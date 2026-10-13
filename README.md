# CB2-Affinity-V2

A small "research team" of AI models, wired together with LangGraph, that
studies binding affinity at the CB2 receptor. A human approves each step.

## How a run works

You give a research question (or leave it blank for the default one, about CB2). The
agents are told only that the study uses ChEMBL data; the question decides the topic. The team works
through stages; each ends with a pause where you type `approve`, write feedback, or press
Enter to revise from the reviewer's critique.

1. **Plan:** the planner writes a research plan; the reviewer critiques it.
2. **Get the data:** the planner searches ChEMBL's targets and counts their activity
   records, then proposes which targets to download and why. The reviewer critiques the
   choice (with searches of its own) and the **auditor** checks every number in both
   against the search or count it cites. Nothing is downloaded until you approve.
3. **Explore the data:** every activity record for the approved targets is downloaded,
   unfiltered. The planner starts knowing only the column names and row count, and
   explores with **looks**: questions such as "how many rows have each value of this
   column" that code answers exactly. It writes findings, citing a look for every number;
   the reviewer critiques them (with looks of its own) and the auditor checks the numbers.
4. **Clean the data:** the planner chooses a cleaning **recipe**: settings from a fixed menu
   (see Cleaning below), trying them first with dry runs that show exactly how many records
   each step keeps and drops. The reviewer critiques the recipe (with dry runs of its own) and
   the auditor checks the numbers. At the pause you see the recipe's step-by-step counts,
   computed by code. Once you approve, code makes the dataset; no agent touches it.

Each stage is the same loop (budget check, worker, reviewer, auditor where there are numbers
to check, then your approval), so in Studio the run reads plan → get_data → download →
explore → clean → make_dataset, and each stage opens to show its loop.

The agents decide what matters; the code only answers their questions and checks their
numbers. Every search, count, look and dry run is numbered (L1, L2, ...) so numbers can be traced.

Every pause is written to `runs/<run>/log.md`: each round's work, review, audit and costs,
and what you answered. `runs/<run>/ledger.json` holds every lookup with its full result.

To try things out without adding to the record, tick `practice` (or pass `--practice`): the
run behaves exactly the same, including real API costs and downloads, but logs to
`practice-runs/`, which git ignores. A cheap team (e.g. claude-haiku-5-5 planning,
gpt-6-luna reviewing, a local auditor) keeps practice to fractions of a cent per round.

## Models

By default **Claude** (claude-opus-5-5) plans, picks the data and explores, **GPT** (gpt-6-astra) reviews,
and a local model (gemma4:26b) audits. Any role
can be played by any model in `MODELS` in `cb2/llm.py`: Claude Opus, Sonnet or Haiku 5.5,
GPT-6 Astra, Sol or Luna, or local models through [Ollama](https://ollama.com) (gemma4:26b, qwen3:14b).
To use another local model, add one line there.

Choose per run, with effort levels for Claude and GPT (local models ignore effort):

- **LangGraph Studio:** the settings form on the assistant (`planner_model`,
  `planner_effort`, `reviewer_model`, `reviewer_effort`, `auditor_model`, `max_lookups`,
  `fresh_download`, `practice`). Save combinations as assistants, e.g. a local-only team.
- **Terminal:** `python -m cb2.graph --planner-model gemma4:26b --reviewer-effort high --budget 0.50`

## Setup

Needs Python 3.10+ and, for the local models, Ollama.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your Anthropic and OpenAI API keys
```

Check that one model from each provider answers (or name models to try):

```bash
python -m cb2.llm
```

Approved downloads are saved in `data/raw/` (git ignores it), with a manifest recording
the ChEMBL release and the proposal that was approved. A later run that approves exactly
the same targets reuses the download unless `fresh_download` is set. To download targets
yourself: `python -m cb2.data CHEMBL...`

A download is checked every time it is loaded: every file must match the checksum in its
manifest, the row counts must match, and no activity record may appear twice. If anything is
off, the run stops and lists what is wrong. To check a download yourself:
`python -m cb2.checks data/raw/<folder>`

## Cleaning

Cleaning turns a download into one row per compound with one value: pChEMBL, which is
-log10 of the value in molar (so pKi for Ki records). A **recipe** picks settings from a fixed
menu; the steps, their order, the unit conversion and the bookkeeping are fixed in
`cb2/clean.py`:

```json
{"targets": ["CHEMBL253"], "types": ["Ki"], "min_confidence": 9, "drop_flagged": true,
 "combine": "median", "max_spread": 1.0}
```

1. keep the chosen targets
2. keep the chosen measurement types (from Ki, Kd, IC50)
3. keep binding assays, at or above the minimum assay confidence (none, 8 or 9)
4. drop records ChEMBL flags (validity warnings, potential duplicates), if chosen
5. keep exact values (drops censored ones such as "> 10,000 nM")
6. keep usable values (a value in nM above zero, a structure and a parent compound)
7. combine repeat measurements of each parent compound (median or mean), dropping compounds
   whose measurements are further apart than `max_spread` log units (0.5, 1.0, 1.5, or none)

`python -m cb2.clean data/raw/<folder> recipe.json` (or `--dry-run` to only see the counts)
saves `data/clean/<timestamp>/` with `clean.csv`, `dropped.csv` (every removed record, with
its step and reason) and a manifest of the source download, recipe, code version, step-by-step
counts and checksums. Before saving, the code checks that every record is either kept or dropped
exactly once, that every step's counts add up, and that every value is between 2 and 12 and
agrees with ChEMBL's own `pchembl_value`.

## Molecules and splits

Before modeling, `cb2/split.py` prepares a clean dataset and splits it. None of this is chosen
by the team:

- **Standardizing:** each structure keeps its largest piece (dropping salts and solvents) and
  is neutralized. Compounds that turn out to be the same molecule are merged (median value), or
  dropped if their values are further apart than the recipe's `max_spread`; unreadable
  structures are dropped. Every compound is either in one molecule or dropped, exactly once.
- **Scaffold split (the headline):** molecules are grouped by their ring system (Bemis–Murcko
  scaffold), and whole groups go to training or test, so the test asks about new kinds of
  molecule. About 20% go to test.
- **Random split (for comparison):** the same, with every molecule its own group.
- **Folds:** each split's training part is divided into 5 cross-validation folds, again
  keeping groups together. A fixed seed makes all of it repeat exactly.

`python -m cb2.split data/clean/<folder>` (or `--dry-run`) saves `data/split/<timestamp>/`
with `molecules.csv` (each molecule's structure, value, scaffold, part and fold in both
splits), `dropped.csv` and a manifest of the source, settings, RDKit and code versions, counts
and checksums. It checks the clean dataset's checksums before reading it, and checks the splits
(no group in both parts or in two folds, no test molecule in a fold) before saving.

`cb2/features.py` turns structures into numbers, from a menu: Morgan fingerprint counts
(radius 2 or 3, 2048 bits), RDKit descriptors, or both.

## Tests

Run `pytest` (no API keys or internet needed). The cleaning tests use a small made-up table
whose right answers were worked out by hand. Other tests break a download, the cleaning
bookkeeping or the splits on purpose (a changed file, a repeated record, a lost dropped row, a value off the
menu, ...) and check that the checks catch it.

## Costs

Every model call records its tokens, how long it took (and so output tokens per
second) and what it cost (prices in `cb2/costs.py`).
At each approval pause the run shows this round's cost and the run total. There is
no limit on revision rounds; instead each run has a budget (default $2, or pass
`"budget"` in the input). When the next round would go over it, the run pauses and
asks for a higher budget or stops.
