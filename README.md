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

The agents decide what matters; the code only answers their questions and checks their
numbers. Every search, count and look is numbered (L1, L2, ...) so numbers can be traced.

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

## Costs

Every model call records its tokens, how long it took (and so output tokens per
second) and what it cost (prices in `cb2/costs.py`).
At each approval pause the run shows this round's cost and the run total. There is
no limit on revision rounds; instead each run has a budget (default $2, or pass
`"budget"` in the input). When the next round would go over it, the run pauses and
asks for a higher budget or stops.
