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
   computed by code. Once you approve, code makes the dataset, then standardizes its molecules
   and splits them (see Models below); no agent touches either.
5. **Choose the models:** the planner chooses the features, a classifier and a regressor from
   the menu, comparing them with **tries**: practice scores from cross-validation on the
   training part, next to the baselines. The reviewer critiques the choice (with tries of its
   own) and the auditor checks the numbers. Each try has a time limit (10 minutes by default,
   `--max-try-minutes`): after every fit, code projects how long the whole try will take and
   stops it early if that's over the limit. At the pause you see the practice scores of what
   was chosen and a rough estimate of how long training and testing will take, both computed by
   code. Once you approve, code trains the models, scores them once on the locked test part and
   saves them.
6. **Conclusions:** the planner writes what the test scores say about the question, how the
   models compare with the baselines and the main limitations. The reviewer critiques it and
   the auditor checks every number against the saved scores.

Each stage is the same loop (budget check, worker, reviewer, auditor where there are numbers
to check, then your approval), so in Studio the run reads plan → get_data → download →
explore → clean → make_dataset → make_split → model → train_and_test → results, and each stage
opens to show its loop.

The agents decide what matters; the code only answers their questions and checks their
numbers. Every search, count, look, dry run and try is numbered (L1, L2, ...), as are the split
and the test scores, so numbers can be traced.

Every pause is written to `runs/<run>/log.md`: each round's work, review, audit and costs,
and what you answered. `runs/<run>/ledger.json` holds every lookup with its full result.

To try things out without adding to the record, tick `practice` (or pass `--practice`): the
run behaves exactly the same, including real API costs and downloads, but logs to
`practice-runs/`, which git ignores. A cheap team (e.g. claude-haiku-5-5 planning,
gpt-6-luna reviewing, a local auditor) keeps practice to fractions of a cent per round.

## Models

`cb2/train.py` trains two separate models on a saved split: a **classifier** (active means
pKi ≥ 7) and a **regressor** (the pKi itself). What is chosen and what is fixed:

- **Chosen:** the features, and for each model any scikit-learn classifier or regressor on the
  menu (`python -m cb2.train --menu` lists them), with the settings to try, e.g.
  `{"C": [0.1, 1, 10]}`. The menu is built from scikit-learn itself: every model that can be
  built with its defaults and passes a quick fit-and-predict check. Settings are checked against
  the model's own list, and one choice may try at most 12 combinations.
- **Fixed:** the seed (`random_state`, wherever a model has one), how features are prepared
  (missing values filled with the training median, then scaled), 5-fold cross-validation inside
  the training part to pick the best settings (by ROC AUC for the classifier, RMSE for the
  regressor), the metrics (classifier: ROC AUC, average precision, balanced accuracy, MCC;
  regressor: RMSE, MAE, R², Spearman) and the cutoff.
- **Baselines:** every model is shown next to "always predict the commonest class / the average"
  and "copy the most similar training molecule" (nearest neighbour by Tanimoto similarity).
- **The test part is locked:** only the final evaluation reads it, and every time it does, a
  line is added to `data/test-use.jsonl`. Test scores come with 95% bootstrap intervals.
- **Sensitivity:** the classifier is also scored at cutoffs 6 and 8, with the settings chosen
  at 7, to show whether conclusions depend on where the line is drawn.

```
python -m cb2.train data/split/<folder> --features morgan2 \
  --classifier RandomForestClassifier --classifier-settings '{"n_estimators": 300, "n_jobs": -1}' \
  --regressor SVR --regressor-settings '{"C": [1, 10]}' --practice
```

With `--practice` it shows practice scores only (the test part stays closed). Without it, it
trains on each split's whole training part, scores both splits' test parts and saves
`data/models/<timestamp>/`: the scores, the test predictions, the fitted scaffold-split models
and a manifest of the source split, choices, settings, library and code versions, how often this
split's test part had been opened before, and checksums.

## Tests

Run `pytest` (no API keys or internet needed). The cleaning tests use a small made-up table
whose right answers were worked out by hand. Other tests break a download, the cleaning
bookkeeping or the splits on purpose (a changed file, a repeated record, a lost dropped row, a
value off the menu, ...) and check that the checks catch it, or change the test part's values
and check that practice scores don't move. Whole runs use stand-in models that call the same
tools a real model would, from the plan to the trained models.

## Costs

Every model call records its tokens, how long it took (and so output tokens per
second) and what it cost (prices in `cb2/costs.py`).
At each approval pause the run shows this round's cost and the run total. There is
no limit on revision rounds; instead each run has a budget (default $2, or pass
`"budget"` in the input). When the next round would go over it, the run pauses and
asks for a higher budget or stops.
