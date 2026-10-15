"""The research team's workflow, built as a LangGraph graph.

Stage 1, plan: the planner writes a research plan and the reviewer critiques it.
Stage 2, get the data: the planner searches ChEMBL and proposes which targets to download;
the reviewer critiques the choice (with searches of its own) and the auditor checks its numbers.
Stage 3, explore the data: the planner explores the download with looks and writes findings;
the reviewer checks them (with looks of its own) and the auditor checks every number.
Stage 4, clean the data: the planner chooses a cleaning recipe from a fixed menu, trying settings
with dry runs; the reviewer critiques it (with dry runs of its own) and the auditor checks every
number. Once the lead approves, code makes the dataset, then standardizes and splits its molecules.
Stage 5, choose the models: the planner chooses the features, a classifier and a regressor from
the menu, comparing practice scores on the training part; the reviewer critiques the choice (with
tries of its own) and the auditor checks every number. Once the lead approves, code trains the
models and scores them once on the test part.
Stage 6, conclusions: the planner writes what the test scores say; the reviewer critiques it and
the auditor checks every number against the scores.
A run can also start from a dataset saved by an earlier run: it is checked against its manifest,
then the run goes from the plan straight to the split, skipping stages 2 to 4.
Every stage is the same loop, built once by make_stage and used six times as a subgraph:
budget check → worker → reviewer → (auditor) → the lead's approval → back round, or on to the
next stage. Every pause is written to the run log in runs/."""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import requests
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from cb2 import chembl, clean, data, runlog, split, train
from cb2.audit import audit
from cb2.costs import round_cost, spent, summary
from cb2.features import FEATURES
from cb2.llm import EFFORTS, MODELS, ask, ask_with_tools, check_model
from cb2.looks import LOOK_TOOL, LookError, look, overview

# The prompts only say the study uses ChEMBL data; the question decides what it is about
DEFAULT_QUESTION = "Which ligand features predict high binding affinity at the CB2 receptor?"

# What code fixes, so plans don't promise what the tools won't do
FIXED_RULES = """What the study's code fixes, which the plan must work within:
- Cleaning keeps exact values only: censored values such as "> 10,000 nM" are dropped. Each
  compound gets one value (pKi for Ki records); compounds that are the same molecule are merged.
- Two splits, each holding out {test:.0%} of the molecules as a locked test part that is scored
  once: by scaffold (the headline) and at random (for comparison), with {folds} folds of the
  training part for practice scores and seed {seed}. There are no other splits, such as by time.
- Two separate models: a classifier, where active means pKi >= {active:g} ({nM:g} nM or stronger),
  also checked at pKi {sensitivity}; and a regressor that predicts pKi. The team chooses the
  features, the models and their settings from menus.
- Baselines: always guessing the commonest class or the average, and copying the most similar
  training molecule. Scores: ROC AUC, average precision, balanced accuracy and MCC for the
  classifier; RMSE, MAE, R² and Spearman for the regressor; each test score with a 95% interval.""".format(
    test=split.TEST_FRACTION, folds=split.FOLDS, seed=split.SEED, active=split.ACTIVE_AT,
    nM=10 ** (9 - split.ACTIVE_AT), sensitivity=" and ".join(f"{cutoff:g}" for cutoff in train.SENSITIVITY))

PLANNER_PROMPT = """You are planning a small study that will only use data downloaded from
ChEMBL and analysed in Python.

""" + FIXED_RULES + """

Write a short research plan (3 to 5 numbered steps) for this question:

{question}"""

REVISION_PROMPT = PLANNER_PROMPT + """

Your previous plan:
{plan}

A reviewer's critique of it:
{review}

Feedback from the project lead:
{feedback}

Write an improved plan that addresses the lead's feedback and the points of the
critique you judge most important. Keep it to 3 to 5 numbered steps: rewrite the
plan, don't add to it. If a point in the critique is wrong or doesn't fit the study,
don't follow it; list any such points in a short note at the end, at most 3 bullets."""

REVIEWER_PROMPT = """You are a critical reviewer of research plans for studies that use ChEMBL data.

Question: {question}

Plan:
{plan}

""" + FIXED_RULES + """

List the three most important weaknesses or risks in this plan, one or two sentences each. Don't
ask for what the code rules out; point out anything in the plan that it can't do."""

SOURCE_PROMPT = """You are the planner of a small study using data from ChEMBL, analysed in Python.

Research question: {question}

The approved research plan:
{plan}

Next, decide which data to download from ChEMBL. You have tools to search ChEMBL's targets and
count their activity records; you can make up to {max_lookups} searches and counts. When you have
decided, call propose_download with the target IDs to download (every activity record for each,
with its assay details) and your reasons. Then write a short summary of the proposal for the
project lead: what you would download and leave out, and why.

Whenever you state a number from a search or count, cite it in brackets, like [L3].
Keep the summary under 300 words."""

SOURCE_REVISION_PROMPT = SOURCE_PROMPT + """

Your previous proposal: {proposal}
Your previous summary:
{summary}

Searches and counts made so far (you can repeat or extend them):
{lookup_list}

A reviewer's critique of the proposal:
{review}

The auditor's check of the numbers in it:
{audit}

Feedback from the project lead:
{feedback}

Make an improved proposal (call propose_download again) that addresses the lead's feedback and
the points of the critique you judge most important, and write a new summary. If a point in the
critique is wrong, don't follow it; say so in a short note at the end."""

SOURCE_REVIEWER_PROMPT = """You are a critical reviewer of studies that use ChEMBL data.

Question: {question}

Plan:
{plan}

The planner searched ChEMBL and proposes downloading these targets:
{proposal}

The planner's summary:
{summary}

The planner's searches and counts, with their results:
{lookups}

The plan has already been approved, so don't review it again. Review the download proposal:
whether these targets are the right data for the question and the plan, and whether anything
the plan needs is left out or anything included doesn't belong. You can make up to
{max_lookups} searches and counts of your own. Then list the three most important problems
with what is proposed (or left out), one or two sentences each. Cite results in brackets, like
[L3], for any numbers."""

EXPLORE_PROMPT = """You are the planner of a small study using data from ChEMBL, analysed in Python.

Research question: {question}

The approved research plan:
{plan}

The lead approved downloading every activity record for these ChEMBL targets, with details of
each record's assay and target: {targets}
Nothing has been filtered out. This is all you know about the downloaded table so far:

{overview}

Use the look tool to explore the data; you can make up to {max_lookups} looks. Then write your findings:
1. What you learned about the data that matters for the plan.
2. How you would turn it into the dataset the plan needs, step by step, with your reasons.

Whenever you state a number about the data, cite the look it came from in brackets, like [L3].
Keep the findings under 500 words."""

EXPLORE_REVISION_PROMPT = EXPLORE_PROMPT + """

Your previous findings:
{findings}

Looks made so far (you can look again, or make new looks):
{look_list}

A reviewer's critique of the findings:
{review}

The auditor's check of the numbers in them:
{audit}

Feedback from the project lead:
{feedback}

Write improved findings that address the lead's feedback, the points of the critique you judge
most important and any numbers the auditor could not confirm. If a point in the critique is
wrong, don't follow it; say so in a short note at the end."""

DATA_REVIEWER_PROMPT = """You are a critical reviewer of studies that use ChEMBL data.

Question: {question}

Plan:
{plan}

The planner explored the downloaded ChEMBL data with looks (numbered queries answered exactly
by code) and wrote these findings:
{findings}

The planner's looks and their results:
{looks}

You can make up to {max_lookups} looks of your own to check claims or follow up on something
the planner missed. Then list the three most important problems with the findings or the
proposed steps, one or two sentences each. Cite looks in brackets, like [L3], for any numbers."""

CLEAN_PROMPT = """You are the planner of a small study using data from ChEMBL, analysed in Python.

Research question: {question}

The approved research plan:
{plan}

The downloaded table holds every activity record for these ChEMBL targets: {targets}
{overview}

Your approved findings from exploring it:
{findings}

Next, choose how to clean the table into the dataset the plan needs: one row per compound, with
one affinity value. Code runs these steps in this order; the settings in brackets are yours to
choose, and the rest is fixed:
{steps}

Use try_recipe to see exactly how many records each step keeps and drops with given settings, and
look to check anything else; you can make up to {max_lookups} tries and looks. When you have
decided, call propose_recipe with the settings and your reasons. Then write a short summary for
the project lead: the settings you chose, what they cost in records and compounds, and why.

Whenever you state a number from a try or look, cite it in brackets, like [L3].
Keep the summary under 300 words."""

CLEAN_REVISION_PROMPT = CLEAN_PROMPT + """

Your previous recipe:
{recipe}

Your previous summary:
{summary}

Tries and looks made so far (you can repeat or extend them):
{lookup_list}

A reviewer's critique of the recipe:
{review}

The auditor's check of the numbers in it:
{audit}

Feedback from the project lead:
{feedback}

Make an improved recipe (call propose_recipe again) that addresses the lead's feedback and the
points of the critique you judge most important, and write a new summary. If a point in the
critique is wrong, don't follow it; say so in a short note at the end."""

CLEAN_REVIEWER_PROMPT = """You are a critical reviewer of studies that use ChEMBL data.

Question: {question}

Plan:
{plan}

The planner proposes cleaning the downloaded data with a recipe. Code runs these steps in this
order; the settings in brackets were the planner's to choose, and the rest is fixed:
{steps}

The proposed recipe, and what code found it does to the data:
{recipe}

The planner's summary:
{summary}

The planner's tries and looks, with their results:
{lookups}

The plan and the findings have already been approved, so don't review them again. Review the
recipe: whether each setting suits the question and the plan, and what it costs in data. You can
make up to {max_lookups} tries and looks of your own. Then list the three most important problems
with the recipe, one or two sentences each. Cite results in brackets, like [L3], for any numbers."""

MODEL_PROMPT = """You are the planner of a small study using data from ChEMBL, analysed in Python.

Research question: {question}

The approved research plan:
{plan}

The approved dataset has one pKi value per compound. Code has standardized the molecules and
split them; the scaffold split is the headline, the random split is for comparison:
{split}

Next, choose how to model it. Code trains two separate models: a classifier (active means
pKi >= {active_at:g}) and a regressor that predicts pKi. You choose:
- the features, one of:
{features}
- the classifier and the regressor: any model on this menu, with settings to try (at most
  {max_settings} combinations each; settings you leave out keep the model's defaults). TanimotoSVC
  and TanimotoSVR compare molecules by Tanimoto similarity, so they need morgan2 or morgan3:
{menu}

Fixed by code: the splits and folds, the seed, how the features are prepared (missing values
filled with the training median, then scaled), the metrics, the baselines and the test part. Code
picks each model's settings by practice {pick}, and also checks the classifier at pKi
{sensitivity} with the settings chosen at {active_at:g}.

Use try_models to get practice scores for a model and its settings (cross-validation on the
training part of the scaffold split), next to the baselines; you can make up to {max_lookups}
tries. Each try may take at most {try_minutes:g} minutes on this computer; one that would take
longer is stopped early, so when two choices score about the same, prefer the faster one. When you
have decided, call propose_models with your choices and reasons. Then write a short summary for the
project lead: what you chose, how it compares with the alternatives you
tried and with the baselines, and why.

Whenever you state a number from a try or the split, cite it in brackets, like [L3].
Keep the summary under 300 words."""

MODEL_REVISION_PROMPT = MODEL_PROMPT + """

Your previous proposal:
{models}

Your previous summary:
{summary}

Tries made so far (you can repeat or extend them):
{lookup_list}

A reviewer's critique of the proposal:
{review}

The auditor's check of the numbers in it:
{audit}

Feedback from the project lead:
{feedback}

Make an improved proposal (call propose_models again) that addresses the lead's feedback and the
points of the critique you judge most important, and write a new summary. If a point in the
critique is wrong, don't follow it; say so in a short note at the end."""

MODEL_REVIEWER_PROMPT = """You are a critical reviewer of studies that use ChEMBL data.

Question: {question}

Plan:
{plan}

The dataset and its split:
{split}

The planner proposes these features and models, chosen on practice scores (cross-validation on the
training part of the scaffold split). Code computed the practice scores shown:
{models}

The planner's summary:
{summary}

The planner's tries, with their results:
{lookups}

The plan and the dataset have already been approved, so don't review them again. Review the choice
of features and models: whether the practice scores support it, whether it beats the baselines by
enough to matter, and whether a better option was missed. You can make up to {max_lookups} tries
of your own. Then list the three most important problems with the choice, one or two sentences
each. Cite results in brackets, like [L3], for any numbers."""

RESULTS_PROMPT = """You are the planner of a small study using data from ChEMBL, analysed in Python.

Research question: {question}

The approved research plan:
{plan}

The approved features and models:
{models}

Code trained them on the whole training part and scored them once on the test part, which nothing
was tuned on. The scaffold split is the headline: its test part holds whole families of molecules
the models never saw. The random split is for comparison. The test scores, each with its 95%
bootstrap interval, next to the baselines:
{scores}

Write the conclusions for the project lead:
1. What the results say about the research question, and how sure we can be.
2. How the models compare with the baselines, and the scaffold split with the random split.
3. The main limitations.

Only quote numbers from the scores above or the practice scores, and cite them in brackets, like
[L3]. Keep the conclusions under 400 words."""

RESULTS_REVISION_PROMPT = RESULTS_PROMPT + """

Your previous conclusions:
{conclusions}

A reviewer's critique of them:
{review}

The auditor's check of the numbers in them:
{audit}

Feedback from the project lead:
{feedback}

Write improved conclusions that address the lead's feedback, the points of the critique you judge
most important and any numbers the auditor could not confirm. If a point in the critique is
wrong, don't follow it; say so in a short note at the end."""

RESULTS_REVIEWER_PROMPT = """You are a critical reviewer of studies that use ChEMBL data.

Question: {question}

Plan:
{plan}

The approved features and models:
{models}

Their test scores, each with its 95% bootstrap interval, next to the baselines:
{scores}

The planner's conclusions:
{conclusions}

The models have already been chosen and tested, so don't review that choice. Review the
conclusions: whether each one follows from the scores and their intervals, and what is overclaimed
or missing. List the three most important problems, one or two sentences each. Cite the scores in
brackets, like [L3], for any numbers."""

# Dollars a run may spend before it pauses to ask for more; pass "budget" in the input to change it
DEFAULT_BUDGET = 2.00
REVIEWER_LOOKUPS = 5

# Used when the lead leaves the answer blank: revise from the critique alone
NO_FEEDBACK = "None given. Address the most important points of the critique."
ANSWER_PROMPT = ("Type 'approve' (or 'ok') to go on, write feedback (or leave it blank to revise from the review) "
                 "for another round, or 'stop' to end the run")
# Answers at an approval pause that end the run
STOP_WORDS = ("stop", "end")
# Answers that mean go on: anything else is feedback, and feedback starts another (paid) round
APPROVE_WORDS = ("approve", "approved", "yes", "y", "ok", "okay", "continue", "go on", "next")


def one_of(answer: str, words: tuple[str, ...]) -> bool:
    return answer.lower().strip(" .!'\"") in words


ModelName = Literal[tuple(MODELS)]
Effort = Literal[("default",) + EFFORTS]


# Which model plays each role, chosen per run. LangGraph Studio shows these as a form,
# and each saved combination becomes an "assistant" you can pick from.
# Effort is ignored by models that don't have it (the local ones).
@dataclass
class Settings:
    planner_model: ModelName = "claude-opus-5-5"
    planner_effort: Effort = "default"
    reviewer_model: ModelName = "gpt-6-astra"
    reviewer_effort: Effort = "default"
    auditor_model: ModelName = "gemma4:26b"
    # Searches, counts or looks the planner may make in each round
    max_lookups: int = 15
    # Download again even if the same targets were downloaded before
    fresh_download: bool = False
    # Practice runs log to practice-runs/ (git-ignored) instead of runs/, so they stay out of the record
    practice: bool = False
    # The longest one try of models may take; a try that would run longer is stopped early
    max_try_minutes: float = train.TRY_MINUTES
    # Answer every approval pause automatically: the plan is revised from the review until it has had
    # auto_plan_rounds rounds, every other stage is approved after one. Budget pauses still ask you
    auto_approve: bool = False
    auto_plan_rounds: int = 3


def effort(value: str) -> str | None:
    # "default" means don't send an effort, so the model uses its own default
    return None if value == "default" else value


def grow(existing: list, new: list) -> list:
    """How the growing lists (costs, ledger, history) take updates. A node returns new items, which
    are appended. A stage hands back its whole list when it finishes, and that list starts with
    what's already here: take it as the full list rather than adding everything a second time."""
    if existing and new[:len(existing)] == existing:
        return new
    return existing + new


# The shared notebook every node reads from and writes to
class State(TypedDict, total=False):
    question: str
    # A saved clean dataset to start from (data/clean/<stamp>), instead of downloading and cleaning
    start_from: str
    run_id: str
    stage: Literal["plan", "source", "data", "clean", "model", "results"]
    # Counts rounds across all stages; every cost record says which round it belongs to
    round: int
    budget: float
    stopped: bool
    # Set by each approval: did the lead approve this round?
    approved: bool
    # Lists that grow as the run goes on
    costs: Annotated[list[dict], grow]      # one usage record per model call or tool loop
    ledger: Annotated[list[dict], grow]     # every search, count and look, numbered L1, L2, ...
    history: Annotated[list[dict], grow]    # every round and the lead's reply, for the run log
    audit_report: str
    # Stage 1: plan
    plan: str
    review: str
    plan_feedback: str
    # Stage 2: get the data
    proposal: dict
    source_summary: str
    source_review: str
    source_feedback: str
    # Stage 3: explore the data
    data_folder: str
    overview_id: str
    findings: str
    data_review: str
    data_feedback: str
    # Stage 4: clean the data
    recipe: dict
    recipe_summary: str
    clean_review: str
    clean_feedback: str
    clean_folder: str
    split_folder: str
    split_id: str
    # Stage 5: choose the models
    models: dict
    models_summary: str
    models_review: str
    models_feedback: str
    models_folder: str
    scores_id: str
    # Stage 6: conclusions
    conclusions: str
    conclusions_review: str
    results_feedback: str



def cost_record(node: str, state: State, usage: dict) -> dict:
    return {"node": node, "stage": state["stage"], "round": state["round"], **usage}


def pause(state: State, runtime: Runtime[Settings], kind: str, sections: dict[str, str],
          how_to_answer: str | None = None, auto_answer: str | None = None) -> tuple[str, dict]:
    """Show sections to the lead and wait for an answer (or take auto_answer); log the round before
    and after. how_to_answer is shown with the pause but not written to the log."""
    item = runlog.entry(kind, state.get("round", 0), list(sections.items()))
    history = state.get("history", [])
    args = (state["run_id"], state["question"], runtime.context or Settings())
    runlog.write(*args, history + [item], state.get("ledger", []))
    if auto_answer is None:
        shown = {"kind": kind, **sections, **({"How to answer": how_to_answer} if how_to_answer else {})}
        answer = str(interrupt(shown) or "").strip()
    else:
        answer, item["auto"] = auto_answer, True
    item["feedback"] = answer
    runlog.write(*args, history + [item], state.get("ledger", []))
    return answer, item


# --- The loop every stage shares ---

def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def make_budget_gate(stage: str, worker: str):
    def budget_gate(state: State, runtime: Runtime[Settings]) -> Command:
        settings = runtime.context or Settings()
        for role in ("planner", "reviewer", "auditor"):
            check_model(role, getattr(settings, f"{role}_model"))
        defaults = {
            "budget": state.get("budget", DEFAULT_BUDGET),
            "stage": stage,
            "question": state.get("question") or DEFAULT_QUESTION,
            "run_id": state.get("run_id") or new_run_id(),
        }

        # Guess the next round costs about what this stage's last round did (nothing to go on before the first)
        costs = state.get("costs", [])
        rounds_in_stage = [call["round"] for call in costs if call["stage"] == stage]
        estimate = round_cost(costs, max(rounds_in_stage)) if rounds_in_stage else 0.0
        if spent(costs) + estimate <= defaults["budget"]:
            return Command(update=defaults, goto=worker)

        # Over budget: pause and let the person decide whether the next round is worth it
        message = (f"Spent ${spent(costs):.3f} of the ${defaults['budget']:.2f} budget, and the next round "
                   f"should cost about ${estimate:.3f}. Type a new budget in dollars to continue, "
                   f"or anything else to stop.")
        answer, item = pause({**state, **defaults}, runtime, "budget", {"Budget": message})
        try:
            new_budget = float(answer.removeprefix("$"))
        except ValueError:
            return Command(update={**defaults, "stopped": True, "history": [item]}, goto=END)
        # Check again, in case the new budget still doesn't cover the next round
        return Command(update={**defaults, "budget": new_budget, "history": [item]}, goto="budget_gate")
    return budget_gate


def make_auditor(sections):
    def auditor(state: State, runtime: Runtime[Settings]) -> dict:
        settings = runtime.context or Settings()
        report, usage = audit(sections(state), state["ledger"], settings.auditor_model)
        return {"audit_report": report, "costs": [cost_record("auditor", state, usage)] if usage else []}
    return auditor


def make_approval(stage: str, feedback_key: str, sections, not_ready=lambda state: None):
    """not_ready(state) can return a reason the lead can't approve yet; it becomes the feedback."""
    def approval(state: State, runtime: Runtime[Settings]) -> dict:
        settings = runtime.context or Settings()
        auto_answer = None
        if settings.auto_approve:
            # Blank means revise from the review
            rounds = 1 + sum(item["stage"] == stage for item in state.get("history", []))
            auto_answer = "" if stage == "plan" and rounds < settings.auto_plan_rounds else "approve"
        answer, item = pause(state, runtime, stage, {
            **sections(state), "Costs": summary(state["costs"], state["round"], state["budget"])}, ANSWER_PROMPT,
            auto_answer)
        if one_of(answer, STOP_WORDS):
            return {feedback_key: answer, "approved": False, "stopped": True, "history": [item]}
        approved = one_of(answer, APPROVE_WORDS)
        reason = not_ready(state) if approved else None
        if reason:
            approved, answer = False, reason
        return {feedback_key: answer, "approved": approved, "history": [item]}
    return approval


def make_stage(stage: str, worker, reviewer, feedback_key: str, show, audit_sections=None, not_ready=lambda state: None):
    """One stage as a subgraph: budget check → worker → reviewer → (auditor) → approval.
    show(state) gives the sections the lead sees; audit_sections(state) the texts to audit."""
    builder = StateGraph(State, context_schema=Settings)
    builder.add_node("budget_gate", make_budget_gate(stage, worker.__name__),
                     destinations=(worker.__name__, "budget_gate", END))
    builder.add_node(worker.__name__, worker)
    builder.add_node("reviewer", reviewer)
    builder.add_node("approval", make_approval(stage, feedback_key, show, not_ready))
    builder.add_edge(START, "budget_gate")
    builder.add_edge(worker.__name__, "reviewer")
    if audit_sections:
        builder.add_node("auditor", make_auditor(audit_sections))
        builder.add_edge("reviewer", "auditor")
        builder.add_edge("auditor", "approval")
    else:
        builder.add_edge("reviewer", "approval")
    # Approved or stopped: the stage is done. Otherwise go round again, budget permitting
    builder.add_conditional_edges("approval", lambda state: END if state["approved"] or state.get("stopped")
                                  else "budget_gate", ["budget_gate", END])
    return builder.compile()


def lookup_runner(state: State, ledger: list[dict], by: str, try_minutes: float | None = None):
    """A run_tool function for ask_with_tools: runs searches, counts, looks and tries, numbering each
    and adding it to ledger. A failed lookup tells the agent what went wrong instead of stopping."""
    def run(name: str, request: dict) -> str:
        look_id = f"L{len(ledger) + 1}"
        try:
            if name == "search_targets":
                entry = chembl.search_targets(str(request.get("query", "")), look_id)
            elif name == "count_activities":
                entry = chembl.count_activities(list(request.get("target_ids") or []),
                                                request.get("standard_types"), look_id)
            elif name == "look":
                entry = look(data.load(state["data_folder"]), request, look_id)
            elif name == "try_recipe":
                entry = clean.try_recipe(data.load(state["data_folder"]), request, look_id)
            elif name == "try_models":
                entry = train.try_models(Path(state["split_folder"]), request, look_id, try_minutes)
            else:
                return f"Unknown tool {name!r}."
        except (ValueError, LookError) as error:
            return f"{name} error: {error}"
        except requests.RequestException as error:
            return f"{name} error: ChEMBL didn't answer ({error}). Try again or continue without it."
        ledger.append({**entry, "by": by, "round": state["round"]})
        return entry["text"]
    return run


LOOK_KINDS = {"value_counts", "crosstab", "describe", "sample"}


def lookup_list(ledger: list[dict], stage_kinds: set[str]) -> str:
    return "\n".join(entry["text"].splitlines()[0] for entry in ledger if entry["request"]["kind"] in stage_kinds)


# --- Stage 1: plan ---

def planner(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    # First pass: plan from the question. Later passes: revise the existing plan
    if "plan" not in state:
        prompt = PLANNER_PROMPT.format(question=state["question"])
    else:
        prompt = REVISION_PROMPT.format(question=state["question"], plan=state["plan"],
                                        review=state["review"], feedback=state["plan_feedback"] or NO_FEEDBACK)
    state = {**state, "round": state.get("round", 0) + 1}
    plan, usage = ask(prompt, settings.planner_model, effort(settings.planner_effort))
    return {"plan": plan, "round": state["round"], "costs": [cost_record("planner", state, usage)]}


def plan_reviewer(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    prompt = REVIEWER_PROMPT.format(question=state["question"], plan=state["plan"])
    review, usage = ask(prompt, settings.reviewer_model, effort(settings.reviewer_effort))
    return {"review": review, "costs": [cost_record("reviewer", state, usage)]}


# --- Stage 2: get the data ---

def format_proposal(proposal: dict | None) -> str:
    if not proposal:
        return "(no proposal made)"
    targets = "\n".join(f"- {t['target_chembl_id']}: {t['pref_name']} ({t['organism'] or '-'}, {t['target_type']})"
                        for t in proposal["targets"])
    return f"{targets}\nReason: {proposal['reason']}"


def sourcer(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    state = {**state, "round": state["round"] + 1}
    ledger = list(state.get("ledger", []))
    run_lookup = lookup_runner(state, ledger, "sourcer")
    proposal = {}

    def run(name: str, request: dict) -> str:
        if name != "propose_download":
            return run_lookup(name, request)
        target_ids = [str(target_id).strip() for target_id in request.get("target_ids") or []]
        try:
            if not target_ids:
                raise ValueError("No target IDs given.")
            targets = chembl.describe_targets(target_ids)
        except (ValueError, requests.RequestException) as error:
            return f"Not proposed: {error}"
        proposal.update(target_ids=target_ids, reason=str(request.get("reason", "")),
                        targets=[{key: target[key] for key in ("target_chembl_id", "pref_name", "organism", "target_type")}
                                 for target in targets])
        return "Proposal recorded:\n" + format_proposal(proposal) + "\nNow write your summary for the lead."

    if "source_summary" not in state:
        prompt = SOURCE_PROMPT.format(question=state["question"], plan=state["plan"], max_lookups=settings.max_lookups)
    else:
        prompt = SOURCE_REVISION_PROMPT.format(
            question=state["question"], plan=state["plan"], max_lookups=settings.max_lookups,
            proposal=format_proposal(state.get("proposal")), summary=state["source_summary"],
            lookup_list=lookup_list(ledger, {"search_targets", "count_activities"}),
            review=state["source_review"], audit=state["audit_report"],
            feedback=state["source_feedback"] or NO_FEEDBACK,
        )
    summary_text, usage = ask_with_tools(prompt, settings.planner_model, effort(settings.planner_effort),
                                         [chembl.SEARCH_TOOL, chembl.COUNT_TOOL, chembl.PROPOSE_TOOL], run,
                                         settings.max_lookups, uncounted=("propose_download",))
    return {"source_summary": summary_text, "proposal": proposal or state.get("proposal") or {},
            "round": state["round"], "ledger": ledger[len(state.get("ledger", [])):],
            "costs": [cost_record("sourcer", state, usage)]}


def source_reviewer(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    ledger = list(state["ledger"])
    this_round = [entry["text"] for entry in ledger if entry.get("by") == "sourcer" and entry["round"] == state["round"]]
    prompt = SOURCE_REVIEWER_PROMPT.format(question=state["question"], plan=state["plan"],
                                           proposal=format_proposal(state["proposal"]), summary=state["source_summary"],
                                           lookups="\n\n".join(this_round) or "(none)", max_lookups=REVIEWER_LOOKUPS)
    review, usage = ask_with_tools(prompt, settings.reviewer_model, effort(settings.reviewer_effort),
                                   [chembl.SEARCH_TOOL, chembl.COUNT_TOOL], lookup_runner(state, ledger, "reviewer"),
                                   REVIEWER_LOOKUPS)
    return {"source_review": review, "ledger": ledger[len(state["ledger"]):],
            "costs": [cost_record("reviewer", state, usage)]}


def download(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    target_ids = state["proposal"]["target_ids"]
    folder = None if settings.fresh_download else data.find_download(target_ids)
    if folder is None:
        folder = data.download(target_ids, state["proposal"]["reason"], state["run_id"])
    # The column names and row count are all the agents know before their first look
    entry = overview(data.load(str(folder)), f"L{len(state.get('ledger', [])) + 1}")
    return {"data_folder": str(folder), "overview_id": entry["id"],
            "ledger": [{**entry, "by": "download", "round": state["round"]}]}


# --- Stage 3: explore the data ---

def about_the_data(state: State) -> dict:
    # The approved targets and the table overview: what the agents know before their first look
    targets = ", ".join(f"{t['target_chembl_id']} ({t['pref_name']}, {t['organism'] or '-'}, {t['target_type']})"
                        for t in state["proposal"]["targets"])
    table_overview = next(entry["text"] for entry in state["ledger"] if entry["id"] == state["overview_id"])
    return {"question": state["question"], "plan": state["plan"], "targets": targets, "overview": table_overview}


def explorer(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    state = {**state, "round": state["round"] + 1}
    ledger = list(state["ledger"])
    fields = dict(**about_the_data(state), max_lookups=settings.max_lookups)
    if "findings" not in state:
        prompt = EXPLORE_PROMPT.format(**fields)
    else:
        prompt = EXPLORE_REVISION_PROMPT.format(
            **fields, findings=state["findings"], look_list=lookup_list(ledger, LOOK_KINDS),
            review=state["data_review"], audit=state["audit_report"], feedback=state["data_feedback"] or NO_FEEDBACK,
        )
    findings, usage = ask_with_tools(prompt, settings.planner_model, effort(settings.planner_effort),
                                     [LOOK_TOOL], lookup_runner(state, ledger, "explorer"), settings.max_lookups)
    return {"findings": findings, "round": state["round"], "ledger": ledger[len(state["ledger"]):],
            "costs": [cost_record("explorer", state, usage)]}


def data_reviewer(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    ledger = list(state["ledger"])
    this_round = [entry["text"] for entry in ledger if entry.get("by") == "explorer" and entry["round"] == state["round"]]
    prompt = DATA_REVIEWER_PROMPT.format(question=state["question"], plan=state["plan"], findings=state["findings"],
                                         looks="\n\n".join(this_round) or "(none)", max_lookups=REVIEWER_LOOKUPS)
    review, usage = ask_with_tools(prompt, settings.reviewer_model, effort(settings.reviewer_effort),
                                   [LOOK_TOOL], lookup_runner(state, ledger, "reviewer"), REVIEWER_LOOKUPS)
    return {"data_review": review, "ledger": ledger[len(state["ledger"]):],
            "costs": [cost_record("reviewer", state, usage)]}


# --- Stage 4: clean the data ---

def format_recipe(recipe: dict | None) -> str:
    if not recipe:
        return "(no recipe proposed)"
    settings = " · ".join(f"{key}={json.dumps(value)}" for key, value in recipe["settings"].items())
    return f"{settings}\nReason: {recipe['reason']}\n\n{recipe['steps']}"


def cleaner(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    state = {**state, "round": state["round"] + 1}
    ledger = list(state["ledger"])
    run_lookup = lookup_runner(state, ledger, "cleaner")
    proposal = {}

    def run(name: str, request: dict) -> str:
        if name != "propose_recipe":
            return run_lookup(name, request)
        table = data.load(state["data_folder"])
        try:
            recipe = clean.make_recipe({key: request[key] for key in clean.RECIPE_PROPERTIES if key in request}, table)
        except clean.RecipeError as error:
            return f"Not proposed: {error}"
        # The lead sees what the recipe does, computed by code, next to the planner's summary
        steps = clean.waterfall_text(clean.run(table, recipe).waterfall)
        proposal.update(settings=json.loads(json.dumps(asdict(recipe))), reason=str(request.get("reason", "")),
                        steps=steps)
        return "Recipe recorded:\n" + format_recipe(proposal) + "\nNow write your summary for the lead."

    fields = dict(**about_the_data(state), findings=state["findings"], steps=clean.STEPS_TEXT,
                  max_lookups=settings.max_lookups)
    if "recipe_summary" not in state:
        prompt = CLEAN_PROMPT.format(**fields)
    else:
        prompt = CLEAN_REVISION_PROMPT.format(
            **fields, recipe=format_recipe(state.get("recipe")), summary=state["recipe_summary"],
            lookup_list=lookup_list(ledger, LOOK_KINDS | {"try_recipe"}), review=state["clean_review"],
            audit=state["audit_report"], feedback=state["clean_feedback"] or NO_FEEDBACK,
        )
    summary_text, usage = ask_with_tools(prompt, settings.planner_model, effort(settings.planner_effort),
                                         [LOOK_TOOL, clean.TRY_TOOL, clean.PROPOSE_TOOL], run,
                                         settings.max_lookups, uncounted=("propose_recipe",))
    return {"recipe_summary": summary_text, "recipe": proposal or state.get("recipe") or {},
            "round": state["round"], "ledger": ledger[len(state["ledger"]):],
            "costs": [cost_record("cleaner", state, usage)]}


def clean_reviewer(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    ledger = list(state["ledger"])
    this_round = [entry["text"] for entry in ledger if entry.get("by") == "cleaner" and entry["round"] == state["round"]]
    prompt = CLEAN_REVIEWER_PROMPT.format(question=state["question"], plan=state["plan"], steps=clean.STEPS_TEXT,
                                          recipe=format_recipe(state["recipe"]), summary=state["recipe_summary"],
                                          lookups="\n\n".join(this_round) or "(none)", max_lookups=REVIEWER_LOOKUPS)
    review, usage = ask_with_tools(prompt, settings.reviewer_model, effort(settings.reviewer_effort),
                                   [LOOK_TOOL, clean.TRY_TOOL], lookup_runner(state, ledger, "reviewer"),
                                   REVIEWER_LOOKUPS)
    return {"clean_review": review, "ledger": ledger[len(state["ledger"]):],
            "costs": [cost_record("reviewer", state, usage)]}


def record(state: State, runtime: Runtime[Settings], kind: str, sections: list, new_entries=()) -> dict:
    """Log a step code takes after the lead's approval."""
    item = runlog.entry(kind, state["round"], sections)
    runlog.write(state["run_id"], state["question"], runtime.context or Settings(),
                 state.get("history", []) + [item], state.get("ledger", []) + list(new_entries))
    return item


def make_dataset(state: State, runtime: Runtime[Settings]) -> dict:
    """Code, not an agent: make the dataset from the approved recipe, check it and save it."""
    table = data.load(state["data_folder"])
    recipe = clean.make_recipe(state["recipe"]["settings"], table)
    result = clean.run(table, recipe)
    folder = clean.save(result, Path(state["data_folder"]), recipe, state["recipe"]["reason"], state["run_id"])
    text = f"Saved {len(result.clean):,} compounds to {clean.shown(folder)}\n\n{clean.waterfall_text(result.waterfall)}"
    return {"clean_folder": str(folder), "history": [record(state, runtime, "dataset", [("Steps", text)])]}


def use_dataset(state: State, runtime: Runtime[Settings]) -> dict:
    """Code, not an agent: check a saved dataset before anything else runs, so a changed file stops
    the run before any money is spent."""
    folder = Path(state["start_from"]).resolve()
    table, manifest = split.load_clean(folder)
    state = {**state, "question": state.get("question") or DEFAULT_QUESTION, "run_id": new_run_id(), "round": 0}
    text = (f"Started from the saved dataset {clean.shown(folder)}: {len(table):,} compounds, every file matches "
            f"its manifest.\nMade by run {manifest.get('run') or '(none recorded)'}"
            f"{', recipe reason: ' + manifest['reason'] if manifest.get('reason') else ''}")
    item = record(state, runtime, "dataset", [("Steps", text)])
    return {"question": state["question"], "run_id": state["run_id"], "clean_folder": str(folder), "history": [item]}


def make_split(state: State, runtime: Runtime[Settings]) -> dict:
    """Code, not an agent: standardize the dataset's molecules and split them, the same way every time."""
    source = Path(state["clean_folder"])
    table, manifest = split.load_clean(source)
    molecules, dropped = split.prepare(table, manifest["recipe"]["max_spread"])
    splits = split.make_splits(molecules)
    folder = split.save(splits, dropped, source, state["run_id"])
    entry = {**split.summary_entry(splits, dropped, len(table), f"L{len(state.get('ledger', [])) + 1}"),
             "by": "split", "round": state["round"]}
    item = record(state, runtime, "split", [("Steps", f"Saved to {clean.shown(folder)}\n\n{entry['text']}")], [entry])
    return {"split_folder": str(folder), "split_id": entry["id"], "ledger": [entry], "history": [item]}


# --- Stage 5: choose the models ---

def entry_text(state: State, entry_id: str) -> str:
    return next(entry["text"] for entry in state["ledger"] if entry["id"] == entry_id)


def format_models(models: dict | None) -> str:
    if not models:
        return "(no models proposed)"
    chosen = "\n".join(f"{task}: {models[task]['model']}, settings to try {json.dumps(models[task]['settings'])}"
                       for task in train.PICK_BY)
    return f"features: {models['features']}\n{chosen}\nReason: {models['reason']}\n\n{models['practice']}"


def about_modeling(state: State) -> dict:
    return {"question": state["question"], "plan": state["plan"], "split": entry_text(state, state["split_id"])}


def modeler(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    state = {**state, "round": state["round"] + 1}
    ledger = list(state["ledger"])
    run_lookup = lookup_runner(state, ledger, "modeler", settings.max_try_minutes)
    proposal = {}

    def run(name: str, request: dict) -> str:
        if name != "propose_models":
            return run_lookup(name, request)
        try:
            proposed = train.propose(Path(state["split_folder"]), request, settings.max_try_minutes)
        except train.ChoiceError as error:
            return f"Not proposed: {error}"
        # The practice scores become a lookup, so the summary can cite them
        look_id = f"L{len(ledger) + 1}"
        ledger.append({"id": look_id, "request": {"kind": "propose_models", **request},
                       "numbers": proposed.pop("numbers"), "text": f"{look_id} {proposed['practice']}",
                       "by": "modeler", "round": state["round"]})
        proposal.clear()
        proposal.update(proposed)
        return (f"Models recorded as {look_id} (cite it for these practice scores):\n" + format_models(proposal)
                + "\nNow write your summary for the lead.")

    fields = dict(**about_modeling(state), active_at=split.ACTIVE_AT, max_settings=train.MAX_SETTINGS,
                  features="\n".join(f"  {name}: {description}" for name, (description, _) in FEATURES.items()),
                  menu=train.menu_text(), pick=" and ".join(metric for metric, _ in train.PICK_BY.values()),
                  sensitivity=" and ".join(f"{cutoff:g}" for cutoff in train.SENSITIVITY),
                  max_lookups=settings.max_lookups, try_minutes=settings.max_try_minutes)
    if "models_summary" not in state:
        prompt = MODEL_PROMPT.format(**fields)
    else:
        prompt = MODEL_REVISION_PROMPT.format(
            **fields, models=format_models(state.get("models")), summary=state["models_summary"],
            lookup_list=lookup_list(ledger, {"try_models"}), review=state["models_review"],
            audit=state["audit_report"], feedback=state["models_feedback"] or NO_FEEDBACK,
        )
    summary_text, usage = ask_with_tools(prompt, settings.planner_model, effort(settings.planner_effort),
                                         [train.TRY_TOOL, train.PROPOSE_TOOL], run, settings.max_lookups,
                                         uncounted=("propose_models",))
    return {"models_summary": summary_text, "models": proposal or state.get("models") or {},
            "round": state["round"], "ledger": ledger[len(state["ledger"]):],
            "costs": [cost_record("modeler", state, usage)]}


def model_reviewer(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    ledger = list(state["ledger"])
    this_round = [entry["text"] for entry in ledger if entry.get("by") == "modeler" and entry["round"] == state["round"]]
    prompt = MODEL_REVIEWER_PROMPT.format(**about_modeling(state), models=format_models(state["models"]),
                                          summary=state["models_summary"], lookups="\n\n".join(this_round) or "(none)",
                                          max_lookups=REVIEWER_LOOKUPS)
    review, usage = ask_with_tools(prompt, settings.reviewer_model, effort(settings.reviewer_effort),
                                   [train.TRY_TOOL], lookup_runner(state, ledger, "reviewer", settings.max_try_minutes),
                                   REVIEWER_LOOKUPS)
    return {"models_review": review, "ledger": ledger[len(state["ledger"]):],
            "costs": [cost_record("reviewer", state, usage)]}


def train_and_test(state: State, runtime: Runtime[Settings]) -> dict:
    """Code, not an agent: train the approved models, score them once on the test part and save them."""
    folder, models = Path(state["split_folder"]), state["models"]
    table, _ = train.load_split(folder)
    classifier, regressor = train.choices(models)
    uses = train.count_test_uses(folder)
    results, predictions, fitted = train.evaluate(table, folder, models["features"], classifier, regressor)
    saved = train.save(results, predictions, fitted, folder, models["features"], classifier, regressor, uses,
                       state["run_id"], models["reason"])
    entry = {**train.scores_entry(results, f"L{len(state['ledger']) + 1}"), "by": "train_and_test",
             "round": state["round"]}
    item = record(state, runtime, "models", [("Scores", f"Saved to {clean.shown(saved)}\n\n{entry['text']}")], [entry])
    return {"models_folder": str(saved), "scores_id": entry["id"], "ledger": [entry], "history": [item]}


# --- Stage 6: conclusions ---

def about_results(state: State) -> dict:
    return {"question": state["question"], "plan": state["plan"], "models": format_models(state["models"]),
            "scores": entry_text(state, state["scores_id"])}


def writer(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    state = {**state, "round": state["round"] + 1}
    if "conclusions" not in state:
        prompt = RESULTS_PROMPT.format(**about_results(state))
    else:
        prompt = RESULTS_REVISION_PROMPT.format(
            **about_results(state), conclusions=state["conclusions"], review=state["conclusions_review"],
            audit=state["audit_report"], feedback=state["results_feedback"] or NO_FEEDBACK)
    conclusions, usage = ask(prompt, settings.planner_model, effort(settings.planner_effort))
    return {"conclusions": conclusions, "round": state["round"], "costs": [cost_record("writer", state, usage)]}


def results_reviewer(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    prompt = RESULTS_REVIEWER_PROMPT.format(**about_results(state), conclusions=state["conclusions"])
    review, usage = ask(prompt, settings.reviewer_model, effort(settings.reviewer_effort))
    return {"conclusions_review": review, "costs": [cost_record("reviewer", state, usage)]}


def make_builder() -> StateGraph:
    plan = make_stage(
        "plan", planner, plan_reviewer, "plan_feedback",
        show=lambda state: {"Plan": state["plan"], "Review": state["review"]})
    get_data = make_stage(
        "source", sourcer, source_reviewer, "source_feedback",
        show=lambda state: {"Proposal": format_proposal(state["proposal"]), "Summary": state["source_summary"],
                            "Review": state["source_review"], "Audit": state["audit_report"]},
        audit_sections=lambda state: {"Summary": f"{state['proposal'].get('reason', '')}\n{state['source_summary']}",
                                      "Review": state["source_review"]},
        not_ready=lambda state: None if state["proposal"] else
        "You haven't proposed a download yet. Call propose_download.")
    explore = make_stage(
        "data", explorer, data_reviewer, "data_feedback",
        show=lambda state: {"Findings": state["findings"], "Review": state["data_review"],
                            "Audit": state["audit_report"]},
        audit_sections=lambda state: {"Findings": state["findings"], "Review": state["data_review"]})
    clean_data = make_stage(
        "clean", cleaner, clean_reviewer, "clean_feedback",
        show=lambda state: {"Recipe": format_recipe(state["recipe"]), "Summary": state["recipe_summary"],
                            "Review": state["clean_review"], "Audit": state["audit_report"]},
        audit_sections=lambda state: {"Summary": f"{state['recipe'].get('reason', '')}\n{state['recipe_summary']}",
                                      "Review": state["clean_review"]},
        not_ready=lambda state: None if state["recipe"] else "You haven't proposed a recipe yet. Call propose_recipe.")
    choose_models = make_stage(
        "model", modeler, model_reviewer, "models_feedback",
        show=lambda state: {"Models": format_models(state["models"]), "Summary": state["models_summary"],
                            "Review": state["models_review"], "Audit": state["audit_report"]},
        audit_sections=lambda state: {"Summary": f"{state['models'].get('reason', '')}\n{state['models_summary']}",
                                      "Review": state["models_review"]},
        not_ready=lambda state: None if state["models"] else "You haven't proposed models yet. Call propose_models.")
    conclude = make_stage(
        "results", writer, results_reviewer, "results_feedback",
        show=lambda state: {"Conclusions": state["conclusions"], "Review": state["conclusions_review"],
                            "Audit": state["audit_report"]},
        audit_sections=lambda state: {"Conclusions": state["conclusions"], "Review": state["conclusions_review"]})

    builder = StateGraph(State, context_schema=Settings)
    builder.add_node("use_dataset", use_dataset)
    builder.add_node("plan", plan)
    builder.add_node("get_data", get_data)
    builder.add_node("download", download)
    builder.add_node("explore", explore)
    builder.add_node("clean", clean_data)
    builder.add_node("make_dataset", make_dataset)
    builder.add_node("make_split", make_split)
    builder.add_node("model", choose_models)
    builder.add_node("train_and_test", train_and_test)
    builder.add_node("results", conclude)
    # Each stage ends either approved (go on) or stopped at the budget (end the run)
    builder.add_conditional_edges(START, lambda state: "use_dataset" if state.get("start_from") else "plan",
                                  ["use_dataset", "plan"])
    builder.add_edge("use_dataset", "plan")
    # A run that started from a saved dataset goes from the plan straight to the split
    builder.add_conditional_edges("plan", lambda state: END if state.get("stopped") else
                                  "make_split" if state.get("start_from") else "get_data",
                                  ["get_data", "make_split", END])
    builder.add_conditional_edges("get_data", lambda state: END if state.get("stopped") else "download", ["download", END])
    builder.add_edge("download", "explore")
    builder.add_conditional_edges("explore", lambda state: END if state.get("stopped") else "clean", ["clean", END])
    builder.add_conditional_edges("clean", lambda state: END if state.get("stopped") else "make_dataset",
                                  ["make_dataset", END])
    builder.add_edge("make_dataset", "make_split")
    builder.add_edge("make_split", "model")
    builder.add_conditional_edges("model", lambda state: END if state.get("stopped") else "train_and_test",
                                  ["train_and_test", END])
    builder.add_edge("train_and_test", "results")
    builder.add_edge("results", END)
    return builder


def build_graph():
    # LangGraph Studio supplies its own checkpointer for saving paused runs
    return make_builder().compile()


if __name__ == "__main__":
    # Usage: python -m cb2.graph [--planner-model gemma4:26b] [--budget 0.50] [--from data/clean/<folder>] ...
    parser = argparse.ArgumentParser(description="Run the research team from the terminal.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--planner-model", default=Settings.planner_model, choices=list(MODELS))
    parser.add_argument("--planner-effort", default="default", choices=["default", *EFFORTS])
    parser.add_argument("--reviewer-model", default=Settings.reviewer_model, choices=list(MODELS))
    parser.add_argument("--reviewer-effort", default="default", choices=["default", *EFFORTS])
    parser.add_argument("--auditor-model", default=Settings.auditor_model, choices=list(MODELS))
    parser.add_argument("--max-lookups", type=int, default=Settings.max_lookups)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--fresh-download", action="store_true")
    start.add_argument("--from", dest="start_from", metavar="data/clean/<folder>",
                       help="start from a saved clean dataset instead of downloading and cleaning")
    parser.add_argument("--practice", action="store_true", help="log to practice-runs/, which git ignores")
    parser.add_argument("--max-try-minutes", type=float, default=Settings.max_try_minutes)
    parser.add_argument("--auto-approve", action="store_true",
                        help="answer every approval pause for you (budget pauses still ask)")
    parser.add_argument("--auto-plan-rounds", type=int, default=Settings.auto_plan_rounds,
                        help="with --auto-approve, rounds of planning and review before the plan is approved")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="dollars")
    args = parser.parse_args()
    settings = Settings(args.planner_model, args.planner_effort, args.reviewer_model, args.reviewer_effort,
                        args.auditor_model, args.max_lookups, args.fresh_download, args.practice,
                        args.max_try_minutes, args.auto_approve, args.auto_plan_rounds)

    graph = make_builder().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "cli"}}
    inputs = {"question": args.question, "budget": args.budget}
    if args.start_from:
        inputs["start_from"] = args.start_from
    result = graph.invoke(inputs, config, context=settings)
    # Each pause shows what the team produced and asks for an answer; resuming runs to the next pause
    while "__interrupt__" in result:
        pause_value = result["__interrupt__"][0].value
        for heading, text in pause_value.items():
            if heading not in ("kind", "How to answer"):
                print(f"\n=== {heading.upper()} ===\n{text}")
        prompt = "> " if pause_value["kind"] == "budget" else f"\n{ANSWER_PROMPT}: "
        result = graph.invoke(Command(resume=input(prompt)), config, context=settings)

    status = "stopped" if result.get("stopped") else "approved"
    log_folder = "practice-runs" if settings.practice else "runs"
    print(f"\nFinished ({status}, ${spent(result.get('costs', [])):.3f} spent). "
          f"Log: {log_folder}/{result['run_id']}/log.md")
    for name, key in [("Dataset", "clean_folder"), ("Split", "split_folder"), ("Models", "models_folder")]:
        if result.get(key):
            print(f"{name}: {result[key]}")
