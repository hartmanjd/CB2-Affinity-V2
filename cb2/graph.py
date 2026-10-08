"""The research team's workflow, built as a LangGraph graph.

Stage 1, plan: the planner writes a research plan and the reviewer critiques it.
Stage 2, get the data: the planner searches ChEMBL and proposes which targets to download;
the reviewer critiques the choice (with searches of its own) and the auditor checks its numbers.
Stage 3, explore the data: the planner explores the download with looks and writes findings;
the reviewer checks them (with looks of its own) and the auditor checks every number.
Each stage ends with the lead approving or giving feedback, every pause is written to the run
log in runs/, and a budget gate runs before every round."""

import argparse
import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

import requests
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from cb2 import chembl, data, runlog
from cb2.audit import audit
from cb2.costs import round_cost, spent, summary
from cb2.llm import EFFORTS, MODELS, ask, ask_with_tools, check_model
from cb2.looks import LOOK_TOOL, LookError, look, overview

# The prompts only say the study uses ChEMBL data; the question decides what it is about
DEFAULT_QUESTION = "Which ligand features predict high binding affinity at the CB2 receptor?"

PLANNER_PROMPT = """You are planning a small study that will only use data downloaded from
ChEMBL and analysed in Python.

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

List the three most important weaknesses or risks in this plan, one or two sentences each."""

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

You can make up to {max_lookups} searches and counts of your own. Then list the three most
important problems with what is proposed (or left out), one or two sentences each. Cite results
in brackets, like [L3], for any numbers."""

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

# Dollars a run may spend before it pauses to ask for more; pass "budget" in the input to change it
DEFAULT_BUDGET = 2.00
REVIEWER_LOOKUPS = 5

# Used when the lead leaves the answer blank: revise from the critique alone
NO_FEEDBACK = "None given. Address the most important points of the critique."
ANSWER_PROMPT = "Type 'approve', write feedback, or press Enter to revise from the review"


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


def effort(value: str) -> str | None:
    # "default" means don't send an effort, so the model uses its own default
    return None if value == "default" else value


# The shared notebook every node reads from and writes to
class State(TypedDict, total=False):
    question: str
    run_id: str
    stage: Literal["plan", "source", "data"]
    # Counts rounds across all stages; every cost record says which round it belongs to
    round: int
    budget: float
    stopped: bool
    # Lists that grow: operator.add appends what each node returns
    costs: Annotated[list[dict], operator.add]      # one usage record per model call or tool loop
    ledger: Annotated[list[dict], operator.add]     # every search, count and look, numbered L1, L2, ...
    history: Annotated[list[dict], operator.add]    # every round and the lead's reply, for the run log
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
    findings_approved: bool


def cost_record(node: str, state: State, usage: dict) -> dict:
    return {"node": node, "stage": state["stage"], "round": state["round"], **usage}


def pause(state: State, runtime: Runtime[Settings], kind: str, sections: dict[str, str]) -> tuple[str, dict]:
    """Show sections to the lead and wait for an answer; log the round before and after."""
    item = runlog.entry(kind, state.get("round", 0), list(sections.items()))
    history = state.get("history", [])
    args = (state["run_id"], state["question"], runtime.context or Settings())
    runlog.write(*args, history + [item], state.get("ledger", []))
    answer = str(interrupt({"kind": kind, **sections}) or "").strip()
    item["feedback"] = answer
    runlog.write(*args, history + [item], state.get("ledger", []))
    return answer, item


def budget_gate(state: State, runtime: Runtime[Settings]) -> Command[Literal["planner", "sourcer", "explorer", "budget_gate", "__end__"]]:
    settings = runtime.context or Settings()
    for role in ("planner", "reviewer", "auditor"):
        check_model(role, getattr(settings, f"{role}_model"))
    stage = state.get("stage", "plan")
    worker = {"plan": "planner", "source": "sourcer", "data": "explorer"}[stage]
    defaults = {
        "budget": state.get("budget", DEFAULT_BUDGET),
        "stage": stage,
        "question": state.get("question") or DEFAULT_QUESTION,
        "run_id": state.get("run_id") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
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


def lookup_runner(state: State, ledger: list[dict], by: str):
    """A run_tool function for ask_with_tools: runs searches, counts and looks, numbering each
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
            else:
                return f"Unknown tool {name!r}."
        except (ValueError, LookError) as error:
            return f"{name} error: {error}"
        except requests.RequestException as error:
            return f"{name} error: ChEMBL didn't answer ({error}). Try again or continue without it."
        ledger.append({**entry, "by": by, "round": state["round"]})
        return entry["text"]
    return run


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


def plan_approval(state: State, runtime: Runtime[Settings]) -> dict:
    answer, item = pause(state, runtime, "plan", {
        "Plan": state["plan"], "Review": state["review"],
        "Costs": summary(state["costs"], state["round"], state["budget"]),
    })
    # Approved: move on to getting the data. Either way the budget gate decides what runs next
    return {"plan_feedback": answer, "history": [item],
            **({"stage": "source"} if answer.lower() == "approve" else {})}


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


def source_approval(state: State, runtime: Runtime[Settings]) -> Command[Literal["download", "budget_gate"]]:
    answer, item = pause(state, runtime, "source", {
        "Proposal": format_proposal(state["proposal"]), "Summary": state["source_summary"],
        "Review": state["source_review"], "Audit": state["audit_report"],
        "Costs": summary(state["costs"], state["round"], state["budget"]),
    })
    update = {"source_feedback": answer, "history": [item]}
    if answer.lower() == "approve" and state["proposal"]:
        return Command(update=update, goto="download")
    if answer.lower() == "approve":
        # Nothing to approve yet: ask the planner for a proposal
        update["source_feedback"] = "You haven't proposed a download yet. Call propose_download."
    return Command(update=update, goto="budget_gate")


def download(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    target_ids = state["proposal"]["target_ids"]
    folder = None if settings.fresh_download else data.find_download(target_ids)
    if folder is None:
        folder = data.download(target_ids, state["proposal"]["reason"], state["run_id"])
    # The column names and row count are all the agents know before their first look
    entry = overview(data.load(str(folder)), f"L{len(state.get('ledger', [])) + 1}")
    return {"stage": "data", "data_folder": str(folder), "overview_id": entry["id"],
            "ledger": [{**entry, "by": "download", "round": state["round"]}]}


# --- Stage 3: explore the data ---

def explorer(state: State, runtime: Runtime[Settings]) -> dict:
    settings = runtime.context or Settings()
    state = {**state, "round": state["round"] + 1}
    ledger = list(state["ledger"])
    table_overview = next(entry["text"] for entry in ledger if entry["id"] == state["overview_id"])
    targets = ", ".join(f"{t['target_chembl_id']} ({t['pref_name']}, {t['organism'] or '-'}, {t['target_type']})"
                        for t in state["proposal"]["targets"])
    fields = dict(question=state["question"], plan=state["plan"], targets=targets,
                  overview=table_overview, max_lookups=settings.max_lookups)
    if "findings" not in state:
        prompt = EXPLORE_PROMPT.format(**fields)
    else:
        prompt = EXPLORE_REVISION_PROMPT.format(
            **fields, findings=state["findings"], look_list=lookup_list(ledger, {"value_counts", "crosstab", "describe", "sample"}),
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


def auditor(state: State, runtime: Runtime[Settings]) -> Command[Literal["source_approval", "data_approval"]]:
    settings = runtime.context or Settings()
    if state["stage"] == "source":
        reason = (state.get("proposal") or {}).get("reason", "")
        sections, goto = {"Summary": f"{reason}\n{state['source_summary']}", "Review": state["source_review"]}, "source_approval"
    else:
        sections, goto = {"Findings": state["findings"], "Review": state["data_review"]}, "data_approval"
    report, usage = audit(sections, state["ledger"], settings.auditor_model)
    return Command(update={"audit_report": report, "costs": [cost_record("auditor", state, usage)] if usage else []},
                   goto=goto)


def data_approval(state: State, runtime: Runtime[Settings]) -> dict:
    answer, item = pause(state, runtime, "data", {
        "Findings": state["findings"], "Review": state["data_review"], "Audit": state["audit_report"],
        "Costs": summary(state["costs"], state["round"], state["budget"]),
    })
    return {"data_feedback": answer, "findings_approved": answer.lower() == "approve", "history": [item]}


def after_data_approval(state: State) -> str:
    return END if state["findings_approved"] else "budget_gate"


def make_builder() -> StateGraph:
    builder = StateGraph(State, context_schema=Settings)
    for node in (budget_gate, planner, plan_reviewer, plan_approval, sourcer, source_reviewer, source_approval,
                 download, explorer, data_reviewer, auditor, data_approval):
        builder.add_node(node.__name__, node)
    builder.add_edge(START, "budget_gate")
    builder.add_edge("planner", "plan_reviewer")
    builder.add_edge("plan_reviewer", "plan_approval")
    builder.add_edge("plan_approval", "budget_gate")
    builder.add_edge("sourcer", "source_reviewer")
    builder.add_edge("source_reviewer", "auditor")
    builder.add_edge("download", "budget_gate")
    builder.add_edge("explorer", "data_reviewer")
    builder.add_edge("data_reviewer", "auditor")
    builder.add_conditional_edges("data_approval", after_data_approval, ["budget_gate", END])
    return builder


def build_graph():
    # LangGraph Studio supplies its own checkpointer for saving paused runs
    return make_builder().compile()


if __name__ == "__main__":
    # Usage: python -m cb2.graph [--planner-model gemma4:26b] [--budget 0.50] ...
    parser = argparse.ArgumentParser(description="Run the research team from the terminal.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--planner-model", default=Settings.planner_model, choices=list(MODELS))
    parser.add_argument("--planner-effort", default="default", choices=["default", *EFFORTS])
    parser.add_argument("--reviewer-model", default=Settings.reviewer_model, choices=list(MODELS))
    parser.add_argument("--reviewer-effort", default="default", choices=["default", *EFFORTS])
    parser.add_argument("--auditor-model", default=Settings.auditor_model, choices=list(MODELS))
    parser.add_argument("--max-lookups", type=int, default=Settings.max_lookups)
    parser.add_argument("--fresh-download", action="store_true")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="dollars")
    args = parser.parse_args()
    settings = Settings(args.planner_model, args.planner_effort, args.reviewer_model, args.reviewer_effort,
                        args.auditor_model, args.max_lookups, args.fresh_download)

    graph = make_builder().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "cli"}}
    result = graph.invoke({"question": args.question, "budget": args.budget}, config, context=settings)
    # Each pause shows what the team produced and asks for an answer; resuming runs to the next pause
    while "__interrupt__" in result:
        pause_value = result["__interrupt__"][0].value
        for heading, text in pause_value.items():
            if heading != "kind":
                print(f"\n=== {heading.upper()} ===\n{text}")
        prompt = "> " if pause_value["kind"] == "budget" else f"\n{ANSWER_PROMPT}: "
        result = graph.invoke(Command(resume=input(prompt)), config, context=settings)

    status = "approved" if result.get("findings_approved") else "stopped at the budget"
    print(f"\nFinished ({status}, ${spent(result.get('costs', [])):.3f} spent). Log: runs/{result['run_id']}/log.md")
