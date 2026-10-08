"""The research team's workflow, built as a LangGraph graph."""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from cb2.costs import round_cost, spent, summary
from cb2.llm import ask_claude, ask_openai

PLANNER_PROMPT = """You are planning a small computational chemistry study of the
cannabinoid receptor 2 (CB2). The study will only use published binding affinity
data downloaded from ChEMBL and analysed in Python.

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

REVIEWER_PROMPT = """You are a critical reviewer of computational chemistry research plans.

Question: {question}

Plan:
{plan}

List the three most important weaknesses or risks in this plan, one or two sentences each."""

# Dollars a run may spend before it pauses to ask for more; pass "budget" in the input to change it
DEFAULT_BUDGET = 2.00

# Used when the lead leaves the answer blank: revise from the critique alone
NO_FEEDBACK = "None given. Address the most important points of the critique."


# The shared notebook every node reads from and writes to
class State(TypedDict, total=False):
    question: str
    plan: str
    review: str
    human_feedback: str
    approved: bool
    stopped: bool
    revisions: int
    budget: float
    # One usage record per model call; operator.add means each node's records are appended
    costs: Annotated[list[dict], operator.add]


def budget_gate(state: State) -> Command[Literal["planner", "budget_gate", "__end__"]]:
    # Before each round, guess it will cost about what the last one did (nothing to go on before the first)
    budget = state.get("budget", DEFAULT_BUDGET)
    costs = state.get("costs", [])
    estimate = round_cost(costs, state["revisions"]) if "plan" in state else 0.0
    if spent(costs) + estimate <= budget:
        return Command(update={"budget": budget}, goto="planner")

    # Over budget: pause and let the person decide whether the next round is worth it
    message = (f"Spent ${spent(costs):.3f} of the ${budget:.2f} budget, and the next round "
               f"should cost about ${estimate:.3f}. Type a new budget in dollars to continue, "
               f"or anything else to stop.")
    answer = str(interrupt({"kind": "budget", "message": message}) or "").strip()
    try:
        new_budget = float(answer.removeprefix("$"))
    except ValueError:
        return Command(update={"stopped": True}, goto=END)
    # Check again, in case the new budget still doesn't cover the next round
    return Command(update={"budget": new_budget}, goto="budget_gate")


def planner(state: State) -> dict:
    # First pass: plan from the question. Later passes: revise the existing plan
    if "plan" not in state:
        prompt = PLANNER_PROMPT.format(question=state["question"])
        revisions = 0
    else:
        prompt = REVISION_PROMPT.format(
            question=state["question"],
            plan=state["plan"],
            review=state["review"],
            feedback=state["human_feedback"] or NO_FEEDBACK,
        )
        revisions = state["revisions"] + 1
    plan, usage = ask_claude(prompt)
    return {"plan": plan, "revisions": revisions, "costs": [{"node": "planner", "round": revisions, **usage}]}


def reviewer(state: State) -> dict:
    review, usage = ask_openai(REVIEWER_PROMPT.format(question=state["question"], plan=state["plan"]))
    return {"review": review, "costs": [{"node": "reviewer", "round": state["revisions"], **usage}]}


def human_approval(state: State) -> dict:
    # Pause the run until a person answers; their answer becomes the return value
    shown = {
        "kind": "approval",
        "plan": state["plan"],
        "review": state["review"],
        "costs": summary(state["costs"], state["revisions"], state["budget"]),
    }
    answer = str(interrupt(shown) or "").strip()
    return {"human_feedback": answer, "approved": answer.lower() == "approve"}


def after_approval(state: State) -> str:
    # Finish when approved; otherwise check the budget and go round again
    return END if state["approved"] else "budget_gate"


def make_builder() -> StateGraph:
    builder = StateGraph(State)
    builder.add_node("budget_gate", budget_gate)
    builder.add_node("planner", planner)
    builder.add_node("reviewer", reviewer)
    builder.add_node("human_approval", human_approval)
    builder.add_edge(START, "budget_gate")
    builder.add_edge("planner", "reviewer")
    builder.add_edge("reviewer", "human_approval")
    builder.add_conditional_edges("human_approval", after_approval, ["budget_gate", END])
    return builder


def build_graph():
    # LangGraph Studio supplies its own checkpointer for saving paused runs
    return make_builder().compile()


if __name__ == "__main__":
    graph = make_builder().compile(checkpointer=InMemorySaver())
    for edge in graph.get_graph().edges:
        print(f"{edge.source} -> {edge.target}")

    config = {"configurable": {"thread_id": "cli"}}
    result = graph.invoke(
        {"question": "Which ligand features predict high binding affinity at the CB2 receptor?"},
        config,
    )
    # Each pause asks for an answer; resuming runs until the next pause or the end
    while "__interrupt__" in result:
        pause = result["__interrupt__"][0].value
        if pause["kind"] == "budget":
            answer = input(f"\n=== BUDGET ===\n{pause['message']}\n> ")
        else:
            print(f"\n=== PLAN (Claude, revision {result['revisions']}) ===\n" + pause["plan"])
            print("\n=== REVIEW (OpenAI) ===\n" + pause["review"])
            print("\n=== COSTS ===\n" + pause["costs"])
            answer = input("\nType 'approve', write feedback, or press Enter to revise from the review: ")
        result = graph.invoke(Command(resume=answer), config)

    status = "approved" if result.get("approved") else "stopped at the budget"
    print(f"\nFinal plan ({status}, ${spent(result['costs']):.3f} spent):\n" + result["plan"])
