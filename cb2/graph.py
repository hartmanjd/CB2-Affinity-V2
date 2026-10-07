"""The research team's workflow, built as a LangGraph graph."""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from cb2.llm import ask_claude, ask_ollama

PLANNER_PROMPT = """You are planning a small computational chemistry study of the
cannabinoid receptor 2 (CB2). The study will only use published binding affinity
data (such as Ki and IC50 values) downloaded from ChEMBL and analysed in Python.

Write a short research plan (3 to 5 numbered steps) for this question:

{question}"""

REVISION_PROMPT = PLANNER_PROMPT + """

Your previous plan:
{plan}

A reviewer's critique of it:
{review}

Feedback from the project lead:
{feedback}

Write an improved plan that addresses the lead's feedback and the most important
points of the critique. If a point in the critique is wrong or doesn't fit the study,
say so briefly and don't follow it."""

REVIEWER_PROMPT = """You are a critical reviewer of computational chemistry research plans.

Question: {question}

Plan:
{plan}

List the three most important weaknesses or risks in this plan, one or two sentences each."""

MAX_REVISIONS = 2

# Used when the lead leaves the answer blank: revise from the critique alone
NO_FEEDBACK = "None given. Address the most important points of the critique."


# The shared notebook every node reads from and writes to
class State(TypedDict, total=False):
    question: str
    plan: str
    review: str
    human_feedback: str
    approved: bool
    revisions: int


def planner(state: State) -> dict:
    # First pass: plan from the question. Later passes: revise the existing plan
    if "plan" not in state:
        return {"plan": ask_claude(PLANNER_PROMPT.format(question=state["question"])), "revisions": 0}
    prompt = REVISION_PROMPT.format(
        question=state["question"],
        plan=state["plan"],
        review=state["review"],
        feedback=state["human_feedback"] or NO_FEEDBACK,
    )
    return {"plan": ask_claude(prompt), "revisions": state["revisions"] + 1}


def reviewer(state: State) -> dict:
    prompt = REVIEWER_PROMPT.format(question=state["question"], plan=state["plan"])
    return {"review": ask_ollama(prompt)}


def human_approval(state: State) -> dict:
    # Pause the run until a person answers; their answer becomes the return value
    answer = str(interrupt({"plan": state["plan"], "review": state["review"]}) or "").strip()
    return {"human_feedback": answer, "approved": answer.lower() == "approve"}


def after_approval(state: State) -> str:
    # Finish when approved or out of revision rounds; otherwise send it back to the planner
    if state["approved"] or state["revisions"] >= MAX_REVISIONS:
        return END
    return "planner"


def make_builder() -> StateGraph:
    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("reviewer", reviewer)
    builder.add_node("human_approval", human_approval)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "reviewer")
    builder.add_edge("reviewer", "human_approval")
    builder.add_conditional_edges("human_approval", after_approval, ["planner", END])
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
        print(f"\n=== PLAN (Claude, revision {result['revisions']}) ===\n" + result["plan"])
        print("\n=== REVIEW (local model) ===\n" + result["review"])
        answer = input("\nType 'approve', write feedback, or press Enter to revise from the review: ")
        result = graph.invoke(Command(resume=answer), config)

    status = "approved" if result["approved"] else f"stopped after {MAX_REVISIONS} revisions"
    print(f"\nFinal plan ({status}):\n" + result["plan"])
