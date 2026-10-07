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

REVIEWER_PROMPT = """You are a critical reviewer of computational chemistry research plans.

Question: {question}

Plan:
{plan}

List the three most important weaknesses or risks in this plan, one or two sentences each."""


# The shared notebook every node reads from and writes to
class State(TypedDict, total=False):
    question: str
    plan: str
    review: str
    human_feedback: str


def planner(state: State) -> dict:
    # Return only the entries this node changes
    return {"plan": ask_claude(PLANNER_PROMPT.format(question=state["question"]))}


def reviewer(state: State) -> dict:
    prompt = REVIEWER_PROMPT.format(question=state["question"], plan=state["plan"])
    return {"review": ask_ollama(prompt)}


def human_approval(state: State) -> dict:
    # Pause the run until a person answers; their answer becomes the return value
    answer = interrupt({"plan": state["plan"], "review": state["review"]})
    return {"human_feedback": str(answer).strip()}


def make_builder() -> StateGraph:
    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("reviewer", reviewer)
    builder.add_node("human_approval", human_approval)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "reviewer")
    builder.add_edge("reviewer", "human_approval")
    builder.add_edge("human_approval", END)
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
    print("\n=== PLAN (Claude) ===\n" + result["plan"])
    print("\n=== REVIEW (local model) ===\n" + result["review"])

    answer = input("\nType 'approve', or write feedback for the planner: ")
    result = graph.invoke(Command(resume=answer), config)
    print("\nRecorded: " + result["human_feedback"])
