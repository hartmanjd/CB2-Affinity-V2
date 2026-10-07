"""The research team's workflow, built as a LangGraph graph."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

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


def planner(state: State) -> dict:
    # Return only the entries this node changes
    return {"plan": ask_claude(PLANNER_PROMPT.format(question=state["question"]))}


def reviewer(state: State) -> dict:
    prompt = REVIEWER_PROMPT.format(question=state["question"], plan=state["plan"])
    return {"review": ask_ollama(prompt)}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("reviewer", reviewer)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "reviewer")
    builder.add_edge("reviewer", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    for edge in graph.get_graph().edges:
        print(f"{edge.source} -> {edge.target}")

    result = graph.invoke(
        {"question": "Which ligand features predict high binding affinity at the CB2 receptor?"}
    )
    print("\n=== PLAN (Claude) ===\n" + result["plan"])
    print("\n=== REVIEW (local model) ===\n" + result["review"])
