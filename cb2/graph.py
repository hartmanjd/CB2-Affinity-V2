"""The research team's workflow, built as a LangGraph graph."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from cb2.llm import ask_claude

PLANNER_PROMPT = """You are planning a small computational chemistry study.
Write a short research plan (3 to 5 numbered steps) for this question:

{question}"""


# The shared notebook every node reads from and writes to
class State(TypedDict, total=False):
    question: str
    plan: str


def planner(state: State) -> dict:
    # Return only the entries this node changes
    return {"plan": ask_claude(PLANNER_PROMPT.format(question=state["question"]))}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    for edge in graph.get_graph().edges:
        print(f"{edge.source} -> {edge.target}")
    print()

    result = graph.invoke(
        {"question": "Which ligand features predict high binding affinity at the CB2 receptor?"}
    )
    print(result["plan"])
