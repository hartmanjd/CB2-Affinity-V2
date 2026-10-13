"""Whole runs with stand-in models: no API calls, no internet. The stand-ins call the same tools a
real model would, so everything around them (the loop, the tools, the dataset) is the real code."""

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from cb2 import clean, graph, runlog
from cb2.llm import usage_record
from tests.fakes import TARGET, activity
from tests.test_clean import RECIPE, ROWS

USAGE = usage_record("claude-haiku-5-5", 0.1, 10, 10)


@pytest.fixture
def team(save, tmp_path, monkeypatch):
    """Stand-in models, a made-up download, and every output written under tmp_path."""
    folder = save([activity(i, **fields) for i, (fields, _) in ROWS.items() if i != 6])
    monkeypatch.setattr(graph.data, "find_download", lambda target_ids: folder)
    monkeypatch.setattr(graph.chembl, "describe_targets", lambda target_ids: [TARGET])
    monkeypatch.setattr(graph, "audit", lambda sections, ledger, model: ("audit report", None))
    monkeypatch.setattr(graph, "ask", lambda prompt, model, effort, **kw: ("a plan or review", USAGE))
    monkeypatch.setattr(runlog, "PRACTICE_RUNS", tmp_path / "practice-runs")
    monkeypatch.setattr(clean, "CLEAN", tmp_path / "clean")
    recipes = [{**RECIPE, "reason": "binding data only"}]

    def ask_with_tools(prompt, model, effort, tools, run_tool, max_tool_calls, uncounted=()):
        names = {tool["name"] for tool in tools}
        if "propose_download" in names:
            run_tool("propose_download", {"target_ids": ["CHEMBL253"], "reason": "human CB2"})
        elif "propose_recipe" in names:
            tried = run_tool("try_recipe", RECIPE)
            assert "3 compounds" in tried
            recipes.append(run_tool("propose_recipe", recipes[0]))
        elif "try_recipe" in names:
            run_tool("try_recipe", {**RECIPE, "max_spread": None})
        elif "look" in names:
            run_tool("look", {"kind": "value_counts", "column": "standard_type"})
        return "summary [L1]", USAGE

    monkeypatch.setattr(graph, "ask_with_tools", ask_with_tools)
    return recipes


def run(answers: list[str]):
    """Run the whole graph, answering each pause in turn; returns the final state and the pauses seen."""
    app = graph.make_builder().compile(checkpointer=InMemorySaver())
    config, settings = {"configurable": {"thread_id": "test"}}, graph.Settings(practice=True)
    result, pauses = app.invoke({"question": "q"}, config, context=settings), []
    for answer in answers:
        pauses.append(result["__interrupt__"][0].value)
        result = app.invoke(Command(resume=answer), config, context=settings)
    assert "__interrupt__" not in result
    return result, pauses


def test_an_approved_recipe_makes_the_dataset(team, tmp_path):
    result, pauses = run(["approve"] * 4)
    assert [pause["kind"] for pause in pauses] == ["plan", "source", "data", "clean"]
    assert team[1].startswith("Recipe recorded")
    # The lead sees what the recipe does, computed by code
    assert "-> 3 compounds" in pauses[-1]["Recipe"]

    manifest = json.loads((Path(result["clean_folder"]) / "manifest.json").read_text())
    assert manifest["reason"] == "binding data only" and manifest["run"] == result["run_id"]
    # Row 6 needs a second assay, so this download leaves it out: 15 records, 11 dropped
    assert manifest["rows"] == {"compounds": 3, "dropped": 11}
    try_entry = next(entry for entry in result["ledger"] if entry["request"]["kind"] == "try_recipe")
    assert try_entry["numbers"]["compounds"] == 3 and try_entry["by"] == "cleaner"

    log = (tmp_path / "practice-runs" / result["run_id"] / "log.md").read_text()
    assert "Round 4 · Clean the data" in log and "## The dataset" in log and "Saved 3 compounds" in log


def test_stopping_at_the_recipe_makes_no_dataset(team, tmp_path):
    result, _ = run(["approve"] * 3 + ["stop"])
    assert result["stopped"] and "clean_folder" not in result
    assert not (tmp_path / "clean").exists()


def test_feedback_on_the_recipe_goes_round_again(team):
    result, pauses = run(["approve"] * 3 + ["use a looser spread", "approve"])
    assert [pause["kind"] for pause in pauses][-2:] == ["clean", "clean"]
    assert result["clean_feedback"] == "approve" and result["clean_folder"]


def test_an_off_menu_recipe_is_not_recorded(team):
    team[0] = {**RECIPE, "max_spread": 3.0, "reason": "keep everything"}
    result, pauses = run(["approve"] * 3 + ["approve", "stop"])
    assert team[1].startswith("Not proposed: max_spread must be one of")
    # Approving without a recipe sends the planner back to propose one
    assert pauses[-1]["Recipe"] == "(no recipe proposed)" and result["clean_feedback"] == "stop"
