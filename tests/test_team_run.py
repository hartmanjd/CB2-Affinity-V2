"""Whole runs with stand-in models: no API calls, no internet. The stand-ins call the same tools a
real model would, so everything around them (the loop, the tools, the dataset, the models) is the
real code."""

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from cb2 import graph, runlog, split, train
from cb2.llm import usage_record
from tests.fakes import TARGET, activity
from tests.test_clean import RECIPE, ROWS

USAGE = usage_record("claude-haiku-5-5", 0.1, 10, 10)
MODELS = {"features": "morgan2", "classifier": {"model": "LogisticRegression", "settings": {"C": [0.1, 1.0]}},
          "regressor": {"model": "Ridge", "settings": {"alpha": [1.0, 10.0]}}, "reason": "simple models first"}


@pytest.fixture
def team(save, clean, tmp_path, monkeypatch):
    """Stand-in models, a made-up download, and every output written under tmp_path. The made-up
    download cleans to only 3 compounds, too few to split, so the split is made from 47."""
    folder = save([activity(i, **fields) for i, (fields, _) in ROWS.items() if i != 6])
    monkeypatch.setattr(graph.data, "find_download", lambda target_ids: folder)
    monkeypatch.setattr(graph.chembl, "describe_targets", lambda target_ids: [TARGET])
    monkeypatch.setattr(split, "load_clean", lambda source: (clean, {"recipe": {"max_spread": 1.0}}))
    monkeypatch.setattr(graph, "audit", lambda sections, ledger, model: ("audit report", None))
    monkeypatch.setattr(graph, "ask", lambda prompt, model, effort, **kw: ("a plan or review", USAGE))
    monkeypatch.setattr(runlog, "PRACTICE_RUNS", tmp_path / "practice-runs")
    monkeypatch.setattr(graph.clean, "CLEAN", tmp_path / "clean")
    monkeypatch.setattr(split, "SPLIT", tmp_path / "split")
    monkeypatch.setattr(train, "TEST_LOG", tmp_path / "test-use.jsonl")
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train, "BOOTSTRAPS", 100)
    # What the stand-in planner proposes, and the replies it gets
    team = {"recipe": {**RECIPE, "reason": "binding data only"}, "models": MODELS, "replies": []}

    def ask_with_tools(prompt, model, effort, tools, run_tool, max_tool_calls, uncounted=()):
        names = {tool["name"] for tool in tools}
        if "propose_download" in names:
            run_tool("propose_download", {"target_ids": ["CHEMBL253"], "reason": "human CB2"})
        elif "propose_recipe" in names:
            assert "3 compounds" in run_tool("try_recipe", RECIPE)
            team["replies"].append(run_tool("propose_recipe", team["recipe"]))
        elif "try_recipe" in names:
            run_tool("try_recipe", {**RECIPE, "max_spread": None})
        elif "propose_models" in names:
            assert "best settings" in run_tool("try_models", {"features": "morgan2", "task": "regressor",
                                                              "model": "Ridge", "settings": {"alpha": [1.0, 10.0]}})
            team["replies"].append(run_tool("propose_models", team["models"]))
        elif "try_models" in names:
            run_tool("try_models", {"features": "morgan2", "task": "classifier", "model": "LogisticRegression"})
        elif "look" in names:
            run_tool("look", {"kind": "value_counts", "column": "standard_type"})
        return "summary [L1]", USAGE

    monkeypatch.setattr(graph, "ask_with_tools", ask_with_tools)
    return team


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


def ledger_kinds(result) -> list[tuple[str, str]]:
    return [(entry["request"]["kind"], entry["by"]) for entry in result["ledger"]]


def test_an_approved_recipe_makes_the_dataset(team, tmp_path):
    result, pauses = run(["approve"] * 4 + ["stop"])
    assert [pause["kind"] for pause in pauses] == ["plan", "source", "data", "clean", "model"]
    assert team["replies"][0].startswith("Recipe recorded")
    # The lead sees what the recipe does, computed by code
    assert "-> 3 compounds" in pauses[3]["Recipe"]

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
    result, pauses = run(["approve"] * 3 + ["use a looser spread", "approve", "stop"])
    assert [pause["kind"] for pause in pauses][-3:] == ["clean", "clean", "model"]
    assert result["clean_feedback"] == "approve" and result["clean_folder"]


def test_an_off_menu_recipe_is_not_recorded(team):
    team["recipe"] = {**RECIPE, "max_spread": 3.0, "reason": "keep everything"}
    result, pauses = run(["approve"] * 3 + ["approve", "stop"])
    assert team["replies"][0].startswith("Not proposed: max_spread must be one of")
    # Approving without a recipe sends the planner back to propose one
    assert pauses[-1]["Recipe"] == "(no recipe proposed)" and result["clean_feedback"] == "stop"


def test_an_approved_run_ends_with_trained_models(team, tmp_path):
    result, pauses = run(["approve"] * 6)
    assert [pause["kind"] for pause in pauses] == ["plan", "source", "data", "clean", "model", "results"]
    assert team["replies"][1].startswith("Models recorded")
    # The lead sees practice scores computed by code, next to the baselines
    assert "Practice scores" in pauses[4]["Models"] and "nearest neighbour" in pauses[4]["Models"]
    assert "Training and testing should take about" in pauses[4]["Models"]

    manifest = json.loads((Path(result["models_folder"]) / "manifest.json").read_text())
    assert manifest["run"] == result["run_id"] and manifest["reason"] == "simple models first"
    assert manifest["choice"]["regressor"] == {"task": "regressor", "model": "Ridge", "settings": {"alpha": [1.0, 10.0]}}
    assert manifest["source"]["folder"] == str(Path(result["split_folder"]))
    # The test part was opened once per split, by code after the approval
    assert len(train.TEST_LOG.read_text().splitlines()) == 2

    kinds = ledger_kinds(result)
    assert {("split", "split"), ("try_models", "modeler"), ("try_models", "reviewer"),
            ("test_scores", "train_and_test")} <= set(kinds)
    split_entry, scores = (next(e for e in result["ledger"] if e["request"]["kind"] == kind)
                           for kind in ("split", "test_scores"))
    assert split_entry["numbers"]["molecules"] == 47
    assert scores["id"] == result["scores_id"] and "scaffold | pKi | Ridge | rmse" in scores["numbers"]

    log = (tmp_path / "practice-runs" / result["run_id"] / "log.md").read_text()
    for heading in ["## The split", "Round 5 · Choose the models", "## The trained models", "Round 6 · Conclusions"]:
        assert heading in log


def test_stopping_at_the_models_never_opens_the_test_part(team, tmp_path):
    result, _ = run(["approve"] * 4 + ["stop"])
    # The team tried models on practice scores, but nothing was trained for good or tested
    assert ("try_models", "modeler") in ledger_kinds(result) and Path(result["split_folder"]).exists()
    assert not train.TEST_LOG.exists() and not (tmp_path / "models").exists()


def test_an_off_menu_model_is_not_proposed(team, tmp_path):
    team["models"] = {**MODELS, "regressor": {"model": "DeepMagicRegressor"}}
    result, pauses = run(["approve"] * 4 + ["approve", "stop"])
    assert team["replies"][1].startswith("Not proposed: regressor must be one of")
    # Approving without a proposal sends the planner back to propose one
    assert pauses[-1]["Models"] == "(no models proposed)" and result["models_feedback"] == "stop"
    assert not train.TEST_LOG.exists()
