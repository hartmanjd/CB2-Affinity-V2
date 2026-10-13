"""A small table where the right answer to every step is worked out by hand, plus tests that break
the bookkeeping on purpose to show the checks catch it."""

import hashlib
import json
import math
from datetime import datetime

import pandas as pd
import pytest

from cb2 import clean
from cb2.checks import CheckFailed, check_dataset, reconcile, verify_download
from cb2.clean import RecipeError, make_recipe, run
from cb2.data import build_table
from tests.fakes import TARGET, activity, assay

RECIPE = {"targets": ["CHEMBL253"], "types": ["Ki"], "min_confidence": 9, "drop_flagged": True,
          "combine": "median", "max_spread": 1.0}


def nM(value):
    # A Ki record of this many nM, with ChEMBL's pchembl_value for it
    return {"standard_value": value, "pchembl_value": round(9 - math.log10(value), 2)}


# activity id: (fields, the step that should drop it, or None if kept)
ROWS = {
    1: ({"parent_molecule_chembl_id": "P1", **nM(10)}, None),
    2: ({"parent_molecule_chembl_id": "P1", **nM(100)}, None),            # a salt form of P1
    3: ({"target_chembl_id": "CHEMBL218"}, "keep chosen targets"),
    4: ({"standard_type": "IC50"}, "keep chosen measurement types"),
    5: ({"assay_type": "F"}, "keep binding assays"),
    6: ({}, "keep binding assays"),                                        # assay with confidence 8
    7: ({"data_validity_comment": "Outside typical range"}, "drop records ChEMBL flags"),
    8: ({"data_validity_comment": "Manually validated", **nM(1)}, None),
    9: ({"potential_duplicate": 1}, "drop records ChEMBL flags"),
    10: ({"standard_relation": ">", "pchembl_value": None}, "keep exact values"),
    11: ({"standard_value": None, "pchembl_value": None}, "keep usable values"),
    12: ({"standard_units": "ug.mL-1", "pchembl_value": None}, "keep usable values"),
    13: ({"canonical_smiles": None}, "keep usable values"),
    14: ({"parent_molecule_chembl_id": "P14", **nM(1)}, "combine repeats"),
    15: ({"parent_molecule_chembl_id": "P14", **nM(1000)}, "combine repeats"),  # 3 log units from 14
    16: ({**nM(1000)}, None),
}


def make_table(rows=ROWS):
    activities = [activity(i, assay="CHEMBL_A3" if i == 6 else "CHEMBL_A1", **fields) for i, (fields, _) in rows.items()]
    assays = [assay("CHEMBL_A1"), {**assay("CHEMBL_A3"), "confidence_score": 8}]
    return build_table(activities, assays, [TARGET])


@pytest.fixture
def table():
    return make_table()


def test_each_step_drops_exactly_the_right_rows(table):
    result = run(table, make_recipe(RECIPE, table))
    dropped = dict(zip(result.dropped["activity_id"], result.dropped["step"]))
    assert dropped == {i: step for i, (_, step) in ROWS.items() if step}


def test_the_clean_dataset(table):
    result = run(table, make_recipe(RECIPE, table))
    rows = result.clean.set_index("parent_molecule_chembl_id")
    assert rows["pchembl"].to_dict() == {"CHEMBL_M16": 6.0, "CHEMBL_M8": 9.0, "P1": 7.5}
    assert rows.loc["P1", "n_measurements"] == 2 and rows.loc["P1", "spread"] == 1.0
    assert json.loads(rows.loc["P1", "activity_ids"]) == [1, 2]


def test_the_waterfall_adds_up(table):
    waterfall = run(table, make_recipe(RECIPE, table)).waterfall
    assert [(s["rows_in"], s["dropped"], s["rows_out"]) for s in waterfall] == [
        (16, 1, 15), (15, 1, 14), (14, 2, 12), (12, 2, 10), (10, 1, 9), (9, 3, 6), (6, 2, 4)]
    assert waterfall[-1]["compounds"] == 3


def test_the_menu_settings_change_the_result(table):
    loose = {**RECIPE, "min_confidence": 8, "drop_flagged": False, "max_spread": None, "combine": "mean"}
    result = run(table, make_recipe(loose, table))
    assert set(result.dropped["activity_id"]) == {3, 4, 5, 10, 11, 12, 13}
    assert result.clean.set_index("parent_molecule_chembl_id").loc["P14", "pchembl"] == 7.5


def test_pchembl_conversion():
    values = pd.DataFrame({"standard_value": [1, 10, 1000], "pchembl_value": [9, 8, 6], "activity_id": [1, 2, 3]})
    assert clean.to_pchembl(values).tolist() == [9.0, 8.0, 6.0]


def test_a_saved_download_cleans_the_same_way(save):
    # Through a real save and load, values come back from CSV as text and numbers
    folder = save([activity(i, **fields) for i, (fields, _) in ROWS.items() if i != 6])
    table = verify_download(folder)
    assert len(run(table, make_recipe(RECIPE, table)).clean) == 3


def test_same_input_and_recipe_give_identical_files(table, tmp_path, monkeypatch):
    # Two saves of the same recipe, into two folders, must give byte-identical files
    stamps = iter([datetime(2026, 1, 1, 0, 0, 0), datetime(2026, 1, 1, 0, 0, 1)])
    monkeypatch.setattr(clean, "CLEAN", tmp_path / "clean")
    monkeypatch.setattr(clean, "datetime", type("Clock", (), {"now": staticmethod(lambda tz: next(stamps))}))
    source = tmp_path / "raw"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    recipe = make_recipe(RECIPE, table)
    folders = [clean.save(run(table, recipe), source, recipe) for _ in range(2)]
    for name in ("clean.csv", "dropped.csv"):
        first, second = (hashlib.sha256((folder / name).read_bytes()).hexdigest() for folder in folders)
        assert first == second
    manifest = json.loads((folders[0] / "manifest.json").read_text())
    assert manifest["rows"] == {"compounds": 3, "dropped": 12} and manifest["recipe"]["targets"] == ["CHEMBL253"]


# --- Break it on purpose ---

@pytest.mark.parametrize("change, expected", [
    ({"min_confidence": 3}, "min_confidence must be one of"),
    ({"combine": "max"}, "combine must be one of"),
    ({"max_spread": 2.0}, "max_spread must be one of"),
    ({"targets": ["CHEMBL999"]}, "targets must be a list of downloaded targets"),
    ({"types": ["EC50"]}, "types must be a list from"),
    ({"drop_flagged": "yes"}, "drop_flagged must be true or false"),
    ({"extra": 1}, "unknown setting 'extra'"),
])
def test_a_recipe_off_the_menu_is_refused(table, change, expected):
    with pytest.raises(RecipeError, match=expected):
        make_recipe({**RECIPE, **change}, table)


def test_a_missing_setting_is_refused(table):
    with pytest.raises(RecipeError, match="missing setting 'combine'"):
        make_recipe({key: value for key, value in RECIPE.items() if key != "combine"}, table)


def test_a_value_that_disagrees_with_chembl_fails(table):
    table.loc[table["activity_id"] == 16, "pchembl_value"] = 5.0
    with pytest.raises(CheckFailed, match="disagree with ChEMBL's pchembl_value"):
        run(table, make_recipe(RECIPE, table))


def test_a_value_outside_the_believable_range_fails(table):
    # 1e-6 nM would be a pKi of 15: almost certainly a unit error
    table.loc[table["activity_id"] == 16, ["standard_value", "pchembl_value"]] = [1e-6, 15.0]
    with pytest.raises(CheckFailed, match="outside 2 to 12"):
        run(table, make_recipe(RECIPE, table))


@pytest.fixture
def books(table):
    return table["activity_id"], run(table, make_recipe(RECIPE, table))


def test_a_lost_dropped_row_fails_reconciliation(books):
    raw, result = books
    with pytest.raises(CheckFailed, match="neither kept nor dropped"):
        reconcile(raw, result.clean, result.dropped.iloc[1:], result.waterfall)


def test_a_row_counted_twice_fails_reconciliation(books):
    raw, result = books
    dropped = pd.concat([result.dropped, result.dropped.iloc[:1]])
    with pytest.raises(CheckFailed, match="counted more than once"):
        reconcile(raw, result.clean, dropped, result.waterfall)


def test_a_waterfall_that_doesnt_add_up_fails(books):
    raw, result = books
    waterfall = [dict(step) for step in result.waterfall]
    waterfall[2]["rows_out"] += 1
    with pytest.raises(CheckFailed, match="doesn't add up"):
        reconcile(raw, result.clean, result.dropped, waterfall)


def test_a_compound_listed_twice_fails_the_contract(table, books):
    _, result = books
    kept = table[table["activity_id"].isin([1, 2, 8, 16])]
    with pytest.raises(CheckFailed, match="appear more than once"):
        check_dataset(pd.concat([result.clean, result.clean.iloc[:1]]), kept, ["CHEMBL253"], ["Ki"], 1.0)


def test_a_record_of_the_wrong_type_fails_the_contract(table, books):
    _, result = books
    kept = table[table["activity_id"].isin([1, 2, 4, 8, 16])]
    with pytest.raises(CheckFailed, match="standard_type outside"):
        check_dataset(result.clean, kept, ["CHEMBL253"], ["Ki"], 1.0)
