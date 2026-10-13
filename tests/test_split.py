import json

import numpy as np
import pandas as pd
import pytest

from cb2 import split
from cb2.checks import CheckFailed, sha256
from cb2.features import DESCRIPTORS, FEATURES, featurize, standardize
from tests.fakes import compound


def test_standardize_keeps_the_largest_piece_neutral():
    assert standardize("CC(=O)[O-].[Na+]") == "CC(=O)O"
    assert standardize("C[NH3+].[Cl-]") == "CN"
    assert standardize("not a molecule") is None and standardize(np.nan) is None


@pytest.mark.parametrize("name, width", [("morgan2", 2048), ("morgan3", 2048), ("descriptors", len(DESCRIPTORS)),
                                         ("morgan2+descriptors", 2048 + len(DESCRIPTORS))])
def test_features_have_a_fixed_shape_and_repeat_exactly(name, width):
    matrix = featurize(["Cc1ccccc1", "CCO"], name)
    assert matrix.shape == (2, width)
    np.testing.assert_array_equal(matrix, featurize(["Cc1ccccc1", "CCO"], name))


def test_features_refuse_what_is_not_on_the_menu():
    assert set(FEATURES) == {"morgan2", "morgan3", "descriptors", "morgan2+descriptors"}
    with pytest.raises(ValueError, match="must be one of"):
        featurize(["CCO"], "morgan9")
    with pytest.raises(ValueError, match="can't read"):
        featurize(["CCO", "nonsense"], "morgan2")


def test_prepare_merges_the_same_molecule_and_drops_unreadable(clean):
    extra = pd.DataFrame([compound(901, "CC(=O)[O-].[Na+]", 6.0), compound(902, "CC(=O)O", 7.0),
                          compound(903, "not a molecule", 6.0)])
    molecules, dropped = split.prepare(pd.concat([clean, extra], ignore_index=True))
    acid = molecules[molecules["smiles"] == "CC(=O)O"].iloc[0]
    assert json.loads(acid["parents"]) == ["CHEMBL901", "CHEMBL902"]
    assert (acid["compound_id"], acid["pchembl"], json.loads(acid["activity_ids"])) == ("CHEMBL901", 6.5, [901, 902])
    assert dropped.to_dict("records") == [{"compound": "CHEMBL903", "reason": "structure can't be read"}]
    assert len(molecules) == len(clean) + 1


def test_prepare_drops_merges_whose_values_disagree(clean):
    extra = pd.DataFrame([compound(901, "CC(=O)[O-].[Na+]", 5.0), compound(902, "CC(=O)O", 7.0)])
    molecules, dropped = split.prepare(pd.concat([clean, extra], ignore_index=True), max_spread=1.0)
    assert "CC(=O)O" not in set(molecules["smiles"])
    assert dropped["compound"].tolist() == ["CHEMBL901", "CHEMBL902"]
    assert dropped["reason"].str.contains("disagree by 2 log units").all()


def test_splits_keep_scaffold_families_together_and_repeat_exactly(clean):
    table = split.make_splits(split.prepare(clean)[0])
    test = table[table["scaffold_part"] == "test"]
    assert set(test["scaffold"]).isdisjoint(table.loc[table["scaffold_part"] == "train", "scaffold"])
    assert 0 < len(test) <= round(0.2 * len(table))
    assert test["scaffold_fold"].isna().all() and table["random_fold"].notna().sum() == (table["random_part"] == "train").sum()
    pd.testing.assert_frame_equal(table, split.make_splits(split.prepare(clean)[0]))


def broken(table, column, rows, value):
    table = table.copy()
    table.loc[rows(table), column] = value
    return table


def test_split_checks_catch_mistakes(clean):
    table = split.make_splits(split.prepare(clean)[0])
    train = table.index[table["scaffold_part"] == "train"]
    # A training molecule whose scaffold has relatives in training moved alone into the test part
    family = table.loc[train].groupby("scaffold").filter(lambda g: len(g) > 1).index[0]
    for bad, message in [
        (broken(table, "scaffold_part", lambda t: [family], "test"), "groups are in both parts"),
        (broken(table, "scaffold_fold", lambda t: [t.index[t["scaffold_part"] == "test"][0]], 0), "in a training fold"),
        (broken(table, "random_fold", lambda t: [t.index[t["random_part"] == "train"][0]], pd.NA), "non-empty folds"),
        (broken(table, "scaffold_fold", lambda t: [family], (table.loc[family, "scaffold_fold"] + 1) % 5),
         "more than one fold"),
        (pd.concat([table, table.iloc[[0]]], ignore_index=True), "appear more than once"),
    ]:
        with pytest.raises(CheckFailed, match=message):
            split.check_splits(bad)


def test_saved_splits_record_their_source(clean, tmp_path, monkeypatch):
    source = tmp_path / "clean"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    monkeypatch.setattr(split, "SPLIT", tmp_path / "split")
    molecules, dropped = split.prepare(clean)
    folder = split.save(split.make_splits(molecules), dropped, source)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["settings"]["seed"] == split.SEED and manifest["rows"]["molecules"] == len(clean)
    assert set(manifest["sha256"]) == {"molecules.csv", "dropped.csv"}


def test_a_changed_clean_dataset_is_refused(tmp_path):
    folder = tmp_path / "clean"
    folder.mkdir()
    (folder / "clean.csv").write_text("parent_molecule_chembl_id\nCHEMBL1\n")
    (folder / "manifest.json").write_text(json.dumps({"rows": {"compounds": 1},
                                                      "sha256": {"clean.csv": sha256(folder / "clean.csv")}}))
    assert len(split.load_clean(folder)[0]) == 1
    (folder / "clean.csv").write_text("parent_molecule_chembl_id\nCHEMBL2\n")
    with pytest.raises(CheckFailed, match="has changed"):
        split.load_clean(folder)
