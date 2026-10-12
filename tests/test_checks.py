"""Each test breaks a download in one way on purpose, and checks that verify_download catches it."""

import json

import pytest

from cb2 import data
from cb2.checks import CheckFailed, verify_download
from tests.fakes import activity, assay


def fails(folder, expected: str):
    with pytest.raises(CheckFailed) as error:
        verify_download(folder)
    assert any(expected in problem for problem in error.value.problems), error.value


def test_a_good_download_passes(save):
    assert len(verify_download(save())) == 5


def test_loading_a_download_checks_it(save):
    folder = save()
    (folder / "assays.json").write_text("[]")
    with pytest.raises(CheckFailed):
        data.load(str(folder))


def test_an_unfinished_download_fails(save):
    folder = save()
    (folder / "manifest.json").unlink()
    fails(folder, "never finished")


def test_a_changed_file_fails(save):
    folder = save()
    path = folder / "activities.json"
    text = path.read_text()
    path.write_text(text.replace('"standard_value": 10.0', '"standard_value": 100.0', 1))
    fails(folder, "activities.json has changed")


def test_a_missing_file_fails(save):
    folder = save()
    (folder / "table.csv.gz").unlink()
    fails(folder, "table.csv.gz is missing")


def test_an_extra_file_fails(save):
    folder = save()
    (folder / "notes.txt").write_text("added later")
    fails(folder, "notes.txt is not in the manifest")


def test_a_wrong_row_count_in_the_manifest_fails(save):
    folder = save()
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["rows"]["activities"] = 6
    (folder / "manifest.json").write_text(json.dumps(manifest))
    fails(folder, "activities has 5 rows but the manifest says 6")


def test_a_repeated_activity_fails(save):
    folder = save([activity(1), activity(2), activity(2)])
    fails(folder, "1 activity IDs appear more than once, e.g. 2")


def test_an_assay_listed_twice_fails(save):
    # Joining assay details onto a duplicated assay doubles that assay's activity rows
    folder = save(assays=[assay("CHEMBL_A1"), assay("CHEMBL_A1")])
    fails(folder, "the table has 10 rows but there are 5 activity records")


def test_a_missing_column_fails(save):
    records = [{key: value for key, value in activity(i).items() if key != "pchembl_value"} for i in range(1, 4)]
    fails(save(records), "no column pchembl_value")
