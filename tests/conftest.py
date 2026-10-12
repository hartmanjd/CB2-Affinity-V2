import pytest

from cb2 import data
from tests.fakes import STATUS, TARGET, activity, assay


@pytest.fixture
def save(tmp_path):
    """Saves a download made of the given records, the same way a real one is saved."""
    def make(activities=None, assays=None):
        folder = tmp_path / "20260101T000000Z"
        data.save(folder, STATUS, [TARGET], activities or [activity(i) for i in range(1, 6)],
                  assays or [assay("CHEMBL_A1")], {"target_ids": ["CHEMBL253"], "reason": "test"})
        return folder
    return make
