import pandas as pd
import pytest

from cb2 import data
from tests.fakes import STATUS, STRUCTURES, TARGET, activity, assay, compound


@pytest.fixture
def save(tmp_path):
    """Saves a download made of the given records, the same way a real one is saved."""
    def make(activities=None, assays=None):
        folder = tmp_path / "20260101T000000Z"
        data.save(folder, STATUS, [TARGET], activities or [activity(i) for i in range(1, 6)],
                  assays or [assay("CHEMBL_A1")], {"target_ids": ["CHEMBL253"], "reason": "test"})
        return folder
    return make


@pytest.fixture
def clean():
    """A clean dataset of 47 compounds with values from 5 to 9."""
    return pd.DataFrame([compound(n, smiles, 5 + n % 9 / 2) for n, smiles in enumerate(STRUCTURES, start=1)])
