"""Made-up ChEMBL records and compounds, shaped like real ones, for building test data."""

import json

STATUS = {"chembl_db_version": "ChEMBL_test", "chembl_release_date": "2026-01-01"}
TARGET = {"target_chembl_id": "CHEMBL253", "pref_name": "Cannabinoid CB2 receptor", "organism": "Homo sapiens",
          "target_type": "SINGLE PROTEIN"}


def activity(activity_id: int, assay: str = "CHEMBL_A1", **fields) -> dict:
    return {"activity_id": activity_id, "assay_chembl_id": assay, "molecule_chembl_id": f"CHEMBL_M{activity_id}",
            "parent_molecule_chembl_id": f"CHEMBL_M{activity_id}", "canonical_smiles": "CCO",
            "target_chembl_id": "CHEMBL253", "standard_type": "Ki", "standard_relation": "=", "standard_value": 10.0,
            "standard_units": "nM", "pchembl_value": 8.0, "data_validity_comment": None, "potential_duplicate": 0,
            "assay_type": "B", **fields}


def assay(assay_id: str) -> dict:
    return {"assay_chembl_id": assay_id, "assay_type": "B", "confidence_score": 9, "description": "binding"}


# 6 ring families of 7 molecules each, plus 5 molecules without rings
RINGS = ["c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "C1CCCCC1", "c1ccc2[nH]ccc2c1", "c1ccsc1"]
SIDES = ["C", "CC", "O", "N", "Cl", "F", "OC"]
CHAINS = ["CCCC", "CCO", "CCN", "CCCCO", "CC(C)C"]
STRUCTURES = [side + ring for ring in RINGS for side in SIDES] + CHAINS


def compound(n: int, smiles: str, pchembl: float) -> dict:
    """A row of a clean dataset."""
    return {"parent_molecule_chembl_id": f"CHEMBL{n:03d}", "canonical_smiles": smiles, "smiles_from": f"CHEMBL{n:03d}",
            "pchembl": pchembl, "n_measurements": 1, "spread": 0.0, "activity_ids": json.dumps([n])}
