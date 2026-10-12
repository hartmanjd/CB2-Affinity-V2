"""Made-up ChEMBL records, shaped like real ones, for building test downloads."""

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
