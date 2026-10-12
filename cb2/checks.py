"""Hard checks on the data. A failed check stops the work with a list of what is wrong,
rather than letting later steps run on data that can't be trusted."""

import hashlib
import json
from pathlib import Path

import pandas as pd

# The columns later steps rely on, from ChEMBL's activity and assay records
REQUIRED_COLUMNS = [
    "activity_id", "assay_chembl_id", "molecule_chembl_id", "parent_molecule_chembl_id", "canonical_smiles",
    "target_chembl_id", "standard_type", "standard_relation", "standard_value", "standard_units",
    "pchembl_value", "data_validity_comment", "potential_duplicate", "assay_type", "assay_confidence_score",
]


class CheckFailed(Exception):
    def __init__(self, what: str, problems: list[str]):
        self.problems = problems
        super().__init__(f"{what} failed its checks:\n" + "\n".join(f"- {problem}" for problem in problems))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_download(folder: Path) -> pd.DataFrame:
    """Check a raw download against its manifest and return its table."""
    what = f"The download in {folder}"
    if not (folder / "manifest.json").exists():
        raise CheckFailed(what, ["it has no manifest.json, so the download never finished"])
    manifest = json.loads((folder / "manifest.json").read_text())

    # Files first: if any file changed, nothing read from it can be trusted
    listed = manifest["sha256"]
    present = {path.name for path in folder.iterdir() if path.name != "manifest.json"}
    problems = [f"{name} is missing" for name in sorted(listed.keys() - present)]
    problems += [f"{name} is not in the manifest" for name in sorted(present - listed.keys())]
    problems += [f"{name} has changed since it was downloaded (its checksum differs)"
                 for name in sorted(listed.keys() & present) if sha256(folder / name) != listed[name]]
    if problems:
        raise CheckFailed(what, problems)

    table = pd.read_csv(folder / "table.csv.gz", low_memory=False)
    activities = json.loads((folder / "activities.json").read_text())
    counts = {"activities": len(activities), "assays": len(json.loads((folder / "assays.json").read_text())),
              "table": len(table)}
    problems = [f"{name} has {n:,} rows but the manifest says {manifest['rows'][name]:,}"
                for name, n in counts.items() if n != manifest["rows"][name]]
    # Each activity record should become exactly one row (an assay listed twice would add rows)
    if len(table) != len(activities):
        problems.append(f"the table has {len(table):,} rows but there are {len(activities):,} activity records")
    if missing := [name for name in REQUIRED_COLUMNS if name not in table.columns]:
        problems.append(f"the table has no column {', '.join(missing)}")
    if "activity_id" in table.columns and len(repeated := table["activity_id"][table["activity_id"].duplicated()]):
        problems.append(f"{repeated.nunique():,} activity IDs appear more than once, e.g. {repeated.iloc[0]}")
    if problems:
        raise CheckFailed(what, problems)
    return table


if __name__ == "__main__":
    # Usage: python -m cb2.checks data/raw/<folder> [...]
    import sys

    failed = False
    for name in sys.argv[1:]:
        try:
            print(f"OK: {name} ({len(verify_download(Path(name))):,} rows)")
        except CheckFailed as error:
            failed = True
            print(error)
    sys.exit(failed)
