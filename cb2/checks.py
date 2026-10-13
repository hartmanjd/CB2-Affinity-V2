"""Hard checks on the data. A failed check stops the work with a list of what is wrong,
rather than letting later steps run on data that can't be trusted."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

# The columns later steps rely on, from ChEMBL's activity and assay records
REQUIRED_COLUMNS = [
    "activity_id", "assay_chembl_id", "molecule_chembl_id", "parent_molecule_chembl_id", "canonical_smiles",
    "target_chembl_id", "standard_type", "standard_relation", "standard_value", "standard_units",
    "pchembl_value", "data_validity_comment", "potential_duplicate", "assay_type", "assay_confidence_score",
]

# A clean value outside this range almost certainly means a unit or typing error upstream
PCHEMBL_RANGE = (2.0, 12.0)


class CheckFailed(Exception):
    def __init__(self, what: str, problems: list[str]):
        self.problems = problems
        super().__init__(f"{what} failed its checks:\n" + "\n".join(f"- {problem}" for problem in problems))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(folder: Path, what: str) -> dict:
    """Check a saved folder's files against its manifest and return the manifest. If any file
    changed, nothing read from it can be trusted."""
    if not (folder / "manifest.json").exists():
        raise CheckFailed(what, ["it has no manifest.json, so it was never finished"])
    manifest = json.loads((folder / "manifest.json").read_text())
    listed = manifest["sha256"]
    present = {path.name for path in folder.iterdir() if path.name != "manifest.json"}
    problems = [f"{name} is missing" for name in sorted(listed.keys() - present)]
    problems += [f"{name} is not in the manifest" for name in sorted(present - listed.keys())]
    problems += [f"{name} has changed since it was saved (its checksum differs)"
                 for name in sorted(listed.keys() & present) if sha256(folder / name) != listed[name]]
    if problems:
        raise CheckFailed(what, problems)
    return manifest


def verify_download(folder: Path) -> pd.DataFrame:
    """Check a raw download against its manifest and return its table."""
    what = f"The download in {folder}"
    manifest = verify_files(folder, what)
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


def check_dataset(clean: pd.DataFrame, kept: pd.DataFrame, targets, types, max_spread: float | None) -> None:
    """The clean dataset's contract: what the modeling code may rely on."""
    problems = []
    compound = clean["parent_molecule_chembl_id"]
    if len(repeated := compound[compound.duplicated()]):
        problems.append(f"{repeated.nunique():,} compounds appear more than once, e.g. {repeated.iloc[0]}")
    if missing := int(clean["canonical_smiles"].isna().sum()):
        problems.append(f"{missing:,} compounds have no structure")
    outside = clean[~clean["pchembl"].between(*PCHEMBL_RANGE)]
    if len(outside):
        example = outside.iloc[0]
        problems.append(f"{len(outside):,} compounds have a value outside {PCHEMBL_RANGE[0]:g} to {PCHEMBL_RANGE[1]:g}, "
                        f"e.g. {example['parent_molecule_chembl_id']} = {example['pchembl']}")
    if max_spread is not None and (clean["spread"] > max_spread).any():
        problems.append(f"some compounds' repeats disagree by more than {max_spread} log units")
    for column, allowed in [("target_chembl_id", targets), ("standard_type", types), ("standard_relation", ["="]),
                            ("standard_units", ["nM"])]:
        if len(wrong := kept[~kept[column].isin(allowed)]):
            problems.append(f"{len(wrong):,} kept records have {column} outside {list(allowed)}")
    if problems:
        raise CheckFailed("The clean dataset", problems)


def reconcile(raw_ids: pd.Series, clean: pd.DataFrame, dropped: pd.DataFrame, waterfall: list[dict]) -> None:
    """The books must balance: every raw activity record is either dropped (once) or combined into
    a compound (once), and every step's counts add up."""
    combined = [i for ids in clean["activity_ids"] for i in json.loads(ids)]
    used = Counter(combined) + Counter(dropped["activity_id"].tolist())
    raw = set(raw_ids.tolist())
    problems = []
    if twice := [i for i, n in used.items() if n > 1]:
        problems.append(f"{len(twice):,} activity records are counted more than once, e.g. {twice[0]}")
    if lost := sorted(raw - used.keys()):
        problems.append(f"{len(lost):,} activity records are neither kept nor dropped, e.g. {lost[0]}")
    if unknown := sorted(used.keys() - raw):
        problems.append(f"{len(unknown):,} activity records aren't in the download, e.g. {unknown[0]}")
    for step in waterfall:
        if step["rows_in"] - step["dropped"] != step["rows_out"]:
            problems.append(f"step '{step['step']}' doesn't add up: {step['rows_in']:,} - {step['dropped']:,} "
                            f"!= {step['rows_out']:,}")
    for before, after in zip(waterfall, waterfall[1:]):
        if before["rows_out"] != after["rows_in"]:
            problems.append(f"step '{after['step']}' starts with {after['rows_in']:,} rows, "
                            f"but the step before it ended with {before['rows_out']:,}")
    if waterfall and (waterfall[0]["rows_in"], waterfall[-1]["rows_out"]) != (len(raw_ids), len(combined)):
        problems.append(f"the steps run from {waterfall[0]['rows_in']:,} to {waterfall[-1]['rows_out']:,} rows, "
                        f"but the download has {len(raw_ids):,} and the clean dataset combines {len(combined):,}")
    if sum(step["dropped"] for step in waterfall) != len(dropped):
        problems.append(f"the steps dropped {sum(step['dropped'] for step in waterfall):,} rows, "
                        f"but the dropped list has {len(dropped):,}")
    if problems:
        raise CheckFailed("The cleaning bookkeeping", problems)


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
