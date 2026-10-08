"""Download the targets the team proposed and the lead approved, and load them as one table.

Every activity record for each approved target is kept, with the details of its assay and
target; nothing is filtered. Each download is saved in its own folder under data/raw, with a
manifest recording the ChEMBL release, the approved proposal, row counts and file checksums."""

import hashlib
import json
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import pandas as pd

from cb2.chembl import describe_targets, get, get_all

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(target_ids: list[str], reason: str = "", run_id: str = "") -> Path:
    status = get("status")
    targets = describe_targets(target_ids)
    joined = ",".join(target_ids)
    print(f"ChEMBL {status['chembl_db_version']}: downloading {joined}")
    activities = get_all("activity", "activities", target_chembl_id__in=joined)
    assays = get_all("assay", "assays", target_chembl_id__in=joined)

    # The raw responses are saved unchanged; the table is built from them
    folder = RAW / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True)
    for name, records in [("targets", targets), ("activities", activities), ("assays", assays)]:
        (folder / f"{name}.json").write_text(json.dumps(records))
    table = build_table(activities, assays, targets)
    table.to_csv(folder / "table.csv.gz", index=False)

    manifest = {
        "downloaded_at_utc": folder.name,
        "chembl_version": status["chembl_db_version"],
        "chembl_release_date": status["chembl_release_date"],
        "run": run_id,
        "proposal": {"target_ids": target_ids, "reason": reason},
        "targets": [{key: target[key] for key in ("target_chembl_id", "pref_name", "organism", "target_type")}
                    for target in targets],
        "rows": {"activities": len(activities), "assays": len(assays), "table": len(table)},
        "sha256": {path.name: sha256(path) for path in sorted(folder.iterdir())},
    }
    # Written last, so a folder without a manifest is an unfinished download
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(table):,} rows to {folder}")
    return folder


def scalars(record: dict) -> dict:
    # Keep every plain field; nested lists (e.g. activity_properties) become JSON text
    return {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in record.items()}


def build_table(activities: list[dict], assays: list[dict], targets: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame([scalars(record) for record in activities])
    # Add every assay field, prefixed with "assay_" where the name doesn't already say so
    assay_table = pd.DataFrame([scalars(record) for record in assays])
    assay_table.columns = [name if name.startswith("assay_") else f"assay_{name}" for name in assay_table.columns]
    table = table.merge(assay_table, on="assay_chembl_id", how="left", suffixes=("", "_from_assay"))
    target_types = {target["target_chembl_id"]: target["target_type"] for target in targets}
    table["target_type"] = table["target_chembl_id"].map(target_types)
    return table


def find_download(target_ids: list[str]) -> Path | None:
    """The latest finished download of exactly these targets, if there is one."""
    for folder in sorted(RAW.glob("*"), reverse=True):
        manifest = folder / "manifest.json"
        if manifest.exists():
            downloaded = {target["target_chembl_id"] for target in json.loads(manifest.read_text())["targets"]}
            if downloaded == set(target_ids):
                return folder
    return None


@cache
def load(folder: str) -> pd.DataFrame:
    # Cached, so every look in a run reads the file only once
    return pd.read_csv(Path(folder) / "table.csv.gz", low_memory=False)


if __name__ == "__main__":
    # Usage: python -m cb2.data CHEMBL... [CHEMBL...]   (downloads those targets)
    import sys

    download(sys.argv[1:])
