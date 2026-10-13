"""Prepare a clean dataset for modeling and split it, the same way every time.

Structures are standardized, and compounds that turn out to be the same molecule are merged.
Each molecule then goes to a training or a test part in two ways: by scaffold (the headline:
whole families of molecules are held out, so the test asks about new kinds of molecule) and at
random (for comparison). The training part is also divided into folds for cross-validation.
None of this is chosen by the agents, and the books must balance before anything is saved."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit.Chem.Scaffolds import MurckoScaffold

from cb2.checks import CheckFailed, sha256, verify_files
from cb2.clean import ROOT, code_version, shown
from cb2.features import standardize

SPLIT = ROOT / "data" / "split"
SEED = 42
TEST_FRACTION = 0.2
FOLDS = 5
ACTIVE_AT = 7.0
# What each split keeps together: a whole scaffold family, or just the molecule itself
GROUPS = {"scaffold": "scaffold", "random": "compound_id"}


def load_clean(folder: Path) -> tuple[pd.DataFrame, dict]:
    manifest = verify_files(folder, f"The clean dataset in {folder}")
    clean = pd.read_csv(folder / "clean.csv")
    if len(clean) != manifest["rows"]["compounds"]:
        raise CheckFailed(f"The clean dataset in {folder}",
                          [f"clean.csv has {len(clean):,} rows but the manifest says {manifest['rows']['compounds']:,}"])
    return clean, manifest


def prepare(clean: pd.DataFrame, max_spread: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per distinct molecule. Compounds that standardize to the same structure are
    merged (median value); unreadable ones, and merges whose values disagree by more than the
    recipe's limit, are dropped with a reason."""
    structures = clean["canonical_smiles"].map(standardize)
    dropped = [pd.DataFrame({"compound": clean.loc[structures.isna(), "parent_molecule_chembl_id"],
                             "reason": "structure can't be read"})]
    rows = []
    kept = clean.assign(smiles=structures).dropna(subset=["smiles"]).sort_values("parent_molecule_chembl_id")
    for structure, group in kept.groupby("smiles", sort=False):
        parents = group["parent_molecule_chembl_id"].tolist()
        spread = float(group["pchembl"].max() - group["pchembl"].min())
        if max_spread is not None and spread > max_spread:
            dropped.append(pd.DataFrame({"compound": parents,
                                         "reason": f"same molecule as another compound, values disagree by "
                                                   f"{spread:.4g} log units ({', '.join(parents)})"}))
            continue
        rows.append({"compound_id": parents[0], "smiles": structure,
                     "pchembl": round(float(group["pchembl"].median()), 4),
                     "n_measurements": int(group["n_measurements"].sum()), "parents": json.dumps(parents),
                     "activity_ids": json.dumps(sorted(i for ids in group["activity_ids"] for i in json.loads(ids)))})
    molecules = pd.DataFrame(rows, columns=["compound_id", "smiles", "pchembl", "n_measurements", "parents",
                                            "activity_ids"]).sort_values("compound_id", ignore_index=True)
    dropped = pd.concat(dropped, ignore_index=True).sort_values("compound", ignore_index=True)

    # Every compound in the clean dataset is either in one molecule or dropped, exactly once
    used = Counter(p for parents in molecules["parents"] for p in json.loads(parents)) + Counter(dropped["compound"])
    problems = []
    if twice := [c for c, n in used.items() if n > 1]:
        problems.append(f"{len(twice):,} compounds are counted more than once, e.g. {twice[0]}")
    if lost := sorted(set(clean["parent_molecule_chembl_id"]) ^ used.keys()):
        problems.append(f"{len(lost):,} compounds are neither kept nor dropped, or unknown, e.g. {lost[0]}")
    if molecules["smiles"].duplicated().any():
        problems.append("two molecules have the same structure")
    if problems:
        raise CheckFailed("Preparing the molecules", problems)
    return molecules, dropped


def scaffold_of(smiles: str) -> str:
    # A molecule without rings has no scaffold, so it's a family of its own
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles) or f"no rings: {smiles}"


def hold_out(groups: pd.Series, rng) -> pd.Series:
    """Whole groups go to the test part, in a seeded random order, until it's as full as it can
    get without going over its share."""
    sizes = groups.value_counts().sort_index()
    room = round(TEST_FRACTION * len(groups))
    test = set()
    for group in sizes.index[rng.permutation(len(sizes))]:
        if sizes[group] <= room:
            test.add(group)
            room -= sizes[group]
    return groups.map(lambda group: "test" if group in test else "train")


def assign_folds(groups: pd.Series, rng) -> pd.Series:
    """Whole groups go to folds, biggest first (ties in seeded random order), each to the
    emptiest fold so far, so folds come out about the same size."""
    sizes = groups.value_counts().sort_index()
    sizes = sizes.iloc[rng.permutation(len(sizes))].sort_values(ascending=False, kind="stable")
    filled, fold = [0] * FOLDS, {}
    for group, size in sizes.items():
        fold[group] = int(np.argmin(filled))
        filled[fold[group]] += size
    return groups.map(fold)


def make_splits(molecules: pd.DataFrame) -> pd.DataFrame:
    table = molecules.assign(scaffold=molecules["smiles"].map(scaffold_of))
    for name, column in GROUPS.items():
        rng = np.random.default_rng(SEED)
        table[f"{name}_part"] = hold_out(table[column], rng)
        train = table[f"{name}_part"] == "train"
        table[f"{name}_fold"] = assign_folds(table.loc[train, column], rng).reindex(table.index).astype("Int64")
    check_splits(table)
    return table


def check_splits(table: pd.DataFrame) -> None:
    problems = []
    if len(repeated := table["compound_id"][table["compound_id"].duplicated()]):
        problems.append(f"{repeated.nunique():,} compounds appear more than once, e.g. {repeated.iloc[0]}")
    for name, column in GROUPS.items():
        part, fold = table[f"{name}_part"], table[f"{name}_fold"]
        train, test = part == "train", part == "test"
        if not (train | test).all():
            problems.append(f"{name} split: some compounds are in neither part")
        if not round(TEST_FRACTION * len(table)) / 2 <= test.sum() <= round(TEST_FRACTION * len(table)):
            problems.append(f"{name} split: the test part has {test.sum():,} of {len(table):,} compounds")
        if fold[test].notna().any():
            problems.append(f"{name} split: some test compounds are in a training fold")
        if not fold[train].isin(range(FOLDS)).all() or fold[train].nunique() != FOLDS:
            problems.append(f"{name} split: training compounds must each be in one of {FOLDS} non-empty folds")
        if len(shared := set(table.loc[train, column]) & set(table.loc[test, column])):
            problems.append(f"{name} split: {len(shared):,} {column} groups are in both parts, e.g. {min(shared)}")
        if (table[train].groupby(column)[f"{name}_fold"].nunique() > 1).any():
            problems.append(f"{name} split: a {column} group is spread over more than one fold")
    if problems:
        raise CheckFailed("The splits", problems)


def summary(table: pd.DataFrame, dropped: pd.DataFrame, compounds: int) -> str:
    merged = int((table["parents"].map(lambda p: len(json.loads(p))) > 1).sum())
    lines = [f"{compounds:,} compounds -> {len(table):,} molecules ({len(dropped):,} dropped, "
             f"{merged:,} merged from several compounds), {table['scaffold'].nunique():,} scaffolds",
             f"{'split':<10}{'train':>8}{'test':>8}{'test share':>12}   active (pKi >= {ACTIVE_AT:g}) train / test"]
    for name in GROUPS:
        parts = {part: table[table[f"{name}_part"] == part] for part in ("train", "test")}
        active = " / ".join(f"{(rows['pchembl'] >= ACTIVE_AT).mean():.0%}" for rows in parts.values())
        lines.append(f"{name:<10}{len(parts['train']):>8,}{len(parts['test']):>8,}"
                     f"{len(parts['test']) / len(table):>12.1%}   {active}")
    return "\n".join(lines)


def save(table: pd.DataFrame, dropped: pd.DataFrame, source: Path, run_id: str = "") -> Path:
    folder = SPLIT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True)
    table.to_csv(folder / "molecules.csv", index=False)
    dropped.to_csv(folder / "dropped.csv", index=False)
    manifest = {
        "made_at_utc": folder.name,
        "run": run_id,
        "source": {"folder": shown(source), "manifest_sha256": sha256(source / "manifest.json")},
        "settings": {"standardize": "largest fragment, neutralized", "seed": SEED, "test_fraction": TEST_FRACTION,
                     "folds": FOLDS, "rdkit": rdkit.__version__},
        "code_version": code_version(),
        "rows": {"molecules": len(table), "dropped": len(dropped)}
                | {f"{name} {part}": int((table[f"{name}_part"] == part).sum())
                   for name in GROUPS for part in ("train", "test")},
        "sha256": {path.name: sha256(path) for path in sorted(folder.iterdir())},
    }
    # Written last, so a folder without a manifest is unfinished
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return folder


if __name__ == "__main__":
    # Usage: python -m cb2.split data/clean/<folder> [--dry-run]
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a clean dataset's molecules and split them.")
    parser.add_argument("clean", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="show the splits without saving anything")
    args = parser.parse_args()
    clean, manifest = load_clean(args.clean)
    molecules, dropped = prepare(clean, manifest["recipe"]["max_spread"])
    table = make_splits(molecules)
    print(summary(table, dropped, len(clean)))
    if not args.dry_run:
        print(f"Saved to {save(table, dropped, args.clean.resolve())}")
