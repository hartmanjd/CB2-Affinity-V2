"""Turn a verified raw download into a modeling-ready dataset: one row per compound, with one
affinity value, following a recipe the lead approved.

The team chooses the recipe's settings from a fixed menu; the order of the steps, the unit
conversion and the bookkeeping are fixed here. The value is ChEMBL's pChEMBL scale,
-log10(molar), so for Ki records it is pKi. Every activity record ends up either combined into a
compound or in the dropped list with its step and reason, and the books must balance (checked
by cb2.checks) before anything is saved."""

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cb2.checks import CheckFailed, check_dataset, reconcile, sha256, verify_download

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"

TYPES = ("Ki", "Kd", "IC50")
MENU = {
    "min_confidence": (None, 8, 9),
    "combine": ("median", "mean"),
    "max_spread": (None, 0.5, 1.0, 1.5),
}
# ChEMBL marks some records as checked by hand; that comment is not a warning
NOT_A_FLAG = "Manually validated"
# Computed values must agree with ChEMBL's own pchembl_value (given to 2 decimals)
AGREEMENT = 0.01


class RecipeError(ValueError):
    pass


# What the agents are told about the steps; the settings they choose are the ones in brackets
STEPS_TEXT = """1. keep the chosen targets [targets: any of the downloaded targets]
2. keep the chosen measurement types [types: any of Ki, Kd, IC50]
3. keep binding assays, at or above a minimum ChEMBL assay confidence score [min_confidence: null, 8 or 9]
4. drop records ChEMBL flags with a data validity warning or as potential duplicates [drop_flagged: true or false]
5. keep exact values only (fixed: censored values such as "> 10,000 nM" are dropped)
6. keep usable values (fixed: a value in nM above zero, a structure and a parent compound), and
   convert each to pChEMBL, -log10 of the value in molar (fixed; for Ki records this is pKi)
7. combine repeat measurements of each parent compound [combine: median or mean], dropping compounds
   whose measurements are further apart than a limit in log units [max_spread: null, 0.5, 1.0 or 1.5]"""

RECIPE_PROPERTIES = {
    "targets": {"type": "array", "items": {"type": "string"}, "description": "Target IDs to keep"},
    "types": {"type": "array", "items": {"type": "string", "enum": list(TYPES)}},
    "min_confidence": {"type": ["integer", "null"], "enum": list(MENU["min_confidence"])},
    "drop_flagged": {"type": "boolean"},
    "combine": {"type": "string", "enum": list(MENU["combine"])},
    "max_spread": {"type": ["number", "null"], "enum": list(MENU["max_spread"])},
}

TRY_TOOL = {
    "name": "try_recipe",
    "description": ("Run the cleaning steps with these settings without saving anything, and see how many "
                    "records each step keeps and drops, how many compounds are left and their values."),
    "input_schema": {"type": "object", "properties": RECIPE_PROPERTIES, "required": list(RECIPE_PROPERTIES)},
}

PROPOSE_TOOL = {
    "name": "propose_recipe",
    "description": ("Propose the cleaning settings. The project lead approves the recipe before the dataset is "
                    "made. Calling it again replaces the earlier proposal."),
    "input_schema": {"type": "object",
                     "properties": {**RECIPE_PROPERTIES,
                                    "reason": {"type": "string", "description": "Why these settings"}},
                     "required": [*RECIPE_PROPERTIES, "reason"]},
}


@dataclass(frozen=True)
class Recipe:
    targets: tuple[str, ...]
    types: tuple[str, ...]
    min_confidence: int | None
    drop_flagged: bool
    combine: str
    max_spread: float | None


def make_recipe(request: dict, table: pd.DataFrame) -> Recipe:
    """A recipe from a request, refusing anything that isn't on the menu."""
    fields = list(Recipe.__dataclass_fields__)
    problems = [f"unknown setting {key!r}" for key in request if key not in fields]
    problems += [f"missing setting {key!r}" for key in fields if key not in request]
    if problems:
        raise RecipeError("; ".join(problems))
    downloaded = set(table["target_chembl_id"].dropna())
    targets, types = request["targets"], request["types"]
    if not isinstance(targets, list) or not targets or not set(targets) <= downloaded:
        problems.append(f"targets must be a list of downloaded targets: {sorted(downloaded)}")
    if not isinstance(types, list) or not types or not set(types) <= set(TYPES):
        problems.append(f"types must be a list from {list(TYPES)}")
    if not isinstance(request["drop_flagged"], bool):
        problems.append("drop_flagged must be true or false")
    for key, options in MENU.items():
        value = request[key]
        if isinstance(value, bool) or value not in options:
            problems.append(f"{key} must be one of {list(options)}")
    if problems:
        raise RecipeError("; ".join(problems))
    return Recipe(tuple(sorted(set(targets))), tuple(sorted(set(types))), request["min_confidence"],
                  request["drop_flagged"], request["combine"], request["max_spread"])


def first_reason(table: pd.DataFrame, rules: list[tuple[pd.Series, pd.Series | str]]) -> pd.Series:
    # Each row's reason to drop is the first rule it breaks; "" keeps it
    if not rules:
        return pd.Series("", index=table.index)
    masks = [mask.fillna(True).to_numpy(bool) for mask, _ in rules]
    reasons = [reason.to_numpy(str) if isinstance(reason, pd.Series) else reason for _, reason in rules]
    return pd.Series(np.select(masks, reasons, default=""), index=table.index)


def keep_targets(table, recipe):
    return first_reason(table, [(~table["target_chembl_id"].isin(recipe.targets), "target not chosen")])


def keep_types(table, recipe):
    return first_reason(table, [(~table["standard_type"].isin(recipe.types), "measurement type not chosen")])


def keep_binding_assays(table, recipe):
    rules = [(table["assay_type"] != "B", "not a binding assay")]
    if recipe.min_confidence is not None:
        confidence = pd.to_numeric(table["assay_confidence_score"], errors="coerce")
        rules.append((~(confidence >= recipe.min_confidence), f"assay confidence below {recipe.min_confidence}"))
    return first_reason(table, rules)


def drop_flagged(table, recipe):
    if not recipe.drop_flagged:
        return first_reason(table, [])
    comment = table["data_validity_comment"]
    duplicate = table["potential_duplicate"].astype(str).str.lower().isin(["1", "1.0", "true"])
    return first_reason(table, [(comment.notna() & (comment != NOT_A_FLAG), "flagged by ChEMBL: " + comment.astype(str)),
                                (duplicate, "flagged by ChEMBL as a potential duplicate")])


def keep_exact_values(table, recipe):
    relation = table["standard_relation"]
    return first_reason(table, [(relation != "=", "not an exact value (relation " + relation.astype(str) + ")")])


def keep_usable_values(table, recipe):
    value = pd.to_numeric(table["standard_value"], errors="coerce")
    return first_reason(table, [(value.isna(), "no value"),
                                (table["standard_units"] != "nM", "units not nM (" + table["standard_units"].astype(str) + ")"),
                                (value <= 0, "value is zero or negative"),
                                (table["canonical_smiles"].isna(), "no structure"),
                                (table["parent_molecule_chembl_id"].isna(), "no parent compound")])


# Fixed order: each step only sees the rows the steps before it kept
STEPS = [("keep chosen targets", keep_targets), ("keep chosen measurement types", keep_types),
         ("keep binding assays", keep_binding_assays), ("drop records ChEMBL flags", drop_flagged),
         ("keep exact values", keep_exact_values), ("keep usable values", keep_usable_values)]


def to_pchembl(table: pd.DataFrame) -> pd.Series:
    """-log10 of the value in molar: 1 nM gives 9, 1,000 nM gives 6."""
    pchembl = 9 - np.log10(pd.to_numeric(table["standard_value"]))
    given = pd.to_numeric(table["pchembl_value"], errors="coerce")
    disagree = table[(pchembl - given).abs() > AGREEMENT]
    if len(disagree):
        row = disagree.iloc[0]
        raise CheckFailed("The unit conversion", [
            f"{len(disagree):,} values disagree with ChEMBL's pchembl_value, e.g. activity {row['activity_id']}: "
            f"{row['standard_value']} nM gives {pchembl[row.name]:.3f}, ChEMBL says {row['pchembl_value']}"])
    return pchembl


def combine_repeats(kept: pd.DataFrame, recipe: Recipe) -> pd.DataFrame:
    """One row per parent compound (salt forms of a molecule share a parent)."""
    kept = kept.sort_values(["parent_molecule_chembl_id", "molecule_chembl_id", "activity_id"])
    rows = []
    for parent, group in kept.groupby("parent_molecule_chembl_id", sort=True):
        # The parent's own structure if it was measured, otherwise the first form's
        own = group[group["molecule_chembl_id"] == parent]
        source = (own if len(own) else group).iloc[0]
        rows.append({"parent_molecule_chembl_id": parent, "canonical_smiles": source["canonical_smiles"],
                     "smiles_from": source["molecule_chembl_id"],
                     "pchembl": round(float(getattr(group["pchembl"], recipe.combine)()), 4),
                     "n_measurements": len(group),
                     "spread": round(float(group["pchembl"].max() - group["pchembl"].min()), 4),
                     "activity_ids": json.dumps(sorted(int(i) for i in group["activity_id"]))})
    return pd.DataFrame(rows, columns=["parent_molecule_chembl_id", "canonical_smiles", "smiles_from", "pchembl",
                                       "n_measurements", "spread", "activity_ids"])


@dataclass
class Result:
    clean: pd.DataFrame
    dropped: pd.DataFrame
    waterfall: list[dict]


def run(table: pd.DataFrame, recipe: Recipe) -> Result:
    """Apply a recipe in memory and check the result; nothing is saved."""
    kept, dropped, waterfall = table, [], []
    for name, step in STEPS:
        reason = step(kept, recipe)
        out = kept[reason != ""]
        dropped.append(pd.DataFrame({"activity_id": out["activity_id"], "step": name, "reason": reason[reason != ""]}))
        waterfall.append({"step": name, "rows_in": len(kept), "dropped": len(out), "rows_out": len(kept) - len(out)})
        kept = kept[reason == ""]

    kept = kept.assign(pchembl=to_pchembl(kept))
    compounds = combine_repeats(kept, recipe)
    step = "combine repeats"
    if recipe.max_spread is not None:
        too_far = compounds["spread"] > recipe.max_spread
        out_ids = {i: f"repeats disagree by {spread:g} log units"
                   for ids, spread in compounds.loc[too_far, ["activity_ids", "spread"]].itertuples(index=False)
                   for i in json.loads(ids)}
        dropped.append(pd.DataFrame({"activity_id": list(out_ids), "step": step, "reason": list(out_ids.values())}))
        compounds = compounds[~too_far].reset_index(drop=True)
        kept = kept[~kept["activity_id"].isin(out_ids)]
    else:
        out_ids = {}
    waterfall.append({"step": step, "rows_in": len(kept) + len(out_ids), "dropped": len(out_ids),
                      "rows_out": len(kept), "compounds": len(compounds)})

    dropped = pd.concat(dropped, ignore_index=True).sort_values("activity_id", ignore_index=True)
    check_dataset(compounds, kept, recipe.targets, recipe.types, recipe.max_spread)
    reconcile(table["activity_id"], compounds, dropped, waterfall)
    return Result(compounds, dropped, waterfall)


def waterfall_text(waterfall: list[dict]) -> str:
    lines = [f"{'step':<32}{'rows in':>10}{'dropped':>10}{'rows out':>10}"]
    for number, step in enumerate(waterfall, start=1):
        lines.append(f"{number} {step['step']:<30}{step['rows_in']:>10,}{step['dropped']:>10,}{step['rows_out']:>10,}"
                     + (f"   -> {step['compounds']:,} compounds" if "compounds" in step else ""))
    return "\n".join(lines)


def describe(recipe: Recipe) -> str:
    return " · ".join(f"{key}={json.dumps(value)}" for key, value in asdict(recipe).items())


def try_recipe(table: pd.DataFrame, request: dict, look_id: str) -> dict:
    """A dry run as a ledger entry, so the auditor can check numbers quoted from it."""
    recipe = make_recipe(request, table)
    result = run(table, recipe)
    values = result.clean["pchembl"]
    overall = {"compounds": len(result.clean),
               "compounds measured more than once": int((result.clean["n_measurements"] > 1).sum())}
    if len(values):
        overall |= {"pchembl min": float(values.min()), "pchembl median": float(values.median()),
                    "pchembl max": float(values.max())}
    numbers = {f"{step['step']} | {column.replace('_', ' ')}": step[column]
               for step in result.waterfall for column in ("rows_in", "dropped", "rows_out")} | overall
    summary = " · ".join(f"{name}: {value:,}" if isinstance(value, int) else f"{name}: {value:g}"
                         for name, value in overall.items())
    text = f"{look_id} try_recipe {describe(recipe)}\n{waterfall_text(result.waterfall)}\n{summary}"
    return {"id": look_id, "request": {"kind": "try_recipe", **request}, "text": text, "numbers": numbers}


def shown(path: Path) -> str:
    # Paths inside the project are recorded relative to it, so records don't depend on where it lives
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def code_version() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        changed = subprocess.run(["git", "status", "--porcelain", "cb2"], cwd=ROOT, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit.stdout.strip() + (" (with uncommitted changes)" if changed.stdout.strip() else "")


def save(result: Result, source: Path, recipe: Recipe, reason: str = "", run_id: str = "") -> Path:
    folder = CLEAN / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True)
    result.clean.to_csv(folder / "clean.csv", index=False)
    result.dropped.to_csv(folder / "dropped.csv", index=False)
    manifest = {
        "made_at_utc": folder.name,
        "run": run_id,
        "source": {"folder": shown(source),
                   "manifest_sha256": sha256(source / "manifest.json")},
        "recipe": asdict(recipe),
        "reason": reason,
        "code_version": code_version(),
        "waterfall": result.waterfall,
        "rows": {"compounds": len(result.clean), "dropped": len(result.dropped)},
        "sha256": {path.name: sha256(path) for path in sorted(folder.iterdir())},
    }
    # Written last, so a folder without a manifest is unfinished
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return folder


if __name__ == "__main__":
    # Usage: python -m cb2.clean data/raw/<folder> recipe.json [--dry-run]
    import argparse

    parser = argparse.ArgumentParser(description="Clean a raw download with a recipe.")
    parser.add_argument("download", type=Path)
    parser.add_argument("recipe", type=Path, help="a JSON file with the recipe's settings")
    parser.add_argument("--dry-run", action="store_true", help="show the steps without saving anything")
    args = parser.parse_args()
    table = verify_download(args.download)
    recipe = make_recipe(json.loads(args.recipe.read_text()), table)
    result = run(table, recipe)
    print(waterfall_text(result.waterfall))
    if not args.dry_run:
        print(f"Saved to {save(result, args.download.resolve(), recipe)}")
