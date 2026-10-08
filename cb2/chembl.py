"""ChEMBL's web API, and the tools the team uses to decide which data to download.

The tools know how ChEMBL is organised (targets have an organism and a type, and activity
records belong to targets) but not which data the study needs: the agents decide that. Like
looks, every search and count is numbered (L1, L2, ...) and kept in the ledger for the auditor."""

from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1000
MAX_SEARCH_RESULTS = 15
MAX_COUNTS = 40

SEARCH_TOOL = {
    "name": "search_targets",
    "description": (
        "Search ChEMBL's targets (proteins, protein families, complexes...) by name or keyword. "
        f"Returns up to {MAX_SEARCH_RESULTS} matches, each with its ChEMBL ID, name, organism, "
        "target type, gene symbols and number of activity records."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Words to search for"}},
        "required": ["query"],
    },
}

COUNT_TOOL = {
    "name": "count_activities",
    "description": (
        "Count a target's activity records in ChEMBL, in total and for each measurement type "
        "you name (ChEMBL's standard_type, e.g. a kind of affinity or potency value). "
        f"At most {MAX_COUNTS} counts (targets x types) per call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "target_ids": {"type": "array", "items": {"type": "string"}, "description": "ChEMBL target IDs"},
            "standard_types": {"type": "array", "items": {"type": "string"},
                               "description": "Measurement types to count (optional)"},
        },
        "required": ["target_ids"],
    },
}

PROPOSE_TOOL = {
    "name": "propose_download",
    "description": (
        "Propose the targets to download: every activity record for each, with its assay "
        "details. The project lead approves the proposal before anything is downloaded. "
        "Calling it again replaces the earlier proposal."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "target_ids": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string", "description": "Why these targets, and why not others"},
        },
        "required": ["target_ids", "reason"],
    },
}


def get(path: str, **params) -> dict:
    response = requests.get(f"{API}/{path}", params={"format": "json", **params}, timeout=120)
    response.raise_for_status()
    return response.json()


def get_all(path: str, key: str, **params) -> list[dict]:
    # ChEMBL returns results in pages; follow page_meta.next until there are none left
    records, offset = [], 0
    while True:
        page = get(path, limit=PAGE_SIZE, offset=offset, **params)
        records += page[key]
        print(f"  {path}: {len(records):,} of {page['page_meta']['total_count']:,}")
        if not page["page_meta"]["next"]:
            return records
        offset += PAGE_SIZE


def count(**filters) -> int:
    return get("activity", limit=1, only="activity_id", **filters)["page_meta"]["total_count"]


def in_parallel(function, items: list) -> list:
    # ChEMBL takes a second or two per request, so ask several things at once
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(function, items))


def genes(target: dict) -> list[str]:
    return sorted({synonym["component_synonym"]
                   for component in target.get("target_components", [])
                   for synonym in component.get("target_component_synonyms", [])
                   if synonym["syn_type"] == "GENE_SYMBOL"})


def search_targets(query: str, look_id: str) -> dict:
    found = get("target/search", q=query, limit=MAX_SEARCH_RESULTS)["targets"]
    counts = in_parallel(lambda target: count(target_chembl_id=target["target_chembl_id"]), found)
    lines = [f"{look_id} search_targets {query!r}: {len(found)} matches"]
    numbers = {"matches": len(found)}
    for target, n in zip(found, counts):
        lines.append(f"  {target['target_chembl_id']:<14} {target['pref_name'][:40]:<40} "
                     f"{(target['organism'] or '-')[:24]:<24} {target['target_type']:<22} "
                     f"{', '.join(genes(target))[:24]:<24} {n:>8,} activities")
        numbers[f"{target['target_chembl_id']} activities"] = n
    return {"id": look_id, "request": {"kind": "search_targets", "query": query},
            "text": "\n".join(lines), "numbers": numbers}


def count_activities(target_ids: list[str], standard_types: list[str] | None, look_id: str) -> dict:
    types = standard_types or []
    pairs = [(target, None) for target in target_ids] + [(t, s) for t in target_ids for s in types]
    if len(pairs) > MAX_COUNTS:
        raise ValueError(f"That's {len(pairs)} counts; at most {MAX_COUNTS} per call.")
    results = in_parallel(lambda pair: count(target_chembl_id=pair[0], **({"standard_type": pair[1]} if pair[1] else {})),
                          pairs)
    found = dict(zip(pairs, results))
    lines = [f"{look_id} count_activities " + ", ".join(target_ids) + (f" by {', '.join(types)}" if types else ""),
             "  " + f"{'target':<14} {'all':>9}" + "".join(f"{s[:12]:>13}" for s in types)]
    numbers = {}
    for target in target_ids:
        lines.append(f"  {target:<14} {found[(target, None)]:>9,}" + "".join(f"{found[(target, s)]:>13,}" for s in types))
        numbers[f"{target} | all"] = found[(target, None)]
        numbers.update({f"{target} | {s}": found[(target, s)] for s in types})
    return {"id": look_id, "request": {"kind": "count_activities", "target_ids": target_ids, "standard_types": types},
            "text": "\n".join(lines), "numbers": numbers}


def describe_targets(target_ids: list[str]) -> list[dict]:
    """The targets in a proposal, checked against ChEMBL; raises ValueError for an unknown ID."""
    def one(target_id: str) -> dict:
        try:
            return get(f"target/{target_id}")
        except requests.HTTPError as error:
            raise ValueError(f"ChEMBL has no target {target_id!r}") from error
    return in_parallel(one, target_ids)
