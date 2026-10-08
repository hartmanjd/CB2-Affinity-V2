"""The look tool: exact answers about the data table, computed by code.

The agents start knowing only the column names and the row count, and ask questions with
looks. Every look is numbered (L1, L2, ...; searches and counts share the numbering) and its result, including every number in it, is
kept in the ledger, so the auditor can check numbers the agents quote against what the code
actually said. Results are capped in size to keep token use down."""

import difflib

import pandas as pd

MAX_VALUES = 20
MAX_ROWS = 5
MAX_COLUMNS = 8
MAX_TEXT = 60
CROSSTAB_ROWS, CROSSTAB_COLUMNS = 12, 8

OPS = ["==", "!=", "in", "not in", "<", "<=", ">", ">=", "contains", "is missing", "not missing"]

LOOK_TOOL = {
    "name": "look",
    "description": (
        "Ask an exact question about the data table; code computes the answer.\n"
        "- value_counts: the values of `column` and how many rows have each (top 20)\n"
        "- crosstab: counts for each pair of values of `column` and `column2`\n"
        "- describe: count, missing, min, quartiles, max and mean of a numeric `column`\n"
        "- sample: a few example rows, showing `columns`\n"
        "Any look can be limited to rows matching every condition in `where`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["value_counts", "crosstab", "describe", "sample"]},
            "column": {"type": "string", "description": "The column to look at (value_counts, crosstab, describe)"},
            "column2": {"type": "string", "description": "The second column (crosstab only)"},
            "columns": {"type": "array", "items": {"type": "string"},
                        "description": f"Columns to show (sample only, at most {MAX_COLUMNS})"},
            "where": {
                "type": "array",
                "description": "Only count rows matching all of these conditions",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "op": {"type": "string", "enum": OPS},
                        "value": {"description": "A value, or a list of values for 'in' / 'not in'"},
                    },
                    "required": ["column", "op"],
                },
            },
        },
        "required": ["kind"],
    },
}


class LookError(ValueError):
    pass


def check_column(table: pd.DataFrame, name: str | None) -> str:
    if not name:
        raise LookError("This look needs a column.")
    if name not in table.columns:
        close = difflib.get_close_matches(name, table.columns, n=3)
        raise LookError(f"There is no column {name!r}." + (f" Did you mean {', '.join(close)}?" if close else ""))
    return name


def short(value) -> str:
    text = "(missing)" if pd.isna(value) else str(value)
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT - 3] + "..."


def apply_where(table: pd.DataFrame, where: list[dict] | None) -> tuple[pd.DataFrame, str]:
    descriptions = []
    for condition in where or []:
        column = table[check_column(table, condition.get("column"))]
        op, value = condition.get("op"), condition.get("value")
        if op not in OPS:
            raise LookError(f"Unknown op {op!r}; use one of {OPS}.")
        if op in ("in", "not in"):
            values = value if isinstance(value, list) else [value]
            keep = column.isin(values) | (column.astype(str).isin([str(v) for v in values]))
            keep = keep if op == "in" else ~keep
        elif op in ("<", "<=", ">", ">="):
            numbers = pd.to_numeric(column, errors="coerce")
            keep = {"<": numbers < value, "<=": numbers <= value, ">": numbers > value, ">=": numbers >= value}[op]
        elif op == "contains":
            keep = column.astype(str).str.contains(str(value), case=False, regex=False) & column.notna()
        elif op == "is missing":
            keep = column.isna()
        elif op == "not missing":
            keep = column.notna()
        else:
            # == and != compare as text too, so 9 matches a column read as "9"
            same = (column == value) | (column.astype(str) == str(value))
            keep = same if op == "==" else ~same
        table = table[keep]
        descriptions.append(f"{condition['column']} {op}" + ("" if "missing" in op else f" {value!r}"))
    return table, " and ".join(descriptions)


def value_counts(table: pd.DataFrame, column: str) -> tuple[list[str], dict]:
    counts = table[column].value_counts(dropna=True)
    missing = int(table[column].isna().sum())
    numbers = {"rows": len(table), "missing": missing, "distinct values": len(counts)}
    lines = [f"rows: {len(table):,} · missing: {missing:,} · distinct values: {len(counts):,}"]
    width = max((len(short(value)) for value in counts.index[:MAX_VALUES]), default=5)
    for value, count in counts.head(MAX_VALUES).items():
        lines.append(f"  {short(value):<{width}}  {count:>9,}")
        numbers[short(value)] = int(count)
    if len(counts) > MAX_VALUES:
        rest = counts.iloc[MAX_VALUES:]
        lines.append(f"  ({len(rest):,} more values, {int(rest.sum()):,} rows)")
        numbers["other values"], numbers["rows in other values"] = len(rest), int(rest.sum())
    return lines, numbers


def crosstab(table: pd.DataFrame, column: str, column2: str) -> tuple[list[str], dict]:
    rows = table[column].fillna("(missing)").astype(str)
    columns = table[column2].fillna("(missing)").astype(str)
    top_rows = rows.value_counts().index[:CROSSTAB_ROWS]
    top_columns = columns.value_counts().index[:CROSSTAB_COLUMNS]
    # Values outside the top ones are grouped as "(other)"
    rows = rows.where(rows.isin(top_rows), "(other)")
    columns = columns.where(columns.isin(top_columns), "(other)")
    counts = pd.crosstab(rows, columns, margins=True, margins_name="total")
    numbers = {"rows": len(table)}
    labels = [short(label)[:20] for label in counts.columns]
    row_width = max(len(short(label)[:28]) for label in counts.index)
    lines = [f"rows: {len(table):,} · rows = {column}, columns = {column2}",
             "  " + " " * row_width + "".join(f"{label:>12}" for label in labels)]
    for row_label, row in counts.iterrows():
        lines.append(f"  {short(row_label)[:28]:<{row_width}}" + "".join(f"{int(n):>12,}" for n in row))
        for column_label, n in row.items():
            numbers[f"{row_label} | {column_label}"] = int(n)
    return lines, numbers


def describe(table: pd.DataFrame, column: str) -> tuple[list[str], dict]:
    values = pd.to_numeric(table[column], errors="coerce")
    not_numbers = int((values.isna() & table[column].notna()).sum())
    present = values.dropna()
    if not len(present) and not_numbers:
        # describe only makes sense for numbers; say so plainly instead of reporting "0 numeric values"
        raise LookError(f"{column} holds text, not numbers ({not_numbers:,} text values, "
                        f"{int(table[column].isna().sum()):,} missing). Use value_counts or sample instead.")
    numbers = {"rows": len(table), "numeric values": len(present), "missing": int(table[column].isna().sum()),
               "not numeric": not_numbers}
    if len(present):
        stats = {"min": present.min(), "25%": present.quantile(0.25), "median": present.median(),
                 "75%": present.quantile(0.75), "max": present.max(), "mean": present.mean()}
        numbers.update({name: float(f"{value:.4g}") for name, value in stats.items()})
    return [" · ".join(f"{name}: {value:,}" if isinstance(value, int) else f"{name}: {value:g}"
                       for name, value in numbers.items())], numbers


def sample(table: pd.DataFrame, columns: list[str] | None) -> tuple[list[str], dict]:
    if not columns:
        raise LookError("A sample needs `columns`: the columns to show.")
    columns = [check_column(table, name) for name in columns[:MAX_COLUMNS]]
    # A fixed random_state, so the same look always shows the same rows
    rows = table.sample(min(MAX_ROWS, len(table)), random_state=0) if len(table) else table
    numbers = {"rows": len(table)}
    lines = [f"rows: {len(table):,} · showing {len(rows)}"]
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        lines.append(f"  row {i}: " + " · ".join(f"{name}={short(row[name])}" for name in columns))
        for name in columns:
            value = pd.to_numeric(row[name], errors="coerce")
            if pd.notna(value):
                numbers[f"row {i} {name}"] = float(value)
    return lines, numbers


def look(table: pd.DataFrame, request: dict, look_id: str) -> dict:
    """Run one look. Returns a ledger entry; a bad request gives an entry with an error message."""
    kind = request.get("kind")
    try:
        subset, condition = apply_where(table, request.get("where"))
        if kind == "value_counts":
            lines, numbers = value_counts(subset, check_column(table, request.get("column")))
        elif kind == "crosstab":
            lines, numbers = crosstab(subset, check_column(table, request.get("column")),
                                      check_column(table, request.get("column2")))
        elif kind == "describe":
            lines, numbers = describe(subset, check_column(table, request.get("column")))
        elif kind == "sample":
            lines, numbers = sample(subset, request.get("columns"))
        else:
            raise LookError(f"Unknown kind {kind!r}; use value_counts, crosstab, describe or sample.")
    except LookError as error:
        return {"id": look_id, "request": request, "text": f"{look_id} error: {error}", "numbers": {}}
    target = request.get("column", "") + (f" by {request['column2']}" if kind == "crosstab" else "")
    header = f"{look_id} {kind} {target}".rstrip() + (f" where {condition}" if condition else "")
    return {"id": look_id, "request": request, "text": "\n".join([header, *lines]), "numbers": numbers}


def overview(table: pd.DataFrame, look_id: str) -> dict:
    # All the agents know about the downloaded table before their first look
    text = f"{look_id} the table has {len(table):,} rows and {len(table.columns)} columns:\n" + ", ".join(table.columns)
    return {"id": look_id, "request": {"kind": "overview"}, "text": text,
            "numbers": {"rows": len(table), "columns": len(table.columns)}}
