"""The auditor: checks every number the agents write about the data against the ledger.

Code finds the sentences that contain numbers. A model (the local one by default) reads a few
sentences at a time and labels each number: a claim about the data (and which look it came
from), a choice the team is making, outside knowledge, or something else. Code then decides:
a data claim must match a number in the look it cites. The model labels; it never judges."""

import json
import math
import re

from cb2.llm import ask, combine

SENTENCES_PER_CALL = 4

AUDIT_PROMPT = """You label the numbers in a few sentences from a research team's report about a
dataset. The report cites query results as look ids in brackets, like [L4].

For every number in the sentences, give:
- text: the number exactly as written, e.g. "1,640", "15%", "6.5"
- quote: the shortest exact phrase from the sentences that contains the number
- kind: one of
  - "data": a fact about this dataset, such as a count of rows or values, or a statistic
  - "choice": a threshold, cutoff or setting the team proposes to use
  - "outside": a fact from outside this dataset, such as a unit conversion
  - "other": anything else, such as part of a name or a look id
- look: for "data", the look id the sentence cites for it (e.g. "L4"), otherwise ""
- percent: true if the number is a percentage
- approximate: true if the sentence says it is approximate (about, roughly, ~, nearly)

Sentences:
{sentences}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "numbers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "quote": {"type": "string"},
                    "kind": {"type": "string", "enum": ["data", "choice", "outside", "other"]},
                    "look": {"type": "string"},
                    "percent": {"type": "boolean"},
                    "approximate": {"type": "boolean"},
                },
                "required": ["text", "quote", "kind", "look", "percent", "approximate"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["numbers"],
    "additionalProperties": False,
}

LOOK_REF = re.compile(r"\bL\d+\b")
APPROXIMATE = re.compile(r"(about|around|approximately|roughly|nearly|almost|over|under|~|≈)\s*$", re.IGNORECASE)
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def sentences_with_numbers(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        # Drop markdown emphasis and list markers like "1." or "-" so they don't count as numbers
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•#]+)\s+", "", line.replace("**", "").replace("`", ""))
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", line):
            if NUMBER.search(LOOK_REF.sub("", sentence)):
                found.append(sentence.strip())
    return found


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9.%<>=]+", text.lower().replace(",", "")))


def parse_number(text: str) -> float | None:
    match = NUMBER.search(text.replace("~", ""))
    return float(match.group().replace(",", "")) if match else None


def decimals(text: str) -> int:
    match = re.search(r"\.(\d+)", text)
    return len(match.group(1)) if match else 0


def matches(claimed: float, actual: float, text: str, approximate: bool) -> bool:
    if approximate:
        return math.isclose(claimed, actual, rel_tol=0.05)
    # Exact to the digits written: rounded or cut short both count ("15.2%" or "15%" for 15.23,
    # "66%" for 66.8), but not off by a whole unit of the last digit
    return abs(claimed - actual) < 10 ** -decimals(text) or math.isclose(claimed, actual, rel_tol=1e-9)


def label_overlap(name: str, sentence: str) -> int:
    # How many words of a ledger entry's label (e.g. "CHEMBL2096981 | Ki") appear in the sentence
    words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    return sum(word in words for word in re.findall(r"[a-z0-9]+", name.lower()))


def said_approximately(text: str, sentence_block: str) -> bool:
    # Code, not the model, checks for words like "about" just before the number
    position = sentence_block.find(text)
    return position > 0 and bool(APPROXIMATE.search(sentence_block[max(0, position - 15):position]))


def check(label: dict, sentence_block: str, ledger: dict[str, dict]) -> tuple[str, str] | None:
    """Returns (mark, explanation) for one labelled number, or None to leave it out of the report."""
    quote, text = label["quote"], label["text"]
    if not text.strip() or not quote.strip():
        return None
    if normalize(quote) not in normalize(sentence_block) or normalize(text) not in normalize(quote):
        return "?", "the auditor's quote isn't in the text, so this number wasn't checked"
    approximate = label["approximate"] or said_approximately(text, sentence_block)
    if label["kind"] == "other":
        return None
    if label["kind"] == "choice":
        return "·", "a choice the team proposes"
    if label["kind"] == "outside":
        return "?", "outside knowledge, not checked against the data"
    claimed = parse_number(text)
    if claimed is None:
        return "?", "couldn't read the number"
    # Only trust a look id the text really cites; a model can invent one. If the model gave none,
    # use the looks cited in the same sentence
    # Check the look the model named first (if the text really cites it), then any other look
    # cited in the same sentence
    look_id = label["look"].strip("[] ")
    sentence = next((line for line in sentence_block.splitlines() if normalize(quote) in normalize(line)), "")
    named = [look_id] if look_id and re.search(rf"\b{re.escape(look_id)}\b", sentence_block) else []
    look_ids = [look_id for look_id in dict.fromkeys(named + LOOK_REF.findall(sentence)) if look_id in ledger]
    if not look_ids:
        return "!", "a claim about the data with no lookup cited"
    closest = None
    for look_id in look_ids:
        numbers = ledger[look_id]["numbers"]
        if label["percent"]:
            # A percentage must be one number in the look as a share of another
            candidates = {f"{part} of {whole}": 100 * numbers[part] / numbers[whole]
                          for part in numbers for whole in numbers
                          if numbers[whole] and 0 <= numbers[part] <= numbers[whole] and part != whole}
        else:
            candidates = numbers
        # Several entries can share a value: prefer the one whose label words appear in the sentence,
        # so "25 Ki" for a target matches that target's Ki count, not another target's 25 IC50 values
        found = [(name, actual) for name, actual in candidates.items() if matches(claimed, actual, text, approximate)]
        if found:
            name, actual = max(found, key=lambda item: label_overlap(item[0], sentence))
            shown = f"{actual:.1f}%" if label["percent"] else f"{actual:,g}"
            return "✓", f"matches {look_id}: {name} = {shown}"
        for name, actual in candidates.items():
            if closest is None or abs(actual - claimed) < abs(closest[2] - claimed):
                closest = (look_id, name, actual)
    # Not quoted directly: maybe calculated, as a sum or difference of two numbers in one look
    if not label["percent"]:
        for look_id in look_ids:
            numbers = list(ledger[look_id]["numbers"].items())
            for i, (name_a, a) in enumerate(numbers):
                for name_b, b in numbers[i + 1:]:
                    for result, sign in ((a + b, "+"), (abs(a - b), "−")):
                        if matches(claimed, result, text, approximate):
                            first, second = (name_a, name_b) if sign == "+" or a >= b else (name_b, name_a)
                            return "≈", f"calculated from {look_id}: {first} {sign} {second}"
    hint = f"; closest is {closest[0]}: {closest[1]} = {closest[2]:,.4g}" if closest else ""
    return "✗", f"{', '.join(look_ids)} has no such number{hint}"


def audit(sections: dict[str, str], ledger: list[dict], model: str) -> tuple[str, dict | None]:
    """Audit each named text (e.g. findings, review). Returns the report and combined usage."""
    by_id = {entry["id"]: entry for entry in ledger}
    lines, records = [], []
    for name, text in sections.items():
        sentences = sentences_with_numbers(text)
        results = []
        for start in range(0, len(sentences), SENTENCES_PER_CALL):
            block = "\n".join(sentences[start:start + SENTENCES_PER_CALL])
            try:
                # Labelling is a simple job, so paid models use low effort (local ones skip thinking)
                answer, usage = ask(AUDIT_PROMPT.format(sentences=block), model, "low", schema=SCHEMA)
            except Exception as error:
                # The audit is a check, not a step the run depends on: if the auditor's model
                # can't run (e.g. a local model that won't load), report it and carry on
                lines.append(f"Audit unavailable: {model} failed ({str(error)[:200]}). "
                             f"Pick another auditor_model to check the numbers.")
                return "\n".join(lines), combine(records) if records else None
            records.append(usage)
            for label in parse_labels(answer):
                result = check(label, block, by_id)
                if result:
                    results.append((*result, label["quote"]))
        lines.append(f"{name}: {sum(mark in '✓≈' for mark, _, _ in results)} of "
                     f"{sum(mark in '✓≈✗!' for mark, _, _ in results)} data claims match their lookups")
        lines += [f"  {mark} \"{quote}\": {why}" for mark, why, quote in results]
    lines.append("Key: ✓ matches · ≈ matches a sum or difference of two numbers in the lookup · "
                 "✗ doesn't match · ! no lookup cited · ? not checked · · a choice")
    return "\n".join(lines), combine(records) if records else None


def parse_labels(answer: str) -> list[dict]:
    try:
        return json.loads(answer)["numbers"]
    except (ValueError, KeyError, TypeError):
        return []
