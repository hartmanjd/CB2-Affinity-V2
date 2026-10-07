"""Check a research plan against the items in checklist.md, using the local model."""

import json
import re
import sys
from pathlib import Path

from cb2.llm import ask_ollama

CHECKLIST = Path(__file__).with_name("checklist.md")

CHECKER_PROMPT = """You check a research plan against a checklist. For every checklist item,
decide whether the plan covers it:
- "covered": the plan says how it handles the item; quote the exact words.
- "missing": the plan does not mention it.
- "unclear": the plan touches on it but not clearly enough to tell.

Checklist:
{checklist}

Plan:
{plan}"""

# Forces the answer into this shape: one entry per item, with a status and a quote
SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"enum": ["covered", "missing", "unclear"]},
                    "quote": {"type": "string"},
                },
                "required": ["id", "status", "quote"],
            },
        }
    },
    "required": ["items"],
}


def item_ids(checklist: str) -> list[str]:
    return re.findall(r"^## (\S+)", checklist, re.MULTILINE)


def normalize(text: str) -> str:
    # Ignore case, punctuation and markdown so small formatting differences still match
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def quote_in_plan(quote: str, plan: str) -> bool:
    quote = normalize(quote)
    return bool(quote) and quote in normalize(plan)


def check_plan(plan: str) -> str:
    checklist = CHECKLIST.read_text()
    answer = json.loads(ask_ollama(CHECKER_PROMPT.format(checklist=checklist, plan=plan), schema=SCHEMA))
    results = {item["id"]: item for item in answer["items"]}

    # The code, not the model, decides whether the answer can be trusted
    lines = []
    for item_id in item_ids(checklist):
        item = results.get(item_id)
        if item is None:
            lines.append(f"- {item_id}: NOT CHECKED (the model skipped this item)")
        elif item["status"] == "covered" and not quote_in_plan(item["quote"], plan):
            lines.append(f"- {item_id}: UNVERIFIED (claimed covered, but the quote is not in the plan)")
        elif item["status"] == "covered":
            lines.append(f'- {item_id}: covered ("{item["quote"].strip()}")')
        else:
            lines.append(f"- {item_id}: {item['status']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Usage: python -m cb2.checklist plan.txt
    print(check_plan(Path(sys.argv[1]).read_text()))
