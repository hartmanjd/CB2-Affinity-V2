"""The run log: a readable record of every round and every answer the lead gave.

Each run gets a folder under runs/ (committed to git) with:
- log.md: round by round, what the agents produced, the audit, the costs and the lead's reply
- ledger.json: every lookup (search, count, look) with its full result

Both are rewritten in full at every pause, from the run's history, so writing them twice
(LangGraph re-runs a paused node when it resumes) does no harm."""

import json
from dataclasses import asdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"

STAGE_TITLES = {"plan": "Plan", "source": "Get the data", "data": "Explore the data", "budget": "Budget"}


def entry(stage: str, round_number: int, sections: list[tuple[str, str]], feedback: str | None = None) -> dict:
    """One round (or budget question) for the history. sections are (title, text) pairs."""
    return {"stage": stage, "round": round_number, "sections": [list(section) for section in sections],
            "feedback": feedback}


def render(run_id: str, question: str, settings, history: list[dict]) -> str:
    lines = [f"# Run {run_id}", "", f"**Question:** {question}", ""]
    if settings is not None:
        lines += ["**Team:** " + " · ".join(f"{key} = {value}" for key, value in asdict(settings).items()), ""]
    for item in history:
        title = STAGE_TITLES.get(item["stage"], item["stage"])
        lines += [f"## Round {item['round']} · {title}" if item["stage"] != "budget" else "## Budget", ""]
        for heading, text in item["sections"]:
            lines += [f"### {heading}", ""]
            # Tables and cost lines keep their columns in a code block
            lines += ["```", text, "```", ""] if heading in ("Costs", "Audit", "Lookups") else [text, ""]
        reply = item["feedback"]
        lines += ["### Lead", "", "*(waiting for the lead)*" if reply is None else
                  "> " + (reply or "(blank: revise from the review)").replace("\n", "\n> "), ""]
    return "\n".join(lines)


def write(run_id: str, question: str, settings, history: list[dict], ledger: list[dict]) -> Path:
    folder = RUNS / run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "log.md").write_text(render(run_id, question, settings, history))
    (folder / "ledger.json").write_text(json.dumps(ledger, indent=1, default=str))
    return folder
