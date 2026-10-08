"""Turn token counts from each model call into dollars, and summarise a run's spending."""

# Dollars per million tokens. Check the providers' pricing pages before trusting these:
# https://platform.claude.com/docs/en/about-claude/pricing
# https://developers.openai.com/api/docs/pricing
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00},
    "gpt-6-astra": {"input": 10.00, "cache_write": 12.50, "cache_read": 1.00, "output": 50.00},
}


def dollars(model: str, input_tokens: int, output_tokens: int,
            cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float | None:
    # A model missing from the table gets None, so it shows up as "price unknown" instead of $0
    if model not in PRICES:
        return None
    price = PRICES[model]
    total = (input_tokens * price["input"] + output_tokens * price["output"]
             + cache_read_tokens * price["cache_read"] + cache_write_tokens * price["cache_write"])
    return total / 1_000_000


def spent(costs: list[dict]) -> float:
    return sum(call["dollars"] or 0 for call in costs)


def round_cost(costs: list[dict], round_number: int) -> float:
    return spent([call for call in costs if call["round"] == round_number])


def format_call(call: dict) -> str:
    price = "price unknown" if call["dollars"] is None else f"${call['dollars']:.3f}"
    # Output tokens per second of the whole call: how fast the answer arrived, waiting included
    speed = call["output_tokens"] / call["seconds"] if call["seconds"] else 0
    return (f"{call['node']:<10} {call['model']:<18} {price:>14}   "
            f"{call['input_tokens']:>7,} in {call['cache_read_tokens']:>7,} cached {call['output_tokens']:>6,} out   "
            f"{call['seconds']:>5.1f}s {speed:>5.0f} tok/s")


def summary(costs: list[dict], round_number: int, budget: float) -> str:
    lines = [format_call(call) for call in costs if call["round"] == round_number]
    seconds = sum(call["seconds"] for call in costs if call["round"] == round_number)
    lines.append(f"This round: ${round_cost(costs, round_number):.3f} in {seconds:.0f}s   "
                 f"Run total: ${spent(costs):.3f} of ${budget:.2f} budget")
    return "\n".join(lines)
