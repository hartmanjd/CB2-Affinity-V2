"""Helpers for asking Claude, OpenAI and local Ollama models a question.

Each helper returns the answer text and a usage record: the model, its settings, its token
counts, how long it took and what the call cost in dollars. ask() picks the right helper for
any model in MODELS, so any role in the team can be played by any model."""

import os
import time

import anthropic
import ollama
import openai
from dotenv import load_dotenv

from cb2.costs import dollars

load_dotenv()

EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Every model the team can use, by the name shown in the settings. To add one (say, a bigger
# local model), add a line here, and for a paid model also add its prices to cb2/costs.py.
MODELS = {
    "claude-opus-5-5": {"provider": "anthropic", "efforts": EFFORTS, "fallbacks": True},
    # Same model in fast mode: faster output at double the price. Anthropic has to enable it for
    # the account first (https://claude.com/fast-mode); until then its rate limit is 0
    "claude-opus-5-5 (fast)": {"provider": "anthropic", "api_model": "claude-opus-5-5", "fast": True,
                               "efforts": EFFORTS, "fallbacks": True},
    "claude-sonnet-5-5": {"provider": "anthropic", "efforts": EFFORTS, "fallbacks": True},
    # Haiku 5.5 has no server-side fallback model, so a refusal simply stops the run
    "claude-haiku-5-5": {"provider": "anthropic", "efforts": EFFORTS, "fallbacks": False},
    "gpt-6-astra": {"provider": "openai", "efforts": EFFORTS},
    "gemma4:26b": {"provider": "ollama", "efforts": ()},
    "qwen3:14b": {"provider": "ollama", "efforts": ()},
}

# Tokens the local model can hold at once (prompt plus answer); Ollama's default is too small for long plans
OLLAMA_CONTEXT = 8192


def check_model(role: str, model: str) -> None:
    # Stop at the start of a run with a clear message, rather than fail halfway through
    if model not in MODELS:
        raise ValueError(f"{role}: unknown model {model!r}; add it to MODELS in cb2/llm.py")


def usage_record(model: str, seconds: float, input_tokens: int, output_tokens: int,
                 cache_read_tokens: int = 0, cache_write_tokens: int = 0,
                 effort: str | None = None, fast: bool = False) -> dict:
    local = MODELS.get(model, {}).get("provider") == "ollama"
    return {
        "model": model,
        "effort": effort,
        "fast": fast,
        # Wall-clock time for the whole call, including waiting for the first token and any thinking
        "seconds": seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "dollars": 0.0 if local else dollars(
            model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, fast),
    }


def ask(prompt: str, model: str, effort: str | None = None) -> tuple[str, dict]:
    check_model("ask", model)
    entry = MODELS[model]
    # Models without effort levels (the local ones) ignore the setting
    effort = effort if effort in entry["efforts"] else None
    api_model = entry.get("api_model", model)
    if entry["provider"] == "anthropic":
        return ask_claude(prompt, api_model, effort, entry.get("fast", False))
    if entry["provider"] == "openai":
        return ask_openai(prompt, api_model, effort)
    return ask_ollama(prompt, api_model)


def ask_claude(prompt: str, model: str = "claude-opus-5-5", effort: str | None = None,
               fast: bool = False, max_tokens: int = 16000) -> tuple[str, dict]:
    options = {"betas": []}
    if MODELS[model]["fallbacks"]:
        # If a safety filter declines the request, retry it on Anthropic's recommended fallback model
        options["betas"].append("server-side-fallback-2026-07-01")
        options["fallbacks"] = "default"
    if effort:
        options["output_config"] = {"effort": effort}
    if fast:
        # Same model, faster output, double the price
        options["betas"].append("fast-mode-2026-02-01")
        options["speed"] = "fast"
    start = time.perf_counter()
    try:
        response = anthropic.Anthropic().beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **options,
        )
    except anthropic.RateLimitError as error:
        if not fast:
            raise
        # Fast mode is a research preview with its own rate limit, which is 0 until Anthropic enables it
        raise RuntimeError(
            "Fast mode was rate limited. If the limit is 0, fast mode isn't enabled for this "
            "Anthropic account yet (https://claude.com/fast-mode); pick claude-opus-5-5 instead. "
            f"Details: {error.message}"
        ) from error
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise RuntimeError(f"Claude declined the request (refusal category: {category})")
    # Keep the answer text, skip the thinking blocks
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text:
        raise RuntimeError(f"Claude returned no answer text (stop_reason: {response.stop_reason})")
    # response.model is whichever model answered; after a fallback, a declined first attempt isn't counted.
    # usage.speed says whether fast mode was actually used, which decides the price
    usage = response.usage
    return text, usage_record(
        response.model, time.perf_counter() - start, usage.input_tokens, usage.output_tokens,
        usage.cache_read_input_tokens or 0, usage.cache_creation_input_tokens or 0,
        effort, getattr(usage, "speed", None) == "fast",
    )


def ask_openai(prompt: str, model: str = "gpt-6-astra", effort: str | None = None,
               max_tokens: int = 16000) -> tuple[str, dict]:
    options = {"reasoning": {"effort": effort}} if effort else {}
    start = time.perf_counter()
    response = openai.OpenAI().responses.create(model=model, input=prompt, max_output_tokens=max_tokens, **options)
    seconds = time.perf_counter() - start
    if response.status != "completed":
        reason = response.incomplete_details.reason if response.incomplete_details else None
        raise RuntimeError(f"OpenAI answer incomplete (status: {response.status}, reason: {reason})")
    refusals = [part.refusal for item in response.output if item.type == "message"
                for part in item.content if part.type == "refusal"]
    if refusals:
        raise RuntimeError(f"OpenAI declined the request: {refusals[0]}")
    if not response.output_text:
        raise RuntimeError("OpenAI returned no answer text")
    # OpenAI counts cached tokens inside input_tokens, so take them out to price them separately
    usage = response.usage
    cached = usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
    return response.output_text, usage_record(
        model, seconds, usage.input_tokens - cached, usage.output_tokens, cached, effort=effort)


def ask_ollama(prompt: str, model: str = "gemma4:26b", schema: dict | None = None) -> tuple[str, dict]:
    # A schema makes the model answer in JSON of that exact shape
    client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    start = time.perf_counter()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"num_ctx": OLLAMA_CONTEXT},
    )
    if not response.message.content:
        raise RuntimeError(
            f"{model} returned no answer text (done_reason: {response.done_reason}, "
            f"prompt tokens: {response.prompt_eval_count}, context: {OLLAMA_CONTEXT})"
        )
    # Local models cost nothing, but the token counts still show how much work they did
    seconds = time.perf_counter() - start
    return response.message.content, usage_record(
        model, seconds, response.prompt_eval_count or 0, response.eval_count or 0)


if __name__ == "__main__":
    # Usage: python -m cb2.llm [model ...]   (default: one model from each provider)
    import sys

    question = "In two sentences, what is the CB2 receptor?"
    for model in sys.argv[1:] or ["claude-opus-5-5", "gpt-6-astra", "gemma4:26b"]:
        text, usage = ask(question, model)
        print(f"{model}:\n{text}\n{usage}\n")
