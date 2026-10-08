"""Helpers for asking Claude, OpenAI and local Ollama models a question.

Each helper returns the answer text and a usage record: the model, its settings, its token
counts, how long it took and what the call cost in dollars. ask() picks the right helper for
any model in MODELS, so any role in the team can be played by any model."""

import json
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
    "claude-sonnet-5-5": {"provider": "anthropic", "efforts": EFFORTS, "fallbacks": True},
    # Haiku 5.5 has no server-side fallback model, so a refusal simply stops the run
    "claude-haiku-5-5": {"provider": "anthropic", "efforts": EFFORTS, "fallbacks": False},
    "gpt-6-astra": {"provider": "openai", "efforts": EFFORTS},
    "gpt-6-sol": {"provider": "openai", "efforts": EFFORTS},
    "gpt-6-luna": {"provider": "openai", "efforts": EFFORTS},
    "gemma4:26b": {"provider": "ollama", "efforts": ()},
    "qwen3:14b": {"provider": "ollama", "efforts": ()},
}

# Tokens a local model can hold at once (prompt plus answer); Ollama's default is too small for long plans.
# Exploring with looks needs more room, which also needs more GPU memory
OLLAMA_CONTEXT = 8192
OLLAMA_TOOL_CONTEXT = 32768


def check_model(role: str, model: str) -> None:
    # Stop at the start of a run with a clear message, rather than fail halfway through
    if model not in MODELS:
        raise ValueError(f"{role}: unknown model {model!r}; add it to MODELS in cb2/llm.py")


def usage_record(model: str, seconds: float, input_tokens: int, output_tokens: int,
                 cache_read_tokens: int = 0, cache_write_tokens: int = 0, effort: str | None = None,
                 requested: str | None = None) -> dict:
    """requested is the model that was asked for, when a fallback model may have answered instead."""
    local = MODELS.get(model, {}).get("provider") == "ollama"
    price = 0.0 if local else dollars(model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
    # Never count an unpriced model as free: estimate it at the requested model's price
    estimated = price is None and requested is not None
    if estimated:
        price = dollars(requested, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
    return {
        "model": model,
        "effort": effort,
        # Wall-clock time for the whole call, including waiting for the first token and any thinking
        "seconds": seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "dollars": price,
        "estimated": estimated,
    }


def ask(prompt: str, model: str, effort: str | None = None, schema: dict | None = None) -> tuple[str, dict]:
    # A schema makes the model answer in JSON of that exact shape
    check_model("ask", model)
    entry = MODELS[model]
    # Models without effort levels (the local ones) ignore the setting
    effort = effort if effort in entry["efforts"] else None
    if entry["provider"] == "anthropic":
        return ask_claude(prompt, model, effort, schema=schema)
    if entry["provider"] == "openai":
        return ask_openai(prompt, model, effort, schema=schema)
    return ask_ollama(prompt, model, schema)


def ask_claude(prompt: str, model: str = "claude-opus-5-5", effort: str | None = None,
               max_tokens: int = 16000, schema: dict | None = None) -> tuple[str, dict]:
    options = {"betas": []}
    if MODELS[model]["fallbacks"]:
        # If a safety filter declines the request, retry it on Anthropic's recommended fallback model
        options["betas"].append("server-side-fallback-2026-07-01")
        options["fallbacks"] = "default"
    output_config = {}
    if effort:
        output_config["effort"] = effort
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    if output_config:
        options["output_config"] = output_config
    start = time.perf_counter()
    response = anthropic.Anthropic().beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **options,
    )
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise RuntimeError(f"Claude declined the request (refusal category: {category})")
    # Keep the answer text, skip the thinking blocks
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text:
        raise RuntimeError(f"Claude returned no answer text (stop_reason: {response.stop_reason})")
    # response.model is whichever model answered; after a fallback, a declined first attempt isn't counted
    usage = response.usage
    return text, usage_record(
        response.model, time.perf_counter() - start, usage.input_tokens, usage.output_tokens,
        usage.cache_read_input_tokens or 0, usage.cache_creation_input_tokens or 0, effort, requested=model,
    )


def ask_openai(prompt: str, model: str = "gpt-6-astra", effort: str | None = None,
               max_tokens: int = 16000, schema: dict | None = None) -> tuple[str, dict]:
    options = {"reasoning": {"effort": effort}} if effort else {}
    if schema:
        options["text"] = {"format": {"type": "json_schema", "name": "answer", "schema": schema, "strict": True}}
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
        # A fixed-format answer is a labelling job: thinking first made gemma4 take ~10x longer
        # and sometimes fill its whole context before answering
        think=False if schema else None,
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


def combine(records: list[dict]) -> dict:
    # One usage record for a whole tool loop: tokens, time and dollars added up over its API calls
    total = {key: sum(record[key] for record in records)
             for key in ("seconds", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
    prices = [record["dollars"] for record in records]
    # If a fallback model took over partway, show both, e.g. "claude-opus-5-5 → claude-opus-5"
    models = " → ".join(dict.fromkeys(record["model"] for record in records))
    return {"model": models, "effort": records[-1]["effort"], **total,
            "dollars": None if None in prices else sum(prices), "calls": len(records),
            "estimated": any(record.get("estimated") for record in records)}


LIMIT_REACHED = "No more lookups are allowed this round. Write your answer now from what you have."


def ask_with_tools(prompt: str, model: str, effort: str | None, tools: list[dict],
                   run_tool, max_tool_calls: int, uncounted: tuple[str, ...] = ()) -> tuple[str, dict]:
    """Let the model call tools (e.g. look) up to max_tool_calls times, then answer.

    run_tool(name, request) takes a tool's name and input and returns the result text for the
    model. Tools named in uncounted (e.g. submitting a proposal) don't count towards the limit."""
    check_model("ask_with_tools", model)
    entry = MODELS[model]
    effort = effort if effort in entry["efforts"] else None
    loop = {"anthropic": claude_tool_loop, "openai": openai_tool_loop, "ollama": ollama_tool_loop}[entry["provider"]]
    used = 0

    def answer(name: str, request: dict) -> str:
        nonlocal used
        if name in uncounted:
            return run_tool(name, request)
        used += 1
        return run_tool(name, request) if used <= max_tool_calls else LIMIT_REACHED

    # After the limit, a model that keeps asking gets a few refusals, then one last call without tools
    def out_of_tool_calls() -> bool:
        return used >= max_tool_calls + 3

    text, records = loop(prompt, model, effort, tools, answer, out_of_tool_calls)
    return text, combine(records)


def claude_tool_loop(prompt, model, effort, tools, answer, out_of_tool_calls):
    client = anthropic.Anthropic()
    options = {"betas": []}
    if MODELS[model]["fallbacks"]:
        options["betas"].append("server-side-fallback-2026-07-01")
        options["fallbacks"] = "default"
    if effort:
        options["output_config"] = {"effort": effort}
    messages, records = [{"role": "user", "content": prompt}], []
    while True:
        start = time.perf_counter()
        # cache_control caches the conversation so far, so each new request pays full price only for what's new
        response = client.beta.messages.create(
            model=model, max_tokens=16000, messages=messages, tools=tools,
            tool_choice={"type": "none"} if out_of_tool_calls() else {"type": "auto"},
            cache_control={"type": "ephemeral"}, **options,
        )
        usage = response.usage
        records.append(usage_record(
            response.model, time.perf_counter() - start, usage.input_tokens, usage.output_tokens,
            usage.cache_read_input_tokens or 0, usage.cache_creation_input_tokens or 0, effort, requested=model))
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request (refusal category: "
                               f"{response.stop_details.category if response.stop_details else None})")
        calls = [block for block in response.content if block.type == "tool_use"]
        if not calls:
            text = "".join(block.text for block in response.content if block.type == "text")
            if not text:
                raise RuntimeError(f"Claude returned no answer text (stop_reason: {response.stop_reason})")
            return text, records
        # Send the whole reply back (thinking included), then every tool result in one message
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, "content": answer(call.name, call.input)}
            for call in calls]})


def openai_tool_loop(prompt, model, effort, tools, answer, out_of_tool_calls):
    client = openai.OpenAI()
    functions = [{"type": "function", "name": tool["name"], "description": tool["description"],
                  "parameters": tool["input_schema"]} for tool in tools]
    options = {"reasoning": {"effort": effort}} if effort else {}
    previous, new_input, records = None, [{"role": "user", "content": prompt}], []
    while True:
        start = time.perf_counter()
        # previous_response_id continues the conversation on OpenAI's side, reasoning included
        response = client.responses.create(
            model=model, input=new_input, previous_response_id=previous, tools=functions,
            tool_choice="none" if out_of_tool_calls() else "auto", max_output_tokens=16000, **options,
        )
        usage = response.usage
        cached = usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
        records.append(usage_record(model, time.perf_counter() - start, usage.input_tokens - cached,
                                    usage.output_tokens, cached, effort=effort))
        if response.status != "completed":
            reason = response.incomplete_details.reason if response.incomplete_details else None
            raise RuntimeError(f"OpenAI answer incomplete (status: {response.status}, reason: {reason})")
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            if not response.output_text:
                raise RuntimeError("OpenAI returned no answer text")
            return response.output_text, records
        previous = response.id
        new_input = [{"type": "function_call_output", "call_id": call.call_id,
                      "output": answer(call.name, json.loads(call.arguments))} for call in calls]


def ollama_tool_loop(prompt, model, effort, tools, answer, out_of_tool_calls):
    client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    functions = [{"type": "function", "function": {"name": tool["name"], "description": tool["description"],
                                                   "parameters": tool["input_schema"]}} for tool in tools]
    messages, records = [{"role": "user", "content": prompt}], []
    while True:
        start = time.perf_counter()
        response = client.chat(model=model, messages=messages, tools=None if out_of_tool_calls() else functions,
                               options={"num_ctx": OLLAMA_TOOL_CONTEXT})
        records.append(usage_record(model, time.perf_counter() - start,
                                    response.prompt_eval_count or 0, response.eval_count or 0))
        # Ollama quietly drops the start of a conversation that no longer fits, so stop instead
        if (response.prompt_eval_count or 0) >= OLLAMA_TOOL_CONTEXT - 1000:
            raise RuntimeError(f"{model} ran out of context ({OLLAMA_TOOL_CONTEXT} tokens); allow fewer looks")
        calls = response.message.tool_calls or []
        if not calls:
            if not response.message.content:
                raise RuntimeError(f"{model} returned no answer text (done_reason: {response.done_reason})")
            return response.message.content, records
        messages.append(response.message)
        for call in calls:
            messages.append({"role": "tool", "tool_name": call.function.name,
                             "content": answer(call.function.name, dict(call.function.arguments))})


if __name__ == "__main__":
    # Usage: python -m cb2.llm [model ...]   (default: one model from each provider)
    import sys

    question = "In two sentences, what is the CB2 receptor?"
    for model in sys.argv[1:] or ["claude-opus-5-5", "gpt-6-astra", "gemma4:26b"]:
        text, usage = ask(question, model)
        print(f"{model}:\n{text}\n{usage}\n")
