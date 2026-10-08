"""Helpers for asking Claude, OpenAI and local Ollama models a question.

Each helper returns the answer text and a usage record: the model, its token counts and
what the call cost in dollars."""

import os
import time

import anthropic
import ollama
import openai
from dotenv import load_dotenv

from cb2.costs import dollars

load_dotenv()

CLAUDE_MODEL = "claude-opus-5-5"
OPENAI_MODEL = "gpt-6-astra"
OLLAMA_MODEL = "gemma4:26b"
# Tokens the local model can hold at once (prompt plus answer); Ollama's default is too small for long plans
OLLAMA_CONTEXT = 8192


def usage_record(model: str, seconds: float, input_tokens: int, output_tokens: int,
                 cache_read_tokens: int = 0, cache_write_tokens: int = 0, local: bool = False) -> dict:
    return {
        "model": model,
        # Wall-clock time for the whole call, including waiting for the first token and any thinking
        "seconds": seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "dollars": 0.0 if local else dollars(model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens),
    }


def ask_claude(prompt: str, model: str = CLAUDE_MODEL, max_tokens: int = 16000) -> tuple[str, dict]:
    # If a safety filter declines the request, retry it on Anthropic's recommended fallback model
    start = time.perf_counter()
    response = anthropic.Anthropic().beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
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
        usage.cache_read_input_tokens or 0, usage.cache_creation_input_tokens or 0,
    )


def ask_openai(prompt: str, model: str = OPENAI_MODEL, max_tokens: int = 16000) -> tuple[str, dict]:
    start = time.perf_counter()
    response = openai.OpenAI().responses.create(model=model, input=prompt, max_output_tokens=max_tokens)
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
    return response.output_text, usage_record(model, seconds, usage.input_tokens - cached, usage.output_tokens, cached)


def ask_ollama(prompt: str, model: str = OLLAMA_MODEL, schema: dict | None = None) -> tuple[str, dict]:
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
    usage = usage_record(model, seconds, response.prompt_eval_count or 0, response.eval_count or 0, local=True)
    return response.message.content, usage


if __name__ == "__main__":
    question = "In two sentences, what is the CB2 receptor?"
    for name, ask in [("Claude", ask_claude), ("OpenAI", ask_openai), ("Ollama", ask_ollama)]:
        text, usage = ask(question)
        print(f"{name}:\n{text}\n{usage}\n")
