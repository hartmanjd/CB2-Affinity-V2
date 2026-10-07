"""Helpers for asking Claude and local Ollama models a question."""

import os

import anthropic
import ollama
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = "claude-opus-5-5"
OLLAMA_MODEL = "gemma4:26b"
# Tokens the local model can hold at once (prompt plus answer); Ollama's default is too small for long plans
OLLAMA_CONTEXT = 8192


def ask_claude(prompt: str, model: str = CLAUDE_MODEL, max_tokens: int = 16000) -> str:
    # If a safety filter declines the request, retry it on Anthropic's recommended fallback model
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
    return text


def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": OLLAMA_CONTEXT},
    )
    if not response.message.content:
        raise RuntimeError(
            f"{model} returned no answer text (done_reason: {response.done_reason}, "
            f"prompt tokens: {response.prompt_eval_count}, context: {OLLAMA_CONTEXT})"
        )
    return response.message.content


if __name__ == "__main__":
    question = "In two sentences, what is the CB2 receptor?"
    print("Claude:\n" + ask_claude(question))
    print("\nOllama:\n" + ask_ollama(question))
