"""Helpers for asking Claude and local Ollama models a question."""

import os

import anthropic
import ollama
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = "claude-opus-5-5"
OLLAMA_MODEL = "gemma4:26b"


def ask_claude(prompt: str, model: str = CLAUDE_MODEL, max_tokens: int = 16000) -> str:
    response = anthropic.Anthropic().messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    # Keep the answer text, skip the thinking blocks
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text:
        raise RuntimeError(f"Claude returned no answer text (stop_reason: {response.stop_reason})")
    return text


def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response.message.content


if __name__ == "__main__":
    question = "In two sentences, what is the CB2 receptor?"
    print("Claude:\n" + ask_claude(question))
    print("\nOllama:\n" + ask_ollama(question))
