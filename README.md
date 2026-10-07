# CB2-Affinity-V2

A small "research team" of AI models, wired together with LangGraph, that
studies binding affinity at the CB2 receptor. A human approves each step.

## Models

- **Claude** (Anthropic API) for planning
- **GPT** (OpenAI API) for reviewing
- **Local models** through [Ollama](https://ollama.com) (gemma4:26b, qwen3:14b) for checking plans against a checklist and small tasks

## Setup

Needs Python 3.10+ and, for the local models, Ollama.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your Anthropic and OpenAI API keys
```

Check that all three models answer:

```bash
python -m cb2.llm
```
