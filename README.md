# CB2-Affinity-V2

A small "research team" of AI models, wired together with LangGraph, that
studies binding affinity at the CB2 receptor. A human approves each step.

## Models

By default **Claude** (claude-opus-5-5) plans and **GPT** (gpt-6-astra) reviews. Any role
can be played by any model in `MODELS` in `cb2/llm.py`: Claude Opus 5.5 (also in fast
mode, once Anthropic enables it for your account at https://claude.com/fast-mode),
Sonnet or Haiku 5.5, GPT-6 Astra, or local models through [Ollama](https://ollama.com) (gemma4:26b, qwen3:14b).
To use another local model, add one line there.

Choose per run, with effort levels for Claude and GPT (local models ignore effort):

- **LangGraph Studio:** the settings form on the assistant (`planner_model`,
  `planner_effort`, `reviewer_model`, `reviewer_effort`). Save
  combinations as assistants, e.g. a local-only team.
- **Terminal:** `python -m cb2.graph --planner-model gemma4:26b --reviewer-effort high --budget 0.50`

## Setup

Needs Python 3.10+ and, for the local models, Ollama.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your Anthropic and OpenAI API keys
```

Check that one model from each provider answers (or name models to try):

```bash
python -m cb2.llm
```

## Costs

Every model call records its tokens, how long it took (and so output tokens per
second) and what it cost (prices in `cb2/costs.py`).
At each approval pause the run shows this round's cost and the run total. There is
no limit on revision rounds; instead each run has a budget (default $2, or pass
`"budget"` in the input). When the next round would go over it, the run pauses and
asks for a higher budget or stops.
