# CLAUDE.md

## Project Overview

Dive Bar is a terminal UI app that simulates AI characters
having unscripted conversations at a dive bar. Supports
both local LLM inference (llama-cpp-python) and Anthropic
API mode with per-agent engine instances. Textual TUI
frontend with DuckDB conversation logging.

## Architecture

```
main.py              Entry point
config.toml          Bar, LLM, display settings
agents.toml          Character definitions

dive_bar/
  app.py             Textual app, tick loop, worker threads
  agent.py           Prompt construction, topic rotation
  bartender.py       Speaker selection + named addressing
  api_engine.py      Anthropic API inference engine
  inference.py       Thread-safe llama-cpp-python wrapper
  config.py          TOML parsing into dataclasses
  models.py          All dataclasses (no logic)
  db.py              DuckDB schema + logging
  widgets/
    chat_panel.py    Scrolling Rich log with agent colors
    agent_sidebar.py Agent status indicators + stats
    controls.py      Pause, speed, stranger message input
```

## Key Patterns

- **Per-agent engines (API mode)**: Each agent gets its
  own APIEngine instance with a separate Anthropic client.
  A house engine handles non-agent calls (opener, topic).
  `_get_engine(name)` routes all generate calls.
- **Single engine (local mode)**: One InferenceEngine
  instance serves all agents. A threading lock serializes
  generation calls.
- **Bartender orchestrator**: Pure Python scoring algorithm
  picks the next speaker. No LLM calls. Weights: time
  since last spoke (0.35), chattiness (0.25), addressed
  by name (0.30), randomness (0.10).
- **Deterministic named addressing**: If the last message
  mentions an agent by name, that agent speaks next
  (skips scoring). A pair-streak limiter
  (`MAX_PAIR_STREAK=2`) suppresses ping-pong lock-in
  between the same dyad by falling back to scoring.
- **Topic rotation**: A counter (`_subject_count`) triggers
  a topic change after `max_subject_chat` consecutive
  turns. Uses a meta-prompt ("what you'd overhear at a
  dive bar") with a rolling avoidance list of the last
  5 topics to prevent subject fixation.
- **Script-style prompting**: Conversation history is packed
  into a single user message as a script. The system prompt
  defines the character. The user message ends with "reply
  as {name}".

## Code Standards

- PEP-8 compliant
- Max 79 characters per line
- Max 35 lines per function (excluding docstrings)
- All Python files have `#!/usr/bin/env python3` shebang
- Imports at the top of every file
- Minimal dependencies: textual, llama-cpp-python,
  duckdb, anthropic, tomllib (stdlib)

## Configuration

Two TOML files at project root:

- `config.toml`: Bar settings, LLM model/params, display,
  database path
- `agents.toml`: Character definitions with backstory,
  personality traits, chattiness, speaking style

Key tuning knobs in `config.toml`:
- `bar.max_subject_chat`: Turns before forced topic change
- `bar.tick_interval`: Seconds between speaker selections
- `llm.generation.temperature`: Higher = more creative
- `llm.generation.repeat_penalty`: Fights echo/repetition
- `llm.generation.max_tokens`: Cap on response length

## Running

```bash
python main.py
```

Keybindings: `p` pause, `q` quit, `+`/`-` speed.
Type in the bottom input to speak as "A stranger".

## Model Requirements

Uses GGUF models via llama-cpp-python. The model file
goes in `models/` and is referenced by `config.toml`.
Tested with:
- Mistral 7B Instruct v0.3 (Q5_K_M) — fast, but
  sycophantic and echo-prone
- Llama 3 70B Instruct (Q4_K_M) — recommended, follows
  personality prompts well, needs ~40GB unified memory

## Database

DuckDB at `data/dive_bar.duckdb`. Four tables:
- `sessions`: Start/end time, config hash
- `messages`: Full conversation log with generation stats
- `agent_states`: Context snapshots (future use)
- `regenerations`: Diversity-triggered regen events

## Common Tasks

- **Add a new character**: Add `[[agent]]` block to
  `agents.toml` with name, backstory, traits, chattiness,
  responsiveness, drink, speaking_style.
- **Swap models (local)**: Change `llm.model_path` and
  `llm.chat_format` in `config.toml`.
- **Use API mode**: Set `llm.mode = "api"` and configure
  `llm.api.model` and API key in `config.toml` or
  `ANTHROPIC_API_KEY` env var.
- **Tune conversation pace**: Adjust `bar.tick_interval`
  and agent `chattiness` values.
- **Reduce echoing**: Increase `repeat_penalty`, lower
  `max_subject_chat`, or use a larger model.
