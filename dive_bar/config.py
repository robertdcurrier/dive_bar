#!/usr/bin/env python3
"""Configuration loading and validation for Dive Bar."""

import os
from pathlib import Path

import tomllib

from dive_bar.models import (
    AgentConfig,
    AppConfig,
    BarConfig,
    DatabaseConfig,
    DisplayConfig,
    DiversityConfig,
    LLMConfig,
    LLMGeneration,
)


def _find_project_root() -> Path:
    """Find the project root directory."""
    current = Path(os.getcwd())
    if (current / "config.toml").exists():
        return current
    script_dir = Path(__file__).resolve().parent.parent
    if (script_dir / "config.toml").exists():
        return script_dir
    return current


def _load_bar_config(data: dict) -> BarConfig:
    """Parse [bar] section from config."""
    bar = data.get("bar", {})
    return BarConfig(
        name=bar.get("name", "The Rusty Nail"),
        max_agents=bar.get("max_agents", 5),
        tick_interval=bar.get("tick_interval", 2.0),
        max_subject_chat=bar.get(
            "max_subject_chat", 3
        ),
    )


def _load_llm_config(data: dict) -> LLMConfig:
    """Parse [llm] section from config."""
    llm = data.get("llm", {})
    gen_data = llm.get("generation", {})
    generation = LLMGeneration(
        temperature=gen_data.get("temperature", 0.85),
        top_p=gen_data.get("top_p", 0.9),
        top_k=gen_data.get("top_k", 50),
        min_p=gen_data.get("min_p", 0.05),
        max_tokens=gen_data.get("max_tokens", 200),
        repeat_penalty=gen_data.get(
            "repeat_penalty", 1.1
        ),
        frequency_penalty=gen_data.get(
            "frequency_penalty", 0.0
        ),
        presence_penalty=gen_data.get(
            "presence_penalty", 0.0
        ),
    )
    return LLMConfig(
        model_path=llm.get("model_path", ""),
        n_ctx=llm.get("n_ctx", 4096),
        n_gpu_layers=llm.get("n_gpu_layers", -1),
        chat_format=llm.get(
            "chat_format", "mistral-instruct"
        ),
        seed=llm.get("seed", 42),
        generation=generation,
    )


def _load_display_config(data: dict) -> DisplayConfig:
    """Parse [display] section from config."""
    disp = data.get("display", {})
    return DisplayConfig(
        response_speed=disp.get("response_speed", 30),
        show_timestamps=disp.get(
            "show_timestamps", True
        ),
        color_scheme=disp.get(
            "color_scheme", "default"
        ),
    )


def _load_database_config(data: dict) -> DatabaseConfig:
    """Parse [database] section from config."""
    db = data.get("database", {})
    return DatabaseConfig(
        path=db.get("path", "data/dive_bar.duckdb"),
    )


def _load_diversity_config(data: dict) -> DiversityConfig:
    """Parse [diversity] section from config."""
    div = data.get("diversity", {})
    return DiversityConfig(
        enabled=div.get("enabled", True),
        threshold=div.get("threshold", 0.6),
        max_retries=div.get("max_retries", 3),
        window_size=div.get("window_size", 10),
        ngram_min=div.get("ngram_min", 3),
        ngram_max=div.get("ngram_max", 6),
        refresh_interval=div.get("refresh_interval", 20),
    )


def _load_agents(data: dict) -> list[AgentConfig]:
    """Parse [[agent]] entries from agents.toml."""
    agents = []
    for entry in data.get("agent", []):
        agents.append(AgentConfig(
            name=entry["name"],
            backstory=entry.get("backstory", "").strip(),
            personality_traits=entry.get(
                "personality_traits", []
            ),
            chattiness=entry.get("chattiness", 0.5),
            responsiveness=entry.get(
                "responsiveness", 0.5
            ),
            drink=entry.get("drink", "Beer"),
            speaking_style=entry.get(
                "speaking_style", ""
            ),
            model_override=entry.get("model_override"),
        ))
    return agents


def load_config(
    config_path: str | None = None,
    agents_path: str | None = None,
) -> AppConfig:
    """Load and validate all configuration."""
    root = _find_project_root()
    if config_path is None:
        config_path = str(root / "config.toml")
    if agents_path is None:
        agents_path = str(root / "agents.toml")

    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)
    with open(agents_path, "rb") as f:
        agents_data = tomllib.load(f)

    return AppConfig(
        bar=_load_bar_config(config_data),
        llm=_load_llm_config(config_data),
        display=_load_display_config(config_data),
        database=_load_database_config(config_data),
        agents=_load_agents(agents_data),
        diversity=_load_diversity_config(config_data),
    )
