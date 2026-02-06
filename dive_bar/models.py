#!/usr/bin/env python3
"""Data models for Dive Bar."""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for a single bar agent."""

    name: str
    backstory: str
    personality_traits: list[str]
    chattiness: float
    responsiveness: float
    drink: str
    speaking_style: str
    model_override: str | None = None


@dataclass
class Message:
    """A single message in the bar conversation."""

    agent_name: str
    content: str
    turn_number: int
    timestamp: float


@dataclass
class GenerationResult:
    """Result from LLM inference."""

    content: str
    tokens_prompt: int
    tokens_completion: int
    generation_time_ms: float


@dataclass
class BarConfig:
    """Global bar configuration."""

    name: str
    max_agents: int
    tick_interval: float
    max_subject_chat: int = 3


@dataclass
class LLMGeneration:
    """LLM generation parameters."""

    temperature: float = 0.85
    top_p: float = 0.9
    top_k: int = 50
    min_p: float = 0.05
    max_tokens: int = 200
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class APIConfig:
    """API provider configuration."""

    provider: str = "anthropic"
    api_key: str = ""
    model: str = "claude-opus-4-6"
    base_url: str = ""


@dataclass
class LLMConfig:
    """LLM configuration."""

    mode: str = "local"
    model_path: str = ""
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    chat_format: str = "mistral-instruct"
    seed: int = 42
    generation: LLMGeneration = field(
        default_factory=LLMGeneration
    )
    api: APIConfig = field(
        default_factory=APIConfig
    )


@dataclass
class DisplayConfig:
    """Display settings."""

    response_speed: int = 30
    show_timestamps: bool = True
    color_scheme: str = "default"


@dataclass
class DatabaseConfig:
    """Database settings."""

    path: str = "data/dive_bar.duckdb"


@dataclass
class DiversityConfig:
    """Diversity checking configuration."""

    enabled: bool = True
    threshold: float = 0.6
    max_retries: int = 3
    window_size: int = 10
    ngram_min: int = 3
    ngram_max: int = 6
    refresh_interval: int = 20


@dataclass
class AppConfig:
    """Top-level application configuration."""

    bar: BarConfig
    llm: LLMConfig
    display: DisplayConfig
    database: DatabaseConfig
    agents: list[AgentConfig]
    diversity: DiversityConfig = field(
        default_factory=DiversityConfig
    )
