#!/usr/bin/env python3
"""Anthropic API inference engine for Dive Bar."""

import os
import threading
import time

import anthropic

from dive_bar.models import GenerationResult, LLMConfig


class APIEngine:
    """Anthropic API backend with same interface
    as InferenceEngine."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.gen_params = config.generation
        self._lock = threading.Lock()
        self._client = None

    def load_model(self):
        """Initialize the Anthropic client."""
        api_cfg = self.config.api
        api_key = (
            api_cfg.api_key
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        if not api_key:
            raise RuntimeError(
                "No API key: set llm.api.api_key in "
                "config.toml or ANTHROPIC_API_KEY env var"
            )
        kwargs = {"api_key": api_key}
        if api_cfg.base_url:
            kwargs["base_url"] = api_cfg.base_url
        self._client = anthropic.Anthropic(**kwargs)

    def generate(
        self,
        messages: list[dict],
        stop: list[str] | None = None,
        **overrides,
    ) -> GenerationResult:
        """Generate a response via Anthropic API.

        Thread-safe: acquires lock so only one agent
        generates at a time.
        """
        if self._client is None:
            raise RuntimeError("API client not loaded")
        params = self._merge_params(overrides)
        with self._lock:
            return self._do_generate(
                messages, params, stop
            )

    def _merge_params(
        self, overrides: dict
    ) -> dict:
        """Merge generation defaults with overrides."""
        return {
            "temperature": self.gen_params.temperature,
            "top_p": self.gen_params.top_p,
            "top_k": self.gen_params.top_k,
            "max_tokens": self.gen_params.max_tokens,
            **overrides,
        }

    def _build_api_kwargs(
        self,
        params: dict,
        chat_msgs: list[dict],
        system_text: str,
        stop: list[str] | None = None,
    ) -> dict:
        """Build kwargs dict for Anthropic API call."""
        kwargs = {
            "model": self.config.api.model,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "top_k": params["top_k"],
            "messages": chat_msgs,
        }
        if system_text:
            kwargs["system"] = system_text
        if stop:
            # Anthropic rejects whitespace-only stops
            cleaned = [
                s for s in stop if s.strip()
            ]
            if cleaned:
                kwargs["stop_sequences"] = cleaned
        return kwargs

    def _do_generate(
        self,
        messages: list[dict],
        params: dict,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Run inference via API (must hold lock)."""
        system_text, chat_msgs = self._split_system(
            messages
        )
        t0 = time.perf_counter()
        kwargs = self._build_api_kwargs(
            params, chat_msgs, system_text, stop
        )
        result = self._client.messages.create(**kwargs)
        elapsed_ms = (
            (time.perf_counter() - t0) * 1000
        )
        content = self._extract_content(result)
        return GenerationResult(
            content=content.strip(),
            tokens_prompt=result.usage.input_tokens,
            tokens_completion=(
                result.usage.output_tokens
            ),
            generation_time_ms=elapsed_ms,
        )

    def _split_system(
        self, messages: list[dict]
    ) -> tuple[str, list[dict]]:
        """Extract system prompt from messages.

        Anthropic API takes system as a separate
        parameter, not in the messages list.
        """
        if (
            messages
            and messages[0].get("role") == "system"
        ):
            return (
                messages[0]["content"],
                messages[1:],
            )
        return "", messages

    def _extract_content(self, result) -> str:
        """Pull text from Anthropic response."""
        for block in result.content:
            if block.type == "text":
                return block.text
        return ""

    def unload(self):
        """No-op for API mode."""
        self._client = None
