#!/usr/bin/env python3
"""LLM inference engine for Dive Bar."""

import threading
import time

from llama_cpp import Llama

from dive_bar.models import GenerationResult, LLMConfig


class InferenceEngine:
    """Thread-safe wrapper around llama-cpp-python."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.gen_params = config.generation
        self._lock = threading.Lock()
        self.llm = None

    def load_model(self):
        """Load the model into memory."""
        self.llm = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            chat_format=self.config.chat_format,
            seed=self.config.seed,
            verbose=False,
        )

    def generate(
        self,
        messages: list[dict],
        stop: list[str] | None = None,
        **overrides,
    ) -> GenerationResult:
        """Generate a response from the LLM.

        Thread-safe: acquires lock so only one agent
        generates at a time.
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded")
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
            "min_p": self.gen_params.min_p,
            "max_tokens": self.gen_params.max_tokens,
            "repeat_penalty": (
                self.gen_params.repeat_penalty
            ),
            **overrides,
        }

    def _do_generate(
        self,
        messages: list[dict],
        params: dict,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Run inference (must hold lock)."""
        t0 = time.perf_counter()
        kwargs = {
            "messages": messages,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "top_k": params["top_k"],
            "min_p": params["min_p"],
            "max_tokens": params["max_tokens"],
            "repeat_penalty": (
                params["repeat_penalty"]
            ),
        }
        if stop:
            kwargs["stop"] = stop
        result = self.llm.create_chat_completion(
            **kwargs
        )
        elapsed_ms = (
            (time.perf_counter() - t0) * 1000
        )
        content = (
            result["choices"][0]["message"]["content"]
        )
        usage = result.get("usage", {})
        return GenerationResult(
            content=content.strip(),
            tokens_prompt=usage.get(
                "prompt_tokens", 0
            ),
            tokens_completion=usage.get(
                "completion_tokens", 0
            ),
            generation_time_ms=elapsed_ms,
        )

    def unload(self):
        """Free the model from memory."""
        with self._lock:
            self.llm = None
