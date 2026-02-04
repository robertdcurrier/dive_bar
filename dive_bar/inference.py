#!/usr/bin/env python3
"""LLM inference engine for Dive Bar."""

import threading
import time

from llama_cpp import Llama

from dive_bar.models import GenerationResult, LLMConfig


_SUPPRESSED_VARIANTS = [
    " tattoo", " tattoos", " tattooed",
    " tattooing", " Tattoo", " Tattoos",
    " TATTOO",
    " Girl", " Honey",
]
SUPPRESS_BIAS = -100.0


class InferenceEngine:
    """Thread-safe wrapper around llama-cpp-python."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.gen_params = config.generation
        self._lock = threading.Lock()
        self.llm = None
        self._logit_bias: dict[int, float] = {}

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
        self._build_logit_bias()

    def _build_logit_bias(self):
        """Build logit bias from suppressed words.

        Only suppresses single-token forms and unique
        subword tokens. Skips common subtokens like
        't', 'o', 'ed' that would break generation.
        """
        bias: dict[int, float] = {}
        common = self._find_common_tokens()
        for word in _SUPPRESSED_VARIANTS:
            tokens = self.llm.tokenize(
                word.encode("utf-8"),
                add_bos=False,
            )
            if len(tokens) == 1:
                bias[tokens[0]] = SUPPRESS_BIAS
            else:
                for tid in tokens:
                    if tid not in common:
                        bias[tid] = SUPPRESS_BIAS
        self._logit_bias = bias

    def _find_common_tokens(self) -> set[int]:
        """Find tokens too common to suppress."""
        probe = [
            " the", " and", " to", " of",
            " it", " that", " ed", " ing",
        ]
        common: set[int] = set()
        for word in probe:
            tokens = self.llm.tokenize(
                word.encode("utf-8"),
                add_bos=False,
            )
            common.update(tokens)
        return common

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
            "frequency_penalty": (
                self.gen_params.frequency_penalty
            ),
            "presence_penalty": (
                self.gen_params.presence_penalty
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
            "frequency_penalty": (
                params["frequency_penalty"]
            ),
            "presence_penalty": (
                params["presence_penalty"]
            ),
        }
        if stop:
            kwargs["stop"] = stop
        if self._logit_bias:
            kwargs["logit_bias"] = self._logit_bias
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
