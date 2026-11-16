"""A thin wrapper around the Groq chat API, so the rest of the code (and the tests) only see `complete()`."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

log = logging.getLogger(__name__)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLM(Protocol):
    def complete(
        self, system: str, user: str, *, model: str | None = None, temperature: float = 0.1, max_tokens: int = 900
    ) -> str: ...


class GroqLLM:
    def __init__(self, api_key: str, default_model: str):
        from groq import Groq

        self.client = Groq(api_key=api_key, max_retries=2, timeout=60)
        self.default_model = default_model

    def complete(
        self, system: str, user: str, *, model: str | None = None, temperature: float = 0.1, max_tokens: int = 900
    ) -> str:
        model = model or self.default_model
        kwargs = {}
        if "gpt-oss" in model:
            kwargs = {"reasoning_effort": "low", "include_reasoning": False}
        elif "qwen" in model:
            kwargs = {"reasoning_format": "hidden"}
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **kwargs,
        )
        return _THINK.sub("", resp.choices[0].message.content or "").strip()


def parse_json(text: str) -> dict | list | None:
    """Pull the first JSON object or array out of a model reply (models like to wrap JSON in prose)."""
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        log.warning("Model returned malformed JSON: %s", text[:200])
        return None
