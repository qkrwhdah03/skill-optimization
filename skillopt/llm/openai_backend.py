"""OpenAI-compatible chat completions backend (OpenAI API, or a local vLLM server). stdlib only."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, ClassVar

from .base import BaseLLM, RawGeneration
from .registry import register_llm
from .thinking import split_thinking

_PAYLOAD_KEYS = ("max_tokens", "temperature", "top_p", "seed", "top_k")


@register_llm("openai")
class OpenAICompatLLM(BaseLLM):
    """Engine keys: ``base_url``, ``api_key_env`` (env var NAME, value never stored), ``timeout``."""

    uses_gpu: ClassVar[bool] = False

    def __init__(self, config: dict, role: str = "default") -> None:
        super().__init__(config, role=role)
        engine = dict(config.get("engine", {}) or {})
        self.base_url: str = str(engine.get("base_url", "http://localhost:8000/v1")).rstrip("/")
        self.api_key_env: str | None = engine.get("api_key_env")
        self.timeout: float = float(engine.get("timeout", 600))

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if key:
                headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _generate(self, prompt: str, gen: dict) -> RawGeneration:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        for k in _PAYLOAD_KEYS:
            if k in gen:
                payload[k] = gen[k]
        if "enable_thinking" in gen:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(gen["enable_thinking"])}
        data = self._post(payload)

        message = ((data.get("choices") or [{}])[0] or {}).get("message", {}) or {}
        content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None:
            reasoning, visible = str(reasoning_content).strip(), content.strip()
        else:
            reasoning, visible = split_thinking(content)

        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return RawGeneration(
            text=visible,
            reasoning_text=reasoning,
            prompt_text=prompt,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            raw=data,
        )
