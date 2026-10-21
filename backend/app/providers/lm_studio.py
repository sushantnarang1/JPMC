from __future__ import annotations

import time
from typing import Any

import httpx

from .base import ChatResult, InferenceProvider


class LMStudioProvider(InferenceProvider):
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.api_key = api_key

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = kwargs.pop("headers", {})
        if self.api_key:
            headers = {**headers, "Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds, transport=self.transport
        ) as client:
            response = await client.request(
                method, f"{self.base_url}{path}", headers=headers, **kwargs
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("LM Studio returned an invalid response")
            return payload

    async def list_models(self) -> list[str]:
        payload = await self._request("GET", "/models")
        entries = payload.get("data", [])
        if not isinstance(entries, list):
            raise ValueError("LM Studio returned an invalid model list")
        return [
            str(entry["id"])
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        ]

    async def chat(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> ChatResult:
        started = time.perf_counter()
        request_body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model == "qwen/qwen3.6-35b-a3b":
            request_body["reasoning_effort"] = "none"
        payload = await self._request(
            "POST",
            "/chat/completions",
            json=request_body,
        )
        elapsed = time.perf_counter() - started
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LM Studio returned no chat choices")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            # Some reasoning models return reasoning_content with empty content
            # when the generation budget is too small. Reasoning is never the
            # user-visible assistant response and must not be used as fallback.
            raise ValueError("LM Studio returned no user-visible chat content")
        usage = payload.get("usage", {})
        usage = usage if isinstance(usage, dict) else {}
        return ChatResult(
            model=model,
            response=content,
            latency_seconds=elapsed,
            input_tokens=_nonnegative_int(usage.get("prompt_tokens")),
            output_tokens=_nonnegative_int(usage.get("completion_tokens")),
        )

    async def embed(self, model: str, text: str) -> list[float]:
        payload = await self._request(
            "POST",
            "/embeddings",
            json={"model": model, "input": text},
        )
        entries = payload.get("data")
        if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
            raise ValueError("LM Studio returned no embedding")
        vector = entries[0].get("embedding")
        if (
            not isinstance(vector, list)
            or not vector
            or any(not isinstance(value, (int, float)) for value in vector)
        ):
            raise ValueError("LM Studio returned an invalid embedding")
        return [float(value) for value in vector]


def _nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        converted = int(value)
        return converted if converted >= 0 else None
    except (TypeError, ValueError):
        return None
