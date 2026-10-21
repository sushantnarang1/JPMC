from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ChatResult:
    model: str
    response: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None

    @property
    def tokens_per_second(self) -> float | None:
        if self.output_tokens is None or self.latency_seconds <= 0:
            return None
        return self.output_tokens / self.latency_seconds


class InferenceProvider(ABC):
    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return model identifiers currently advertised by the provider."""

    @abstractmethod
    async def chat(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> ChatResult:
        """Generate a chat completion."""

    @abstractmethod
    async def embed(self, model: str, text: str) -> list[float]:
        """Return a vector embedding."""

    async def health_check(self) -> bool:
        try:
            await self.list_models()
            return True
        except (httpx.TimeoutException, httpx.HTTPError, ValueError):
            return False

    async def generate(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> ChatResult:
        """Generate a response through the provider's chat-compatible endpoint."""
        return await self.chat(model, prompt, temperature, max_tokens)
