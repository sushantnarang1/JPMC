from __future__ import annotations

from typing import Protocol

import httpx

from ..providers.base import ChatResult
from ..providers.lm_studio import LMStudioProvider as _LMStudioProvider


class LLMProvider(Protocol):
    async def list_models(self) -> list[str]: ...

    async def health_check(self) -> bool: ...

    async def generate(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> ChatResult: ...

    async def embed(self, model: str, text: str) -> list[float]: ...


class LMStudioProvider(_LMStudioProvider):
    async def health_check(self) -> bool:
        try:
            await self.list_models()
            return True
        except (httpx.TimeoutException, httpx.HTTPError, ValueError):
            return False

    async def generate(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> ChatResult:
        return await self.chat(model, prompt, temperature, max_tokens)
