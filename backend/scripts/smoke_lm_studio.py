#!/usr/bin/env python3
"""Exercise model discovery, chat, and embeddings against a running LM Studio server."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import settings
from app.providers.lm_studio import LMStudioProvider


async def run() -> int:
    provider = LMStudioProvider(
        settings.lm_studio_base_url,
        settings.request_timeout_seconds,
        api_key=settings.lm_studio_api_key,
    )
    print(f"LM Studio endpoint: {settings.lm_studio_base_url}")
    try:
        discovered = await provider.list_models()
    except Exception as exc:
        print(f"Model discovery failed: {exc}", file=sys.stderr)
        return 1
    print("Discovered models:")
    for model_id in discovered:
        print(f"  - {model_id}")

    chat_model = os.getenv("SMOKE_CHAT_MODEL", "qwen/qwen3.6-35b-a3b")
    embedding_model = os.getenv(
        "SMOKE_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"
    )
    missing = [model for model in (chat_model, embedding_model) if model not in discovered]
    if missing:
        print(
            "Smoke test not run: load the configured model(s) in LM Studio first: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 2

    try:
        result = await provider.chat(
            chat_model, "Reply with the single word ready.", temperature=0, max_tokens=256
        )
        vector = await provider.embed(embedding_model, "ModelOps Hub smoke test")
    except Exception as exc:
        print(f"Chat or embedding request failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Chat succeeded: model={result.model}, latency={result.latency_seconds:.3f}s, "
        f"output_tokens={result.output_tokens}"
    )
    print(f"Embedding succeeded: model={embedding_model}, dimensions={len(vector)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
