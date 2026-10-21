from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.lm_studio import LMStudioProvider

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATABASE = BACKEND_ROOT / "tests" / ".test-modelops.sqlite"
MODELS = [
    "qwen/qwen3.6-35b-a3b",
    "qwen/qwen2.5-coder-14b",
    "text-embedding-nomic-embed-text-v1.5",
]


@pytest.fixture
def model_state():
    return {
        "models": [{"id": model} for model in MODELS],
        "chat_count": 0,
        "offline": False,
        "timeout_chat": False,
    }


def mock_handler(state):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            if state.get("offline"):
                return httpx.Response(503, json={"error": "LM Studio unavailable"})
            return httpx.Response(200, json={"data": state["models"]})
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            if state.get("timeout_chat"):
                raise httpx.ReadTimeout("mock model request timed out")
            body = json.loads(request.content)
            state["chat_count"] += 1
            if "Review this" in body["messages"][0]["content"]:
                content = json.dumps(
                    {
                        "findings": [
                            {
                                "severity": "low",
                                "title": "AI note",
                                "explanation": "Consider a rollout health check.",
                                "recommendation": "Configure a readiness probe.",
                            }
                        ]
                    }
                )
            else:
                content = "Mock completion"
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                },
            )
        if request.method == "POST" and request.url.path == "/v1/embeddings":
            text = json.loads(request.content)["input"].lower()
            vector = [1.0, 0.0, 0.0] if "alpha" in text else [0.0, 1.0, 0.0]
            return httpx.Response(200, json={"data": [{"embedding": vector}]})
        return httpx.Response(404, json={"error": "not found"})

    return handle


@pytest.fixture
def client(model_state):
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    if DATABASE.exists():
        DATABASE.unlink()
    model_artifact = DATABASE.with_suffix(".fraud.joblib")
    if model_artifact.exists():
        model_artifact.unlink()
    transport = httpx.MockTransport(mock_handler(model_state))
    provider = LMStudioProvider("http://lm-studio.test/v1", transport=transport)
    test_settings = Settings(
        lm_studio_base_url="http://lm-studio.test/v1",
        configured_model_ids=tuple(MODELS),
        request_timeout_seconds=1,
        database_path=DATABASE,
        cors_origins=("http://localhost:5173",),
    )
    application = create_app(test_settings, provider=provider)
    with TestClient(application) as test_client:
        yield test_client
    if DATABASE.exists():
        DATABASE.unlink()


@pytest.fixture
def database_path():
    return DATABASE
