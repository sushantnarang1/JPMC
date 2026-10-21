from __future__ import annotations

import json

import httpx
import pytest

from app.providers.lm_studio import LMStudioProvider
from app.main import _ai_findings


@pytest.mark.asyncio
async def test_lm_studio_provider_uses_discovery_chat_and_embeddings():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen/qwen3.6-35b-a3b"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                },
            )
        return httpx.Response(200, json={"data": [{"embedding": [0.25, 0.5]}]})

    provider = LMStudioProvider(
        "http://lm-studio.test/v1",
        transport=httpx.MockTransport(handler),
    )
    assert await provider.list_models() == ["qwen/qwen3.6-35b-a3b"]
    result = await provider.chat("qwen/qwen3.6-35b-a3b", "hello", 0.1, 20)
    assert result.response == "hello"
    assert result.input_tokens == 4
    assert result.output_tokens == 2
    assert result.latency_seconds >= 0
    assert await provider.embed("nomic", "hello") == [0.25, 0.5]
    assert [request.url.path for request in requests] == [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/embeddings",
    ]
    chat_request = json.loads(requests[1].content)
    assert chat_request["messages"][0]["content"] == "hello"
    assert chat_request["temperature"] == 0.1
    assert chat_request["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_chat_uses_content_and_never_exposes_reasoning_content():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "User-facing answer",
                            "reasoning_content": "private internal reasoning",
                        }
                    }
                ]
            },
        )

    provider = LMStudioProvider(
        "http://lm-studio.test/v1",
        transport=httpx.MockTransport(handler),
    )
    result = await provider.chat("qwen/qwen3.6-35b-a3b", "question", 0, 256)
    assert result.response == "User-facing answer"
    assert "private internal reasoning" not in result.response


@pytest.mark.asyncio
async def test_empty_content_does_not_fall_back_to_reasoning_content():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "reasoning only, not a response",
                        }
                    }
                ]
            },
        )

    provider = LMStudioProvider(
        "http://lm-studio.test/v1",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError, match="user-visible chat content"):
        await provider.chat("qwen/qwen3.6-35b-a3b", "question", 0, 64)


@pytest.mark.asyncio
async def test_missing_usage_is_not_reported_as_zero_tokens():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}}]},
        )

    provider = LMStudioProvider(
        "http://lm-studio.test/v1",
        transport=httpx.MockTransport(handler),
    )
    result = await provider.chat("qwen/qwen3.6-35b-a3b", "question", 0, 256)
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.tokens_per_second is None


def test_ai_review_accepts_json_wrapped_in_markdown_but_rejects_unstructured_text():
    findings = _ai_findings(
        '```json\n{"findings":[{"severity":"HIGH","title":"Risk","explanation":"Why","recommendation":"Fix"}]}\n```'
    )
    assert findings == [
        {
            "source": "ai",
            "severity": "high",
            "title": "Risk",
            "explanation": "Why",
            "recommendation": "Fix",
        }
    ]
    assert _ai_findings("No structured findings") is None
    assert _ai_findings('{"findings":[]}') == []
