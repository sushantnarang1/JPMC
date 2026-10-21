from __future__ import annotations

import sqlite3

import yaml

from conftest import MODELS


def test_health_and_dynamic_model_discovery(client, model_state):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["provider"]["available"] is True

    response = client.get("/api/llm/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert [item["id"] for item in models[:3]] == MODELS
    assert all(item["discovered"] and not item["verified"] for item in models[:3])
    assert models[0]["model_type"] == "chat"
    assert models[2]["model_type"] == "embedding"

    chatted = client.post(
        "/api/llm/chat", json={"model": MODELS[0], "prompt": "confirm loaded model"}
    )
    assert chatted.status_code == 200
    verified_models = client.get("/api/llm/models").json()["models"]
    assert verified_models[0]["verified"] is True
    assert verified_models[1]["verified"] is False
    assert verified_models[2]["verified"] is False

    model_state["models"] = [{"id": MODELS[1]}]
    updated = client.get("/api/llm/models").json()["models"]
    assert [item["discovered"] for item in updated[:3]] == [False, True, False]


def test_chat_and_request_validation(client, database_path):
    result = client.post(
        "/api/llm/chat",
        json={"model": MODELS[0], "prompt": "hello", "temperature": 0.5, "max_tokens": 32},
    )
    assert result.status_code == 200
    assert result.json()["response"] == "Mock completion"
    assert result.json()["input_tokens"] == 12
    assert result.json()["tokens_per_second"] >= 0
    assert client.post(
        "/api/llm/chat", json={"model": MODELS[2], "prompt": "hello"}
    ).status_code == 422
    assert client.post(
        "/api/llm/chat", json={"model": MODELS[0], "prompt": "  "}
    ).status_code == 422
    assert client.post(
        "/api/llm/chat", json={"model": "unconfigured/arbitrary", "prompt": "hello"}
    ).status_code == 404
    with sqlite3.connect(database_path) as database:
        row = database.execute("SELECT action, details_json FROM audit_records").fetchone()
    assert row[0] == "chat"
    assert "hello" not in row[1]


def test_provider_unavailable_is_graceful(client, model_state):
    model_state["offline"] = True
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["provider"]["available"] is False
    unknown = client.get("/api/llm/models").json()["models"]
    assert all(item["availability_status"] == "unknown" for item in unknown[:3])
    assert client.post(
        "/api/llm/chat", json={"model": MODELS[0], "prompt": "hello"}
    ).status_code == 503


def test_inference_timeout_is_reported_and_audited(client, model_state):
    model_state["timeout_chat"] = True
    response = client.post(
        "/api/llm/chat", json={"model": MODELS[0], "prompt": "timeout probe"}
    )
    assert response.status_code == 502
    assert "ReadTimeout" in response.json()["detail"]
    assert client.get("/api/llm/models").json()["models"][0]["last_inference_status"] == "error"

    model_state["offline"] = False
    model_state["models"] = []
    # An empty successful model list is a live provider with no loaded models.
    assert client.get("/api/health").json()["provider"]["available"] is True
    assert client.post(
        "/api/llm/chat", json={"model": MODELS[0], "prompt": "hello"}
    ).status_code == 503


def test_benchmark_persists_metadata(client, database_path):
    response = client.post("/api/benchmark", json={"prompt": "benchmark prompt"})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
    assert all(item["status"] == "success" for item in response.json()["results"])
    with sqlite3.connect(database_path) as database:
        count = database.execute("SELECT COUNT(*) FROM benchmark_records").fetchone()[0]
        columns = [row[1] for row in database.execute("PRAGMA table_info(benchmark_records)")]
    assert count == 2
    assert "prompt" not in columns


def test_rag_chunks_embed_search_and_answer(client):
    added_alpha = client.post(
        "/api/rag/documents",
        json={"name": "alpha-doc", "content": "Alpha explains retrieval augmented generation."},
    )
    added_beta = client.post(
        "/api/rag/documents",
        json={"name": "beta-doc", "content": "Beta is an unrelated document."},
    )
    assert added_alpha.status_code == 200
    assert added_beta.status_code == 200
    assert added_alpha.json()["chunks"] == 1
    assert len(client.get("/api/rag/documents").json()["documents"]) == 2
    embeddings_verified = client.get("/api/llm/models").json()["models"]
    assert embeddings_verified[2]["verified"] is True
    assert embeddings_verified[2]["last_inference_status"] == "success"

    result = client.post(
        "/api/rag/query",
        json={"question": "alpha", "model": MODELS[0]},
    )
    assert result.status_code == 200
    assert result.json()["answer"] == "Mock completion"
    sources = result.json()["sources"]
    assert sources[0]["document"] == "alpha-doc"
    assert sources[0]["score"] > sources[1]["score"]


def test_rag_requires_local_embedding_model(client, model_state):
    model_state["models"] = [{"id": MODELS[0]}]
    added = client.post(
        "/api/rag/documents",
        json={"name": "local", "content": "alpha retrieval context"},
    )
    result = client.post(
        "/api/rag/query",
        json={"question": "alpha retrieval", "model": MODELS[0]},
    )
    assert added.status_code == 503
    assert result.status_code == 409


def test_seed_knowledge_base_indexes_synthetic_starter_documents(client):
    seeded = client.post("/api/rag/seed").json()
    assert len(seeded["seeded"]) == 5
    assert seeded["indexed_chunks"] == 5
    assert seeded["embedding_model"] == MODELS[2]
    assert client.post("/api/rag/seed").json()["seeded"] == []


def test_ai_review_runs_deterministic_then_coder(client, model_state):
    response = client.post(
        "/api/ai/review",
        json={
            "format": "kubernetes",
            "code": "containers:\n- image: demo:latest\n  securityContext:\n    privileged: true",
        },
    )
    assert response.status_code == 200
    findings = response.json()["findings"]
    assert [finding["source"] for finding in findings].count("deterministic") == 3
    assert [finding["source"] for finding in findings].count("ai") == 1
    assert response.json()["model"] == MODELS[1]
    assert model_state["chat_count"] == 1
    assert client.post(
        "/api/ai/review", json={"format": "yaml", "code": "x"}
    ).status_code == 422


def test_deterministic_review_parses_kubernetes_and_terraform(client):
    kubernetes = client.post(
        "/api/ai/review",
        json={
            "format": "kubernetes",
            "code": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: risky
spec:
  template:
    spec:
      hostNetwork: true
      containers:
        - name: app
          image: nginx:latest
          securityContext:
            privileged: true
            allowPrivilegeEscalation: true
""",
        },
    ).json()
    deterministic = [finding for finding in kubernetes["findings"] if finding["source"] == "deterministic"]
    titles = {finding["title"] for finding in deterministic}
    assert {"Host network enabled", "Privileged container", "Privilege escalation enabled", "Resource limits missing"} <= titles

    terraform = client.post(
        "/api/ai/review",
        json={
            "format": "terraform",
            "code": '''resource "aws_iam_policy" "wide" {
  policy = jsonencode({
    Statement = [{
      Effect = "Allow"
      Action = "*"
      Resource = "*"
    }]
  })
}''',
        },
    ).json()
    assert any(
        finding["title"] == "Broadly permissive IAM policy"
        for finding in terraform["findings"]
    )


def test_synthetic_fraud_inference(client):
    result = client.post(
        "/api/inference/fraud",
        json={
            "amount": 2500,
            "merchant_risk_score": 0.8,
            "transaction_hour": 2,
            "transaction_velocity": 12,
            "international": True,
        },
    )
    assert result.status_code == 200
    assert result.json()["prediction"] in {"fraud", "legitimate"}
    assert 0 <= result.json()["fraud_probability"] <= 1
    assert result.json()["model_version"] == "synthetic-fraud-logreg-v1"
    assert result.json()["latency_seconds"] >= 0
    assert "synthetic data" in result.json()["disclaimer"]
    offline = client.get("/api/inference/fraud/metrics").json()
    assert {"accuracy", "precision", "recall", "f1"} <= offline.keys()
    assert "synthetic data" in offline["disclaimer"]


def test_deployment_validation_manifest_and_metrics(client):
    valid = client.post(
        "/api/deployments/validate",
        json={"name": "demo", "image": "example/demo:1.2", "replicas": 2},
    )
    assert valid.status_code == 200
    assert valid.json()["valid"] is True
    invalid = client.post(
        "/api/deployments/validate",
        json={"name": "demo", "image": "example/demo:1.2", "replicas": 0},
    )
    assert invalid.status_code == 200
    assert invalid.json()["valid"] is False

    created = client.post(
        "/api/deployments",
        json={"name": "demo", "image": "example/demo:1.2", "replicas": 2, "port": 8080},
    )
    assert created.status_code == 200
    deployment_id = created.json()["id"]
    assert created.json()["status"] == "manifest_generated"
    assert client.get("/api/deployments").json()["deployments"][0]["id"] == deployment_id
    assert client.get(f"/api/deployments/{deployment_id}").status_code == 200
    manifest = client.get(f"/api/deployments/{deployment_id}/manifest")
    assert manifest.status_code == 200
    assert "kind: Deployment" in manifest.text
    assert "containerPort: 8080" in manifest.text
    assert len(list(yaml.safe_load_all(manifest.text))) == 2
    assert client.get("/api/deployments/unknown").status_code == 404

    summary = client.get("/api/metrics/summary").json()
    assert summary["benchmarks"]["count"] == 0
    assert summary["deployments"] == 1
    metrics = client.get("/metrics").text
    assert "llm_requests_total" in metrics
    assert "llm_request_duration_seconds" in metrics
    assert "deployment_validation_failures_total" in metrics


def test_deployment_policy_requires_approved_production_model_and_gpu_policy(client):
    production = client.post(
        "/api/deployments/validate",
        json={
            "name": "prod-model",
            "model": MODELS[0],
            "environment": "production",
            "production_approval": False,
        },
    )
    assert production.status_code == 200
    assert production.json()["valid"] is False
    assert any(
        finding["title"] == "Production approval required"
        for finding in production.json()["findings"]
    )

    gpu_in_development = client.post(
        "/api/deployments/validate",
        json={
            "name": "gpu-model",
            "environment": "development",
            "compute_profile": "gpu-inference",
            "gpu_count": 1,
        },
    )
    assert gpu_in_development.status_code == 200
    assert gpu_in_development.json()["valid"] is False


def test_gpu_deployment_manifest_includes_device_resource_and_scheduling(client):
    request = {
        "name": "gpu-model",
        "environment": "staging",
        "compute_profile": "gpu-inference",
        "gpu_count": 1,
    }
    assert client.post("/api/deployments/validate", json=request).json()["valid"] is True
    created = client.post("/api/deployments", json=request)
    assert created.status_code == 200
    manifest = client.get(f"/api/deployments/{created.json()['id']}/manifest").text
    assert "nvidia.com/gpu: 1" in manifest
    assert "nodeSelector:" in manifest
    assert "tolerations:" in manifest


def test_audit_events_are_metadata_only(client):
    client.post("/api/llm/chat", json={"model": MODELS[0], "prompt": "private prompt marker"})
    events = client.get("/api/audit").json()["events"]
    assert events[0]["action"] == "chat"
    assert "private prompt marker" not in str(events)


def test_cors_for_vite(client):
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
