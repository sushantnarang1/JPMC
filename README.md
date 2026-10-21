# ModelOps Hub

**Enterprise AI/ML Platform Engineering** — a local-first platform demo for model discovery, inference, RAG, infrastructure review, deployment policy, and observability.

## Run locally

Prerequisites: Docker Desktop with Compose, and LM Studio running its local server on the Mac host at port `1234`. No AWS credentials or external model API are needed.

```bash
cp .env.example .env
docker compose up --build
```

Open the UI at <http://localhost:5174>, the FastAPI OpenAPI page at <http://localhost:8000/docs>, and Prometheus metrics at <http://localhost:8000/metrics>. Compose binds only to loopback; frontend port `5174` avoids conflicting with common local Vite servers on `5173`. The default Compose URL is `http://host.docker.internal:1234/v1`. For direct host execution, set `LM_STUDIO_BASE_URL=http://localhost:1234/v1` and start the backend and frontend separately.

The backend tolerates LM Studio being unavailable: API health and platform features remain available, while inference-dependent actions show their actual errors. A listed model is not necessarily loaded or compatible. Start by refreshing Model Registry, then execute the smoke test after backend dependencies are installed:

```bash
cd backend
python -m pip install -r requirements.txt
python scripts/smoke_lm_studio.py
```

The smoke test tries chat and embeddings separately against the configured exact IDs; failures are reported rather than substituted. This project does not download models.

## Features

- Dynamic LM Studio model discovery, allowlisted chat and timed sequential benchmarking.
- Local embedding-backed text/Markdown RAG with NumPy cosine retrieval and source snippets.
- Deterministic Terraform/Kubernetes findings followed by optional local coder-model review.
- Policy-validated deployment requests and Kubernetes YAML previews; no cluster provisioning.
- Reproducible synthetic scikit-learn fraud classifier, with metrics explicitly labeled synthetic.
- Prometheus metrics, SQLite historical metadata, audit events, and AWS EKS Terraform examples.

See [ARCHITECTURE.md](ARCHITECTURE.md), [INTERVIEW_DEMO.md](INTERVIEW_DEMO.md), and [INTERVIEW_QA.md](INTERVIEW_QA.md) for design and interview guidance.

## Important boundaries

This is an interview/demo platform, not a production deployment. It has no production authentication, does not ingest PDFs, does not apply Terraform or Kubernetes manifests, and does not send prompts/documents to external services. Deployment records are requests/previews, not live deployments. Synthetic fraud scores are not bank fraud-detection performance. AI review is advisory and is not a substitute for policy-as-code, security scanning, or human approval. Retrieved documents are untrusted and prompt-injection defenses are not foolproof.

## Development checks

```bash
cd backend && python -m pytest
cd frontend && npm ci && npm run build
```

The real LM Studio smoke test is separate from mocked unit tests. Terraform files are examples only; do not run `terraform apply`.
