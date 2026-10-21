# ModelOps Hub architecture

ModelOps Hub is a local-first self-service platform demo. The React application only talks to FastAPI; model prompts and uploaded documents never go directly from the browser to LM Studio or an external service.

```text
Browser (Vite / React)
       │ HTTP, same-origin development CORS
       ▼
FastAPI gateway ───── SQLite (audit metadata, requests, deployment requests)
       │              Local files (synthetic classifier, embedded chunk index)
       ├── LM Studio OpenAI-compatible API
       │     ├── qwen/qwen3.6-35b-a3b (general chat / RAG)
       │     ├── qwen/qwen2.5-coder-14b (infrastructure review)
       │     └── text-embedding-nomic-embed-text-v1.5 (RAG embeddings)
       └── scikit-learn synthetic fraud classifier

Prometheus scrape: FastAPI /metrics
Infrastructure examples: Terraform AWS VPC + EKS (not applied)
```

## Inference and model lifecycle

The backend discovers model IDs from LM Studio at runtime. Explicit local configuration maps known IDs to intended capabilities; discovery is not evidence that a model is loaded, inference-compatible, approved, or deployed. Chat and embedding calls verify those capabilities when requested. Requests have allowlisted IDs, bounded input/output, request timeouts, timing measurements, and metadata-only audit events. LM Studio connectivity failures are reported to clients without taking down the API.

The browser cannot select an inference URL or call LM Studio. To run the backend directly on macOS, use `http://localhost:1234/v1`; in Docker Desktop, use `http://host.docker.internal:1234/v1`. The backend container uses the latter by default.

## Retrieval-augmented generation

UTF-8 Markdown/text is validated and chunked with overlap, then embedded by the local embedding model. Vectors and chunk metadata are stored locally. Query embeddings are compared with NumPy cosine similarity, and the top matches are passed as explicitly untrusted evidence to the selected approved local chat model. Retrieval is not a security boundary; prompt injection resistance is not guaranteed.

## Infrastructure review

Supported Terraform and Kubernetes text is checked deterministically before it is sent to the local coding model for explanation. Findings identify their source. The LLM may miss or misstate risks. It does not mutate files or replace policy-as-code, dedicated scanners, human review, or approval.

## Deployment lifecycle

The API validates resource quotas, model and environment policy, and production approval requirements, and creates a Kubernetes YAML preview. A stored request represents a request/manifest only; no Kubernetes API or cloud provisioning is invoked. The deployment status must not be interpreted as a live cluster deployment.

## Future EKS / vLLM path

The provider interface is the seam for replacing LM Studio with a vLLM OpenAI-compatible endpoint on EKS. Deploy vLLM behind an internal service, use GPU node groups with the NVIDIA device plugin, request `nvidia.com/gpu`, and configure workload identity, private networking, autoscaling, model artifact controls, and service authentication. Keep model policy and API contracts in FastAPI while routing to a configured provider. Production adoption also requires secrets management, identity/RBAC, network policies, artifact provenance, policy-as-code, load testing, SLOs, backup/retention, and incident response. The Terraform examples are illustrative architecture and have not been applied.
