# Interview Q&A

**Why FastAPI?** It provides typed request/response validation, OpenAPI documentation, async HTTP integration, and a lightweight Python deployment surface.

**Why LM Studio locally?** It exposes an OpenAI-compatible local API, allowing real inference on the interview machine without cloud credentials or paid API calls. It is a development runtime, not a production serving recommendation.

**How would you integrate vLLM on EKS?** Implement a provider against the OpenAI-compatible vLLM endpoint and select it through secure backend configuration. Keep browser access routed through FastAPI, with private service networking and workload identity.

**How does GPU scheduling work in Kubernetes?** The NVIDIA device plugin advertises GPU resources such as `nvidia.com/gpu`; a pod requests an integer resource and the scheduler places it on a node with capacity. Node selectors/affinity, taints, tolerations, and node groups further constrain placement.

**How do you scale inference?** Profile throughput and memory first, then scale replicas and GPU node capacity, use batching where supported, and apply queue/backpressure limits. Autoscaling should account for GPU utilization, queue depth, and latency rather than CPU alone.

**How do you monitor P95 latency?** Instrument request duration as a Prometheus histogram, aggregate histogram buckets with `histogram_quantile(0.95, sum by (le, model) (rate(llm_request_duration_seconds_bucket[5m])))`, and alert against a defined SLO. Avoid high-cardinality labels.

**How do you enforce model governance?** Use a configured model allowlist, explicit approval/version policies, metadata-only audit events, constrained environments and quotas, and human approval before production release.

**How does RAG work here?** The service chunks validated text, embeds and stores chunks locally, embeds a question, ranks chunks with cosine similarity, then provides retrieved evidence to a local chat model with grounding and untrusted-content instructions.

**How is AI-generated code validated?** Deterministic checks run first; model findings are labeled separately, reviewed by a human, then validated with policy-as-code and dedicated scanners. Suggestions are never applied automatically.

**Terraform versus GitOps?** Terraform provisions and manages infrastructure resources; GitOps reconciles application and configuration state from version-controlled desired state. They complement each other and should have clear ownership boundaries.

**What changes for production?** Add real identity and authorization, secrets management, private inference serving, artifact/model supply-chain controls, stronger policy-as-code, durable managed storage, resource isolation, load and failure testing, SLO-based alerts, retention controls, disaster recovery, and operational ownership.

**Which parts are real versus illustrative?** LM Studio calls, local embeddings/retrieval, deterministic checks, SQLite audit/deployment records, generated manifests, Prometheus metrics, and a fitted scikit-learn model operate locally. Fraud data is synthetic, deployment manifests are not applied, AWS Terraform is not applied, and development identity is not production authentication.
