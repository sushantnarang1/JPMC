# Five-minute interview demonstration

Use `docker compose up --build`, with LM Studio running on the host and its local server enabled. The application stays usable if LM Studio is unavailable, but live inference, embeddings, and LLM-assisted review require the relevant models to be loaded and compatible.

1. **Model discovery (30 sec):** Open Model Registry. Refresh to query LM Studio and distinguish discovered models from inference-verified, platform-approved, and Kubernetes-deployed status.
2. **Real Qwen inference (45 sec):** Open LLM Playground, select a discovered chat model, submit a short prompt, and show the actual response, measured request latency, and token counts only when returned by the server.
3. **Benchmark (30 sec):** Submit the same prompt to the general and coder models sequentially. Explain that timings are local single-run measurements and not production guarantees.
4. **Infrastructure review (60 sec):** Review the intentionally insecure Kubernetes sample. Identify deterministic flags separately from model-generated explanations; highlight that the review does not auto-apply fixes or replace security scanners and approvals.
5. **Local RAG (60 sec):** Load the synthetic starter documents, query the knowledge base, and show retrieved source snippets and similarity scores alongside the generated answer. Explain that document text is untrusted.
6. **Deployment preview (45 sec):** Submit a non-production deployment request, inspect policy validation and generated Kubernetes manifest. State explicitly that no cluster was provisioned.
7. **Terraform and observability (30 sec):** Show the VPC/EKS example and metrics/audit views. Point out that Terraform has not been applied and the fraud classifier is trained only on synthetic data.

## Live inference fallback

If a model is listed but not loaded or does not support a tested endpoint, demonstrate the explicit unavailable/error state rather than substituting a response. The backend smoke test is `python scripts/smoke_lm_studio.py` from the `backend/` directory.
