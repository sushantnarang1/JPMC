# ModelOps Hub backend

The API runs locally against LM Studio's OpenAI-compatible API. Configured model
IDs are not treated as loaded: `GET /api/llm/models` and inference operations
check the live `/v1/models` response before using a model.
Model discovery sets `discovered` and availability only. The API sets
`verified` after a successful chat completion with non-empty `message.content`
or embedding inference; any separate `reasoning_content` is never returned as
the assistant answer.

## Run

```sh
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

LM Studio defaults to `http://127.0.0.1:1234/v1`. Copy `.env.example` to `.env`, adjust the values,
and start with `uvicorn app.main:app --reload --env-file .env`. The SQLite
database defaults to `backend/modelops.db`.
The API remains healthy when LM Studio is offline; model availability is
reported as unknown until the provider is reachable.

Infrastructure deployment requests only validate inputs and generate/store
Kubernetes manifests. They do not invoke `kubectl` or deploy workloads.
Fraud scoring uses a reproducible synthetic scikit-learn training dataset and
is for demonstration only.

## Test and smoke test

```sh
cd backend
python -m pytest
python scripts/smoke_lm_studio.py
```

The pytest suite uses mocked LM Studio HTTP responses. The smoke test makes real
requests to the configured LM Studio server and only runs chat/embedding checks
when both respective model IDs are discovered as loaded.
