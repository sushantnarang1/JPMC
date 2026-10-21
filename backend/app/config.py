from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "config" / "model_registry.json"
MODEL_REGISTRY = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))["models"]
DEFAULT_MODELS = tuple(model["id"] for model in MODEL_REGISTRY)


@dataclass(frozen=True)
class Settings:
    lm_studio_base_url: str
    configured_model_ids: tuple[str, ...]
    request_timeout_seconds: float
    database_path: Path
    cors_origins: tuple[str, ...]
    max_output_tokens: int = 512
    lm_studio_api_key: str | None = None
    production_approved_model_ids: tuple[str, ...] = ()
    gpu_allowed_environments: tuple[str, ...] = ("staging", "production")

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("LLM_PROVIDER", "lmstudio").strip().lower()
        if provider != "lmstudio":
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
        configured = os.getenv("MODEL_OPS_MODEL_IDS")
        model_ids = tuple(
            item.strip()
            for item in configured.split(",")
            if item.strip()
        ) if configured else DEFAULT_MODELS
        origins = os.getenv(
            "MODEL_OPS_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
        timeout_seconds = float(
            os.getenv("LLM_REQUEST_TIMEOUT", os.getenv("MODEL_OPS_TIMEOUT_SECONDS", "120"))
        )
        max_output_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "512"))
        if not 0 < timeout_seconds <= 600:
            raise ValueError("LLM_REQUEST_TIMEOUT must be between 0 and 600 seconds")
        if not 1 <= max_output_tokens <= 8192:
            raise ValueError("LLM_MAX_OUTPUT_TOKENS must be between 1 and 8192")
        return cls(
            lm_studio_base_url=os.getenv(
                "LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"
            ).rstrip("/"),
            configured_model_ids=model_ids,
            request_timeout_seconds=timeout_seconds,
            database_path=Path(
                os.getenv(
                    "MODEL_OPS_DATABASE_PATH",
                    str(Path(__file__).resolve().parents[1] / "modelops.db"),
                )
            ),
            cors_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
            max_output_tokens=max_output_tokens,
            lm_studio_api_key=os.getenv("LM_STUDIO_API_KEY") or None,
            production_approved_model_ids=tuple(
                item.strip()
                for item in os.getenv("MODEL_OPS_PRODUCTION_APPROVED_MODELS", "").split(",")
                if item.strip()
            ),
            gpu_allowed_environments=tuple(
                item.strip()
                for item in os.getenv(
                    "MODEL_OPS_GPU_ALLOWED_ENVIRONMENTS", "staging,production"
                ).split(",")
                if item.strip()
            ),
        )


settings = Settings.from_env()
