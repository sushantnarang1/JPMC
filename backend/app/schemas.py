from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
import re


class ChatRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=20_000)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=8192)

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class BenchmarkRequest(BaseModel):
    prompt: str = Field(default="Explain the role of model monitoring.", min_length=1, max_length=20_000)

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class DocumentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=100_000)

    @field_validator("name", "content")
    @classmethod
    def document_fields_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be blank")
        return value

    @field_validator("name")
    @classmethod
    def reject_unsupported_document_type(cls, value: str) -> str:
        if value.lower().endswith(".pdf"):
            raise ValueError("PDF ingestion is not supported; provide UTF-8 text or Markdown")
        return value


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)
    model: str = Field(min_length=1, max_length=200)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class ReviewRequest(BaseModel):
    code: str = Field(min_length=1, max_length=100_000)
    format: str

    @field_validator("code")
    @classmethod
    def code_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("code must not be blank")
        return value

    @field_validator("format")
    @classmethod
    def supported_format(cls, value: str) -> str:
        if value not in {"terraform", "kubernetes"}:
            raise ValueError("format must be terraform or kubernetes")
        return value


class DeploymentRequest(BaseModel):
    name: str = Field(default="modelops-inference", min_length=1, max_length=63)
    image: str = Field(default="modelops/inference:1.0.0", min_length=1, max_length=512)
    replicas: int = Field(default=1, ge=0, le=100)
    port: int = Field(default=8000, ge=1, le=65535)
    namespace: str = Field(default="default", min_length=1, max_length=63)
    model: str = Field(default="qwen/qwen3.6-35b-a3b", min_length=1, max_length=200)
    environment: str = Field(default="development")
    compute_profile: str = Field(default="cpu-standard")
    cpu_request: str = Field(default="500m")
    cpu_limit: str = Field(default="1")
    memory_request: str = Field(default="512Mi")
    memory_limit: str = Field(default="1Gi")
    gpu_count: int = Field(default=0, ge=0, le=4)
    production_approval: bool = False

    @field_validator("name", "namespace")
    @classmethod
    def dns_label(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value):
            raise ValueError("must be a valid lowercase Kubernetes DNS label")
        return value

    @field_validator("environment")
    @classmethod
    def valid_environment(cls, value: str) -> str:
        if value not in {"development", "staging", "production"}:
            raise ValueError("environment must be development, staging, or production")
        return value

    @field_validator("compute_profile")
    @classmethod
    def valid_compute_profile(cls, value: str) -> str:
        if value not in {"cpu-standard", "gpu-inference"}:
            raise ValueError("compute_profile must be cpu-standard or gpu-inference")
        return value

    @field_validator("cpu_request", "cpu_limit")
    @classmethod
    def valid_cpu_quantity(cls, value: str) -> str:
        if not re.fullmatch(r"(?:[1-9]\d*m|(?:0?\.)?[1-9]\d*(?:\.\d+)?)", value):
            raise ValueError("CPU must be a positive Kubernetes quantity (for example 500m or 2)")
        return value

    @field_validator("memory_request", "memory_limit")
    @classmethod
    def valid_memory_quantity(cls, value: str) -> str:
        if not re.fullmatch(r"[1-9]\d*(?:Ki|Mi|Gi|Ti)", value):
            raise ValueError("Memory must be a positive quantity using Ki, Mi, Gi, or Ti")
        return value


class FraudRequest(BaseModel):
    amount: float = Field(ge=0, le=10_000_000)
    merchant_risk_score: float = Field(ge=0, le=1)
    transaction_hour: int = Field(ge=0, le=23)
    transaction_velocity: int = Field(ge=0, le=100_000)
    international: bool = False
