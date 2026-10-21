from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    model TEXT,
                    status TEXT NOT NULL,
                    latency_seconds REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS benchmark_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_seconds REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    tokens_per_second REAL
                );
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding_json TEXT,
                    embedding_model TEXT
                );
                CREATE TABLE IF NOT EXISTS deployments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    image TEXT NOT NULL,
                    replicas INTEGER NOT NULL,
                    port INTEGER NOT NULL,
                    namespace TEXT NOT NULL,
                    manifest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            deployment_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(deployments)")
            }
            if "details_json" not in deployment_columns:
                db.execute(
                    "ALTER TABLE deployments ADD COLUMN details_json TEXT NOT NULL DEFAULT '{}'"
                )

    def audit(
        self,
        action: str,
        *,
        model: str | None = None,
        status: str,
        latency_seconds: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO audit_records
                   (created_at, action, model, status, latency_seconds,
                    input_tokens, output_tokens, details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    utc_now(), action, model, status, latency_seconds,
                    input_tokens, output_tokens,
                    json.dumps(details or {}, separators=(",", ":")),
                ),
            )

    def benchmark(self, model: str, status: str, latency: float | None,
                  input_tokens: int | None, output_tokens: int | None,
                  tokens_per_second: float | None) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO benchmark_records
                   (created_at, model, status, latency_seconds, input_tokens,
                    output_tokens, tokens_per_second)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (utc_now(), model, status, latency, input_tokens, output_tokens, tokens_per_second),
            )

    def add_document(self, name: str, content: str, chunks: list[str]) -> int:
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO rag_documents(name, content, created_at) VALUES (?, ?, ?)",
                (name, content, utc_now()),
            )
            document_id = cursor.lastrowid
            if document_id is None:
                raise RuntimeError("SQLite did not return the new document ID")
            db.executemany(
                """INSERT INTO rag_chunks(document_id, chunk_index, text)
                   VALUES (?, ?, ?)""",
                [(document_id, index, text) for index, text in enumerate(chunks)],
            )
            return document_id

    def update_chunk_embedding(
        self, chunk_id: int, vector: list[float], model: str | None
    ) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE rag_chunks SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                (json.dumps(vector), model, chunk_id),
            )

    def documents(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT d.id, d.name, d.created_at, COUNT(c.id) AS chunks
                   FROM rag_documents d LEFT JOIN rag_chunks c ON c.document_id = d.id
                   GROUP BY d.id ORDER BY d.id"""
            ).fetchall()
        return [dict(row) for row in rows]

    def chunks(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT c.id, c.document_id, c.chunk_index, c.text,
                          c.embedding_json, c.embedding_model, d.name
                   FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id
                   ORDER BY c.id"""
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["embedding"] = json.loads(item.pop("embedding_json")) if item["embedding_json"] else None
            result.append(item)
        return result

    def add_deployment(self, deployment: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO deployments
                   (id, name, image, replicas, port, namespace, manifest, created_at, status, details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    deployment["id"], deployment["name"], deployment["image"],
                    deployment["replicas"], deployment["port"], deployment["namespace"],
                    deployment["manifest"], deployment["created_at"], deployment["status"],
                    json.dumps(deployment.get("details", {}), separators=(",", ":")),
                ),
            )

    def deployments(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, name, image, replicas, port, namespace, created_at, status, details_json FROM deployments ORDER BY created_at DESC"
            ).fetchall()
        return [self._deployment_record(row) for row in rows]

    def deployment(self, deployment_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM deployments WHERE id = ?", (deployment_id,)).fetchone()
        return self._deployment_record(row) if row else None

    @staticmethod
    def _deployment_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        details_json = item.pop("details_json", "{}")
        item.update(json.loads(details_json))
        return item

    def summary(self) -> dict[str, Any]:
        with self.connect() as db:
            benchmark = db.execute(
                """SELECT COUNT(*) AS count,
                          AVG(latency_seconds) AS average_latency_seconds,
                          AVG(tokens_per_second) AS average_tokens_per_second
                   FROM benchmark_records"""
            ).fetchone()
            audit_count = db.execute("SELECT COUNT(*) FROM audit_records").fetchone()[0]
            deployment_count = db.execute("SELECT COUNT(*) FROM deployments").fetchone()[0]
            inference = db.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
                          AVG(latency_seconds) AS average_latency_seconds
                   FROM audit_records WHERE action IN ('chat', 'fraud_inference')"""
            ).fetchone()
            latency_rows = db.execute(
                """SELECT latency_seconds FROM audit_records
                   WHERE action IN ('chat', 'fraud_inference') AND status = 'success'
                     AND latency_seconds IS NOT NULL ORDER BY latency_seconds"""
            ).fetchall()
        latencies = [float(row["latency_seconds"]) for row in latency_rows]

        def percentile(p: float) -> float | None:
            if not latencies:
                return None
            return latencies[max(0, math.ceil(p * len(latencies)) - 1)]

        return {
            "inference": {
                "requests": inference["total"],
                "errors": inference["errors"] or 0,
                "error_rate": (
                    (inference["errors"] or 0) / inference["total"]
                    if inference["total"] else None
                ),
                "average_latency_seconds": inference["average_latency_seconds"],
                "p50_latency_seconds": percentile(0.50),
                "p95_latency_seconds": percentile(0.95),
                "p99_latency_seconds": percentile(0.99),
            },
            "benchmarks": {
                "count": benchmark["count"],
                "average_latency_seconds": benchmark["average_latency_seconds"],
                "average_tokens_per_second": benchmark["average_tokens_per_second"],
            },
            "audit_records": audit_count,
            "deployments": deployment_count,
        }

    def recent_audit_records(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        with self.connect() as db:
            rows = db.execute(
                """SELECT created_at, action, model, status, latency_seconds,
                          input_tokens, output_tokens, details_json
                   FROM audit_records ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["details"] = json.loads(record.pop("details_json"))
            records.append(record)
        return records

    def latest_model_inference_statuses(self) -> dict[str, str]:
        inference_actions = (
            "chat",
            "benchmark_model",
            "fraud_inference",
            "rag_document_add",
            "rag_query",
            "rag_query_embedding",
            "infrastructure_review",
        )
        placeholders = ",".join("?" for _ in inference_actions)
        with self.connect() as db:
            rows = db.execute(
                f"""SELECT model, status, action, latency_seconds FROM audit_records
                    WHERE model IS NOT NULL AND action IN ({placeholders})
                    ORDER BY id DESC""",
                inference_actions,
            ).fetchall()
        latest: dict[str, str] = {}
        for row in rows:
            status = row["status"]
            if row["action"] == "infrastructure_review" and status == "complete":
                status = "success" if row["latency_seconds"] else "error"
            latest.setdefault(row["model"], status)
        return latest

    def models_with_successful_inference(self) -> set[str]:
        inference_actions = (
            "chat",
            "benchmark_model",
            "fraud_inference",
            "rag_document_add",
            "rag_query",
            "rag_query_embedding",
            "infrastructure_review",
        )
        placeholders = ",".join("?" for _ in inference_actions)
        with self.connect() as db:
            rows = db.execute(
                f"""SELECT DISTINCT model FROM audit_records
                    WHERE model IS NOT NULL AND status = 'success'
                    AND action IN ({placeholders})""",
                inference_actions,
            ).fetchall()
        return {row["model"] for row in rows}
