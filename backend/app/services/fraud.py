from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "amount",
    "merchant_risk_score",
    "transaction_hour",
    "transaction_velocity",
    "international",
)
MODEL_VERSION = "synthetic-fraud-logreg-v1"


class SyntheticFraudClassifier:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists():
            artifact = joblib.load(model_path)
            self.classifier = artifact["classifier"]
            self.training_metrics = artifact["metrics"]
        else:
            self.classifier, self.training_metrics = self._train()
            joblib.dump(
                {"classifier": self.classifier, "metrics": self.training_metrics},
                model_path,
            )

    @staticmethod
    def _train() -> tuple[Any, dict[str, float | int | str]]:
        rng = np.random.default_rng(20261020)
        count = 4000
        amount = rng.lognormal(mean=5.0, sigma=1.2, size=count)
        merchant_risk = rng.beta(2.0, 5.0, size=count)
        transaction_hour = rng.integers(0, 24, size=count)
        velocity = rng.poisson(lam=3.0, size=count)
        international = rng.integers(0, 2, size=count)
        features = np.column_stack(
            (amount, merchant_risk, transaction_hour, velocity, international)
        )
        score = (
            (amount > 1200).astype(float) * 1.0
            + (merchant_risk > 0.72).astype(float) * 1.4
            + ((transaction_hour < 5) | (transaction_hour > 22)).astype(float) * 0.55
            + (velocity > 8).astype(float) * 1.1
            + international * 0.45
            + rng.normal(0, 0.25, size=count)
        )
        labels = (score >= 1.7).astype(int)
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=41, stratify=labels
        )
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=41, max_iter=400, class_weight="balanced"),
        )
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        metrics: dict[str, float | int | str] = {
            "model_version": MODEL_VERSION,
            "training_samples": int(len(x_train)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "disclaimer": "Offline metrics on generated synthetic data; not genuine bank fraud-detection accuracy.",
        }
        return classifier, metrics

    def predict(self, features: list[float | int]) -> tuple[str, float]:
        probability = float(self.classifier.predict_proba([features])[0, 1])
        return ("fraud" if probability >= 0.5 else "legitimate", probability)
