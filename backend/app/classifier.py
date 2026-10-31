"""TF-IDF + Logistic Regression complaint product classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .config import Settings, get_settings
from .schemas import ClassificationResult


@dataclass
class ClassifierBundle:
    pipeline: Pipeline
    label_encoder: LabelEncoder
    metrics: dict[str, Any]

    @property
    def products(self) -> list[str]:
        return [str(x) for x in self.label_encoder.classes_]


class ComplaintClassifier:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._bundle: ClassifierBundle | None = None

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    @property
    def products(self) -> list[str]:
        if not self._bundle:
            return []
        return self._bundle.products

    @property
    def metrics(self) -> dict[str, Any]:
        if not self._bundle:
            return {}
        return self._bundle.metrics

    def load(self) -> None:
        settings = self.settings
        required = [
            settings.classifier_path,
            settings.label_encoder_path,
            settings.vectorizer_path,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Classifier artifacts missing. Run `python -m backend.train` first. "
                f"Missing: {', '.join(missing)}"
            )

        vectorizer: TfidfVectorizer = joblib.load(settings.vectorizer_path)
        classifier: LogisticRegression = joblib.load(settings.classifier_path)
        label_encoder: LabelEncoder = joblib.load(settings.label_encoder_path)
        pipeline = Pipeline(
            steps=[
                ("tfidf", vectorizer),
                ("clf", classifier),
            ]
        )
        metrics: dict[str, Any] = {}
        if settings.metrics_path.exists():
            metrics = json.loads(settings.metrics_path.read_text(encoding="utf-8"))

        self._bundle = ClassifierBundle(
            pipeline=pipeline,
            label_encoder=label_encoder,
            metrics=metrics,
        )

    def predict(self, narrative: str, top_n: int = 3) -> ClassificationResult:
        if not self._bundle:
            raise RuntimeError("Classifier is not loaded")

        text = narrative.strip()
        proba = self._bundle.pipeline.predict_proba([text])[0]
        classes = self._bundle.label_encoder.classes_
        order = np.argsort(proba)[::-1]
        top_n = min(top_n, len(classes))
        top = [
            {str(classes[idx]): float(round(float(proba[idx]), 4))}
            for idx in order[:top_n]
        ]
        best_idx = int(order[0])
        return ClassificationResult(
            product=str(classes[best_idx]),
            confidence=float(round(float(proba[best_idx]), 4)),
            top_probabilities=top,
        )

    def transform(self, narratives: list[str]) -> np.ndarray:
        if not self._bundle:
            raise RuntimeError("Classifier is not loaded")
        matrix = self._bundle.pipeline.named_steps["tfidf"].transform(narratives)
        return matrix.astype(np.float32)

    def transform_dense(self, narratives: list[str], dim: int | None = None) -> np.ndarray:
        """Return L2-normalized dense vectors truncated/padded to dim for pgvector."""
        sparse = self.transform(narratives)
        dense = sparse.toarray().astype(np.float32)
        target_dim = dim or self.settings.embedding_dim
        if dense.shape[1] > target_dim:
            dense = dense[:, :target_dim]
        elif dense.shape[1] < target_dim:
            pad = np.zeros((dense.shape[0], target_dim - dense.shape[1]), dtype=np.float32)
            dense = np.hstack([dense, pad])
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms


def ensure_artifacts_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
