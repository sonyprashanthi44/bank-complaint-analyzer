"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseModel):
    app_name: str = "Bank Complaint Analyzer"
    app_version: str = "1.0.0"
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
            ).split(",")
            if origin.strip()
        ]
    )
    database_url: str | None = Field(default_factory=lambda: os.getenv("DATABASE_URL") or None)
    embedding_dim: int = 384
    top_k_similar: int = Field(default_factory=lambda: int(os.getenv("TOP_K_SIMILAR", "5")))
    groq_api_key: str | None = Field(default_factory=lambda: os.getenv("GROQ_API_KEY") or None)
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    artifacts_dir: Path = ARTIFACTS_DIR
    data_dir: Path = DATA_DIR
    classifier_path: Path = ARTIFACTS_DIR / "classifier.joblib"
    label_encoder_path: Path = ARTIFACTS_DIR / "label_encoder.joblib"
    vectorizer_path: Path = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
    local_index_path: Path = ARTIFACTS_DIR / "local_case_index.joblib"
    metrics_path: Path = ARTIFACTS_DIR / "metrics.json"
    sample_data_path: Path = DATA_DIR / "sample_complaints.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
