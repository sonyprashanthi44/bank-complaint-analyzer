"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ComplaintRequest(BaseModel):
    narrative: str = Field(..., min_length=20, description="Customer complaint narrative")
    customer_name: str | None = Field(default=None, max_length=120)
    account_last4: str | None = Field(default=None, max_length=4)
    channel: str = Field(default="web", description="Intake channel: web, phone, email, branch")
    top_k: int = Field(default=5, ge=1, le=10)


class ClassificationResult(BaseModel):
    product: str
    confidence: float
    top_probabilities: list[dict[str, float]]


class SimilarCase(BaseModel):
    case_id: str
    product: str
    issue: str
    narrative: str
    resolution_summary: str
    company_response: str
    similarity: float
    source: str = "local"


class DraftReply(BaseModel):
    subject: str
    body: str
    tone: str
    grounded_case_ids: list[str]
    provider: str
    is_fallback: bool = False


class AnalyzeResponse(BaseModel):
    classification: ClassificationResult
    similar_cases: list[SimilarCase]
    draft_reply: DraftReply
    retrieval_backend: str
    notes: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    classifier_loaded: bool
    retrieval_backend: str
    llm_ready: bool
    case_count: int


class MetricsResponse(BaseModel):
    metrics: dict[str, Any]
    products: list[str]
