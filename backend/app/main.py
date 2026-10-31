"""FastAPI entrypoint for Bank Complaint Analyzer."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .classifier import ComplaintClassifier
from .config import get_settings
from .llm_draft import ReplyDrafter
from .retrieval import CaseRetriever
from .schemas import (
    AnalyzeResponse,
    ComplaintRequest,
    HealthResponse,
    MetricsResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bank_complaint_analyzer")

settings = get_settings()
classifier = ComplaintClassifier(settings)
retriever = CaseRetriever(classifier, settings)
drafter = ReplyDrafter(settings)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading classifier artifacts...")
    classifier.load()
    logger.info("Loading retrieval backend...")
    retriever.load()
    logger.info(
        "Startup complete | retrieval=%s cases=%s llm_ready=%s",
        retriever.backend,
        retriever.case_count,
        drafter.ready,
    )
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Classifies bank complaints, finds similar past cases, "
        "and drafts a first-response email for agent review."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "app": settings.app_name,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        classifier_loaded=classifier.loaded,
        retrieval_backend=retriever.backend,
        llm_ready=drafter.ready,
        case_count=retriever.case_count,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["meta"])
def metrics() -> MetricsResponse:
    if not classifier.loaded:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    return MetricsResponse(metrics=classifier.metrics, products=classifier.products)


@app.get("/products", tags=["meta"])
def products() -> dict[str, list[str]]:
    return {"products": classifier.products}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analyze"])
def analyze(payload: ComplaintRequest) -> AnalyzeResponse:
    if not classifier.loaded:
        raise HTTPException(status_code=503, detail="Classifier not loaded")

    narrative = payload.narrative.strip()
    if len(narrative) < 20:
        raise HTTPException(status_code=422, detail="Narrative must be at least 20 characters")

    classification = classifier.predict(narrative)
    similar = retriever.search(narrative, top_k=payload.top_k)
    draft = drafter.draft(
        narrative=narrative,
        classification=classification,
        similar_cases=similar,
        customer_name=payload.customer_name,
        account_last4=payload.account_last4,
        channel=payload.channel,
    )

    # Keep notes short and agent-facing — no setup/env instructions in the UI.
    notes: list[str] = []
    if classification.confidence < 0.45:
        notes.append("Confidence is on the lower side. Have a specialist double-check before sending.")
    if draft.is_fallback:
        notes.append("This draft used the standard reply template.")
    notes.append("Please review this draft before it goes to the customer.")

    return AnalyzeResponse(
        classification=classification,
        similar_cases=similar,
        draft_reply=draft,
        retrieval_backend=retriever.backend,
        notes=notes,
    )
