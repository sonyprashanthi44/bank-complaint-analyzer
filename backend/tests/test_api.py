"""Smoke tests for the analyze API."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import app  # noqa: E402


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["classifier_loaded"] is True
        assert payload["case_count"] > 0


def test_analyze_credit_card() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            json={
                "narrative": (
                    "A merchant charge of $250 appears twice on my credit card ending in 4412. "
                    "The store says they only billed once and I need the duplicate removed."
                ),
                "customer_name": "Alex",
                "account_last4": "4412",
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert "classification" in payload
        assert payload["classification"]["product"]
        assert len(payload["similar_cases"]) >= 1
        assert payload["draft_reply"]["body"]
