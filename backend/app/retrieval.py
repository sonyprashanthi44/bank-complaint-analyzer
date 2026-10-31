"""Similar-case retrieval via PostgreSQL/pgvector or local TF-IDF index."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np

from .classifier import ComplaintClassifier
from .config import Settings, get_settings
from .schemas import SimilarCase

logger = logging.getLogger(__name__)


@dataclass
class LocalCaseIndex:
    case_ids: list[str]
    products: list[str]
    issues: list[str]
    narratives: list[str]
    resolution_summaries: list[str]
    company_responses: list[str]
    vectors: np.ndarray  # (n, dim) L2-normalized


class CaseRetriever:
    def __init__(
        self,
        classifier: ComplaintClassifier,
        settings: Settings | None = None,
    ) -> None:
        self.classifier = classifier
        self.settings = settings or get_settings()
        self._local_index: LocalCaseIndex | None = None
        self._pg_ready = False
        self._backend = "uninitialized"
        self._engine = None

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def case_count(self) -> int:
        if self._backend.startswith("postgres") and self._engine is not None:
            try:
                from sqlalchemy import text

                with self._engine.connect() as conn:
                    return int(conn.execute(text("SELECT COUNT(*) FROM complaint_cases")).scalar_one())
            except Exception:  # noqa: BLE001
                pass
        if self._local_index is not None:
            return len(self._local_index.case_ids)
        return 0

    def load(self) -> None:
        self._load_local_index()
        if self.settings.database_url:
            try:
                self._init_postgres()
                self._backend = "postgres+pgvector"
                logger.info("Retrieval backend: postgres+pgvector")
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Postgres/pgvector unavailable (%s). Falling back to local index.", exc)
        if self._local_index is None:
            raise RuntimeError(
                "No retrieval index available. Run training to build local_case_index.joblib "
                "or configure DATABASE_URL with pgvector."
            )
        self._backend = "local-tfidf"
        logger.info("Retrieval backend: local-tfidf (%s cases)", len(self._local_index.case_ids))

    def _load_local_index(self) -> None:
        path = self.settings.local_index_path
        if not path.exists():
            self._local_index = None
            return
        payload = joblib.load(path)
        self._local_index = LocalCaseIndex(
            case_ids=list(payload["case_ids"]),
            products=list(payload["products"]),
            issues=list(payload["issues"]),
            narratives=list(payload["narratives"]),
            resolution_summaries=list(payload["resolution_summaries"]),
            company_responses=list(payload["company_responses"]),
            vectors=np.asarray(payload["vectors"], dtype=np.float32),
        )

    def _init_postgres(self) -> None:
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import Engine

        url = self.settings.database_url
        assert url is not None
        # Railway/Render sometimes provide postgres://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://") and "+psycopg" not in url:
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)

        engine: Engine = create_engine(url, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS complaint_cases (
                        case_id TEXT PRIMARY KEY,
                        product TEXT NOT NULL,
                        issue TEXT NOT NULL,
                        narrative TEXT NOT NULL,
                        resolution_summary TEXT NOT NULL,
                        company_response TEXT NOT NULL,
                        embedding vector({self.settings.embedding_dim}) NOT NULL
                    )
                    """
                )
            )
        self._engine = engine
        self._pg_ready = True
        # Seed from local sample if table empty
        with engine.connect() as conn:
            count = int(conn.execute(text("SELECT COUNT(*) FROM complaint_cases")).scalar_one())
        if count == 0:
            self.seed_postgres_from_sample()
        # IVFFlat needs rows before index build; ignore if extension/plan disallows it
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        CREATE INDEX IF NOT EXISTS complaint_cases_embedding_idx
                        ON complaint_cases
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 50)
                        """
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create ivfflat index (%s). Sequential scan will be used.", exc)

    def seed_postgres_from_sample(self) -> int:
        if not self._engine:
            raise RuntimeError("Postgres engine not initialized")
        sample_path = self.settings.sample_data_path
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample data not found: {sample_path}")
        cases = json.loads(sample_path.read_text(encoding="utf-8"))
        narratives = [c["narrative"] for c in cases]
        vectors = self.classifier.transform_dense(narratives, dim=self.settings.embedding_dim)

        from sqlalchemy import text

        inserted = 0
        with self._engine.begin() as conn:
            for case, vector in zip(cases, vectors, strict=True):
                embedding = "[" + ",".join(f"{float(x):.8f}" for x in vector.tolist()) + "]"
                conn.execute(
                    text(
                        """
                        INSERT INTO complaint_cases (
                            case_id, product, issue, narrative,
                            resolution_summary, company_response, embedding
                        ) VALUES (
                            :case_id, :product, :issue, :narrative,
                            :resolution_summary, :company_response, CAST(:embedding AS vector)
                        )
                        ON CONFLICT (case_id) DO UPDATE SET
                            product = EXCLUDED.product,
                            issue = EXCLUDED.issue,
                            narrative = EXCLUDED.narrative,
                            resolution_summary = EXCLUDED.resolution_summary,
                            company_response = EXCLUDED.company_response,
                            embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "case_id": case["case_id"],
                        "product": case["product"],
                        "issue": case["issue"],
                        "narrative": case["narrative"],
                        "resolution_summary": case["resolution_summary"],
                        "company_response": case["company_response"],
                        "embedding": embedding,
                    },
                )
                inserted += 1
        return inserted

    def search(self, narrative: str, top_k: int = 5) -> list[SimilarCase]:
        if self._backend.startswith("postgres") and self._engine is not None:
            try:
                return self._search_postgres(narrative, top_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Postgres search failed (%s). Using local index.", exc)
        return self._search_local(narrative, top_k)

    def _search_local(self, narrative: str, top_k: int) -> list[SimilarCase]:
        if self._local_index is None:
            return []
        query = self.classifier.transform_dense([narrative], dim=self._local_index.vectors.shape[1])[0]
        scores = self._local_index.vectors @ query
        top_idx = np.argsort(scores)[::-1][:top_k]
        results: list[SimilarCase] = []
        for idx in top_idx:
            results.append(
                SimilarCase(
                    case_id=self._local_index.case_ids[idx],
                    product=self._local_index.products[idx],
                    issue=self._local_index.issues[idx],
                    narrative=self._local_index.narratives[idx],
                    resolution_summary=self._local_index.resolution_summaries[idx],
                    company_response=self._local_index.company_responses[idx],
                    similarity=float(round(float(scores[idx]), 4)),
                    source="local-tfidf",
                )
            )
        return results

    def _search_postgres(self, narrative: str, top_k: int) -> list[SimilarCase]:
        from sqlalchemy import text

        vector = self.classifier.transform_dense([narrative], dim=self.settings.embedding_dim)[0]
        embedding = "[" + ",".join(f"{float(x):.8f}" for x in vector.tolist()) + "]"
        sql = text(
            """
            SELECT
                case_id, product, issue, narrative, resolution_summary, company_response,
                1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM complaint_cases
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
            """
        )
        assert self._engine is not None
        with self._engine.connect() as conn:
            rows = conn.execute(sql, {"embedding": embedding, "top_k": top_k}).mappings().all()
        return [
            SimilarCase(
                case_id=row["case_id"],
                product=row["product"],
                issue=row["issue"],
                narrative=row["narrative"],
                resolution_summary=row["resolution_summary"],
                company_response=row["company_response"],
                similarity=float(round(float(row["similarity"]), 4)),
                source="postgres+pgvector",
            )
            for row in rows
        ]

    def status(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "case_count": self.case_count,
            "pg_ready": self._pg_ready,
        }
