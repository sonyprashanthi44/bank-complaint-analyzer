"""Train TF-IDF + Logistic Regression and build local case retrieval index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.classifier import ensure_artifacts_dir  # noqa: E402
from backend.app.config import get_settings  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Sample data not found at {path}. Run scripts/generate_sample_data.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    settings = get_settings()
    ensure_artifacts_dir(settings.artifacts_dir)
    cases = load_cases(settings.sample_data_path)

    narratives = [c["narrative"] for c in cases]
    products = [c["product"] for c in cases]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(products)

    x_train, x_test, y_train, y_test = train_test_split(
        narratives,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=8000,
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        C=2.0,
    )
    clf.fit(x_train_vec, y_train)

    y_pred = clf.predict(x_test_vec)
    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(x) for x in label_encoder.classes_],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "n_samples": len(cases),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "products": [str(x) for x in label_encoder.classes_],
        "classification_report": report,
        "vectorizer": {
            "max_features": 8000,
            "ngram_range": [1, 2],
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
    }

    joblib.dump(clf, settings.classifier_path)
    joblib.dump(vectorizer, settings.vectorizer_path)
    joblib.dump(label_encoder, settings.label_encoder_path)
    settings.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Build local retrieval index using the same TF-IDF space (truncated/padded + L2-normalized)
    full_matrix = vectorizer.transform(narratives).toarray().astype(np.float32)
    dim = settings.embedding_dim
    if full_matrix.shape[1] > dim:
        full_matrix = full_matrix[:, :dim]
    elif full_matrix.shape[1] < dim:
        pad = np.zeros((full_matrix.shape[0], dim - full_matrix.shape[1]), dtype=np.float32)
        full_matrix = np.hstack([full_matrix, pad])
    norms = np.linalg.norm(full_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = full_matrix / norms

    index_payload = {
        "case_ids": [c["case_id"] for c in cases],
        "products": [c["product"] for c in cases],
        "issues": [c["issue"] for c in cases],
        "narratives": narratives,
        "resolution_summaries": [c["resolution_summary"] for c in cases],
        "company_responses": [c["company_response"] for c in cases],
        "vectors": vectors,
        "embedding_dim": dim,
    }
    joblib.dump(index_payload, settings.local_index_path)

    print("Training complete")
    print(f"  accuracy   : {metrics['accuracy']}")
    print(f"  macro_f1   : {metrics['macro_f1']}")
    print(f"  weighted_f1: {metrics['weighted_f1']}")
    print(f"  artifacts  : {settings.artifacts_dir}")
    print(f"  cases      : {len(cases)}")


if __name__ == "__main__":
    main()
