import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.negotiation.service_ranker import rank_services
from core.negotiation.preference_ml_ranker import get_preference_ml_ranker
from database.models.entities import Service, UserPreference

DATA = ROOT / "datasets" / "ranking" / "starter_examples.csv"
OUTPUT = ROOT / "artifacts" / "ranking" / "evaluation_comparison.json"


def ndcg_at_k(labels, k):
    import math

    gains = [(2 ** label - 1) / math.log2(i + 2)
             for i, label in enumerate(labels[:k])]
    dcg = sum(gains)
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum((2 ** label - 1) / math.log2(i + 2)
               for i, label in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate(frame, ml_ranker):
    results = {"Deterministic": [], "ML": []}

    for query_id, group in frame.groupby("query_id", sort=True):
        group = group.reset_index(drop=True)
        query_texts = group["preference_text"].dropna().unique()

        if len(query_texts) != 1:
            raise ValueError(f"{query_id}: expected one preference text")

        services = [
            Service(
                id=i + 1,
                name=str(row.service_name),
                category=str(row.category),
                description=str(row.description),
                active=True,
            )
            for i, row in group.iterrows()
        ]
        preferences = [
            UserPreference(
                preference_key="evaluation_query",
                preference_value=str(query_texts[0]),
                is_sensitive=False,
            )
        ]
        labels_by_name = dict(zip(group["service_name"], group["label"].astype(int)))

        ranked_methods = {
            "Deterministic": rank_services(services, preferences),
            "ML": ml_ranker.rank(services, preferences),
        }

        for method, ranked in ranked_methods.items():
            labels = [labels_by_name[service.name] for service, _ in ranked]
            first_relevant = next(
                (i + 1 for i, label in enumerate(labels) if label == 1),
                None,
            )
            results[method].append({
                "query_id": str(query_id),
                "recall_at_1": float(any(label == 1 for label in labels[:1])),
                "recall_at_2": float(any(label == 1 for label in labels[:2])),
                "mrr_at_2": 1.0 / first_relevant if first_relevant else 0.0,
                "ndcg_at_2": ndcg_at_k(labels, 2),
                "top_service": ranked[0][0].name,
                "top_service_relevant": labels[0] == 1,
            })

    summary = {}
    for method, rows in results.items():
        summary[method] = {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in (
                "recall_at_1",
                "recall_at_2",
                "mrr_at_2",
                "ndcg_at_2",
                "top_service_relevant",
            )
        }
        summary[method]["queries_evaluated"] = len(rows)

    return {
        "dataset": str(DATA.relative_to(ROOT)),
        "dataset_type": "synthetic smoke-test data; not a real-world benchmark",
        "warning": (
            "Only 16 query groups and two candidates per query. "
            "Do not use these results to claim production performance."
        ),
        "summary": summary,
        "per_query": results,
    }


def main():
    frame = pd.read_csv(DATA)
    required = {
        "query_id", "preference_text", "service_name",
        "category", "description", "label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Dataset is empty or contains missing required values.")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary.")

    for query_id, group in frame.groupby("query_id"):
        if group["label"].nunique() != 2:
            raise ValueError(f"{query_id} must contain both labels 0 and 1.")

    ml_ranker = get_preference_ml_ranker()
    report = evaluate(frame, ml_ranker)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Preference Ranker Comparison ===")
    print("Dataset: synthetic smoke-test data")
    print(f"Queries: {frame['query_id'].nunique()}")
    print("\nMethod         Recall@1  MRR@2   NDCG@2  Top-1 accuracy")
    for method, metrics in report["summary"].items():
        print(
            f"{method:14} "
            f"{metrics['recall_at_1']:.3f}     "
            f"{metrics['mrr_at_2']:.3f}   "
            f"{metrics['ndcg_at_2']:.3f}   "
            f"{metrics['top_service_relevant']:.3f}"
        )

    print(f"\nSaved report: {OUTPUT.relative_to(ROOT)}")
    print(report["warning"])


if __name__ == "__main__":
    main()
