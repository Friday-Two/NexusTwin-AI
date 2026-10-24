import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.negotiation.service_ranker import rank_services
from core.negotiation.preference_ml_ranker import get_preference_ml_ranker
from database.models.entities import Service, UserPreference

DEFAULT_DATA = ROOT / "datasets/ranking/starter_examples.csv"
OUTPUT_DIR = ROOT / "artifacts/ranking"


def ndcg_at_k(labels, k):
    def dcg(values):
        return sum(
            (2**label - 1) / math.log2(i + 2)
            for i, label in enumerate(values[:k])
        )

    ideal = sorted(labels, reverse=True)
    ideal_dcg = dcg(ideal)
    return dcg(labels) / ideal_dcg if ideal_dcg else 0.0


def evaluate(frame, ml_ranker, dataset_path):
    results = {"Deterministic": [], "ML": []}

    for query_id, group in frame.groupby("query_id", sort=True):
        group = group.reset_index(drop=True)
        queries = group["preference_text"].unique()

        if len(queries) != 1:
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
                preference_value=str(queries[0]),
                is_sensitive=False,
            )
        ]
        labels_by_name = dict(zip(group.service_name, group.label.astype(int)))

        ranked_methods = {
            "Deterministic": rank_services(services, preferences),
            "ML": ml_ranker.rank(services, preferences),
        }

        for method, ranked in ranked_methods.items():
            labels = [labels_by_name[s.name] for s, _ in ranked]
            first_relevant = next(
                (i + 1 for i, label in enumerate(labels) if label == 1),
                None,
            )
            results[method].append({
                "query_id": str(query_id),
                "candidate_count": len(labels),
                "recall_at_1": float(any(labels[:1])),
                "recall_at_5": (
                    sum(labels[:5]) / sum(labels) if sum(labels) else 0.0
                ),
                "mrr_at_10": (
                    1.0 / first_relevant
                    if first_relevant is not None and first_relevant <= 10
                    else 0.0
                ),
                "ndcg_at_5": ndcg_at_k(labels, 5),
                "top_service": ranked[0][0].name if ranked else None,
                "top_service_relevant": bool(labels[0]) if labels else False,
            })

    metrics = (
        "recall_at_1", "recall_at_5", "mrr_at_10",
        "ndcg_at_5", "top_service_relevant",
    )
    summary = {
        method: {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in metrics
        }
        for method, rows in results.items()
    }

    label_source = (
        sorted(frame["label_source"].astype(str).unique().tolist())
        if "label_source" in frame.columns
        else ["unspecified"]
    )
    return {
        "dataset": str(dataset_path),
        "dataset_type": "synthetic; not a real-world benchmark",
        "label_sources": label_source,
        "rows": len(frame),
        "queries": int(frame.query_id.nunique()),
        "candidate_count_min": int(frame.groupby("query_id").size().min()),
        "candidate_count_max": int(frame.groupby("query_id").size().max()),
        "warning": (
            "Labels are synthetic or unverified. These results measure "
            "pipeline behavior only, not real-world relevance or production quality."
        ),
        "summary": summary,
        "per_query": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA.relative_to(ROOT)),
        help="CSV dataset path, relative to project root or absolute",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    data_path = data_path.resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    required = {
        "query_id", "preference_text", "service_name",
        "category", "description", "label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Dataset is empty or has missing required values.")
    if not set(frame.label.unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary.")
    if frame.duplicated(["query_id", "service_name"]).any():
        raise ValueError("Duplicate service within a query.")
    for query_id, group in frame.groupby("query_id"):
        if group.preference_text.nunique() != 1:
            raise ValueError(f"{query_id}: inconsistent query text.")
        if group.label.nunique() != 2:
            raise ValueError(f"{query_id}: needs both positive and negative labels.")

    report = evaluate(frame, get_preference_ml_ranker(), data_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"evaluation_{data_path.stem}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Preference Ranker Comparison ===")
    print(f"Dataset: {report['dataset']}")
    print(f"Rows: {report['rows']} | Queries: {report['queries']}")
    print(
        f"Candidates/query: {report['candidate_count_min']}-"
        f"{report['candidate_count_max']}"
    )
    print("Label sources:", ", ".join(report["label_sources"]))
    print("\nMethod         Recall@1  Recall@5  MRR@10  NDCG@5  Top-1 accuracy")
    for method, values in report["summary"].items():
        print(
            f"{method:14} "
            f"{values['recall_at_1']:.3f}     "
            f"{values['recall_at_5']:.3f}     "
            f"{values['mrr_at_10']:.3f}   "
            f"{values['ndcg_at_5']:.3f}   "
            f"{values['top_service_relevant']:.3f}"
        )
    print(f"\nSaved report: {output.relative_to(ROOT)}")
    print(report["warning"])


if __name__ == "__main__":
    main()
