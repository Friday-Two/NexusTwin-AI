import math

import pandas as pd
import pytest

from scripts.evaluate_preference_rankers import (
    evaluate,
    ndcg_at_k,
)


def test_ndcg_perfect_ranking():
    assert ndcg_at_k([1, 0], 2) == 1.0


def test_ndcg_reversed_ranking():
    assert ndcg_at_k([0, 1], 2) == pytest.approx(
        1 / math.log2(3)
    )


def test_ndcg_at_one():
    assert ndcg_at_k([1, 0], 1) == 1.0


def test_ndcg_empty_labels():
    assert ndcg_at_k([], 5) == 0.0


def test_ndcg_no_relevant_candidates():
    assert ndcg_at_k([0, 0], 2) == 0.0


def test_ndcg_respects_cutoff():
    assert ndcg_at_k([0, 0, 1], 2) == 0.0


def test_evaluate_reports_dataset_metadata():
    frame = pd.DataFrame([
        {
            "query_id": "q1",
            "preference_text": "cheap hotel",
            "service_name": "Budget Hotel",
            "category": "hotel",
            "description": "low cost",
            "label": 1,
            "label_source": "synthetic_rule",
        },
        {
            "query_id": "q1",
            "preference_text": "cheap hotel",
            "service_name": "Luxury Hotel",
            "category": "hotel",
            "description": "high cost",
            "label": 0,
            "label_source": "synthetic_rule",
        },
    ])

    class FakeRanker:
        def rank(self, services, preferences):
            return [(services[0], 1.0), (services[1], 0.0)]

    report = evaluate(frame, FakeRanker(), "sample.csv")

    assert report["rows"] == 2
    assert report["queries"] == 1
    assert report["candidate_count_min"] == 2
    assert report["candidate_count_max"] == 2
    assert report["label_sources"] == ["synthetic_rule"]
    assert report["dataset_type"].startswith("synthetic")
    assert report["summary"]["ML"]["recall_at_1"] == 1.0


def test_evaluate_rejects_inconsistent_query_text():
    frame = pd.DataFrame([
        {
            "query_id": "q1",
            "preference_text": "cheap hotel",
            "service_name": "Budget Hotel",
            "category": "hotel",
            "description": "low cost",
            "label": 1,
        },
        {
            "query_id": "q1",
            "preference_text": "luxury hotel",
            "service_name": "Luxury Hotel",
            "category": "hotel",
            "description": "high cost",
            "label": 0,
        },
    ])

    class FakeRanker:
        def rank(self, services, preferences):
            return [(service, 0.0) for service in services]

    with pytest.raises(ValueError, match="expected one preference text"):
        evaluate(frame, FakeRanker(), "sample.csv")
