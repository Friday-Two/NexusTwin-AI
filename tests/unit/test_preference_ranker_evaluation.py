import math

import pytest

from scripts.evaluate_preference_rankers import ndcg_at_k


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
