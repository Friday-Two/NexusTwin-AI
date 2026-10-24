import numpy as np
import pytest
import torch
from torch import nn

from core.negotiation.preference_ml_ranker import (
    PairRanker,
    PreferenceMLRanker,
    get_preference_ml_ranker,
)
from database.models.entities import Service, UserPreference


def make_service(service_id, name, category="Accommodation", description=""):
    return Service(
        id=service_id,
        name=name,
        category=category,
        description=description,
        active=True,
    )


def make_preference(value, sensitive=False):
    return UserPreference(
        preference_key="test",
        preference_value=value,
        is_sensitive=sensitive,
    )


class FakeEncoder:
    def __init__(self):
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        # Deterministic, normalized 384-dimensional vectors.
        result = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            result[i, hash(text) % 384] = 1.0
        return result


def make_ranker():
    ranker = PreferenceMLRanker.__new__(PreferenceMLRanker)
    ranker.encoder = FakeEncoder()
    ranker.model = PairRanker(384 * 3)
    ranker.model.eval()
    ranker.threshold = 0.5
    return ranker


def test_no_preferences_returns_alphabetical_zero_scores():
    ranker = make_ranker()
    services = [
        make_service(2, "Zebra Hotel"),
        make_service(1, "Alpha Hotel"),
    ]

    ranked = ranker.rank(services, [])

    assert [service.name for service, _ in ranked] == [
        "Alpha Hotel", "Zebra Hotel"
    ]
    assert [score for _, score in ranked] == [0.0, 0.0]
    assert ranker.encoder.seen == []


def test_sensitive_preference_text_is_not_encoded():
    ranker = make_ranker()
    services = [make_service(1, "City Hotel")]

    ranked = ranker.rank(
        services,
        [
            make_preference("ordinary budget preference"),
            make_preference("private-secret-value", sensitive=True),
        ],
    )

    assert len(ranked) == 1
    assert all("private-secret-value" not in text for text in ranker.encoder.seen)


def test_empty_services_returns_empty_list():
    ranker = make_ranker()

    assert ranker.rank([], [make_preference("budget")]) == []


def test_missing_checkpoint_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing preference checkpoint"):
        PreferenceMLRanker(tmp_path / "missing.pt")


def test_invalid_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save({"unexpected": 1}, path)

    with pytest.raises(ValueError, match="Checkpoint missing keys"):
        PreferenceMLRanker(path)


def test_loader_is_cached():
    get_preference_ml_ranker.cache_clear()
    assert get_preference_ml_ranker.cache_info().maxsize == 1
    get_preference_ml_ranker.cache_clear()


def test_fallback_uses_deterministic_ranker_when_ml_disabled():
    from core.negotiation.preference_ml_ranker import rank_services_with_fallback
    from core.negotiation.service_ranker import rank_services

    services = [
        make_service(2, "Zebra Taxi", "Transport"),
        make_service(1, "Budget Hotel", "Accommodation"),
    ]
    preferences = [make_preference("budget")]

    actual = rank_services_with_fallback(services, preferences, use_ml=False)
    expected = rank_services(services, preferences)

    assert actual == expected


def test_fallback_uses_deterministic_ranker_when_ml_fails(monkeypatch):
    import core.negotiation.preference_ml_ranker as module

    services = [
        make_service(2, "Zebra Taxi", "Transport"),
        make_service(1, "Budget Hotel", "Accommodation"),
    ]
    preferences = [make_preference("budget")]

    def fail_to_load():
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(module, "get_preference_ml_ranker", fail_to_load)

    actual = module.rank_services_with_fallback(
        services, preferences, use_ml=True
    )
    from core.negotiation.service_ranker import rank_services
    expected = rank_services(services, preferences)

    assert actual == expected
