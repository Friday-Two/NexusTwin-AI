from fastapi.testclient import TestClient

from apps.backend.main import app


def test_retrieval_route_keeps_baseline_when_flag_disabled(monkeypatch):
    baseline = [
        {
            "document_id": "helper",
            "description": "Categories of establishments",
            "api_text": "Categories",
            "ranking_score": 0.9,
        },
        {
            "document_id": "nearby",
            "description": "Search nearby places",
            "api_text": "Nearby search",
            "ranking_score": 0.8,
        },
    ]

    class FakeRetriever:
        def rank(self, query, top_k):
            return [dict(item) for item in baseline]

    monkeypatch.setattr(
        "apps.backend.routes.retrieval.get_retriever",
        lambda: FakeRetriever(),
    )
    monkeypatch.setattr(
        "apps.backend.routes.retrieval.get_settings",
        lambda: type("Settings", (), {"enable_geo_reranking": False})(),
    )

    # Bypass authentication for this focused route test.
    from apps.backend.routes.retrieval import rank_apis, RetrievalRequest
    from unittest.mock import Mock

    response = rank_apis(
        RetrievalRequest(query="Find nearby places using coordinates"),
        current_user=Mock(),
    )

    assert [item.document_id for item in response.results] == ["helper", "nearby"]


def test_retrieval_route_reranks_when_flag_enabled(monkeypatch):
    baseline = [
        {
            "document_id": "helper",
            "description": "Categories of establishments",
            "api_text": "Categories",
            "ranking_score": 0.9,
        },
        {
            "document_id": "nearby",
            "description": "Search nearby places",
            "api_text": "Nearby search",
            "ranking_score": 0.8,
        },
    ]

    class FakeRetriever:
        def rank(self, query, top_k):
            return [dict(item) for item in baseline]

    monkeypatch.setattr(
        "apps.backend.routes.retrieval.get_retriever",
        lambda: FakeRetriever(),
    )
    monkeypatch.setattr(
        "apps.backend.routes.retrieval.get_settings",
        lambda: type("Settings", (), {"enable_geo_reranking": True})(),
    )

    from apps.backend.routes.retrieval import rank_apis, RetrievalRequest
    from unittest.mock import Mock

    response = rank_apis(
        RetrievalRequest(query="Find nearby places using coordinates"),
        current_user=Mock(),
    )

    assert [item.document_id for item in response.results] == ["nearby", "helper"]
