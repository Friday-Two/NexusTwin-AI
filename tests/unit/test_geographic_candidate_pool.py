from unittest.mock import Mock

from apps.backend.routes.retrieval import RetrievalRequest, rank_apis


def test_geo_enabled_retrieves_broader_pool_and_limits_response(monkeypatch):
    calls = []

    class FakeRetriever:
        def rank(self, query, top_k):
            calls.append(top_k)
            return [
                {
                    "document_id": f"api-{i}",
                    "description": "Search nearby places",
                    "api_text": "Nearby search",
                    "ranking_score": float(100 - i),
                }
                for i in range(top_k)
            ]

    monkeypatch.setattr(
        "apps.backend.routes.retrieval.get_retriever",
        lambda: FakeRetriever(),
    )
    monkeypatch.setattr(
        "apps.backend.routes.retrieval.get_settings",
        lambda: type("Settings", (), {"enable_geo_reranking": True})(),
    )

    response = rank_apis(
        RetrievalRequest(
            query="Find nearby places using geographic coordinates",
            top_k=3,
        ),
        current_user=Mock(),
    )

    assert calls == [12]
    assert len(response.results) == 3


def test_geo_disabled_uses_requested_candidate_count(monkeypatch):
    calls = []

    class FakeRetriever:
        def rank(self, query, top_k):
            calls.append(top_k)
            return [
                {
                    "document_id": f"api-{i}",
                    "description": "Search nearby places",
                    "api_text": "Nearby search",
                    "ranking_score": float(100 - i),
                }
                for i in range(top_k)
            ]

    monkeypatch.setattr(
        "apps.backend.routes.retrieval.get_retriever",
        lambda: FakeRetriever(),
    )
    monkeypatch.setattr(
        "apps.backend.routes.retrieval.get_settings",
        lambda: type("Settings", (), {"enable_geo_reranking": False})(),
    )

    response = rank_apis(
        RetrievalRequest(
            query="Find nearby places using geographic coordinates",
            top_k=3,
        ),
        current_user=Mock(),
    )

    assert calls == [3]
    assert len(response.results) == 3
