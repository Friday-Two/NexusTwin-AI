from core.retrieval.geographic_reranker import (
    geographic_adjustment,
    is_geographic_query,
    rerank,
)


def test_detects_geographic_query():
    assert is_geographic_query(
        "Find nearby places using geographic coordinates"
    )


def test_does_not_detect_hotel_query():
    assert not is_geographic_query("Find hotels in Hyderabad")


def test_reranks_nearby_search_above_helper():
    results = [
        {
            "document_id": "helper",
            "description": '{"api_name":"Categories","api_description":"Categories of establishments"}',
            "api_text": "",
            "ranking_score": 0.90,
        },
        {
            "document_id": "nearby",
            "description": '{"api_name":"Search Nearby","api_description":"Search businesses near the requested coordinates"}',
            "api_text": "",
            "ranking_score": 0.80,
        },
    ]

    ranked = rerank("Find nearby places using geographic coordinates", results)

    assert [item["document_id"] for item in ranked] == ["nearby", "helper"]


def test_non_geographic_query_preserves_order():
    results = [
        {"document_id": "a", "ranking_score": 0.5},
        {"document_id": "b", "ranking_score": 0.9},
    ]

    ranked = rerank("Find hotels in Hyderabad", results)

    assert [item["document_id"] for item in ranked] == ["a", "b"]


def test_rerank_does_not_mutate_original_results():
    results = [
        {
            "document_id": "nearby",
            "description": "Search nearby places",
            "api_text": "",
            "ranking_score": 0.5,
        }
    ]
    original = [dict(item) for item in results]

    rerank("Find nearby places using coordinates", results)

    assert results == original


def test_helper_endpoint_gets_no_bonus():
    item = {
        "description": "Categories of establishments",
        "api_text": "",
        "ranking_score": 0.8,
    }

    assert geographic_adjustment(item) == 0.0
