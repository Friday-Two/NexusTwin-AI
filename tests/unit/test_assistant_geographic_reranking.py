from unittest.mock import Mock

from apps.backend.routes.assistant import (
    AssistantPlanRequest,
    create_assistant_plan,
)


def test_assistant_plan_keeps_baseline_when_flag_disabled(monkeypatch):
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
        "apps.backend.routes.assistant.get_retriever",
        lambda: FakeRetriever(),
    )
    monkeypatch.setattr(
        "apps.backend.routes.assistant.get_settings",
        lambda: type("Settings", (), {"enable_geo_reranking": False})(),
    )

    monkeypatch.setattr(
        "apps.backend.routes.assistant.interpret_intent",
        lambda *args, **kwargs: {
            "intent": "Find nearby places",
            "category": "Travel",
            "constraints": [],
            "missing_information": [],
            "summary": "Find places using geographic coordinates",
        },
    )

    response = create_assistant_plan(
        AssistantPlanRequest(
            message="Find nearby places using coordinates",
            top_k=2,
        ),
        db=Mock(),
        current_user=Mock(),
    )

    assert [item.document_id for item in response.retrieved_apis] == [
        "helper", "nearby"
    ]


def test_assistant_plan_reranks_when_flag_enabled(monkeypatch):
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
        "apps.backend.routes.assistant.get_retriever",
        lambda: FakeRetriever(),
    )
    monkeypatch.setattr(
        "apps.backend.routes.assistant.get_settings",
        lambda: type("Settings", (), {"enable_geo_reranking": True})(),
    )

    monkeypatch.setattr(
        "apps.backend.routes.assistant.interpret_intent",
        lambda *args, **kwargs: {
            "intent": "Find nearby places",
            "category": "Travel",
            "constraints": [],
            "missing_information": [],
            "summary": "Find places using geographic coordinates",
        },
    )

    response = create_assistant_plan(
        AssistantPlanRequest(
            message="Find nearby places using coordinates",
            top_k=2,
        ),
        db=Mock(),
        current_user=Mock(),
    )

    assert [item.document_id for item in response.retrieved_apis] == [
        "nearby", "helper"
    ]
