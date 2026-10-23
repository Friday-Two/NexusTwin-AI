from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.backend.dependencies import get_db
from apps.backend.main import app
from database.models.entities import Base


def test_assistant_interpret_uses_validated_llm_output(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(
        "apps.backend.routes.assistant.ollama_chat_json",
        lambda system_prompt, user_prompt: {
            "intent": "find affordable accommodation near the airport",
            "category": "Accommodation",
            "constraints": ["affordable", "near airport"],
            "missing_information": ["travel dates"],
            "summary": "Find affordable accommodation near the airport.",
        },
    )

    try:
        with TestClient(app) as client:
            registration = client.post("/api/v1/auth/register", json={
                "email": "llm-test@example.com",
                "display_name": "LLM Test",
                "password": "TestPassword-123!",
            })
            assert registration.status_code == 201
            login = client.post("/api/v1/auth/login", json={
                "email": "llm-test@example.com",
                "password": "TestPassword-123!",
            })
            assert login.status_code == 200
            client.headers["Authorization"] = "Bearer " + login.json()["access_token"]

            response = client.post("/api/v1/assistant/interpret", json={
                "message": "Find affordable accommodation near the airport",
                "budget_limit": 10000,
            })

        assert response.status_code == 200
        assert response.json()["category"] == "Accommodation"
        assert response.json()["constraints"] == ["affordable", "near airport"]
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_assistant_rejects_invalid_request_without_calling_model(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Model must not be called for invalid input")

    monkeypatch.setattr(
        "apps.backend.routes.assistant.ollama_chat_json",
        fail_if_called,
    )
    response = TestClient(app).post("/api/v1/assistant/interpret", json={"message": "x"})
    assert response.status_code == 422
