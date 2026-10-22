import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.backend.dependencies import get_db
from apps.backend.main import app
from database.models.entities import (
    Base,
    Service,
)


@pytest.fixture
def negotiation_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        users = []

        for email in (
            "negotiation-api-a@example.com",
            "negotiation-api-b@example.com",
        ):
            registration = client.post(
                "/api/v1/auth/register",
                json={
                    "email": email,
                    "display_name": "Negotiation API Test",
                    "password": "TestPassword-123!",
                },
            )
            assert registration.status_code == 201

            login = client.post(
                "/api/v1/auth/login",
                json={
                    "email": email,
                    "password": "TestPassword-123!",
                },
            )
            assert login.status_code == 200
            users.append(login.json()["access_token"])

        db = TestingSessionLocal()
        try:
            service = Service(
                name="Budget Hotel",
                category="Accommodation",
                description="Affordable hotel",
                active=True,
            )
            db.add(service)
            db.commit()
            db.refresh(service)
            service_id = service.id
        finally:
            db.close()

        client.headers.update(
            {"Authorization": f"Bearer {users[0]}"}
        )

        yield {
            "client": client,
            "tokens": users,
            "service_id": service_id,
            "session_factory": TestingSessionLocal,
        }

    app.dependency_overrides.clear()
    engine.dispose()


def create_negotiation(context):
    response = context["client"].post(
        "/api/v1/negotiations",
        json={
            "service_id": context["service_id"],
            "budget_limit": 1000,
            "max_offers": 3,
        },
    )
    assert response.status_code == 201
    return response.json()


def test_create_negotiation_returns_offers(negotiation_client):
    result = create_negotiation(negotiation_client)

    assert result["status"] == "NEGOTIATING"
    assert result["service_id"] == negotiation_client["service_id"]
    assert len(result["offers"]) == 3
    assert all(
        offer["amount"] <= result["budget_limit"]
        for offer in result["offers"]
    )


def test_create_negotiation_requires_authentication(negotiation_client):
    client = negotiation_client["client"]
    client.headers.pop("Authorization", None)

    response = client.post(
        "/api/v1/negotiations",
        json={
            "service_id": negotiation_client["service_id"],
            "budget_limit": 1000,
        },
    )

    assert response.status_code == 401


def test_get_negotiation_returns_own_session(negotiation_client):
    result = create_negotiation(negotiation_client)

    response = negotiation_client["client"].get(
        f"/api/v1/negotiations/{result['id']}"
    )

    assert response.status_code == 200
    assert response.json()["id"] == result["id"]


def test_other_user_cannot_access_negotiation(negotiation_client):
    result = create_negotiation(negotiation_client)
    client = negotiation_client["client"]
    client.headers.update(
        {"Authorization": f"Bearer {negotiation_client['tokens'][1]}"}
    )

    response = client.get(
        f"/api/v1/negotiations/{result['id']}"
    )

    assert response.status_code == 404


def test_accept_valid_offer(negotiation_client):
    result = create_negotiation(negotiation_client)
    offer_id = result["offers"][0]["id"]
    original_message = result["offers"][0]["message"]
    client = negotiation_client["client"]

    response = client.post(
        f"/api/v1/negotiations/{result['id']}/decision",
        json={
            "status": "ACCEPTED",
            "selected_offer_id": offer_id,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ACCEPTED"
    assert body["selected_offer_id"] == offer_id

    # Verify the offer message was not modified.
    updated = client.get(f"/api/v1/negotiations/{result['id']}")
    assert updated.status_code == 200
    accepted_offer = next(
        offer for offer in updated.json()["offers"]
        if offer["id"] == offer_id
    )
    assert accepted_offer["message"] == original_message
    assert "[ACCEPTED]" not in accepted_offer["message"]


def test_reject_negotiation(negotiation_client):
    result = create_negotiation(negotiation_client)

    response = negotiation_client["client"].post(
        f"/api/v1/negotiations/{result['id']}/decision",
        json={"status": "REJECTED"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "REJECTED"
    assert body["selected_offer_id"] is None


def test_cannot_accept_offer_from_another_session(negotiation_client):
    first = create_negotiation(negotiation_client)
    second = create_negotiation(negotiation_client)

    response = negotiation_client["client"].post(
        f"/api/v1/negotiations/{first['id']}/decision",
        json={
            "status": "ACCEPTED",
            "selected_offer_id": second["offers"][0]["id"],
        },
    )

    assert response.status_code == 422


def test_cannot_decide_negotiation_twice(negotiation_client):
    result = create_negotiation(negotiation_client)
    client = negotiation_client["client"]
    url = f"/api/v1/negotiations/{result['id']}/decision"

    first = client.post(url, json={"status": "REJECTED"})
    second = client.post(url, json={"status": "REJECTED"})

    assert first.status_code == 200
    assert second.status_code == 409


def test_nonexistent_negotiation_returns_404(negotiation_client):
    response = negotiation_client["client"].get(
        "/api/v1/negotiations/999999"
    )

    assert response.status_code == 404


def test_invalid_budget_is_rejected_by_api(negotiation_client):
    response = negotiation_client["client"].post(
        "/api/v1/negotiations",
        json={
            "service_id": negotiation_client["service_id"],
            "budget_limit": 0,
        },
    )

    assert response.status_code == 422