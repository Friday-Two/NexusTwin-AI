from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.backend.dependencies import get_db
from apps.backend.main import app
from database.models.entities import Base, Service


@pytest.fixture
def execution_client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/register", json={
            "email": "execution-test@example.com",
            "display_name": "Execution Test",
            "password": "TestPassword-123!",
        })
        assert response.status_code == 201
        login = client.post("/api/v1/auth/login", json={
            "email": "execution-test@example.com",
            "password": "TestPassword-123!",
        })
        assert login.status_code == 200
        client.headers["Authorization"] = "Bearer " + login.json()["access_token"]
        with factory() as db:
            service = Service(name="Test Hotel", category="Accommodation",
                              description="Local test service", active=True)
            db.add(service)
            db.commit()
            service_id = service.id
        yield {"client": client, "service_id": service_id}
    app.dependency_overrides.clear()
    engine.dispose()


def make_accepted_negotiation(context):
    client = context["client"]
    response = client.post("/api/v1/negotiations", json={
        "service_id": context["service_id"], "budget_limit": 1000, "max_offers": 3
    })
    assert response.status_code == 201
    negotiation = response.json()
    decision = client.post(f"/api/v1/negotiations/{negotiation['id']}/decision", json={
        "status": "ACCEPTED", "selected_offer_id": negotiation["offers"][0]["id"]
    })
    assert decision.status_code == 200
    return negotiation["id"]


def test_execution_requires_accepted_negotiation(execution_client):
    client = execution_client["client"]
    response = client.post("/api/v1/negotiations", json={
        "service_id": execution_client["service_id"], "budget_limit": 1000
    })
    assert response.status_code == 201
    response = client.post(f"/api/v1/negotiations/{response.json()['id']}/execute")
    assert response.status_code == 409


def test_execution_is_idempotent_and_verifiable(execution_client):
    client = execution_client["client"]
    session_id = make_accepted_negotiation(execution_client)
    first = client.post(f"/api/v1/negotiations/{session_id}/execute")
    assert first.status_code == 201
    assert first.json()["status"] == "SIMULATED_SUCCESS"
    assert '"external_side_effects": false' in first.json()["result_details"]
    second = client.post(f"/api/v1/negotiations/{session_id}/execute")
    assert second.status_code == 201
    assert second.json()["id"] == first.json()["id"]
    verification = client.post(f"/api/v1/executions/{first.json()['id']}/verify")
    assert verification.status_code == 200
    assert verification.json()["verified"] is True
    assert "local simulation only" in verification.json()["details"]


def test_user_cannot_verify_another_users_execution(execution_client):
    client = execution_client["client"]
    session_id = make_accepted_negotiation(execution_client)
    execution = client.post(f"/api/v1/negotiations/{session_id}/execute").json()
    other = client.post("/api/v1/auth/register", json={
        "email": "other-execution@example.com", "display_name": "Other User",
        "password": "TestPassword-123!",
    })
    assert other.status_code == 201
    login = client.post("/api/v1/auth/login", json={
        "email": "other-execution@example.com", "password": "TestPassword-123!",
    })
    client.headers["Authorization"] = "Bearer " + login.json()["access_token"]
    response = client.post(f"/api/v1/executions/{execution['id']}/verify")
    assert response.status_code == 404
