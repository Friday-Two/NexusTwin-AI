
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.backend.dependencies import get_db
from apps.backend.main import app
from database.models.entities import Base, Service


@pytest.fixture
def client():
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

    with TestClient(app) as test_client:
        registration = test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "services-test@example.com",
                "display_name": "Services Test",
                "password": "TestPassword-123!",
            },
        )
        assert registration.status_code == 201

        login = test_client.post(
            "/api/v1/auth/login",
            json={
                "email": "services-test@example.com",
                "password": "TestPassword-123!",
            },
        )
        assert login.status_code == 200

        test_client.headers.update(
            {"Authorization": f"Bearer {login.json()['access_token']}"}
        )

        db = TestingSessionLocal()
        try:
            db.add_all(
                [
                    Service(
                        name="Budget Hotel",
                        category="Accommodation",
                        description="Affordable hotel near the city center",
                        active=True,
                    ),
                    Service(
                        name="Luxury Hotel",
                        category="Accommodation",
                        description="Premium accommodation with spa",
                        active=True,
                    ),
                    Service(
                        name="Airport Taxi",
                        category="Transport",
                        description="Affordable airport transportation",
                        active=True,
                    ),
                    Service(
                        name="Inactive Hotel",
                        category="Accommodation",
                        description="This service is unavailable",
                        active=False,
                    ),
                ]
            )
            db.commit()
        finally:
            db.close()

        yield test_client

    app.dependency_overrides.clear()
    engine.dispose()


def test_discover_services_returns_active_services(client):
    response = client.get("/api/v1/services")

    assert response.status_code == 200
    names = {service["name"] for service in response.json()}
    assert names == {"Budget Hotel", "Luxury Hotel", "Airport Taxi"}


def test_discover_services_searches_keyword(client):
    response = client.get(
        "/api/v1/services",
        params={"query": "airport"},
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["name"] == "Airport Taxi"


def test_discover_services_filters_by_category(client):
    response = client.get(
        "/api/v1/services",
        params={"category": "accommodation"},
    )

    assert response.status_code == 200
    names = {service["name"] for service in response.json()}
    assert names == {"Budget Hotel", "Luxury Hotel"}


def test_discover_services_respects_limit(client):
    response = client.get(
        "/api/v1/services",
        params={"limit": 1},
    )

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_discover_services_rejects_invalid_limit(client):
    response = client.get(
        "/api/v1/services",
        params={"limit": 101},
    )

    assert response.status_code == 422


def test_discover_services_requires_authentication(client):
    client.headers.pop("Authorization", None)

    response = client.get("/api/v1/services")

    assert response.status_code == 401
