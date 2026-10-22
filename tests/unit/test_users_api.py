import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.backend.dependencies import get_db
from apps.backend.main import app
from database.models.entities import Base


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
        yield test_client

    app.dependency_overrides.clear()
    engine.dispose()


def register_user(client, email="student@example.com"):
    return client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "display_name": "Student",
            "password": "TestPassword-123!",
        },
    )


def login_user(client, email="student@example.com"):
    return client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": "TestPassword-123!",
        },
    )


def test_register_user_returns_201_without_password_hash(client):
    response = register_user(client)

    assert response.status_code == 201
    assert response.json()["email"] == "student@example.com"
    assert response.json()["display_name"] == "Student"
    assert "password_hash" not in response.json()
    assert "password" not in response.json()


def test_duplicate_email_returns_409(client):
    assert register_user(client).status_code == 201
    assert register_user(client).status_code == 409


def test_login_returns_bearer_token(client):
    register_user(client)
    response = login_user(client)

    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["access_token"]
    assert response.json()["expires_in"] > 0


def test_wrong_password_is_rejected(client):
    register_user(client)

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "student@example.com",
            "password": "WrongPassword-123!",
        },
    )

    assert response.status_code == 401


def test_profile_requires_authentication(client):
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401


def test_authenticated_user_can_get_own_profile(client):
    register_user(client)
    login = login_user(client)
    token = login.json()["access_token"]

    response = client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["email"] == "student@example.com"
    assert "password_hash" not in response.json()


def test_legacy_user_endpoints_are_not_exposed(client):
    assert client.post(
        "/api/v1/users",
        json={
            "email": "legacy@example.com",
            "display_name": "Legacy",
        },
    ).status_code == 404

    assert client.get("/api/v1/users").status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {
            "email": "invalid-email",
            "display_name": "Student",
            "password": "TestPassword-123!",
        },
        {
            "email": "student@example.com",
            "display_name": "",
            "password": "TestPassword-123!",
        },
        {
            "email": "student@example.com",
            "display_name": "A" * 101,
            "password": "TestPassword-123!",
        },
        {
            "email": "student@example.com",
            "display_name": "Student",
            "password": "short",
        },
    ],
)
def test_invalid_registration_payload_returns_422(client, payload):
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 422
