# import pytest
# from fastapi.testclient import TestClient
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.pool import StaticPool

# from apps.backend.dependencies import get_db
# from apps.backend.main import app
# from database.models.entities import Base


# @pytest.fixture
# def client():
#     engine = create_engine(
#         "sqlite://",
#         connect_args={"check_same_thread": False},
#         poolclass=StaticPool,
#     )
#     TestingSessionLocal = sessionmaker(
#         bind=engine,
#         autoflush=False,
#         expire_on_commit=False,
#     )
#     Base.metadata.create_all(bind=engine)

#     def override_get_db():
#         db = TestingSessionLocal()
#         try:
#             yield db
#         finally:
#             db.close()

#     app.dependency_overrides[get_db] = override_get_db

#     with TestClient(app) as test_client:
#         registration = test_client.post(
#             "/api/v1/auth/register",
#             json={
#                 "email": "preferences-test@example.com",
#                 "display_name": "Preferences Test",
#                 "password": "TestPassword-123!",
#             },
#         )
#         assert registration.status_code == 201

#         login = test_client.post(
#             "/api/v1/auth/login",
#             json={
#                 "email": "preferences-test@example.com",
#                 "password": "TestPassword-123!",
#             },
#         )
#         assert login.status_code == 200

#         test_client.headers.update(
#             {"Authorization": f"Bearer {login.json()['access_token']}"}
#         )
#         yield test_client

#     app.dependency_overrides.clear()
#     Base.metadata.drop_all(bind=engine)
#     engine.dispose()


# def test_create_preference(client):
#     response = client.put(
#         "/api/v1/users/1/preferences/travel_budget",
#         json={
#             "preference_key": "travel_budget",
#             "preference_value": "15000 INR",
#             "is_sensitive": False,
#         },
#     )
#     assert response.status_code == 200
#     assert response.json()["preference_value"] == "15000 INR"


# def test_update_existing_preference(client):
#     url = "/api/v1/users/1/preferences/travel_budget"
#     payload = {
#         "preference_key": "travel_budget",
#         "preference_value": "15000 INR",
#         "is_sensitive": False,
#     }
#     client.put(url, json=payload)

#     payload["preference_value"] = "20000 INR"
#     response = client.put(url, json=payload)

#     assert response.status_code == 200
#     assert response.json()["preference_value"] == "20000 INR"
#     assert response.json()["id"] == 1


# def test_sensitive_preference_hidden_by_default(client):
#     client.put(
#         "/api/v1/users/1/preferences/private_note",
#         json={
#             "preference_key": "private_note",
#             "preference_value": "Test-only private data",
#             "is_sensitive": True,
#         },
#     )

#     response = client.get("/api/v1/users/1/preferences")
#     assert response.status_code == 200
#     assert response.json() == []


# def test_sensitive_preference_included_when_requested(client):
#     client.put(
#         "/api/v1/users/1/preferences/private_note",
#         json={
#             "preference_key": "private_note",
#             "preference_value": "Test-only private data",
#             "is_sensitive": True,
#         },
#     )

#     response = client.get(
#         "/api/v1/users/1/preferences",
#         params={"include_sensitive": True},
#     )
#     assert response.status_code == 200
#     assert len(response.json()) == 1
#     assert response.json()[0]["is_sensitive"] is True


# def test_delete_preference(client):
#     url = "/api/v1/users/1/preferences/travel_budget"
#     client.put(
#         url,
#         json={
#             "preference_key": "travel_budget",
#             "preference_value": "15000 INR",
#             "is_sensitive": False,
#         },
#     )

#     response = client.delete(url)
#     assert response.status_code == 204
#     assert client.get("/api/v1/users/1/preferences").json() == []


# def test_other_user_access_is_forbidden(client):
#     response = client.get("/api/v1/users/999/preferences")
#     assert response.status_code == 403


# def test_preferences_require_authentication(client):
#     client.headers.pop("Authorization", None)

#     response = client.get("/api/v1/users/1/preferences")
#     assert response.status_code == 401


# def test_missing_token_cannot_delete_preference(client):
#     client.headers.pop("Authorization", None)

#     response = client.delete(
#         "/api/v1/users/1/preferences/travel_budget"
#     )
#     assert response.status_code == 401




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
    testing_session_local = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        registration = test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "preferences-test@example.com",
                "display_name": "Preferences Test",
                "password": "TestPassword-123!",
            },
        )
        assert registration.status_code == 201

        login = test_client.post(
            "/api/v1/auth/login",
            json={
                "email": "preferences-test@example.com",
                "password": "TestPassword-123!",
            },
        )
        assert login.status_code == 200

        test_client.headers.update(
            {"Authorization": f"Bearer {login.json()['access_token']}"}
        )
        yield test_client

    app.dependency_overrides.clear()
    engine.dispose()


def test_create_preference(client):
    response = client.put(
        "/api/v1/users/1/preferences/travel_budget",
        json={
            "preference_key": "travel_budget",
            "preference_value": "15000 INR",
            "is_sensitive": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["preference_value"] == "15000 INR"


def test_update_existing_preference(client):
    url = "/api/v1/users/1/preferences/travel_budget"
    payload = {
        "preference_key": "travel_budget",
        "preference_value": "15000 INR",
        "is_sensitive": False,
    }
    client.put(url, json=payload)

    payload["preference_value"] = "20000 INR"
    response = client.put(url, json=payload)

    assert response.status_code == 200
    assert response.json()["preference_value"] == "20000 INR"
    assert response.json()["id"] == 1


def test_sensitive_preference_hidden_by_default(client):
    client.put(
        "/api/v1/users/1/preferences/private_note",
        json={
            "preference_key": "private_note",
            "preference_value": "Test-only private data",
            "is_sensitive": True,
        },
    )

    response = client.get("/api/v1/users/1/preferences")
    assert response.status_code == 200
    assert response.json() == []


def test_sensitive_preference_included_when_requested(client):
    client.put(
        "/api/v1/users/1/preferences/private_note",
        json={
            "preference_key": "private_note",
            "preference_value": "Test-only private data",
            "is_sensitive": True,
        },
    )

    response = client.get(
        "/api/v1/users/1/preferences",
        params={"include_sensitive": True},
    )
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["is_sensitive"] is True


def test_delete_preference(client):
    url = "/api/v1/users/1/preferences/travel_budget"
    client.put(
        url,
        json={
            "preference_key": "travel_budget",
            "preference_value": "15000 INR",
            "is_sensitive": False,
        },
    )

    response = client.delete(url)
    assert response.status_code == 204
    assert client.get("/api/v1/users/1/preferences").json() == []


def test_other_user_access_is_forbidden(client):
    response = client.get("/api/v1/users/999/preferences")
    assert response.status_code == 403


def test_preferences_require_authentication(client):
    client.headers.pop("Authorization", None)

    response = client.get("/api/v1/users/1/preferences")
    assert response.status_code == 401


def test_missing_token_cannot_delete_preference(client):
    client.headers.pop("Authorization", None)

    response = client.delete(
        "/api/v1/users/1/preferences/travel_budget"
    )
    assert response.status_code == 401
