import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from database.models.entities import Base, User, UserPreference


@pytest.fixture
def test_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def test_create_user(test_engine):
    with Session(test_engine) as session:
        user = User(
            email="student@example.com",
            display_name="Test Student",
        )
        session.add(user)
        session.commit()

        saved_user = session.scalar(
            select(User).where(
                User.email == "student@example.com"
            )
        )

        assert saved_user is not None
        assert saved_user.display_name == "Test Student"


def test_save_user_preference(test_engine):
    with Session(test_engine) as session:
        user = User(
            email="preferences@example.com",
            display_name="Preference Tester",
        )
        session.add(user)
        session.flush()

        preference = UserPreference(
            user_id=user.id,
            preference_key="currency",
            preference_value="INR",
            is_sensitive=False,
        )
        session.add(preference)
        session.commit()

        saved = session.scalar(
            select(UserPreference).where(
                UserPreference.user_id == user.id
            )
        )

        assert saved is not None
        assert saved.preference_key == "currency"
        assert saved.preference_value == "INR"


def test_failed_transaction_rolls_back(test_engine):
    with (
        pytest.raises(RuntimeError),
        Session(test_engine) as session,
        session.begin(),
    ):
        session.add(
            User(
                email="rollback@example.com",
                display_name="Rollback Tester",
            )
        )
        session.flush()
        raise RuntimeError("Simulated failure")

    with Session(test_engine) as session:
        saved = session.scalar(
            select(User).where(
                User.email == "rollback@example.com"
            )
        )
        assert saved is None