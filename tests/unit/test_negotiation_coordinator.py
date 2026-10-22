
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from core.negotiation.coordinator import start_negotiation
from database.models.entities import (
    Base,
    NegotiationOffer,
    NegotiationSession,
    Service,
    User,
)


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    with Session(engine, expire_on_commit=False) as session:
        user = User(
            email="negotiation-test@example.com",
            display_name="Negotiation Test",
        )
        service = Service(
            name="Budget Hotel",
            category="Accommodation",
            description="Affordable hotel",
            active=True,
        )
        inactive_service = Service(
            name="Closed Hotel",
            category="Accommodation",
            description="Unavailable hotel",
            active=False,
        )
        session.add_all([user, service, inactive_service])
        session.commit()
        session.refresh(user)
        session.refresh(service)
        session.refresh(inactive_service)

        session.info["test_user_id"] = user.id
        session.info["active_service_id"] = service.id
        session.info["inactive_service_id"] = inactive_service.id

        yield session
    engine.dispose()


def test_creates_session_and_offers(db):
    user = db.get(User, db.info["test_user_id"])

    result = start_negotiation(
        db=db,
        user=user,
        service_id=db.info["active_service_id"],
        budget_limit=1000,
    )

    offers = list(
        db.scalars(
            select(NegotiationOffer).where(
                NegotiationOffer.session_id == result.id
            )
        ).all()
    )

    assert result.id is not None
    assert result.user_id == user.id
    assert result.status == "NEGOTIATING"
    assert len(offers) == 3
    assert all(offer.amount <= result.budget_limit for offer in offers)


def test_respects_max_offers(db):
    user = db.get(User, db.info["test_user_id"])

    result = start_negotiation(
        db=db,
        user=user,
        service_id=db.info["active_service_id"],
        budget_limit=1000,
        max_offers=2,
    )

    offers = list(
        db.scalars(
            select(NegotiationOffer).where(
                NegotiationOffer.session_id == result.id
            )
        ).all()
    )

    assert len(offers) == 2


@pytest.mark.parametrize("budget", [0, -10, float("inf"), float("nan")])
def test_rejects_invalid_budget(db, budget):
    user = db.get(User, db.info["test_user_id"])

    with pytest.raises(ValueError, match="Budget limit"):
        start_negotiation(
            db=db,
            user=user,
            service_id=db.info["active_service_id"],
            budget_limit=budget,
        )

    assert db.scalar(select(NegotiationSession)) is None


def test_rejects_inactive_service(db):
    user = db.get(User, db.info["test_user_id"])

    with pytest.raises(ValueError, match="Active service not found"):
        start_negotiation(
            db=db,
            user=user,
            service_id=db.info["inactive_service_id"],
            budget_limit=1000,
        )

    assert db.scalar(select(NegotiationSession)) is None


def test_rejects_nonexistent_service(db):
    user = db.get(User, db.info["test_user_id"])

    with pytest.raises(ValueError, match="Active service not found"):
        start_negotiation(
            db=db,
            user=user,
            service_id=99999,
            budget_limit=1000,
        )

    assert db.scalar(select(NegotiationSession)) is None


@pytest.mark.parametrize("max_offers", [0, -1, 11])
def test_rejects_invalid_offer_count(db, max_offers):
    user = db.get(User, db.info["test_user_id"])

    with pytest.raises(ValueError, match="max_offers"):
        start_negotiation(
            db=db,
            user=user,
            service_id=db.info["active_service_id"],
            budget_limit=1000,
            max_offers=max_offers,
        )

    assert db.scalar(select(NegotiationSession)) is None
