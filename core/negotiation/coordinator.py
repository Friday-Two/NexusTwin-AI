import math

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models.entities import (
    NegotiationOffer,
    NegotiationSession,
    Service,
    User,
)


def start_negotiation(
    db: Session,
    user: User,
    service_id: int,
    budget_limit: float,
    max_offers: int = 3,
) -> NegotiationSession:
    """Create a negotiation session and persist deterministic agent offers."""
    if not math.isfinite(budget_limit) or budget_limit <= 0:
        raise ValueError("Budget limit must be a finite number greater than zero")
    if not 1 <= max_offers <= 10:
        raise ValueError("max_offers must be between 1 and 10")

    service = db.scalar(
        select(Service).where(
            Service.id == service_id,
            Service.active.is_(True),
        )
    )

    if service is None:
        raise ValueError("Active service not found")

    session = NegotiationSession(
        user_id=user.id,
        service_id=service.id,
        status="NEGOTIATING",
        budget_limit=budget_limit,
    )

    try:
        db.add(session)
        db.flush()

        # Initial deterministic agents. These are simulated offers,
        # not live quotes from external service providers.
        offer_candidates = [
            (
                "BUDGET_AGENT",
                round(budget_limit * 0.70, 2),
                "Budget agent: cost-conscious offer.",
            ),
            (
                "SERVICE_AGENT",
                round(budget_limit * 0.85, 2),
                "Service agent: balanced offer.",
            ),
            (
                "POLICY_AGENT",
                round(budget_limit, 2),
                "Policy agent: offer within the stated budget.",
            ),
        ]

        for role, amount, message in offer_candidates[:max_offers]:
            db.add(
                NegotiationOffer(
                    session_id=session.id,
                    agent_role=role,
                    amount=amount,
                    message=message,
                )
            )

        db.flush()
        db.commit()
        db.refresh(session)

    except Exception:
        db.rollback()
        raise

    return session
