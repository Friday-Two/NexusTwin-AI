from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.backend.auth_dependencies import get_current_user
from apps.backend.dependencies import get_db
from apps.backend.schemas import (
    NegotiationCreate,
    NegotiationDecision,
    NegotiationOfferRead,
    NegotiationRead,
)
from core.negotiation.coordinator import start_negotiation
from database.models.entities import (
    NegotiationOffer,
    NegotiationSession,
    User,
)

router = APIRouter(
    prefix="/api/v1/negotiations",
    tags=["Negotiations"],
)


def _get_owned_session(
    db: Session,
    session_id: int,
    user_id: int,
) -> NegotiationSession:
    session = db.scalar(
        select(NegotiationSession).where(
            NegotiationSession.id == session_id,
            NegotiationSession.user_id == user_id,
        )
    )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Negotiation session not found",
        )

    return session


def _serialize_session(
    db: Session,
    session: NegotiationSession,
) -> NegotiationRead:
    offers = list(
        db.scalars(
            select(NegotiationOffer)
            .where(NegotiationOffer.session_id == session.id)
            .order_by(NegotiationOffer.amount.asc(), NegotiationOffer.id.asc())
        ).all()
    )

    return NegotiationRead(
        id=session.id,
        service_id=session.service_id,
        status=session.status,
        budget_limit=session.budget_limit,
        selected_offer_id=session.selected_offer_id,
        offers=[
            NegotiationOfferRead.model_validate(offer)
            for offer in offers
        ],
    )


@router.post(
    "",
    response_model=NegotiationRead,
    status_code=status.HTTP_201_CREATED,
)
def create_negotiation(
    payload: NegotiationCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> NegotiationRead:
    try:
        session = start_negotiation(
            db=db,
            user=current_user,
            service_id=payload.service_id,
            budget_limit=payload.budget_limit,
            max_offers=payload.max_offers,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return _serialize_session(db, session)


@router.get(
    "/{session_id}",
    response_model=NegotiationRead,
)
def get_negotiation(
    session_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> NegotiationRead:
    session = _get_owned_session(db, session_id, current_user.id)
    return _serialize_session(db, session)


@router.post(
    "/{session_id}/decision",
    response_model=NegotiationRead,
)
def decide_negotiation(
    session_id: int,
    payload: NegotiationDecision,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> NegotiationRead:
    session = _get_owned_session(db, session_id, current_user.id)

    if session.status != "NEGOTIATING":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This negotiation has already been decided",
        )

    if payload.status == "ACCEPTED":
        offer = db.scalar(
            select(NegotiationOffer).where(
                NegotiationOffer.id == payload.selected_offer_id,
                NegotiationOffer.session_id == session.id,
            )
        )

        if offer is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Selected offer does not belong to this negotiation",
            )

        if offer.amount > session.budget_limit:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Selected offer exceeds the negotiation budget",
            )

        session.selected_offer_id = offer.id
        session.status = "ACCEPTED"
    else:
        session.selected_offer_id = None
        session.status = "REJECTED"

    db.commit()
    db.refresh(session)

    return _serialize_session(db, session)