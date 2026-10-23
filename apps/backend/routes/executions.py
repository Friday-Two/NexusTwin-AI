"""Explicitly authorized simulated execution and outcome verification endpoints."""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.backend.auth_dependencies import get_current_user
from apps.backend.dependencies import get_db
from core.execution.runner import execute_accepted_session, verify_execution
from database.models.entities import (
    ExecutionRecord,
    NegotiationSession,
    User,
    VerificationResult,
)

router = APIRouter(prefix="/api/v1", tags=["Execution & Verification"])


class ExecutionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    session_id: int
    status: str
    external_reference: str | None
    result_details: str


class VerificationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    execution_id: int
    verified: bool
    details: str


def owned_session(db: Session, session_id: int, user_id: int) -> NegotiationSession:
    item = db.scalar(select(NegotiationSession).where(
        NegotiationSession.id == session_id, NegotiationSession.user_id == user_id
    ))
    if item is None:
        raise HTTPException(status_code=404, detail="Negotiation session not found.")
    return item


def owned_execution(db: Session, execution_id: int, user_id: int) -> ExecutionRecord:
    item = db.scalar(select(ExecutionRecord).join(
        NegotiationSession, ExecutionRecord.session_id == NegotiationSession.id
    ).where(
        ExecutionRecord.id == execution_id, NegotiationSession.user_id == user_id
    ))
    if item is None:
        raise HTTPException(status_code=404, detail="Execution record not found.")
    return item


@router.post("/negotiations/{session_id}/execute", response_model=ExecutionRead,
             status_code=status.HTTP_201_CREATED)
def execute_negotiation(session_id: int, db: Annotated[Session, Depends(get_db)],
                        current_user: Annotated[User, Depends(get_current_user)]) -> ExecutionRead:
    session = owned_session(db, session_id, current_user.id)
    try:
        record = execute_accepted_session(db, session)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ExecutionRead.model_validate(record)


@router.post("/executions/{execution_id}/verify", response_model=VerificationRead)
def verify_endpoint(execution_id: int, db: Annotated[Session, Depends(get_db)],
                    current_user: Annotated[User, Depends(get_current_user)]) -> VerificationRead:
    record = owned_execution(db, execution_id, current_user.id)
    return VerificationRead.model_validate(verify_execution(db, record))


@router.get("/executions/{execution_id}/verification", response_model=VerificationRead)
def get_verification(execution_id: int, db: Annotated[Session, Depends(get_db)],
                     current_user: Annotated[User, Depends(get_current_user)]) -> VerificationRead:
    record = owned_execution(db, execution_id, current_user.id)
    result = db.scalar(select(VerificationResult).where(
        VerificationResult.execution_id == record.id
    ).order_by(VerificationResult.id.desc()))
    if result is None:
        raise HTTPException(status_code=404, detail="Verification result not found.")
    return VerificationRead.model_validate(result)
