"""Local-only execution simulator and independent verifier."""
import json
from sqlalchemy import select
from sqlalchemy.orm import Session
from database.models.entities import ExecutionRecord, NegotiationOffer, NegotiationSession, Service, VerificationResult


def execute_accepted_session(db: Session, session: NegotiationSession) -> ExecutionRecord:
    """Record a simulated execution; never contact external providers or take payment."""
    if session.status != "ACCEPTED" or session.selected_offer_id is None:
        raise ValueError("Only an accepted negotiation with a selected offer can execute")
    existing = db.scalar(select(ExecutionRecord).where(ExecutionRecord.session_id == session.id))
    if existing is not None:
        return existing
    offer = db.scalar(select(NegotiationOffer).where(
        NegotiationOffer.id == session.selected_offer_id,
        NegotiationOffer.session_id == session.id,
    ))
    service = db.scalar(select(Service).where(Service.id == session.service_id))
    if offer is None or service is None:
        raise ValueError("The accepted offer or associated service could not be found")
    if offer.amount <= 0 or offer.amount > session.budget_limit:
        raise ValueError("Selected offer failed the budget safety check")
    record = ExecutionRecord(
        session_id=session.id,
        status="SIMULATED_SUCCESS",
        external_reference=None,
        result_details=json.dumps({
            "mode": "simulation", "service_id": service.id,
            "service_name": service.name, "selected_offer_id": offer.id,
            "amount": offer.amount, "currency": "UNSPECIFIED",
            "external_side_effects": False,
        }, sort_keys=True),
    )
    try:
        db.add(record)
        db.flush()
        db.add(VerificationResult(execution_id=record.id, verified=False, details="Verification pending."))
        db.commit()
        db.refresh(record)
    except Exception:
        db.rollback()
        raise
    return record


def verify_execution(db: Session, record: ExecutionRecord) -> VerificationResult:
    """Verify persisted facts against the accepted offer, budget, and execution payload."""
    session = db.scalar(select(NegotiationSession).where(NegotiationSession.id == record.session_id))
    issues: list[str] = []
    verified = True
    if session is None:
        verified = False
        issues.append("Negotiation session does not exist.")
    else:
        if session.status != "ACCEPTED":
            verified = False
            issues.append("Negotiation is not in ACCEPTED state.")
        offer = None
        if session.selected_offer_id is None:
            verified = False
            issues.append("No selected offer is recorded.")
        else:
            offer = db.scalar(select(NegotiationOffer).where(
                NegotiationOffer.id == session.selected_offer_id,
                NegotiationOffer.session_id == session.id,
            ))
            if offer is None:
                verified = False
                issues.append("Selected offer does not belong to the session.")
            elif offer.amount <= 0 or offer.amount > session.budget_limit:
                verified = False
                issues.append("Selected offer violates the budget limit.")
        try:
            payload = json.loads(record.result_details)
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if offer is not None:
            if payload.get("selected_offer_id") != offer.id:
                verified = False
                issues.append("Execution offer does not match the accepted offer.")
            if payload.get("amount") != offer.amount:
                verified = False
                issues.append("Execution amount does not match the accepted offer.")
        if record.status != "SIMULATED_SUCCESS":
            verified = False
            issues.append("Execution status is not a successful simulation.")
    result = db.scalar(select(VerificationResult).where(
        VerificationResult.execution_id == record.id
    ).order_by(VerificationResult.id.desc()))
    message = " ".join(issues) if issues else (
        "Persisted session, selected offer, and amount are consistent. "
        "This verifies a local simulation only, not an external transaction."
    )
    if result is None:
        result = VerificationResult(execution_id=record.id, verified=verified, details=message)
        db.add(result)
    else:
        result.verified = verified
        result.details = message
    db.commit()
    db.refresh(result)
    return result
