# Execution and verification

This implementation is a local-only simulation. It does not contact service providers,
book anything, transfer money, or claim an external transaction occurred.

## Flow

1. Register and log in through the authentication endpoints.
2. Discover a service and create a negotiation.
3. Accept a valid offer with POST /api/v1/negotiations/{id}/decision.
4. Explicitly execute the accepted negotiation with POST /api/v1/negotiations/{id}/execute.
5. Independently verify the execution with POST /api/v1/executions/{id}/verify.
6. Read the stored result with GET /api/v1/executions/{id}/verification.

All endpoints require a bearer token and only return records belonging to the signed-in user.
Execution requires ACCEPTED status and a selected offer inside the budget. Repeated execution
returns the existing execution record rather than creating a duplicate simulation record.
Verification re-reads the session and offer and compares the selected offer ID, amount, budget,
and execution status against the stored result. Passing verification proves only local data
consistency, not that a real-world booking or payment occurred.

## Local startup

```powershell
python -m pip install -r requirements.txt
python -m alembic upgrade head
python -m uvicorn apps.backend.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API.
