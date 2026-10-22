# NexusTwin AI

A privacy-aware, multi-agent digital twin platform for service discovery, preference-aware ranking, offer negotiation, explicitly authorized task execution, and independent outcome verification.

## Current capabilities

- FastAPI backend with OpenAPI documentation.
- Email/password registration and JWT bearer authentication.
- User-owned preferences; sensitive preferences are excluded from ranking by default.
- Active service discovery and preference relevance ranking.
- Simulated agent offers, explicit accept/reject decisions, and budget checks.
- Explicitly triggered simulated execution with owner checks and idempotent records.
- Independent verification of persisted execution facts.
- SQLAlchemy models, Alembic migrations, unit tests, and GitHub Actions CI.

> Safety boundary: execution is currently a local simulation. No external bookings, payments, or provider actions are performed. Verification confirms local data consistency only.

## Requirements

- Python 3.11
- Dependencies in requirements.txt

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set a strong, unique JWT_SECRET_KEY in .env before running. Never commit .env or real credentials. The example secret is only a placeholder.

## Database and backend

```powershell
python -m alembic upgrade head
python -m uvicorn apps.backend.main:app --reload
```

- API root: http://127.0.0.1:8000/
- Health: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

## Quality checks

```powershell
python -m ruff check .
python -m pytest tests/unit -v
python -m alembic check
```

See docs/EXECUTION_AND_VERIFICATION.md for the full execution workflow.
