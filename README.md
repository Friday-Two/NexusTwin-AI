# NexusTwin AI

A privacy-aware, multi-agent digital twin platform for service discovery,
negotiation, authorized task execution, and independent outcome verification.

## Requirements

- Python 3.11
- Dependencies listed in requirements.txt

## Setup

Activate the virtual environment:

    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

## Run the backend

    python -m uvicorn apps.backend.main:app --reload

API documentation:

    http://127.0.0.1:8000/docs

Health endpoint:

    http://127.0.0.1:8000/health

## Run tests

    python -m pytest tests/unit -v

The current backend is a development baseline. No external bookings or
real-world transactions are performed.
