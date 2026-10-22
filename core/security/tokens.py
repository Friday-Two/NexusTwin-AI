from datetime import UTC, datetime, timedelta

import jwt

from configs.settings import get_settings


def create_access_token(subject: str | int) -> str:
    """Create a signed JWT access token for a user."""
    settings = get_settings()
    now = datetime.now(UTC)

    payload = {
        "sub": str(subject),
        "iat": now,
        "exp": now + timedelta(
            minutes=settings.access_token_expire_minutes
        ),
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict:
    """Validate and decode a JWT access token."""
    settings = get_settings()

    payload = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
        options={"require": ["sub", "exp", "iat"]},
    )

    if not isinstance(payload.get("sub"), str) or not payload["sub"]:
        raise jwt.InvalidTokenError("Invalid token subject")

    return payload
