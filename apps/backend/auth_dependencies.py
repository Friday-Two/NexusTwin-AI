from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.backend.dependencies import get_db
from configs.settings import get_settings
from database.models.entities import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    settings = get_settings()

    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp", "iat"]},
        )
        subject = payload.get("sub")

        if not isinstance(subject, str) or not subject.isdigit():
            raise credentials_error

        user = db.scalar(
            select(User).where(User.id == int(subject))
        )
        if user is None:
            raise credentials_error

        return user

    except jwt.InvalidTokenError:
        raise credentials_error
