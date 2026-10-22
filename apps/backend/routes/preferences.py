from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.backend.auth_dependencies import get_current_user
from apps.backend.dependencies import get_db
from apps.backend.schemas import PreferenceCreate, PreferenceRead
from database.models.entities import User, UserPreference

router = APIRouter(
    prefix="/api/v1/users/{user_id}/preferences",
    tags=["Preferences"],
)


def verify_owner(user_id: int, current_user: User) -> None:
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot access another user's preferences.",
        )


@router.put("/{preference_key}", response_model=PreferenceRead)
def upsert_preference(
    user_id: int,
    preference_key: str,
    payload: PreferenceCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    verify_owner(user_id, current_user)

    if payload.preference_key != preference_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Preference key in the URL and body must match.",
        )

    preference = db.scalar(
        select(UserPreference).where(
            UserPreference.user_id == user_id,
            UserPreference.preference_key == preference_key,
        )
    )

    if preference is None:
        preference = UserPreference(
            user_id=user_id,
            preference_key=preference_key,
            preference_value=payload.preference_value,
            is_sensitive=payload.is_sensitive,
        )
        db.add(preference)
    else:
        preference.preference_value = payload.preference_value
        preference.is_sensitive = payload.is_sensitive

    db.commit()
    db.refresh(preference)
    return preference


@router.get("", response_model=list[PreferenceRead])
def list_preferences(
    user_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    include_sensitive: Annotated[bool, Query()] = False,
):
    verify_owner(user_id, current_user)

    query = select(UserPreference).where(
        UserPreference.user_id == user_id
    )

    if not include_sensitive:
        query = query.where(UserPreference.is_sensitive.is_(False))

    return db.scalars(
        query.order_by(UserPreference.preference_key)
    ).all()


@router.delete(
    "/{preference_key}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_preference(
    user_id: int,
    preference_key: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    verify_owner(user_id, current_user)

    preference = db.scalar(
        select(UserPreference).where(
            UserPreference.user_id == user_id,
            UserPreference.preference_key == preference_key,
        )
    )

    if preference is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preference not found.",
        )

    db.delete(preference)
    db.commit()