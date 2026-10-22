from typing import Annotated

from fastapi import APIRouter, Depends

from apps.backend.auth_dependencies import get_current_user
from apps.backend.schemas import UserRead
from database.models.entities import User

router = APIRouter(prefix="/api/v1/users", tags=["Users"])


@router.get("/me", response_model=UserRead)
def get_my_profile(
    current_user: Annotated[User, Depends(get_current_user)],
):
    return current_user
