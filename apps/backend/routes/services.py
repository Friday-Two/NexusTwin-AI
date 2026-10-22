
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from apps.backend.auth_dependencies import get_current_user
from apps.backend.dependencies import get_db
from apps.backend.schemas import RankedServiceRead
from core.negotiation.service_ranker import rank_services
from database.models.entities import Service, User, UserPreference

router = APIRouter(prefix="/api/v1/services", tags=["Service Discovery"])


@router.get("", response_model=list[RankedServiceRead])
def discover_services(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    query: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    category: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict]:
    statement = select(Service).where(Service.active.is_(True))

    if query:
        keyword = query.strip()
        if keyword:
            pattern = f"%{keyword}%"
            statement = statement.where(
                or_(
                    Service.name.ilike(pattern),
                    Service.description.ilike(pattern),
                    Service.category.ilike(pattern),
                )
            )

    if category:
        statement = statement.where(
            Service.category.ilike(category.strip())
        )

    services = list(
        db.scalars(statement.order_by(Service.name.asc())).all()
    )

    preferences = list(
        db.scalars(
            select(UserPreference).where(
                UserPreference.user_id == current_user.id,
                UserPreference.is_sensitive.is_(False),
            )
        ).all()
    )

    ranked = rank_services(services, preferences)[:limit]

    return [
        {
            "id": service.id,
            "name": service.name,
            "category": service.category,
            "description": service.description,
            "active": service.active,
            "relevance_score": score,
        }
        for service, score in ranked
    ]
