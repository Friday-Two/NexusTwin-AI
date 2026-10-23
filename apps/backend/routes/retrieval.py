import json
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from apps.backend.auth_dependencies import get_current_user
from configs.settings import get_settings
from core.retrieval.geographic_reranker import rerank as geographic_rerank
from core.retrieval.toolbench_ranker import get_retriever

router = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["API Retrieval"],
)


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedAPI(BaseModel):
    document_id: str
    description: str
    api_text: str
    ranking_score: float


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievedAPI]


def is_car_rental_search(query: str) -> bool:
    query = query.lower()
    rental_terms = (
        "car rental",
        "car rentals",
        "rental car",
        "rental cars",
        "rent a car",
        "car hire",
        "vehicle rental",
        "vehicle hire",
    )
    return any(term in query for term in rental_terms)


def supports_car_rental_search(result: dict) -> bool:
    """Require documented vehicle-rental search capability."""
    raw = str(result.get("description", ""))
    api_text = str(result.get("api_text", ""))
    combined = f"{raw} {api_text}".lower()

    # Prefer structured API metadata when available.
    try:
        metadata = json.loads(raw)
        if isinstance(metadata, dict):
            name = str(metadata.get("api_name", "")).lower()
            description = str(metadata.get("api_description", "")).lower()
        else:
            name, description = "", combined
    except (ValueError, TypeError):
        name, description = "", combined

    operation_text = f"{name} {description}"

    rental_terms = (
        "car rental",
        "car rentals",
        "rental car",
        "rental cars",
        "rent a car",
        "car hire",
        "vehicle rental",
        "vehicle hire",
    )
    has_rental_capability = any(
        term in operation_text for term in rental_terms
    )

    search_actions = (
        "search",
        "find",
        "list",
        "availability",
        "available",
        "inventory",
        "searching",
    )
    has_search_operation = any(
        re.search(rf"\b{re.escape(action)}\b", operation_text)
        for action in search_actions
    )

    # Reject property-rental and generic market/reference endpoints.
    property_terms = (
        "real estate",
        "property",
        "properties",
        "apartments",
        "zillow",
        "realtor",
        "trulia",
    )
    is_property_api = any(term in combined for term in property_terms)
    is_market_lookup = (
        "markets" in name and "market countries" in description
    )

    return (
        has_rental_capability
        and has_search_operation
        and not is_property_api
        and not is_market_lookup
    )


@router.post("/rank", response_model=RetrievalResponse)
def rank_apis(
    payload: RetrievalRequest,
    current_user: Annotated[object, Depends(get_current_user)],
) -> RetrievalResponse:
    """Rank APIs and reject unsupported car-rental search matches."""
    try:
        candidate_k = (
            min(20, max(payload.top_k, payload.top_k * 4))
            if get_settings().enable_geo_reranking
            else payload.top_k
        )

        results = get_retriever().rank(
            payload.query,
            top_k=candidate_k,
        )

        if get_settings().enable_geo_reranking:
            results = geographic_rerank(payload.query, results)

        if is_car_rental_search(payload.query):
            results = [
                result for result in results if supports_car_rental_search(result)
            ]

        results = results[: payload.top_k]

        return RetrievalResponse(
            query=payload.query,
            results=results,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="API retrieval model files are missing.",
        ) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="API retrieval is temporarily unavailable.",
        ) from exc