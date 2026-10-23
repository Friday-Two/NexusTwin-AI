"""Experimental geographic-query reranking; the base ranker is unchanged."""

def is_geographic_query(query: str) -> bool:
    query = query.lower()
    return any(term in query for term in (
        "nearby", "geographic coordinates", "coordinates",
        "near me", "nearby places", "places near",
    ))


def geographic_adjustment(item: dict) -> float:
    text = (
        str(item.get("description", "")) + " " +
        str(item.get("api_text", ""))
    ).lower()

    # Suppress bonuses for helper operations rather than place searches.
    helper_terms = (
        "categories of establishments",
        "place details",
        "list of establishment types",
        "get a list of place types",
        "geocoding",
        "coordinates by placename",
        "coordinates to address",
        "random place",
        "random (google)",
        "air quality",
    )
    if any(term in text for term in helper_terms):
        return 0.0

    strong_terms = (
        "search businesses near",
        "search nearby places",
        "search for nearby places",
        "returns nearby places sorted by distance",
        "find places nearby",
        "nearby search",
    )
    if any(term in text for term in strong_terms):
        return 0.30

    if any(term in text for term in (
        "nearby places", "search places", "places in radius",
    )):
        return 0.10

    return 0.0


def rerank(query: str, results: list[dict]) -> list[dict]:
    """Return a new sorted list; preserve the original results and scores."""
    if not is_geographic_query(query):
        return list(results)

    adjusted = []
    for item in results:
        row = dict(item)
        row["_geo_adjustment"] = geographic_adjustment(item)
        row["_geo_adjusted_score"] = (
            float(item.get("ranking_score", 0.0)) + row["_geo_adjustment"]
        )
        adjusted.append(row)

    adjusted.sort(key=lambda item: item["_geo_adjusted_score"], reverse=True)

    # Keep the public response shape unchanged.
    for row in adjusted:
        row.pop("_geo_adjustment", None)
        row.pop("_geo_adjusted_score", None)

    return adjusted
