
import re

from database.models.entities import Service, UserPreference


def rank_services(
    services: list[Service],
    preferences: list[UserPreference],
) -> list[tuple[Service, float]]:
    """Rank services by overlap with non-sensitive preference terms."""
    preference_terms: set[str] = set()

    for preference in preferences:
        if preference.is_sensitive:
            continue

        value = preference.preference_value.strip().lower()
        preference_terms.update(
            term
            for term in re.findall(r"[a-z0-9]+", value)
            if len(term) >= 3
        )

    ranked: list[tuple[Service, float]] = []

    for service in services:
        searchable_text = (
            f"{service.name} {service.category} {service.description}"
        ).lower()

        matched_terms = {
            term for term in preference_terms if term in searchable_text
        }

        score = (
            len(matched_terms) / len(preference_terms)
            if preference_terms
            else 0.0
        )

        ranked.append((service, round(score, 4)))

    ranked.sort(key=lambda item: (-item[1], item[0].name.lower(), item[0].id))
    return ranked
