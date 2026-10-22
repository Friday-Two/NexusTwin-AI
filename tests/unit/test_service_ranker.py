
from core.negotiation.service_ranker import rank_services
from database.models.entities import Service, UserPreference


def make_service(
    service_id: int,
    name: str,
    category: str,
    description: str = "",
) -> Service:
    return Service(
        id=service_id,
        name=name,
        category=category,
        description=description,
        active=True,
    )


def make_preference(
    key: str,
    value: str,
    sensitive: bool = False,
) -> UserPreference:
    return UserPreference(
        preference_key=key,
        preference_value=value,
        is_sensitive=sensitive,
    )


def test_matching_preference_ranks_service_higher():
    budget_hotel = make_service(
        1, "Budget Hotel", "Accommodation", "Affordable rooms"
    )
    luxury_hotel = make_service(
        2, "Luxury Hotel", "Accommodation", "Premium rooms"
    )

    ranked = rank_services(
        [luxury_hotel, budget_hotel],
        [make_preference("travel_style", "budget affordable")],
    )

    assert ranked[0][0].name == "Budget Hotel"
    assert ranked[0][1] > ranked[1][1]


def test_sensitive_preferences_do_not_affect_ranking():
    hotel = make_service(1, "City Hotel", "Accommodation")
    taxi = make_service(2, "City Taxi", "Transport")

    ranked = rank_services(
        [taxi, hotel],
        [make_preference("private_note", "accommodation", sensitive=True)],
    )

    # With no usable terms, ties are resolved by service name.
    assert [service.name for service, _ in ranked] == [
        "City Hotel",
        "City Taxi",
    ]
    assert all(score == 0.0 for _, score in ranked)


def test_repeated_terms_are_counted_once():
    hotel = make_service(1, "Budget Hotel", "Accommodation")
    taxi = make_service(2, "Airport Taxi", "Transport")

    ranked = rank_services(
        [taxi, hotel],
        [make_preference("travel", "budget budget budget")],
    )

    assert ranked[0][0].name == "Budget Hotel"
    assert ranked[0][1] == 1.0
    assert ranked[1][1] == 0.0


def test_no_preferences_produce_zero_scores():
    services = [
        make_service(2, "Zebra Taxi", "Transport"),
        make_service(1, "Alpha Hotel", "Accommodation"),
    ]

    ranked = rank_services(services, [])

    assert [service.name for service, _ in ranked] == [
        "Alpha Hotel",
        "Zebra Taxi",
    ]
    assert all(score == 0.0 for _, score in ranked)


def test_matching_is_case_insensitive():
    hotel = make_service(1, "BUDGET HOTEL", "Accommodation")
    ranked = rank_services(
        [hotel],
        [make_preference("travel", "budget")],
    )

    assert ranked[0][1] == 1.0
