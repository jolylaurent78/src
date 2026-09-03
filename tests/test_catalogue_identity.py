import uuid

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_identity import (
    ApplicationContext,
    SystemCatalogueIdProvider,
    UserCatalogueIdProvider,
    is_catalogue_beacon_id,
    is_catalogue_city_id,
    is_catalogue_map_id,
    is_catalogue_template_id,
    is_catalogue_triangle_id,
    is_system_catalogue_id,
    is_user_catalogue_id,
    load_project_dotenv,
)


def _cities(catalogue: Catalogue):
    return (
        catalogue.add_city("Ouverture", 47.0, 2.0),
        catalogue.add_city("Base", 46.0, 3.0),
        catalogue.add_city("Lumiere", 48.0, 4.0),
    )


@pytest.mark.parametrize(
    ("environ", "mode", "provider_type"),
    [
        ({}, "USER", UserCatalogueIdProvider),
        ({"ASSEMBLEUR_MODE": " user "}, "USER", UserCatalogueIdProvider),
        ({"ASSEMBLEUR_MODE": "SYS"}, "SYS", SystemCatalogueIdProvider),
    ],
)
def test_application_context_resolves_mode_once(environ, mode, provider_type):
    context = ApplicationContext.from_environment(environ)
    assert context.mode == mode
    assert isinstance(context.catalogue_id_provider, provider_type)


def test_application_context_rejects_unknown_mode():
    with pytest.raises(ValueError, match="ASSEMBLEUR_MODE invalide"):
        ApplicationContext.from_environment({"ASSEMBLEUR_MODE": "FOO"})


def test_project_dotenv_is_optional_and_user_remains_the_default(tmp_path):
    environment = {}

    assert load_project_dotenv(environ=environment, env_path=tmp_path / ".env") is False
    context = ApplicationContext.from_environment(environment)

    assert context.mode == "USER"
    assert isinstance(context.catalogue_id_provider, UserCatalogueIdProvider)


@pytest.mark.parametrize(
    ("dotenv_mode", "expected_mode", "provider_type"),
    [
        ("SYS", "SYS", SystemCatalogueIdProvider),
        ("USER", "USER", UserCatalogueIdProvider),
    ],
)
def test_project_dotenv_supplies_the_mode_when_the_process_environment_is_absent(
    tmp_path, dotenv_mode, expected_mode, provider_type,
):
    env_path = tmp_path / ".env"
    env_path.write_text(f"ASSEMBLEUR_MODE={dotenv_mode}\n", encoding="utf-8")
    environment = {}

    assert load_project_dotenv(environ=environment, env_path=env_path) is True
    context = ApplicationContext.from_environment(environment)

    assert context.mode == expected_mode
    assert isinstance(context.catalogue_id_provider, provider_type)


@pytest.mark.parametrize(
    ("dotenv_mode", "environment_mode"),
    [("SYS", "USER"), ("USER", "SYS")],
)
def test_process_environment_has_priority_over_project_dotenv(tmp_path, dotenv_mode, environment_mode):
    env_path = tmp_path / ".env"
    env_path.write_text(f"ASSEMBLEUR_MODE={dotenv_mode}\n", encoding="utf-8")
    environment = {"ASSEMBLEUR_MODE": environment_mode}

    load_project_dotenv(environ=environment, env_path=env_path)

    assert ApplicationContext.from_environment(environment).mode == environment_mode


def test_invalid_project_dotenv_mode_keeps_the_existing_explicit_error(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("ASSEMBLEUR_MODE=TOTO\n", encoding="utf-8")
    environment = {}

    load_project_dotenv(environ=environment, env_path=env_path)

    with pytest.raises(ValueError, match="ASSEMBLEUR_MODE invalide"):
        ApplicationContext.from_environment(environment)


def test_system_ids_use_monotonic_counters_and_all_catalogue_kinds():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    opening, base, light = _cities(catalogue)
    beacon = catalogue.add_beacon(opening.city_id)
    triangle = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
    template = catalogue.add_template("Principal")

    assert (opening.city_id, base.city_id, light.city_id) == (
        "CITY-SYS-000001", "CITY-SYS-000002", "CITY-SYS-000003",
    )
    assert beacon.beacon_id == "BEA-SYS-000001"
    assert triangle.triangle_id == "TRI-SYS-000001"
    assert template.template_id == "TPL-SYS-000001"
    assert catalogue.id_counters == {"city": 3, "beacon": 1, "triangle": 1, "template": 1, "map": 0, "book": 0}

    catalogue.delete_triangle(triangle.triangle_id)
    catalogue.delete_city(light.city_id)
    replacement = catalogue.add_city("Remplacement", 45.0, 5.0)
    assert replacement.city_id == "CITY-SYS-000004"


def test_rejected_beacon_creation_does_not_consume_a_system_counter():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    with pytest.raises(ValueError, match="ville Catalogue introuvable"):
        catalogue.add_beacon("CITY-SYS-999999")
    assert catalogue.id_counters["beacon"] == 0

    first_city = catalogue.add_city("A", 47.0, 2.0)
    second_city = catalogue.add_city("B", 46.0, 3.0)
    first_beacon = catalogue.add_beacon(first_city.city_id)
    assert first_beacon.beacon_id == "BEA-SYS-000001"

    with pytest.raises(ValueError, match="d.j. une balise"):
        catalogue.add_beacon(first_city.city_id)
    assert catalogue.id_counters["beacon"] == 1
    assert catalogue.add_beacon(second_city.city_id).beacon_id == "BEA-SYS-000002"


def test_clone_keeps_provider_and_counters_without_mutating_source():
    system = Catalogue(id_provider=SystemCatalogueIdProvider())
    system.add_city("A", 47.0, 2.0)
    cloned_system = system.clone()
    assert cloned_system.add_city("B", 46.0, 3.0).city_id == "CITY-SYS-000002"
    assert system.id_counters["city"] == 1

    user = Catalogue(id_provider=UserCatalogueIdProvider())
    cloned_user = user.clone()
    created = cloned_user.add_city("A", 47.0, 2.0)
    assert is_user_catalogue_id(created.city_id)
    assert user.id_counters == {"city": 0, "beacon": 0, "triangle": 0, "template": 0, "map": 0, "book": 0}


def test_user_ids_are_uuid4_and_distinct_for_each_kind():
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    opening, base, light = _cities(catalogue)
    beacon = catalogue.add_beacon(opening.city_id)
    triangle = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
    template = catalogue.add_template("Principal")

    values = [opening.city_id, beacon.beacon_id, triangle.triangle_id, template.template_id]
    assert all(is_user_catalogue_id(value) for value in values)
    assert len(set(values)) == len(values)
    for value in values:
        assert uuid.UUID(value.split("-USR-", 1)[1]).version == 4


@pytest.mark.parametrize(
    ("value", "validator"),
    [
        ("CITY-SYS-000001", is_catalogue_city_id),
        ("BEA-SYS-1000000", is_catalogue_beacon_id),
        ("TRI-USR-550e8400-e29b-41d4-a716-446655440000", is_catalogue_triangle_id),
        ("TPL-USR-550e8400-e29b-41d4-a716-446655440000", is_catalogue_template_id),
        ("MAP-SYS-000001", is_catalogue_map_id),
        ("MAP-USR-550e8400-e29b-41d4-a716-446655440000", is_catalogue_map_id),
    ],
)
def test_catalogue_identity_validation_accepts_contract(value, validator):
    assert validator(value)


@pytest.mark.parametrize(
    "value",
    [
        "CITY-0001",
        "CITY-SYS-000000",
        "CITY-SYS-00001",
        "CITY-SYS-ABCDEF",
        "CITY-USR-550E8400-E29B-41D4-A716-446655440000",
        "CITY-USR-550e8400-e29b-11d4-a716-446655440000",
    ],
)
def test_catalogue_identity_validation_rejects_legacy_and_invalid_values(value):
    assert not is_catalogue_city_id(value)


def test_template_creation_always_allocates_a_new_identity_from_the_provider():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    source = catalogue.add_template("Source")
    duplicate = catalogue.add_template("Source - copie", description=source.description)
    assert source.template_id == "TPL-SYS-000001"
    assert duplicate.template_id == "TPL-SYS-000002"
    assert is_system_catalogue_id(duplicate.template_id)
