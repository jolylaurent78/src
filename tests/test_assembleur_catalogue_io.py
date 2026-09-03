import json

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_identity import (
    SystemCatalogueIdProvider,
    UserCatalogueIdProvider,
    is_user_catalogue_id,
)
from src.assembleur_catalogue_io import catalogue_from_dict, catalogue_to_dict, load_catalogue, save_catalogue


def _catalogue() -> Catalogue:
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    opening = catalogue.add_city("Bourges", 47.081, 2.398)
    base = catalogue.add_city("Rocamadour", 44.799, 1.619)
    light = catalogue.add_city("Loches", 47.128, 0.995)
    triangle = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
    template = catalogue.add_template("Ordre principal", "Description")
    catalogue.set_template_rank(template.template_id, 1, triangle.triangle_id)
    return catalogue


def test_v5_round_trip_keeps_ids_references_counters_and_injected_provider(tmp_path):
    catalogue = _catalogue()
    city_ids = list(catalogue.cities)
    beacon_one = catalogue.add_beacon(city_ids[0])
    beacon_two = catalogue.add_beacon(city_ids[1])
    catalogue.update_beacon(beacon_two.beacon_id, archived=True)
    catalogue.update_city(city_ids[1], archived=True)
    triangle_id = next(iter(catalogue.triangles))
    catalogue.update_triangle(triangle_id, archived=True)
    catalogue.get_city_lambert(city_ids[0])
    path = tmp_path / "nested" / "catalogue.json"
    user_provider = UserCatalogueIdProvider()

    save_catalogue(catalogue, path)
    serialized = json.loads(path.read_text(encoding="utf-8"))
    loaded = load_catalogue(path, id_provider=user_provider)

    assert serialized["version"] == 5
    assert serialized["idCounters"] == {"city": 3, "beacon": 2, "triangle": 1, "template": 1, "map": 0, "book": 0}
    assert list(serialized["idCounters"]) == ["city", "beacon", "triangle", "template", "map", "book"]
    assert set(loaded.cities) == set(catalogue.cities)
    assert set(loaded.beacons) == set(catalogue.beacons)
    assert set(loaded.triangles) == set(catalogue.triangles)
    assert set(loaded.templates) == set(catalogue.templates)
    assert loaded.default_template_id == catalogue.default_template_id
    assert loaded.catalogue_reference_map_id == catalogue.catalogue_reference_map_id
    assert loaded.id_counters == catalogue.id_counters
    assert loaded.id_provider is user_provider
    assert loaded.get_beacon(beacon_one.beacon_id).city_id == city_ids[0]
    assert loaded.get_beacon(beacon_two.beacon_id).archived is True
    assert loaded._city_lambert_cache == {}
    created = loaded.add_city("Ville utilisateur", 42.0, 0.0)
    assert is_user_catalogue_id(created.city_id)
    assert loaded.id_counters == catalogue.id_counters
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("value", [True, 1.0, "1", -1])
def test_id_counters_reject_invalid_value_types(value):
    data = catalogue_to_dict(_catalogue())
    data["idCounters"]["city"] = value
    with pytest.raises(ValueError, match="idCounters.city"):
        catalogue_from_dict(data)


@pytest.mark.parametrize(
    "counters",
    [
        {"city": 1, "beacon": 0, "triangle": 0, "template": 0},
        {"city": 1, "beacon": 0, "triangle": 0, "template": 0, "map": 0, "triangles": 3},
    ],
)
def test_id_counters_require_exactly_five_known_keys(counters):
    data = catalogue_to_dict(_catalogue())
    data["idCounters"] = counters
    with pytest.raises(ValueError, match="idCounters doit contenir exactement"):
        catalogue_from_dict(data)


def test_id_counters_reject_non_object():
    data = catalogue_to_dict(_catalogue())
    data["idCounters"] = []
    with pytest.raises(ValueError, match="idCounters doit être un objet JSON"):
        catalogue_from_dict(data)


def test_counter_must_not_be_lower_than_an_existing_system_id():
    data = catalogue_to_dict(_catalogue())
    data["idCounters"]["city"] = 2
    with pytest.raises(ValueError, match="Compteur d'identifiants Catalogue incohérent pour city"):
        catalogue_from_dict(data)


def test_incoherent_memory_counter_cannot_be_serialized():
    catalogue = _catalogue()
    catalogue.id_counters["city"] = 2

    with pytest.raises(ValueError, match="Compteur d'identifiants Catalogue incohérent pour city"):
        catalogue_to_dict(catalogue)


def test_counter_equal_or_greater_than_max_system_id_is_valid():
    equal_data = catalogue_to_dict(_catalogue())
    assert catalogue_from_dict(equal_data).id_counters["city"] == 3

    greater_data = catalogue_to_dict(_catalogue())
    greater_data["idCounters"]["city"] = 42
    assert catalogue_from_dict(greater_data).id_counters["city"] == 42


def test_user_ids_do_not_affect_system_counters():
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    catalogue.add_city("Ville utilisateur", 42.0, 0.0)
    data = catalogue_to_dict(catalogue)
    data["idCounters"]["city"] = 42

    loaded = catalogue_from_dict(data)
    assert loaded.id_counters["city"] == 42


def test_system_ids_are_never_reused_after_delete_save_and_load(tmp_path):
    provider = SystemCatalogueIdProvider()
    catalogue = Catalogue(id_provider=provider)
    first = catalogue.add_city("A", 47.0, 2.0)
    second = catalogue.add_city("B", 46.0, 3.0)
    third = catalogue.add_city("C", 45.0, 4.0)
    first_template = catalogue.add_template("Premier")
    second_template = catalogue.add_template("Second")
    catalogue.delete_city(third.city_id)
    catalogue.delete_template(second_template.template_id)
    path = tmp_path / "catalogue.json"

    save_catalogue(catalogue, path)
    loaded = load_catalogue(path, id_provider=provider)

    assert loaded.id_counters == {"city": 3, "beacon": 0, "triangle": 0, "template": 2, "map": 0, "book": 0}
    assert first.city_id in loaded.cities
    assert second.city_id in loaded.cities
    assert first_template.template_id in loaded.templates
    assert loaded.add_city("D", 44.0, 5.0).city_id == "CITY-SYS-000004"
    assert loaded.add_template("Troisième").template_id == "TPL-SYS-000003"


def test_system_provider_after_loading_a_mixed_catalogue_starts_after_persisted_counter():
    data = {
        "version": 5,
        "idCounters": {"city": 8, "beacon": 0, "triangle": 0, "template": 0, "map": 0},
        "defaultTemplateId": None,
        "defaultMapId": None,
        "catalogueReferenceMapId": None,
        "cities": [
            {"cityId": "CITY-SYS-000005", "name": "Système", "latitude": 47.0, "longitude": 2.0, "archived": False},
            {
                "cityId": "CITY-USR-550e8400-e29b-41d4-a716-446655440000",
                "name": "Utilisateur",
                "latitude": 46.0,
                "longitude": 3.0,
                "archived": False,
            },
        ],
        "beacons": [],
        "triangles": [],
        "templates": [],
        "maps": [],
    }
    loaded = catalogue_from_dict(data, id_provider=SystemCatalogueIdProvider())

    assert loaded.add_city("Nouvelle", 45.0, 4.0).city_id == "CITY-SYS-000009"


def test_older_versions_are_rejected_explicitly_and_json_errors_stay_contextual(tmp_path):
    syntax_path = tmp_path / "syntax.json"
    syntax_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON catalogue invalide"):
        load_catalogue(syntax_path)
    with pytest.raises(ValueError, match="Version de catalogue non supportée : 1"):
        catalogue_from_dict({"version": 1})
    with pytest.raises(ValueError, match="Version de catalogue non supportée : 2"):
        catalogue_from_dict({"version": 2})


def test_save_replaces_existing_file_atomically(tmp_path):
    path = tmp_path / "catalogue.json"
    first = _catalogue()
    save_catalogue(first, path)
    second = _catalogue()
    second.update_template(next(iter(second.templates)), description="Nouvelle description")
    save_catalogue(second, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 5
    assert data["templates"][0]["description"] == "Nouvelle description"
    assert not path.with_suffix(".json.tmp").exists()
