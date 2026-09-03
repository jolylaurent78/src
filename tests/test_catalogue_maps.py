import math

import pytest

from src.assembleur_catalogue import Catalogue, CatalogueMap, WorldRect, centered_world_rect
from src.assembleur_catalogue_identity import (
    SystemCatalogueIdProvider,
    UserCatalogueIdProvider,
    is_catalogue_map_id,
    is_user_catalogue_id,
)
from src.assembleur_catalogue_io import catalogue_from_dict, catalogue_to_dict


def _rect() -> WorldRect:
    return WorldRect(-2963.38, 1642.99, 4293.65, 5282.16)


def _add_map(catalogue: Catalogue, *, name: str = "899 - Alsace", **overrides) -> str:
    values = {
        "name": name,
        "image_file": "899 - Alsace.jpg",
        "calibration_file": "899 - Alsace.json",
        "projection": "EPSG:2154",
        "default_world_rect": _rect(),
        "default_scale_factor": 12.0,
    }
    values.update(overrides)
    return catalogue.add_map(**values)


def test_map_identity_and_system_allocation_are_catalogue_native():
    catalogue = Catalogue(provider=SystemCatalogueIdProvider())

    map_id = _add_map(catalogue)

    assert map_id == "MAP-SYS-000001"
    assert is_catalogue_map_id(map_id)
    assert catalogue.id_counters["map"] == 1
    assert catalogue.get_map(map_id).default_scale_factor == 12.0


def test_user_map_allocation_does_not_change_system_counter():
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())

    map_id = _add_map(catalogue)

    assert is_user_catalogue_id(map_id)
    assert is_catalogue_map_id(map_id)
    assert catalogue.id_counters["map"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": " "},
        {"image_file": ""},
        {"image_file": r"C:\\maps\\map.jpg"},
        {"image_file": "../map.jpg"},
        {"image_file": "./map.jpg"},
        {"default_world_rect": WorldRect(0, 0, 0, 1)},
        {"default_world_rect": WorldRect(0, 0, 1, float("nan"))},
        {"default_scale_factor": 0},
        {"default_scale_factor": float("inf")},
        {"default_scale_factor": True},
        {"calibration_file": "map.json", "projection": "EPSG:4326"},
        {"calibration_file": None, "projection": "EPSG:2154"},
        {"archived": 1},
    ],
)
def test_invalid_map_creation_does_not_consume_system_counter(overrides):
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())

    with pytest.raises(ValueError):
        _add_map(catalogue, **overrides)

    assert catalogue.id_counters["map"] == 0
    assert _add_map(catalogue, name="Valide") == "MAP-SYS-000001"


def test_partial_and_uncalibrated_map_states_are_valid():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    visual_map_id = _add_map(
        catalogue,
        name="Fond visuel",
        calibration_file=None,
        projection=None,
    )
    assert catalogue.get_map(visual_map_id).projection is None
    catalogue.validate()


def test_map_names_follow_existing_catalogue_global_uniqueness_rule():
    catalogue = Catalogue()
    _add_map(catalogue)

    with pytest.raises(ValueError, match="déjà"):
        _add_map(catalogue, name="899 - alsace")


def test_archive_preserves_identity_counter_and_clears_default_map():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    first = _add_map(catalogue, name="Première")
    catalogue.set_default_map(first)
    catalogue.archive_map(first)

    assert catalogue.get_map(first).archived is True
    assert catalogue.default_map_id is None
    assert _add_map(catalogue, name="Seconde") == "MAP-SYS-000002"
    assert catalogue.id_counters["map"] == 2


def test_default_map_must_resolve_to_an_active_map():
    catalogue = Catalogue()
    map_id = _add_map(catalogue)
    catalogue.set_default_map(map_id)
    assert catalogue.default_map_id == map_id
    catalogue.archive_map(map_id)

    with pytest.raises(KeyError, match="Carte inconnue"):
        catalogue.set_default_map("MAP-SYS-999999")
    with pytest.raises(ValueError, match="archivée"):
        catalogue.set_default_map(map_id)


def test_global_validation_rejects_corrupted_map_state_and_counter():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    map_id = _add_map(catalogue)
    catalogue.maps[map_id].default_scale_factor = math.nan
    with pytest.raises(ValueError, match="default_scale_factor"):
        catalogue.validate()

    catalogue.maps[map_id].default_scale_factor = 12.0
    catalogue.id_counters["map"] = 0
    with pytest.raises(ValueError, match="incohérent pour map"):
        catalogue.validate()


def test_v3_map_round_trip_preserves_all_map_fields_and_default():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    map_id = _add_map(catalogue)
    catalogue.set_default_map(map_id)
    catalogue.archive_map(map_id)
    active_map_id = _add_map(
        catalogue,
        name="Fond non calibré",
        calibration_file=None,
        projection=None,
    )
    catalogue.set_default_map(active_map_id)

    restored = catalogue_from_dict(catalogue_to_dict(catalogue))
    archived = restored.get_map(map_id)

    assert restored.default_map_id == active_map_id
    assert restored.id_counters["map"] == 2
    assert archived.archived is True
    assert archived.image_file == "899 - Alsace.jpg"
    assert archived.calibration_file == "899 - Alsace.json"
    assert archived.projection == "EPSG:2154"
    assert archived.default_world_rect == _rect()
    assert archived.default_scale_factor == 12.0


def test_v3_loader_requires_map_counter_default_and_maps_collection():
    data = catalogue_to_dict(Catalogue())
    data["idCounters"].pop("map")
    with pytest.raises(ValueError, match="idCounters doit contenir exactement"):
        catalogue_from_dict(data)

    data = catalogue_to_dict(Catalogue())
    data.pop("defaultMapId")
    with pytest.raises(ValueError, match="la racine.*clés manquantes.*defaultMapId"):
        catalogue_from_dict(data)

    data = catalogue_to_dict(Catalogue())
    data.pop("maps")
    with pytest.raises(ValueError, match="la racine.*clés manquantes.*maps"):
        catalogue_from_dict(data)

    data = catalogue_to_dict(Catalogue())
    data["idCounters"]["map"] = True
    with pytest.raises(ValueError, match="idCounters.map"):
        catalogue_from_dict(data)


def test_v3_loader_rejects_invalid_map_json_and_default_reference():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    map_id = _add_map(catalogue)
    data = catalogue_to_dict(catalogue)

    data["maps"][0]["imageFile"] = "/absolute/map.jpg"
    with pytest.raises(ValueError, match="référence relative sûre"):
        catalogue_from_dict(data)

    data = catalogue_to_dict(catalogue)
    data["defaultMapId"] = "MAP-SYS-999999"
    with pytest.raises(ValueError, match="carte par défaut .* absente"):
        catalogue_from_dict(data)

    data = catalogue_to_dict(catalogue)
    data["maps"][0]["unknown"] = "value"
    with pytest.raises(ValueError, match=r"maps\[1\].*clés inconnues"):
        catalogue_from_dict(data)

    data = catalogue_to_dict(catalogue)
    data["defaultMapId"] = 1
    with pytest.raises(ValueError, match="defaultMapId doit être une chaîne ou null"):
        catalogue_from_dict(data)

    data = catalogue_to_dict(catalogue)
    data["version"] = 99
    with pytest.raises(ValueError, match="Version de catalogue non supportée : 99"):
        catalogue_from_dict(data)


def test_clone_owns_map_collection_world_rect_and_counter():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    map_id = _add_map(catalogue)
    catalogue.set_default_map(map_id)
    clone = catalogue.clone()

    assert clone.default_map_id == map_id
    assert clone.maps[map_id] is not catalogue.maps[map_id]
    assert clone.maps[map_id].default_world_rect is not catalogue.maps[map_id].default_world_rect
    clone.update_map(map_id, name="Carte clone", default_scale_factor=15.0)
    clone.maps[map_id].default_world_rect.w = 5000.0

    assert catalogue.maps[map_id].name == "899 - Alsace"
    assert catalogue.maps[map_id].default_scale_factor == 12.0
    assert catalogue.maps[map_id].default_world_rect.w == _rect().w
    assert clone.add_map(
        name="Autre carte",
        image_file="autre.jpg",
        default_world_rect=WorldRect(0, 0, 1, 1),
        default_scale_factor=1.0,
    ) == "MAP-SYS-000002"
    assert catalogue.id_counters["map"] == 1


def test_map_default_scale_factor_is_bounded_to_twenty():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    map_id = _add_map(catalogue)

    catalogue.update_map(map_id, default_scale_factor=20.0)

    assert catalogue.get_map(map_id).default_scale_factor == 20.0
    with pytest.raises(ValueError, match="inferieur ou egal a 20"):
        catalogue.update_map(map_id, default_scale_factor=20.1)


def test_centered_world_rect_uses_the_intrinsic_map_origin():
    assert centered_world_rect(100.0, 80.0) == WorldRect(-50.0, -40.0, 100.0, 80.0)


def test_direct_catalogue_map_validation_checks_identity_and_archived_type():
    catalogue = Catalogue()
    invalid = CatalogueMap(
        "CITY-SYS-000001",
        "Carte",
        "map.jpg",
        None,
        None,
        WorldRect(0, 0, 1, 1),
        1.0,
        False,
    )
    catalogue.maps[invalid.map_id] = invalid
    with pytest.raises(ValueError, match="Identifiant carte invalide"):
        catalogue.validate()
