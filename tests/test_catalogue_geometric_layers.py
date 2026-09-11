import json

import pytest

from src.assembleur_catalogue import Catalogue, CatalogueGeometricLayer
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_catalogue_io import catalogue_to_dict, load_catalogue, save_catalogue
from tools.migrate_catalogue_v5_to_v6 import migrate_catalogue_data_v5_to_v6, migrate_catalogue_file_v5_to_v6


def _catalogue_with_triangles(*, shared_base: bool = False) -> tuple[Catalogue, str, str, str]:
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    base = catalogue.add_city("Base", 47.0, 2.0)
    other_base = catalogue.add_city("Autre base", 46.0, 3.0)
    opening = catalogue.add_city("Ouverture", 45.0, 4.0)
    light = catalogue.add_city("Lumière", 44.0, 5.0)
    first = catalogue.add_triangle("Premier", opening.city_id, base.city_id, light.city_id)
    if shared_base:
        other_opening = catalogue.add_city("Ouverture 2", 43.0, 6.0)
        other_light = catalogue.add_city("Lumière 2", 42.0, 7.0)
        catalogue.add_triangle("Second", other_opening.city_id, base.city_id, other_light.city_id, archived=True)
    return catalogue, base.city_id, other_base.city_id, first.triangle_id


def test_geometric_layers_are_empty_by_default() -> None:
    catalogue, _base, _other_base, _triangle_id = _catalogue_with_triangles()
    assert catalogue.get_geometric_layers() == ()
    assert catalogue_to_dict(catalogue)["geometricLayers"] == {}


def test_set_refuses_an_existing_city_that_is_not_a_triangle_base() -> None:
    catalogue, _base, other_base, _triangle_id = _catalogue_with_triangles()
    with pytest.raises(ValueError, match="Base d'aucun triangle"):
        catalogue.set_geometric_layer(other_base, "geometric-layers/a.traces.json")


def test_set_accepts_a_real_triangle_base_and_replaces_its_layer() -> None:
    catalogue, base, _other_base, _triangle_id = _catalogue_with_triangles()
    catalogue.set_geometric_layer(base, "geometric-layers/one.traces.json")
    replacement = catalogue.set_geometric_layer(base, "geometric-layers/two.traces.json")
    assert catalogue.get_geometric_layer(base) == replacement
    assert catalogue.get_geometric_layers() == (replacement,)


@pytest.mark.parametrize("asset", ["C:/temp/a.traces.json", "../a.traces.json", "maps/a.traces.json"])
def test_geometric_layer_rejects_invalid_assets(asset: str) -> None:
    catalogue, base, _other_base, _triangle_id = _catalogue_with_triangles()
    with pytest.raises(ValueError):
        catalogue.set_geometric_layer(base, asset)


def test_geometric_layer_rejects_unknown_base() -> None:
    catalogue, _base, _other_base, _triangle_id = _catalogue_with_triangles()
    with pytest.raises(KeyError, match="Ville inconnue"):
        catalogue.set_geometric_layer("CITY-SYS-999999", "geometric-layers/a.traces.json")


def test_deleting_one_of_two_triangles_with_the_same_base_is_allowed() -> None:
    catalogue, base, _other_base, first_id = _catalogue_with_triangles(shared_base=True)
    catalogue.set_geometric_layer(base, "geometric-layers/a.traces.json")
    catalogue.delete_triangle(first_id)
    assert catalogue.is_triangle_base_city(base)


def test_deleting_the_last_triangle_for_a_layer_base_is_refused() -> None:
    catalogue, base, _other_base, triangle_id = _catalogue_with_triangles()
    catalogue.set_geometric_layer(base, "geometric-layers/a.traces.json")
    with pytest.raises(ValueError, match="calque géométrique"):
        catalogue.delete_triangle(triangle_id)


def test_changing_the_last_triangle_base_for_a_layer_is_refused() -> None:
    catalogue, base, other_base, triangle_id = _catalogue_with_triangles()
    catalogue.set_geometric_layer(base, "geometric-layers/a.traces.json")
    with pytest.raises(ValueError, match="calque géométrique"):
        catalogue.update_triangle(triangle_id, base_city_id=other_base)


def test_changing_a_triangle_base_is_allowed_when_another_triangle_keeps_it() -> None:
    catalogue, base, other_base, triangle_id = _catalogue_with_triangles(shared_base=True)
    catalogue.set_geometric_layer(base, "geometric-layers/a.traces.json")
    catalogue.update_triangle(triangle_id, base_city_id=other_base)
    assert catalogue.get_triangle(triangle_id).base_city_id == other_base
    assert catalogue.is_triangle_base_city(base)


def test_validate_rejects_a_layer_on_a_city_that_is_not_a_triangle_base() -> None:
    catalogue, _base, other_base, _triangle_id = _catalogue_with_triangles()
    catalogue.geometric_layers[other_base] = CatalogueGeometricLayer(other_base, "geometric-layers/a.traces.json")
    with pytest.raises(ValueError, match="Base d'aucun triangle"):
        catalogue.validate()


def test_geometric_layers_round_trip_uses_camel_case_and_migrates_v5(tmp_path) -> None:
    catalogue, base, _other_base, _triangle_id = _catalogue_with_triangles()
    catalogue.set_geometric_layer(base, "geometric-layers/a.traces.json")
    path = tmp_path / "catalogue.json"
    save_catalogue(catalogue, path)
    serialized = json.loads(path.read_text(encoding="utf-8"))
    assert serialized["geometricLayers"] == {
        base: {"asset": "geometric-layers/a.traces.json"},
    }
    assert serialized["geometricLayerDisplayOverrides"] == {}
    assert "geometric_layers" not in serialized
    assert catalogue_to_dict(load_catalogue(path)) == serialized

    v5 = dict(serialized)
    v5["version"] = 5
    v5.pop("geometricLayers")
    v5.pop("geometricLayerDisplayOverrides")
    migrated = migrate_catalogue_data_v5_to_v6(v5)
    assert migrated["version"] == 6
    assert migrated["geometricLayers"] == {}
    assert "geometric_layers" not in migrated

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(v5), encoding="utf-8")
    backup = migrate_catalogue_file_v5_to_v6(legacy_path)
    assert json.loads(backup.read_text(encoding="utf-8")) == v5
    assert json.loads(legacy_path.read_text(encoding="utf-8")) == migrated
