import json

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_geometric_layer_assets import CatalogueGeometricLayerAssetController
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_catalogue_io import catalogue_from_dict, catalogue_to_dict, load_catalogue
from src.assembleur_geometric_layer_display import GeometricLayerModuleDisplayOverride
from src.assembleur_paths import ApplicationPaths
from tools.migrate_catalogue_v6_to_v7 import migrate_catalogue_data_v6_to_v7
from tools.migrate_catalogue_v7_to_v8 import migrate_catalogue_data_v7_to_v8, migrate_catalogue_file_v7_to_v8


def _catalogue():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    base = catalogue.add_city("Base", 47, 2)
    opening = catalogue.add_city("Ouverture", 46, 3)
    light = catalogue.add_city("Lumière", 45, 4)
    catalogue.add_triangle("Trace", opening.city_id, base.city_id, light.city_id)
    catalogue.set_geometric_layer(base.city_id, f"geometric-layers/{base.city_id}.traces.json")
    return catalogue, base.city_id


def test_global_display_override_api_validates_and_removes_empty_overrides():
    catalogue, _base = _catalogue()
    catalogue.set_geometric_layer_display_override("lumiere", color_bgr=(36, 28, 237))
    assert catalogue.get_geometric_layer_display_override("lumiere") == GeometricLayerModuleDisplayOverride((36, 28, 237), None)
    catalogue.set_geometric_layer_display_override("lumiere", width=3)
    assert catalogue.get_geometric_layer_display_override("lumiere") == GeometricLayerModuleDisplayOverride(None, 3)
    catalogue.set_geometric_layer_display_override("lumiere", color_bgr=(1, 2, 3), width=4)
    catalogue.remove_geometric_layer_display_override("lumiere")
    assert catalogue.get_geometric_layer_display_override("lumiere") is None


@pytest.mark.parametrize("color", ([1, 2, 3], (1, 2), (True, 0, 0), (-1, 0, 0), (256, 0, 0)))
def test_global_display_override_rejects_invalid_bgr(color):
    catalogue, _base = _catalogue()
    with pytest.raises(ValueError):
        catalogue.set_geometric_layer_display_override("lumiere", color_bgr=color)


@pytest.mark.parametrize("module_id,width", (("", 2), ("lumiere", True), ("lumiere", 0), ("lumiere", 1.5)))
def test_global_display_override_rejects_invalid_module_or_width(module_id, width):
    catalogue, _base = _catalogue()
    with pytest.raises(ValueError):
        catalogue.set_geometric_layer_display_override(module_id, width=width)


def test_clone_and_layer_mutations_leave_global_overrides_independent():
    catalogue, base = _catalogue()
    catalogue.set_geometric_layer_display_override("lumiere", color_bgr=(1, 2, 3), width=3)
    cloned = catalogue.clone()
    cloned.set_geometric_layer_display_override("lumiere", width=5)
    cloned.set_geometric_layer(base, f"geometric-layers/{base}.traces.json")
    cloned.remove_geometric_layer(base)
    assert catalogue.get_geometric_layer_display_override("lumiere") == GeometricLayerModuleDisplayOverride((1, 2, 3), 3)
    assert cloned.get_geometric_layer_display_override("lumiere") == GeometricLayerModuleDisplayOverride(None, 5)


def test_one_global_override_is_shared_by_distinct_layer_bases():
    catalogue, base = _catalogue()
    opening = catalogue.add_city("Ouverture 2", 44, 3)
    other_base = catalogue.add_city("Base 2", 43, 4)
    light = catalogue.add_city("Lumière 2", 42, 5)
    catalogue.add_triangle("Trace 2", opening.city_id, other_base.city_id, light.city_id)
    catalogue.set_geometric_layer(other_base.city_id, f"geometric-layers/{other_base.city_id}.traces.json")
    catalogue.set_geometric_layer_display_override("commun", width=3)
    assert catalogue.get_geometric_layer(base).asset_file.endswith(".traces.json")
    assert catalogue.get_geometric_layer_display_overrides() == {"commun": GeometricLayerModuleDisplayOverride(None, 3)}


def test_validate_rejects_empty_or_invalid_global_override():
    catalogue, _base = _catalogue()
    catalogue.geometric_layer_display_overrides["vide"] = GeometricLayerModuleDisplayOverride()
    with pytest.raises(ValueError, match="Override graphique global vide"):
        catalogue.validate()
    catalogue.geometric_layer_display_overrides = {"ok": object()}
    with pytest.raises(ValueError, match="Override graphique global invalide"):
        catalogue.validate()


def test_v8_serialization_is_global_and_strict():
    catalogue, base = _catalogue()
    catalogue.set_geometric_layer_display_override("ombre", width=2)
    catalogue.set_geometric_layer_display_override("lumiere", color_bgr=(36, 28, 237), width=3)
    data = catalogue_to_dict(catalogue)
    assert data["version"] == 8
    assert list(data["geometricLayerDisplayOverrides"]) == ["lumiere", "ombre"]
    assert data["geometricLayers"][base] == {"asset": f"geometric-layers/{base}.traces.json"}
    assert catalogue_from_dict(data).get_geometric_layer_display_override("ombre") == GeometricLayerModuleDisplayOverride(None, 2)
    broken = json.loads(json.dumps(data))
    broken["geometricLayers"][base]["displayOverrides"] = {}
    with pytest.raises(ValueError):
        catalogue_from_dict(broken)


@pytest.mark.parametrize("override", [{}, {"colorBgr": [1, 2]}, {"colorBgr": [True, 0, 0]}, {"width": True}, {"width": 0}, {"style": "Arrow"}])
def test_v8_rejects_malformed_global_display_override_shapes(override):
    catalogue, _base = _catalogue()
    data = catalogue_to_dict(catalogue)
    data["geometricLayerDisplayOverrides"] = {"module": override}
    with pytest.raises(ValueError):
        catalogue_from_dict(data)


def test_v7_to_v8_discards_every_per_layer_override_and_v6_loads_through_both_migrations(tmp_path):
    catalogue, base = _catalogue()
    v8 = catalogue_to_dict(catalogue)
    v7 = dict(v8)
    v7["version"] = 7
    v7.pop("geometricLayerDisplayOverrides")
    v7["geometricLayers"] = {base: {"asset": f"geometric-layers/{base}.traces.json", "displayOverrides": {"old": {"width": 2}}}}
    migrated = migrate_catalogue_data_v7_to_v8(v7)
    assert migrated["geometricLayerDisplayOverrides"] == {}
    assert migrated["geometricLayers"][base] == {"asset": f"geometric-layers/{base}.traces.json"}
    v7_path = tmp_path / "v7.json"
    v7_path.write_text(json.dumps(v7), encoding="utf-8")
    backup = migrate_catalogue_file_v7_to_v8(v7_path)
    assert json.loads(backup.read_text(encoding="utf-8"))["version"] == 7
    assert json.loads(v7_path.read_text(encoding="utf-8"))["geometricLayerDisplayOverrides"] == {}
    v6 = dict(v7)
    v6["version"] = 6
    v6["geometricLayers"] = {base: {"asset": f"geometric-layers/{base}.traces.json"}}
    assert migrate_catalogue_data_v6_to_v7(v6)["version"] == 7
    source = tmp_path / "v6.json"
    source.write_text(json.dumps(v6), encoding="utf-8")
    loaded = load_catalogue(source)
    assert loaded.version == 8
    assert loaded.get_geometric_layer_display_overrides() == {}


def test_asset_controller_rollback_does_not_change_global_overrides(tmp_path):
    catalogue, base = _catalogue()
    catalogue.set_geometric_layer_display_override("lumiere", color_bgr=(1, 2, 3), width=3)
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    source = tmp_path / "source.traces.json"
    source.write_text(json.dumps({
        "schema_version": 2, "source": "AlgoSimulator", "algorithm": "Test", "segment": "1",
        "scope": {"type": "automatic_aggregation"}, "modules": [],
    }), encoding="utf-8")
    controller.stage_geometric_layer(base, source)
    created = controller.commit()
    controller.rollback(created)
    assert catalogue.get_geometric_layer_display_override("lumiere") == GeometricLayerModuleDisplayOverride((1, 2, 3), 3)
