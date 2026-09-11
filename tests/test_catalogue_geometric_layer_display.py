import json

import pytest

from src.assembleur_catalogue import Catalogue, CatalogueGeometricLayer
from src.assembleur_catalogue_geometric_layer_assets import CatalogueGeometricLayerAssetController
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_catalogue_io import catalogue_from_dict, catalogue_to_dict, load_catalogue
from src.assembleur_geometric_layer_display import GeometricLayerModuleDisplayOverride
from src.assembleur_paths import ApplicationPaths
from tools.migrate_catalogue_v6_to_v7 import migrate_catalogue_data_v6_to_v7


def _catalogue():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    base = catalogue.add_city("Base", 47, 2)
    opening = catalogue.add_city("Ouverture", 46, 3)
    light = catalogue.add_city("Lumière", 45, 4)
    catalogue.add_triangle("Trace", opening.city_id, base.city_id, light.city_id)
    catalogue.set_geometric_layer(base.city_id, f"geometric-layers/{base.city_id}.traces.json")
    return catalogue, base.city_id


def test_display_override_api_validates_and_removes_empty_overrides():
    catalogue, base = _catalogue()
    catalogue.set_geometric_layer_display_override(base, "lumiere", color_bgr=(36, 28, 237))
    assert catalogue.get_geometric_layer_display_override(base, "lumiere") == GeometricLayerModuleDisplayOverride((36, 28, 237), None)
    catalogue.set_geometric_layer_display_override(base, "lumiere", width=3)
    assert catalogue.get_geometric_layer_display_override(base, "lumiere") == GeometricLayerModuleDisplayOverride(None, 3)
    catalogue.set_geometric_layer_display_override(base, "lumiere", color_bgr=(1, 2, 3), width=4)
    catalogue.remove_geometric_layer_display_override(base, "lumiere")
    assert catalogue.get_geometric_layer_display_override(base, "lumiere") is None


@pytest.mark.parametrize("color", ([1, 2, 3], (1, 2), (True, 0, 0), (-1, 0, 0), (256, 0, 0)))
def test_display_override_rejects_invalid_bgr(color):
    catalogue, base = _catalogue()
    with pytest.raises(ValueError):
        catalogue.set_geometric_layer_display_override(base, "lumiere", color_bgr=color)


@pytest.mark.parametrize("module_id,width", (("", 2), ("lumiere", True), ("lumiere", 0), ("lumiere", 1.5)))
def test_display_override_rejects_invalid_module_or_width(module_id, width):
    catalogue, base = _catalogue()
    with pytest.raises(ValueError):
        catalogue.set_geometric_layer_display_override(base, module_id, width=width)


def test_clone_and_reimport_keep_independent_overrides():
    catalogue, base = _catalogue()
    catalogue.set_geometric_layer_display_override(base, "lumiere", color_bgr=(1, 2, 3), width=3)
    cloned = catalogue.clone()
    cloned.set_geometric_layer_display_override(base, "lumiere", width=5)
    cloned.set_geometric_layer(base, f"geometric-layers/{base}.traces.json")
    assert catalogue.get_geometric_layer_display_override(base, "lumiere") == GeometricLayerModuleDisplayOverride((1, 2, 3), 3)
    assert cloned.get_geometric_layer_display_override(base, "lumiere") == GeometricLayerModuleDisplayOverride(None, 5)


def test_layer_defensively_freezes_its_override_mapping_and_validate_rejects_empty_override():
    catalogue, base = _catalogue()
    source = {"lumiere": GeometricLayerModuleDisplayOverride((1, 2, 3), None)}
    layer = CatalogueGeometricLayer(base, f"geometric-layers/{base}.traces.json", source)
    source.clear()
    assert layer.display_overrides == {"lumiere": GeometricLayerModuleDisplayOverride((1, 2, 3), None)}
    with pytest.raises(TypeError):
        layer.display_overrides["ombre"] = GeometricLayerModuleDisplayOverride(None, 2)
    catalogue.geometric_layers[base] = CatalogueGeometricLayer(
        base, f"geometric-layers/{base}.traces.json", {"vide": GeometricLayerModuleDisplayOverride()},
    )
    with pytest.raises(ValueError, match="override vide"):
        catalogue.validate()


def test_v7_serialization_and_migration_are_strict(tmp_path):
    catalogue, base = _catalogue()
    catalogue.set_geometric_layer_display_override(base, "ombre", width=2)
    catalogue.set_geometric_layer_display_override(base, "lumiere", color_bgr=(36, 28, 237), width=3)
    data = catalogue_to_dict(catalogue)
    assert data["version"] == 7
    assert list(data["geometricLayers"][base]["displayOverrides"]) == ["lumiere", "ombre"]
    assert catalogue_from_dict(data).get_geometric_layer_display_override(base, "ombre") == GeometricLayerModuleDisplayOverride(None, 2)
    broken = json.loads(json.dumps(data))
    broken["geometricLayers"][base]["displayOverrides"]["lumiere"] = {}
    with pytest.raises(ValueError):
        catalogue_from_dict(broken)
    v6 = json.loads(json.dumps(data))
    v6["version"] = 6
    v6["geometricLayers"][base] = {"asset": v6["geometricLayers"][base]["asset"]}
    migrated = migrate_catalogue_data_v6_to_v7(v6)
    assert migrated["geometricLayers"][base]["displayOverrides"] == {}
    source = tmp_path / "v6.json"
    source.write_text(json.dumps(v6), encoding="utf-8")
    assert load_catalogue(source).version == 7


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"colorBgr": [1, 2]},
        {"colorBgr": [1, 2, 3, 4]},
        {"colorBgr": [True, 0, 0]},
        {"colorBgr": [-1, 0, 0]},
        {"colorBgr": [256, 0, 0]},
        {"colorBgr": ["1", 2, 3]},
        {"width": True},
        {"width": 0},
        {"width": -1},
        {"width": 1.5},
        {"width": "2"},
        {"style": "Arrow"},
    ],
)
def test_v7_rejects_malformed_display_override_shapes(override):
    catalogue, base = _catalogue()
    data = catalogue_to_dict(catalogue)
    data["geometricLayers"][base]["displayOverrides"] = {"module": override}
    with pytest.raises(ValueError):
        catalogue_from_dict(data)


def test_v7_rejects_empty_module_identifier_in_display_overrides():
    catalogue, base = _catalogue()
    data = catalogue_to_dict(catalogue)
    data["geometricLayers"][base]["displayOverrides"] = {"": {"width": 2}}
    with pytest.raises(ValueError, match="module_id vide"):
        catalogue_from_dict(data)


def test_rollback_restores_the_previous_layer_overrides(tmp_path):
    catalogue, base = _catalogue()
    catalogue.set_geometric_layer_display_override(base, "lumiere", color_bgr=(1, 2, 3), width=3)
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
    assert catalogue.get_geometric_layer_display_override(base, "lumiere") == GeometricLayerModuleDisplayOverride((1, 2, 3), 3)
