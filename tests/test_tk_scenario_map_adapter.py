import json

import pytest
from PIL import Image

from src.assembleur_background_map_layer import BackgroundMapLayer
from src.assembleur_catalogue import Catalogue, WorldRect
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_core import ScenarioAssemblage
from src.assembleur_paths import ApplicationPaths
from src.assembleur_scenario_map import ScenarioMapPosition, ScenarioMapState
from src.assembleur_scenario_map_controller import ScenarioMapController


def _controller(tmp_path, scenario=None):
    paths = ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user-root",
        catalogue_mode="SYS",
    )
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    map_id = catalogue.add_map(
        name="Carte",
        image_file="map.jpg",
        calibration_file="map.json",
        projection="EPSG:2154",
        default_world_rect=WorldRect(10, 20, 400, 200),
        default_scale_factor=12,
    )
    paths.default_catalogue_maps_dir.mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(
        paths.default_catalogue_maps_dir / "map.jpg"
    )
    (paths.default_catalogue_maps_dir / "map.json").write_text(
        json.dumps(
            {"projection": "EPSG:2154", "A": [[0.01, 0], [0, 0.01]], "offset": [0, 0]}
        ),
        encoding="utf-8",
    )
    scenario = scenario or ScenarioAssemblage("Carte")
    holder = {}
    layer = BackgroundMapLayer(
        lambda point: point,
        lambda x, y: (x, y),
        lambda: holder["controller"].sync_active_state_from_background(),
    )
    controller = ScenarioMapController(catalogue, paths, layer, lambda: scenario)
    holder["controller"] = controller
    return controller, layer, scenario, catalogue, map_id


def test_controller_projects_resolved_map_and_uses_its_transform(tmp_path) -> None:
    controller, layer, scenario, _catalogue, map_id = _controller(tmp_path)
    state = ScenarioMapState(map_id, ScenarioMapPosition(30, 40), 18, False)
    scenario.map_state = state

    controller.apply_state(state)

    assert layer.world_rect.x0 == 30
    assert layer.world_rect.w == 600
    assert layer.base_image.mode == "RGBA"
    assert controller.resolved_map.visible is False
    assert controller.lambert_to_world(0, 0) == (30, 340)
    assert controller.scale_factor == 18

    layer.start_move(0, 0)
    assert layer.update_move(5, 7) is True

    assert scenario.map_state.position_override == ScenarioMapPosition(35, 47)
    assert controller.lambert_to_world(0, 0) == (35, 347)


def test_controller_default_capture_clear_and_visibility(tmp_path) -> None:
    controller, layer, scenario, catalogue, map_id = _controller(tmp_path)

    assert controller.new_default_state() == ScenarioMapState(catalogue.default_map_id)
    assert controller.capture_active_state() == ScenarioMapState(catalogue.default_map_id)

    state = ScenarioMapState(map_id, visible=True)
    scenario.map_state = state
    assert controller.capture_active_state() is state
    controller.apply_state(state)
    controller.set_active_visibility(False)
    assert scenario.map_state.visible is False
    assert controller.resolved_map.visible is False

    controller.apply_state(ScenarioMapState(None))
    assert layer.has_map is False
    assert controller.resolved_map is None
    with pytest.raises(RuntimeError, match="Aucune carte calibrée active"):
        controller.lambert_to_world(0, 0)


def test_controller_visibility_changes_only_the_active_scenario(tmp_path) -> None:
    controller, _layer, first, _catalogue, map_id = _controller(tmp_path)
    second = ScenarioAssemblage("Autre carte")
    first.map_state = ScenarioMapState(map_id, visible=True)
    second.map_state = ScenarioMapState(map_id, visible=True)
    active = {"scenario": first}
    controller._active_scenario_provider = lambda: active["scenario"]

    controller.set_active_visibility(False)

    assert first.map_state.visible is False
    assert second.map_state.visible is True


def test_controller_sync_restores_default_overrides_and_preserves_visibility(tmp_path) -> None:
    controller, layer, scenario, catalogue, map_id = _controller(tmp_path)
    state = ScenarioMapState(map_id, visible=False)
    scenario.map_state = state
    controller.apply_state(state)

    layer.start_resize("br", 0, 0)
    assert layer.update_resize(810, 420) is True
    assert scenario.map_state.position_override == ScenarioMapPosition(10, 220)
    assert scenario.map_state.scale_factor_override == pytest.approx(24)
    assert scenario.map_state.visible is False

    default = catalogue.get_map(map_id).default_world_rect
    layer.set_map(
        layer.base_image,
        type(layer.world_rect)(default.x0, default.y0, default.w, default.h),
    )
    controller.sync_active_state_from_background()
    assert scenario.map_state.position_override is None
    assert scenario.map_state.scale_factor_override is None
    assert scenario.map_state.visible is False


def test_controller_set_catalogue_rebuilds_runtime_resolvers(tmp_path) -> None:
    controller, _layer, _scenario, catalogue, _map_id = _controller(tmp_path)
    previous_resolver = controller._resolver
    previous_assets = controller._assets
    replacement = catalogue.clone()

    controller.set_catalogue(replacement)

    assert controller._resolver is not previous_resolver
    assert controller._assets is not previous_assets
    assert controller.new_default_state() == ScenarioMapState(replacement.default_map_id)


def test_controller_set_catalogue_reloads_the_active_map_from_new_calibration(tmp_path) -> None:
    controller, layer, scenario, catalogue, map_id = _controller(tmp_path)
    scenario.map_state = ScenarioMapState(map_id)
    controller.apply_state(scenario.map_state)
    assert controller.lambert_to_world(0, 0) == (10, 220)

    replacement = catalogue.clone()
    replacement.update_map(
        map_id,
        name="Carte B",
        image_file="map-b.jpg",
        calibration_file="map-b.json",
        default_world_rect=WorldRect(100, 200, 800, 400),
        default_scale_factor=16,
    )
    Image.new("RGB", (200, 100), "red").save(
        controller._paths.default_catalogue_maps_dir / "map-b.jpg"
    )
    (controller._paths.default_catalogue_maps_dir / "map-b.json").write_text(
        json.dumps(
            {"projection": "EPSG:2154", "A": [[0.01, 0], [0, 0.01]], "offset": [5, 7]}
        ),
        encoding="utf-8",
    )

    controller.set_catalogue(replacement)

    assert controller.resolved_map.catalogue_map.name == "Carte B"
    assert controller.resolved_map.world_rect == WorldRect(100, 200, 800, 400)
    assert layer.world_rect.x0 == 100
    assert layer.base_image.getpixel((0, 0)) == pytest.approx((255, 0, 0, 255), abs=1)
    assert controller.lambert_to_world(0, 0) == (120, 572)
