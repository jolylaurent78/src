import json
from pathlib import Path

import pytest
from PIL import Image

from src.assembleur_catalogue import Catalogue, WorldRect
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider, UserCatalogueIdProvider
from src.assembleur_catalogue_io import load_catalogue
from src.assembleur_catalogue_map_assets import CatalogueMapAssetResolver
from src.assembleur_paths import ApplicationPaths
from src.assembleur_scenario_map import ScenarioMapPosition, ScenarioMapState
from src.assembleur_scenario_map_runtime import ScenarioMapResolver, resolve_scenario_map


def _paths(tmp_path: Path) -> ApplicationPaths:
    return ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user-data",
    )


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 100), "white").save(path)


def _write_calibration(path: Path) -> None:
    path.write_text(
        json.dumps({"projection": "EPSG:2154", "A": [[0.01, 0], [0, 0.01]], "offset": [0, 0]}),
        encoding="utf-8",
    )


def _catalogue(provider, *, archived: bool = False, calibrated: bool = True) -> tuple[Catalogue, str]:
    catalogue = Catalogue(id_provider=provider)
    map_id = catalogue.add_map(
        name="Carte test",
        image_file="map.jpg",
        calibration_file="map.json" if calibrated else None,
        projection="EPSG:2154" if calibrated else None,
        default_world_rect=WorldRect(100, 200, 400, 200),
        default_scale_factor=12.0,
        archived=archived,
    )
    return catalogue, map_id


def _prepare_assets(paths: ApplicationPaths, map_id: str) -> None:
    root = paths.default_catalogue_maps_dir if map_id.startswith("MAP-SYS-") else paths.user_catalogue_maps_dir
    _write_image(root / "map.jpg")
    _write_calibration(root / "map.json")


def test_state_none_resolves_to_no_map(tmp_path):
    catalogue, _ = _catalogue(SystemCatalogueIdProvider())

    assert resolve_scenario_map(catalogue, ScenarioMapState(None), CatalogueMapAssetResolver(_paths(tmp_path))) is None


@pytest.mark.parametrize("opacity", [-0.1, 1.1, float("nan"), True])
def test_state_rejects_invalid_opacity(opacity):
    with pytest.raises(ValueError, match="opacity"):
        ScenarioMapState(None, opacity=opacity)


def test_state_rejects_invalid_visibility():
    with pytest.raises(ValueError, match="visible"):
        ScenarioMapState(None, visible=1)


@pytest.mark.parametrize(
    ("x0", "y0"),
    [
        (float("nan"), 0),
        (0, float("nan")),
        (float("inf"), 0),
        (0, float("-inf")),
        (True, 0),
        (0, False),
        ("1", 0),
        (0, "1"),
    ],
)
def test_position_rejects_non_finite_or_non_numeric_coordinates(x0, y0):
    with pytest.raises(ValueError, match="position_override"):
        ScenarioMapPosition(x0, y0)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_state_rejects_invalid_scale_override(value):
    with pytest.raises(ValueError, match="scale_factor_override"):
        ScenarioMapState("MAP-SYS-000001", scale_factor_override=value)


def test_resolver_uses_catalogue_defaults_and_preserves_visibility_opacity(tmp_path):
    paths = _paths(tmp_path)
    catalogue, map_id = _catalogue(SystemCatalogueIdProvider())
    _prepare_assets(paths, map_id)

    resolved = ScenarioMapResolver(catalogue, CatalogueMapAssetResolver(paths)).resolve(
        ScenarioMapState(map_id, visible=False, opacity=0.35)
    )

    assert resolved is not None
    assert resolved.map_id == map_id
    assert resolved.world_rect == WorldRect(100, 200, 400, 200)
    assert resolved.scale_factor == 12.0
    assert resolved.visible is False
    assert resolved.opacity == 0.35


def test_resolver_supports_user_map_archived_map_scale_and_position_overrides(tmp_path):
    user_paths = _paths(tmp_path / "user")
    user_catalogue, user_map_id = _catalogue(UserCatalogueIdProvider())
    _prepare_assets(user_paths, user_map_id)
    scale_resolved = resolve_scenario_map(
        user_catalogue,
        ScenarioMapState(user_map_id, scale_factor_override=18.0),
        CatalogueMapAssetResolver(user_paths),
    )
    assert scale_resolved is not None
    assert scale_resolved.world_rect == WorldRect(100, 200, 600, 300)
    assert scale_resolved.scale_factor == 18.0

    system_paths = _paths(tmp_path / "archived")
    system_catalogue, system_map_id = _catalogue(SystemCatalogueIdProvider(), archived=True)
    _prepare_assets(system_paths, system_map_id)
    position_resolved = resolve_scenario_map(
        system_catalogue,
        ScenarioMapState(system_map_id, position_override=ScenarioMapPosition(-50, 20)),
        CatalogueMapAssetResolver(system_paths),
    )
    assert position_resolved is not None
    assert position_resolved.world_rect == WorldRect(-50, 20, 400, 200)
    assert position_resolved.scale_factor == 12.0


def test_resolver_composes_position_and_scale_and_keeps_round_trips(tmp_path):
    paths = _paths(tmp_path)
    catalogue, map_id = _catalogue(SystemCatalogueIdProvider())
    _prepare_assets(paths, map_id)

    resolved = resolve_scenario_map(
        catalogue,
        ScenarioMapState(
            map_id,
            position_override=ScenarioMapPosition(-2500.0, 1800.0),
            scale_factor_override=15.0,
        ),
        CatalogueMapAssetResolver(paths),
    )

    assert resolved is not None
    assert resolved.world_rect == WorldRect(-2500.0, 1800.0, 500.0, 250.0)
    assert resolved.scale_factor == 15.0
    lambert = (1234.5, 6789.0)
    world = resolved.transform.lambert_to_world(*lambert)
    assert resolved.transform.world_to_lambert(*world) == pytest.approx(lambert, abs=1e-9)
    assert resolved.transform.lambert_to_world(*resolved.transform.world_to_lambert(*world)) == pytest.approx(
        world, abs=1e-9
    )


def test_resolver_fails_explicitly_for_unknown_missing_or_uncalibrated_map(tmp_path):
    paths = _paths(tmp_path)
    catalogue, map_id = _catalogue(SystemCatalogueIdProvider())
    resolver = CatalogueMapAssetResolver(paths)

    with pytest.raises(KeyError, match="Carte inconnue"):
        resolve_scenario_map(catalogue, ScenarioMapState("MAP-SYS-999999"), resolver)
    with pytest.raises(FileNotFoundError, match="MAP-SYS-000001.*image"):
        resolve_scenario_map(catalogue, ScenarioMapState(map_id), resolver)

    uncalibrated, uncalibrated_id = _catalogue(SystemCatalogueIdProvider(), calibrated=False)
    _write_image(paths.default_catalogue_maps_dir / "map.jpg")
    with pytest.raises(ValueError, match="n'est pas calibrée"):
        resolve_scenario_map(uncalibrated, ScenarioMapState(uncalibrated_id), resolver)


def test_real_default_map_resolves_and_round_trips_lambert_and_catalogue_cities():
    root = Path(__file__).resolve().parents[1]
    paths = ApplicationPaths.from_runtime(
        installation_root=root,
        user_data_root=root / ".scenario-map-runtime-test-data",
    )
    catalogue = load_catalogue(paths.default_catalogue_path)
    resolved = resolve_scenario_map(
        catalogue,
        ScenarioMapState("MAP-SYS-000001"),
        CatalogueMapAssetResolver(paths),
    )
    assert resolved is not None
    assert resolved.world_rect == resolved.catalogue_map.default_world_rect
    assert resolved.scale_factor == 12.0
    for city_name in ("Bourges", "Strasbourg", "Bordeaux", "Calais"):
        city = next(city for city in catalogue.cities.values() if city.name == city_name)
        lambert = catalogue.get_city_lambert(city.city_id)
        assert resolved.transform.world_to_lambert(*resolved.transform.lambert_to_world(*lambert)) == pytest.approx(
            lambert, abs=1e-6
        )


def test_real_default_map_composes_position_and_scale_without_changing_legacy_defaults():
    root = Path(__file__).resolve().parents[1]
    paths = ApplicationPaths.from_runtime(
        installation_root=root,
        user_data_root=root / ".scenario-map-runtime-test-data",
    )
    catalogue = load_catalogue(paths.default_catalogue_path)
    resolver = CatalogueMapAssetResolver(paths)
    baseline = resolve_scenario_map(catalogue, ScenarioMapState("MAP-SYS-000001"), resolver)
    moved_and_scaled = resolve_scenario_map(
        catalogue,
        ScenarioMapState(
            "MAP-SYS-000001",
            position_override=ScenarioMapPosition(-2500.0, 1800.0),
            scale_factor_override=15.0,
        ),
        resolver,
    )

    assert baseline is not None
    assert moved_and_scaled is not None
    assert baseline.world_rect == baseline.catalogue_map.default_world_rect
    assert baseline.scale_factor == 12.0
    assert moved_and_scaled.world_rect.x0 == -2500.0
    assert moved_and_scaled.world_rect.y0 == 1800.0
    assert moved_and_scaled.world_rect.w == pytest.approx(baseline.world_rect.w * 15.0 / 12.0)
    assert moved_and_scaled.world_rect.h == pytest.approx(
        moved_and_scaled.world_rect.w / (2124.0 / 2613.0)
    )
    assert moved_and_scaled.scale_factor == 15.0
    lambert = (950000.0, 6800000.0)
    world = moved_and_scaled.transform.lambert_to_world(*lambert)
    assert moved_and_scaled.transform.world_to_lambert(*world) == pytest.approx(lambert, abs=1e-6)


def test_real_default_transform_composes_catalogue_calibration_with_y_up_world():
    root = Path(__file__).resolve().parents[1]
    paths = ApplicationPaths.from_runtime(
        installation_root=root,
        user_data_root=root / ".scenario-map-runtime-test-data",
    )
    catalogue = load_catalogue(paths.default_catalogue_path)
    resolved = resolve_scenario_map(
        catalogue,
        ScenarioMapState("MAP-SYS-000001"),
        CatalogueMapAssetResolver(paths),
    )
    assert resolved is not None

    image_width, image_height = resolved.calibrated_map.image_size
    world_rect = resolved.world_rect
    for pixel_x, pixel_y in (
        (0.0, 0.0),
        (float(image_width), 0.0),
        (0.0, float(image_height)),
        (image_width / 2.0, image_height / 2.0),
    ):
        expected = (
            world_rect.x0 + pixel_x * world_rect.w / image_width,
            world_rect.y0 + world_rect.h - pixel_y * world_rect.h / image_height,
        )
        world = resolved.transform.pixel_to_world(pixel_x, pixel_y)
        assert world == pytest.approx(expected, abs=1e-9)
        assert resolved.transform.world_to_pixel(*world) == pytest.approx(
            (pixel_x, pixel_y), abs=1e-9
        )

    for city_name in ("Bourges", "Strasbourg", "Bordeaux", "Calais"):
        city = next(city for city in catalogue.cities.values() if city.name == city_name)
        lambert = catalogue.get_city_lambert(city.city_id)
        world = resolved.transform.lambert_to_world(*lambert)
        assert resolved.transform.world_to_lambert(*world) == pytest.approx(lambert, abs=1e-6)

    assert resolved.scale_factor == 12.0
