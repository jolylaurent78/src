import json
from pathlib import Path

import pytest
from PIL import Image

from src.assembleur_catalogue import CatalogueMap, WorldRect
from src.assembleur_catalogue_map_assets import (
    CatalogueMapAssetResolver,
    load_calibrated_catalogue_map,
)
from src.assembleur_catalogue_io import load_catalogue
from src.assembleur_geo_map_view import CalibratedGeoMap
from src.assembleur_paths import ApplicationPaths


def _paths(tmp_path: Path) -> ApplicationPaths:
    return ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user-root",
    )


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), "white").save(path)


def _write_calibration(path: Path, *, projection: str = "EPSG:2154", matrix=None, offset=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "projection": projection,
        "A": matrix if matrix is not None else [[0.01, 0.0], [0.0, -0.01]],
        "offset": offset if offset is not None else [5.0, 6.0],
    }), encoding="utf-8")


def _map(map_id: str, *, points: str | None = "points.json", calibration: str | None = "calibration.json", projection: str | None = "EPSG:2154") -> CatalogueMap:
    return CatalogueMap(
        map_id=map_id,
        name="Carte",
        image_file="image.jpg",
        calibration_points_file=points,
        calibration_file=calibration,
        projection=projection,
        default_world_rect=WorldRect(0, 0, 10, 10),
        default_scale_factor=1.0,
    )


def test_resolver_resolves_system_assets_only_from_default_catalogue_maps(tmp_path):
    paths = _paths(tmp_path)
    _write_image(paths.default_catalogue_maps_dir / "image.jpg")
    (paths.default_catalogue_maps_dir / "points.json").parent.mkdir(parents=True, exist_ok=True)
    (paths.default_catalogue_maps_dir / "points.json").write_text("{}", encoding="utf-8")
    _write_calibration(paths.default_catalogue_maps_dir / "calibration.json")

    resolved = CatalogueMapAssetResolver(paths).resolve(_map("MAP-SYS-000001"))

    assert resolved.image_path == (paths.default_catalogue_maps_dir / "image.jpg").resolve()
    assert resolved.calibration_points_path == (paths.default_catalogue_maps_dir / "points.json").resolve()
    assert resolved.calibration_path == (paths.default_catalogue_maps_dir / "calibration.json").resolve()


def test_resolver_resolves_user_assets_only_from_catalogue_maps_store(tmp_path):
    paths = _paths(tmp_path)
    paths.ensure_user_data_directories()
    _write_image(paths.user_catalogue_maps_dir / "image.jpg")
    (paths.user_catalogue_maps_dir / "points.json").write_text("{}", encoding="utf-8")
    _write_calibration(paths.user_catalogue_maps_dir / "calibration.json")

    resolved = CatalogueMapAssetResolver(paths).resolve(
        _map("MAP-USR-550e8400-e29b-41d4-a716-446655440000")
    )

    assert resolved.image_path == (paths.user_catalogue_maps_dir / "image.jpg").resolve()
    assert resolved.calibration_path == (paths.user_catalogue_maps_dir / "calibration.json").resolve()


def test_resolver_returns_none_for_optional_assets(tmp_path):
    paths = _paths(tmp_path)
    _write_image(paths.default_catalogue_maps_dir / "image.jpg")

    resolved = CatalogueMapAssetResolver(paths).resolve(
        _map("MAP-SYS-000001", points=None, calibration=None, projection=None)
    )

    assert resolved.calibration_points_path is None
    assert resolved.calibration_path is None


def test_resolver_never_falls_back_between_system_and_user_asset_roots(tmp_path):
    paths = _paths(tmp_path)
    _write_image(paths.resource_maps_dir / "image.jpg")
    _write_image(paths.user_catalogue_maps_dir / "image.jpg")

    with pytest.raises(FileNotFoundError, match="MAP-SYS-000001.*image"):
        CatalogueMapAssetResolver(paths).resolve(_map("MAP-SYS-000001"))

    user_paths = _paths(tmp_path / "other-installation")
    _write_image(user_paths.resource_maps_dir / "image.jpg")
    _write_image(user_paths.default_catalogue_maps_dir / "image.jpg")
    with pytest.raises(FileNotFoundError, match="MAP-USR-550e8400-e29b-41d4-a716-446655440000.*image"):
        CatalogueMapAssetResolver(user_paths).resolve(
            _map("MAP-USR-550e8400-e29b-41d4-a716-446655440000")
        )


@pytest.mark.parametrize(
    ("field", "expected_role"),
    [
        ("image", "image"),
        ("points", "calibration points"),
        ("calibration", "calibration"),
    ],
)
def test_resolver_fails_explicitly_for_missing_assets(tmp_path, field, expected_role):
    paths = _paths(tmp_path)
    if field != "image":
        _write_image(paths.default_catalogue_maps_dir / "image.jpg")
    if field == "calibration":
        (paths.default_catalogue_maps_dir / "points.json").parent.mkdir(parents=True, exist_ok=True)
        (paths.default_catalogue_maps_dir / "points.json").write_text("{}", encoding="utf-8")
    if field == "points":
        _write_calibration(paths.default_catalogue_maps_dir / "calibration.json")

    with pytest.raises(FileNotFoundError, match=rf"MAP-SYS-000001.*{expected_role}"):
        CatalogueMapAssetResolver(paths).resolve(_map("MAP-SYS-000001"))


def test_resolver_never_escapes_its_asset_root(tmp_path):
    paths = _paths(tmp_path)
    outside = tmp_path / "outside.jpg"
    _write_image(outside)
    catalogue_map = _map("MAP-SYS-000001")
    catalogue_map.image_file = "../outside.jpg"

    with pytest.raises(ValueError, match="hors de sa racine"):
        CatalogueMapAssetResolver(paths).resolve(catalogue_map)


def test_resolver_rejects_non_map_namespace(tmp_path):
    with pytest.raises(ValueError, match="Identifiant CatalogueMap invalide"):
        CatalogueMapAssetResolver(_paths(tmp_path)).resolve(_map("CITY-SYS-000001"))


def test_load_from_assets_validates_files_and_calibration_contract(tmp_path):
    image = tmp_path / "image.jpg"
    calibration = tmp_path / "calibration.json"
    _write_image(image)
    _write_calibration(calibration)

    geo_map = CalibratedGeoMap.load_from_assets(
        map_id="MAP-SYS-000001", image_path=image, calibration_path=calibration
    )
    assert geo_map.map_id == "MAP-SYS-000001"

    with pytest.raises(FileNotFoundError, match="Image de carte"):
        CalibratedGeoMap.load_from_assets(
            map_id="MAP-SYS-000001", image_path=tmp_path / "missing.jpg", calibration_path=calibration
        )
    with pytest.raises(FileNotFoundError, match="Calibration de carte"):
        CalibratedGeoMap.load_from_assets(
            map_id="MAP-SYS-000001", image_path=image, calibration_path=tmp_path / "missing.json"
        )
    with pytest.raises(ValueError, match="max_image_dimension"):
        CalibratedGeoMap.load_from_assets(
            map_id="MAP-SYS-000001", image_path=image, calibration_path=calibration, max_image_dimension=True
        )


def test_load_from_assets_round_trips_lambert_and_geographic_coordinates(tmp_path):
    image = tmp_path / "image.jpg"
    calibration = tmp_path / "calibration.json"
    _write_image(image)
    _write_calibration(calibration)

    geo_map = CalibratedGeoMap.load_from_assets(
        map_id="MAP-SYS-000001", image_path=image, calibration_path=calibration
    )
    lambert = (700_000.0, 6_600_000.0)
    pixel = geo_map.lambert_to_pixel(*lambert)
    geographic = geo_map.lambert_to_geographic(*lambert)

    assert geo_map.pixel_to_lambert(*pixel) == pytest.approx(lambert, abs=1e-6)
    assert geo_map.pixel_to_geographic(*pixel) == pytest.approx(geographic, abs=1e-9)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{", "JSON invalide"),
        ({"A": [[1, 2, 3]], "offset": [0, 0]}, "A ou offset"),
        ({"A": [[1, 0], [0, 1]], "offset": [0]}, "A ou offset"),
        ({"A": [[1, 0], [0, 0]], "offset": [0, 0]}, "non inversible"),
    ],
)
def test_load_from_assets_rejects_invalid_calibrations(tmp_path, payload, message):
    image = tmp_path / "image.jpg"
    calibration = tmp_path / "calibration.json"
    _write_image(image)
    calibration.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        CalibratedGeoMap.load_from_assets(
            map_id="MAP-SYS-000001", image_path=image, calibration_path=calibration
        )


def test_default_catalogue_map_resolves_loads_and_round_trips_lambert():
    root = Path(__file__).resolve().parents[1]
    paths = ApplicationPaths.from_runtime(
        installation_root=root,
        user_data_root=root / ".pytest-user-root",
    )
    catalogue = load_catalogue(paths.default_catalogue_path)
    catalogue_map = catalogue.get_map(catalogue.default_map_id)

    assets = CatalogueMapAssetResolver(paths).resolve(catalogue_map)
    geo_map = load_calibrated_catalogue_map(catalogue_map, CatalogueMapAssetResolver(paths))
    source = (950000.0, 6800000.0)
    pixel = geo_map.lambert_to_pixel(*source)

    assert assets.image_path == paths.default_catalogue_maps_dir / "899 - Alsace.jpg"
    assert assets.calibration_points_path == paths.default_catalogue_maps_dir / "899 - Alsace.calib_points.json"
    assert assets.calibration_path == paths.default_catalogue_maps_dir / "899 - Alsace.json"
    assert geo_map.map_id == "MAP-SYS-000001"
    assert geo_map.pixel_to_lambert(*pixel) == pytest.approx(source, abs=1e-6)


def test_catalogue_map_loader_rejects_an_uncalibrated_map(tmp_path):
    paths = _paths(tmp_path)
    _write_image(paths.default_catalogue_maps_dir / "image.jpg")
    catalogue_map = _map("MAP-SYS-000001", points=None, calibration=None, projection=None)

    with pytest.raises(ValueError, match="n'est pas calibrée"):
        load_calibrated_catalogue_map(catalogue_map, CatalogueMapAssetResolver(paths))


def test_catalogue_map_loader_rejects_incoherent_projection(tmp_path):
    paths = _paths(tmp_path)
    _write_image(paths.default_catalogue_maps_dir / "image.jpg")
    (paths.default_catalogue_maps_dir / "points.json").parent.mkdir(parents=True, exist_ok=True)
    (paths.default_catalogue_maps_dir / "points.json").write_text("{}", encoding="utf-8")
    _write_calibration(paths.default_catalogue_maps_dir / "calibration.json", projection="EPSG:4326")

    with pytest.raises(ValueError, match="projection non supportée"):
        load_calibrated_catalogue_map(_map("MAP-SYS-000001"), CatalogueMapAssetResolver(paths))
