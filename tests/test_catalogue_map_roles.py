import json
from pathlib import Path

import pytest

from src.assembleur_catalogue import Catalogue, WorldRect
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_catalogue_io import catalogue_from_dict, catalogue_to_dict, load_catalogue
from src.assembleur_catalogue_map_assets import CatalogueMapAssetResolver, load_calibrated_catalogue_map
from src.assembleur_catalogue_map_calibration import CatalogueMapCalibrationController, STATUS_VALID
from src.assembleur_paths import ApplicationPaths
from src import assembleur_catalogue_window as catalogue_window_module
from src.assembleur_catalogue_window import CatalogueWindow
from tools.migrate_catalogue_v3_to_v4 import (
    migrate_catalogue_data_v3_to_v4,
    migrate_catalogue_file_v3_to_v4,
)
from tools.migrate_catalogue_v4_to_v5 import migrate_catalogue_data_v4_to_v5


def _maps_catalogue(*, calibrated_reference: bool = True) -> tuple[Catalogue, str, str]:
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    default_id = catalogue.add_map(
        name="Assemblage", image_file="assembly.jpg", calibration_file="assembly.json",
        projection="EPSG:2154", default_world_rect=WorldRect(0, 0, 2, 1), default_scale_factor=12,
    )
    reference_id = catalogue.add_map(
        name="Référence", image_file="reference.jpg",
        calibration_file="reference.json" if calibrated_reference else None,
        projection="EPSG:2154" if calibrated_reference else None,
        default_world_rect=WorldRect(0, 0, 2, 1), default_scale_factor=1,
    )
    return catalogue, default_id, reference_id


def _as_v3_fixture(data: dict) -> dict:
    data["version"] = 3
    data["idCounters"].pop("book")
    data.pop("defaultBookId")
    data.pop("books")
    data.pop("catalogueReferenceMapId")
    data["maps"] = data["maps"][:1]
    for catalogue_map in data["maps"]:
        catalogue_map["calibrationPointsFile"] = None
        catalogue_map.pop("description")
        catalogue_map.pop("calibrationCityIds")
    data["idCounters"]["map"] = 1
    return data


def test_catalogue_has_two_independent_map_roles() -> None:
    catalogue, default_id, reference_id = _maps_catalogue()
    catalogue.set_default_map(default_id)
    catalogue.set_catalogue_reference_map(reference_id)

    assert catalogue.default_map_id == default_id
    assert catalogue.catalogue_reference_map_id == reference_id
    catalogue.validate()


def test_catalogue_reference_role_rejects_unknown_or_uncalibrated_map() -> None:
    catalogue, _default_id, reference_id = _maps_catalogue(calibrated_reference=False)
    with pytest.raises(ValueError, match="calibrée"):
        catalogue.set_catalogue_reference_map(reference_id)
    catalogue.catalogue_reference_map_id = "MAP-SYS-999999"
    with pytest.raises(ValueError, match="absente"):
        catalogue.validate()


def test_v3_to_v4_migration_adds_the_reference_role_and_system_map() -> None:
    catalogue, default_id, _reference_id = _maps_catalogue()
    catalogue.set_default_map(default_id)
    data = catalogue_to_dict(catalogue)
    data = _as_v3_fixture(data)

    migrated = migrate_catalogue_data_v3_to_v4(data)
    loaded = catalogue_from_dict(migrate_catalogue_data_v4_to_v5(migrated), id_provider=SystemCatalogueIdProvider())

    assert migrated["version"] == 4
    assert loaded.default_map_id == "MAP-SYS-000001"
    assert loaded.catalogue_reference_map_id == "MAP-SYS-000002"
    assert loaded.id_counters["map"] == 2


def test_v3_to_v4_file_migration_keeps_a_v3_backup(tmp_path) -> None:
    catalogue, default_id, _reference_id = _maps_catalogue()
    catalogue.set_default_map(default_id)
    source = catalogue_to_dict(catalogue)
    source = _as_v3_fixture(source)
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    backup = migrate_catalogue_file_v3_to_v4(path)

    assert json.loads(backup.read_text(encoding="utf-8")) == source
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["version"] == 4
    assert migrated["catalogueReferenceMapId"] == "MAP-SYS-000002"


def test_delivered_reference_map_resolves_calibration_and_city_coordinates() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = ApplicationPaths.from_runtime(installation_root=root, user_data_root=root / ".pytest-user-root", catalogue_mode="SYS")
    catalogue = load_catalogue(paths.default_catalogue_path)
    reference_id = catalogue.catalogue_reference_map_id
    assert reference_id == "MAP-SYS-000002"
    reference_map = catalogue.get_map(reference_id)

    assets = CatalogueMapAssetResolver(paths).resolve(reference_map)
    calibrated_map = load_calibrated_catalogue_map(reference_map, CatalogueMapAssetResolver(paths))
    lambert = catalogue.get_city_lambert(next(iter(catalogue.cities)))
    pixel = calibrated_map.lambert_to_pixel(*lambert)

    assert assets.image_path == paths.default_catalogue_maps_dir / "france_michelin.jpg"
    assert assets.calibration_path == paths.default_catalogue_maps_dir / "france_michelin.json"
    assert calibrated_map.map_id == reference_id
    assert 0 <= pixel[0] <= calibrated_map.image_size[0]
    assert 0 <= pixel[1] <= calibrated_map.image_size[1]


def test_delivered_system_maps_have_five_pointed_calibration_cities() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = ApplicationPaths.from_runtime(installation_root=root, user_data_root=root / ".pytest-user-root", catalogue_mode="SYS")
    catalogue = load_catalogue(paths.default_catalogue_path)
    controller = CatalogueMapCalibrationController(catalogue, paths)

    for map_id in ("MAP-SYS-000001", "MAP-SYS-000002"):
        catalogue_map = catalogue.get_map(map_id)
        assert len(catalogue_map.calibration_city_ids) == 5
        assert len(controller.points_for(catalogue_map)) == 5
        assert controller.status_for(catalogue_map) == STATUS_VALID


def test_catalogue_window_loads_the_catalogue_reference_role(monkeypatch) -> None:
    catalogue, default_id, reference_id = _maps_catalogue()
    catalogue.set_default_map(default_id)
    catalogue.set_catalogue_reference_map(reference_id)
    loaded_maps = []
    calibrated_map = object()

    def load_reference(map_definition, _resolver):
        loaded_maps.append(map_definition.map_id)
        return calibrated_map

    class MapView:
        def __init__(self) -> None:
            self.map = None

        def set_map(self, value) -> None:
            self.map = value

    monkeypatch.setattr(catalogue_window_module, "load_calibrated_catalogue_map", load_reference)
    window = object.__new__(CatalogueWindow)
    window.catalogue = catalogue
    window._map_view = MapView()
    window._beacon_map_view = MapView()
    window._triangle_map_view = MapView()

    CatalogueWindow._load_map(window)

    assert loaded_maps == [reference_id]
    assert [view.map for view in (window._map_view, window._beacon_map_view, window._triangle_map_view)] == [
        calibrated_map,
        calibrated_map,
        calibrated_map,
    ]
