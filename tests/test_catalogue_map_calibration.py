import json
import math

import numpy as np
import pytest
from PIL import Image

from src.assembleur_catalogue import Catalogue, WorldRect
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider, UserCatalogueIdProvider
from src.assembleur_catalogue_io import catalogue_from_dict, catalogue_to_dict, load_catalogue
from src.assembleur_catalogue_map_calibration import (
    CatalogueMapCalibrationController,
    STATUS_INCOMPLETE,
    STATUS_UNCALIBRATED,
    STATUS_VALID,
)
from src.assembleur_paths import ApplicationPaths
from tools.migrate_catalogue_v4_to_v5 import migrate_catalogue_data_v4_to_v5


def _catalogue_with_map(tmp_path):
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    cities = [
        catalogue.add_city("A", 47.0, 2.0),
        catalogue.add_city("B", 48.0, 3.0),
        catalogue.add_city("C", 46.5, 4.0),
    ]
    map_id = catalogue.add_map(
        name="Utilisateur", image_file="map.png", calibration_file="map.json", projection=None,
        default_world_rect=WorldRect(0, 0, 100, 100), default_scale_factor=1,
        description="Une carte de test", calibration_city_ids=[city.city_id for city in cities],
    )
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path, user_data_root=tmp_path / "user")
    paths.user_catalogue_maps_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(paths.user_catalogue_maps_dir / "map.png")
    (paths.user_catalogue_maps_dir / "map.json").write_text(json.dumps({"points": []}), encoding="utf-8")
    return catalogue, map_id, cities, paths


def _five_point_affine_controller(tmp_path):
    catalogue, map_id, cities, paths = _catalogue_with_map(tmp_path)
    for name, latitude, longitude in (("D", 47.5, 4.5), ("E", 46.8, 2.8)):
        city = catalogue.add_city(name, latitude, longitude)
        cities.append(city)
        catalogue.update_map(
            map_id,
            calibration_city_ids=[*catalogue.get_map(map_id).calibration_city_ids, city.city_id],
        )
    controller = CatalogueMapCalibrationController(catalogue, paths)
    matrix = np.asarray(((0.00012, -0.00003), (0.00004, 0.00009)))
    offset = np.asarray((-32.0, 91.0))
    points = []
    for city in cities:
        lambert = np.asarray(catalogue.get_city_lambert(city.city_id))
        pixel_x, pixel_y = matrix @ lambert + offset
        points.append({"cityId": city.city_id, "pixelX": float(pixel_x), "pixelY": float(pixel_y)})
    controller._documents[map_id] = {"points": points}
    return catalogue, map_id, cities, controller, matrix, offset


def test_description_and_calibration_city_ids_round_trip(tmp_path):
    catalogue, _map_id, cities, _paths = _catalogue_with_map(tmp_path)
    restored = catalogue_from_dict(catalogue_to_dict(catalogue), id_provider=UserCatalogueIdProvider())

    item = next(iter(restored.maps.values()))
    assert item.description == "Une carte de test"
    assert item.calibration_city_ids == [city.city_id for city in cities]


def test_calibration_city_ids_reject_unknown_and_duplicates(tmp_path):
    catalogue, map_id, cities, _paths = _catalogue_with_map(tmp_path)
    with pytest.raises(ValueError, match="absente"):
        catalogue.update_map(map_id, calibration_city_ids=["CITY-SYS-999999"])
    with pytest.raises(ValueError, match="dupliquée"):
        catalogue.update_map(map_id, calibration_city_ids=[cities[0].city_id, cities[0].city_id])


def test_status_and_pointing_recalculate_a_calibration(tmp_path):
    catalogue, map_id, cities, paths = _catalogue_with_map(tmp_path)
    controller = CatalogueMapCalibrationController(catalogue, paths)
    catalogue_map = catalogue.get_map(map_id)

    assert controller.status_for(catalogue_map) == STATUS_INCOMPLETE
    controller.remove_city(map_id, cities[2].city_id)
    controller.remove_city(map_id, cities[1].city_id)
    assert controller.status_for(catalogue_map) == STATUS_INCOMPLETE
    controller.remove_city(map_id, cities[0].city_id)
    assert controller.status_for(catalogue_map) == STATUS_UNCALIBRATED

    for city in cities:
        controller.add_city(map_id, city.city_id)
    controller.set_pixel(map_id, cities[0].city_id, 10, 10)
    controller.set_pixel(map_id, cities[1].city_id, 60, 20)
    assert controller.status_for(catalogue_map) == STATUS_INCOMPLETE
    controller.set_pixel(map_id, cities[2].city_id, 30, 70)
    assert controller.status_for(catalogue_map) == STATUS_VALID
    assert set(controller.points_for(catalogue_map)) == {city.city_id for city in cities}
    controller.commit()
    stored = json.loads((paths.user_catalogue_maps_dir / "map.json").read_text(encoding="utf-8"))
    assert {point["cityId"] for point in stored["points"]} == {city.city_id for city in cities}
    assert "A" in stored and "offset" in stored


def test_preview_uses_uncommitted_calibration_and_keeps_disk_unchanged(tmp_path):
    catalogue, map_id, cities, paths = _catalogue_with_map(tmp_path)
    controller = CatalogueMapCalibrationController(catalogue, paths)
    calibration_path = paths.user_catalogue_maps_dir / "map.json"
    disk_before = calibration_path.read_bytes()

    for city, pixel in zip(cities, ((10, 10), (60, 20), (30, 70))):
        controller.set_pixel(map_id, city.city_id, *pixel)

    preview = controller.preview_map(catalogue.get_map(map_id))
    assert preview.calibration["A"] == controller._documents[map_id]["A"]
    assert calibration_path.read_bytes() == disk_before

    controller.commit()
    persisted = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert persisted["A"] == preview.calibration["A"]


def test_overdetermined_calibration_recomputes_residuals_for_every_city(tmp_path):
    catalogue, map_id, cities, paths = _catalogue_with_map(tmp_path)
    extra_cities = [catalogue.add_city("D", 47.5, 4.5), catalogue.add_city("E", 46.8, 2.8)]
    for city in extra_cities:
        catalogue.update_map(
            map_id,
            calibration_city_ids=[*catalogue.get_map(map_id).calibration_city_ids, city.city_id],
        )
    cities = [*cities, *extra_cities]
    controller = CatalogueMapCalibrationController(catalogue, paths)
    for city, pixel in zip(cities, ((10, 10), (60, 20), (30, 70), (55, 55), (22, 44))):
        controller.set_pixel(map_id, city.city_id, *pixel)
    catalogue_map = catalogue.get_map(map_id)
    before = controller.leave_one_out_residuals(catalogue_map)

    controller.set_pixel(map_id, cities[0].city_id, 18, 13)
    after = controller.leave_one_out_residuals(catalogue_map)

    assert set(after) == {city.city_id for city in cities}
    assert any(after[city.city_id].error_px != before[city.city_id].error_px for city in cities[1:])


def test_solve_affine_recovers_a_known_transformation(tmp_path):
    catalogue, map_id, _cities, controller, matrix, offset = _five_point_affine_controller(tmp_path)

    solved_matrix, solved_offset = controller._solve_affine(
        list(controller.points_for(catalogue.get_map(map_id)).values())
    )

    assert solved_matrix == pytest.approx(matrix)
    assert solved_offset == pytest.approx(offset)


def test_leave_one_out_is_zero_for_a_perfect_affine(tmp_path):
    catalogue, map_id, cities, controller, _matrix, _offset = _five_point_affine_controller(tmp_path)

    residuals = controller.leave_one_out_residuals(catalogue.get_map(map_id))

    assert set(residuals) == {city.city_id for city in cities}
    assert all(residual.error_px == pytest.approx(0.0, abs=1e-7) for residual in residuals.values())


def test_leave_one_out_identifies_a_displaced_observation(tmp_path):
    catalogue, map_id, cities, controller, _matrix, _offset = _five_point_affine_controller(tmp_path)
    controller._documents[map_id]["points"][0]["pixelX"] += 20.0
    controller._documents[map_id]["points"][0]["pixelY"] -= 10.0

    residuals = controller.leave_one_out_residuals(catalogue.get_map(map_id))

    assert residuals[cities[0].city_id].error_px == pytest.approx(math.hypot(20.0, -10.0), abs=1e-7)


def test_leave_one_out_differs_from_the_global_residual(tmp_path):
    catalogue, map_id, cities, controller, _matrix, _offset = _five_point_affine_controller(tmp_path)
    controller._documents[map_id]["points"][0]["pixelX"] += 20.0
    catalogue_map = catalogue.get_map(map_id)
    points = controller.points_for(catalogue_map)
    global_matrix, global_offset = controller._solve_affine(list(points.values()))
    lambert = np.asarray(catalogue.get_city_lambert(cities[0].city_id))
    global_error = float(np.hypot(*(global_matrix @ lambert + global_offset - np.asarray((points[cities[0].city_id].pixel_x, points[cities[0].city_id].pixel_y)))))

    residual = controller.leave_one_out_residuals(catalogue_map)[cities[0].city_id]

    assert residual.error_px != pytest.approx(global_error)


def test_leave_one_out_is_unavailable_with_only_three_observations(tmp_path):
    catalogue, map_id, _cities, controller, _matrix, _offset = _five_point_affine_controller(tmp_path)
    controller._documents[map_id]["points"] = controller._documents[map_id]["points"][:3]

    assert controller.leave_one_out_residuals(catalogue.get_map(map_id)) == {}


def test_alsace_leave_one_out_residuals_are_finite():
    paths = ApplicationPaths.from_runtime()
    catalogue = load_catalogue(paths.default_catalogue_path)
    catalogue_map = catalogue.get_map("MAP-SYS-000001")
    controller = CatalogueMapCalibrationController(catalogue, paths)

    residuals = controller.leave_one_out_residuals(catalogue_map)

    assert set(residuals) == set(catalogue_map.calibration_city_ids)
    assert all(np.isfinite(residual.error_px) and residual.error_px >= 0.0 for residual in residuals.values())


def test_v4_to_v5_migration_adds_empty_fields_to_user_maps():
    v4 = {
        "version": 4, "idCounters": {"city": 0, "beacon": 0, "triangle": 0, "template": 0, "map": 1},
        "defaultTemplateId": None, "defaultMapId": None, "catalogueReferenceMapId": None,
        "cities": [], "beacons": [], "triangles": [], "templates": [],
        "maps": [{
            "mapId": "MAP-SYS-000001", "name": "Carte", "imageFile": "map.jpg",
            "calibrationPointsFile": None, "calibrationFile": None, "projection": None,
            "defaultWorldRect": {"x0": 0, "y0": 0, "w": 1, "h": 1},
            "defaultScaleFactor": 1, "archived": False,
        }],
    }
    migrated = migrate_catalogue_data_v4_to_v5(v4)
    assert migrated["version"] == 5
    assert migrated["maps"][0]["description"] == "Carte d'assemblage Vosges"
    assert migrated["maps"][0]["calibrationCityIds"] == []
    assert "calibrationPointsFile" not in migrated["maps"][0]


def test_user_map_assets_stay_staged_until_commit(tmp_path):
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path, user_data_root=tmp_path / "user")
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(source)
    controller = CatalogueMapCalibrationController(catalogue, paths)

    map_id = controller.stage_user_map(source, name="Nouvelle", description="Brouillon")
    catalogue_map = catalogue.get_map(map_id)
    assert map_id.startswith("MAP-USR-")
    assert not (paths.user_catalogue_maps_dir / catalogue_map.image_file).exists()
    controller.discard()
    assert not (paths.user_catalogue_maps_dir / catalogue_map.image_file).exists()


def test_system_map_editing_permission_is_explicit(tmp_path):
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path, user_data_root=tmp_path / "user")
    system_catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    system_map_id = system_catalogue.add_map(
        name="SYS", image_file="map.png", calibration_file=None, projection=None,
        default_world_rect=WorldRect(0, 0, 1, 1), default_scale_factor=1,
    )
    user_catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    user_map_id = user_catalogue.add_map(
        name="USR", image_file="map.png", calibration_file=None, projection=None,
        default_world_rect=WorldRect(0, 0, 1, 1), default_scale_factor=1,
    )

    assert CatalogueMapCalibrationController(system_catalogue, paths).is_readonly(system_catalogue.get_map(system_map_id))
    assert not CatalogueMapCalibrationController(
        system_catalogue, paths, allow_system_map_editing=True
    ).is_readonly(system_catalogue.get_map(system_map_id))
    assert not CatalogueMapCalibrationController(user_catalogue, paths).is_readonly(user_catalogue.get_map(user_map_id))


def test_sys_calibration_recalculation_persists_modern_a_offset(tmp_path):
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path, user_data_root=tmp_path / "user")
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    cities = [
        catalogue.add_city("A", 47.0, 2.0),
        catalogue.add_city("B", 48.0, 3.0),
        catalogue.add_city("C", 46.5, 4.0),
    ]
    map_id = catalogue.add_map(
        name="Alsace SYS", image_file="map.png", calibration_file="map.json", projection=None,
        default_world_rect=WorldRect(0, 0, 100, 100), default_scale_factor=1,
        calibration_city_ids=[city.city_id for city in cities],
    )
    paths.default_catalogue_maps_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(paths.default_catalogue_maps_dir / "map.png")
    (paths.default_catalogue_maps_dir / "map.json").write_text('{"points": []}', encoding="utf-8")
    controller = CatalogueMapCalibrationController(catalogue, paths, allow_system_map_editing=True)
    for city, pixel in zip(cities, ((10, 10), (60, 20), (30, 70))):
        controller.set_pixel(map_id, city.city_id, *pixel)
    controller.commit()

    persisted = json.loads((paths.default_catalogue_maps_dir / "map.json").read_text(encoding="utf-8"))
    assert persisted["projection"] == "EPSG:2154"
    assert "A" in persisted and "offset" in persisted
    assert catalogue.get_map(map_id).projection == "EPSG:2154"
