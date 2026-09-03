import json

import pytest

from src.assembleur_catalogue import Catalogue, WorldRect
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider, UserCatalogueIdProvider
from src.assembleur_catalogue_io import catalogue_to_dict
from tools.migrate_catalogue_v2_to_v3 import (
    InitialMapDefinition,
    build_delivered_default_map_definition,
    migrate_catalogue_data_v2_to_v3,
    migrate_catalogue_file_v2_to_v3,
    parse_catalogue_v2,
)


def _initial_map() -> InitialMapDefinition:
    return InitialMapDefinition(
        name="899 - Alsace",
        image_file="899 - Alsace.jpg",
        calibration_points_file="899 - Alsace.calib_points.json",
        calibration_file="899 - Alsace.json",
        projection="EPSG:2154",
        default_world_rect=WorldRect(-2963.38, 1642.9905325627296, 4293.6497163554395, 5282.159467437271),
        default_scale_factor=12.0,
    )


def _catalogue_v2(provider=None) -> dict:
    catalogue = Catalogue(id_provider=provider or SystemCatalogueIdProvider())
    opening = catalogue.add_city("Ouverture", 47.0, 2.0)
    base = catalogue.add_city("Base", 46.0, 3.0)
    light = catalogue.add_city("Lumière", 48.0, 4.0)
    beacon = catalogue.add_beacon(opening.city_id)
    triangle = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
    template = catalogue.add_template("Principal")
    catalogue.set_template_rank(template.template_id, 1, triangle.triangle_id)
    data = catalogue_to_dict(catalogue)
    data["version"] = 2
    data["idCounters"].pop("book")
    data["idCounters"].pop("map")
    data.pop("defaultBookId")
    data.pop("books")
    data.pop("defaultMapId")
    data.pop("catalogueReferenceMapId")
    data.pop("maps")
    assert beacon.beacon_id in data["beacons"][0]["beaconId"]
    return data


def test_v2_parser_accepts_current_v2_contract_without_using_v3_loader():
    data = _catalogue_v2()

    parsed = parse_catalogue_v2(data)

    assert parsed.id_counters == {"city": 3, "beacon": 1, "triangle": 1, "template": 1, "map": 0, "book": 1}
    assert parsed.default_map_id is None
    assert parsed.default_book_id == "BOOK-SYS-000001"
    assert set(parsed.cities) == {"CITY-SYS-000001", "CITY-SYS-000002", "CITY-SYS-000003"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.__setitem__("version", 3),
        lambda data: data["idCounters"].pop("template"),
        lambda data: data["idCounters"].__setitem__("map", 0),
        lambda data: data["cities"][0].__setitem__("cityId", "CITY-0001"),
        lambda data: data["beacons"][0].__setitem__("cityId", "CITY-SYS-999999"),
    ],
)
def test_v2_parser_rejects_invalid_contracts(mutate):
    data = _catalogue_v2()
    mutate(data)

    with pytest.raises(ValueError):
        parse_catalogue_v2(data)


def test_migration_adds_only_the_initial_sys_map_and_preserves_existing_ids():
    source = _catalogue_v2()
    migrated = migrate_catalogue_data_v2_to_v3(source, initial_map=_initial_map())

    assert migrated["version"] == 3
    assert migrated["idCounters"] == {"city": 3, "beacon": 1, "triangle": 1, "template": 1, "map": 1}
    assert migrated["defaultMapId"] == "MAP-SYS-000001"
    assert migrated["cities"] == source["cities"]
    assert migrated["beacons"] == source["beacons"]
    assert migrated["triangles"] == source["triangles"]
    assert migrated["templates"] == source["templates"]
    assert migrated["defaultTemplateId"] == source["defaultTemplateId"]
    assert migrated["maps"] == [{
        "mapId": "MAP-SYS-000001",
        "name": "899 - Alsace",
        "imageFile": "899 - Alsace.jpg",
        "calibrationPointsFile": "899 - Alsace.calib_points.json",
        "calibrationFile": "899 - Alsace.json",
        "projection": "EPSG:2154",
        "defaultWorldRect": {
            "x0": -2963.38,
            "y0": 1642.9905325627296,
            "w": 4293.6497163554395,
            "h": 5282.159467437271,
        },
        "defaultScaleFactor": 12.0,
        "archived": False,
    }]


def test_user_v2_migration_keeps_user_ids_and_receives_the_sys_map():
    source = _catalogue_v2(UserCatalogueIdProvider())

    migrated = migrate_catalogue_data_v2_to_v3(source, initial_map=_initial_map())

    assert all("-USR-" in item["cityId"] for item in migrated["cities"])
    assert "-USR-" in migrated["beacons"][0]["beaconId"]
    assert "-USR-" in migrated["triangles"][0]["triangleId"]
    assert "-USR-" in migrated["templates"][0]["templateId"]
    assert migrated["maps"][0]["mapId"] == "MAP-SYS-000001"
    assert migrated["idCounters"]["map"] == 1


def test_delivered_default_map_definition_reads_config_pose_and_audits_assets(tmp_path):
    maps_dir = tmp_path / "resources" / "maps"
    maps_dir.mkdir(parents=True)
    for asset in ("899 - Alsace.jpg", "899 - Alsace.calib_points.json", "899 - Alsace.json"):
        (maps_dir / asset).write_bytes(b"asset")
    config = tmp_path / "assembleur_config.json"
    config.write_text(json.dumps({
        "bgMap": "899 - Alsace.jpg",
        "bgWorldRect": {"x0": 1, "y0": 2, "w": 3, "h": 4},
    }), encoding="utf-8")

    definition = build_delivered_default_map_definition(config_path=config, resource_maps_dir=maps_dir)

    assert definition.default_world_rect == WorldRect(1.0, 2.0, 3.0, 4.0)
    assert definition.default_scale_factor == 12.0
    assert definition.projection == "EPSG:2154"


def test_file_migration_creates_backup_keeps_v2_backup_and_refuses_second_run(tmp_path):
    source = tmp_path / "catalogue.json"
    original = _catalogue_v2()
    source.write_text(json.dumps(original), encoding="utf-8")

    result = migrate_catalogue_file_v2_to_v3(source, initial_map=_initial_map())

    assert result.destination == source
    assert result.backup == tmp_path / "catalogue.v2.pre-catalog-map-001c1.json"
    assert json.loads(result.backup.read_text(encoding="utf-8")) == original
    assert json.loads(source.read_text(encoding="utf-8"))["version"] == 3
    with pytest.raises(FileExistsError, match="Backup de migration déjà existant"):
        migrate_catalogue_file_v2_to_v3(source, initial_map=_initial_map())


def test_file_migration_failure_preserves_source_and_does_not_create_backup(tmp_path):
    source = tmp_path / "catalogue.json"
    invalid = _catalogue_v2()
    invalid["idCounters"].pop("city")
    original_text = json.dumps(invalid)
    source.write_text(original_text, encoding="utf-8")

    with pytest.raises(ValueError):
        migrate_catalogue_file_v2_to_v3(source, initial_map=_initial_map())

    assert source.read_text(encoding="utf-8") == original_text
    assert not (tmp_path / "catalogue.v2.pre-catalog-map-001c1.json").exists()


def test_file_migration_to_distinct_destination_does_not_change_source(tmp_path):
    source = tmp_path / "catalogue-v2.json"
    destination = tmp_path / "catalogue-v3.json"
    original = _catalogue_v2()
    source.write_text(json.dumps(original), encoding="utf-8")

    result = migrate_catalogue_file_v2_to_v3(
        source,
        output_path=destination,
        initial_map=_initial_map(),
    )

    assert result.backup is None
    assert json.loads(source.read_text(encoding="utf-8")) == original
    assert json.loads(destination.read_text(encoding="utf-8"))["version"] == 3


def test_file_migration_refuses_an_already_v3_source_explicitly(tmp_path):
    source = tmp_path / "catalogue-v3.json"
    source.write_text(json.dumps(migrate_catalogue_data_v2_to_v3(_catalogue_v2(), initial_map=_initial_map())), encoding="utf-8")

    with pytest.raises(ValueError, match="Catalogue V2 attendu"):
        migrate_catalogue_file_v2_to_v3(
            source,
            output_path=tmp_path / "other.json",
            initial_map=_initial_map(),
        )
