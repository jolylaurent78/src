import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_catalogue_io import catalogue_from_dict, load_catalogue
from tools.migrate_catalogue_v1_to_v2 import (
    build_catalogue_mappings,
    migrate_catalogue_data_v1_to_v2,
    migrate_paths,
    migrate_scenario_xml,
)


def _legacy_catalogue():
    return {
        "version": 1,
        "defaultTemplateId": "TPL-0001",
        "cities": [
            {"cityId": "CITY-0001", "name": "A", "latitude": 47.0, "longitude": 2.0, "archived": False},
            {"cityId": "CITY-0002", "name": "B", "latitude": 46.0, "longitude": 3.0, "archived": False},
            {"cityId": "CITY-0042", "name": "C", "latitude": 45.0, "longitude": 4.0, "archived": False},
        ],
        "beacons": [{"beaconId": "BEA-0007", "cityId": "CITY-0001", "archived": False}],
        "triangles": [{
            "triangleId": "TRI-0042", "note": "Do", "openingCityId": "CITY-0001",
            "baseCityId": "CITY-0002", "lightCityId": "CITY-0042", "archived": False,
        }],
        "templates": [{
            "templateId": "TPL-0001", "name": "Principal", "description": "", "archived": False,
            "triangleIdsByRank": [None] * 32,
        }],
    }


def _scenario_xml() -> str:
    return """<?xml version='1.0' encoding='utf-8'?>
<scenario version="6" topo_tx_orientation="cw">
  <scenarioReference><cities>
    <city cityRefId="SCITY-0001" name="Locale" latitude="47" longitude="2" catalogueSourceCityId="CITY-0001" />
  </cities><triangles>
    <triangle triangleRefId="STRI-0001" note="Local" openingCityRefId="SCITY-0001" baseCityRefId="CITY-0002" lightCityRefId="CITY-0042" catalogueSourceTriangleId="TRI-0042" />
  </triangles></scenarioReference>
  <hypothesis sourceTemplateId="TPL-0001">
    <rank number="1" triangleId="TRI-0042" />
    <rank number="2" triangleId="STRI-0001" />
  </hypothesis>
  <topoSnapshot encoding="json">{"elements":[{"element_id":"T01","source_triangle_id":"TRI-0042","vertex_business_ids":["CITY-0001","SCITY-0001",null],"meta":{"free":"TRI-0042"}}],"group_anchors":[{"anchor_id":"AN001","group_id":"G001","node_id":"T01:N0","beacon_id":"BEA-0007"}],"attachments":[{"attachment_id":"A001","feature_a":{"element_id":"T01"}}]}</topoSnapshot>
  <clockRef topoGroupId="G001" nodeId="T01:N0" edgeId="T01:E0" />
</scenario>"""


def test_mappings_preserve_numeric_suffixes_and_reject_normalization_collisions():
    mappings = build_catalogue_mappings(_legacy_catalogue())
    assert mappings.city["CITY-0001"] == "CITY-SYS-000001"
    assert mappings.city["CITY-0042"] == "CITY-SYS-000042"
    assert mappings.triangle["TRI-0042"] == "TRI-SYS-000042"
    assert mappings.counters == {"city": 42, "beacon": 7, "triangle": 42, "template": 1}

    collision = _legacy_catalogue()
    collision["cities"].append({"cityId": "CITY-1", "name": "D", "latitude": 44.0, "longitude": 5.0, "archived": False})
    with pytest.raises(ValueError, match="collision de normalisation city"):
        build_catalogue_mappings(collision)


def test_catalogue_v1_migrates_to_a_runtime_valid_v2():
    legacy = _legacy_catalogue()
    migrated = migrate_catalogue_data_v1_to_v2(legacy)
    loaded = catalogue_from_dict(migrated, id_provider=SystemCatalogueIdProvider())

    assert migrated["version"] == 2
    assert migrated["idCounters"] == {"city": 42, "beacon": 7, "triangle": 42, "template": 1}
    assert migrated["defaultTemplateId"] == "TPL-SYS-000001"
    assert loaded.get_beacon("BEA-SYS-000007").city_id == "CITY-SYS-000001"
    assert loaded.get_triangle("TRI-SYS-000042").light_city_id == "CITY-SYS-000042"


def test_scenario_migration_changes_only_known_catalogue_references(tmp_path):
    source = tmp_path / "legacy.xml"
    source.write_text(_scenario_xml(), encoding="utf-8")
    xml_bytes, report = migrate_scenario_xml(source, build_catalogue_mappings(_legacy_catalogue()))
    root = ET.fromstring(xml_bytes)
    snapshot = json.loads(root.find("topoSnapshot").text)

    assert root.find("hypothesis").get("sourceTemplateId") == "TPL-SYS-000001"
    assert [rank.get("triangleId") for rank in root.findall("./hypothesis/rank")] == ["TRI-SYS-000042", "STRI-0001"]
    assert root.find("./scenarioReference/cities/city").get("cityRefId") == "SCITY-0001"
    assert root.find("./scenarioReference/cities/city").get("catalogueSourceCityId") == "CITY-SYS-000001"
    triangle = root.find("./scenarioReference/triangles/triangle")
    assert triangle.get("triangleRefId") == "STRI-0001"
    assert triangle.get("baseCityRefId") == "CITY-SYS-000002"
    assert triangle.get("catalogueSourceTriangleId") == "TRI-SYS-000042"
    assert snapshot["elements"][0]["source_triangle_id"] == "TRI-SYS-000042"
    assert snapshot["elements"][0]["vertex_business_ids"] == ["CITY-SYS-000001", "SCITY-0001", None]
    assert snapshot["elements"][0]["meta"] == {"free": "TRI-0042"}
    assert snapshot["group_anchors"][0]["beacon_id"] == "BEA-SYS-000007"
    assert snapshot["elements"][0]["element_id"] == "T01"
    assert root.find("clockRef").get("nodeId") == "T01:N0"
    assert report.local_scities > 0 and report.local_stris > 0


def test_migration_is_dry_run_then_publishes_all_destinations(tmp_path):
    catalogue_in = tmp_path / "catalogue-v1.json"
    scenario_in = tmp_path / "scenario-v6.xml"
    catalogue_out = tmp_path / "out" / "catalogue-v2.json"
    scenario_out = tmp_path / "out" / "scenario-v6.xml"
    catalogue_in.write_text(json.dumps(_legacy_catalogue()), encoding="utf-8")
    scenario_in.write_text(_scenario_xml(), encoding="utf-8")

    report = migrate_paths(catalogue_in, catalogue_out, [(scenario_in, scenario_out)], dry_run=True)
    assert report.mappings.counters["city"] == 42
    assert not catalogue_out.exists() and not scenario_out.exists()

    migrate_paths(catalogue_in, catalogue_out, [(scenario_in, scenario_out)])
    assert load_catalogue(catalogue_out, id_provider=SystemCatalogueIdProvider()).version == 2
    assert scenario_out.exists()


def test_migration_rejects_unresolved_references_without_publishing_outputs(tmp_path):
    catalogue_in = tmp_path / "catalogue-v1.json"
    scenario_in = tmp_path / "broken.xml"
    catalogue_out = tmp_path / "catalogue-v2.json"
    scenario_out = tmp_path / "scenario-v6.xml"
    catalogue_in.write_text(json.dumps(_legacy_catalogue()), encoding="utf-8")
    scenario_in.write_text(_scenario_xml().replace("BEA-0007", "BEA-9999"), encoding="utf-8")

    with pytest.raises(ValueError, match="référence beacon legacy introuvable"):
        migrate_paths(catalogue_in, catalogue_out, [(scenario_in, scenario_out)])
    assert not catalogue_out.exists() and not scenario_out.exists()
