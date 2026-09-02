from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from src.assembleur_catalogue import Catalogue, WorldRect
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_io import _parse_map_state
from src.assembleur_paths import ApplicationPaths
from src.assembleur_scenario_map import migrate_scenario_map_path


def _maps_dir(tmp_path: Path) -> Path:
    maps_dir = tmp_path / "resources" / "maps"
    maps_dir.mkdir(parents=True)
    (maps_dir / "899 - Alsace.jpg").write_bytes(b"map")
    return maps_dir


def _write_scenario(path: Path, map_attributes: dict[str, str] | None) -> None:
    root = ET.Element("scenario", {"version": "6"})
    if map_attributes is not None:
        ET.SubElement(root, "map", map_attributes)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_migration_converts_delivered_map_path_and_preserves_other_attributes(tmp_path) -> None:
    maps_dir = _maps_dir(tmp_path)
    scenario = tmp_path / "scenario.xml"
    original_path = r"D:\Old\data\maps\899 - Alsace.jpg"
    _write_scenario(scenario, {"path": original_path, "x0": "1", "opacity": "55"})

    result = migrate_scenario_map_path(scenario, resource_maps_dir=maps_dir)
    map_element = ET.parse(scenario).getroot().find("map")

    assert result.changed is True
    assert result.old_path == original_path
    assert result.resource == "899 - Alsace.jpg"
    assert result.backup_path is not None and result.backup_path.is_file()
    assert map_element is not None
    assert map_element.get("resource") == "899 - Alsace.jpg"
    assert map_element.get("path") is None
    assert map_element.get("x0") == "1"
    assert map_element.get("opacity") == "55"


def test_migration_preserves_external_map_and_xml_without_map(tmp_path) -> None:
    maps_dir = _maps_dir(tmp_path)
    external = tmp_path / "external.xml"
    no_map = tmp_path / "no-map.xml"
    _write_scenario(external, {"path": r"Z:\Cartes\personnelle.jpg"})
    _write_scenario(no_map, None)

    external_result = migrate_scenario_map_path(external, resource_maps_dir=maps_dir)
    no_map_result = migrate_scenario_map_path(no_map, resource_maps_dir=maps_dir)

    assert external_result.changed is False
    assert ET.parse(external).getroot().find("map").get("path") == r"Z:\Cartes\personnelle.jpg"
    assert no_map_result.changed is False


def test_resource_map_state_is_migrated_to_an_explicit_catalogue_reference(tmp_path) -> None:
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    map_id = catalogue.add_map(
        name="Alsace", image_file="899 - Alsace.jpg", calibration_file="map.json",
        projection="EPSG:2154", default_world_rect=WorldRect(1, 2, 4, 2), default_scale_factor=12,
    )

    state = _parse_map_state(catalogue, ET.Element("map", {"resource": "899 - Alsace.jpg"}))
    assert state.map_ref_id == map_id
    with pytest.raises(ValueError, match="mutuellement exclusifs"):
        _parse_map_state(catalogue, ET.Element("map", {"resource": "899 - Alsace.jpg", "path": "x"}))
    with pytest.raises(ValueError, match="non migrable"):
        _parse_map_state(catalogue, ET.Element("map", {"resource": "../899 - Alsace.jpg"}))
