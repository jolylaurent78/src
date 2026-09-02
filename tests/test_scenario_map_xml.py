import xml.etree.ElementTree as ET

import pytest

from src.assembleur_catalogue import Catalogue, WorldRect
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_io import _parse_map_state
from src.assembleur_scenario_map import (
    ScenarioMapPosition,
    ScenarioMapState,
    scenario_map_state_from_xml_attributes,
    scenario_map_state_to_xml_attributes,
)


def _catalogue() -> Catalogue:
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    catalogue.add_map(
        name="Alsace",
        image_file="899 - Alsace.jpg",
        calibration_file="899 - Alsace.json",
        projection="EPSG:2154",
        default_world_rect=WorldRect(-10, 20, 400, 200),
        default_scale_factor=12,
    )
    return catalogue


@pytest.mark.parametrize(
    "state",
    [
        ScenarioMapState("MAP-SYS-000001"),
        ScenarioMapState("MAP-SYS-000001", ScenarioMapPosition(1, 2)),
        ScenarioMapState("MAP-SYS-000001", scale_factor_override=15),
        ScenarioMapState("MAP-SYS-000001", ScenarioMapPosition(1, 2), 15, False, 0.35),
    ],
)
def test_xml_target_round_trip_preserves_all_supported_overrides(state) -> None:
    attributes = scenario_map_state_to_xml_attributes(state)

    assert "resource" not in attributes and "path" not in attributes
    assert "w" not in attributes and "h" not in attributes
    assert scenario_map_state_from_xml_attributes(attributes) == state


@pytest.mark.parametrize(
    "attributes",
    [
        {"refId": "CITY-SYS-000001"},
        {"refId": "MAP-SYS-000001", "x0": "1"},
        {"refId": "MAP-SYS-000001", "y0": "2"},
        {"refId": "MAP-SYS-000001", "scale": "0"},
        {"refId": "MAP-SYS-000001", "scale": "nan"},
        {"refId": "MAP-SYS-000001", "visible": "1"},
        {"refId": "MAP-SYS-000001", "opacity": "1.1"},
    ],
)
def test_xml_target_rejects_invalid_values(attributes) -> None:
    with pytest.raises(ValueError):
        scenario_map_state_from_xml_attributes(attributes)


def test_xml_map_without_element_means_no_map() -> None:
    assert ScenarioMapState(map_ref_id=None).map_ref_id is None


def test_legacy_default_pose_normalizes_without_false_overrides() -> None:
    catalogue = _catalogue()
    state = _parse_map_state(catalogue, ET.Element("map", {
        "resource": "899 - Alsace.jpg", "x0": "-10", "y0": "20", "w": "400", "h": "200",
        "scale": "11.983520258273956", "visible": "1", "opacity": "70",
    }))

    assert state == ScenarioMapState("MAP-SYS-000001", opacity=0.7)


def test_legacy_six_significant_digit_rounding_normalizes_to_defaults() -> None:
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    default = WorldRect(-2963.38, 1642.9905325627296, 4293.6497163554395, 5282.159467437271)
    catalogue.add_map(
        name="Alsace", image_file="899 - Alsace.jpg", calibration_file="899 - Alsace.json",
        projection="EPSG:2154", default_world_rect=default, default_scale_factor=12,
    )
    state = _parse_map_state(catalogue, ET.Element("map", {
        "resource": "899 - Alsace.jpg",
        "x0": f"{default.x0:.6g}", "y0": f"{default.y0:.6g}",
        "w": f"{default.w:.6g}", "h": f"{default.h:.6g}",
    }))

    assert state == ScenarioMapState("MAP-SYS-000001")


def test_legacy_migration_keeps_position_and_derives_scale() -> None:
    catalogue = _catalogue()
    state = _parse_map_state(catalogue, ET.Element("map", {
        "path": r"D:\Old\899 - Alsace.jpg", "x0": "3", "y0": "4", "w": "500", "h": "250",
    }))

    assert state.position_override == ScenarioMapPosition(3, 4)
    assert state.scale_factor_override == 15
