import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import src.assembleur_tk as assembleur_tk

from src.assembleur_core import ScenarioAssemblage, TopologyElement, TopologyWorld
from src.assembleur_io import loadScenarioXml, saveScenarioXml
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


class _Listbox:
    def size(self):
        return 0

    def get(self, _index):
        raise IndexError

    def delete(self, *_args):
        pass

    def insert(self, *_args):
        pass


class _Canvas:
    def delete(self, *_args):
        pass

    def focus_set(self):
        pass


class _Viewer:
    # Intentionally has no groups or _next_group_id attributes.
    def __init__(self, world, entries):
        scenario = ScenarioAssemblage("XML test")
        scenario.topoWorld = world
        scenario.topoScenarioId = "SCENARIO"
        self.scenarios = [scenario]
        self.active_scenario_index = 0
        self.canvas_objects = CanvasObjectsCollection(entries)
        self._last_drawn = self.canvas_objects.entries
        self.excel_path = None
        self._bg = None
        self._clock_cx = 0.0
        self._clock_cy = 0.0
        self._clock_state = {"hour": 0, "minute": 0, "label": ""}
        self.listbox = _Listbox()
        self.canvas = _Canvas()
        self.df = None
        self.zoom = 1.0
        self.offset = np.array([0.0, 0.0])
        self._clock_ref_azimuth_deg = 0.0

    def _get_active_scenario(self):
        return self.scenarios[self.active_scenario_index]

    _bind_canvas_objects = TriangleViewerManual._bind_canvas_objects
    _strip_core_duplicates_from_last_drawn_entry = staticmethod(
        TriangleViewerManual._strip_core_duplicates_from_last_drawn_entry
    )
    _build_scenario_projection_from_core = TriangleViewerManual._build_scenario_projection_from_core
    _rebuild_active_projection_from_core = TriangleViewerManual._rebuild_active_projection_from_core

    def _pt_to_xml(self, point):
        return f"{float(point[0])},{float(point[1])}"

    def _xml_to_pt(self, text):
        return np.array([float(value) for value in text.split(",")])

    def _bg_clear(self, persist=False):
        pass

    def _clear_nearest_line(self):
        pass

    def _clear_edge_highlights(self):
        pass

    def _hide_tooltip(self):
        pass

    def _update_triangle_listbox_colors(self):
        pass

    def _bind_canvas_handlers(self):
        pass

    def _redraw_from(self, _entries):
        pass

    def _redraw_overlay_only(self):
        pass

    def _rebuild_pick_cache(self):
        pass


def _world_with_t28():
    world = TopologyWorld()
    element = TopologyElement(
        element_id="T28",
        name="Triangle 28",
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )
    world.add_element_as_new_group(element)
    return world


def _entry():
    return {
        "topoElementId": "T28",
        "pts": {
            "O": np.array([0.0, 0.0]),
            "B": np.array([3.0, 0.0]),
            "L": np.array([0.0, 4.0]),
        },
    }


def test_xml_v5_persists_only_the_core_and_rebuilds_projection(tmp_path):
    source = tmp_path / "source.xml"
    viewer = _Viewer(_world_with_t28(), [_entry()])
    saveScenarioXml(viewer, str(source))

    tree = ET.parse(source)
    root = tree.getroot()
    assert root.get("version") == "5"
    assert root.find("triangles") is None
    assert root.find("topoSnapshot") is not None
    assert root.find("view") is not None

    loaded = _Viewer(TopologyWorld(), [])
    loadScenarioXml(loaded, str(source))

    assert len(loaded._last_drawn) == 1
    assert loaded._last_drawn[0]["topoElementId"] == "T28"
    assert "id" not in loaded._last_drawn[0]
    assert loaded._last_drawn[0]["topoElementId"] in loaded._get_active_scenario().topoWorld.elements
    assert "group_id" not in loaded._last_drawn[0]
    assert not {"labels", "orient", "topoGroupId", "mirrored"}.intersection(loaded._last_drawn[0])
    assert tuple(loaded._get_active_scenario().topoWorld.elements["T28"].vertex_labels) == ("O", "B", "L")
    assert not hasattr(loaded, "groups")

    saved = tmp_path / "round-trip.xml"
    saveScenarioXml(loaded, str(saved))
    saved_root = ET.parse(saved).getroot()
    saved_text = saved.read_text(encoding="utf-8")
    assert saved_root.get("version") == "5"
    assert saved_root.find("triangles") is None
    assert saved_root.find("groups") is None
    assert saved_root.find("words") is None
    assert "edge_in" not in saved_text
    assert "edge_out" not in saved_text

    reloaded = _Viewer(TopologyWorld(), [])
    loadScenarioXml(reloaded, str(saved))
    assert reloaded._last_drawn[0]["topoElementId"] == "T28"


def test_xml_v5_rejects_legacy_triangles_section(tmp_path):
    source = tmp_path / "source.xml"
    saveScenarioXml(_Viewer(_world_with_t28(), [_entry()]), str(source))
    tree = ET.parse(source)
    ET.SubElement(tree.getroot(), "triangles")
    legacy_only = tmp_path / "invalid-v5.xml"
    tree.write(legacy_only, encoding="utf-8", xml_declaration=True)

    with pytest.raises(ValueError, match="legacy <triangles>"):
        loadScenarioXml(_Viewer(TopologyWorld(), []), str(legacy_only))


def test_xml_mirrored_round_trip_uses_core_without_cache_duplication(tmp_path):
    world = _world_with_t28()
    world.setElementPose("T28", np.eye(2), np.zeros(2), mirrored=True)
    viewer = _Viewer(world, [_entry()])

    path = tmp_path / "mirrored.xml"
    saveScenarioXml(viewer, str(path))

    assert ET.parse(path).getroot().find("triangles") is None
    assert "mirrored" not in viewer._last_drawn[0]

    reloaded = _Viewer(TopologyWorld(), [])
    loadScenarioXml(reloaded, str(path))
    reloaded_world = reloaded._get_active_scenario().topoWorld
    assert reloaded_world.getElementPose("T28")[2] is True
    assert "mirrored" not in reloaded._last_drawn[0]


def test_xml_core_first_rejects_repository_v4_scenario_without_topology_ids():
    legacy_xml = (
        Path(__file__).resolve().parents[1]
        / "scenario"
        / "Scenario_FrontiereActuelle v6.xml"
    )
    viewer = _Viewer(TopologyWorld(), [])

    with pytest.raises(ValueError, match="expected 5"):
        loadScenarioXml(viewer, str(legacy_xml))


def test_f11_geo_orient_dump_contains_core_and_projection_diagnostics(tmp_path):
    world = _world_with_t28()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    entry = _entry()
    viewer._last_drawn = [entry]

    dump = tmp_path / "TopoDump.xml"
    world.export_topo_dump_xml(str(dump))
    viewer._append_geo_orient_debug_to_topodump(str(dump), world)

    triangle = ET.parse(dump).getroot().find("./GeoOrientationDebug/Triangle")
    assert triangle is not None
    assert triangle.get("topoElementId") == "T28"
    assert triangle.find("Catalogue").get("orient") == "<absent>"
    assert triangle.find("Core/VertexLocalXY/Point[@vertex='O']") is not None
    assert triangle.find("LastDrawn").get("coreGroupId") == str(world.get_group_of_element("T28"))
    assert triangle.find("GeometricOrientation").get("world") in {"CW", "CCW"}
    assert triangle.find("XML").get("mirrored") == "0"


def test_f12_toggles_geo_orient_debug_and_logs_state(monkeypatch):
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.debug_geo_orient = False
    messages = []
    monkeypatch.setattr(
        assembleur_tk.MIG_GEO_LOGGER,
        "info",
        lambda message, state: messages.append(message % state),
    )

    assert viewer._toggle_geo_orient_debug() == "break"
    assert viewer.debug_geo_orient is True
    assert viewer._geo_orient_debug_enabled() is True
    assert messages == ["DEBUG GEO-ORIENT : ON"]

    assert viewer._toggle_geo_orient_debug() == "break"
    assert viewer.debug_geo_orient is False
    assert messages[-1] == "DEBUG GEO-ORIENT : OFF"
