import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
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
        self._tri_words = {}
        self.listbox = _Listbox()
        self.canvas = _Canvas()
        self.df = None
        self.zoom = 1.0
        self.offset = np.array([0.0, 0.0])
        self._clock_ref_azimuth_deg = 0.0

    def _get_active_scenario(self):
        return self.scenarios[self.active_scenario_index]

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
        "id": 28,
        "mirrored": False,
        "topoElementId": "T28",
        "pts": {
            "O": np.array([0.0, 0.0]),
            "B": np.array([3.0, 0.0]),
            "L": np.array([0.0, 4.0]),
        },
    }


def test_xml_core_first_load_legacy_id_then_round_trip(tmp_path):
    source = tmp_path / "source.xml"
    viewer = _Viewer(_world_with_t28(), [_entry()])
    saveScenarioXml(viewer, str(source))

    tree = ET.parse(source)
    root = tree.getroot()
    triangle = root.find("./triangles/triangle")
    assert triangle is not None
    assert triangle.get("topoElementId") == "T28"

    # Existing v4 compatibility: no topoElementId and unusable legacy groups.
    del triangle.attrib["topoElementId"]
    groups = ET.SubElement(root, "groups")
    ET.SubElement(groups, "group", {"id": "99"})
    ET.SubElement(groups[-1], "node", {"tid": "not-an-index", "edge_in": "INVALID"})
    legacy = tmp_path / "legacy-v4.xml"
    tree.write(legacy, encoding="utf-8", xml_declaration=True)

    loaded = _Viewer(TopologyWorld(), [])
    loadScenarioXml(loaded, str(legacy))

    assert len(loaded._last_drawn) == 1
    assert loaded._last_drawn[0]["topoElementId"] == "T28"
    assert loaded._last_drawn[0]["topoElementId"] in loaded._get_active_scenario().topoWorld.elements
    assert "group_id" not in loaded._last_drawn[0]
    assert not hasattr(loaded, "groups")

    saved = tmp_path / "round-trip.xml"
    saveScenarioXml(loaded, str(saved))
    saved_root = ET.parse(saved).getroot()
    saved_text = saved.read_text(encoding="utf-8")
    saved_triangle = saved_root.find("./triangles/triangle")
    assert saved_triangle.get("topoElementId") == "T28"
    assert saved_root.find("groups") is None
    assert saved_triangle.get("group") is None
    assert "edge_in" not in saved_text
    assert "edge_out" not in saved_text

    reloaded = _Viewer(TopologyWorld(), [])
    loadScenarioXml(reloaded, str(saved))
    assert reloaded._last_drawn[0]["topoElementId"] == "T28"


def test_xml_core_first_loads_repository_v4_scenario_without_ui_groups(tmp_path):
    legacy_xml = (
        Path(__file__).resolve().parents[1]
        / "scenario"
        / "Scenario_FrontiereActuelle v6.xml"
    )
    viewer = _Viewer(TopologyWorld(), [])

    loadScenarioXml(viewer, str(legacy_xml))

    assert len(viewer._last_drawn) == 32
    assert not hasattr(viewer, "groups")
    assert all(
        entry["topoElementId"] in viewer._get_active_scenario().topoWorld.elements
        for entry in viewer._last_drawn
    )

    saved = tmp_path / "repository-round-trip.xml"
    saveScenarioXml(viewer, str(saved))
    saved_triangles = ET.parse(saved).getroot().findall("./triangles/triangle")
    assert len(saved_triangles) == 32
    assert all(triangle.get("topoElementId") for triangle in saved_triangles)
    assert ET.parse(saved).getroot().find("groups") is None

    reloaded = _Viewer(TopologyWorld(), [])
    loadScenarioXml(reloaded, str(saved))
    assert len(reloaded._last_drawn) == 32


def test_f11_geo_orient_dump_contains_core_and_projection_diagnostics(tmp_path):
    world = _world_with_t28()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    entry = _entry()
    entry["orient"] = "CW"
    entry["topoGroupId"] = str(world.get_group_of_element("T28"))
    viewer._last_drawn = [entry]

    dump = tmp_path / "TopoDump.xml"
    world.export_topo_dump_xml(str(dump))
    viewer._append_geo_orient_debug_to_topodump(str(dump), world)

    triangle = ET.parse(dump).getroot().find("./GeoOrientationDebug/Triangle")
    assert triangle is not None
    assert triangle.get("topoElementId") == "T28"
    assert triangle.find("Catalogue").get("orient") == "<absent>"
    assert triangle.find("Core/VertexLocalXY/Point[@vertex='O']") is not None
    assert triangle.find("LastDrawn").get("orient") == "CW"
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
