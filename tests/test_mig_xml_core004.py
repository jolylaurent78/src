"""Contrat XML v5 : le snapshot Core est l'unique persistence geometrique."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.assembleur_core import ScenarioAssemblage, TopologyElement, TopologyWorld
from src.assembleur_io import loadScenarioXml, saveScenarioXml
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


class _Listbox:
    def size(self): return 0
    def get(self, _index): raise IndexError
    def delete(self, *_args): pass
    def insert(self, *_args): pass


class _Canvas:
    def delete(self, *_args): pass
    def focus_set(self): pass


class _Viewer:
    _bind_canvas_objects = TriangleViewerManual._bind_canvas_objects
    _build_scenario_projection_from_core = TriangleViewerManual._build_scenario_projection_from_core
    _rebuild_active_projection_from_core = TriangleViewerManual._rebuild_active_projection_from_core
    _strip_core_duplicates_from_last_drawn_entry = staticmethod(
        TriangleViewerManual._strip_core_duplicates_from_last_drawn_entry
    )

    def __init__(self, world, entries, source_type="auto"):
        scenario = ScenarioAssemblage("XML Core 004", source_type=source_type)
        scenario.topoWorld = world
        scenario.topoScenarioId = "SCENARIO"
        scenario.orderedElementIds = list(world.elements)
        self.scenarios, self.active_scenario_index = [scenario], 0
        self.canvas_objects = CanvasObjectsCollection(entries)
        self._last_drawn = self.canvas_objects.entries
        scenario.last_drawn = self._last_drawn
        self.excel_path, self._bg = None, None
        self._clock_cx, self._clock_cy = 0.0, 0.0
        self._clock_state = {"hour": 0, "minute": 0, "label": ""}
        self.listbox, self.canvas, self.df = _Listbox(), _Canvas(), None
        self.zoom, self.offset = 1.0, np.zeros(2)
        self._clock_ref_azimuth_deg = 0.0

    def _get_active_scenario(self): return self.scenarios[self.active_scenario_index]
    def _bg_clear(self, persist=False): pass
    def _clear_nearest_line(self): pass
    def _clear_edge_highlights(self): pass
    def _hide_tooltip(self): pass
    def _update_triangle_listbox_colors(self): pass
    def _bind_canvas_handlers(self): pass
    def _redraw_from(self, _entries): pass
    def _redraw_overlay_only(self): pass
    def _rebuild_pick_cache(self): pass


def _world():
    world = TopologyWorld()
    world.add_element_as_new_group(TopologyElement(
        element_id="T01", name="Triangle 1", vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"], edge_lengths_km=[3.0, 5.0, 4.0],
        meta={"triRank": 1},
    ))
    return world


def test_v5_writer_ignores_projection_and_loader_rebuilds_manual_cache(tmp_path):
    world = _world()
    poisoned_cache = [{"topoElementId": "T01", "pts": {"O": np.array([999.0, 999.0])}}]
    source = tmp_path / "core-only.xml"
    source_viewer = _Viewer(world, poisoned_cache, source_type="auto")
    source_viewer._get_active_scenario().last_drawn = poisoned_cache
    source_viewer._last_drawn = poisoned_cache
    saveScenarioXml(source_viewer, str(source))

    root = ET.parse(source).getroot()
    xml_text = source.read_text(encoding="utf-8")
    assert root.get("version") == "5"
    assert root.find("topoSnapshot") is not None
    assert root.find("triangles") is None
    assert "999" not in xml_text
    assert "FAKE" not in xml_text

    viewer = _Viewer(TopologyWorld(), [], source_type="auto")
    loadScenarioXml(viewer, str(source))
    scen = viewer._get_active_scenario()
    assert scen.source_type == "manual"
    assert len(viewer._last_drawn) == 1
    assert viewer._last_drawn[0]["topoElementId"] == "T01"
    assert not np.allclose(viewer._last_drawn[0]["pts"]["O"], [999.0, 999.0])


def test_v5_loader_rejects_missing_snapshot_and_legacy_triangles(tmp_path):
    path = tmp_path / "invalid.xml"
    path.write_text('<scenario version="5" topo_tx_orientation="cw"><triangles /></scenario>', encoding="utf-8")
    with pytest.raises(ValueError, match="legacy <triangles>"):
        loadScenarioXml(_Viewer(TopologyWorld(), []), str(path))


def test_v5_loader_rejects_version_4_without_legacy_fallback(tmp_path):
    path = tmp_path / "v4.xml"
    path.write_text('<scenario version="4" topo_tx_orientation="cw" />', encoding="utf-8")
    with pytest.raises(ValueError, match="expected 5, got 4"):
        loadScenarioXml(_Viewer(TopologyWorld(), []), str(path))


def test_v5_loader_rejects_missing_or_invalid_snapshot_before_replacing_world(tmp_path):
    original_world = _world()
    viewer = _Viewer(original_world, [])
    missing = tmp_path / "missing-snapshot.xml"
    missing.write_text('<scenario version="5" topo_tx_orientation="cw" />', encoding="utf-8")
    with pytest.raises(ValueError, match="Missing topoSnapshot"):
        loadScenarioXml(viewer, str(missing))
    assert viewer._get_active_scenario().topoWorld is original_world


def test_v5_round_trip_preserves_core_poses_and_handles_empty_world(tmp_path):
    world = _world()
    second = TopologyElement(
        element_id="T02", name="Triangle 2", vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"], edge_lengths_km=[6.0, 10.0, 8.0],
        meta={"triRank": 2},
    )
    world.add_element_as_new_group(second)
    world.setElementPose("T01", np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([2.0, 3.0]), mirrored=True)
    world.setElementPose("T02", np.eye(2), np.array([-4.0, 7.0]), mirrored=False)
    before = {element_id: world.getElementPose(element_id) for element_id in world.elements}
    path = tmp_path / "poses.xml"
    saveScenarioXml(_Viewer(world, []), str(path))
    loaded = _Viewer(TopologyWorld(), [])
    loadScenarioXml(loaded, str(path))
    restored = loaded._get_active_scenario().topoWorld
    assert len(restored.elements) == 2
    assert len(restored.getLiveGroupIds()) == 2
    for element_id, (rotation, translation, mirrored) in before.items():
        actual_rotation, actual_translation, actual_mirrored = restored.getElementPose(element_id)
        assert np.allclose(actual_rotation, rotation)
        assert np.allclose(actual_translation, translation)
        assert actual_mirrored is mirrored
    assert len(loaded._last_drawn) == 2
    loaded.canvas_objects.validate_against_world(restored)

    empty_path = tmp_path / "empty.xml"
    saveScenarioXml(_Viewer(TopologyWorld(), []), str(empty_path))
    empty_loaded = _Viewer(_world(), [{"topoElementId": "T01", "pts": {}}])
    loadScenarioXml(empty_loaded, str(empty_path))
    assert empty_loaded._get_active_scenario().last_drawn == []
    assert empty_loaded._last_drawn == []

    invalid = tmp_path / "invalid-snapshot.xml"
    invalid.write_text(
        '<scenario version="5" topo_tx_orientation="cw">'
        '<topoSnapshot encoding="json">[]</topoSnapshot></scenario>',
        encoding="utf-8",
    )
    original_world = _world()
    viewer = _Viewer(original_world, [])
    with pytest.raises(ValueError, match="Missing topoSnapshot"):
        loadScenarioXml(viewer, str(invalid))
    assert viewer._get_active_scenario().topoWorld is original_world
