from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.assembleur_balises import Beacon, BeaconCatalog
from src.assembleur_core import TopologyElement, TopologyWorld
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


def _catalog():
    catalog = BeaconCatalog()
    catalog._by_id = {
        "BAL-1": Beacon("BAL-1", "Balise 1", 0.0, 0.0, 0.0, 0.0, 10.0, 5.0),
    }
    return catalog


def _element(element_id="T01"):
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[3.0, 4.0, 5.0],
    )


def _viewer_with_one_group():
    catalog = _catalog()
    world = TopologyWorld(beacon_catalog=catalog)
    group_id = world.add_element_as_new_group(_element())
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.canvas_objects = CanvasObjectsCollection([{"topoElementId": "T01", "pts": {}}])
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.scenarios = [SimpleNamespace(topoWorld=world, source_type="manual")]
    viewer.active_scenario_index = 0
    viewer.beacon_catalog = catalog
    viewer.canvas = _CanvasRecorder()
    viewer.offset = np.array((0.0, 0.0))
    viewer.zoom = 1.0
    viewer._hit_px = 12
    viewer._edge_choice = None
    viewer._edge_highlights = None
    viewer._group_drag_snap_candidate = None
    viewer._nearest_line_id = None
    viewer._edge_highlight_ids = []
    viewer._update_assist_line_to_world = lambda *_args, **_kwargs: None
    viewer._project_core_group_to_last_drawn(world, group_id)
    return viewer, world, group_id


def _add_target(viewer, world, translation):
    target_group_id = world.add_element_as_new_group(_element("T02"))
    world.setElementPose("T02", np.eye(2), np.asarray(translation, dtype=float))
    viewer.canvas_objects.add({"topoElementId": "T02", "pts": {}})
    viewer._project_core_group_to_last_drawn(world, target_group_id)
    return target_group_id


def _set_beacon_world(viewer, beacon_id, world_xy):
    current = viewer.beacon_catalog._by_id.get(beacon_id)
    if current is None:
        viewer.beacon_catalog._by_id[beacon_id] = Beacon(
            beacon_id, beacon_id, 0.0, 0.0, 0.0, 0.0, *world_xy
        )
    else:
        viewer.beacon_catalog._by_id[beacon_id] = replace(
            current, world_x=float(world_xy[0]), world_y=float(world_xy[1])
        )


def test_beacon_wins_when_closer_than_a_raccordable_vertex():
    viewer, world, group_id = _viewer_with_one_group()
    _add_target(viewer, world, (3.0, 0.0))
    mobile_world = np.asarray(viewer._last_drawn[0]["pts"]["O"], dtype=float)
    _set_beacon_world(viewer, "BAL-1", mobile_world + np.array((1.0, 0.0)))
    viewer._update_nearest_line = lambda *_args, **_kwargs: None
    viewer._update_edge_highlights = lambda *_args, **_kwargs: setattr(
        viewer, "_edge_choice", object()
    )

    candidate = viewer._update_group_drag_snap_assist(
        mobile_world, 0, "O", group_id
    )

    assert candidate["type"] == "beacon"
    assert candidate["beacon_id"] == "BAL-1"
    assert viewer._edge_choice is None


def test_raccordable_vertex_wins_when_closer_than_beacon():
    viewer, world, group_id = _viewer_with_one_group()
    _add_target(viewer, world, (1.0, 0.0))
    mobile_world = np.asarray(viewer._last_drawn[0]["pts"]["O"], dtype=float)
    _set_beacon_world(viewer, "BAL-1", mobile_world + np.array((3.0, 0.0)))
    viewer._update_nearest_line = lambda *_args, **_kwargs: None
    viewer._update_edge_highlights = lambda *_args, **_kwargs: setattr(
        viewer, "_edge_choice", object()
    )

    candidate = viewer._update_group_drag_snap_assist(
        mobile_world, 0, "O", group_id
    )

    assert candidate["type"] == "vertex"
    assert candidate["triangle_idx"] == 1
    assert candidate["distance2"] < 9.0


def test_raccordable_vertex_wins_on_equal_distance_with_beacon():
    viewer, world, group_id = _viewer_with_one_group()
    _add_target(viewer, world, (1.0, 0.0))
    mobile_world = np.asarray(viewer._last_drawn[0]["pts"]["O"], dtype=float)
    _set_beacon_world(viewer, "BAL-1", mobile_world + np.array((1.0, 0.0)))
    viewer._update_nearest_line = lambda *_args, **_kwargs: None
    viewer._update_edge_highlights = lambda *_args, **_kwargs: setattr(
        viewer, "_edge_choice", object()
    )

    candidate = viewer._update_group_drag_snap_assist(
        mobile_world, 0, "O", group_id
    )

    assert candidate["type"] == "vertex"


def test_non_raccordable_vertex_is_eliminated_in_favor_of_beacon():
    viewer, world, group_id = _viewer_with_one_group()
    _add_target(viewer, world, (1.0, 0.0))
    mobile_world = np.asarray(viewer._last_drawn[0]["pts"]["O"], dtype=float)
    _set_beacon_world(viewer, "BAL-1", mobile_world + np.array((4.0, 0.0)))
    viewer._update_nearest_line = lambda *_args, **_kwargs: None
    viewer._update_edge_highlights = lambda *_args, **_kwargs: None

    candidate = viewer._update_group_drag_snap_assist(
        mobile_world, 0, "O", group_id
    )

    assert candidate["type"] == "beacon"


def test_nearest_beacon_is_selected_without_capture_radius():
    viewer, _world, _group_id = _viewer_with_one_group()
    _set_beacon_world(viewer, "BAL-1", (100.0, 0.0))
    _set_beacon_world(viewer, "BAL-2", (20.0, 0.0))

    candidate = viewer._find_nearest_beacon_candidate(np.array((-10.0, 0.0)))
    assert candidate["beacon_id"] == "BAL-2"
    assert candidate["distance2"] == 900.0


def test_single_distant_beacon_remains_a_candidate():
    viewer, _world, _group_id = _viewer_with_one_group()
    _set_beacon_world(viewer, "BAL-1", (100.0, 0.0))

    candidate = viewer._find_nearest_beacon_candidate(np.array((0.0, 0.0)))

    assert candidate["beacon_id"] == "BAL-1"
    assert candidate["distance2"] == 10000.0


def test_beacon_without_world_coordinates_is_ignored():
    viewer, _world, _group_id = _viewer_with_one_group()
    viewer.beacon_catalog._by_id["BAL-1"] = replace(
        viewer.beacon_catalog._by_id["BAL-1"], world_x=None, world_y=None
    )

    assert viewer._find_nearest_beacon_candidate(np.array((0.0, 0.0))) is None


def test_iter_beacons_programming_failure_is_not_silenced():
    viewer, _world, _group_id = _viewer_with_one_group()

    class ExplodingCatalog:
        @staticmethod
        def iter_beacons():
            raise RuntimeError("catalogue invalide")

    viewer.beacon_catalog = ExplodingCatalog()
    with pytest.raises(RuntimeError, match="catalogue invalide"):
        viewer._find_nearest_beacon_candidate(np.array((0.0, 0.0)))


class _CanvasRecorder:
    def __init__(self):
        self.deleted = []
        self.ovals = []
        self.raised = []

    def delete(self, tag):
        self.deleted.append(tag)

    def create_oval(self, *coords, **kwargs):
        self.ovals.append((coords, kwargs))
        return len(self.ovals)

    def tag_raise(self, tag):
        self.raised.append(tag)


def test_beacon_candidate_draws_the_dedicated_compass_style_ring():
    viewer, _world, _group_id = _viewer_with_one_group()
    canvas = _CanvasRecorder()
    viewer.canvas = canvas
    viewer.offset = np.array((0.0, 0.0))
    viewer.zoom = 2.0

    viewer._group_drag_update_beacon_target(
        {"type": "beacon", "beacon_id": "BAL-1", "world": np.array((3.0, 4.0))}
    )

    assert canvas.deleted == ["group_drag_beacon_target"]
    assert canvas.ovals == [
        ((-4.0, -18.0, 16.0, 2.0), {
            "outline": "#FF0000",
            "width": 3,
            "fill": "",
            "tags": "group_drag_beacon_target",
        })
    ]
    assert canvas.raised == ["group_drag_beacon_target"]


def test_beacon_ring_is_cleared_when_vertex_candidate_becomes_winner():
    viewer, world, group_id = _viewer_with_one_group()
    _add_target(viewer, world, (1.0, 0.0))
    canvas = _CanvasRecorder()
    viewer.canvas = canvas
    viewer.offset = np.array((0.0, 0.0))
    mobile_world = np.asarray(viewer._last_drawn[0]["pts"]["O"], dtype=float)
    _set_beacon_world(viewer, "BAL-1", mobile_world + np.array((0.5, 0.0)))
    viewer._update_nearest_line = lambda *_args, **_kwargs: None
    viewer._update_edge_highlights = lambda *_args, **_kwargs: setattr(
        viewer, "_edge_choice", object()
    )
    viewer._update_group_drag_snap_assist(mobile_world, 0, "O", group_id)
    assert viewer._group_drag_snap_candidate["type"] == "beacon"

    _set_beacon_world(viewer, "BAL-1", mobile_world + np.array((4.0, 0.0)))
    viewer._update_group_drag_snap_assist(mobile_world, 0, "O", group_id)

    assert viewer._group_drag_snap_candidate["type"] == "vertex"
    assert canvas.deleted[-1] == "group_drag_beacon_target"


def test_beacon_candidate_creates_and_applies_anchor_without_free_move():
    viewer, world, group_id = _viewer_with_one_group()
    viewer._sel = {
        "mode": "move_group",
        "core_group_id": group_id,
        "mouse_world_start": np.array((0.0, 0.0)),
        "anchor": {"type": "vertex", "tid": 0, "vkey": "O"},
        "suppress_assist": False,
    }
    viewer._group_drag_snap_candidate = {
        "type": "beacon",
        "beacon_id": "BAL-1",
        "world": np.array((10.0, 5.0)),
    }
    viewer._drag = None
    viewer._ctrl_down = False
    viewer._clock_dragging = False
    viewer._bg_resizing = None
    viewer._bg_moving = None
    viewer._pan_anchor = None
    viewer.auto_geom_state = None
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)
    viewer._is_active_auto_scenario = lambda: False
    viewer._clear_edge_highlights = lambda: None
    viewer._reset_assist = lambda: None
    viewer._redraw_from = lambda _entries: None
    world.move_group = lambda *_args: (_ for _ in ()).throw(
        AssertionError("a beacon anchor must not commit a free move")
    )

    viewer._on_canvas_left_up(SimpleNamespace(x=0.0, y=0.0))

    anchor = world.getAnchorForGroup(group_id)
    assert anchor is not None
    assert anchor.beacon_id == "BAL-1"
    assert world.getConceptNodeWorldXY(anchor.node_id, group_id) == (10.0, 5.0)
    np.testing.assert_allclose(viewer._last_drawn[0]["pts"]["O"], (10.0, 5.0))


def test_drag_of_an_anchored_group_is_refused_before_preview_starts():
    viewer, world, group_id = _viewer_with_one_group()
    node_id = world.get_element_vertex_node_id_by_type("T01", "O")
    world.createGroupAnchor(group_id, "BAL-1", node_id)
    viewer._clock_arc_active = False
    viewer._clock_trace_active = False
    viewer._clock_measure_active = False
    viewer._clock_setref_active = False
    viewer._ensure_pick_cache = lambda: None
    viewer._screen_to_world = lambda x, y: (float(x), float(y))
    viewer.offset = np.array((0.0, 0.0))
    viewer._bg_calib_active = False
    viewer._is_in_clock = lambda *_args: False
    viewer.bg_resize_mode = SimpleNamespace(get=lambda: False)
    viewer._bg = None
    viewer._sel = None
    viewer._drag = None
    viewer._hide_tooltip = lambda: None
    viewer._reset_assist = lambda: None
    viewer._hit_test = lambda *_args: ("vertex", 0, "O")
    viewer._resolve_core_vertex_move_members = lambda *_args: {
        "core_group_id": group_id,
        "entries": [viewer._last_drawn[0]],
    }
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)

    assert viewer._on_canvas_left_down(SimpleNamespace(x=0.0, y=0.0)) == "break"
    assert viewer._sel is None


def _prepare_escape_from_assisted_manual_move(viewer, world, group_id):
    initial_pts = viewer._capture_move_preview_initial_pts(world, group_id)
    viewer._sel = {
        "mode": "move_group",
        "core_group_id": group_id,
        "mouse_world_start": np.array((0.0, 0.0)),
        "move_preview_initial_pts": initial_pts,
        "anchor": {"type": "vertex", "tid": 0, "vkey": "O"},
        "suppress_assist": False,
    }
    viewer._drag = None
    viewer._clock_trace_active = False
    viewer._clock_arc_active = False
    viewer._clock_measure_active = False
    viewer._clock_setref_active = False
    viewer._bg_calib_active = False
    viewer._is_active_auto_scenario = lambda: False
    viewer._invalidate_pick_cache = lambda: None
    viewer._redraw_from = lambda _entries: None
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)
    viewer._preview_move_group_from_snapshot(initial_pts, 4.0, -2.0)
    return initial_pts


def test_escape_clears_beacon_candidate_and_restores_manual_preview():
    viewer, world, group_id = _viewer_with_one_group()
    mobile_world = np.asarray(viewer._last_drawn[0]["pts"]["O"], dtype=float)
    viewer._update_group_drag_snap_assist(mobile_world, 0, "O", group_id)
    assert viewer._group_drag_snap_candidate["type"] == "beacon"
    initial_pts = _prepare_escape_from_assisted_manual_move(viewer, world, group_id)

    assert viewer._on_escape_key(SimpleNamespace()) == "break"

    assert viewer._sel is None
    assert viewer._group_drag_snap_candidate is None
    assert viewer._edge_choice is None
    assert viewer._edge_highlights is None
    assert world.groupAnchors == {}
    np.testing.assert_allclose(viewer._last_drawn[0]["pts"]["O"], initial_pts["T01"]["O"])


def test_escape_clears_vertex_candidate_without_creating_attachment():
    viewer, world, group_id = _viewer_with_one_group()
    _add_target(viewer, world, (1.0, 0.0))
    mobile_world = np.asarray(viewer._last_drawn[0]["pts"]["O"], dtype=float)
    viewer._update_nearest_line = lambda *_args, **_kwargs: None
    viewer._update_edge_highlights = lambda *_args, **_kwargs: setattr(
        viewer, "_edge_choice", object()
    )
    viewer._update_group_drag_snap_assist(mobile_world, 0, "O", group_id)
    assert viewer._group_drag_snap_candidate["type"] == "vertex"
    _prepare_escape_from_assisted_manual_move(viewer, world, group_id)

    assert viewer._on_escape_key(SimpleNamespace()) == "break"

    assert viewer._sel is None
    assert viewer._group_drag_snap_candidate is None
    assert viewer._edge_choice is None
    assert viewer._edge_highlights is None
    assert world.attachments == {}
