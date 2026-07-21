from types import SimpleNamespace

import numpy as np

from src.assembleur_core import TopologyElement, TopologyNodeType, TopologyWorld
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


def _element(element_id):
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=[TopologyNodeType.OUVERTURE, TopologyNodeType.BASE, TopologyNodeType.LUMIERE],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )


def _viewer_with_group():
    world = TopologyWorld()
    first, second = _element("T01"), _element("T02")
    group_id = world.union_groups(
        world.add_element_as_new_group(first), world.add_element_as_new_group(second)
    )
    world.setElementPose("T01", np.eye(2), np.array((2.0, 0.0)), mirrored=True)
    world.setElementPose("T02", np.eye(2), np.array((0.0, 3.0)), mirrored=False)
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.canvas_objects = CanvasObjectsCollection([
        {"topoElementId": "T01", "pts": {}},
        {"topoElementId": "T02", "pts": {}},
    ])
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.scenarios = [SimpleNamespace(topoWorld=world, source_type="manual")]
    viewer.active_scenario_index = 0
    viewer._is_active_auto_scenario = lambda: False
    viewer._project_core_group_to_last_drawn(world, group_id)
    return viewer, world, group_id, first, second


def _prepare_manual_rotate(viewer, world, group_id):
    viewer._sel = {
        "mode": "rotate_group",
        "core_group_id": group_id,
        "pivot_world": np.array((0.0, 0.0)),
        "mouse_angle_start": 0.0,
        "rotate_preview_initial_pts": viewer._capture_move_preview_initial_pts(world, group_id),
        "auto_geom": False,
    }
    viewer._drag = None
    viewer._clock_trace_active = False
    viewer._clock_measure_active = False
    viewer._clock_arc_active = False
    viewer._clock_setref_active = False
    viewer._ensure_pick_cache = lambda: None
    viewer._redraw_from = lambda _entries: None
    viewer._screen_to_world = lambda x, y: (float(x), -float(y))
    viewer.offset = np.array((0.0, 0.0))
    viewer.zoom = 1.0


def test_manual_rotate_preview_does_not_write_core_and_uses_snapshot():
    viewer, world, group_id, first, second = _viewer_with_group()
    _prepare_manual_rotate(viewer, world, group_id)
    before = {element_id: world.getElementPose(element_id) for element_id in ("T01", "T02")}
    calls = []
    original_rotate = world.rotate_group
    world.rotate_group = lambda *args: (calls.append(args), original_rotate(*args))[1]

    viewer._on_canvas_motion_update_drag(SimpleNamespace(x=0.0, y=-1.0))
    viewer._on_canvas_motion_update_drag(SimpleNamespace(x=0.0, y=-1.0))

    assert calls == []
    for element_id in ("T01", "T02"):
        actual = world.getElementPose(element_id)
        np.testing.assert_allclose(actual[0], before[element_id][0])
        np.testing.assert_allclose(actual[1], before[element_id][1])
        assert actual[2] is before[element_id][2]
    np.testing.assert_allclose(viewer._last_drawn[0]["pts"]["O"], np.array((0.0, 2.0)), atol=1e-12)
    np.testing.assert_allclose(viewer._last_drawn[1]["pts"]["O"], np.array((-3.0, 0.0)), atol=1e-12)


def test_manual_rotate_commit_recomputes_click_angle_once_and_projects_core():
    viewer, world, group_id, first, second = _viewer_with_group()
    _prepare_manual_rotate(viewer, world, group_id)
    calls = []
    original_rotate = world.rotate_group
    world.rotate_group = lambda *args: (calls.append(args), original_rotate(*args))[1]
    viewer._clock_arc_active = viewer._clock_trace_active = False
    viewer._clock_measure_active = viewer._clock_setref_active = False
    viewer._bg_calib_active = False
    viewer._is_in_clock = lambda *_args: False
    viewer._hide_tooltip = viewer._reset_assist = lambda: None
    viewer._redraw_from = lambda _entries: None
    viewer._bg = None
    viewer.bg_resize_mode = SimpleNamespace(get=lambda: False)
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)

    # Le dernier Motion serait +90°, mais le clic valide -90° : seul le clic compte.
    viewer._on_canvas_motion_update_drag(SimpleNamespace(x=0.0, y=-1.0))
    viewer._on_canvas_left_down(SimpleNamespace(x=0.0, y=1.0))

    assert len(calls) == 1
    committed_group, pivot, angle = calls[0]
    assert committed_group == group_id
    np.testing.assert_allclose(pivot, np.array((0.0, 0.0)))
    np.testing.assert_allclose(angle, -np.pi / 2.0)
    assert viewer._sel is None
    for element, expected_translation, mirrored in (
        (first, (0.0, -2.0), True),
        (second, (3.0, 0.0), False),
    ):
        rotation, translation, mirrored_after = world.getElementPose(element.element_id)
        np.testing.assert_allclose(translation, expected_translation, atol=1e-12)
        assert mirrored_after is mirrored
        for point in viewer.canvas_objects.get_by_topology_id(element.element_id)["pts"].values():
            assert np.all(np.isfinite(point))


def test_manual_rotate_escape_discards_preview_without_core_write():
    viewer, world, group_id, first, second = _viewer_with_group()
    _prepare_manual_rotate(viewer, world, group_id)
    viewer._on_canvas_motion_update_drag(SimpleNamespace(x=0.0, y=-1.0))
    viewer._clock_trace_active = viewer._clock_arc_active = False
    viewer._clock_measure_active = viewer._clock_setref_active = False
    viewer._bg_calib_active = False
    viewer._drag = None
    viewer._clock_trace_active = False
    viewer._clock_measure_active = False
    viewer._clock_arc_active = False
    viewer._clock_setref_active = False
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)
    viewer._clear_nearest_line = viewer._clear_edge_highlights = lambda: None
    world.rotate_group = lambda *_args: (_ for _ in ()).throw(
        AssertionError("rotate forbidden during discard")
    )

    assert viewer._on_escape_key(SimpleNamespace()) == "break"
    assert viewer._sel is None
    np.testing.assert_allclose(viewer._last_drawn[0]["pts"]["O"], first.localToWorld(first.vertex_local_xy[0]))
    np.testing.assert_allclose(viewer._last_drawn[1]["pts"]["O"], second.localToWorld(second.vertex_local_xy[0]))


def test_automatic_rotate_motion_is_preview_only():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._drag = None
    viewer._clock_trace_active = False
    viewer._clock_measure_active = False
    viewer._clock_arc_active = False
    viewer._clock_setref_active = False
    viewer._ensure_pick_cache = lambda: None
    viewer.offset = np.array((0.0, 0.0))
    viewer.zoom = 1.0
    viewer.canvas_objects = CanvasObjectsCollection([
        {"topoElementId": "T01", "pts": {
            "O": np.array((1.0, 0.0)), "B": np.array((2.0, 0.0)), "L": np.array((1.0, 1.0)),
        }},
    ])
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.auto_geom_state = {"ox": 0.0, "oy": 0.0, "thetaDeg": 10.0}
    viewer._sel = {
        "mode": "rotate_group", "auto_geom": True, "pivot": np.array((0.0, 0.0)),
        "start_angle": 0.0,
        "auto_preview_initial_pts": viewer._capture_active_auto_preview_pts(),
    }
    calls = []
    viewer._redraw_from = lambda _entries: calls.append("redraw")
    viewer._invalidate_pick_cache = lambda: None

    viewer._on_canvas_motion_update_drag(SimpleNamespace(x=0.0, y=-1.0))

    np.testing.assert_allclose(viewer.auto_geom_state["thetaDeg"], 10.0)
    np.testing.assert_allclose(viewer._last_drawn[0]["pts"]["O"], (0.0, 1.0), atol=1e-12)
    assert calls == ["redraw"]
