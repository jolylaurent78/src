from types import SimpleNamespace

import numpy as np

from src.assembleur_core import TopologyElement, TopologyNodeType, TopologyWorld
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


def _element(element_id: str) -> TopologyElement:
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=[
            TopologyNodeType.OUVERTURE,
            TopologyNodeType.BASE,
            TopologyNodeType.LUMIERE,
        ],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )


def _auto_scenario(element_id: str, translation) -> SimpleNamespace:
    world = TopologyWorld()
    group_id = world.add_element_as_new_group(_element(element_id))
    world.setElementPose(element_id, np.eye(2), np.asarray(translation, dtype=float), False)
    return SimpleNamespace(
        source_type="auto",
        topoWorld=world,
        last_drawn=[{"topoElementId": element_id, "pts": {}}],
        orderedElementIds=[element_id],
        group_id=group_id,
    )


def _viewer_with_auto_scenarios():
    auto_active = _auto_scenario("T01", (1.0, 2.0))
    auto_inactive = _auto_scenario("T02", (-4.0, 3.0))
    manual = _auto_scenario("T03", (20.0, 30.0))
    manual.source_type = "manual"

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [auto_active, auto_inactive, manual]
    viewer.active_scenario_index = 0
    viewer.canvas_objects = CanvasObjectsCollection(auto_active.last_drawn)
    viewer._last_drawn = viewer.canvas_objects.entries
    auto_active.last_drawn = viewer._last_drawn
    viewer.auto_geom_state = {"ox": 0.0, "oy": 0.0, "thetaDeg": 0.0}
    viewer.offset = np.array((0.0, 0.0))
    viewer.zoom = 1.0
    viewer._drag = None
    viewer._clock_dragging = False
    viewer._bg_moving = None
    viewer._bg_resizing = None
    viewer._pan_anchor = None
    viewer._edge_choice = None
    viewer._ctrl_down = False
    viewer._clear_edge_highlights = lambda: None
    viewer._clear_nearest_line = lambda: None
    viewer._reset_assist = lambda: None
    viewer._redraw_from = lambda _entries: None
    viewer._invalidate_pick_cache = lambda: None
    viewer._clock_apply_auto_ref_sync = lambda: None
    viewer._simulationPersistCurrentAutoPlacement = lambda **_kwargs: None
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)
    viewer._project_all_auto_scenarios_from_core()
    return viewer, auto_active, auto_inactive, manual


def test_auto_move_preview_does_not_change_core_then_commits_once_globally():
    viewer, active, inactive, manual = _viewer_with_auto_scenarios()
    before_active = active.topoWorld.getElementPose("T01")[1].copy()
    before_inactive = inactive.topoWorld.getElementPose("T02")[1].copy()
    before_manual = manual.topoWorld.getElementPose("T03")[1].copy()
    viewer._sel = {
        "mode": "move_group",
        "core_group_id": active.group_id,
        "mouse_world_start": np.array((0.0, 0.0)),
        "auto_geom": True,
        "auto_state0": dict(viewer.auto_geom_state),
        "auto_move_preview_pts0": viewer._capture_active_auto_preview_pts(),
    }

    viewer._on_canvas_left_move(SimpleNamespace(x=5.0, y=-2.0))

    np.testing.assert_allclose(active.topoWorld.getElementPose("T01")[1], before_active)
    np.testing.assert_allclose(inactive.topoWorld.getElementPose("T02")[1], before_inactive)
    assert viewer.auto_geom_state == {"ox": 0.0, "oy": 0.0, "thetaDeg": 0.0}
    np.testing.assert_allclose(viewer._last_drawn[0]["pts"]["O"], before_active + (5.0, 2.0))
    np.testing.assert_allclose(inactive.last_drawn[0]["pts"]["O"], before_inactive)

    viewer._on_canvas_left_up(SimpleNamespace(x=5.0, y=-2.0))

    np.testing.assert_allclose(active.topoWorld.getElementPose("T01")[1], before_active + (5.0, 2.0))
    np.testing.assert_allclose(inactive.topoWorld.getElementPose("T02")[1], before_inactive + (5.0, 2.0))
    np.testing.assert_allclose(manual.topoWorld.getElementPose("T03")[1], before_manual)
    assert viewer.auto_geom_state == {"ox": 5.0, "oy": 2.0, "thetaDeg": 0.0}
    np.testing.assert_allclose(inactive.last_drawn[0]["pts"]["O"], before_inactive + (5.0, 2.0))


def test_auto_rotation_preview_and_discard_leave_core_and_state_unchanged():
    viewer, active, inactive, _manual = _viewer_with_auto_scenarios()
    before_active = active.topoWorld.getElementPose("T01")[1].copy()
    before_inactive = inactive.topoWorld.getElementPose("T02")[1].copy()
    snapshot = viewer._capture_active_auto_preview_pts()

    viewer._preview_auto_rotation_from_snapshot(snapshot, (0.0, 0.0), np.pi / 2.0)

    np.testing.assert_allclose(active.topoWorld.getElementPose("T01")[1], before_active)
    np.testing.assert_allclose(inactive.topoWorld.getElementPose("T02")[1], before_inactive)
    assert viewer.auto_geom_state == {"ox": 0.0, "oy": 0.0, "thetaDeg": 0.0}
    viewer._sel = {"mode": "rotate_group", "auto_geom": True}
    assert viewer._discard_auto_transform_preview() is True
    np.testing.assert_allclose(viewer._last_drawn[0]["pts"]["O"], before_active)


def test_auto_rigid_rotation_is_applied_once_to_all_auto_cores_only():
    viewer, active, inactive, manual = _viewer_with_auto_scenarios()
    before_active = active.topoWorld.getElementPose("T01")[1].copy()
    before_inactive = inactive.topoWorld.getElementPose("T02")[1].copy()
    before_manual = manual.topoWorld.getElementPose("T03")[1].copy()
    rotation = np.array(((0.0, -1.0), (1.0, 0.0)))

    viewer._apply_rigid_transform_to_all_auto_scenarios(rotation, np.zeros(2))
    viewer._project_all_auto_scenarios_from_core()

    np.testing.assert_allclose(active.topoWorld.getElementPose("T01")[1], rotation @ before_active)
    np.testing.assert_allclose(inactive.topoWorld.getElementPose("T02")[1], rotation @ before_inactive)
    np.testing.assert_allclose(manual.topoWorld.getElementPose("T03")[1], before_manual)
    np.testing.assert_allclose(inactive.last_drawn[0]["pts"]["O"], rotation @ before_inactive)


def test_auto_anchor_is_read_from_first_ordered_core_element_only():
    viewer, active, _inactive, _manual = _viewer_with_auto_scenarios()
    second = _element("T99")
    active.topoWorld.add_element_as_new_group(second)
    active.topoWorld.setElementPose("T99", np.eye(2), np.array((50.0, -25.0)), False)
    active.orderedElementIds = ["T99", "T01"]
    active.last_drawn[:] = list(reversed(active.last_drawn))

    anchor = viewer._auto_get_anchor_world_from_core(active)

    np.testing.assert_allclose(
        anchor,
        viewer._get_core_triangle_world_points(active.topoWorld, "T99")["L"],
    )
    viewer.auto_geom_state = None
    viewer._auto_ensure_geometry_state(active)
    assert viewer.auto_geom_state == {
        "ox": float(anchor[0]), "oy": float(anchor[1]), "thetaDeg": 0.0,
    }


def test_auto_anchor_rejects_empty_or_unknown_ordered_element_ids():
    viewer, active, _inactive, _manual = _viewer_with_auto_scenarios()

    active.orderedElementIds = []
    try:
        viewer._auto_get_anchor_world_from_core(active)
    except ValueError as exc:
        assert "orderedElementIds" in str(exc)
    else:
        raise AssertionError("une liste orderedElementIds vide doit être rejetée")

    active.orderedElementIds = ["T404"]
    try:
        viewer._auto_get_anchor_world_from_core(active)
    except KeyError as exc:
        assert "T404" in str(exc)
    else:
        raise AssertionError("un ElementID absent du Core doit être rejeté")


def test_auto_core_transform_projects_from_core():
    viewer, active, inactive, _manual = _viewer_with_auto_scenarios()
    active_pose_before = active.topoWorld.getElementPose("T01")[1].copy()
    inactive_pose_before = inactive.topoWorld.getElementPose("T02")[1].copy()

    viewer._apply_rigid_transform_to_all_auto_scenarios(
        np.eye(2), np.array((3.0, -4.0))
    )
    viewer._project_all_auto_scenarios_from_core()

    np.testing.assert_allclose(active.topoWorld.getElementPose("T01")[1], active_pose_before + (3.0, -4.0))
    np.testing.assert_allclose(inactive.topoWorld.getElementPose("T02")[1], inactive_pose_before + (3.0, -4.0))
    np.testing.assert_allclose(viewer._last_drawn[0]["pts"]["O"], active_pose_before + (3.0, -4.0))
    np.testing.assert_allclose(inactive.last_drawn[0]["pts"]["O"], inactive_pose_before + (3.0, -4.0))
