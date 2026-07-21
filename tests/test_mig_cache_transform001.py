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
        vertex_types=[
            TopologyNodeType.OUVERTURE,
            TopologyNodeType.BASE,
            TopologyNodeType.LUMIERE,
        ],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )


def _viewer_with_group():
    world = TopologyWorld()
    first = _element("T01")
    second = _element("T02")
    outsider = _element("T03")
    first_group = world.add_element_as_new_group(first)
    second_group = world.add_element_as_new_group(second)
    world.add_element_as_new_group(outsider)
    group_id = world.union_groups(first_group, second_group)

    world.setElementPose("T01", np.eye(2), np.array((1.0, 2.0)), mirrored=True)
    world.setElementPose("T02", np.eye(2), np.array((8.0, -3.0)), mirrored=False)
    world.setElementPose("T03", np.eye(2), np.array((30.0, 40.0)), mirrored=False)

    entries = [
        {"id": 1, "topoElementId": "T01", "mirrored": True, "pts": {"O": (-99, -99)}},
        {"id": 2, "topoElementId": "T02", "mirrored": False, "pts": {"O": (-98, -98)}},
        {"id": 3, "topoElementId": "T03", "mirrored": False, "pts": {"O": (-97, -97)}},
    ]
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.canvas_objects = CanvasObjectsCollection(entries)
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.scenarios = [SimpleNamespace(topoWorld=world, source_type="manual")]
    viewer.active_scenario_index = 0
    viewer._is_active_auto_scenario = lambda: False
    return viewer, world, group_id, first, second, outsider


def _expected_points(element):
    return {
        vertex_type: element.localToWorld(element.vertex_local_xy[index])
        for index, vertex_type in enumerate(element.vertex_types)
    }


def test_project_core_element_rebuilds_pts_from_vertex_types_and_local_to_world():
    viewer, world, _group_id, first, _second, _outsider = _viewer_with_group()

    entry = viewer._project_core_element_to_last_drawn(world, "T01")

    expected = _expected_points(first)
    for vertex_type in ("O", "B", "L"):
        np.testing.assert_allclose(entry["pts"][vertex_type], expected[vertex_type])
    # La projection ne réécrit pas ce flag de cache.
    assert entry["mirrored"] is True


def test_project_core_group_updates_only_its_members():
    viewer, world, group_id, first, second, _outsider = _viewer_with_group()
    outsider_before = dict(viewer._last_drawn[2]["pts"])

    projected = viewer._project_core_group_to_last_drawn(world, group_id)

    assert {entry["topoElementId"] for entry in projected} == {"T01", "T02"}
    for element, entry in ((first, viewer._last_drawn[0]), (second, viewer._last_drawn[1])):
        for vertex_type, expected in _expected_points(element).items():
            np.testing.assert_allclose(entry["pts"][vertex_type], expected)
    assert viewer._last_drawn[2]["pts"] == outsider_before


def test_manual_move_updates_core_then_projects_without_reverse_sync():
    viewer, world, group_id, first, second, _outsider = _viewer_with_group()
    before = {
        element_id: world.getElementPose(element_id)
        for element_id in ("T01", "T02")
    }
    local_before = {
        element_id: dict(world.elements[element_id].vertex_local_xy)
        for element_id in ("T01", "T02")
    }
    calls = []
    original_move_group = world.move_group

    def record_move(core_group_id, dx, dy):
        calls.append((core_group_id, dx, dy))
        return original_move_group(core_group_id, dx, dy)

    world.move_group = record_move
    viewer._sync_group_elements_pose_to_core = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(AssertionError("reverse sync interdit"))
    )

    for dx, dy in ((1.0, 2.0), (3.0, -1.0), (-2.0, 4.0)):
        viewer._move_group_world(group_id, dx, dy, move_member_entries=[])

    assert calls == [
        (group_id, 1.0, 2.0),
        (group_id, 3.0, -1.0),
        (group_id, -2.0, 4.0),
    ]
    total_delta = np.array((2.0, 5.0))
    for element_id, element in (("T01", first), ("T02", second)):
        rotation_after, translation_after, mirrored_after = world.getElementPose(element_id)
        rotation_before, translation_before, mirrored_before = before[element_id]
        np.testing.assert_allclose(rotation_after, rotation_before)
        np.testing.assert_allclose(translation_after, translation_before + total_delta)
        assert mirrored_after is mirrored_before
        assert element.vertex_local_xy == local_before[element_id]
        for vertex_type, expected in _expected_points(element).items():
            np.testing.assert_allclose(
                viewer.canvas_objects.get_by_topology_id(element_id)["pts"][vertex_type],
                expected,
            )


def test_free_move_release_does_not_trigger_reverse_sync():
    viewer, _world, group_id, _first, _second, _outsider = _viewer_with_group()
    viewer._sel = {
        "mode": "move_group",
        "core_group_id": group_id,
        "move_member_entries": [viewer._last_drawn[0], viewer._last_drawn[1]],
        "mouse_world_start": np.array((0.0, 0.0)),
    }
    viewer._edge_choice = None
    viewer._ctrl_down = False
    viewer._clock_dragging = False
    viewer._bg_resizing = None
    viewer._bg_moving = None
    viewer._pan_anchor = None
    viewer._drag = None
    viewer.auto_geom_state = None
    viewer._clear_edge_highlights = lambda: None
    viewer._reset_assist = lambda: None
    viewer._redraw_from = lambda _entries: None
    viewer._screen_to_world = lambda x, y: (float(x), float(y))
    viewer._sync_group_elements_pose_to_core = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(AssertionError("reverse sync interdit au relâchement libre"))
    )

    viewer._on_canvas_left_up(SimpleNamespace(x=0.0, y=0.0))

    assert viewer._sel is None


def test_automatic_move_commit_uses_the_shared_core_first_path():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.auto_geom_state = {"ox": 10.0, "oy": -4.0, "thetaDeg": 0.0}
    viewer.scenarios = []
    viewer._is_active_auto_scenario = lambda: True
    viewer._autoEnsureLocalGeometry = lambda _scen: None
    viewer._get_active_scenario = lambda: SimpleNamespace(source_type="auto")

    viewer._move_group_world("G-AUTO", 2.5, -3.0)

    assert viewer.auto_geom_state["ox"] == 12.5
    assert viewer.auto_geom_state["oy"] == -7.0
