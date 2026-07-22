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
    first, second, outsider = _element("T01"), _element("T02"), _element("T03")
    first_group = world.add_element_as_new_group(first)
    second_group = world.add_element_as_new_group(second)
    world.add_element_as_new_group(outsider)
    group_id = world.union_groups(first_group, second_group)
    world.setElementPose("T01", np.eye(2), np.array((1.0, 2.0)), mirrored=True)
    world.setElementPose("T02", np.eye(2), np.array((8.0, -3.0)), mirrored=False)
    world.setElementPose("T03", np.eye(2), np.array((30.0, 40.0)), mirrored=False)
    entries = [
        {"topoElementId": "T01", "pts": {"O": (-99, -99)}},
        {"topoElementId": "T02", "pts": {"O": (-98, -98)}},
        {"topoElementId": "T03", "pts": {"O": (-97, -97)}},
    ]
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.canvas_objects = CanvasObjectsCollection(entries)
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.scenarios = [SimpleNamespace(topoWorld=world, source_type="manual")]
    viewer.active_scenario_index = 0
    viewer._is_active_auto_scenario = lambda: False
    viewer._group_drag_snap_candidate = None
    return viewer, world, group_id, first, second, outsider


def _expected_points(element):
    return {
        vertex_type: element.localToWorld(element.vertex_local_xy[index])
        for index, vertex_type in enumerate(element.vertex_types)
    }


def _prepare_manual_move(viewer, world, group_id, start=(0.0, 0.0)):
    viewer._project_core_group_to_last_drawn(world, group_id)
    viewer._sel = {
        "mode": "move_group",
        "core_group_id": group_id,
        "mouse_world_start": np.array(start, dtype=float),
        "move_preview_initial_pts": viewer._capture_move_preview_initial_pts(world, group_id),
    }
    viewer._drag = None
    viewer._edge_choice = None
    viewer._ctrl_down = False
    viewer._screen_to_world = lambda x, y: (float(x), float(y))
    viewer._clear_nearest_line = lambda: None
    viewer._clear_edge_highlights = lambda: None
    viewer._clock_apply_auto_ref_sync = lambda: None
    viewer._redraw_from = lambda _entries: None
    viewer._reset_assist = lambda: None
    viewer._bg_resizing = None
    viewer._bg_moving = None
    viewer._clock_dragging = False
    viewer._pan_anchor = None
    viewer.auto_rotation_state = None


def test_manual_move_preview_does_not_modify_core():
    viewer, world, group_id, first, second, _outsider = _viewer_with_group()
    _prepare_manual_move(viewer, world, group_id)
    before = {element_id: world.getElementPose(element_id) for element_id in ("T01", "T02")}
    calls = []
    original_move_group = world.move_group

    def record_move(*args):
        calls.append(args)
        return original_move_group(*args)

    world.move_group = record_move

    for event in (SimpleNamespace(x=1.0, y=2.0), SimpleNamespace(x=3.0, y=5.0)):
        viewer._on_canvas_left_move(event)

    assert calls == []
    for element_id in ("T01", "T02"):
        after = world.getElementPose(element_id)
        for actual, expected in zip(after[:2], before[element_id][:2]):
            np.testing.assert_allclose(actual, expected)
        assert after[2] is before[element_id][2]
    for element in (first, second):
        entry = viewer.canvas_objects.get_by_topology_id(element.element_id)
        for vertex, expected in _expected_points(element).items():
            np.testing.assert_allclose(entry["pts"][vertex], expected + np.array((3.0, 5.0)))


def test_manual_move_commits_one_total_delta_on_release():
    viewer, world, group_id, first, second, _outsider = _viewer_with_group()
    _prepare_manual_move(viewer, world, group_id, start=(10.0, -4.0))
    calls = []
    original_move_group = world.move_group
    world.move_group = lambda *args: (calls.append(args), original_move_group(*args))[1]

    for event in (SimpleNamespace(x=12.0, y=-1.0), SimpleNamespace(x=16.0, y=3.0)):
        viewer._on_canvas_left_move(event)
    viewer._on_canvas_left_up(SimpleNamespace(x=16.0, y=3.0))

    assert calls == [(group_id, 6.0, 7.0)]
    for element in (first, second):
        _rotation, translation, _mirrored = world.getElementPose(element.element_id)
        expected_translation = {"T01": (7.0, 9.0), "T02": (14.0, 4.0)}[element.element_id]
        np.testing.assert_allclose(translation, expected_translation)
        entry = viewer.canvas_objects.get_by_topology_id(element.element_id)
        for vertex, expected in _expected_points(element).items():
            np.testing.assert_allclose(entry["pts"][vertex], expected)


def test_manual_move_result_is_independent_of_motion_event_count():
    def run(events):
        viewer, world, group_id, first, second, _outsider = _viewer_with_group()
        _prepare_manual_move(viewer, world, group_id)
        for event in events:
            viewer._on_canvas_left_move(SimpleNamespace(x=event[0], y=event[1]))
        viewer._on_canvas_left_up(SimpleNamespace(x=20.0, y=-6.0))
        return [world.getElementPose(element.element_id) for element in (first, second)]

    one_event = run([(20.0, -6.0)])
    twenty_events = run([(float(i), -0.3 * i) for i in range(1, 21)])
    for actual, expected in zip(twenty_events, one_event):
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
        assert actual[2] is expected[2]


def test_escape_restores_core_projection_without_committing_preview():
    viewer, world, group_id, first, second, _outsider = _viewer_with_group()
    _prepare_manual_move(viewer, world, group_id)
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)
    viewer._hide_tooltip = lambda: None
    viewer._clock_trace_active = False
    viewer._clock_arc_active = False
    viewer._clock_measure_active = False
    viewer._clock_setref_active = False
    viewer._bg_calib_active = False
    discard_calls = []
    discard = viewer._discard_manual_move_preview
    viewer._discard_manual_move_preview = lambda: (discard_calls.append(True), discard())[1]
    original_move_group = world.move_group
    world.move_group = lambda *_args: (_ for _ in ()).throw(
        AssertionError("core move forbidden during ESC")
    )
    viewer._on_canvas_left_move(SimpleNamespace(x=9.0, y=-2.0))
    assert viewer._on_escape_key(SimpleNamespace()) == "break"

    assert discard_calls == [True]
    assert viewer._sel is None
    for element in (first, second):
        entry = viewer.canvas_objects.get_by_topology_id(element.element_id)
        for vertex, expected in _expected_points(element).items():
            np.testing.assert_allclose(entry["pts"][vertex], expected)


def test_manual_move_preserves_rotation_mirror_and_local_coordinates():
    viewer, world, group_id, first, second, _outsider = _viewer_with_group()
    rotation = np.array(((0.0, -1.0), (1.0, 0.0)))
    world.setElementPose("T01", rotation, np.array((4.0, -7.0)), mirrored=True)
    world.setElementPose("T02", rotation, np.array((-2.0, 11.0)), mirrored=False)
    local_before = {
        element.element_id: {index: np.array(value, copy=True) for index, value in element.vertex_local_xy.items()}
        for element in (first, second)
    }
    _prepare_manual_move(viewer, world, group_id)
    viewer._on_canvas_left_move(SimpleNamespace(x=-3.0, y=8.0))
    viewer._on_canvas_left_up(SimpleNamespace(x=-3.0, y=8.0))

    for element, translation0, mirrored in (
        (first, (4.0, -7.0), True),
        (second, (-2.0, 11.0), False),
    ):
        rotation_after, translation_after, mirrored_after = world.getElementPose(element.element_id)
        np.testing.assert_allclose(rotation_after, rotation)
        np.testing.assert_allclose(translation_after, np.array(translation0) + np.array((-3.0, 8.0)))
        assert mirrored_after is mirrored
        for index, local_xy in local_before[element.element_id].items():
            np.testing.assert_allclose(element.vertex_local_xy[index], local_xy)


def test_legacy_snap_does_not_also_commit_a_core_translation():
    viewer, world, group_id, _first, _second, _outsider = _viewer_with_group()
    _prepare_manual_move(viewer, world, group_id)

    class EdgeChoice:
        kind = "vertex-edge"

        @staticmethod
        def computeRigidTransform():
            return np.array(((0.0, -1.0), (1.0, 0.0))), np.array((2.0, 3.0))

        @staticmethod
        def createTopologyAttachments(**_kwargs):
            return [object()]

    viewer._sel.update({
        "anchor": {"type": "vertex", "tid": 0, "vkey": "O"},
        "suppress_assist": False,
    })
    viewer._edge_choice = (0, "O", 2, "O", EdgeChoice())
    original_group_of_element = world.get_group_of_element
    world.get_group_of_element = lambda element_id: (
        "G-final" if str(element_id) == "T01" else original_group_of_element(element_id)
    )
    world.beginTopoTransaction = lambda: None
    world.apply_attachments = lambda _attachments: "G-final"
    world.commitTopoTransaction = lambda: None
    rigid_transform_calls = []
    world.apply_group_rigid_transform = lambda *args: rigid_transform_calls.append(args)
    projected_group_ids = []
    viewer._project_core_group_to_last_drawn = lambda _world, core_group_id: projected_group_ids.append(core_group_id)
    viewer._commit_move_group_to_core = lambda *_args: (_ for _ in ()).throw(
        AssertionError("free-move commit forbidden after a legacy snap")
    )

    viewer._on_canvas_left_up(SimpleNamespace(x=10.0, y=5.0))

    assert len(rigid_transform_calls) == 1
    assert rigid_transform_calls[0][0] == group_id
    np.testing.assert_allclose(rigid_transform_calls[0][1], np.array(((0.0, -1.0), (1.0, 0.0))))
    np.testing.assert_allclose(rigid_transform_calls[0][2], np.array((-3.0, 13.0)))
    assert projected_group_ids == [group_id, "G-final"]
