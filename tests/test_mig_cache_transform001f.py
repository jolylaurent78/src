from types import SimpleNamespace

import numpy as np

from src.assembleur_core import TopologyElement, TopologyNodeType, TopologyWorld
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


class _BeaconResolver:
    def contains(self, beacon_id):
        return beacon_id == "BEA-0001"

    def get_world(self, beacon_id):
        if beacon_id != "BEA-0001":
            raise KeyError(beacon_id)
        return (0.0, 0.0)


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


def _viewer_with_split_groups(anchored_group: str | None = None):
    world = TopologyWorld(beacon_resolver=_BeaconResolver())
    main_group_id = world.add_element_as_new_group(_element("T01"))
    new_group_id = world.add_element_as_new_group(_element("T02"))
    world.setElementPose("T01", np.eye(2), np.array((0.0, 0.0)), mirrored=False)
    world.setElementPose("T02", np.eye(2), np.array((0.0, 0.0)), mirrored=False)

    if anchored_group == "main":
        world.createGroupAnchor(main_group_id, "BEA-0001", "T01:N0")
    elif anchored_group == "new":
        world.createGroupAnchor(new_group_id, "BEA-0001", "T02:N0")

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.canvas_objects = CanvasObjectsCollection([
        {"topoElementId": "T01", "pts": {}},
        {"topoElementId": "T02", "pts": {}},
    ])
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.scenarios = [SimpleNamespace(topoWorld=world)]
    viewer.active_scenario_index = 0
    viewer.zoom = 1.0
    viewer.offset = (0.0, 0.0)
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)
    viewer._sel = {"stale": True}
    viewer._reset_assist = lambda: None
    viewer._redraw_from = lambda _entries: None
    viewer.refreshCheminTreeView = lambda: None
    return viewer, world, main_group_id, new_group_id


def _assert_pose_unchanged(actual, expected) -> None:
    actual_rotation, actual_translation, actual_mirrored = actual
    expected_rotation, expected_translation, expected_mirrored = expected
    np.testing.assert_allclose(actual_rotation, expected_rotation)
    np.testing.assert_allclose(actual_translation, expected_translation)
    assert actual_mirrored is expected_mirrored


def test_split_post_layout_moves_core_then_reprojects_without_reverse_sync():
    viewer, world, main_group_id, new_group_id = _viewer_with_split_groups()
    move_calls = []
    project_calls = []
    original_move = world.move_group
    original_project = viewer._project_core_group_to_last_drawn

    def record_move(group_id, dx, dy):
        move_calls.append((group_id, dx, dy))
        return original_move(group_id, dx, dy)

    def record_project(projected_world, group_id):
        project_calls.append(group_id)
        return original_project(projected_world, group_id)

    world.move_group = record_move
    viewer._project_core_group_to_last_drawn = record_project

    viewer._applyDegrouperResultToTk({
        "mainGroupId": main_group_id,
        "newGroupIds": [new_group_id],
    })

    assert project_calls[:2] == [main_group_id, new_group_id]
    assert len(move_calls) == 1
    moved_group_id, dx, dy = move_calls[0]
    assert moved_group_id == new_group_id
    np.testing.assert_allclose((dx, dy), (0.0, -30.0))
    np.testing.assert_allclose(world.getElementPose("T01")[1], (0.0, 0.0))
    np.testing.assert_allclose(world.getElementPose("T02")[1], (0.0, -30.0))
    np.testing.assert_allclose(viewer._last_drawn[1]["pts"]["O"], (0.0, -30.0))
    assert project_calls[-1] == new_group_id
    assert viewer._sel is None


def test_split_post_layout_keeps_an_anchored_main_group_fixed():
    viewer, world, main_group_id, new_group_id = _viewer_with_split_groups(
        anchored_group="main"
    )
    main_pose_before = world.getElementPose("T01")
    move_calls = []
    original_move = world.move_group

    def record_move(group_id, dx, dy):
        move_calls.append(group_id)
        return original_move(group_id, dx, dy)

    world.move_group = record_move

    viewer._applyDegrouperResultToTk({
        "mainGroupId": main_group_id,
        "newGroupIds": [new_group_id],
    })

    assert move_calls == [new_group_id]
    _assert_pose_unchanged(world.getElementPose("T01"), main_pose_before)


def test_split_post_layout_does_not_move_an_anchored_new_group():
    viewer, world, main_group_id, new_group_id = _viewer_with_split_groups(
        anchored_group="new"
    )
    new_pose_before = world.getElementPose("T02")
    move_calls = []
    original_move = world.move_group

    def record_move(group_id, dx, dy):
        move_calls.append(group_id)
        return original_move(group_id, dx, dy)

    world.move_group = record_move

    viewer._applyDegrouperResultToTk({
        "mainGroupId": main_group_id,
        "newGroupIds": [new_group_id],
    })

    assert move_calls == []
    _assert_pose_unchanged(world.getElementPose("T02"), new_pose_before)


def test_screen_delta_conversion_is_pure_and_respects_canvas_y_axis():
    viewer, _world, _main_group_id, _new_group_id = _viewer_with_split_groups()
    viewer.zoom = 2.0
    viewer.offset = (100.0, 50.0)
    before = [dict(entry) for entry in viewer._last_drawn]

    delta = viewer._screen_delta_to_world_delta(30.0, -14.0)

    np.testing.assert_allclose(delta, (15.0, 7.0))
    assert viewer._last_drawn == before
