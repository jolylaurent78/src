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
    world.setElementPose("T01", np.eye(2), np.array((1.0, 2.0)), mirrored=False)
    world.setElementPose("T02", np.eye(2), np.array((7.0, -1.0)), mirrored=True)
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.canvas_objects = CanvasObjectsCollection([
        {"topoElementId": "T01", "pts": {}},
        {"topoElementId": "T02", "pts": {}},
    ])
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.scenarios = [SimpleNamespace(topoWorld=world, source_type="manual")]
    viewer.active_scenario_index = 0
    viewer._is_active_auto_scenario = lambda: False
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)
    viewer._redraw_from = lambda _entries: None
    viewer._project_core_group_to_last_drawn(world, group_id)
    return viewer, world, group_id, first, second


def test_manual_flip_uses_core_once_and_preserves_cache_mirrored_field():
    viewer, world, group_id, _first, _second = _viewer_with_group()
    points_before = viewer._get_core_triangle_world_points(world, "T01")
    centroid_before = viewer._get_core_group_world_centroid(world, group_id)
    calls = []
    original_flip = world.flip_group
    world.flip_group = lambda *args: (calls.append(args), original_flip(*args))[1]
    viewer._ctx_target_element_id = "T01"

    viewer._ctx_flip_selected()

    assert len(calls) == 1
    committed_group, pivot, axis = calls[0]
    assert committed_group == group_id
    np.testing.assert_allclose(pivot, centroid_before)
    np.testing.assert_allclose(axis, points_before["L"] - points_before["O"])
    assert all("mirrored" not in entry for entry in viewer._last_drawn)
    assert world.getElementPose("T01")[2] is True
    assert world.getElementPose("T02")[2] is False
    assert viewer._get_core_element_mirrored("T01") is True
    assert viewer._get_core_element_mirrored("T02") is False


def test_double_manual_flip_returns_exact_core_geometry():
    viewer, world, _group_id, first, second = _viewer_with_group()
    before = {
        element.element_id: viewer._get_core_triangle_world_points(world, element.element_id)
        for element in (first, second)
    }
    mirrored_before = {element.element_id: world.getElementPose(element.element_id)[2] for element in (first, second)}

    viewer._ctx_target_element_id = "T01"
    viewer._ctx_flip_selected()
    viewer._ctx_target_element_id = "T01"
    viewer._ctx_flip_selected()

    for element in (first, second):
        after = viewer._get_core_triangle_world_points(world, element.element_id)
        for vertex in ("O", "B", "L"):
            np.testing.assert_allclose(after[vertex], before[element.element_id][vertex], atol=1e-12)
        assert world.getElementPose(element.element_id)[2] is mirrored_before[element.element_id]


def test_core_mirrored_state_is_used_even_when_cache_disagrees():
    viewer, world, _group_id, _first, _second = _viewer_with_group()
    world.setElementPose("T01", *world.getElementPose("T01")[:2], mirrored=True)

    assert viewer._get_core_element_mirrored("T01") is True
    assert "mirrored" not in viewer._last_drawn[0]


def test_auto_flip_remains_disabled_without_core_change():
    viewer, world, _group_id, _first, _second = _viewer_with_group()
    viewer._is_active_auto_scenario = lambda: True
    before = world.getElementPose("T01")
    viewer._ctx_target_element_id = "T01"

    viewer._ctx_flip_selected()

    after = world.getElementPose("T01")
    np.testing.assert_allclose(after[0], before[0])
    np.testing.assert_allclose(after[1], before[1])
    assert after[2] is before[2]
