from types import SimpleNamespace

import numpy as np

from src.assembleur_balises import Beacon, BeaconCatalog
from src.assembleur_core import TopologyElement, TopologyNodeType, TopologyWorld
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


def _element(element_id, vertex_types=None):
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=vertex_types or [
            TopologyNodeType.OUVERTURE,
            TopologyNodeType.BASE,
            TopologyNodeType.LUMIERE,
        ],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )


def _viewer_with_group():
    world = TopologyWorld()
    first, second = _element("T01"), _element("T02")
    group_id = world.union_groups(
        world.add_element_as_new_group(first), world.add_element_as_new_group(second)
    )
    world.setElementPose("T01", np.eye(2), np.array((2.0, -1.0)), mirrored=False)
    world.setElementPose("T02", np.eye(2), np.array((9.0, 4.0)), mirrored=True)
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


def test_core_world_points_use_vertex_types_and_local_to_world_with_mirror():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    world = TopologyWorld()
    element = _element(
        "T99",
        [TopologyNodeType.LUMIERE, TopologyNodeType.OUVERTURE, TopologyNodeType.BASE],
    )
    world.add_element_as_new_group(element)
    world.setElementPose("T99", np.array(((0.0, -1.0), (1.0, 0.0))), np.array((7.0, -3.0)), mirrored=True)

    points = viewer._get_core_triangle_world_points(world, "T99")

    type_to_key = {
        TopologyNodeType.OUVERTURE: "O",
        TopologyNodeType.BASE: "B",
        TopologyNodeType.LUMIERE: "L",
    }
    for index, vertex_type in enumerate(element.vertex_types):
        np.testing.assert_allclose(
            points[type_to_key[vertex_type]], element.localToWorld(element.vertex_local_xy[index])
        )


def test_manual_orient_north_commits_core_once_and_overwrites_divergent_cache():
    viewer, world, group_id, first, second = _viewer_with_group()
    cache_before = {
        "O": np.array((1000.0, 1000.0)), "B": np.array((1001.0, 1000.0)), "L": np.array((1002.0, 1000.0)),
    }
    viewer._last_drawn[0]["pts"] = {key: value.copy() for key, value in cache_before.items()}
    centroid_before = sum(viewer._get_core_triangle_world_points(world, "T01").values()) / 3.0
    distance_before = np.linalg.norm(
        viewer._get_core_triangle_world_points(world, "T01")["O"]
        - viewer._get_core_triangle_world_points(world, "T02")["O"]
    )
    calls = []
    original_rotate = world.rotate_group
    world.rotate_group = lambda *args: (calls.append(args), original_rotate(*args))[1]
    projected = []
    original_project = viewer._project_core_group_to_last_drawn
    viewer._project_core_group_to_last_drawn = lambda w, gid: (projected.append(gid), original_project(w, gid))[1]
    viewer._ctx_target_element_id = "T01"

    viewer._ctx_orient_OL_north()

    assert len(calls) == 1
    assert calls[0][0] == group_id
    assert projected == [world.get_group_of_element("T01")]
    points_after = viewer._last_drawn[0]["pts"]
    delta = points_after["L"] - points_after["O"]
    assert abs(float(delta[0])) < 1e-10
    assert float(delta[1]) > 0.0
    centroid_after = sum(viewer._get_core_triangle_world_points(world, "T01").values()) / 3.0
    np.testing.assert_allclose(centroid_after, centroid_before)
    distance_after = np.linalg.norm(
        viewer._get_core_triangle_world_points(world, "T01")["O"]
        - viewer._get_core_triangle_world_points(world, "T02")["O"]
    )
    np.testing.assert_allclose(distance_after, distance_before)
    assert world.getElementPose("T01")[2] is False
    assert world.getElementPose("T02")[2] is True
    assert not np.allclose(points_after["O"], cache_before["O"])


def test_auto_orient_north_commits_a_core_first_global_rotation():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    catalog = BeaconCatalog()
    catalog._by_id = {
        "BAL-1": Beacon("BAL-1", "Balise", 0, 0, 0, 0, 10.0, 5.0),
    }
    world = TopologyWorld(beacon_catalog=catalog)
    group_id = world.add_element_as_new_group(_element("T01"))
    viewer.canvas_objects = CanvasObjectsCollection([
        {"topoElementId": "T01", "pts": {}},
    ])
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.scenarios = [SimpleNamespace(
        topoWorld=world, source_type="auto", last_drawn=viewer._last_drawn,
        orderedElementIds=["T01"],
        name="Auto test",
    )]
    viewer.active_scenario_index = 0
    viewer.auto_rotation_state = {"thetaDeg": 0.0}
    world.setElementPose(
        "T01",
        np.array(((0.0, -1.0), (1.0, 0.0))),
        np.zeros(2),
        mirrored=False,
    )
    anchor = world.createGroupAnchor(
        group_id,
        "BAL-1",
        world.get_element_vertex_node_id_by_type("T01", "L"),
    )
    world.applyGroupAnchor(anchor.anchor_id)
    viewer._project_core_group_to_last_drawn(world, group_id)
    viewer._redraw_from = lambda _entries: None
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)
    viewer._ctx_target_element_id = "T01"

    viewer._ctx_orient_OL_north()

    np.testing.assert_allclose(viewer.auto_rotation_state["thetaDeg"], 270.0)
    vector = viewer._last_drawn[0]["pts"]["L"] - viewer._last_drawn[0]["pts"]["O"]
    assert abs(float(vector[0])) < 1e-10
    assert float(vector[1]) > 0.0
