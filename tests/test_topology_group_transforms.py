import math

import numpy as np
import pytest

from src.assembleur_core import TopologyElement, TopologyWorld


def _element(element_id: str) -> TopologyElement:
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )


def _world_with_two_element_group():
    world = TopologyWorld()
    first = _element("T01")
    second = _element("T02")
    group_a = world.add_element_as_new_group(first)
    group_b = world.add_element_as_new_group(second)
    group_id = world.union_groups(group_a, group_b)
    return world, group_id


def _world_points(world, element_id: str):
    element = world.elements[element_id]
    return [
        world.elementLocalToWorld(element_id, element.vertex_local_xy[index])
        for index in range(3)
    ]


def test_move_group_translates_every_pose_and_preserves_relative_distances():
    world, group_id = _world_with_two_element_group()
    world.setElementPose("T01", np.eye(2), np.array((1.0, -2.0)))
    world.setElementPose("T02", np.eye(2), np.array((8.0, 5.0)))

    before = {element_id: _world_points(world, element_id) for element_id in ("T01", "T02")}
    before_distance = np.linalg.norm(before["T02"][0] - before["T01"][0])

    world.move_group(group_id, 3.5, -4.0)

    delta = np.array((3.5, -4.0))
    after = {element_id: _world_points(world, element_id) for element_id in ("T01", "T02")}
    for element_id in ("T01", "T02"):
        for point_before, point_after in zip(before[element_id], after[element_id]):
            np.testing.assert_allclose(point_after, point_before + delta)
    assert np.linalg.norm(after["T02"][0] - after["T01"][0]) == pytest.approx(before_distance)


@pytest.mark.parametrize("angle_rad", [math.pi / 2.0, math.pi])
def test_rotate_group_rotates_all_element_geometry_around_world_center(angle_rad):
    world, group_id = _world_with_two_element_group()
    rotation_90 = np.array(((0.0, -1.0), (1.0, 0.0)))
    world.setElementPose("T01", np.eye(2), np.array((2.0, 1.0)))
    world.setElementPose("T02", rotation_90, np.array((-3.0, 4.0)))

    center = np.array((1.5, -2.0))
    before = {element_id: _world_points(world, element_id) for element_id in ("T01", "T02")}
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    Q = np.array(((c, -s), (s, c)))

    world.rotate_group(group_id, center, angle_rad)

    for element_id in ("T01", "T02"):
        for point_before, point_after in zip(before[element_id], _world_points(world, element_id)):
            np.testing.assert_allclose(point_after, Q @ (point_before - center) + center)

    # Une rotation rigide laisse les distances entre les éléments inchangées.
    before_distance = np.linalg.norm(before["T02"][1] - before["T01"][1])
    after_distance = np.linalg.norm(
        _world_points(world, "T02")[1] - _world_points(world, "T01")[1]
    )
    assert after_distance == pytest.approx(before_distance)


def test_flip_group_reflects_world_geometry_and_refactorizes_mirrored_state():
    world, group_id = _world_with_two_element_group()
    rotation_90 = np.array(((0.0, -1.0), (1.0, 0.0)))
    world.setElementPose("T01", np.eye(2), np.array((2.0, 3.0)), mirrored=False)
    world.setElementPose("T02", rotation_90, np.array((-1.0, 5.0)), mirrored=True)
    before = {element_id: _world_points(world, element_id) for element_id in ("T01", "T02")}

    # Axe horizontal y=1, donc (x, y) devient (x, 2-y).
    world.flip_group(group_id, axis_point=(0.0, 1.0), axis_direction=(1.0, 0.0))

    for element_id in ("T01", "T02"):
        for point_before, point_after in zip(before[element_id], _world_points(world, element_id)):
            np.testing.assert_allclose(point_after, (point_before[0], 2.0 - point_before[1]))

    assert world.getElementPose("T01")[2] is True
    assert world.getElementPose("T02")[2] is False


def test_apply_group_rigid_transform_preserves_geometry_and_mirrored_state():
    world, group_id = _world_with_two_element_group()
    rotation_90 = np.array(((0.0, -1.0), (1.0, 0.0)))
    world.setElementPose("T01", np.eye(2), np.array((2.0, -3.0)), mirrored=False)
    world.setElementPose("T02", rotation_90, np.array((-5.0, 4.0)), mirrored=True)
    before = {element_id: _world_points(world, element_id) for element_id in ("T01", "T02")}
    mirrored_before = {element_id: world.getElementPose(element_id)[2] for element_id in ("T01", "T02")}
    R = np.array(((0.0, -1.0), (1.0, 0.0)))
    T = np.array((7.0, -2.0))

    world.apply_group_rigid_transform(group_id, R, T)

    for element_id in ("T01", "T02"):
        for point_before, point_after in zip(before[element_id], _world_points(world, element_id)):
            np.testing.assert_allclose(point_after, R @ point_before + T)
        assert world.getElementPose(element_id)[2] is mirrored_before[element_id]


@pytest.mark.parametrize(
    "R, T",
    [
        (np.array(((1.0, 0.0), (0.0, -1.0))), np.zeros(2)),
        (np.array(((2.0, 0.0), (0.0, 1.0))), np.zeros(2)),
        (np.eye(2), np.array((float("nan"), 0.0))),
    ],
)
def test_apply_group_rigid_transform_rejects_invalid_parameters(R, T):
    world, group_id = _world_with_two_element_group()
    with pytest.raises(ValueError):
        world.apply_group_rigid_transform(group_id, R, T)


def test_group_transformations_require_a_live_canonical_core_group():
    world = TopologyWorld()
    with pytest.raises(ValueError, match="canonique"):
        world.move_group("G-inconnu", 1.0, 2.0)
    with pytest.raises(ValueError, match="canonique"):
        world.rotate_group("G-inconnu", (0.0, 0.0), 0.0)
    with pytest.raises(ValueError, match="canonique"):
        world.flip_group("G-inconnu", (0.0, 0.0), (1.0, 0.0))


def test_transforming_a_cloned_world_never_changes_the_original_world():
    world, group_id = _world_with_two_element_group()
    world.setElementPose("T01", np.eye(2), np.array((2.0, 3.0)))
    world.setElementPose("T02", np.eye(2), np.array((-4.0, 1.0)), mirrored=True)
    before = {element_id: _world_points(world, element_id) for element_id in ("T01", "T02")}

    clone = world.clonePhysicalState()
    clone.move_group(group_id, 10.0, -7.0)
    clone.rotate_group(group_id, (0.0, 0.0), math.pi / 2.0)
    clone.flip_group(group_id, (0.0, 0.0), (1.0, 0.0))

    for element_id in ("T01", "T02"):
        for point_before, point_after in zip(before[element_id], _world_points(world, element_id)):
            np.testing.assert_allclose(point_after, point_before)
