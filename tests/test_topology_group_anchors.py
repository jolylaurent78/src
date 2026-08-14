import xml.etree.ElementTree as ET
import math
from types import SimpleNamespace

import pytest

from src.assembleur_core import TopologyElement, TopologyWorld
from src.assembleur_tk import TriangleViewerManual


class _BeaconResolver:
    _world_by_id = {"BEA-0001": (10.0, -5.0), "BEA-0002": (-7.0, 3.0)}

    def contains(self, beacon_id):
        return beacon_id in self._world_by_id

    def get_world(self, beacon_id):
        return self._world_by_id[beacon_id]


class _MutableBeaconResolver(_BeaconResolver):
    def set_world(self, beacon_id, world_xy):
        self._world_by_id = {**self._world_by_id, beacon_id: world_xy}


def _element(element_id: str) -> TopologyElement:
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )


def _world_with_two_groups() -> tuple[TopologyWorld, str, str]:
    world = TopologyWorld(beacon_resolver=_BeaconResolver())
    first = world.add_element_as_new_group(_element("T01"))
    second = world.add_element_as_new_group(_element("T02"))
    return world, first, second


def _node_id(world: TopologyWorld, element_id: str = "T01") -> str:
    return str(world.elements[element_id].vertexes[0].node_id)


def test_group_anchor_lifecycle_and_clone_share_the_catalog() -> None:
    world, first_group, _second_group = _world_with_two_groups()
    anchor = world.createGroupAnchor(first_group, "BEA-0001", _node_id(world))

    assert anchor.anchor_id == "AN001"
    assert world.getGroupAnchor("AN001") is anchor
    assert world.getAnchorForGroup(first_group) is anchor

    world.setGroupAnchorBeacon(anchor.anchor_id, "BEA-0002")
    assert anchor.beacon_id == "BEA-0002"

    clone = world.clonePhysicalState()
    cloned_anchor = clone.getGroupAnchor(anchor.anchor_id)
    assert cloned_anchor is not anchor
    assert cloned_anchor.beacon_id == "BEA-0002"
    assert cloned_anchor.node_id == _node_id(world)
    assert clone._beacon_resolver is world._beacon_resolver

    assert world.removeGroupAnchor(anchor.anchor_id) is anchor
    assert world.getAnchorForGroup(first_group) is None


def test_group_anchor_rejects_unknown_entities_and_duplicate_group_anchor() -> None:
    world, first_group, _second_group = _world_with_two_groups()
    with pytest.raises(ValueError, match="groupe inexistant"):
        world.createGroupAnchor("G999", "BEA-0001", _node_id(world))
    with pytest.raises(ValueError, match="balise inexistante"):
        world.createGroupAnchor(first_group, "BEA-9999", _node_id(world))
    with pytest.raises(ValueError, match="nœud d'ancrage inexistant"):
        world.createGroupAnchor(first_group, "BEA-0001", "T99:N0")

    world.createGroupAnchor(first_group, "BEA-0001", _node_id(world))
    with pytest.raises(ValueError, match="déjà ancré"):
        world.createGroupAnchor(first_group, "BEA-0002", _node_id(world))


def test_group_union_refuses_two_anchored_groups_and_preserves_one_anchor() -> None:
    world, first_group, second_group = _world_with_two_groups()
    first_anchor = world.createGroupAnchor(first_group, "BEA-0001", _node_id(world, "T01"))
    second_anchor = world.createGroupAnchor(second_group, "BEA-0002", _node_id(world, "T02"))

    with pytest.raises(ValueError, match="fusion interdite"):
        world.union_groups(first_group, second_group)
    assert world.getGroupAnchor(first_anchor.anchor_id) is first_anchor
    assert world.getGroupAnchor(second_anchor.anchor_id) is second_anchor

    world.removeGroupAnchor(second_anchor.anchor_id)
    merged_group = world.union_groups(first_group, second_group)
    assert world.getAnchorForGroup(merged_group) is first_anchor
    assert first_anchor.group_id == merged_group


def test_topodump_exposes_group_anchors(tmp_path) -> None:
    world, first_group, _second_group = _world_with_two_groups()
    anchor = world.createGroupAnchor(first_group, "BEA-0001", _node_id(world))

    dump = tmp_path / "TopoDump.xml"
    world.export_topo_dump_xml(str(dump))
    xml_anchor = ET.parse(dump).getroot().find("GroupAnchors/GroupAnchor")

    assert xml_anchor is not None
    assert xml_anchor.attrib == {
        "id": anchor.anchor_id,
        "group": first_group,
        "beacon": "BEA-0001",
        "node": _node_id(world),
    }


def test_apply_group_anchor_translates_rigidly_and_blocks_user_transforms() -> None:
    world, first_group, second_group = _world_with_two_groups()
    group_id = world.union_groups(first_group, second_group)
    world.setElementPose("T01", [[1.0, 0.0], [0.0, 1.0]], [2.0, 4.0])
    world.setElementPose("T02", [[1.0, 0.0], [0.0, 1.0]], [-3.0, 8.0])
    node_id = _node_id(world, "T01")
    anchor = world.createGroupAnchor(group_id, "BEA-0001", node_id)
    node_before = world.getConceptNodeWorldXY(node_id, group_id)
    poses_before = {
        element_id: world.getElementPose(element_id)
        for element_id in world.getGroupElementIds(group_id)
    }
    attachments_before = dict(world.attachments)

    world.applyGroupAnchor(anchor.anchor_id)

    assert world.getConceptNodeWorldXY(node_id, group_id) == pytest.approx((10.0, -5.0))
    delta = (10.0 - node_before[0], -5.0 - node_before[1])
    for element_id, (rotation, translation, mirrored) in poses_before.items():
        actual_rotation, actual_translation, actual_mirrored = world.getElementPose(element_id)
        assert actual_rotation == pytest.approx(rotation)
        assert actual_translation == pytest.approx((translation[0] + delta[0], translation[1] + delta[1]))
        assert actual_mirrored is mirrored
    assert world.attachments == attachments_before

    poses_after_first_apply = {
        element_id: world.getElementPose(element_id)
        for element_id in world.getGroupElementIds(group_id)
    }
    world.applyGroupAnchor(anchor.anchor_id)
    for element_id, pose in poses_after_first_apply.items():
        actual = world.getElementPose(element_id)
        assert actual[0] == pytest.approx(pose[0])
        assert actual[1] == pytest.approx(pose[1])
        assert actual[2] is pose[2]

    with pytest.raises(ValueError, match="MOVE interdit"):
        world.move_group(group_id, 1.0, 2.0)
    with pytest.raises(ValueError, match="ROTATE interdit"):
        world.rotate_group(group_id, (0.0, 0.0), 1.0)
    with pytest.raises(ValueError, match="FLIP interdit"):
        world.flip_group(group_id, (0.0, 0.0), (1.0, 0.0))


def test_restored_anchor_is_reapplied_after_runtime_map_change() -> None:
    resolver = _MutableBeaconResolver()
    world = TopologyWorld(beacon_resolver=resolver)
    group_id = world.add_element_as_new_group(_element("T01"))
    node_id = _node_id(world)
    anchor = world.createGroupAnchor(group_id, "BEA-0001", node_id)
    world.applyGroupAnchor(anchor.anchor_id)
    snapshot = world._exportPhysicalSnapshot()

    resolver.set_world("BEA-0001", (10.0001, -5.0001))
    restored = TopologyWorld(beacon_resolver=resolver)
    restored._importPhysicalSnapshot(snapshot)
    restored_anchor = restored.getGroupAnchor(anchor.anchor_id)
    assert restored.getConceptNodeWorldXY(node_id, group_id) != pytest.approx(
        resolver.get_world("BEA-0001")
    )

    viewer = SimpleNamespace(_beacon_world_resolver=resolver)
    viewer._attach_beacon_resolver_to_world = (
        TriangleViewerManual._attach_beacon_resolver_to_world.__get__(viewer)
    )
    TriangleViewerManual._reapply_scenario_group_anchors(
        viewer, SimpleNamespace(topoWorld=restored),
    )

    beacon_world = resolver.get_world("BEA-0001")
    assert restored.getConceptNodeWorldXY(node_id, group_id) == pytest.approx(beacon_world)
    restored.rotate_group(group_id, beacon_world, math.pi / 6.0)
    assert restored.getConceptNodeWorldXY(node_id, group_id) == pytest.approx(beacon_world)
    restored.applyGroupAnchor(restored_anchor.anchor_id)
    assert restored.getConceptNodeWorldXY(node_id, group_id) == pytest.approx(beacon_world)


def test_reapply_scenario_group_anchors_does_not_change_a_free_group() -> None:
    resolver = _MutableBeaconResolver()
    world = TopologyWorld(beacon_resolver=resolver)
    group_id = world.add_element_as_new_group(_element("T01"))
    before = world.getElementPose("T01")
    viewer = SimpleNamespace(_beacon_world_resolver=resolver)
    viewer._attach_beacon_resolver_to_world = (
        TriangleViewerManual._attach_beacon_resolver_to_world.__get__(viewer)
    )

    TriangleViewerManual._reapply_scenario_group_anchors(
        viewer, SimpleNamespace(topoWorld=world),
    )

    rotation, translation, mirrored = world.getElementPose("T01")
    assert rotation == pytest.approx(before[0])
    assert translation == pytest.approx(before[1])
    assert mirrored is before[2]
    assert world.getAnchorForGroup(group_id) is None


def test_apply_group_rigid_transform_rejected_for_anchored_group() -> None:
    world, first_group, _second_group = _world_with_two_groups()
    world.createGroupAnchor(first_group, "BEA-0001", _node_id(world))

    with pytest.raises(ValueError, match="RIGID_TRANSFORM interdit"):
        world.apply_group_rigid_transform(
            first_group,
            [[1.0, 0.0], [0.0, 1.0]],
            [3.0, -2.0],
        )


def test_apply_group_rigid_transform_allowed_after_anchor_removal() -> None:
    world, first_group, _second_group = _world_with_two_groups()
    anchor = world.createGroupAnchor(first_group, "BEA-0001", _node_id(world))
    world.removeGroupAnchor(anchor.anchor_id)

    world.apply_group_rigid_transform(
        first_group,
        [[1.0, 0.0], [0.0, 1.0]],
        [3.0, -2.0],
    )

    _rotation, translation, _mirrored = world.getElementPose("T01")
    assert translation == pytest.approx((3.0, -2.0))


def test_anchored_group_rotation_is_allowed_only_around_its_beacon() -> None:
    world, first_group, _second_group = _world_with_two_groups()
    node_id = _node_id(world)
    anchor = world.createGroupAnchor(first_group, "BEA-0001", node_id)
    world.applyGroupAnchor(anchor.anchor_id)

    world.rotate_group(first_group, (10.0, -5.0), math.pi / 2.0)

    assert world.getAnchorForGroup(first_group) is anchor
    assert world.getConceptNodeWorldXY(node_id, first_group) == pytest.approx(
        (10.0, -5.0)
    )


def test_free_group_rotation_accepts_an_arbitrary_valid_pivot() -> None:
    world, first_group, _second_group = _world_with_two_groups()
    _rotation_before, translation_before, _mirrored_before = world.getElementPose("T01")

    world.rotate_group(first_group, (3.0, -2.0), math.pi / 2.0)

    _rotation_after, translation_after, _mirrored_after = world.getElementPose("T01")
    assert translation_after != pytest.approx(translation_before)


def test_anchored_group_rotation_rejects_wrong_pivot_without_mutating_poses() -> None:
    world, first_group, _second_group = _world_with_two_groups()
    anchor = world.createGroupAnchor(first_group, "BEA-0001", _node_id(world))
    world.applyGroupAnchor(anchor.anchor_id)
    poses_before = {
        element_id: world.getElementPose(element_id)
        for element_id in world.getGroupElementIds(first_group)
    }

    with pytest.raises(ValueError, match="ROTATE interdit"):
        world.rotate_group(first_group, (0.0, 0.0), math.pi / 2.0)

    for element_id, expected_pose in poses_before.items():
        actual_pose = world.getElementPose(element_id)
        assert actual_pose[0] == pytest.approx(expected_pose[0])
        assert actual_pose[1] == pytest.approx(expected_pose[1])
        assert actual_pose[2] is expected_pose[2]


def test_anchored_group_rotation_rejects_preexisting_anchor_node_mismatch() -> None:
    world, first_group, _second_group = _world_with_two_groups()
    anchor = world.createGroupAnchor(first_group, "BEA-0001", _node_id(world))
    poses_before = {
        element_id: world.getElementPose(element_id)
        for element_id in world.getGroupElementIds(first_group)
    }

    with pytest.raises(RuntimeError, match="invariant géométrique rompu"):
        world.rotate_group(first_group, (10.0, -5.0), math.pi / 2.0)

    for element_id, expected_pose in poses_before.items():
        actual_pose = world.getElementPose(element_id)
        assert actual_pose[0] == pytest.approx(expected_pose[0])
        assert actual_pose[1] == pytest.approx(expected_pose[1])
        assert actual_pose[2] is expected_pose[2]


def test_group_anchor_is_removed_when_its_group_is_deleted() -> None:
    world, first_group, _second_group = _world_with_two_groups()
    anchor = world.createGroupAnchor(first_group, "BEA-0001", _node_id(world))

    world.removeElementsAndRebuild(["T01"])

    assert anchor.anchor_id not in world.groupAnchors
