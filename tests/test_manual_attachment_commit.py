import math

import numpy as np
import src.assembleur_edgechoice as attachment_api

from src.assembleur_core import (
    ResolvedEdgeEdgeAttachment,
    ResolvedVertexEdgeAttachment,
    TopologyElement,
    TopologyVertexEdgeAttachment,
    TopologyWorld,
)
from src.assembleur_edgechoice import (
    ManualAttachmentIntent,
    commitManualAttachment,
    previewManualAttachment,
)


def _triangle(element_id: str, light_xy: tuple[float, float]) -> TopologyElement:
    return TopologyElement(
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[
            10.0,
            math.dist((10.0, 0.0), light_xy),
            math.dist((0.0, 0.0), light_xy),
        ],
        vertex_local_xy={0: (0.0, 0.0), 1: (10.0, 0.0), 2: light_xy},
        element_id=element_id,
    )


def _world(*, same_shape: bool = False) -> TopologyWorld:
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle("T01", (3.0, 4.0)))
    world.add_element_as_new_group(
        _triangle("T02", (3.0, 4.0) if same_shape else (6.0, 8.0))
    )
    return world


def _ve_intent(mob="T01", dest="T02") -> ManualAttachmentIntent:
    return ManualAttachmentIntent("vertex-edge", mob, "O", "OB", dest, "B", "OB")


def _world_with_same_side_equal_length_overlap() -> TopologyWorld:
    world = _world(same_shape=True)
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
    return world


def _pose(world: TopologyWorld, element_id: str):
    rotation, translation, mirrored = world.getElementPose(element_id)
    return rotation.copy(), translation.copy(), mirrored


def _assert_pose_equal(left, right) -> None:
    assert np.array_equal(left[0], right[0])
    assert np.array_equal(left[1], right[1])
    assert left[2] is right[2]


def test_commit_vertex_edge_replays_the_real_world_with_destination_fixed():
    world = _world()
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
    destination_pose = _pose(world, "T02")
    preview = previewManualAttachment(world, _ve_intent())
    assert preview.accepted
    world_identity = id(world)

    attachment, merged_group_id = commitManualAttachment(world, _ve_intent())

    assert id(world) == world_identity
    assert attachment.attachment_id in world.attachments
    assert isinstance(world.getResolvedAttachment(attachment.attachment_id), ResolvedVertexEdgeAttachment)
    assert world.get_group_of_element("T01") == merged_group_id
    assert world.get_group_of_element("T02") == merged_group_id
    _assert_pose_equal(destination_pose, _pose(world, "T02"))
    for element_id in ("T01", "T02"):
        _assert_pose_equal(_pose(preview.world, element_id), _pose(world, element_id))


def test_commit_edge_edge_matches_the_accepted_preview():
    world = _world(same_shape=True)
    intent = ManualAttachmentIntent("edge-edge", "T01", "O", "LO", "T02", "O", "LO")
    preview = previewManualAttachment(world, intent)
    assert preview.accepted

    attachment, merged_group_id = commitManualAttachment(world, intent)

    assert world.get_group_of_element("T01") == merged_group_id
    assert isinstance(world.getResolvedAttachment(attachment.attachment_id), ResolvedEdgeEdgeAttachment)
    for element_id in ("T01", "T02"):
        _assert_pose_equal(_pose(preview.world, element_id), _pose(world, element_id))


def test_topological_preview_rejects_an_invalid_existing_mobile_group_without_commit():
    world = _world()
    world.add_element_as_new_group(_triangle("T03", (3.0, 4.0)))
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
    world.apply_attachment(
        TopologyVertexEdgeAttachment("A001", "T01", "O", "OB", "T02", "B", "OB", "CCW", "CW")
    )
    world.replay_group_attachment_poses(world.get_group_of_element("T01"), "T02")
    world.setElementPose("T02", np.eye(2), np.zeros(2), mirrored=True)
    intent = _ve_intent("T02", "T03")
    real_attachments = dict(world.attachments)
    real_resolved_attachments = dict(world.resolved_attachments)
    preview = previewManualAttachment(world, intent)
    assert preview.accepted is False
    assert preview.rejection_reason == "Chevauchement géométrique incompatible."
    assert world.get_group_of_element("T01") == world.get_group_of_element("T02")
    assert world.get_group_of_element("T02") != world.get_group_of_element("T03")
    assert world.attachments == real_attachments
    assert world.resolved_attachments == real_resolved_attachments


def test_a_refused_preview_is_never_committed_by_the_orchestration_contract(monkeypatch):
    world = _world_with_same_side_equal_length_overlap()
    intent = _ve_intent()
    real_attachments = dict(world.attachments)
    real_resolved_attachments = dict(world.resolved_attachments)
    commit_calls = []
    monkeypatch.setattr(
        attachment_api,
        "commitManualAttachment",
        lambda actual_world, actual_intent: commit_calls.append((actual_world, actual_intent)),
    )
    orchestration_calls = []

    def orchestrate_manual_attachment(actual_world, actual_intent):
        orchestration_calls.append(actual_intent)
        preview = attachment_api.previewManualAttachment(actual_world, actual_intent)
        if preview.accepted:
            attachment_api.commitManualAttachment(actual_world, actual_intent)
        return preview

    refused_preview = orchestrate_manual_attachment(world, intent)

    assert refused_preview.accepted is False
    assert orchestration_calls == [intent]
    assert commit_calls == []
    assert world.attachments == real_attachments
    assert world.resolved_attachments == real_resolved_attachments


def test_commits_are_deterministic_on_independent_identical_worlds():
    left = _world()
    right = _world()
    for world in (left, right):
        world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
        assert previewManualAttachment(world, _ve_intent()).accepted
        commitManualAttachment(world, _ve_intent())

    assert tuple(left.attachments) == tuple(right.attachments)
    for element_id in ("T01", "T02"):
        _assert_pose_equal(_pose(left, element_id), _pose(right, element_id))
