import math
from pathlib import Path

import numpy as np

from src.assembleur_core import (
    ResolvedEdgeEdgeAttachment,
    ResolvedVertexEdgeAttachment,
    TopologyEdgeEdgeAttachment,
    TopologyElement,
    TopologyVertexEdgeAttachment,
    TopologyWorld,
)
from src.assembleur_edgechoice import (
    ManualAttachmentIntent,
    buildTopologyAttachmentFromManualIntent,
    previewManualAttachment,
)


def _triangle(element_id: str, light_xy: tuple[float, float]) -> TopologyElement:
    opening = (0.0, 0.0)
    base = (10.0, 0.0)
    return TopologyElement(
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[
            10.0,
            math.dist(base, light_xy),
            math.dist(light_xy, opening),
        ],
        vertex_local_xy={0: opening, 1: base, 2: light_xy},
        element_id=element_id,
    )


def _world_with_two_elements(*, same_shape: bool = False) -> TopologyWorld:
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle("T01", (3.0, 4.0)))
    world.add_element_as_new_group(
        _triangle("T02", (3.0, 4.0) if same_shape else (6.0, 8.0))
    )
    return world


def _ve_intent(mob_element_id="T01", dest_element_id="T02") -> ManualAttachmentIntent:
    return ManualAttachmentIntent(
        kind="vertex-edge",
        mob_element_id=mob_element_id,
        mob_vertex="O",
        mob_edge="LO",
        dest_element_id=dest_element_id,
        dest_vertex="O",
        dest_edge="LO",
    )


def _pose_signature(world: TopologyWorld, element_id: str):
    rotation, translation, mirrored = world.getElementPose(element_id)
    return (rotation.copy(), translation.copy(), mirrored)


def _assert_pose_equal(before, after) -> None:
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])
    assert before[2] is after[2]


def test_manual_intent_builder_materializes_vertex_edge_without_resolution_fields():
    attachment = buildTopologyAttachmentFromManualIntent(
        _ve_intent(), attachment_id="PREVIEW_MANUAL"
    )

    assert attachment == TopologyVertexEdgeAttachment(
        attachment_id="PREVIEW_MANUAL",
        mob_element_id="T01",
        mob_vertex="O",
        mob_edge="LO",
        dest_element_id="T02",
        dest_vertex="O",
        dest_edge="LO",
    )
    assert not hasattr(attachment, "t")
    assert not hasattr(attachment, "edgeFrom")


def test_manual_intent_builder_materializes_edge_edge_without_mapping():
    intent = ManualAttachmentIntent(
        kind="edge-edge",
        mob_element_id="T01",
        mob_vertex="O",
        mob_edge="LO",
        dest_element_id="T02",
        dest_vertex="O",
        dest_edge="LO",
    )
    attachment = buildTopologyAttachmentFromManualIntent(
        intent, attachment_id="PREVIEW_MANUAL"
    )

    assert attachment == TopologyEdgeEdgeAttachment(
        attachment_id="PREVIEW_MANUAL",
        mob_element_id="T01",
        mob_edge="LO",
        dest_element_id="T02",
        dest_edge="LO",
    )
    assert not hasattr(attachment, "mapping")


def test_vertex_edge_preview_is_accepted_and_leaves_the_real_world_unchanged():
    world = _world_with_two_elements()
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
    real_dest_pose = _pose_signature(world, "T02")
    real_mob_pose = _pose_signature(world, "T01")
    real_groups = dict(world.element_to_group)
    real_attachments = dict(world.attachments)
    real_resolved = dict(world.resolved_attachments)

    preview = previewManualAttachment(world, _ve_intent())

    assert preview.accepted is True
    assert preview.rejection_reason is None
    assert preview.world is not None
    assert preview.attachment.attachment_id in preview.world.attachments
    assert isinstance(
        preview.world.getResolvedAttachment(preview.attachment.attachment_id),
        ResolvedVertexEdgeAttachment,
    )
    _assert_pose_equal(real_dest_pose, _pose_signature(preview.world, "T02"))
    assert not np.array_equal(real_mob_pose[0], preview.world.getElementPose("T01")[0])
    assert world.attachments == real_attachments
    assert world.resolved_attachments == real_resolved
    assert world.element_to_group == real_groups
    _assert_pose_equal(real_dest_pose, _pose_signature(world, "T02"))
    _assert_pose_equal(real_mob_pose, _pose_signature(world, "T01"))


def test_vertex_edge_preview_uses_resolver_when_destination_carries_the_vertex_side():
    world = _world_with_two_elements()
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
    intent = ManualAttachmentIntent(
        kind="vertex-edge",
        mob_element_id="T01",
        mob_vertex="O",
        mob_edge="OB",
        dest_element_id="T02",
        dest_vertex="B",
        dest_edge="BL",
    )

    preview = previewManualAttachment(world, intent)

    assert preview.accepted is True
    assert isinstance(
        preview.world.getResolvedAttachment(preview.attachment.attachment_id),
        ResolvedVertexEdgeAttachment,
    )


def test_equal_length_vertex_edge_preview_remains_vertex_edge():
    world = _world_with_two_elements(same_shape=True)
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)

    preview = previewManualAttachment(world, _ve_intent())

    assert preview.accepted is True
    assert isinstance(
        preview.world.getResolvedAttachment(preview.attachment.attachment_id),
        ResolvedVertexEdgeAttachment,
    )


def test_edge_edge_preview_keeps_destination_fixed_and_uses_resolved_edge_edge():
    world = _world_with_two_elements(same_shape=True)
    real_dest_pose = _pose_signature(world, "T02")
    intent = ManualAttachmentIntent(
        kind="edge-edge",
        mob_element_id="T01",
        mob_vertex="O",
        mob_edge="LO",
        dest_element_id="T02",
        dest_vertex="O",
        dest_edge="LO",
    )

    preview = previewManualAttachment(world, intent)

    assert preview.accepted is True
    assert isinstance(
        preview.world.getResolvedAttachment(preview.attachment.attachment_id),
        ResolvedEdgeEdgeAttachment,
    )
    _assert_pose_equal(real_dest_pose, _pose_signature(preview.world, "T02"))


def test_preview_rejects_elements_already_in_the_same_group_without_mutating_real_world():
    world = _world_with_two_elements()
    world.apply_attachment(
        TopologyVertexEdgeAttachment("A001", "T01", "O", "LO", "T02", "O", "LO")
    )
    real_attachments = dict(world.attachments)

    preview = previewManualAttachment(world, _ve_intent())

    assert preview.accepted is False
    assert preview.world is None
    assert "même groupe" in preview.rejection_reason
    assert world.attachments == real_attachments


def test_preview_rejects_an_incompatible_overlap_without_mutating_real_world():
    world = _world_with_two_elements()
    preview = previewManualAttachment(world, _ve_intent())

    assert preview.accepted is False
    assert preview.world is None
    assert preview.rejection_reason == "Chevauchement géométrique incompatible."
    assert world.attachments == {}
    assert world.resolved_attachments == {}


def test_topological_preview_rejection_is_deterministic_and_leaves_real_world_unchanged():
    world = _world_with_two_elements()
    world.add_element_as_new_group(_triangle("T03", (3.0, 4.0)))
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
    world.apply_attachment(
        TopologyVertexEdgeAttachment("A001", "T01", "O", "LO", "T02", "O", "LO")
    )
    world.replay_group_attachment_poses(
        world.get_group_of_element("T01"), "T02"
    )
    world.setElementPose("T02", np.eye(2), np.zeros(2), mirrored=True)
    preview_intent = _ve_intent("T02", "T03")
    real_poses = {
        element_id: _pose_signature(world, element_id)
        for element_id in ("T01", "T02", "T03")
    }

    first = previewManualAttachment(world, preview_intent)
    second = previewManualAttachment(world, preview_intent)

    assert first.accepted is False
    assert second.accepted is False
    assert first.rejection_reason == "Chevauchement géométrique incompatible."
    assert second.rejection_reason == first.rejection_reason
    for element_id in ("T01", "T02", "T03"):
        _assert_pose_equal(
            real_poses[element_id],
            _pose_signature(world, element_id),
        )


def test_tk_drag_preview_never_projects_the_preview_world():
    """Le clone de preview peut différer, mais ne doit pas devenir la vue du drag."""
    viewer_source = Path("src/assembleur_tk.py").read_text(encoding="utf-8")

    assert "_project_attachment_preview_to_last_drawn" not in viewer_source
    assert "self._attachment_preview.world" not in viewer_source
