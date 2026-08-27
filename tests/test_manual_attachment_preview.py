import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import src.assembleur_edgechoice as edgechoice
from src.assembleur_core import (
    ResolvedEdgeEdgeAttachment,
    ResolvedVertexEdgeAttachment,
    TopologyAttachmentResolutionError,
    TopologyAttachmentResolver,
    TopologyConstraintGeometryError,
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
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


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


def _world_with_same_side_equal_length_overlap() -> TopologyWorld:
    """Deux triangles égaux qui se recouvrent après la pose VE candidate."""
    world = _world_with_two_elements(same_shape=True)
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
    return world


def _ve_intent(mob_element_id="T01", dest_element_id="T02") -> ManualAttachmentIntent:
    return ManualAttachmentIntent(
        kind="vertex-edge",
        mob_element_id=mob_element_id,
        mob_vertex="O",
        mob_edge="OB",
        dest_element_id=dest_element_id,
        dest_vertex="B",
        dest_edge="OB",
    )


def _pose_signature(world: TopologyWorld, element_id: str):
    rotation, translation, mirrored = world.getElementPose(element_id)
    return (rotation.copy(), translation.copy(), mirrored)


def _assert_pose_equal(before, after) -> None:
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])
    assert before[2] is after[2]


def test_manual_intent_builder_materializes_vertex_edge_without_resolution_fields():
    world = _world_with_two_elements()
    attachment = buildTopologyAttachmentFromManualIntent(
        _ve_intent(), attachment_id="PREVIEW_MANUAL", world=world
    )

    assert attachment == TopologyVertexEdgeAttachment(
        attachment_id="PREVIEW_MANUAL",
        mob_element_id="T01",
        mob_vertex="O",
        creation_mob_edge="OB",
        dest_element_id="T02",
        dest_vertex="B",
        creation_dest_edge="OB",
        mob_orientation="CCW",
        dest_orientation="CW",
    )
    assert not hasattr(attachment, "t")
    assert not hasattr(attachment, "edgeFrom")


def test_vertex_edge_preview_uses_dynamic_short_long_resolution():
    world = _world_with_two_elements(same_shape=True)
    intent = ManualAttachmentIntent(
        kind="vertex-edge",
        mob_element_id="T01",
        mob_vertex="O",
        mob_edge="OB",
        dest_element_id="T02",
        dest_vertex="O",
        dest_edge="LO",
    )

    preview = previewManualAttachment(world, intent)

    assert preview.accepted is True
    resolved = preview.world.getResolvedAttachment(preview.attachment.attachment_id)
    assert resolved.vertex_element_id == "T02"
    assert resolved.edge_element_id == "T01"
    assert resolved.position_from_anchor == 0.5


def test_t3_t4_light_to_light_vertex_edge_preview_is_accepted():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle("T3", (3.0, -4.0)))
    world.add_element_as_new_group(_triangle("T4", (6.0, 8.0)))
    intent = ManualAttachmentIntent(
        kind="vertex-edge",
        mob_element_id="T3",
        mob_vertex="L",
        mob_edge="LO",
        dest_element_id="T4",
        dest_vertex="L",
        dest_edge="LO",
    )

    preview = previewManualAttachment(world, intent)

    assert preview.accepted is True
    assert preview.attachment.mob_orientation == "CW"
    assert preview.attachment.dest_orientation == "CCW"
    assert preview.world.getResolvedAttachment(preview.attachment.attachment_id).position_from_anchor <= 1.0


def test_manual_intent_builder_materializes_edge_edge_without_mapping():
    world = _world_with_two_elements()
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
        intent, attachment_id="PREVIEW_MANUAL", world=world
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
        dest_edge="OB",
    )

    preview = previewManualAttachment(world, intent)

    assert preview.accepted is True
    assert isinstance(
        preview.world.getResolvedAttachment(preview.attachment.attachment_id),
        ResolvedVertexEdgeAttachment,
    )


def test_equal_length_vertex_edge_resolution_remains_vertex_edge():
    world = _world_with_same_side_equal_length_overlap()
    attachment = buildTopologyAttachmentFromManualIntent(
        _ve_intent(), attachment_id="RESOLVE_EQUAL_LENGTH", world=world
    )

    resolved = TopologyAttachmentResolver.resolve(world, attachment)

    assert isinstance(resolved, ResolvedVertexEdgeAttachment)
    assert resolved.mob_effective_edge == "OB"
    assert resolved.dest_effective_edge == "OB"
    assert resolved.vertex_element_id == "T01"
    assert resolved.edge_element_id == "T02"
    assert resolved.position_from_anchor == 1.0


def test_equal_length_vertex_edge_same_side_overlap_is_rejected():
    world = _world_with_same_side_equal_length_overlap()

    preview = previewManualAttachment(world, _ve_intent())

    assert preview.accepted is False
    assert preview.world is None
    assert preview.rejection_reason == "Chevauchement géométrique incompatible."


def test_preview_rejects_same_vertex_edge_orientation_without_mutating_real_world(monkeypatch):
    world = _world_with_two_elements()
    real_attachments = dict(world.attachments)
    real_resolved = dict(world.resolved_attachments)
    real_groups = dict(world.element_to_group)

    original_builder = edgechoice.buildTopologyAttachmentFromManualIntent

    def build_invalid_orientation(*args, **kwargs):
        attachment = original_builder(*args, **kwargs)
        return replace(attachment, dest_orientation=attachment.mob_orientation)

    monkeypatch.setattr(
        edgechoice, "buildTopologyAttachmentFromManualIntent", build_invalid_orientation,
    )

    preview = previewManualAttachment(world, _ve_intent())

    assert preview.accepted is False
    assert preview.world is None
    assert preview.attachment.mob_orientation == preview.attachment.dest_orientation
    assert "orientations must be opposite" in preview.rejection_reason
    assert world.attachments == real_attachments
    assert world.resolved_attachments == real_resolved
    assert world.element_to_group == real_groups


@pytest.mark.parametrize(
    "expected_error",
    [TopologyAttachmentResolutionError, TopologyConstraintGeometryError],
)
def test_preview_rejects_expected_core_errors_without_mutating_real_world(
    monkeypatch, expected_error,
):
    world = _world_with_two_elements()
    real_attachments = dict(world.attachments)
    real_resolved = dict(world.resolved_attachments)

    def reject_candidate(*_args):
        raise expected_error("candidate impossible")

    monkeypatch.setattr(TopologyWorld, "simulate_topological_overlap", reject_candidate)

    preview = previewManualAttachment(world, _ve_intent())

    assert preview.accepted is False
    assert preview.world is None
    assert preview.rejection_reason == "candidate impossible"
    assert world.attachments == real_attachments
    assert world.resolved_attachments == real_resolved


def test_preview_does_not_absorb_unexpected_runtime_errors(monkeypatch):
    world = _world_with_two_elements()

    def broken_preview(*_args):
        raise RuntimeError("unexpected preview bug")

    monkeypatch.setattr(TopologyWorld, "simulate_topological_overlap", broken_preview)

    with pytest.raises(RuntimeError, match="unexpected preview bug"):
        previewManualAttachment(world, _ve_intent())


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
        TopologyVertexEdgeAttachment("A001", "T01", "O", "OB", "T02", "B", "OB", "CCW", "CW")
    )
    real_attachments = dict(world.attachments)

    preview = previewManualAttachment(world, _ve_intent())

    assert preview.accepted is False
    assert preview.world is None
    assert "même groupe" in preview.rejection_reason
    assert world.attachments == real_attachments


def test_preview_rejects_an_incompatible_overlap_without_mutating_real_world():
    world = _world_with_same_side_equal_length_overlap()
    real_poses = {
        element_id: _pose_signature(world, element_id)
        for element_id in ("T01", "T02")
    }
    preview = previewManualAttachment(world, _ve_intent())

    assert preview.accepted is False
    assert preview.world is None
    assert preview.rejection_reason == "Chevauchement géométrique incompatible."
    assert world.attachments == {}
    assert world.resolved_attachments == {}
    for element_id in ("T01", "T02"):
        _assert_pose_equal(real_poses[element_id], _pose_signature(world, element_id))


def test_topological_preview_rejection_is_deterministic_and_leaves_real_world_unchanged():
    world = _world_with_two_elements()
    world.add_element_as_new_group(_triangle("T03", (3.0, 4.0)))
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
    world.apply_attachment(
        TopologyVertexEdgeAttachment("A001", "T01", "O", "OB", "T02", "B", "OB", "CCW", "CW")
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
    real_attachments = dict(world.attachments)
    real_resolved_attachments = dict(world.resolved_attachments)

    first = previewManualAttachment(world, preview_intent)
    second = previewManualAttachment(world, preview_intent)

    assert first.accepted is False
    assert second.accepted is False
    assert first.rejection_reason == "Chevauchement géométrique incompatible."
    assert second.rejection_reason == first.rejection_reason
    assert world.attachments == real_attachments
    assert world.resolved_attachments == real_resolved_attachments
    for element_id in ("T01", "T02", "T03"):
        _assert_pose_equal(
            real_poses[element_id],
            _pose_signature(world, element_id),
        )


def test_tk_drag_projects_an_accepted_preview_world_without_installing_it():
    world = _world_with_two_elements()
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
    preview = previewManualAttachment(world, _ve_intent())
    assert preview.accepted
    real_pose = _pose_signature(world, "T01")

    entries = []
    for element_id in ("T01", "T02"):
        element = world.elements[element_id]
        entries.append({
            "topoElementId": element_id,
            "pts": {
                vertex: element.localToWorld(element.vertex_local_xy[index])
                for index, vertex in enumerate(("O", "B", "L"))
            },
        })
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.canvas_objects = CanvasObjectsCollection(entries)
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.scenarios = [SimpleNamespace(topoWorld=world)]
    viewer.active_scenario_index = 0
    viewer._sel = {
        "mode": "move_group",
        "core_group_id": world.get_group_of_element("T01"),
        "anchor": {"type": "vertex", "tid": 0, "vkey": "O"},
    }
    viewer._invalidate_pick_cache = lambda: None
    viewer._attachment_intent = _ve_intent()
    drag_delta = np.array((4.0, -1.0))
    viewer._last_drawn[0]["pts"] = {
        vertex: np.asarray(point, dtype=float) + drag_delta
        for vertex, point in viewer._last_drawn[0]["pts"].items()
    }
    free_mobile_points = {
        vertex: np.array(point, copy=True)
        for vertex, point in viewer._last_drawn[0]["pts"].items()
    }
    viewer._edge_highlights = {
        "best": ((-1.0, 0.0), (-2.0, 0.0), (20.0, 0.0), (30.0, 0.0)),
        "mob_outline": [((-1.0, 0.0), (-2.0, 0.0))],
        "mob_inc": [((-1.0, 0.0), (-2.0, 0.0))],
        "all": [],
        "tgt_inc": [],
        "tgt_outline": [],
    }

    viewer._preview_attachment_rotation_to_last_drawn(preview)

    np.testing.assert_allclose(viewer._last_drawn[0]["pts"]["O"], free_mobile_points["O"])
    preview_points = viewer._get_core_triangle_world_points(preview.world, "T01")
    mobile_edge = (
        np.asarray(viewer._last_drawn[0]["pts"]["B"])
        - np.asarray(viewer._last_drawn[0]["pts"]["O"])
    )
    preview_edge = np.asarray(preview_points["B"]) - np.asarray(preview_points["O"])
    assert np.isclose(
        mobile_edge[0] * preview_edge[1] - mobile_edge[1] * preview_edge[0], 0.0
    )
    highlighted_mobile_edge = viewer._edge_highlights["best"][:2]
    np.testing.assert_allclose(highlighted_mobile_edge[0], viewer._last_drawn[0]["pts"]["O"])
    np.testing.assert_allclose(highlighted_mobile_edge[1], viewer._last_drawn[0]["pts"]["B"])
    assert viewer._edge_highlights["mob_outline"] == []
    assert viewer._edge_highlights["mob_inc"] == []
    _assert_pose_equal(real_pose, _pose_signature(world, "T01"))


def test_tk_ctrl_preview_renders_the_current_mobile_candidate_edge_in_green():
    class Canvas:
        def __init__(self):
            self.lines = []

        def delete(self, _item):
            pass

        def create_line(self, *coords, **kwargs):
            self.lines.append((coords, kwargs))
            return len(self.lines)

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.canvas = Canvas()
    viewer._world_to_screen = lambda point: point
    viewer._edge_highlight_ids = []
    viewer._attachment_preview = SimpleNamespace(accepted=True)
    viewer._edge_highlights = {
        "all": [],
        "mob_inc": [],
        "tgt_inc": [],
        "mob_outline": [],
        "tgt_outline": [],
        "best": ((4.0, -1.0), (4.0, 9.0), (20.0, 0.0), (30.0, 0.0)),
    }

    viewer._redraw_edge_highlights()

    green_lines = [
        coords for coords, kwargs in viewer.canvas.lines if kwargs["fill"] == "#178C3A"
    ]
    assert (4.0, -1.0, 4.0, 9.0) in green_lines
