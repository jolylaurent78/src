import math
import inspect

import numpy as np
import pytest

from src.assembleur_core import (
    ResolvedEdgeEdgeAttachment,
    ResolvedVertexEdgeAttachment,
    TopologyAttachmentResolutionError,
    TopologyAttachmentValidationError,
    TopologyConstraintGeometryError,
    TopologyEdgeEdgeAttachment,
    TopologyElement,
    TopologyVertexEdgeAttachment,
    TopologyAttachmentResolver,
    TopologyWorld,
    compute_triangle_intrinsic_orientation,
    compute_vertex_edge_attachment_orientation,
    resolve_vertex_edge_effective_edge,
)


def _triangle(element_id: str, light_xy: tuple[float, float]) -> TopologyElement:
    opening = (0.0, 0.0)
    base = (10.0, 0.0)
    light = light_xy
    return TopologyElement(
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[
            10.0,
            math.dist(base, light),
            math.dist(light, opening),
        ],
        vertex_local_xy={0: opening, 1: base, 2: light},
        element_id=element_id,
    )


def _world_with_two_triangles(
    mob_light: tuple[float, float] = (3.0, 4.0),
    dest_light: tuple[float, float] = (6.0, 8.0),
) -> TopologyWorld:
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle("T01", mob_light))
    world.add_element_as_new_group(_triangle("T02", dest_light))
    return world


def _vertex_edge_attachment(attachment_id: str = "A001") -> TopologyVertexEdgeAttachment:
    return TopologyVertexEdgeAttachment(
        attachment_id=attachment_id,
        mob_element_id="T01",
        mob_vertex="O",
        creation_mob_edge="OB",
        dest_element_id="T02",
        dest_vertex="O",
        creation_dest_edge="LO",
        mob_orientation="CCW",
        dest_orientation="CW",
    )


def _triangle_with_lo_length(element_id: str, length: float) -> TopologyElement:
    return _triangle(element_id, (0.0, length))


def _vertex_edge_from_light_anchor(
    attachment_id: str,
    mob_element_id: str,
    dest_element_id: str,
) -> TopologyVertexEdgeAttachment:
    return TopologyVertexEdgeAttachment(
        attachment_id=attachment_id,
        mob_element_id=mob_element_id,
        mob_vertex="L",
        creation_mob_edge="LO",
        dest_element_id=dest_element_id,
        dest_vertex="O",
        creation_dest_edge="LO",
        mob_orientation="CCW",
        dest_orientation="CW",
    )


def _edge_split_points(world: TopologyWorld, element_id: str, edge_name: str) -> list[dict]:
    edge = world.get_element_edge_by_vertex_types(element_id, edge_name)
    return world.buildDerivedSplitPointsForPhysEdge(element_id, edge.edge_index)


def _coverage_intervals(world: TopologyWorld, element_id: str, edge_name: str) -> list[tuple[float, float]]:
    edge = world.get_element_edge_by_vertex_types(element_id, edge_name)
    return [(coverage.t0, coverage.t1) for coverage in edge.coverages]


def _set_pose(
    world: TopologyWorld,
    element_id: str,
    angle: float,
    translation: tuple[float, float],
    mirrored: bool = False,
) -> None:
    cosine, sine = math.cos(angle), math.sin(angle)
    world.setElementPose(
        element_id,
        R=np.array(((cosine, -sine), (sine, cosine))),
        T=np.array(translation),
        mirrored=mirrored,
    )


def _pose_signature(world: TopologyWorld, element_id: str) -> tuple[np.ndarray, np.ndarray, bool]:
    rotation, translation, mirrored = world.getElementPose(element_id)
    return (rotation.copy(), translation.copy(), mirrored)


def _world_attachment_side(
    world: TopologyWorld,
    element_id: str,
    anchor_vertex: str,
    edge: str,
) -> str:
    endpoints = {"OB": ("O", "B"), "BL": ("B", "L"), "LO": ("L", "O")}[edge]
    other = endpoints[1] if anchor_vertex == endpoints[0] else endpoints[0]
    third = next(vertex for vertex in ("O", "B", "L") if vertex not in endpoints)

    def point(vertex: str) -> np.ndarray:
        return world.elementLocalToWorld(
            element_id, world._local_vertex_point_by_type(element_id, vertex)
        )

    anchor = point(anchor_vertex)
    ray = point(other) - anchor
    third_vector = point(third) - anchor
    cross = float(ray[0] * third_vector[1] - ray[1] * third_vector[0])
    return "CCW" if cross > 0.0 else "CW"


@pytest.mark.parametrize(
    "attachment",
    [
        TopologyVertexEdgeAttachment("", "T01", "O", "LO", "T02", "O", "LO", "CW", "CW"),
        TopologyVertexEdgeAttachment("A001", "T99", "O", "LO", "T02", "O", "LO", "CW", "CW"),
        TopologyVertexEdgeAttachment("A001", "T01", "O", "LO", "T01", "O", "LO", "CW", "CW"),
        TopologyVertexEdgeAttachment("A001", "T01", "O", "XX", "T02", "O", "LO", "CW", "CW"),
    ],
)
def test_v2_attachment_validation_rejects_invalid_structure(attachment):
    world = _world_with_two_triangles()

    with pytest.raises(TopologyAttachmentValidationError):
        world.apply_attachment(attachment)


@pytest.mark.parametrize(
    ("vertex", "edge", "orientation"),
    [
        ("O", "OB", "CCW"), ("B", "OB", "CW"),
        ("B", "BL", "CCW"), ("L", "BL", "CW"),
        ("L", "LO", "CCW"), ("O", "LO", "CW"),
    ],
)
def test_vertex_edge_orientation_uses_anchor_to_edge_and_third_vertex(vertex, edge, orientation):
    world = _world_with_two_triangles()
    assert compute_vertex_edge_attachment_orientation(world, "T01", vertex, edge) == orientation
    world.replace_element_intrinsic_geometry("T01", _triangle("T01", (3.0, -4.0)))
    assert compute_vertex_edge_attachment_orientation(world, "T01", vertex, edge) != orientation


@pytest.mark.parametrize(("vertex", "edge"), [("L", "OB"), ("O", "XX")])
def test_vertex_edge_orientation_rejects_non_incident_or_unknown(vertex, edge):
    with pytest.raises(TopologyAttachmentResolutionError):
        compute_vertex_edge_attachment_orientation(
            _world_with_two_triangles(), "T01", vertex, edge
        )


@pytest.mark.parametrize(
    "attachment",
    [
        TopologyVertexEdgeAttachment("A001", "T01", "O", "OB", "T02", "O", "LO", "NOPE", "CW"),
        TopologyVertexEdgeAttachment("A001", "T01", "O", "OB", "T02", "O", "LO", "NOPE", "CCW"),
    ],
)
def test_vertex_edge_validation_rejects_invalid_or_inconsistent_orientations(attachment):
    with pytest.raises(TopologyAttachmentValidationError):
        _world_with_two_triangles().apply_attachment(attachment)


@pytest.mark.parametrize(
    ("vertex", "intrinsic", "pose", "expected"),
    [
        ("O", "CCW", "CCW", "OB"), ("O", "CCW", "CW", "LO"),
        ("O", "CW", "CW", "OB"), ("O", "CW", "CCW", "LO"),
        ("B", "CCW", "CCW", "BL"), ("B", "CCW", "CW", "OB"),
        ("B", "CW", "CW", "BL"), ("B", "CW", "CCW", "OB"),
        ("L", "CCW", "CCW", "LO"), ("L", "CCW", "CW", "BL"),
        ("L", "CW", "CW", "LO"), ("L", "CW", "CCW", "BL"),
    ],
)
def test_vertex_edge_effective_edge_matrix(vertex, intrinsic, pose, expected):
    assert resolve_vertex_edge_effective_edge(vertex, intrinsic, pose) == expected


def test_triangle_intrinsic_orientation_uses_only_current_local_geometry():
    world = _world_with_two_triangles((3.0, 4.0), (3.0, -4.0))
    assert compute_triangle_intrinsic_orientation(world.elements["T01"]) == "CCW"
    assert compute_triangle_intrinsic_orientation(world.elements["T02"]) == "CW"


def test_vertex_edge_resolution_ignores_creation_edges():
    world = _world_with_two_triangles()
    first = TopologyVertexEdgeAttachment(
        "A001", "T01", "O", "OB", "T02", "B", "BL", "CCW", "CW"
    )
    second = TopologyVertexEdgeAttachment(
        "A001", "T01", "O", "LO", "T02", "B", "OB", "CCW", "CW"
    )

    first_resolved = TopologyAttachmentResolver.resolve(world, first)
    second_resolved = TopologyAttachmentResolver.resolve(world, second)

    assert first_resolved == second_resolved


def test_vertex_edge_runtime_resolver_source_does_not_read_creation_edges():
    source = inspect.getsource(TopologyAttachmentResolver._resolve_vertex_edge)
    assert "creation_mob_edge" not in source
    assert "creation_dest_edge" not in source


def test_vertex_edge_recomputes_short_long_after_an_effective_edge_switch():
    world = _world_with_two_triangles((10.0, 30.0), (0.0, 20.0))
    attachment = TopologyVertexEdgeAttachment(
        "A001", "T01", "B", "BL", "T02", "O", "LO", "CCW", "CW"
    )
    world.apply_attachment(attachment)
    before = world.getResolvedAttachment("A001")

    world.replace_element_intrinsic_geometry("T01", _triangle("T01", (10.0, -30.0)))
    after = world.getResolvedAttachment("A001")

    assert (before.mob_effective_edge, before.dest_effective_edge) == ("BL", "LO")
    assert (before.vertex_element_id, before.edge_element_id) == ("T02", "T01")
    assert before.position_from_anchor == pytest.approx(20.0 / 30.0)
    assert (after.mob_effective_edge, after.dest_effective_edge) == ("OB", "LO")
    assert (after.vertex_element_id, after.edge_element_id) == ("T01", "T02")
    assert after.position_from_anchor == pytest.approx(10.0 / 20.0)


def test_vertex_edge_resolver_uses_mobile_as_vertex_when_mobile_edge_is_shorter():
    world = _world_with_two_triangles((3.0, 4.0), (6.0, 8.0))
    world.apply_attachment(_vertex_edge_attachment())

    resolved = world.getResolvedAttachment("A001")

    assert isinstance(resolved, ResolvedVertexEdgeAttachment)
    assert resolved.vertex_element_id == "T01"
    assert resolved.vertex == "B"
    assert resolved.edge_element_id == "T02"
    assert resolved.edge == "LO"
    assert resolved.edge_anchor_vertex == "O"
    assert resolved.position_from_anchor == pytest.approx(1.0)


def test_vertex_edge_resolver_does_not_switch_when_intrinsic_lengths_cross():
    world = _world_with_two_triangles((6.0, 8.0), (3.0, 4.0))
    world.apply_attachment(_vertex_edge_attachment())

    resolved = world.getResolvedAttachment("A001")

    assert isinstance(resolved, ResolvedVertexEdgeAttachment)
    assert resolved.vertex_element_id == "T02"
    assert resolved.vertex == "L"
    assert resolved.edge_element_id == "T01"
    assert resolved.edge == "OB"
    assert resolved.edge_anchor_vertex == "O"
    assert resolved.position_from_anchor == pytest.approx(0.5)


def test_vertex_edge_roles_follow_current_lengths_after_an_intrinsic_side_flip():
    world = _world_with_two_triangles((0.0, 20.0), (0.0, 30.0))
    attachment = TopologyVertexEdgeAttachment(
        "A001", "T01", "O", "OB", "T02", "O", "LO", "CCW", "CW"
    )
    world.apply_attachment(attachment)
    before = world.getResolvedAttachment("A001")

    world.replace_element_intrinsic_geometry("T02", _triangle("T02", (0.0, -5.0)))
    after = world.getResolvedAttachment("A001")

    assert (before.vertex_element_id, before.vertex, before.edge_element_id, before.edge) == (
        "T01", "B", "T02", "LO"
    )
    assert (after.vertex_element_id, after.vertex, after.edge_element_id, after.edge) == (
        "T01", "B", "T02", "OB"
    )
    assert after.dest_effective_edge == "OB"
    assert before.position_from_anchor == pytest.approx(10.0 / 30.0)
    assert after.position_from_anchor == pytest.approx(1.0)
    assert attachment.dest_orientation == "CW"

    world.replay_group_attachment_poses(
        world.get_group_of_element("T02"), root_element_id="T02"
    )

    assert world._resolved_attachment_geometry_is_satisfied(after, tolerance=1e-8)


def test_vertex_edge_replay_accepts_an_intrinsic_side_flip_without_rewriting_intent():
    world = _world_with_two_triangles((0.0, 5.0), (0.0, 10.0))
    attachment = TopologyVertexEdgeAttachment(
        "A001", "T01", "O", "OB", "T02", "O", "LO", "CCW", "CW"
    )
    world.apply_attachment(attachment)
    before = world.getResolvedAttachment("A001")

    world.replace_element_intrinsic_geometry("T01", _triangle("T01", (0.0, -5.0)))
    after = world.getResolvedAttachment("A001")

    assert attachment.mob_orientation == "CCW"
    assert attachment.dest_orientation == "CW"
    assert before.mob_effective_edge == "OB"
    assert after.mob_effective_edge == "LO"
    assert (after.vertex_element_id, after.vertex, after.edge_element_id, after.edge) == (
        "T01", "L", "T02", "LO"
    )
    deformed_pose = _pose_signature(world, "T01")
    world.replay_group_attachment_poses(
        world.get_group_of_element("T01"), root_element_id="T01"
    )

    assert world._resolved_attachment_geometry_is_satisfied(after, tolerance=1e-8)
    replayed_deformed_pose = _pose_signature(world, "T01")
    assert np.array_equal(replayed_deformed_pose[0], deformed_pose[0])
    assert np.array_equal(replayed_deformed_pose[1], deformed_pose[1])
    assert replayed_deformed_pose[2] is deformed_pose[2]
    assert _world_attachment_side(world, "T02", "O", "LO") == "CW"
    assert np.linalg.det(world.getElementPose("T02")[0]) == pytest.approx(1.0)


def test_vertex_edge_replay_keeps_a_ccw_neighbour_around_a_deformed_target():
    world = _world_with_two_triangles((3.0, -4.0), (3.0, 4.0))
    attachment = TopologyVertexEdgeAttachment(
        "A001", "T01", "O", "OB", "T02", "O", "OB", "CW", "CCW"
    )
    world.apply_attachment(attachment)
    world.replace_element_intrinsic_geometry("T01", _triangle("T01", (3.0, 4.0)))
    deformed_pose = _pose_signature(world, "T01")

    world.replay_group_attachment_poses(
        world.get_group_of_element("T01"), root_element_id="T01"
    )

    resolved = world.getResolvedAttachment("A001")
    replayed_deformed_pose = _pose_signature(world, "T01")
    assert world._resolved_attachment_geometry_is_satisfied(resolved, tolerance=1e-8)
    assert np.array_equal(replayed_deformed_pose[0], deformed_pose[0])
    assert np.array_equal(replayed_deformed_pose[1], deformed_pose[1])
    assert _world_attachment_side(world, "T02", "O", "OB") == "CCW"
    assert np.linalg.det(world.getElementPose("T02")[0]) == pytest.approx(1.0)


def test_vertex_edge_chain_replays_after_the_middle_triangle_crosses_its_axes():
    world = _world_with_two_triangles((3.0, 4.0), (3.0, 4.0))
    world.add_element_as_new_group(_triangle("T03", (0.0, 10.0)))
    attachments = (
        TopologyVertexEdgeAttachment(
            "A001", "T01", "L", "LO", "T02", "O", "LO",
            compute_vertex_edge_attachment_orientation(world, "T01", "L", "LO"),
            compute_vertex_edge_attachment_orientation(world, "T02", "O", "LO"),
        ),
        TopologyVertexEdgeAttachment(
            "A002", "T02", "L", "LO", "T03", "O", "LO",
            compute_vertex_edge_attachment_orientation(world, "T02", "L", "LO"),
            compute_vertex_edge_attachment_orientation(world, "T03", "O", "LO"),
        ),
    )
    group_id = world.apply_attachments(attachments)

    world.replace_element_intrinsic_geometry("T02", _triangle("T02", (3.0, -4.0)))
    world.replay_group_attachment_poses(group_id, root_element_id="T02")

    assert tuple(world.attachments.values()) == attachments
    assert all(
        world._resolved_attachment_geometry_is_satisfied(
            world.getResolvedAttachment(attachment.attachment_id), tolerance=1e-8
        )
        for attachment in attachments
    )


def test_edge_edge_then_vertex_edge_replays_with_current_effective_edge():
    """Regression: T01 -- EE -- T02 -- VE -- T03 after T02 flips intrinsically."""
    world = TopologyWorld()
    for element_id in ("T01", "T02", "T03"):
        world.add_element_as_new_group(_triangle(element_id, (3.0, 4.0)))
    ee = TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "OB")
    ve = TopologyVertexEdgeAttachment(
        "A002", "T02", "O", "OB", "T03", "B", "OB", "CCW", "CW"
    )
    group_id = world.apply_attachments([ee, ve])
    before = world.getResolvedAttachment("A002")

    world.replace_element_intrinsic_geometry("T02", _triangle("T02", (3.0, -4.0)))
    after = world.getResolvedAttachment("A002")
    world.replay_group_attachment_poses(group_id, root_element_id="T01")

    assert isinstance(before, ResolvedVertexEdgeAttachment)
    assert isinstance(after, ResolvedVertexEdgeAttachment)
    assert before.mob_effective_edge == "OB"
    assert after.mob_effective_edge == "LO"
    assert after.edge == "OB"
    assert world._resolved_attachment_geometry_is_satisfied(
        world.getResolvedAttachment("A001"), tolerance=1e-8
    )
    assert world._resolved_attachment_geometry_is_satisfied(after, tolerance=1e-8)
    assert world.is_group_contour_valid(group_id)


def test_vertex_edge_five_triangle_chain_replays_outward_from_the_deformed_middle():
    world = TopologyWorld()
    for index in range(1, 6):
        world.add_element_as_new_group(_triangle(f"T{index:02d}", (3.0, 4.0)))
    attachments = [
        TopologyVertexEdgeAttachment(
            f"A{index:03d}", f"T{index:02d}", "O", "OB",
            f"T{index + 1:02d}", "B", "OB", "CCW", "CW",
        )
        for index in range(1, 5)
    ]
    group_id = world.apply_attachments(attachments)
    world.replace_element_intrinsic_geometry("T04", _triangle("T04", (3.0, -4.0)))
    deformed_pose = _pose_signature(world, "T04")

    world.replay_group_attachment_poses(group_id, root_element_id="T04")

    replayed_pose = _pose_signature(world, "T04")
    assert np.array_equal(replayed_pose[0], deformed_pose[0])
    assert np.array_equal(replayed_pose[1], deformed_pose[1])
    assert all(
        world._resolved_attachment_geometry_is_satisfied(
            world.getResolvedAttachment(attachment.attachment_id), tolerance=1e-8
        )
        for attachment in attachments
    )
    assert all(world.getElementPose(f"T{index:02d}")[2] is False for index in range(1, 6))


def test_vertex_edge_equal_lengths_remains_vertex_edge():
    world = _world_with_two_triangles((3.0, 4.0), (3.0, 4.0))
    world.apply_attachment(_vertex_edge_attachment())

    resolved = world.getResolvedAttachment("A001")

    assert isinstance(resolved, ResolvedVertexEdgeAttachment)
    assert resolved.position_from_anchor == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("dest_light", "expected_pairs"),
    [
        ((3.0, 4.0), (("O", "B"), ("B", "O"))),
        ((3.0, -4.0), (("O", "O"), ("B", "B"))),
    ],
)
def test_edge_edge_resolver_exposes_endpoint_pairs(dest_light, expected_pairs):
    world = _world_with_two_triangles((3.0, 4.0), dest_light)
    world.apply_attachment(TopologyEdgeEdgeAttachment(
        attachment_id="A001",
        mob_element_id="T01",
        mob_edge="OB",
        dest_element_id="T02",
        dest_edge="OB",
    ))

    resolved = world.getResolvedAttachment("A001")

    assert isinstance(resolved, ResolvedEdgeEdgeAttachment)
    assert (
        (resolved.mob_vertex_1, resolved.dest_vertex_1),
        (resolved.mob_vertex_2, resolved.dest_vertex_2),
    ) == expected_pairs
    assert not hasattr(resolved, "mapping")


def test_resolved_attachment_cache_is_lazy_and_invalidated_only_for_incident_element():
    world = _world_with_two_triangles()
    world.add_element_as_new_group(_triangle("T03", (4.0, 7.0)))
    world.apply_attachment(_vertex_edge_attachment("A001"))
    world.apply_attachment(_vertex_edge_from_light_anchor("A002", "T02", "T03"))

    first = world.getResolvedAttachment("A001")
    assert world.getResolvedAttachment("A001") is first
    second = world.getResolvedAttachment("A002")
    world.setElementPose("T01", *world.getElementPose("T01")[:2])
    assert world.getResolvedAttachment("A001") is first

    world.replace_element_intrinsic_geometry("T01", _triangle("T01", (4.0, 6.0)))

    assert "A001" not in world.resolved_attachments
    assert world.resolved_attachments["A002"] is second


def test_resolution_error_for_missing_intrinsic_coordinates_is_explicit():
    world = _world_with_two_triangles()
    del world.elements["T01"].vertex_local_xy[1]

    with pytest.raises(TopologyAttachmentResolutionError, match="coordonnées locales absentes"):
        world.apply_attachment(_vertex_edge_attachment())


def test_v2_vertex_edge_application_registers_resolution_and_anchor_union():
    world = _world_with_two_triangles()
    attachment = _vertex_edge_attachment()

    world.apply_attachment(attachment)

    assert world.attachments == {"A001": attachment}
    assert isinstance(world.getResolvedAttachment("A001"), ResolvedVertexEdgeAttachment)
    assert world.get_group_of_element("T01") == world.get_group_of_element("T02")
    assert world.find_node(world.get_element_vertex_node_id_by_type("T01", "O")) == (
        world.find_node(world.get_element_vertex_node_id_by_type("T02", "O"))
    )


def test_v2_vertex_edge_equal_lengths_unions_only_the_anchor():
    world = _world_with_two_triangles((3.0, 4.0), (3.0, 4.0))

    world.apply_attachment(_vertex_edge_attachment())

    resolved = world.getResolvedAttachment("A001")
    assert isinstance(resolved, ResolvedVertexEdgeAttachment)
    assert resolved.position_from_anchor == pytest.approx(0.5)
    assert world.find_node(world.get_element_vertex_node_id_by_type("T01", "O")) == (
        world.find_node(world.get_element_vertex_node_id_by_type("T02", "O"))
    )
    assert world.find_node(world.get_element_vertex_node_id_by_type("T01", "L")) != (
        world.find_node(world.get_element_vertex_node_id_by_type("T02", "L"))
    )


def test_v2_edge_edge_application_uses_resolved_endpoint_pairs():
    world = _world_with_two_triangles()
    attachment = TopologyEdgeEdgeAttachment(
        attachment_id="A001",
        mob_element_id="T01",
        mob_edge="OB",
        dest_element_id="T02",
        dest_edge="OB",
    )

    world.apply_attachment(attachment)

    resolved = world.getResolvedAttachment("A001")
    assert isinstance(resolved, ResolvedEdgeEdgeAttachment)
    assert world.get_group_of_element("T01") == world.get_group_of_element("T02")
    for mob_vertex, dest_vertex in (
        (resolved.mob_vertex_1, resolved.dest_vertex_1),
        (resolved.mob_vertex_2, resolved.dest_vertex_2),
    ):
        assert world.find_node(
            world.get_element_vertex_node_id_by_type("T01", mob_vertex)
        ) == world.find_node(
            world.get_element_vertex_node_id_by_type("T02", dest_vertex)
        )


def test_v2_attachment_chain_builds_one_group_of_three_elements():
    world = _world_with_two_triangles()
    world.add_element_as_new_group(_triangle("T03", (4.0, 7.0)))
    attachments = [
            _vertex_edge_attachment("A001"),
        TopologyVertexEdgeAttachment(
            "A002", "T02", "L", "LO", "T03", "O", "LO", "CCW", "CW"
            ),
    ]

    world.apply_attachments(attachments)

    group_id = world.get_group_of_element("T01")
    assert group_id == world.get_group_of_element("T02")
    assert group_id == world.get_group_of_element("T03")
    assert sorted(world.groups[group_id].element_ids) == ["T01", "T02", "T03"]


def test_v2_rebuild_is_deterministic_without_attachment_duplication():
    world = _world_with_two_triangles()
    world.add_element_as_new_group(_triangle("T03", (4.0, 7.0)))
    world.apply_attachments([
        _vertex_edge_attachment("A001"),
        _vertex_edge_from_light_anchor("A002", "T02", "T03"),
    ])

    def topology_state() -> tuple[list[str], list[str], list[str]]:
        group_id = world.get_group_of_element("T01")
        return (
            sorted(world.attachments),
            sorted(world.groups[group_id].attachment_ids),
            sorted(world.groups[group_id].element_ids),
        )

    expected = topology_state()
    world.rebuild_from_attachments()
    assert topology_state() == expected
    world.rebuild_from_attachments()
    assert topology_state() == expected


def test_v2_rebuild_preserves_resolved_cache_without_intrinsic_change():
    world = _world_with_two_triangles()
    world.apply_attachment(_vertex_edge_attachment())
    resolved = world.getResolvedAttachment("A001")

    world.rebuild_from_attachments()

    assert world.getResolvedAttachment("A001") is resolved


def test_v2_intrinsic_invalidation_is_selective_across_rebuild():
    world = _world_with_two_triangles()
    world.add_element_as_new_group(_triangle("T03", (4.0, 7.0)))
    world.apply_attachments([
        _vertex_edge_attachment("A001"),
        _vertex_edge_from_light_anchor("A002", "T02", "T03"),
    ])
    first = world.getResolvedAttachment("A001")
    second = world.getResolvedAttachment("A002")

    world.replace_element_intrinsic_geometry("T01", _triangle("T01", (4.0, 6.0)))

    assert "A001" not in world.resolved_attachments
    assert world.resolved_attachments["A002"] is second
    world.rebuild_from_attachments()
    assert world.getResolvedAttachment("A001") is not first
    assert world.getResolvedAttachment("A002") is second
    assert world.get_group_of_element("T01") == world.get_group_of_element("T03")


def test_v2_split_point_uses_resolved_mobile_shorter_vertex_edge():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle_with_lo_length("T01", 60.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T02", 100.0))
    world.apply_attachment(_vertex_edge_from_light_anchor("A001", "T01", "T02"))

    split_points = _edge_split_points(world, "T02", "LO")

    assert [point["t"] for point in split_points] == pytest.approx([0.0, 0.4, 1.0])
    assert split_points[1] == {
        "t": pytest.approx(0.4),
        "nodeCanon": world.find_node(
            world.get_element_vertex_node_id_by_type("T01", "O")
        ),
        "source": "vertex-edge",
        "attachmentId": "A001",
    }
    world.assertNoPhysicalSplitPoints()


def test_v2_split_point_uses_resolved_destination_shorter_vertex_edge():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle_with_lo_length("T01", 100.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T02", 60.0))
    world.apply_attachment(_vertex_edge_from_light_anchor("A001", "T01", "T02"))

    split_points = _edge_split_points(world, "T01", "LO")

    assert [point["t"] for point in split_points] == pytest.approx([0.0, 0.6, 1.0])
    assert split_points[1]["nodeCanon"] == world.find_node(
        world.get_element_vertex_node_id_by_type("T02", "L")
    )


def test_v2_split_point_converts_anchor_at_physical_edge_end():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle_with_lo_length("T01", 60.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T02", 100.0))
    world.apply_attachment(_vertex_edge_attachment())

    split_points = _edge_split_points(world, "T02", "LO")

    assert [point["t"] for point in split_points] == pytest.approx([0.0, 0.9, 1.0])


def test_v2_equal_vertex_edge_has_no_interior_split_or_endpoint_union():
    world = _world_with_two_triangles((0.0, 60.0), (0.0, 60.0))
    world.apply_attachment(_vertex_edge_attachment())

    assert [point["t"] for point in _edge_split_points(world, "T02", "LO")] == pytest.approx([0.0, 5.0 / 6.0, 1.0])
    assert world.find_node(world.get_element_vertex_node_id_by_type("T01", "L")) != (
        world.find_node(world.get_element_vertex_node_id_by_type("T02", "L"))
    )


def test_v2_edge_edge_application_derives_only_its_two_coverages():
    world = _world_with_two_triangles()
    world.apply_attachment(TopologyEdgeEdgeAttachment(
        attachment_id="A001",
        mob_element_id="T01",
        mob_edge="OB",
        dest_element_id="T02",
        dest_edge="OB",
    ))

    assert _coverage_intervals(world, "T01", "OB") == [(0.0, 1.0)]
    assert _coverage_intervals(world, "T02", "OB") == [(0.0, 1.0)]
    assert _coverage_intervals(world, "T01", "LO") == []
    assert _coverage_intervals(world, "T02", "LO") == []
    assert [point["t"] for point in _edge_split_points(world, "T01", "OB")] == [0.0, 1.0]


def test_v2_coverage_rebuild_is_deterministic_and_vertex_edge_adds_none():
    world = _world_with_two_triangles()
    world.apply_attachments([
        TopologyEdgeEdgeAttachment(
            attachment_id="A001",
            mob_element_id="T01",
            mob_edge="OB",
            dest_element_id="T02",
            dest_edge="OB",
        ),
        _vertex_edge_attachment("A002"),
    ])

    for _ in range(3):
        world.rebuild_from_attachments()
        assert _coverage_intervals(world, "T01", "OB") == [(0.0, 1.0)]
        assert _coverage_intervals(world, "T02", "OB") == [(0.0, 1.0)]
        assert _coverage_intervals(world, "T01", "LO") == []
        assert _coverage_intervals(world, "T02", "LO") == []


def test_v2_multiple_edge_edge_coverages_stay_on_their_resolved_edges():
    world = _world_with_two_triangles()
    world.add_element_as_new_group(_triangle("T03", (4.0, 7.0)))
    world.apply_attachments([
        TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "OB"),
        TopologyEdgeEdgeAttachment("A002", "T02", "BL", "T03", "BL"),
    ])

    assert _coverage_intervals(world, "T01", "OB") == [(0.0, 1.0)]
    assert _coverage_intervals(world, "T02", "OB") == [(0.0, 1.0)]
    assert _coverage_intervals(world, "T02", "BL") == [(0.0, 1.0)]
    assert _coverage_intervals(world, "T03", "BL") == [(0.0, 1.0)]
    assert _coverage_intervals(world, "T01", "LO") == []
    assert _coverage_intervals(world, "T03", "LO") == []


def test_v2_conflicting_resolved_vertex_edge_splits_raise_explicitly():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle_with_lo_length("T01", 60.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T02", 60.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T03", 100.0))
    world.apply_attachments([
        _vertex_edge_from_light_anchor("A001", "T01", "T03"),
        _vertex_edge_from_light_anchor("A002", "T02", "T03"),
    ])

    with pytest.raises(ValueError, match="conflicting splitpoints"):
        _edge_split_points(world, "T03", "LO")


def test_v2_concept_graph_builds_from_resolved_vertex_edge_splits():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle_with_lo_length("T01", 60.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T02", 100.0))
    world.apply_attachment(_vertex_edge_from_light_anchor("A001", "T01", "T02"))

    cache = world.ensureConceptGraph(world.get_group_of_element("T01"))

    assert cache.graphValid
    world.assertNoPhysicalSplitPoints()


def test_v2_replay_edge_edge_preserves_root_and_satisfies_pairs():
    world = _world_with_two_triangles((3.0, 4.0), (3.0, -4.0))
    world.apply_attachment(TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "OB"))
    _set_pose(world, "T02", 0.43, (24.0, -11.0))
    _set_pose(world, "T01", -0.75, (81.0, 42.0))
    root_pose = _pose_signature(world, "T02")

    world.replay_group_attachment_poses(
        world.get_group_of_element("T02"),
        root_element_id="T02",
    )

    replayed_root = _pose_signature(world, "T02")
    assert np.allclose(replayed_root[0], root_pose[0])
    assert np.allclose(replayed_root[1], root_pose[1])
    assert replayed_root[2] is root_pose[2]
    assert world._resolved_attachment_geometry_is_satisfied(
        world.getResolvedAttachment("A001"),
        1e-8,
    )


def test_v2_replay_edge_edge_uses_inverse_resolved_endpoint_pairs():
    world = _world_with_two_triangles((3.0, 4.0), (3.0, 4.0))
    world.apply_attachment(TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "OB"))
    _set_pose(world, "T02", -0.31, (13.0, 27.0))
    _set_pose(world, "T01", 0.82, (-50.0, 18.0))

    resolved = world.getResolvedAttachment("A001")
    assert isinstance(resolved, ResolvedEdgeEdgeAttachment)
    assert (resolved.mob_vertex_1, resolved.dest_vertex_1) == ("O", "B")
    assert (resolved.mob_vertex_2, resolved.dest_vertex_2) == ("B", "O")
    world.replay_group_attachment_poses(
        world.get_group_of_element("T02"),
        root_element_id="T02",
    )

    assert world._resolved_attachment_geometry_is_satisfied(resolved, 1e-8)


def test_v2_replay_vertex_edge_mobile_shorter():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle_with_lo_length("T01", 60.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T02", 100.0))
    world.apply_attachment(_vertex_edge_from_light_anchor("A001", "T01", "T02"))
    _set_pose(world, "T02", 0.58, (7.0, -15.0))
    _set_pose(world, "T01", -0.62, (74.0, 91.0))

    world.replay_group_attachment_poses(
        world.get_group_of_element("T02"),
        root_element_id="T02",
    )

    assert world._resolved_attachment_geometry_is_satisfied(
        world.getResolvedAttachment("A001"),
        1e-8,
    )


def test_v2_replay_vertex_edge_destination_shorter_places_edge_side_mobile():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle_with_lo_length("T01", 100.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T02", 60.0))
    world.apply_attachment(_vertex_edge_from_light_anchor("A001", "T01", "T02"))
    _set_pose(world, "T02", 0.58, (7.0, -15.0))
    _set_pose(world, "T01", -0.62, (74.0, 91.0))

    resolved = world.getResolvedAttachment("A001")
    assert isinstance(resolved, ResolvedVertexEdgeAttachment)
    assert resolved.vertex_element_id == "T02"
    world.replay_group_attachment_poses(
        world.get_group_of_element("T02"),
        root_element_id="T02",
    )

    assert world._resolved_attachment_geometry_is_satisfied(resolved, 1e-8)


def test_v2_replay_equal_vertex_edge_remains_vertex_edge():
    world = _world_with_two_triangles((0.0, 60.0), (0.0, 60.0))
    world.apply_attachment(_vertex_edge_attachment())
    _set_pose(world, "T02", 0.22, (35.0, 19.0))
    _set_pose(world, "T01", -0.47, (-12.0, 53.0))

    resolved = world.getResolvedAttachment("A001")
    assert isinstance(resolved, ResolvedVertexEdgeAttachment)
    assert resolved.position_from_anchor == pytest.approx(1.0 / 6.0)
    world.replay_group_attachment_poses(
        world.get_group_of_element("T02"),
        root_element_id="T02",
    )

    assert world._resolved_attachment_geometry_is_satisfied(resolved, 1e-8)


def test_v2_replay_preserves_mobile_mirroring():
    world = _world_with_two_triangles((3.0, 4.0), (3.0, -4.0))
    world.apply_attachment(TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "OB"))
    _set_pose(world, "T02", -0.34, (17.0, -8.0))
    _set_pose(world, "T01", 0.95, (32.0, 71.0), mirrored=True)

    world.replay_group_attachment_poses(
        world.get_group_of_element("T02"),
        root_element_id="T02",
    )

    assert world.getElementPose("T01")[2] is True
    assert world._resolved_attachment_geometry_is_satisfied(
        world.getResolvedAttachment("A001"),
        1e-8,
    )


def test_v2_replay_chain_is_deterministic_and_validates_all_constraints():
    world = _world_with_two_triangles((3.0, 4.0), (3.0, -4.0))
    world.add_element_as_new_group(_triangle("T03", (4.0, 7.0)))
    world.apply_attachments([
        TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "OB"),
            TopologyVertexEdgeAttachment(
                "A002", "T02", "L", "LO", "T03", "O", "LO", "CW", "CCW"
        ),
    ])
    _set_pose(world, "T01", 0.29, (10.0, 5.0))
    _set_pose(world, "T02", -0.8, (61.0, 77.0))
    _set_pose(world, "T03", 1.1, (-37.0, 49.0))
    root_pose = _pose_signature(world, "T01")

    group_id = world.get_group_of_element("T01")
    world.replay_group_attachment_poses(group_id, root_element_id="T01")
    first_replay = {
        element_id: _pose_signature(world, element_id)
        for element_id in ("T01", "T02", "T03")
    }
    world.replay_group_attachment_poses(group_id, root_element_id="T01")

    assert np.allclose(_pose_signature(world, "T01")[0], root_pose[0])
    assert np.allclose(_pose_signature(world, "T01")[1], root_pose[1])
    for element_id, pose in first_replay.items():
        current = _pose_signature(world, element_id)
        assert np.allclose(current[0], pose[0])
        assert np.allclose(current[1], pose[1])
        assert current[2] is pose[2]
    for attachment_id in ("A001", "A002"):
        assert world._resolved_attachment_geometry_is_satisfied(
            world.getResolvedAttachment(attachment_id),
            1e-8,
        )


def test_v2_replay_rejects_a_group_with_an_unreachable_element():
    world = _world_with_two_triangles()
    world.add_element_as_new_group(_triangle("T03", (4.0, 7.0)))
    world.apply_attachment(TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "OB"))
    group_id = world.get_group_of_element("T01")
    world.element_to_group["T03"] = group_id
    world.groups[group_id].element_ids.append("T03")

    with pytest.raises(TopologyConstraintGeometryError, match="ne peut pas être rejoué"):
        world.replay_group_attachment_poses(group_id, root_element_id="T01")


def test_v2_resolved_geometry_validation_detects_a_perturbed_pose():
    world = _world_with_two_triangles((3.0, 4.0), (3.0, -4.0))
    world.apply_attachment(TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "OB"))
    world.replay_group_attachment_poses(
        world.get_group_of_element("T02"),
        root_element_id="T02",
    )
    resolved = world.getResolvedAttachment("A001")
    assert world._resolved_attachment_geometry_is_satisfied(resolved, 1e-8)

    _set_pose(world, "T01", 0.0, (100.0, 100.0))

    assert not world._resolved_attachment_geometry_is_satisfied(resolved, 1e-8)


def _assert_no_orphaned_resolved_attachments(world: TopologyWorld) -> None:
    assert set(world.resolved_attachments).issubset(world.attachments)


def test_v2_degrouper_breaks_one_atomic_vertex_edge_attachment():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle_with_lo_length("T01", 60.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T02", 100.0))
    attachment = _vertex_edge_from_light_anchor("A001", "T01", "T02")
    world.apply_attachment(attachment)
    group_id = world.get_group_of_element("T01")
    anchor_node_id = world.get_element_vertex_node_id_by_type("T01", "L")

    assert set(world.attachments) == {"A001"}
    assert world.canDegrouperAtNode(group_id, anchor_node_id)
    world.degrouperAtNode(group_id, anchor_node_id)

    assert world.attachments == {}
    assert world.resolved_attachments == {}
    assert world.get_group_of_element("T01") != world.get_group_of_element("T02")


def test_v2_degrouper_vertex_edge_uses_no_legacy_sibling():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle_with_lo_length("T01", 60.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T02", 100.0))
    world.apply_attachment(_vertex_edge_from_light_anchor("A001", "T01", "T02"))

    world.degrouperAtNode(
        world.get_group_of_element("T01"),
        world.get_element_vertex_node_id_by_type("T01", "L"),
    )

    assert "A001" not in world.attachments
    assert not world.attachments
    _assert_no_orphaned_resolved_attachments(world)


def test_v2_degrouper_breaks_vertex_edge_when_the_second_element_is_vertex_side():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle_with_lo_length("T01", 100.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T02", 60.0))
    world.apply_attachment(_vertex_edge_from_light_anchor("A001", "T02", "T01"))

    world.degrouperAtNode(
        world.get_group_of_element("T02"),
        world.get_element_vertex_node_id_by_type("T02", "L"),
    )

    assert not world.attachments
    assert world.get_group_of_element("T01") != world.get_group_of_element("T02")
    _assert_no_orphaned_resolved_attachments(world)


@pytest.mark.parametrize(
    "dest_light",
    [(3.0, -4.0), (3.0, 4.0)],
)
def test_v2_degrouper_breaks_edge_edge_regardless_of_resolved_orientation(dest_light):
    world = _world_with_two_triangles((3.0, 4.0), dest_light)
    world.apply_attachment(TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "OB"))

    world.degrouperAtNode(
        world.get_group_of_element("T01"),
        world.get_element_vertex_node_id_by_type("T01", "O"),
    )

    assert not world.attachments
    assert world.get_group_of_element("T01") != world.get_group_of_element("T02")
    _assert_no_orphaned_resolved_attachments(world)


def test_v2_degrouper_chain_removes_only_the_requested_connection():
    world = _world_with_two_triangles()
    world.add_element_as_new_group(_triangle("T03", (4.0, 7.0)))
    world.apply_attachments([
        _vertex_edge_attachment("A001"),
        _vertex_edge_from_light_anchor("A002", "T02", "T03"),
    ])
    world.replay_group_attachment_poses(
        world.get_group_of_element("T01"),
        root_element_id="T01",
    )

    world.degrouperAtNode(
        world.get_group_of_element("T02"),
        world.get_element_vertex_node_id_by_type("T02", "L"),
    )

    assert set(world.attachments) == {"A001"}
    assert world.get_group_of_element("T01") == world.get_group_of_element("T02")
    assert world.get_group_of_element("T01") != world.get_group_of_element("T03")
    _assert_no_orphaned_resolved_attachments(world)


def test_v2_remove_elements_purges_incident_attachments_and_resolved_cache():
    world = _world_with_two_triangles()
    world.add_element_as_new_group(_triangle("T03", (4.0, 7.0)))
    world.apply_attachments([
        TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "OB"),
        _vertex_edge_from_light_anchor("A002", "T02", "T03"),
    ])
    assert set(world.resolved_attachments) == {"A001", "A002"}

    world.removeElementsAndRebuild(["T02"])

    assert "T02" not in world.elements
    assert not world.attachments
    assert not world.resolved_attachments
    assert world.get_group_of_element("T01") != world.get_group_of_element("T03")


def test_v2_degrouper_rejects_an_unrelated_node_without_mutating_attachments():
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle_with_lo_length("T01", 60.0))
    world.add_element_as_new_group(_triangle_with_lo_length("T02", 100.0))
    world.apply_attachment(_vertex_edge_from_light_anchor("A001", "T01", "T02"))
    group_id = world.get_group_of_element("T01")
    unrelated_node_id = world.get_element_vertex_node_id_by_type("T01", "B")

    assert not world.canDegrouperAtNode(group_id, unrelated_node_id)
    with pytest.raises(ValueError, match="aucun attachment incident"):
        world.degrouperAtNode(group_id, unrelated_node_id)

    assert set(world.attachments) == {"A001"}
    _assert_no_orphaned_resolved_attachments(world)
