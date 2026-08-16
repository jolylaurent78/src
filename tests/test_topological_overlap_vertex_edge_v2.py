import math

from src.assembleur_core import (
    TopologyElement,
    TopologyVertexEdgeAttachment,
    TopologyWorld,
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


def _world_with_two_triangles(
    mob_light: tuple[float, float],
    dest_light: tuple[float, float],
) -> tuple[TopologyWorld, str, str]:
    world = TopologyWorld()
    group_mob = world.add_element_as_new_group(_triangle("T01", mob_light))
    group_dest = world.add_element_as_new_group(_triangle("T02", dest_light))
    return world, group_mob, group_dest


def test_vertex_edge_topological_overlap_accepts_a_valid_boundary_ring():
    world, group_mob, group_dest = _world_with_two_triangles((3.0, 4.0), (3.0, 4.0))
    attachment = TopologyVertexEdgeAttachment(
        "VE-VALID",
        "T01",
        "O",
        "OB",
        "T02",
        "O",
        "LO",
    )

    overlap, ring_out = world._simulate_topological_overlap_vertex_edge(
        group_dest,
        group_mob,
        attachment,
    )

    assert world.simulate_topological_overlap(group_dest, group_mob, attachment) is False
    assert overlap is False
    assert len(ring_out) == 5
    assert world._isValidPolygon(ring_out) is True


def test_vertex_edge_topological_overlap_rejects_an_invalid_boundary_ring():
    world, group_mob, group_dest = _world_with_two_triangles((3.0, 4.0), (6.0, 8.0))
    attachment = TopologyVertexEdgeAttachment(
        "VE-INVALID",
        "T01",
        "O",
        "LO",
        "T02",
        "O",
        "LO",
    )

    overlap, ring_out = world._simulate_topological_overlap_vertex_edge(
        group_dest,
        group_mob,
        attachment,
    )

    assert world.simulate_topological_overlap(group_dest, group_mob, attachment) is True
    assert overlap is True
    assert world._isValidPolygon(ring_out) is False
