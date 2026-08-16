import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.assembleur_core import (
    TopologyEdgeEdgeAttachment,
    TopologyElement,
    TopologyNodeType,
    TopologyVertexEdgeAttachment,
    TopologyWorld,
)


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


def _single_triangle_world() -> tuple[TopologyWorld, str]:
    world = TopologyWorld()
    return world, world.add_element_as_new_group(_element("T01"))


def _two_triangles_with_shared_edge() -> tuple[TopologyWorld, str]:
    world = TopologyWorld()
    world.add_element_as_new_group(_element("T01"))
    world.add_element_as_new_group(_element("T02"))
    world.setElementPose(
        "T02",
        np.array([[-1.0, 0.0], [0.0, -1.0]]),
        np.array([3.0, 0.0]),
    )
    group_id = world.apply_attachment(
        TopologyEdgeEdgeAttachment("A01", "T01", "OB", "T02", "OB")
    )
    return world, group_id


def test_ensure_boundary_is_lazy_cached_and_recomputed_after_invalidation():
    world, group_id = _single_triangle_world()
    calls = []
    original_compute = world.computeBoundary

    def counted_compute(core_group_id):
        calls.append(core_group_id)
        return original_compute(core_group_id)

    world.computeBoundary = counted_compute

    world.ensureBoundary(group_id)
    world.ensureBoundary(group_id)
    assert calls == [group_id]

    world.invalidateConceptGeom(group_id)
    world.ensureBoundary(group_id)
    assert calls == [group_id, group_id]


def test_ensure_boundary_rejects_unknown_noncanonical_group():
    world, _group_id = _single_triangle_world()

    with pytest.raises(ValueError, match="unknown canonical group"):
        world.ensureBoundary("G-UNKNOWN")


def test_incident_boundary_segments_for_isolated_triangle_are_exactly_two():
    world, group_id = _single_triangle_world()
    segments = world.getBoundarySegments(group_id)

    assert len(segments) == 3
    for vertex_index in range(3):
        node_id = world.format_node_id("T01", vertex_index)
        incident = world.getIncidentBoundarySegments(group_id, node_id)
        assert len(incident) == 2
        assert all(node_id in (segment.conceptA, segment.conceptB) for segment in incident)
        assert all(segment.elementId == "T01" for segment in incident)


def test_shared_edge_is_never_returned_by_boundary_services():
    world, group_id = _two_triangles_with_shared_edge()
    segments = world.getBoundarySegments(group_id)

    assert len(segments) == 4
    assert all(not (segment.elementId in {"T01", "T02"} and segment.edgeIndex == 0) for segment in segments)
    for node_id in {segment.conceptA for segment in segments}.union(segment.conceptB for segment in segments):
        incident = world.getIncidentBoundarySegments(group_id, node_id)
        assert incident == world.getIncidentBoundarySegments(group_id, node_id)
        assert all(segment.edgeIndex != 0 for segment in incident)
        assert all(node_id in (segment.conceptA, segment.conceptB) for segment in incident)


def test_edge_edge_mutation_invalidates_precomputed_boundary_and_rebuilds_it(tmp_path):
    world = TopologyWorld()
    group_t01 = world.add_element_as_new_group(_element("T01"))
    group_t02 = world.add_element_as_new_group(_element("T02"))
    world.setElementPose(
        "T02",
        np.array([[-1.0, 0.0], [0.0, -1.0]]),
        np.array([3.0, 0.0]),
    )

    # Reproduit la précondition du bug : chaque contour existe avant la fusion.
    world.ensureBoundary(group_t01)
    world.ensureBoundary(group_t02)

    group_id = world.apply_attachment(
        TopologyEdgeEdgeAttachment("A01", "T01", "OB", "T02", "OB")
    )

    cache = world._concept_cache(group_id)
    assert cache.boundaryCycle is None
    assert cache.boundaryEdges is None
    assert cache.boundaryIndex is None
    assert cache.boundaryOrientation is None
    assert cache.boundaries.cycle is None

    segments = world.getBoundarySegments(group_id)
    assert len(segments) == 4
    assert all(
        not (segment.elementId in {"T01", "T02"} and segment.edgeIndex == 0)
        for segment in segments
    )

    graph = world.ensureConceptGraph(group_id)
    shared_edge = tuple(sorted((
        world.find_node("T01:N0"),
        world.find_node("T01:N1"),
    )))
    assert len(graph.edges[shared_edge].occurrences) == 2

    dump = tmp_path / "TopoDump.xml"
    world.export_topo_dump_xml(str(dump))
    root = ET.parse(dump).getroot()
    boundary_nodes = root.findall(
        f"./ConceptModels/ConceptModel[@group='{group_id}']/Boundary/Cycle/N"
    )
    assert boundary_nodes
    assert all(node.get("id") == world.find_node(node.get("id")) for node in boundary_nodes)


def test_vertex_edge_mutation_also_invalidates_precomputed_boundary():
    world = TopologyWorld()
    group_t01 = world.add_element_as_new_group(_element("T01"))
    group_t02 = world.add_element_as_new_group(_element("T02"))
    world.ensureBoundary(group_t01)
    world.ensureBoundary(group_t02)

    group_id = world.apply_attachment(
        TopologyVertexEdgeAttachment("A01", "T01", "O", "LO", "T02", "O", "LO")
    )

    cache = world._concept_cache(group_id)
    assert cache.boundaryCycle is None
    assert cache.boundaries.cycle is None
    segments = world.getBoundarySegments(group_id)
    assert all(
        segment.conceptA == world.find_node(segment.conceptA)
        and segment.conceptB == world.find_node(segment.conceptB)
        for segment in segments
    )


def test_boundary_segment_order_and_physical_metadata_are_stable():
    world, group_id = _two_triangles_with_shared_edge()
    first = world.getBoundarySegments(group_id)
    second = world.getBoundarySegments(group_id)

    assert first == second
    assert [(segment.elementId, segment.edgeIndex) for segment in first] == [
        (segment.elementId, segment.edgeIndex) for segment in second
    ]
    assert all(segment.fromNodeId != segment.toNodeId for segment in first)
    assert all(0.0 <= segment.t0 < segment.t1 <= 1.0 for segment in first)
