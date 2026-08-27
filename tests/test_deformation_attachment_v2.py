import numpy as np
import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import (
    ResolvedVertexEdgeAttachment,
    TopologyEdgeEdgeAttachment,
    TopologyVertexEdgeAttachment,
    TopologyWorld,
    compute_vertex_edge_attachment_orientation,
)
from src.assembleur_deformation import (
    simulate_deformation_session,
    simulate_occurrence_deformation,
)
from src.assembleur_deformation_points import WorkingPoint
from src.assembleur_geometry_reference import GeometryReferenceResolver, ScenarioReference
from src.assembleur_scenario import materialize_catalogue_triangle


class _BeaconResolver:
    def contains(self, beacon_id):
        return beacon_id == "BEA-0001"

    def get_world(self, beacon_id):
        if not self.contains(beacon_id):
            raise KeyError(beacon_id)
        return (500.0, -200.0)


def _catalogue_and_v2_chain():
    catalogue = Catalogue()
    opening = catalogue.add_city("Ouverture", 45.0, 2.0)
    base = catalogue.add_city("Base", 45.0, 3.0)
    north = catalogue.add_city("Lumière nord", 46.0, 2.5)
    south = catalogue.add_city("Lumière sud", 44.0, 2.5)
    chain = catalogue.add_city("Lumière chaîne", 43.5, 2.4)
    first = catalogue.add_triangle("Do", opening.city_id, base.city_id, north.city_id)
    second = catalogue.add_triangle("Si", opening.city_id, base.city_id, south.city_id)
    third = catalogue.add_triangle("La", opening.city_id, base.city_id, chain.city_id)
    world = TopologyWorld(beacon_resolver=_BeaconResolver())
    elements = [materialize_catalogue_triangle(catalogue, item.triangle_id) for item in (first, second, third)]
    for element in elements:
        world.add_element_as_new_group(element)
    world.apply_attachment(TopologyEdgeEdgeAttachment("A001", elements[0].element_id, "OB", elements[1].element_id, "OB"))
    world.apply_attachment(TopologyVertexEdgeAttachment(
        "A002", elements[2].element_id, "O", "LO", elements[1].element_id, "L", "LO",
        compute_vertex_edge_attachment_orientation(world, elements[2].element_id, "O", "LO"),
        compute_vertex_edge_attachment_orientation(world, elements[1].element_id, "L", "LO"),
    ))
    group_id = world.get_group_of_element(elements[0].element_id)
    world.replay_group_attachment_poses(group_id, elements[1].element_id)
    anchor = world.createGroupAnchor(group_id, "BEA-0001", world.get_element_vertex_node_id_by_type(elements[0].element_id, "O"))
    world.applyGroupAnchor(anchor.anchor_id)
    return catalogue, world, elements[1].element_id, second, anchor


def test_cow_occurrence_deformation_replays_the_complete_attachment_chain():
    catalogue, world, element_id, triangle, anchor = _catalogue_and_v2_chain()
    light_before = np.asarray(catalogue.get_city_lambert(triangle.light_city_id))
    source_snapshot = world._exportPhysicalSnapshot()
    source_attachments = tuple(world.attachments.values())
    resolved_before = world.getResolvedAttachment("A002")
    third_pose_before = world.getElementPose("T03")
    working_point = WorkingPoint("TMP-0001", tuple(light_before + (1000.0, 0.0)), {(element_id, "L")})
    result = simulate_occurrence_deformation(
        resolver=GeometryReferenceResolver(catalogue, ScenarioReference()),
        initial_world=world,
        occurrence_lambert_overrides={occurrence: working_point.lambert_xy for occurrence in working_point.occurrences},
    )
    assert result.accepted and result.world is not None
    assert tuple(result.world.attachments.values()) == source_attachments
    assert world._exportPhysicalSnapshot() == source_snapshot
    resolved = result.world.getResolvedAttachment("A002")
    assert isinstance(resolved, ResolvedVertexEdgeAttachment)
    assert resolved != resolved_before
    vertex_local = result.world._local_vertex_point_by_type(resolved.vertex_element_id, resolved.vertex)
    edge_local = result.world._resolved_edge_local_point(resolved.edge_element_id, resolved.edge, resolved.edge_anchor_vertex, resolved.position_from_anchor, resolved.attachment_id)
    assert result.world.elementLocalToWorld(resolved.vertex_element_id, vertex_local) == pytest.approx(result.world.elementLocalToWorld(resolved.edge_element_id, edge_local))
    assert not np.allclose(result.world.getElementPose("T03")[1], third_pose_before[1])
    restored = result.world.getGroupAnchor(anchor.anchor_id)
    assert result.world.getConceptNodeWorldXY(restored.node_id, restored.group_id) == (500.0, -200.0)


def test_pivot_session_is_independent_from_geometry_references():
    _catalogue, world, _element_id, _triangle, _anchor = _catalogue_and_v2_chain()
    result = simulate_deformation_session(reference_world=world, pivoted_attachment_ids=())
    assert result.accepted
    assert result.world is world
