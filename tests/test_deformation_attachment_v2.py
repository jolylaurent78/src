import xml.etree.ElementTree as ET

import numpy as np

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import (
    ResolvedVertexEdgeAttachment,
    TopologyEdgeEdgeAttachment,
    TopologyVertexEdgeAttachment,
    TopologyWorld,
)
from src.assembleur_deformation import simulate_triangle_deformation
from src.assembleur_scenario import materialize_catalogue_triangle


class _BeaconResolver:
    def __init__(self, positions):
        self._positions = dict(positions)

    def contains(self, beacon_id):
        return beacon_id in self._positions

    def get_world(self, beacon_id):
        return self._positions[beacon_id]


def _catalogue_and_v2_chain():
    catalogue = Catalogue()
    opening = catalogue.add_city("Ouverture", 45.0, 2.0)
    base = catalogue.add_city("Base", 45.0, 3.0)
    light_north = catalogue.add_city("Lumière nord", 46.0, 2.5)
    light_south = catalogue.add_city("Lumière sud", 44.0, 2.5)
    light_chain = catalogue.add_city("Lumière chaîne", 43.5, 2.4)
    first = catalogue.add_triangle("Do", opening.city_id, base.city_id, light_north.city_id)
    second = catalogue.add_triangle("Si", opening.city_id, base.city_id, light_south.city_id)
    third = catalogue.add_triangle("La", opening.city_id, base.city_id, light_chain.city_id)

    world = TopologyWorld()
    first_element = materialize_catalogue_triangle(catalogue, first.triangle_id)
    second_element = materialize_catalogue_triangle(catalogue, second.triangle_id)
    third_element = materialize_catalogue_triangle(catalogue, third.triangle_id)
    for element in (first_element, second_element, third_element):
        world.add_element_as_new_group(element)

    edge_edge = TopologyEdgeEdgeAttachment(
        "A001", first_element.element_id, "OB", second_element.element_id, "OB"
    )
    vertex_edge = TopologyVertexEdgeAttachment(
        "A002",
        third_element.element_id,
        "L",
        "LO",
        second_element.element_id,
        "L",
        "LO",
    )
    world.apply_attachment(edge_edge)
    group_id = world.apply_attachment(vertex_edge)
    world.replay_group_attachment_poses(group_id, second_element.element_id)

    world.attachBeaconResolver(_BeaconResolver({"BEA-0001": (500.0, -200.0)}))
    anchor = world.createGroupAnchor(
        world.get_group_of_element(first_element.element_id),
        "BEA-0001",
        world.get_element_vertex_node_id_by_type(first_element.element_id, "O"),
    )
    world.applyGroupAnchor(anchor.anchor_id)
    return catalogue, world, third_element.element_id, third.triangle_id, anchor


def _attachment_dump_signature(world, path):
    world.export_topo_dump_xml(str(path))
    return [
        (child.tag, tuple(sorted(child.attrib.items())))
        for child in ET.parse(path).getroot().find("Attachments")
    ]


def test_deformation_keeps_v2_chain_intentions_and_rebuilds_derived_state():
    catalogue, world, element_id, triangle_id, anchor = _catalogue_and_v2_chain()
    attachments_before = tuple(world.attachments.values())
    resolved_before = world.getResolvedAttachment("A002")
    light_city_id = catalogue.get_triangle(triangle_id).light_city_id
    light_before = np.asarray(catalogue.get_city_lambert(light_city_id))

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=world,
        element_id=element_id,
        vertex_lambert_overrides={"L": tuple(light_before + np.asarray((1000.0, 0.0)))},
    )

    assert result.accepted
    assert tuple(result.world.attachments.values()) == attachments_before
    assert set(result.world.attachments) == {"A001", "A002"}
    assert isinstance(result.world.attachments["A001"], TopologyEdgeEdgeAttachment)
    assert isinstance(result.world.attachments["A002"], TopologyVertexEdgeAttachment)
    resolved_after = result.world.getResolvedAttachment("A002")
    assert isinstance(resolved_after, ResolvedVertexEdgeAttachment)
    assert resolved_after.position_from_anchor != resolved_before.position_from_anchor
    assert any(
        point.get("attachmentId") == "A002"
        for edge in result.world.elements[element_id].edges
        for point in result.world.buildDerivedSplitPointsForPhysEdge(element_id, edge.edge_index)
    )
    assert any(
        edge.coverages
        for element in result.world.elements.values()
        for edge in element.edges
    )
    restored_anchor = result.world.getGroupAnchor(anchor.anchor_id)
    assert result.world.getConceptNodeWorldXY(
        restored_anchor.node_id, restored_anchor.group_id
    ) == (500.0, -200.0)
    assert result.world.is_group_contour_valid(restored_anchor.group_id)


def test_deformation_topodump_keeps_attachment_intentions(tmp_path):
    catalogue, world, element_id, triangle_id, _anchor = _catalogue_and_v2_chain()
    attachment_dump_before = _attachment_dump_signature(world, tmp_path / "before.xml")
    light_city_id = catalogue.get_triangle(triangle_id).light_city_id
    light_before = np.asarray(catalogue.get_city_lambert(light_city_id))

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=world,
        element_id=element_id,
        vertex_lambert_overrides={"L": tuple(light_before + np.asarray((1000.0, 0.0)))},
    )

    assert result.accepted
    assert _attachment_dump_signature(result.world, tmp_path / "after.xml") == attachment_dump_before
