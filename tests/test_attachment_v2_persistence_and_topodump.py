import xml.etree.ElementTree as ET

from src.assembleur_core import (
    ResolvedEdgeEdgeAttachment,
    ResolvedVertexEdgeAttachment,
    TopologyEdgeEdgeAttachment,
    TopologyElement,
    TopologyVertexEdgeAttachment,
    TopologyWorld,
)


def _triangle(element_id: str, *, inverted: bool = False) -> TopologyElement:
    points = (
        {0: (0.0, 0.0), 1: (3.0, 0.0), 2: (3.0, -4.0)}
        if inverted
        else {0: (0.0, 0.0), 1: (3.0, 0.0), 2: (0.0, 4.0)}
    )
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[3.0, 5.0, 4.0],
        vertex_local_xy=points,
    )


def _world_with_edge_edge_and_vertex_edge() -> TopologyWorld:
    world = TopologyWorld()
    for element_id, inverted in (("T01", False), ("T02", True), ("T03", False)):
        world.add_element_as_new_group(_triangle(element_id, inverted=inverted))

    first_ee = TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "OB")
    world.apply_attachment(first_ee)
    world.replay_group_attachment_poses(world.get_group_of_element("T01"), "T01")

    ve = TopologyVertexEdgeAttachment(
        "A002", "T03", "L", "LO", "T02", "L", "LO"
    )
    group_id = world.apply_attachment(ve)
    world.replay_group_attachment_poses(group_id, "T02")
    return world


def test_v2_snapshot_round_trip_rebuilds_resolved_attachments_and_replays(tmp_path):
    world = _world_with_edge_edge_and_vertex_edge()
    snapshot = world._exportPhysicalSnapshot()

    assert snapshot["attachments"] == [
        {
            "kind": "edge-edge",
            "attachment_id": "A001",
            "mob_element_id": "T01",
            "mob_edge": "OB",
            "dest_element_id": "T02",
            "dest_edge": "OB",
        },
        {
            "kind": "vertex-edge",
            "attachment_id": "A002",
            "mob_element_id": "T03",
            "mob_vertex": "L",
            "mob_edge": "LO",
            "dest_element_id": "T02",
            "dest_vertex": "L",
            "dest_edge": "LO",
        },
    ]
    assert "resolved_attachments" not in snapshot

    restored = TopologyWorld()
    restored._importPhysicalSnapshot(snapshot)

    assert restored._exportPhysicalSnapshot() == snapshot
    assert isinstance(restored.attachments["A001"], TopologyEdgeEdgeAttachment)
    assert isinstance(restored.attachments["A002"], TopologyVertexEdgeAttachment)
    assert isinstance(restored.getResolvedAttachment("A001"), ResolvedEdgeEdgeAttachment)
    assert isinstance(restored.getResolvedAttachment("A002"), ResolvedVertexEdgeAttachment)
    assert restored.getResolvedAttachment("A001") == world.getResolvedAttachment("A001")
    assert restored.getResolvedAttachment("A002") == world.getResolvedAttachment("A002")

    restored.replay_group_attachment_poses(
        restored.get_group_of_element("T02"), "T02"
    )
    dump_before = tmp_path / "before.xml"
    dump_after = tmp_path / "after.xml"
    world.export_topo_dump_xml(str(dump_before))
    restored.export_topo_dump_xml(str(dump_after))
    attachments_before = ET.parse(dump_before).getroot().find("Attachments")
    attachments_after = ET.parse(dump_after).getroot().find("Attachments")
    assert ET.tostring(attachments_before) == ET.tostring(attachments_after)


def test_topodump_exposes_v2_intentions_resolutions_split_points_and_coverages(tmp_path):
    world = _world_with_edge_edge_and_vertex_edge()
    dump = tmp_path / "TopoDump.xml"
    world.export_topo_dump_xml(str(dump))
    root = ET.parse(dump).getroot()

    attachments = root.find("Attachments")
    assert attachments is not None
    assert attachments.get("count") == "2"
    assert attachments.find("Attachment") is None

    edge_edge = attachments.find("EdgeEdgeAttachment[@id='A001']")
    assert edge_edge is not None
    assert edge_edge.attrib == {
        "id": "A001",
        "mobElement": "T01",
        "mobEdge": "OB",
        "destElement": "T02",
        "destEdge": "OB",
    }
    assert edge_edge.find("Resolved[@status='ok']") is not None

    vertex_edge = attachments.find("VertexEdgeAttachment[@id='A002']")
    assert vertex_edge is not None
    assert vertex_edge.find("Resolved[@status='ok']") is not None
    assert vertex_edge.find("Resolved").get("positionFromAnchor") is not None

    edges = root.findall("./Elements/Element/Edges/Edge")
    assert any(edge.find("Coverages/C") is not None for edge in edges)
    assert any(
        split.get("source") == "vertex-edge"
        for edge in edges
        for split in edge.findall("DerivedSplitPoints/SplitPoint")
    )
    assert root.find("Groups/Group") is not None
    assert root.find("Nodes/Node") is not None
    assert root.find("ConceptModels/ConceptModel") is not None
