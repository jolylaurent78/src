from types import SimpleNamespace

import pytest

from src.assembleur_core import TopologyEdgeEdgeAttachment, TopologyVertexEdgeAttachment
from src.assembleur_topology_comparison import (
    build_attachment_signature,
    build_oriented_step_attachment_signature,
    build_topology_prefix_steps,
    differing_attachment_element_ids,
)


def _world(*attachments):
    return SimpleNamespace(attachments={item.attachment_id: item for item in attachments})


def test_v2_signatures_are_symmetric_and_ignore_ids():
    ee = TopologyEdgeEdgeAttachment("A001", "T01", "OB", "T02", "BL")
    reverse_ee = TopologyEdgeEdgeAttachment("A999", "T02", "BL", "T01", "OB")
    changed_ee = TopologyEdgeEdgeAttachment("A002", "T01", "LO", "T02", "BL")
    ve = TopologyVertexEdgeAttachment("A003", "T01", "L", "LO", "T02", "B", "OB", "CCW", "CW")
    reverse_ve = TopologyVertexEdgeAttachment("A004", "T02", "B", "OB", "T01", "L", "LO", "CW", "CCW")
    changed_ve = TopologyVertexEdgeAttachment("A005", "T01", "O", "LO", "T02", "B", "OB", "CW", "CW")
    assert build_attachment_signature(ee) == build_attachment_signature(reverse_ee)
    assert build_attachment_signature(ee) != build_attachment_signature(changed_ee)
    assert build_attachment_signature(ve) == build_attachment_signature(reverse_ve)
    assert build_attachment_signature(ve) != build_attachment_signature(changed_ve)
    with pytest.raises(TypeError):
        build_attachment_signature(object())


def test_v2_difference_and_oriented_prefix_need_no_resolver():
    ee = TopologyEdgeEdgeAttachment("A001", "T02", "BL", "T01", "OB")
    same = TopologyEdgeEdgeAttachment("A999", "T01", "OB", "T02", "BL")
    changed = TopologyEdgeEdgeAttachment("A002", "T01", "LO", "T02", "BL")
    ve = TopologyVertexEdgeAttachment("A003", "T03", "L", "LO", "T02", "B", "BL", "CCW", "CCW")
    assert differing_attachment_element_ids(_world(ee), _world(same)) == set()
    assert differing_attachment_element_ids(_world(ee), _world(changed)) == {"T01", "T02"}
    assert build_topology_prefix_steps(_world(ee, ve), ["T01", "T02", "T03"], 2) == [
        (("edge-edge", ("T01", "OB"), ("T02", "BL")),),
        (("vertex-edge", ("T02", "B", "CCW"), ("T03", "L", "CCW")),),
    ]
    assert build_topology_prefix_steps(_world(ee), ["T01", "T02", "T03"], 2) is None
    assert build_topology_prefix_steps(_world(), [], 0) == []


def test_vertex_edge_comparison_ignores_creation_edges_but_keeps_intent():
    first = TopologyVertexEdgeAttachment(
        "A001", "T01", "O", "OB", "T02", "B", "BL", "CCW", "CW"
    )
    different_history = TopologyVertexEdgeAttachment(
        "A002", "T01", "O", "LO", "T02", "B", "OB", "CCW", "CW"
    )
    changed_vertex = TopologyVertexEdgeAttachment(
        "A003", "T01", "L", "LO", "T02", "B", "OB", "CCW", "CW"
    )
    changed_orientation = TopologyVertexEdgeAttachment(
        "A004", "T01", "O", "LO", "T02", "B", "OB", "CW", "CCW"
    )

    assert build_attachment_signature(first) == build_attachment_signature(different_history)
    first_oriented = build_oriented_step_attachment_signature(first, "T01", "T02")
    assert first_oriented == build_oriented_step_attachment_signature(
        different_history, "T01", "T02"
    )
    assert build_attachment_signature(first) != build_attachment_signature(changed_vertex)
    assert build_attachment_signature(first) != build_attachment_signature(changed_orientation)
    assert first_oriented != build_oriented_step_attachment_signature(
        changed_vertex, "T01", "T02"
    )
    assert first_oriented != build_oriented_step_attachment_signature(
        changed_orientation, "T01", "T02"
    )
