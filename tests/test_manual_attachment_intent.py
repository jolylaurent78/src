from dataclasses import FrozenInstanceError, fields

import pytest

from src.assembleur_core import TopologyElement, TopologyWorld
from src.assembleur_edgechoice import (
    ManualAttachmentIntent,
    buildManualAttachmentIntentFromBest,
)


def _triangle(element_id: str, labels) -> TopologyElement:
    return TopologyElement(
        name=element_id,
        vertex_labels=labels,
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[10.0, 10.0, 10.0],
        vertex_local_xy={0: (0.0, 0.0), 1: (10.0, 0.0), 2: (0.0, 10.0)},
        element_id=element_id,
    )


def _intent_from_lo_segments(mob_labels, dest_labels) -> ManualAttachmentIntent:
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle("T01", mob_labels))
    world.add_element_as_new_group(_triangle("T02", dest_labels))
    last_drawn = [
        {
            "topoElementId": "T01",
            "pts": {"O": (0.0, 0.0), "B": (10.0, 0.0), "L": (0.0, 10.0)},
        },
        {
            "topoElementId": "T02",
            "pts": {"O": (0.0, 0.0), "B": (10.0, 0.0), "L": (0.0, 10.0)},
        },
    ]
    intent = buildManualAttachmentIntentFromBest(
        (0.0, ((0.0, 0.0), (0.0, 10.0)), ((0.0, 0.0), (0.0, 10.0))),
        world=world,
        mob_idx=0,
        tgt_idx=1,
        mob_tids=[0],
        tgt_tids=[1],
        last_drawn=last_drawn,
        eps_world=1e-9,
        mATmpId=world.get_element_vertex_node_id_by_type("T01", "O"),
        tATmpId=world.get_element_vertex_node_id_by_type("T02", "O"),
    )
    assert intent is not None
    return intent


def test_manual_attachment_intent_vertex_edge_uses_core_business_identities():
    intent = _intent_from_lo_segments(
        ["mobile O", "mobile B", "mobile L"],
        ["destination O", "destination B", "destination L"],
    )

    assert intent == ManualAttachmentIntent(
        kind="vertex-edge",
        mob_element_id="T01",
        mob_vertex="O",
        mob_edge="LO",
        dest_element_id="T02",
        dest_vertex="O",
        dest_edge="LO",
    )


def test_manual_attachment_intent_edge_edge_has_no_mapping():
    intent = _intent_from_lo_segments(
        ["same O", "same B", "same L"],
        ["same O", "same B", "same L"],
    )

    assert intent.kind == "edge-edge"
    assert intent.mob_element_id == "T01"
    assert intent.dest_element_id == "T02"
    assert not hasattr(intent, "mapping")


def test_manual_attachment_intent_is_immutable_and_has_no_resolved_geometry_or_canvas_state():
    intent = _intent_from_lo_segments(
        ["mobile O", "mobile B", "mobile L"],
        ["destination O", "destination B", "destination L"],
    )

    assert {field.name for field in fields(ManualAttachmentIntent)} == {
        "kind",
        "mob_element_id",
        "mob_vertex",
        "mob_edge",
        "dest_element_id",
        "dest_vertex",
        "dest_edge",
    }
    for name in ("t", "tRaw", "position_from_anchor", "edgeFrom", "mapping", "pose", "mob_idx", "tgt_idx", "owner_tid"):
        assert not hasattr(intent, name)
    with pytest.raises(FrozenInstanceError):
        intent.kind = "edge-edge"
