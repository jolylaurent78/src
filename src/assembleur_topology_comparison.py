"""Signatures canoniques des intentions d'attachments V2."""

from __future__ import annotations

from src.assembleur_core import TopologyEdgeEdgeAttachment, TopologyVertexEdgeAttachment


def build_attachment_signature(attachment) -> tuple:
    """Signature topologique V2, indépendante de l'ID et de mob/dest."""
    if isinstance(attachment, TopologyEdgeEdgeAttachment):
        return ("edge-edge", tuple(sorted(((attachment.mob_element_id, attachment.mob_edge), (attachment.dest_element_id, attachment.dest_edge)))))
    if isinstance(attachment, TopologyVertexEdgeAttachment):
        return ("vertex-edge", tuple(sorted(((attachment.mob_element_id, attachment.mob_vertex, attachment.mob_orientation), (attachment.dest_element_id, attachment.dest_vertex, attachment.dest_orientation)))))
    raise TypeError(f"Attachment V2 attendu, reçu: {type(attachment)!r}")


def _element_ids(attachment) -> frozenset[str]:
    if not isinstance(attachment, (TopologyEdgeEdgeAttachment, TopologyVertexEdgeAttachment)):
        raise TypeError(f"Attachment V2 attendu, reçu: {type(attachment)!r}")
    return frozenset((attachment.mob_element_id, attachment.dest_element_id))


def build_world_attachment_connections(world) -> dict[tuple, frozenset[str]]:
    connections: dict[tuple, set[str]] = {}
    for attachment in world.attachments.values():
        connections.setdefault(build_attachment_signature(attachment), set()).update(_element_ids(attachment))
    return {signature: frozenset(ids) for signature, ids in connections.items()}


def differing_attachment_element_ids(reference_world, current_world) -> set[str]:
    reference = build_world_attachment_connections(reference_world)
    current = build_world_attachment_connections(current_world)
    ids: set[str] = set()
    for signature in set(reference) ^ set(current):
        ids.update(reference.get(signature, ()))
        ids.update(current.get(signature, ()))
    return ids


def build_oriented_step_attachment_signature(attachment, element_a, element_b) -> tuple | None:
    element_a, element_b = str(element_a), str(element_b)
    if _element_ids(attachment) != frozenset((element_a, element_b)):
        return None
    if isinstance(attachment, TopologyEdgeEdgeAttachment):
        sides = {attachment.mob_element_id: (attachment.mob_element_id, attachment.mob_edge), attachment.dest_element_id: (attachment.dest_element_id, attachment.dest_edge)}
        return ("edge-edge", sides[element_a], sides[element_b])
    if isinstance(attachment, TopologyVertexEdgeAttachment):
        sides = {attachment.mob_element_id: (attachment.mob_element_id, attachment.mob_vertex, attachment.mob_orientation), attachment.dest_element_id: (attachment.dest_element_id, attachment.dest_vertex, attachment.dest_orientation)}
        return ("vertex-edge", sides[element_a], sides[element_b])
    raise TypeError(f"Attachment V2 attendu, reçu: {type(attachment)!r}")


def build_topology_prefix_steps(world, triangle_ids, upto_index: int):
    if world is None or upto_index < 0:
        return None
    if upto_index == 0:
        return []
    ordered_ids = [str(triangle_id) for triangle_id in (triangle_ids or [])]
    if len(ordered_ids) < upto_index + 1:
        return None
    steps = []
    for index in range(upto_index):
        signatures = [signature for attachment in world.attachments.values() if (signature := build_oriented_step_attachment_signature(attachment, ordered_ids[index], ordered_ids[index + 1])) is not None]
        if not signatures:
            return None
        steps.append(tuple(sorted(signatures)))
    return steps
