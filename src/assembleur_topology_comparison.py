"""Signatures canoniques de connexions topologiques entre scénarios."""

from __future__ import annotations

from typing import Any


def _endpoint(feature) -> tuple[str, str, int]:
    """Identité stable d'une feature, indépendante de ses objets runtime."""
    return (
        str(getattr(feature, "feature_type", "")),
        str(getattr(feature, "element_id", "")),
        int(getattr(feature, "index", -1)),
    )


def _semantic_params(attachment) -> tuple[tuple[str, Any], ...]:
    """Conserve uniquement les paramètres qui changent le fait topologique."""
    kind = str(getattr(attachment, "kind", ""))
    params = dict(getattr(attachment, "params", {}) or {})
    if kind == "edge-edge":
        return (("mapping", str(params.get("mapping", "direct")).lower()),)
    if kind == "vertex-edge":
        t = round(float(params.get("t", 0.0)), 12)
        return (("edgeFrom", str(params.get("edgeFrom", ""))), ("t", t))
    return ()


def build_attachment_signature(attachment) -> tuple:
    """Retourne une signature stable d'un attachment.

    Les endpoints sont triés : ni le côté mobile/statique, ni l'ordre de
    création, ni ``attachment_id``/``source`` ne participent à la comparaison.
    """
    endpoints = tuple(sorted((_endpoint(attachment.feature_a), _endpoint(attachment.feature_b))))
    return (str(getattr(attachment, "kind", "")), endpoints, _semantic_params(attachment))


def build_world_attachment_connections(world) -> dict[tuple, frozenset[str]]:
    """Mappe chaque signature canonique vers les éléments graphiquement concernés."""
    connections: dict[tuple, set[str]] = {}
    for attachment in dict(getattr(world, "attachments", {}) or {}).values():
        signature = build_attachment_signature(attachment)
        ids = {
            str(getattr(attachment.feature_a, "element_id", "")),
            str(getattr(attachment.feature_b, "element_id", "")),
        }
        connections.setdefault(signature, set()).update(element_id for element_id in ids if element_id)
    return {signature: frozenset(element_ids) for signature, element_ids in connections.items()}


def differing_attachment_element_ids(reference_world, current_world) -> set[str]:
    """Éléments touchés par les connexions présentes dans un seul des mondes."""
    ref = build_world_attachment_connections(reference_world)
    cur = build_world_attachment_connections(current_world)
    differing = set(ref) ^ set(cur)
    ids: set[str] = set()
    for signature in differing:
        ids.update(ref.get(signature, ()))
        ids.update(cur.get(signature, ()))
    return ids
