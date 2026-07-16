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


def build_oriented_step_attachment_signature(attachment, element_a, element_b) -> tuple | None:
    """Signature d'un attachment réorientée pour l'étape ``element_a -> element_b``.

    L'attachment est retenu seulement lorsqu'il relie exactement les deux
    éléments de l'étape. Les endpoints sont ensuite placés côté déjà assemblé
    (A) et côté ajouté (B), indépendamment de leur ordre de stockage.
    """
    feature_a = getattr(attachment, "feature_a", None)
    feature_b = getattr(attachment, "feature_b", None)
    endpoint_a = _endpoint(feature_a)
    endpoint_b = _endpoint(feature_b)
    expected = {str(element_a), str(element_b)}
    if {endpoint_a[1], endpoint_b[1]} != expected:
        return None

    if endpoint_a[1] == str(element_a):
        step_a, step_b = endpoint_a, endpoint_b
    else:
        step_a, step_b = endpoint_b, endpoint_a

    return (
        str(getattr(attachment, "kind", "")),
        step_a,
        step_b,
        _semantic_params(attachment),
    )


def build_topology_prefix_steps(world, tri_ids, upto_index: int):
    """Construit les signatures Core des étapes du préfixe de ``tri_ids``.

    Retourne ``[]`` pour zéro étape, ``None`` si le monde, le parcours ou une
    liaison d'étape est indisponible. Chaque étape est un tuple trié de toutes
    les signatures d'attachments reliant les deux triangles consécutifs.
    """
    if world is None:
        return None
    ordered_ids = [str(triangle_id) for triangle_id in list(tri_ids or [])]
    if upto_index < 0:
        return None
    if upto_index == 0:
        return []
    if len(ordered_ids) < upto_index + 1:
        return None

    attachments = list(dict(getattr(world, "attachments", {}) or {}).values())
    steps = []
    for index in range(upto_index):
        element_a = ordered_ids[index]
        element_b = ordered_ids[index + 1]
        signatures = [
            signature
            for attachment in attachments
            if (signature := build_oriented_step_attachment_signature(
                attachment, element_a, element_b
            )) is not None
        ]
        if not signatures:
            return None
        steps.append(tuple(sorted(signatures)))
    return steps
