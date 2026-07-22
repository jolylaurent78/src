"""Projection pure de ``TopologyWorld`` vers le cache graphique runtime."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from src.assembleur_core import TopologyNodeType, TopologyWorld


_VERTEX_KEY_BY_TYPE = {
    TopologyNodeType.OUVERTURE: "O",
    TopologyNodeType.BASE: "B",
    TopologyNodeType.LUMIERE: "L",
}


def getCoreTriangleWorldPoints(
    topologyWorld: TopologyWorld,
    elementId: str,
) -> Dict[str, np.ndarray]:
    """Reconstruit les sommets O/B/L depuis l'élément et sa pose Core."""
    key = str(elementId or "").strip()
    if topologyWorld is None:
        raise RuntimeError("Projection Core: TopologyWorld absent")
    if not key:
        raise ValueError("Projection Core: topoElementId absent")
    element = topologyWorld.elements.get(key)
    if element is None:
        raise KeyError(f"Projection Core: élément absent: {key!r}")

    points: Dict[str, np.ndarray] = {}
    for vertex_index, vertex_type in enumerate(element.vertex_types):
        point_key = _VERTEX_KEY_BY_TYPE.get(vertex_type)
        if point_key is None:
            continue
        if point_key in points:
            raise ValueError(
                f"Projection Core: type sommet dupliqué {point_key!r} "
                f"pour {key}"
            )
        local_xy = element.vertex_local_xy.get(vertex_index)
        if local_xy is None:
            raise ValueError(
                f"Projection Core: coordonnée locale absente "
                f"element={key} vertex_index={vertex_index}"
            )
        world_xy = np.asarray(element.localToWorld(local_xy), dtype=float)
        if world_xy.shape != (2,) or not np.all(np.isfinite(world_xy)):
            raise ValueError(
                f"Projection Core: coordonnée monde invalide "
                f"element={key} vertex={point_key}"
            )
        points[point_key] = np.array(world_xy, dtype=float, copy=True)

    missing = {"O", "B", "L"}.difference(points)
    if missing:
        raise ValueError(
            f"Projection Core: types sommets manquants element={key}: {sorted(missing)}"
        )
    return points


def getManualProjectionElementIds(topologyWorld: TopologyWorld) -> list[str]:
    """Retourne les éléments manuels, contigus par groupe Core vivant."""
    if topologyWorld is None:
        raise RuntimeError("Projection Core: TopologyWorld absent")
    element_ids: list[str] = []
    for core_group_id in topologyWorld.getLiveGroupIds():
        element_ids.extend(
            element_id
            for element_id in topologyWorld.getGroupElementIds(core_group_id)
        )
    _validate_element_ids(topologyWorld, element_ids, require_complete=True)
    return element_ids


def buildLastDrawnFromTopology(
    *,
    topologyWorld: TopologyWorld,
    elementIds: Iterable[str],
) -> list[dict]:
    """Construit une nouvelle projection indépendante exclusivement depuis le Core."""
    normalized_ids = _validate_element_ids(
        topologyWorld,
        elementIds if elementIds is not None else [],
        require_complete=True,
    )
    return [
        {
            "topoElementId": element_id,
            "pts": getCoreTriangleWorldPoints(topologyWorld, element_id),
        }
        for element_id in normalized_ids
    ]


def _validate_element_ids(
    topologyWorld: TopologyWorld,
    elementIds: Iterable[str],
    *,
    require_complete: bool,
) -> list[str]:
    if topologyWorld is None:
        raise RuntimeError("Projection Core: TopologyWorld absent")
    normalized_ids: list[str] = []
    seen_ids: set[str] = set()
    for element_id in elementIds:
        key = str(element_id or "").strip()
        if not key:
            raise ValueError("Projection Core: topoElementId vide")
        if key in seen_ids:
            raise ValueError(f"Projection Core: topoElementId dupliqué: {key!r}")
        if key not in topologyWorld.elements:
            raise KeyError(f"Projection Core: élément absent: {key!r}")
        if topologyWorld.get_group_of_element(key) is None:
            raise ValueError(f"Projection Core: groupe absent pour élément {key!r}")
        seen_ids.add(key)
        normalized_ids.append(key)

    if require_complete and seen_ids != set(topologyWorld.elements):
        missing = sorted(set(topologyWorld.elements).difference(seen_ids))
        extra = sorted(seen_ids.difference(topologyWorld.elements))
        raise ValueError(
            "Projection Core: couverture d'éléments invalide "
            f"missing={missing!r} extra={extra!r}"
        )
    return normalized_ids
