"""Simulation Core pure de déformation temporaire d'un triangle Catalogue."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import (
    TopologyConstraintGeometryError,
    TopologyWorld,
)
from src.assembleur_scenario import materialize_catalogue_triangle


@dataclass(frozen=True)
class DeformationSimulationResult:
    """Résultat immutable d'un candidat de déformation."""

    accepted: bool
    world: TopologyWorld | None
    element_id: str
    vertex_lambert_overrides: Mapping[str, tuple[float, float]]
    rejection_reason: str | None = None


def _normalize_lambert_point(value) -> tuple[float, float] | None:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    return (float(point[0]), float(point[1]))


def _normalize_vertex_lambert_overrides(
    overrides: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]] | None:
    if not isinstance(overrides, Mapping):
        raise ValueError("Les overrides Lambert doivent former une mapping O/B/L")
    normalized: dict[str, tuple[float, float]] = {}
    for role, point in overrides.items():
        if role not in {"O", "B", "L"}:
            raise ValueError(f"Rôle d'override Lambert inconnu: {role!r}")
        normalized_point = _normalize_lambert_point(point)
        if normalized_point is None:
            return None
        normalized[role] = normalized_point
    return normalized


def _require_selected_group_anchor(
    world: TopologyWorld,
    group_id: str,
):
    """Trouve l'unique ancre du groupe depuis son node_id durable."""
    anchors = []
    for anchor in world.groupAnchors.values():
        try:
            anchor_group_id = world.getGroupIdFromConceptNode(anchor.node_id)
        except ValueError as exc:
            raise ValueError(
                f"Ancre {anchor.anchor_id!r}: node_id inexistant {anchor.node_id!r}"
            ) from exc
        if anchor_group_id == group_id:
            anchors.append(anchor)
    if len(anchors) != 1:
        raise ValueError(
            f"La simulation exige exactement une ancre résoluble pour le groupe {group_id!r}"
        )
    anchor = anchors[0]
    if world._beacon_resolver is None:
        raise ValueError("La simulation exige un BeaconResolver")
    if not world._beacon_resolver.contains(anchor.beacon_id):
        raise ValueError(
            f"La simulation exige une balise résoluble: {anchor.beacon_id!r}"
        )
    beacon_world = np.asarray(world.getBeaconWorldXY(anchor.beacon_id), dtype=float)
    if beacon_world.shape != (2,) or not np.all(np.isfinite(beacon_world)):
        raise ValueError(
            f"La simulation exige une position World finie pour {anchor.beacon_id!r}"
        )
    return anchor


def _rejected(
    element_id: str,
    vertex_lambert_overrides: Mapping[str, tuple[float, float]],
    reason: str,
) -> DeformationSimulationResult:
    return DeformationSimulationResult(
        accepted=False,
        world=None,
        element_id=element_id,
        vertex_lambert_overrides=MappingProxyType(dict(vertex_lambert_overrides)),
        rejection_reason=reason,
    )


def simulate_triangle_deformation(
    *,
    catalogue: Catalogue,
    initial_world: TopologyWorld,
    element_id: str,
    vertex_lambert_overrides: Mapping[str, tuple[float, float]],
) -> DeformationSimulationResult:
    """Simule des overrides O/B/L sans muter le Catalogue ni le World source.

    La déformation remplace uniquement la géométrie intrinsèque du triangle.
    Les attachments V2 restent inchangés ; leurs résolutions et les états
    géométriques dérivés sont reconstruits par le Core avant le replay.
    Les coordonnées sont Lambert (mètres), soit le repère canonique utilisé
    par la factory de matérialisation Catalogue.
    """
    normalized_overrides = _normalize_vertex_lambert_overrides(
        vertex_lambert_overrides
    )
    if normalized_overrides is None:
        return _rejected(
            element_id,
            {},
            "Override Lambert candidat invalide",
        )

    element = initial_world.elements.get(element_id)
    if element is None:
        raise ValueError(f"Élément topologique inconnu: {element_id!r}")
    source_triangle_id = (element.source_triangle_id or "").strip()
    if not source_triangle_id:
        raise ValueError(
            f"Élément topologique sans source_triangle_id: {element_id!r}"
        )

    # Valide d'abord la donnée Catalogue durable. Une géométrie catalogue
    # incohérente est une erreur de contrat, pas un simple candidat refusé.
    catalogue.get_triangle(source_triangle_id)
    catalogue.get_triangle_geometry(source_triangle_id)
    group_id = initial_world.get_group_of_element(element_id)
    anchor = _require_selected_group_anchor(initial_world, group_id)

    try:
        replacement = materialize_catalogue_triangle(
            catalogue,
            source_triangle_id,
            vertex_lambert_overrides=normalized_overrides,
        )
    except ValueError as exc:
        return _rejected(element_id, normalized_overrides, str(exc))

    working_world = initial_world.clonePhysicalState()
    working_world.replace_element_intrinsic_geometry(element_id, replacement)
    working_world.rebuild_from_attachments()
    working_world.reconcileGroupAnchorsByNode()

    candidate_group_id = working_world.get_group_of_element(element_id)
    candidate_anchor = working_world.groupAnchors.get(anchor.anchor_id)
    if candidate_anchor is None:
        raise ValueError(f"Ancre perdue pendant la reconstruction: {anchor.anchor_id!r}")
    if candidate_anchor.node_id != anchor.node_id or candidate_anchor.beacon_id != anchor.beacon_id:
        raise ValueError(f"Ancre altérée pendant la reconstruction: {anchor.anchor_id!r}")
    if candidate_anchor.group_id != candidate_group_id:
        raise ValueError(
            f"Ancre {anchor.anchor_id!r} hors du groupe reconstruit {candidate_group_id!r}"
        )

    try:
        working_world.replay_group_attachment_poses(
            candidate_group_id,
            element_id,
        )
        working_world.reconcileGroupAnchorsByNode()
        working_world.applyGroupAnchor(candidate_anchor.anchor_id)
        if not working_world.is_group_contour_valid(candidate_group_id):
            return _rejected(
                element_id,
                normalized_overrides,
                "Contour du groupe invalide",
            )
    except TopologyConstraintGeometryError as exc:
        return _rejected(element_id, normalized_overrides, str(exc))

    return DeformationSimulationResult(
        accepted=True,
        world=working_world,
        element_id=element_id,
        vertex_lambert_overrides=MappingProxyType(dict(normalized_overrides)),
    )
