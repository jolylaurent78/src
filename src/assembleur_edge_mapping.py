"""Règles géométriques partagées pour les raccords edge-edge."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EdgeEdgePose:
    """Pose absolue d'un mobile aligné sur l'arête d'une cible."""

    mapping: str
    rotation: np.ndarray
    translation: np.ndarray
    mirrored: bool


def _orient2d(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _third_vertex(a: str, b: str) -> str:
    for key in ("O", "B", "L"):
        if key != a and key != b:
            return key
    raise ValueError("Arête triangle invalide")


def choose_edge_edge_mapping_by_orientation(
    mobile_points: dict,
    mobile_start: str,
    mobile_end: str,
    target_points: dict,
    target_start: str,
    target_end: str,
) -> str:
    """Choisit le mapping face-à-face à partir de la géométrie courante.

    Le mobile est exprimé dans son repère local et la cible dans son repère
    monde. Aucun mapping historique n'est réutilisé.
    """
    mobile_third = _third_vertex(mobile_start, mobile_end)
    target_third = _third_vertex(target_start, target_end)
    mobile_side = _orient2d(
        mobile_points[mobile_start], mobile_points[mobile_end], mobile_points[mobile_third]
    )
    target_side_direct = _orient2d(
        target_points[target_start], target_points[target_end], target_points[target_third]
    )
    target_side_reverse = _orient2d(
        target_points[target_end], target_points[target_start], target_points[target_third]
    )

    epsilon = 1e-9
    if abs(mobile_side) < epsilon or abs(target_side_direct) < epsilon:
        return "direct"
    if mobile_side * target_side_direct < 0:
        return "direct"
    if mobile_side * target_side_reverse < 0:
        return "reverse"
    return "direct"


def compute_edge_edge_pose(
    mobile_points: dict,
    mobile_start: str,
    mobile_end: str,
    target_points: dict,
    target_start: str,
    target_end: str,
    *,
    mobile_mirrored: bool = False,
) -> EdgeEdgePose:
    """Calcule la pose absolue qui superpose une arête mobile à une cible.

    Les points mobiles sont locaux et ceux de la cible sont monde.  La pose
    retournée respecte le contrat Core ``world = R @ (M @ local) + T`` quand
    ``mirrored`` vaut ``True``.
    """
    normalized_mobile = {
        key: np.asarray(value, dtype=float)
        for key, value in mobile_points.items()
    }
    if mobile_mirrored:
        mirror = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
        effective_mobile = {
            key: mirror @ value for key, value in normalized_mobile.items()
        }
    else:
        effective_mobile = normalized_mobile
    normalized_target = {
        key: np.asarray(value, dtype=float)
        for key, value in target_points.items()
    }

    mapping = choose_edge_edge_mapping_by_orientation(
        effective_mobile,
        mobile_start,
        mobile_end,
        normalized_target,
        target_start,
        target_end,
    )
    target_first, target_second = (
        (target_start, target_end)
        if mapping == "direct"
        else (target_end, target_start)
    )
    mobile_vector = effective_mobile[mobile_end] - effective_mobile[mobile_start]
    target_vector = normalized_target[target_second] - normalized_target[target_first]
    mobile_length = float(np.linalg.norm(mobile_vector))
    target_length = float(np.linalg.norm(target_vector))
    if mobile_length <= 1e-12 or target_length <= 1e-12:
        raise ValueError("Arête dégénérée pour une pose edge-edge")

    angle = math.atan2(float(target_vector[1]), float(target_vector[0])) - math.atan2(
        float(mobile_vector[1]), float(mobile_vector[0])
    )
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    translation = normalized_target[target_first] - rotation @ effective_mobile[mobile_start]
    return EdgeEdgePose(mapping, rotation, translation, bool(mobile_mirrored))


def apply_edge_edge_pose(points: dict, pose: EdgeEdgePose) -> dict:
    """Projette des points locaux par une :class:`EdgeEdgePose` pure."""
    mirror = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    return {
        key: pose.rotation @ (mirror @ np.asarray(value, dtype=float) if pose.mirrored else np.asarray(value, dtype=float))
        + pose.translation
        for key, value in points.items()
    }
