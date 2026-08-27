"""Objets metier temporaires partages par le moteur et l'IHM DEFORM."""

from __future__ import annotations

from dataclasses import dataclass, field


VertexLambertPoint = tuple[float, float]
DeformationOccurrence = tuple[str, str]


@dataclass
class WorkingPoint:
    """Point DEFORM temporaire et identite explicite de son partage."""

    point_id: str
    lambert_xy: VertexLambertPoint
    occurrences: set[DeformationOccurrence] = field(default_factory=set)
