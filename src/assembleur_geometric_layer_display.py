"""Surcharges d'affichage AssembleurTriangles pour les modules Traces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometricLayerModuleDisplayOverride:
    """Valeurs optionnelles qui remplacent les graphiques d'origine d'un module."""

    color_bgr: tuple[int, int, int] | None = None
    width: int | None = None
