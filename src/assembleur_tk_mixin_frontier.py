"""Compatibilité du découpage Tk : aucun calcul de Boundary topologique ici.

Depuis MIG-TOPO-SERVICE-001, le contour extérieur et ses incidences sont
fournis par les APIs publiques de ``TopologyWorld``. Les règles d'interaction
restent dans ``TriangleViewerManual``.
"""

from __future__ import annotations


class TriangleViewerFrontierGraphMixin:
    """Mixin conservé pour la structure historique de la classe Tk."""

    pass
