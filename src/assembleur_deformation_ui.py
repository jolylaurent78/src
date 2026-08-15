"""Etat pur du mode interactif de deformation.

Ce module ne depend pas de Tk.  Il isole la courte machine d'etat qui relie
les gestes de l'IHM au moteur de simulation pur.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.assembleur_core import TopologyWorld


VertexLambertPoint = tuple[float, float]


@dataclass
class DeformationUiState:
    """Etat temporaire, explicitement distinct du scenario actif."""

    active: bool = False
    element_id: str | None = None
    reference_world: TopologyWorld | None = None
    vertex_lambert_overrides: dict[str, VertexLambertPoint] = field(
        default_factory=dict
    )
    dragging_role: str | None = None
    last_accepted_world: TopologyWorld | None = None
    _drag_last_accepted_point: VertexLambertPoint | None = None

    def enter(self) -> None:
        self.active = True
        self.clear_selection()

    def clear_selection(self) -> None:
        self.element_id = None
        self.reference_world = None
        self.vertex_lambert_overrides.clear()
        self.dragging_role = None
        self.last_accepted_world = None
        self._drag_last_accepted_point = None

    def exit(self) -> None:
        self.active = False
        self.clear_selection()

    def select(self, element_id: str, reference_world: TopologyWorld) -> None:
        if not self.active:
            raise RuntimeError("Le mode deformation doit etre actif")
        self.element_id = str(element_id)
        self.reference_world = reference_world
        self.vertex_lambert_overrides.clear()
        self.dragging_role = None
        self.last_accepted_world = reference_world
        self._drag_last_accepted_point = None

    def begin_drag(self, role: str) -> None:
        if role not in {"O", "B", "L"}:
            raise ValueError(f"Role de deformation inconnu: {role!r}")
        if self.element_id is None or self.reference_world is None:
            raise RuntimeError("Aucun triangle de deformation selectionne")
        self.dragging_role = role
        self._drag_last_accepted_point = None

    def candidate_overrides(self, point: VertexLambertPoint) -> dict[str, VertexLambertPoint]:
        if self.dragging_role is None:
            raise RuntimeError("Aucun drag de deformation en cours")
        overrides = dict(self.vertex_lambert_overrides)
        overrides[self.dragging_role] = (float(point[0]), float(point[1]))
        return overrides

    def preview_overrides(self) -> dict[str, VertexLambertPoint]:
        """Retourne les overrides valides visibles, y compris le drag en cours."""
        overrides = dict(self.vertex_lambert_overrides)
        if self.dragging_role is not None and self._drag_last_accepted_point is not None:
            overrides[self.dragging_role] = self._drag_last_accepted_point
        return overrides

    def accept_candidate(
        self,
        point: VertexLambertPoint,
        candidate_world: TopologyWorld,
    ) -> None:
        if self.dragging_role is None:
            raise RuntimeError("Aucun drag de deformation en cours")
        self._drag_last_accepted_point = (float(point[0]), float(point[1]))
        self.last_accepted_world = candidate_world

    def end_drag(self) -> bool:
        if self.dragging_role is None:
            return False
        if self._drag_last_accepted_point is not None:
            self.vertex_lambert_overrides[self.dragging_role] = (
                self._drag_last_accepted_point
            )
            accepted = True
        else:
            accepted = False
        self.dragging_role = None
        self._drag_last_accepted_point = None
        return accepted

    def replace_reference_world(self, reference_world: TopologyWorld) -> None:
        if self.element_id is None:
            raise RuntimeError("Aucun triangle de deformation selectionne")
        self.reference_world = reference_world
        self.last_accepted_world = reference_world
        self.dragging_role = None
        self._drag_last_accepted_point = None
