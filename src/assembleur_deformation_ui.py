"""Etat pur du mode interactif de deformation.

Ce module ne depend pas de Tk.  Il isole la courte machine d'etat qui relie
les gestes de l'IHM au moteur de simulation pur.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.assembleur_core import TopologyWorld


VertexLambertPoint = tuple[float, float]
DeformationOccurrence = tuple[str, str]


@dataclass
class DeformationUiState:
    """Etat temporaire, explicitement distinct du scenario actif."""

    active: bool = False
    element_id: str | None = None
    reference_world: TopologyWorld | None = None
    city_lambert_overrides: dict[str, VertexLambertPoint] = field(
        default_factory=dict
    )
    dragging_role: str | None = None
    dragging_city_id: str | None = None
    last_accepted_world: TopologyWorld | None = None
    modified_occurrences: list[DeformationOccurrence] = field(default_factory=list)
    selected_occurrence: DeformationOccurrence | None = None
    pivoted_attachment_ids: set[str] = field(default_factory=set)
    _drag_last_accepted_point: VertexLambertPoint | None = None

    def enter(self) -> None:
        self.active = True
        self.clear_session()

    def clear_session(self) -> None:
        self.element_id = None
        self.reference_world = None
        self.city_lambert_overrides.clear()
        self.dragging_role = None
        self.dragging_city_id = None
        self.last_accepted_world = None
        self._drag_last_accepted_point = None
        self.modified_occurrences.clear()
        self.selected_occurrence = None
        self.pivoted_attachment_ids.clear()

    def toggle_pivoted_attachment(self, attachment_id: str) -> None:
        attachment_id = str(attachment_id)
        if not attachment_id:
            raise ValueError("attachment_id vide")
        if attachment_id in self.pivoted_attachment_ids:
            self.pivoted_attachment_ids.remove(attachment_id)
        else:
            self.pivoted_attachment_ids.add(attachment_id)

    def clear_selection(self) -> None:
        """Compatibilite locale : vider toute la session de deformation."""
        self.clear_session()

    def exit(self) -> None:
        self.active = False
        self.clear_session()

    def select(self, element_id: str, reference_world: TopologyWorld) -> None:
        if not self.active:
            raise RuntimeError("Le mode deformation doit etre actif")
        self.element_id = str(element_id)
        if self.reference_world is None:
            self.reference_world = reference_world
        self.dragging_role = None
        self.dragging_city_id = None
        if self.last_accepted_world is None:
            self.last_accepted_world = reference_world
        self._drag_last_accepted_point = None

    def begin_drag(self, role: str, city_id: str) -> None:
        if role not in {"O", "B", "L"}:
            raise ValueError(f"Role de deformation inconnu: {role!r}")
        if self.element_id is None or self.reference_world is None:
            raise RuntimeError("Aucun triangle de deformation selectionne")
        self.dragging_role = role
        self.dragging_city_id = str(city_id)
        self._drag_last_accepted_point = None

    def candidate_city_overrides(
        self,
        point: VertexLambertPoint,
    ) -> dict[str, VertexLambertPoint]:
        if self.dragging_city_id is None:
            raise RuntimeError("Aucun drag de deformation en cours")
        overrides = dict(self.city_lambert_overrides)
        overrides[self.dragging_city_id] = (float(point[0]), float(point[1]))
        return overrides

    def preview_city_overrides(self) -> dict[str, VertexLambertPoint]:
        """Retourne les overrides de villes visibles, y compris le drag en cours."""
        overrides = dict(self.city_lambert_overrides)
        if self.dragging_city_id is not None and self._drag_last_accepted_point is not None:
            overrides[self.dragging_city_id] = self._drag_last_accepted_point
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

    def end_drag(self, city_occurrences: tuple[DeformationOccurrence, ...]) -> bool:
        if self.dragging_role is None:
            return False
        if self._drag_last_accepted_point is not None:
            if self.dragging_city_id is None or self.element_id is None:
                raise RuntimeError("Drag de deformation incomplet")
            self.city_lambert_overrides[self.dragging_city_id] = self._drag_last_accepted_point
            occurrence = (self.element_id, self.dragging_role)
            for city_occurrence in city_occurrences:
                if city_occurrence not in self.modified_occurrences:
                    self.modified_occurrences.append(city_occurrence)
            self.selected_occurrence = occurrence
            accepted = True
        else:
            accepted = False
        self.dragging_role = None
        self.dragging_city_id = None
        self._drag_last_accepted_point = None
        return accepted

    def replace_reference_world(self, reference_world: TopologyWorld) -> None:
        if self.element_id is None:
            raise RuntimeError("Aucun triangle de deformation selectionne")
        self.reference_world = reference_world
        self.last_accepted_world = reference_world
        self.dragging_role = None
        self.dragging_city_id = None
        self._drag_last_accepted_point = None

    def modified_roles_for_element(self, element_id: str) -> set[str]:
        return {
            role
            for occurrence_element_id, role in self.modified_occurrences
            if occurrence_element_id == element_id
        }

    def select_occurrence(self, element_id: str, role: str) -> None:
        if role not in {"O", "B", "L"}:
            raise ValueError(f"Role de deformation inconnu: {role!r}")
        self.selected_occurrence = (str(element_id), role)
