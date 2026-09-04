"""Etat pur du mode interactif de deformation.

Ce module ne depend pas de Tk.  Il isole la courte machine d'etat qui relie
les gestes de l'IHM au moteur de simulation pur.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.assembleur_core import TopologyWorld
from src.assembleur_deformation_points import (
    DeformationOccurrence,
    VertexLambertPoint,
    WorkingPoint,
)
from src.assembleur_geometry_reference import ScenarioReference
from src.assembleur_scenario import ScenarioHypothesis


@dataclass
class DeformationUiState:
    """Etat temporaire, explicitement distinct du scenario actif."""

    active: bool = False
    element_id: str | None = None
    reference_world: TopologyWorld | None = None
    dragging_role: str | None = None
    last_accepted_world: TopologyWorld | None = None
    modified_occurrences: list[DeformationOccurrence] = field(default_factory=list)
    selected_occurrence: DeformationOccurrence | None = None
    pivoted_attachment_ids: set[str] = field(default_factory=set)
    working_points: dict[str, WorkingPoint] = field(default_factory=dict)
    working_reference: ScenarioReference | None = None
    working_hypothesis: ScenarioHypothesis | None = None
    working_point_names: dict[str, str] = field(default_factory=dict)
    dirty: bool = False
    _drag_last_accepted_point: VertexLambertPoint | None = None

    def enter(self) -> None:
        self.active = True
        self.clear_session()

    def clear_session(self) -> None:
        self.element_id = None
        self.reference_world = None
        self.dragging_role = None
        self.last_accepted_world = None
        self._drag_last_accepted_point = None
        self.modified_occurrences.clear()
        self.selected_occurrence = None
        self.pivoted_attachment_ids.clear()
        self.working_points.clear()
        self.working_reference = None
        self.working_hypothesis = None
        self.working_point_names.clear()
        self.dirty = False

    def _new_temporary_point_id(self) -> str:
        sequence = 1
        while f"TMP-{sequence:04d}" in self.working_points:
            sequence += 1
        return f"TMP-{sequence:04d}"

    def working_point_for_occurrence(
        self, occurrence: DeformationOccurrence
    ) -> WorkingPoint | None:
        return next(
            (point for point in self.working_points.values()
             if occurrence in point.occurrences),
            None,
        )

    def ensure_working_point(
        self,
        occurrence: DeformationOccurrence,
        lambert_xy: VertexLambertPoint,
        linked_occurrences: tuple[DeformationOccurrence, ...] = (),
    ) -> WorkingPoint:
        """Initialise une identite explicite depuis le lien DEFORM existant."""
        point = self.working_point_for_occurrence(occurrence)
        if point is not None:
            return point
        point = WorkingPoint(
            self._new_temporary_point_id(),
            (float(lambert_xy[0]), float(lambert_xy[1])),
            {occurrence, *linked_occurrences},
        )
        # Une occurrence deja rattachee a un autre point n'est jamais fusionnee
        # implicitement : seul un appel explicite de partage peut la deplacer.
        point.occurrences = {
            item for item in point.occurrences
            if self.working_point_for_occurrence(item) is None
        }
        self.working_points[point.point_id] = point
        return point

    def occurrence_lambert_overrides(
        self,
        candidate_point: tuple[str, VertexLambertPoint] | None = None,
    ) -> dict[DeformationOccurrence, VertexLambertPoint]:
        overrides: dict[DeformationOccurrence, VertexLambertPoint] = {}
        for point in self.working_points.values():
            coordinate = point.lambert_xy
            if candidate_point is not None and point.point_id == candidate_point[0]:
                coordinate = candidate_point[1]
            for occurrence in point.occurrences:
                overrides[occurrence] = coordinate
        return overrides

    def candidate_occurrence_overrides(self, point: VertexLambertPoint) -> dict[DeformationOccurrence, VertexLambertPoint]:
        if self.element_id is None or self.dragging_role is None:
            raise RuntimeError("Aucun drag de deformation en cours")
        working_point = self.working_point_for_occurrence(
            (self.element_id, self.dragging_role)
        )
        if working_point is None:
            raise RuntimeError("WorkingPoint DEFORM absent")
        return self.occurrence_lambert_overrides(
            (working_point.point_id, (float(point[0]), float(point[1])))
        )

    def accept_occurrence_candidate(self, point: VertexLambertPoint, candidate_world: TopologyWorld) -> None:
        if self.element_id is None or self.dragging_role is None:
            raise RuntimeError("Aucun drag de deformation en cours")
        working_point = self.working_point_for_occurrence(
            (self.element_id, self.dragging_role)
        )
        if working_point is None:
            raise RuntimeError("WorkingPoint DEFORM absent")
        working_point.lambert_xy = (float(point[0]), float(point[1]))
        self._drag_last_accepted_point = (float(point[0]), float(point[1]))
        self.last_accepted_world = candidate_world
        self.dirty = True

    def share_working_point(self, source: DeformationOccurrence, destination: DeformationOccurrence) -> None:
        point = self.working_point_for_occurrence(source)
        if point is None:
            raise ValueError("Point temporaire source absent")
        previous = self.working_point_for_occurrence(destination)
        if previous is not None:
            previous.occurrences.discard(destination)
            if not previous.occurrences:
                self.working_points.pop(previous.point_id)
                self.working_point_names.pop(previous.point_id, None)
        point.occurrences.add(destination)
        self.dirty = True

    def set_shared_working_point(
        self,
        occurrences: tuple[DeformationOccurrence, ...],
        point: VertexLambertPoint,
    ) -> None:
        """Lie explicitement plusieurs occurrences au même point temporaire."""
        if not occurrences:
            raise ValueError("Aucune occurrence à lier")
        point = WorkingPoint(
            self._new_temporary_point_id(),
            (float(point[0]), float(point[1])),
            set(),
        )
        for occurrence in occurrences:
            old_point = self.working_point_for_occurrence(occurrence)
            if old_point is not None:
                old_point.occurrences.discard(occurrence)
                if not old_point.occurrences:
                    self.working_points.pop(old_point.point_id)
                    self.working_point_names.pop(old_point.point_id, None)
            point.occurrences.add(occurrence)
            if occurrence not in self.modified_occurrences:
                self.modified_occurrences.append(occurrence)
        self.working_points[point.point_id] = point
        self.selected_occurrence = occurrences[0]
        self.dirty = True

    def end_occurrence_drag(self) -> bool:
        """Finalise un drag COW sans propager l'override aux homonymes."""
        if self.dragging_role is None:
            return False
        if self._drag_last_accepted_point is None or self.element_id is None:
            accepted = False
        else:
            occurrence = (self.element_id, self.dragging_role)
            point = self.working_point_for_occurrence(occurrence)
            if point is None:
                raise RuntimeError("WorkingPoint DEFORM absent")
            for linked_occurrence in sorted(point.occurrences):
                if linked_occurrence not in self.modified_occurrences:
                    self.modified_occurrences.append(linked_occurrence)
            self.selected_occurrence = occurrence
            accepted = True
        self.dragging_role = None
        self._drag_last_accepted_point = None
        return accepted

    def restore_working_point(self, occurrence: DeformationOccurrence) -> set[DeformationOccurrence]:
        """Abandonne le point courant, sans aucune reference historique."""
        point = self.working_point_for_occurrence(occurrence)
        if point is None:
            return set()
        restored = set(point.occurrences)
        self.working_points.pop(point.point_id)
        self.working_point_names.pop(point.point_id, None)
        self.modified_occurrences = [
            item for item in self.modified_occurrences if item not in restored
        ]
        self.dirty = bool(self.working_points or self.pivoted_attachment_ids)
        return restored

    def rebase_after_commit(
        self,
        reference_world: TopologyWorld,
        reference: ScenarioReference | None = None,
        hypothesis: ScenarioHypothesis | None = None,
    ) -> None:
        self.reference_world = reference_world
        self.last_accepted_world = reference_world
        self.working_reference = reference.clone() if reference is not None else None
        self.working_hypothesis = hypothesis.clone() if hypothesis is not None else None
        self.working_points.clear()
        self.working_point_names.clear()
        self.pivoted_attachment_ids.clear()
        self.modified_occurrences.clear()
        self.selected_occurrence = None
        self.dragging_role = None
        self._drag_last_accepted_point = None
        self.dirty = False

    def toggle_pivoted_attachment(self, attachment_id: str) -> None:
        attachment_id = str(attachment_id)
        if not attachment_id:
            raise ValueError("attachment_id vide")
        if attachment_id in self.pivoted_attachment_ids:
            self.pivoted_attachment_ids.remove(attachment_id)
        else:
            self.pivoted_attachment_ids.add(attachment_id)
        self.dirty = bool(self.working_points or self.pivoted_attachment_ids)

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
        if self.last_accepted_world is None:
            self.last_accepted_world = reference_world
        self._drag_last_accepted_point = None

    def begin_drag(self, role: str) -> None:
        if role not in {"O", "B", "L"}:
            raise ValueError(f"Role de deformation inconnu: {role!r}")
        if self.element_id is None or self.reference_world is None:
            raise RuntimeError("Aucun triangle de deformation selectionne")
        self.dragging_role = role
        self._drag_last_accepted_point = None

    def replace_reference_world(self, reference_world: TopologyWorld) -> None:
        if self.element_id is None:
            raise RuntimeError("Aucun triangle de deformation selectionne")
        self.reference_world = reference_world
        self.last_accepted_world = reference_world
        self.dragging_role = None
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
