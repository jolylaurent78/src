"""Simulation Core pure de déformation temporaire d'un triangle Catalogue."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from src.assembleur_core import (
    ScenarioAssemblage,
    TopologyConstraintGeometryError,
    TopologyWorld,
)
from src.assembleur_geometry_reference import GeometryReferenceResolver, ScenarioReference
from src.assembleur_deformation_points import WorkingPoint
from src.assembleur_scenario import (
    ScenarioHypothesis,
    materialize_triangle,
)


@dataclass(frozen=True)
class DeformationSimulationResult:
    """Résultat immutable d'un candidat de déformation."""

    accepted: bool
    world: TopologyWorld | None
    element_id: str
    vertex_lambert_overrides: Mapping[str, tuple[float, float]]
    rejection_reason: str | None = None
    warning_reason: str | None = None


@dataclass(frozen=True)
class DeformationCommitResult:
    """Publication atomique réussie d'une déformation copy-on-write."""

    world: TopologyWorld
    reference: ScenarioReference
    hypothesis: ScenarioHypothesis
    changed_element_ids: tuple[str, ...]


def _normalize_lambert_point(value) -> tuple[float, float] | None:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    return (float(point[0]), float(point[1]))


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


def _root_element_from_group_anchor(
    world: TopologyWorld,
    group_id: str,
    anchor,
) -> str:
    """Resolve the replay root from the anchor's durable physical node ID."""
    owner = world.getElementVertexFromAnyNodeId(anchor.node_id, group_id)
    if owner is None:
        raise ValueError(
            f"Ancre {anchor.anchor_id!r}: node_id non r\u00e9soluble "
            f"{anchor.node_id!r}"
        )
    root_element_id = owner["elementId"]
    if root_element_id not in world.getGroupElementIds(group_id):
        raise ValueError(
            f"Ancre {anchor.anchor_id!r}: propri\u00e9taire hors groupe "
            f"{root_element_id!r}"
        )
    return root_element_id


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
        warning_reason=None,
    )


def _replay_changed_groups_from_anchors(
    initial_world: TopologyWorld,
    working_world: TopologyWorld,
    changed_element_ids: Collection[str],
) -> str | None:
    """Rejoue les poses V2 des groupes modifies depuis leurs ancres durables.

    ``working_world`` doit deja avoir ete reconstruit et ses ancres reconciliees.
    La procedure est commune aux previews DEFORM Catalogue et COW.
    """
    source_anchor_id_by_group: dict[str, str] = {}
    for element_id in changed_element_ids:
        group_id = initial_world.get_group_of_element(element_id)
        if group_id not in source_anchor_id_by_group:
            # Un triangle isole n'a ni pose d'attachment a rejouer ni contrat
            # d'ancre DEFORM. Les groupes relies restent, eux, strictement
            # ancres comme le pipeline historique.
            if len(initial_world.getGroupElementIds(group_id)) <= 1:
                continue
            source_anchor_id_by_group[group_id] = _require_selected_group_anchor(
                initial_world, group_id
            ).anchor_id

    anchor_id_by_group: dict[str, str] = {}
    for element_id in changed_element_ids:
        group_id = working_world.get_group_of_element(element_id)
        source_group_id = initial_world.get_group_of_element(element_id)
        if source_group_id not in source_anchor_id_by_group:
            continue
        source_anchor_id = source_anchor_id_by_group[source_group_id]
        existing = anchor_id_by_group.setdefault(group_id, source_anchor_id)
        if existing != source_anchor_id:
            raise ValueError(
                f"Plusieurs ancres source pour le groupe reconstruit {group_id!r}"
            )

    for group_id, anchor_id in anchor_id_by_group.items():
        anchor = working_world.groupAnchors.get(anchor_id)
        if anchor is None:
            raise ValueError(f"Ancre perdue pendant la reconstruction: {anchor_id!r}")
        working_world.replay_group_attachment_poses(
            group_id,
            _root_element_from_group_anchor(working_world, group_id, anchor),
        )
    working_world.reconcileGroupAnchorsByNode()

    overlap_detected = False
    for group_id, anchor_id in anchor_id_by_group.items():
        anchor = working_world.groupAnchors.get(anchor_id)
        if anchor is None:
            raise ValueError(f"Ancre perdue pendant la reconstruction: {anchor_id!r}")
        if anchor.group_id != group_id:
            raise ValueError(
                f"Ancre {anchor_id!r} hors du groupe reconstruit {group_id!r}"
            )
        working_world.applyGroupAnchor(anchor_id)
        overlap_detected = overlap_detected or not working_world.is_group_contour_valid(
            group_id
        )
    return "Attention : chevauchement détecté." if overlap_detected else None


def simulate_deformation_session(
    *,
    reference_world: TopologyWorld,
    pivoted_attachment_ids: Collection[str],
) -> DeformationSimulationResult:
    """Construit le candidat DEFORM portant uniquement les pivots VE."""
    working_world = reference_world
    overlap_detected = False
    for attachment_id in sorted(pivoted_attachment_ids):
        try:
            result = working_world._build_pivot_vertex_edge_candidate_for_attachment(
                str(attachment_id), reject_overlap=False
            )
        except TopologyConstraintGeometryError as exc:
            return _rejected("", {}, str(exc))
        if result is None:
            return _rejected("", {}, f"Pivot VE DEFORM impossible: {attachment_id}")
        working_world, _attachment_id, overlap = result
        overlap_detected = overlap_detected or overlap

    return DeformationSimulationResult(
        accepted=True,
        world=working_world,
        element_id="",
        vertex_lambert_overrides=MappingProxyType({}),
        warning_reason=("Attention : chevauchement détecté." if overlap_detected else None),
    )


def simulate_occurrence_deformation(
    *,
    resolver: GeometryReferenceResolver,
    initial_world: TopologyWorld,
    occurrence_lambert_overrides: Mapping[tuple[str, str], tuple[float, float]],
) -> DeformationSimulationResult:
    """Prévisualise des déplacements isolés par occurrence Core.

    Les clés sont ``(element_id, role)`` : deux occurrences qui partagent une
    ville Catalogue restent donc indépendantes tant qu'aucun point temporaire
    commun ne les lie explicitement au commit.
    """
    if not isinstance(resolver, GeometryReferenceResolver):
        raise TypeError("simulate_occurrence_deformation attend un GeometryReferenceResolver")
    replacements: dict[str, object] = {}
    for occurrence, point in occurrence_lambert_overrides.items():
        if not isinstance(occurrence, tuple) or len(occurrence) != 2:
            raise ValueError("Occurrence DEFORM invalide")
        element_id, role = occurrence
        if role not in {"O", "B", "L"}:
            raise ValueError(f"Role de deformation inconnu: {role!r}")
        normalized = _normalize_lambert_point(point)
        if normalized is None:
            return _rejected(element_id, {}, "Override Lambert candidat invalide")
        element = initial_world.elements.get(element_id)
        if element is None or not element.source_triangle_id:
            raise ValueError(f"Element topologique sans source_triangle_id: {element_id!r}")
        roles = replacements.setdefault(element_id, {})
        roles[role] = normalized

    if not replacements:
        return _rejected("", {}, "Aucune occurrence de ville a deformer")

    working_world = initial_world.clonePhysicalState()
    changed_element_ids = []
    for element_id, role_overrides in replacements.items():
        source_triangle_id = working_world.elements[element_id].source_triangle_id
        replacement = materialize_triangle(
            resolver,
            source_triangle_id,
            vertex_lambert_overrides=role_overrides,
        )
        working_world.replace_element_intrinsic_geometry(element_id, replacement)
        changed_element_ids.append(element_id)
    working_world.rebuild_from_attachments()
    working_world.reconcileGroupAnchorsByNode()
    try:
        warning_reason = _replay_changed_groups_from_anchors(
            initial_world, working_world, changed_element_ids
        )
    except TopologyConstraintGeometryError as exc:
        return _rejected("", {}, str(exc))
    return DeformationSimulationResult(
        accepted=True,
        world=working_world,
        element_id="",
        vertex_lambert_overrides=MappingProxyType({}),
        warning_reason=warning_reason,
    )


def commit_deformation_copy_on_write(
    *,
    catalogue: Catalogue,
    scenario: ScenarioAssemblage,
    preview_world: TopologyWorld,
    working_points: Mapping[str, WorkingPoint],
    base_reference: ScenarioReference | None = None,
    base_hypothesis: ScenarioHypothesis | None = None,
) -> DeformationCommitResult:
    """Prépare puis publie conceptuellement une définition locale COW.

    Cette fonction ne modifie aucun argument. Son résultat ne devient réel que
    lorsque l'appelant affecte simultanément la référence, l'hypothèse et le
    monde du scénario après retour sans exception.
    """
    if not isinstance(scenario.reference, ScenarioReference):
        raise TypeError("Le scénario doit posséder un ScenarioReference")
    if scenario.hypothesis is None:
        raise ValueError("Le scénario doit posséder une ScenarioHypothesis")
    if base_reference is not None and not isinstance(base_reference, ScenarioReference):
        raise TypeError("base_reference DEFORM doit etre un ScenarioReference")
    if base_hypothesis is not None and not isinstance(base_hypothesis, ScenarioHypothesis):
        raise TypeError("base_hypothesis DEFORM doit etre une ScenarioHypothesis")
    source_reference = base_reference or scenario.reference
    source_hypothesis = base_hypothesis or scenario.hypothesis
    reference = source_reference.clone()
    hypothesis = source_hypothesis.clone()
    source_resolver = GeometryReferenceResolver(catalogue, source_reference)
    candidate_resolver = GeometryReferenceResolver(catalogue, reference)
    normalized_points: dict[str, tuple[tuple[float, float], set[tuple[str, str]]] ] = {}
    occurrence_point_ids: dict[tuple[str, str], str] = {}
    for point_id, working_point in working_points.items():
        if not isinstance(working_point, WorkingPoint):
            raise TypeError(f"WorkingPoint DEFORM invalide : {point_id!r}")
        if working_point.point_id != point_id:
            raise ValueError(f"WorkingPoint DEFORM incoherent : {point_id!r}")
        normalized = _normalize_lambert_point(working_point.lambert_xy)
        if normalized is None:
            raise ValueError(f"Point temporaire DEFORM invalide : {point_id!r}")
        occurrences = set(working_point.occurrences)
        if not occurrences:
            raise ValueError(f"WorkingPoint DEFORM sans occurrence : {point_id!r}")
        normalized_points[point_id] = (normalized, occurrences)
        for occurrence in sorted(occurrences):
            if occurrence in occurrence_point_ids:
                raise ValueError(f"Occurrence DEFORM rattachee a deux points : {occurrence!r}")
            occurrence_point_ids[occurrence] = point_id

    occurrences_by_element: dict[str, dict[str, str]] = {}
    for occurrence, point_id in occurrence_point_ids.items():
        if not isinstance(occurrence, tuple) or len(occurrence) != 2:
            raise ValueError("Occurrence DEFORM invalide")
        element_id, role = occurrence
        if role not in {"O", "B", "L"}:
            raise ValueError(f"Role de deformation inconnu: {role!r}")
        if point_id not in normalized_points:
            raise ValueError(f"Point temporaire DEFORM inconnu : {point_id!r}")
        if element_id not in preview_world.elements:
            raise ValueError(f"Element topologique inconnu: {element_id!r}")
        occurrences_by_element.setdefault(element_id, {})[role] = point_id

    if not occurrences_by_element and not preview_world.elements:
        raise ValueError("Aucune déformation à valider")

    shared_city_by_point_id: dict[str, str] = {}
    changed_element_ids = []
    local_triangle_ref_id_by_element: dict[str, str] = {}
    for element_id, role_point_ids in occurrences_by_element.items():
        element = preview_world.elements[element_id]
        triangle_ref_id = element.source_triangle_id
        if not triangle_ref_id:
            raise ValueError(f"Element topologique sans source_triangle_id: {element_id!r}")
        triangle = candidate_resolver.resolve_triangle(triangle_ref_id)
        if triangle.origin == "catalogue":
            rank = hypothesis.get_rank_for_triangle_ref(triangle_ref_id)
            local_triangle = reference.create_triangle(
                triangle.note,
                triangle.opening_city_ref_id,
                triangle.base_city_ref_id,
                triangle.light_city_ref_id,
                catalogue_source_triangle_id=triangle.ref_id,
            )
            hypothesis.triangle_ids_by_rank[rank - 1] = local_triangle.triangle_ref_id
        else:
            local_triangle = reference.triangles[triangle.ref_id]
        local_triangle_ref_id_by_element[element_id] = local_triangle.triangle_ref_id

        role_city_attributes = {
            "O": "opening_city_ref_id",
            "B": "base_city_ref_id",
            "L": "light_city_ref_id",
        }
        for role, point_id in role_point_ids.items():
            current_city_ref_id = getattr(local_triangle, role_city_attributes[role])
            shared_city_ref_id = shared_city_by_point_id.get(point_id)
            if shared_city_ref_id is None:
                point = normalized_points[point_id][0]
                if current_city_ref_id in reference.cities:
                    city = reference.cities[current_city_ref_id]
                    latitude, longitude = candidate_resolver.lambert_to_geographic(*point)
                    city.latitude = latitude
                    city.longitude = longitude
                    shared_city_ref_id = city.city_ref_id
                else:
                    source_city = source_resolver.resolve_city(current_city_ref_id)
                    latitude, longitude = candidate_resolver.lambert_to_geographic(*point)
                    city = reference.create_city(
                        f"Temp {source_city.name}",
                        latitude,
                        longitude,
                        catalogue_source_city_id=(
                            source_city.catalogue_source_city_id or source_city.ref_id
                        ),
                    )
                    shared_city_ref_id = city.city_ref_id
                shared_city_by_point_id[point_id] = shared_city_ref_id
            setattr(local_triangle, role_city_attributes[role], shared_city_ref_id)
        changed_element_ids.append(element_id)

    # Une ville scenario qui n'est plus referencee apres un changement de
    # WorkingPoint ne constitue pas un historique : elle est retiree du nouvel
    # etat de reference.
    referenced_city_ids = {
        city_ref_id
        for triangle in reference.triangles.values()
        for city_ref_id in (
            triangle.opening_city_ref_id,
            triangle.base_city_ref_id,
            triangle.light_city_ref_id,
        )
    }
    reference.cities = {
        city_ref_id: city
        for city_ref_id, city in reference.cities.items()
        if city_ref_id in referenced_city_ids
    }
    hypothesis.validate(candidate_resolver)

    working_world = preview_world.clonePhysicalState()
    for element_id in changed_element_ids:
        triangle_ref_id = working_world.elements[element_id].source_triangle_id
        if element_id in local_triangle_ref_id_by_element:
            # L'élément preview portait déjà STRI-*; il reste identique.
            local_triangle_ref_id = local_triangle_ref_id_by_element[element_id]
        else:
            # Le seul STRI créé pour cet élément est celui qui porte la source TRI.
            local_triangle_ref_id = next(
                triangle.triangle_ref_id
                for triangle in reference.triangles.values()
                if triangle.catalogue_source_triangle_id == triangle_ref_id
                and triangle.triangle_ref_id not in scenario.reference.triangles
            )
        replacement = materialize_triangle(candidate_resolver, local_triangle_ref_id)
        working_world.replace_element_materialized_definition(element_id, replacement)
    working_world.rebuild_from_attachments()
    working_world.reconcileGroupAnchorsByNode()
    for anchor in working_world.groupAnchors.values():
        working_world.applyGroupAnchor(anchor.anchor_id)
    errors = working_world.validate_world()
    if errors:
        raise ValueError("Commit DEFORM invalide : " + " ; ".join(str(error) for error in errors))
    return DeformationCommitResult(
        world=working_world,
        reference=reference,
        hypothesis=hypothesis,
        changed_element_ids=tuple(changed_element_ids),
    )
