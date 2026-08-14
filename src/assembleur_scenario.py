"""Modèles métier propres aux scénarios, indépendants de Tk et de la topologie."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.assembleur_catalogue import Catalogue, HypothesisTemplate
from src.assembleur_core import (
    ScenarioAssemblage,
    TopologyAttachment,
    TopologyElement,
    TopologyFeatureRef,
    TopologyFeatureType,
    build_topology_element_from_catalogue_triangle,
)
from src.assembleur_edge_mapping import compute_edge_edge_pose


@dataclass
class ScenarioHypothesis:
    """Hypothèse complète, possédée par un scénario manuel."""

    triangle_ids_by_rank: list[str]
    source_template_id: str | None = None

    def validate(self, catalogue: Catalogue) -> None:
        ranks = self.triangle_ids_by_rank
        if len(ranks) != 32:
            raise ValueError("Une hypothèse de scénario doit contenir exactement 32 rangs.")
        if any(not isinstance(triangle_id, str) for triangle_id in ranks):
            raise ValueError("Une hypothèse de scénario ne peut contenir aucun rang vide.")
        if len(set(ranks)) != len(ranks):
            raise ValueError("Un triangle ne peut pas être utilisé dans plusieurs rangs.")
        for index in range(0, 32, 2):
            try:
                odd = catalogue.get_triangle(ranks[index])
                even = catalogue.get_triangle(ranks[index + 1])
            except KeyError as exc:
                raise ValueError(
                    "L'hypothèse du scénario référence un triangle Catalogue absent : "
                    f"{exc.args[0]}"
                ) from exc
            if odd.base_city_id != even.base_city_id:
                raise ValueError(f"Les rangs {index + 1} et {index + 2} doivent utiliser la même base.")

    def set_ranks(self, catalogue: Catalogue, triangle_ids_by_rank: list[str]) -> None:
        preview = list(triangle_ids_by_rank)
        candidate = ScenarioHypothesis(preview, self.source_template_id)
        candidate.validate(catalogue)
        self.triangle_ids_by_rank[:] = preview

    def clone(self) -> "ScenarioHypothesis":
        return ScenarioHypothesis(list(self.triangle_ids_by_rank), self.source_template_id)


class HypothesisImpact(Enum):
    """Niveau de migration requis pour appliquer une hypothèse modifiée."""

    NONE = "NONE"
    REPLAY = "REPLAY"
    DETACH = "DETACH"
    RESET = "RESET"


@dataclass(frozen=True)
class HypothesisRankChange:
    """Différence d'un rang, exprimée sans référence à Tk ou au Core."""

    rank: int
    old_triangle_id: str
    new_triangle_id: str
    impact: HypothesisImpact


@dataclass(frozen=True)
class ScenarioHypothesisChangePlan:
    """Plan pur consommable plus tard par les migrations RESET/DETACH/REPLAY."""

    template_changed: bool
    global_impact: HypothesisImpact
    rank_changes: tuple[HypothesisRankChange, ...]


@dataclass(frozen=True)
class HypothesisTopologyApplyResult:
    """Résultat du commit Core-first d'une hypothèse de scénario manuel."""

    plan: ScenarioHypothesisChangePlan
    replaced_element_ids: tuple[str, ...]
    replayed_attachment_count: int


@dataclass(frozen=True)
class _ReplacedElement:
    rank: int
    old_triangle_id: str
    old_element_id: str
    rotation: np.ndarray
    translation: np.ndarray
    mirrored: bool


def _rank_change_impact(catalogue: Catalogue, old_triangle_id: str, new_triangle_id: str) -> HypothesisImpact:
    old_triangle = catalogue.get_triangle(old_triangle_id)
    new_triangle = catalogue.get_triangle(new_triangle_id)
    if (
        old_triangle.opening_city_id == new_triangle.opening_city_id
        and old_triangle.base_city_id == new_triangle.base_city_id
        and old_triangle.light_city_id != new_triangle.light_city_id
    ):
        return HypothesisImpact.REPLAY
    return HypothesisImpact.DETACH


def analyze_hypothesis_change(
    catalogue: Catalogue,
    old_hypothesis: ScenarioHypothesis,
    new_hypothesis: ScenarioHypothesis,
) -> ScenarioHypothesisChangePlan:
    """Compare deux hypothèses validées sans muter ni Catalogue ni scénario."""
    old_hypothesis.validate(catalogue)
    new_hypothesis.validate(catalogue)
    changes = tuple(
        HypothesisRankChange(
            rank=index,
            old_triangle_id=old_triangle_id,
            new_triangle_id=new_triangle_id,
            impact=_rank_change_impact(catalogue, old_triangle_id, new_triangle_id),
        )
        for index, (old_triangle_id, new_triangle_id) in enumerate(
            zip(old_hypothesis.triangle_ids_by_rank, new_hypothesis.triangle_ids_by_rank),
            start=1,
        )
        if old_triangle_id != new_triangle_id
    )
    template_changed = old_hypothesis.source_template_id != new_hypothesis.source_template_id
    if any(change.impact is HypothesisImpact.DETACH for change in changes):
        global_impact = HypothesisImpact.DETACH
    elif any(change.impact is HypothesisImpact.REPLAY for change in changes):
        global_impact = HypothesisImpact.REPLAY
    else:
        global_impact = HypothesisImpact.NONE
    return ScenarioHypothesisChangePlan(template_changed, global_impact, changes)


def create_hypothesis_from_template(catalogue: Catalogue, template: HypothesisTemplate) -> ScenarioHypothesis:
    """Instancie une copie autonome des rangs d'un template Catalogue."""
    hypothesis = ScenarioHypothesis(list(template.triangle_ids_by_rank), template.template_id)
    hypothesis.validate(catalogue)
    return hypothesis


def create_default_scenario_hypothesis(catalogue: Catalogue) -> ScenarioHypothesis:
    """Instancie l'hypothèse du scénario depuis le template Catalogue par défaut."""
    return create_hypothesis_from_template(catalogue, catalogue.require_valid_default_template())


def materialize_catalogue_triangle(
    catalogue: Catalogue,
    triangle_id: str,
) -> TopologyElement:
    """Resolves a Catalogue triangle before passing simple data to the Core."""
    triangle = catalogue.get_triangle(triangle_id)
    opening = catalogue.get_city(triangle.opening_city_id)
    base = catalogue.get_city(triangle.base_city_id)
    light = catalogue.get_city(triangle.light_city_id)
    return build_topology_element_from_catalogue_triangle(
        triangle_id=triangle.triangle_id,
        opening_name=opening.name,
        base_name=base.name,
        light_name=light.name,
        opening_lambert_xy=catalogue.get_city_lambert(opening.city_id),
        base_lambert_xy=catalogue.get_city_lambert(base.city_id),
        light_lambert_xy=catalogue.get_city_lambert(light.city_id),
    )


def _single_materialized_element_id(world, triangle_id: str) -> str | None:
    matches = [
        element.element_id
        for element in world.elements.values()
        if element.source_triangle_id == triangle_id
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Triangle Catalogue {triangle_id!r} matérialisé plusieurs fois dans le scénario"
        )
    return matches[0] if matches else None


def _edge_index_for_vertex_types(element: TopologyElement, first: str, second: str) -> int:
    wanted = {first, second}
    for edge in element.edges:
        edge_types = {
            element.vertex_types[edge.v_start.vertex_index],
            element.vertex_types[edge.v_end.vertex_index],
        }
        if edge_types == wanted:
            return int(edge.edge_index)
    raise ValueError(
        f"Élément {element.element_id!r}: arête {first}{second} introuvable"
    )


def _points_by_vertex_type(element: TopologyElement) -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {}
    for index, vertex_type in enumerate(element.vertex_types):
        point = element.vertex_local_xy.get(index)
        if point is None:
            raise ValueError(
                f"Élément {element.element_id!r}: coordonnées locales absentes pour {vertex_type}"
            )
        points[vertex_type] = np.asarray(point, dtype=float)
    if set(points) != {"O", "B", "L"}:
        raise ValueError(f"Élément {element.element_id!r}: triangle O/B/L attendu")
    return points


def _has_ob_edge_edge_attachment(world, first_element_id: str, second_element_id: str) -> bool:
    expected = {first_element_id, second_element_id}
    for attachment in world.attachments.values():
        if attachment.kind != "edge-edge":
            continue
        if {
            attachment.feature_a.element_id,
            attachment.feature_b.element_id,
        } != expected:
            continue
        first = world.get_edge(attachment.feature_a.element_id, attachment.feature_a.index)
        second = world.get_edge(attachment.feature_b.element_id, attachment.feature_b.index)
        first_types = {
            world.elements[first.element_id].vertex_types[first.v_start.vertex_index],
            world.elements[first.element_id].vertex_types[first.v_end.vertex_index],
        }
        second_types = {
            world.elements[second.element_id].vertex_types[second.v_start.vertex_index],
            world.elements[second.element_id].vertex_types[second.v_end.vertex_index],
        }
        if first_types == {"O", "B"} and second_types == {"O", "B"}:
            return True
    return False


def _replay_ob_edge_edge_attachment(world, mobile_element_id: str, target_element_id: str) -> None:
    mobile = world.elements[mobile_element_id]
    target = world.elements[target_element_id]
    mobile_points = _points_by_vertex_type(mobile)
    target_local_points = _points_by_vertex_type(target)
    target_points = {
        vertex_type: world.elementLocalToWorld(target_element_id, point)
        for vertex_type, point in target_local_points.items()
    }
    _old_rotation, _old_translation, mobile_mirrored = world.getElementPose(
        mobile_element_id
    )
    pose = compute_edge_edge_pose(
        mobile_points,
        "O",
        "B",
        target_points,
        "O",
        "B",
        mobile_mirrored=mobile_mirrored,
    )
    world.setElementPose(
        mobile_element_id,
        pose.rotation,
        pose.translation,
        mirrored=pose.mirrored,
    )
    attachment = TopologyAttachment(
        attachment_id=None,
        kind="edge-edge",
        feature_a=TopologyFeatureRef(
            TopologyFeatureType.EDGE,
            mobile_element_id,
            _edge_index_for_vertex_types(mobile, "O", "B"),
        ),
        feature_b=TopologyFeatureRef(
            TopologyFeatureType.EDGE,
            target_element_id,
            _edge_index_for_vertex_types(target, "O", "B"),
        ),
        params={
            "mapping": pose.mapping,
            "incident_edge_by_element": {
                mobile_element_id: "OB",
                target_element_id: "OB",
            },
        },
        source="manual",
    )
    world.apply_attachments([attachment])


def apply_hypothesis_change_to_manual_scenario(
    catalogue: Catalogue,
    scenario: ScenarioAssemblage,
    draft_hypothesis: ScenarioHypothesis,
) -> HypothesisTopologyApplyResult:
    """Applique une modification d'hypothèse à un manuel non vide, atomiquement.

    Le monde courant n'est jamais modifié pendant le calcul : toutes les
    opérations Core sont réalisées sur un clone physique, qui n'est publié sur
    le scénario qu'après reconstruction et validation complètes.
    """
    if scenario.source_type != "manual":
        raise ValueError("Seul un scénario manuel peut recevoir une ScenarioHypothesis modifiée.")
    if scenario.hypothesis is None:
        raise ValueError("ScenarioHypothesis absente du scénario manuel actif")

    candidate = draft_hypothesis.clone()
    candidate.validate(catalogue)
    plan = analyze_hypothesis_change(catalogue, scenario.hypothesis, candidate)
    if not plan.rank_changes:
        scenario.hypothesis = candidate
        return HypothesisTopologyApplyResult(plan, (), 0)

    source_world = scenario.topoWorld
    for change in plan.rank_changes:
        old_element_id = _single_materialized_element_id(source_world, change.old_triangle_id)
        if old_element_id is None:
            continue
        for anchor in source_world.groupAnchors.values():
            if anchor.node_id.startswith(f"{old_element_id}:"):
                raise ValueError(
                    "Modification d'hypothèse impossible : l'élément à remplacer "
                    f"{old_element_id!r} porte l'ancre {anchor.anchor_id!r}."
                )

    working_world = source_world.clonePhysicalState()
    surviving_anchors = tuple(
        (anchor.anchor_id, anchor.beacon_id, anchor.node_id)
        for anchor in working_world.groupAnchors.values()
    )
    old_by_rank = scenario.hypothesis.triangle_ids_by_rank
    changed_by_rank = {change.rank: change for change in plan.rank_changes}
    replacements: dict[int, _ReplacedElement] = {}

    for change in plan.rank_changes:
        old_element_id = _single_materialized_element_id(working_world, change.old_triangle_id)
        if old_element_id is None:
            continue
        rotation, translation, mirrored = working_world.getElementPose(old_element_id)
        replacements[change.rank] = _ReplacedElement(
            rank=change.rank,
            old_triangle_id=change.old_triangle_id,
            old_element_id=old_element_id,
            rotation=np.array(rotation, dtype=float),
            translation=np.array(translation, dtype=float),
            mirrored=bool(mirrored),
        )

    replay_links: dict[tuple[int, int], tuple[int, int]] = {}
    for change in plan.rank_changes:
        if change.impact is not HypothesisImpact.REPLAY or change.rank not in replacements:
            continue
        companion_rank = change.rank + 1 if change.rank % 2 else change.rank - 1
        companion_old_id = _single_materialized_element_id(
            working_world, old_by_rank[companion_rank - 1]
        )
        if companion_old_id is None:
            continue
        if _has_ob_edge_edge_attachment(
            working_world, replacements[change.rank].old_element_id, companion_old_id
        ):
            pair_key = tuple(sorted((change.rank, companion_rank)))
            companion_change = changed_by_rank.get(companion_rank)
            if companion_change is None:
                replay_links[pair_key] = (change.rank, companion_rank)
            elif companion_change.impact is HypothesisImpact.REPLAY:
                # Si les deux extrémités sont remplacées, conserver la plus
                # basse comme cible et ne rejouer l'attachment qu'une fois.
                replay_links[pair_key] = (max(pair_key), min(pair_key))

    working_world.removeElementsAndRebuild(
        [replacement.old_element_id for replacement in replacements.values()]
    )

    new_element_by_rank: dict[int, str] = {}
    for rank, replacement in replacements.items():
        new_element = materialize_catalogue_triangle(
            catalogue, candidate.triangle_ids_by_rank[rank - 1]
        )
        working_world.add_element_as_new_group(new_element)
        working_world.setElementPose(
            new_element.element_id,
            replacement.rotation,
            replacement.translation,
            mirrored=replacement.mirrored,
        )
        new_element_by_rank[rank] = new_element.element_id

    replayed_attachment_count = 0
    for mobile_rank, target_rank in (
        replay_links[key] for key in sorted(replay_links)
    ):
        mobile_element_id = new_element_by_rank.get(mobile_rank)
        if mobile_element_id is None:
            mobile_element_id = _single_materialized_element_id(
                working_world, candidate.triangle_ids_by_rank[mobile_rank - 1]
            )
        target_element_id = new_element_by_rank.get(target_rank)
        if target_element_id is None:
            target_element_id = _single_materialized_element_id(
                working_world, candidate.triangle_ids_by_rank[target_rank - 1]
            )
        if mobile_element_id is None or target_element_id is None:
            continue
        _replay_ob_edge_edge_attachment(
            working_world, mobile_element_id, target_element_id
        )
        replayed_attachment_count += 1

    working_world.reconcileGroupAnchorsByNode()
    for anchor_id, beacon_id, node_id in surviving_anchors:
        if anchor_id not in working_world.groupAnchors:
            raise ValueError(
                f"Ancre survivante perdue pendant le rebuild : {anchor_id!r}"
            )
        anchor = working_world.getGroupAnchor(anchor_id)
        if anchor.beacon_id != beacon_id or anchor.node_id != node_id:
            raise ValueError(
                f"Ancre survivante altérée pendant le rebuild : {anchor_id!r}"
            )
        working_world.applyGroupAnchor(anchor_id)

    errors = working_world.validate_world()
    if errors:
        raise ValueError(
            "Reconstruction topologique invalide après modification d'hypothèse : "
            + " ; ".join(str(error) for error in errors)
        )

    scenario.topoWorld = working_world
    scenario.hypothesis = candidate
    return HypothesisTopologyApplyResult(
        plan,
        tuple(new_element_by_rank[rank] for rank in sorted(new_element_by_rank)),
        replayed_attachment_count,
    )
