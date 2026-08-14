"""Modèles métier propres aux scénarios, indépendants de Tk et de la topologie."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.assembleur_catalogue import Catalogue, HypothesisTemplate
from src.assembleur_core import (
    TopologyElement,
    build_topology_element_from_catalogue_triangle,
)


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
