"""Modèles métier propres aux scénarios, indépendants de Tk et de la topologie."""

from __future__ import annotations

from dataclasses import dataclass

from src.assembleur_catalogue import Catalogue, HypothesisTemplate


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
                    "L'hypothÃ¨se du scÃ©nario rÃ©fÃ©rence un triangle Catalogue absent : "
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


def create_hypothesis_from_template(catalogue: Catalogue, template: HypothesisTemplate) -> ScenarioHypothesis:
    """Instancie une copie autonome des rangs d'un template Catalogue."""
    hypothesis = ScenarioHypothesis(list(template.triangle_ids_by_rank), template.template_id)
    hypothesis.validate(catalogue)
    return hypothesis


def create_default_scenario_hypothesis(catalogue: Catalogue) -> ScenarioHypothesis:
    """Instancie l'hypothèse du scénario depuis le template Catalogue par défaut."""
    return create_hypothesis_from_template(catalogue, catalogue.require_valid_default_template())
