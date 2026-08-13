import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage
from src.assembleur_scenario import (
    ScenarioHypothesis,
    create_default_scenario_hypothesis,
)


def _valid_catalogue() -> tuple[Catalogue, list[str]]:
    catalogue = Catalogue()
    opening = catalogue.add_city("Ouverture", 47.0, 2.0)
    triangle_ids = []
    for pair_index in range(16):
        base = catalogue.add_city(f"Base {pair_index}", 44.0 + pair_index / 10, 1.0)
        for member_index in range(2):
            light = catalogue.add_city(
                f"Lumière {pair_index}-{member_index}",
                42.0 + pair_index / 10,
                2.0 + member_index / 10,
            )
            triangle = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
            triangle_ids.append(triangle.triangle_id)
    template = catalogue.add_template("Ordre principal")
    catalogue.set_template_ranks(template.template_id, triangle_ids)
    return catalogue, triangle_ids


def test_default_hypothesis_is_complete_owned_and_traceable():
    catalogue, triangle_ids = _valid_catalogue()
    template = catalogue.require_valid_default_template()

    hypothesis = create_default_scenario_hypothesis(catalogue)

    assert hypothesis.source_template_id == template.template_id
    assert hypothesis.triangle_ids_by_rank == triangle_ids
    assert hypothesis.triangle_ids_by_rank is not template.triangle_ids_by_rank
    hypothesis.validate(catalogue)


def test_template_and_scenario_hypothesis_are_independent():
    catalogue, triangle_ids = _valid_catalogue()
    template = catalogue.require_valid_default_template()
    hypothesis = create_default_scenario_hypothesis(catalogue)

    replacement = list(template.triangle_ids_by_rank)
    replacement[0], replacement[1] = replacement[1], replacement[0]
    catalogue.set_template_ranks(template.template_id, replacement)
    assert hypothesis.triangle_ids_by_rank[0] == triangle_ids[0]

    scenario_update = list(hypothesis.triangle_ids_by_rank)
    scenario_update[2], scenario_update[3] = scenario_update[3], scenario_update[2]
    hypothesis.set_ranks(catalogue, scenario_update)
    assert template.triangle_ids_by_rank[2] == triangle_ids[2]


def test_hypothesis_clone_and_atomic_set_ranks():
    catalogue, _triangle_ids = _valid_catalogue()
    hypothesis = create_default_scenario_hypothesis(catalogue)
    clone = hypothesis.clone()
    assert clone is not hypothesis
    assert clone.triangle_ids_by_rank is not hypothesis.triangle_ids_by_rank
    assert clone.triangle_ids_by_rank == hypothesis.triangle_ids_by_rank
    assert clone.source_template_id == hypothesis.source_template_id

    before = list(hypothesis.triangle_ids_by_rank)
    invalid = list(before)
    invalid[1] = invalid[0]
    with pytest.raises(ValueError, match="plusieurs rangs"):
        hypothesis.set_ranks(catalogue, invalid)
    assert hypothesis.triangle_ids_by_rank == before

    invalid = list(before)
    invalid[0], invalid[2] = invalid[2], invalid[0]
    with pytest.raises(ValueError, match="même base"):
        hypothesis.set_ranks(catalogue, invalid)
    assert hypothesis.triangle_ids_by_rank == before

    with pytest.raises(ValueError, match="exactement 32"):
        hypothesis.set_ranks(catalogue, before[:-1])
    assert hypothesis.triangle_ids_by_rank == before

    invalid = list(before)
    invalid[0] = "TRI-9999"
    with pytest.raises(ValueError, match="Triangle inconnu"):
        hypothesis.set_ranks(catalogue, invalid)
    assert hypothesis.triangle_ids_by_rank == before


def test_default_factory_refuses_missing_or_incomplete_template():
    with pytest.raises(ValueError, match="Aucun template par défaut"):
        create_default_scenario_hypothesis(Catalogue())

    catalogue = Catalogue()
    catalogue.add_template("Incomplet")
    with pytest.raises(ValueError, match="incomplet"):
        create_default_scenario_hypothesis(catalogue)


def test_scenario_assemblage_owns_explicit_hypothesis_contract():
    catalogue, _triangle_ids = _valid_catalogue()
    hypothesis = create_default_scenario_hypothesis(catalogue)
    scenario = ScenarioAssemblage("Scénario manuel", source_type="manual", hypothesis=hypothesis)

    assert scenario.hypothesis is hypothesis
    assert len(scenario.hypothesis.triangle_ids_by_rank) == 32
