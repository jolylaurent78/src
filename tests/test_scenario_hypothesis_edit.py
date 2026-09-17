from types import SimpleNamespace

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage, TopologyElement
from src.assembleur_geometry_reference import GeometryReferenceResolver, ScenarioReference
from src.assembleur_hypothesis_window import ScenarioHypothesisDialog
from src.assembleur_scenario import (
    HypothesisImpact,
    ScenarioHypothesis,
    _active_reference_changed_triangle_ids,
    analyze_hypothesis_change,
    apply_hypothesis_change_to_manual_scenario,
    materialize_triangle,
)
from src.assembleur_tk import TriangleViewerManual


def _catalogue_and_hypothesis():
    catalogue = Catalogue()
    ranks = []
    for index in range(16):
        base = catalogue.add_city(f"Base {index}", 40.0 + index, 2.0)
        for parity in range(2):
            opening = catalogue.add_city(f"O {index}-{parity}", 40.0 + index, 3.0 + parity)
            light = catalogue.add_city(f"L {index}-{parity}", 40.0 + index, 4.0 + parity)
            ranks.append(catalogue.add_triangle(f"N {index}-{parity}", opening.city_id, base.city_id, light.city_id).triangle_id)
    first = catalogue.get_triangle(ranks[0])
    replay_light = catalogue.add_city("L replay", 55.0, 6.0)
    replay = catalogue.add_triangle("N replay", first.opening_city_id, first.base_city_id, replay_light.city_id)
    detach_opening = catalogue.add_city("O detach", 56.0, 6.0)
    detach = catalogue.add_triangle("N detach", detach_opening.city_id, first.base_city_id, replay_light.city_id)
    return catalogue, ScenarioHypothesis(ranks, "TPL-A"), replay.triangle_id, detach.triangle_id


def test_hypothesis_change_ignores_template_provenance_when_ranks_are_identical():
    catalogue, old, _replay, _detach = _catalogue_and_hypothesis()
    same = old.clone()
    none = analyze_hypothesis_change(catalogue, old, same)
    assert none.global_impact is HypothesisImpact.NONE
    assert none.rank_changes == ()

    changed_template = old.clone()
    changed_template.source_template_id = "TPL-B"
    plan = analyze_hypothesis_change(catalogue, old, changed_template)
    assert plan.template_changed
    assert plan.global_impact is HypothesisImpact.NONE
    assert plan.rank_changes == ()


def test_hypothesis_change_classifies_replay_and_opening_detach():
    catalogue, old, replay_id, detach_id = _catalogue_and_hypothesis()
    replay = old.clone()
    replay.source_template_id = "TPL-B"
    replay.triangle_ids_by_rank[0] = replay_id
    replay_plan = analyze_hypothesis_change(catalogue, old, replay)
    assert replay_plan.template_changed
    assert replay_plan.rank_changes[0].impact is HypothesisImpact.REPLAY
    assert replay_plan.global_impact is HypothesisImpact.REPLAY

    detach = old.clone()
    detach.source_template_id = "TPL-B"
    detach.triangle_ids_by_rank[0] = detach_id
    detach_plan = analyze_hypothesis_change(catalogue, old, detach)
    assert detach_plan.rank_changes[0].impact is HypothesisImpact.DETACH
    assert detach_plan.global_impact is HypothesisImpact.DETACH


def test_hypothesis_change_detects_base_swap_as_detach():
    catalogue, old, _replay, _detach = _catalogue_and_hypothesis()
    changed = old.clone()
    changed.triangle_ids_by_rank[0:4] = old.triangle_ids_by_rank[2:4] + old.triangle_ids_by_rank[0:2]
    plan = analyze_hypothesis_change(catalogue, old, changed)
    assert len(plan.rank_changes) == 4
    assert all(change.impact is HypothesisImpact.DETACH for change in plan.rank_changes)
    assert plan.global_impact is HypothesisImpact.DETACH


def test_empty_manual_scenario_commits_a_draft_without_touching_an_auto_snapshot():
    catalogue, original, replay_id, _detach = _catalogue_and_hypothesis()
    manual = ScenarioAssemblage("Manuel", source_type="manual", hypothesis=original)
    auto = ScenarioAssemblage("Auto", source_type="auto", hypothesis=original.clone())
    draft = original.clone()
    draft.triangle_ids_by_rank[0] = replay_id
    viewer = SimpleNamespace(catalogue=catalogue)

    plan = TriangleViewerManual._commit_manual_hypothesis_draft(viewer, manual, draft)

    assert plan.global_impact is HypothesisImpact.REPLAY
    assert manual.hypothesis is not draft
    assert manual.hypothesis.triangle_ids_by_rank == draft.triangle_ids_by_rank
    assert auto.hypothesis.triangle_ids_by_rank == original.triangle_ids_by_rank
    assert manual.topoWorld.elements == {}


def test_non_empty_manual_scenario_commits_without_touching_unrelated_topology():
    catalogue, original, replay_id, _detach = _catalogue_and_hypothesis()
    manual = ScenarioAssemblage("Manuel", source_type="manual", hypothesis=original)
    manual.topoWorld.add_element_as_new_group(TopologyElement(
        element_id="T01", name="T01", vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"], edge_lengths_km=[3.0, 4.0, 5.0],
    ))
    world_snapshot = manual.topoWorld._exportPhysicalSnapshot()
    draft = original.clone()
    draft.triangle_ids_by_rank[0] = replay_id
    viewer = SimpleNamespace(catalogue=catalogue)

    plan = TriangleViewerManual._commit_manual_hypothesis_draft(viewer, manual, draft)

    assert plan.global_impact is HypothesisImpact.REPLAY
    assert manual.hypothesis is not original
    assert manual.hypothesis.triangle_ids_by_rank == draft.triangle_ids_by_rank
    assert manual.topoWorld._exportPhysicalSnapshot() == world_snapshot


def test_non_empty_manual_scenario_commits_template_provenance_without_core_mutation():
    catalogue, original, _replay_id, _detach = _catalogue_and_hypothesis()
    manual = ScenarioAssemblage("Manuel", source_type="manual", hypothesis=original)
    manual.topoWorld.add_element_as_new_group(TopologyElement(
        element_id="T01", name="T01", vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"], edge_lengths_km=[3.0, 4.0, 5.0],
    ))
    draft = original.clone()
    draft.source_template_id = "TPL-B"
    viewer = SimpleNamespace(catalogue=catalogue)

    plan = TriangleViewerManual._commit_manual_hypothesis_draft(viewer, manual, draft)

    assert plan.global_impact is HypothesisImpact.NONE
    assert manual.hypothesis is not original
    assert manual.hypothesis.source_template_id == "TPL-B"


def test_hypothesis_dialog_template_replacement_is_draft_only_and_independent():
    catalogue, original, replay_id, _detach = _catalogue_and_hypothesis()
    template = catalogue.add_template("Autre ordre")
    replacement_ranks = list(original.triangle_ids_by_rank)
    replacement_ranks[0] = replay_id
    catalogue.set_template_ranks(template.template_id, replacement_ranks)

    dialog = object.__new__(ScenarioHypothesisDialog)
    dialog.catalogue = catalogue
    dialog._draft = original.clone()
    ScenarioHypothesisDialog._replace_draft_from_template(dialog, template.template_id)

    assert dialog._draft.source_template_id == template.template_id
    assert dialog._draft.triangle_ids_by_rank == replacement_ranks
    assert dialog._draft.triangle_ids_by_rank is not template.triangle_ids_by_rank
    assert original.triangle_ids_by_rank != replacement_ranks
    assert original.triangle_ids_by_rank is not dialog._draft.triangle_ids_by_rank


def test_hypothesis_dialog_ranks_view_uses_the_scenario_hypothesis_order():
    catalogue, hypothesis, _replay_id, _detach = _catalogue_and_hypothesis()

    class PairRow:
        def __init__(self):
            self.values = None

        def set_triangles(self, *values):
            self.values = values

    dialog = object.__new__(ScenarioHypothesisDialog)
    dialog.catalogue = catalogue
    dialog.resolver = GeometryReferenceResolver(catalogue, ScenarioReference())
    dialog._draft = hypothesis.clone()
    dialog._pair_rows = [PairRow() for _ in range(16)]
    dialog._selected_slot = None
    ScenarioHypothesisDialog._refresh_ranks(dialog)

    first = dialog._pair_rows[0].values
    assert first[0] == hypothesis.triangle_ids_by_rank[0]
    assert first[2] == hypothesis.triangle_ids_by_rank[1]
    assert len(dialog._pair_rows) == 16


def test_hypothesis_dialog_drop_updates_only_its_valid_draft_preview():
    catalogue, original, replay_id, _detach = _catalogue_and_hypothesis()
    dialog = object.__new__(ScenarioHypothesisDialog)
    dialog.catalogue = catalogue
    dialog.resolver = GeometryReferenceResolver(catalogue, ScenarioReference())
    dialog._draft = original.clone()
    target_slot = SimpleNamespace(rank_number=1)

    action, valid, message, preview = ScenarioHypothesisDialog._plan_drop(
        dialog, replay_id, target_slot,
    )

    assert action == "replace"
    assert valid is True
    assert message is None
    assert preview[0] == replay_id
    assert dialog._draft.triangle_ids_by_rank == original.triangle_ids_by_rank


def _manual_scenario_with_active_and_orphan_reference():
    catalogue, hypothesis, _replay_id, _detach = _catalogue_and_hypothesis()
    reference = ScenarioReference()
    source = catalogue.get_triangle(hypothesis.triangle_ids_by_rank[0])
    active_city = reference.create_city("Active", 48.0, 2.0)
    orphan_city = reference.create_city("Orpheline", 49.0, 3.0)
    active_triangle = reference.create_triangle(
        "Active", source.opening_city_id, source.base_city_id, active_city.city_ref_id,
        catalogue_source_triangle_id=source.triangle_id,
    )
    orphan_triangle = reference.create_triangle(
        "Orphelin", source.opening_city_id, source.base_city_id, orphan_city.city_ref_id,
        catalogue_source_triangle_id=source.triangle_id,
    )
    hypothesis = hypothesis.clone()
    hypothesis.triangle_ids_by_rank[0] = active_triangle.triangle_ref_id
    hypothesis.validate(GeometryReferenceResolver(catalogue, reference))
    scenario = ScenarioAssemblage("Manuel", source_type="manual", hypothesis=hypothesis)
    scenario.reference = reference
    resolver = GeometryReferenceResolver(catalogue, reference)
    scenario.topoWorld.add_element_as_new_group(
        materialize_triangle(resolver, active_triangle.triangle_ref_id)
    )
    scenario.topoWorld.add_element_as_new_group(
        materialize_triangle(resolver, hypothesis.triangle_ids_by_rank[1])
    )
    return catalogue, scenario, active_city, active_triangle, orphan_city, orphan_triangle


def test_orphan_reference_removal_keeps_the_same_core_world(monkeypatch):
    catalogue, scenario, _active_city, _active_triangle, orphan_city, orphan_triangle = (
        _manual_scenario_with_active_and_orphan_reference()
    )
    candidate_reference = scenario.reference.clone()
    candidate_reference.remove_triangle(orphan_triangle.triangle_ref_id)
    candidate_reference.remove_city(orphan_city.city_ref_id)
    old_world = scenario.topoWorld
    monkeypatch.setattr(
        old_world, "clonePhysicalState",
        lambda: pytest.fail("An orphan-only reference change must not clone Core"),
    )

    result = apply_hypothesis_change_to_manual_scenario(
        catalogue, scenario, scenario.hypothesis.clone(), candidate_reference
    )

    assert result.plan.rank_changes == ()
    assert orphan_city.city_ref_id not in scenario.reference.cities
    assert orphan_triangle.triangle_ref_id not in scenario.reference.triangles
    assert scenario.topoWorld is old_world


def test_active_city_rename_rematerializes_only_affected_active_elements(monkeypatch):
    catalogue, scenario, active_city, active_triangle, _orphan_city, _orphan_triangle = (
        _manual_scenario_with_active_and_orphan_reference()
    )
    candidate_reference = scenario.reference.clone()
    candidate_reference.rename_city(active_city.city_ref_id, "Active renommee")
    affected = _active_reference_changed_triangle_ids(
        catalogue,
        scenario.hypothesis.triangle_ids_by_rank,
        scenario.reference,
        candidate_reference,
    )
    replaced = []
    original_replace = type(scenario.topoWorld).replace_element_materialized_definition

    def track_replace(world, element_id, definition):
        replaced.append((element_id, definition.source_triangle_id))
        return original_replace(world, element_id, definition)

    monkeypatch.setattr(
        type(scenario.topoWorld), "replace_element_materialized_definition", track_replace
    )

    apply_hypothesis_change_to_manual_scenario(
        catalogue, scenario, scenario.hypothesis.clone(), candidate_reference
    )

    assert affected == {active_triangle.triangle_ref_id}
    assert [source_triangle_id for _element_id, source_triangle_id in replaced] == [
        active_triangle.triangle_ref_id
    ]


def test_reference_change_helper_ignores_orphan_city_rename():
    catalogue, scenario, _active_city, _active_triangle, orphan_city, _orphan_triangle = (
        _manual_scenario_with_active_and_orphan_reference()
    )
    candidate_reference = scenario.reference.clone()
    candidate_reference.rename_city(orphan_city.city_ref_id, "Orpheline renommee")

    assert _active_reference_changed_triangle_ids(
        catalogue,
        scenario.hypothesis.triangle_ids_by_rank,
        scenario.reference,
        candidate_reference,
    ) == set()


def test_reference_change_helper_marks_all_active_triangles_sharing_a_city():
    catalogue = Catalogue()
    reference = ScenarioReference()
    shared = reference.create_city("Partagee", 45.0, 2.0)
    cities = [reference.create_city(f"Ville {index}", 46.0 + index, 2.0) for index in range(4)]
    first = reference.create_triangle(
        "Premier", shared.city_ref_id, cities[0].city_ref_id, cities[1].city_ref_id
    )
    second = reference.create_triangle(
        "Second", cities[2].city_ref_id, shared.city_ref_id, cities[3].city_ref_id
    )
    candidate_reference = reference.clone()
    candidate_reference.rename_city(shared.city_ref_id, "Partagee renommee")

    assert _active_reference_changed_triangle_ids(
        catalogue,
        [first.triangle_ref_id, second.triangle_ref_id],
        reference,
        candidate_reference,
    ) == {first.triangle_ref_id, second.triangle_ref_id}
