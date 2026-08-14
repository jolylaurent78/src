from types import SimpleNamespace

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage, TopologyElement
from src.assembleur_hypothesis_window import ScenarioHypothesisDialog
from src.assembleur_scenario import (
    HypothesisImpact,
    ScenarioHypothesis,
    analyze_hypothesis_change,
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


def test_non_empty_manual_scenario_keeps_hypothesis_and_topology_unchanged():
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
    assert manual.hypothesis is original
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
