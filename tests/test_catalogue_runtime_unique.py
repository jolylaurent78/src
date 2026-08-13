import tkinter as tk

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_window import CatalogueWindow
from src.assembleur_core import ScenarioAssemblage
from src.assembleur_scenario import create_default_scenario_hypothesis


def _valid_catalogue() -> Catalogue:
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
            triangle_ids.append(
                catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id).triangle_id
            )
    template = catalogue.add_template("Ordre principal")
    catalogue.set_template_ranks(template.template_id, triangle_ids)
    return catalogue


def test_window_edits_a_working_copy_and_cancel_never_changes_runtime_catalogue(tmp_path):
    runtime = _valid_catalogue()
    original_description = runtime.get_template("TPL-0001").description
    published = []
    root = tk.Tk()
    root.withdraw()
    window = CatalogueWindow(
        root,
        catalogue=runtime,
        catalogue_path=tmp_path / "catalogue.json",
        on_catalogue_applied=published.append,
    )
    try:
        assert window.catalogue is not runtime
        window.catalogue.update_template("TPL-0001", description="Brouillon")
        assert runtime.get_template("TPL-0001").description == original_description
        window._cancel_changes()
        assert window.catalogue.get_template("TPL-0001").description == original_description
        assert runtime.get_template("TPL-0001").description == original_description
        assert published == []
    finally:
        window.destroy()
        root.destroy()


def test_apply_publishes_runtime_catalogue_for_future_scenarios_only(tmp_path):
    runtime = _valid_catalogue()
    existing = ScenarioAssemblage(
        "Avant catalogue",
        source_type="manual",
        hypothesis=create_default_scenario_hypothesis(runtime),
    )
    existing_ranks = list(existing.hypothesis.triangle_ids_by_rank)
    published = []
    root = tk.Tk()
    root.withdraw()
    window = CatalogueWindow(
        root,
        catalogue=runtime,
        catalogue_path=tmp_path / "catalogue.json",
        on_catalogue_applied=published.append,
    )
    try:
        changed_ranks = list(window.catalogue.get_template("TPL-0001").triangle_ids_by_rank)
        changed_ranks[0], changed_ranks[1] = changed_ranks[1], changed_ranks[0]
        window.catalogue.set_template_ranks("TPL-0001", changed_ranks)
        window._mark_dirty()
        window._apply_changes()

        assert len(published) == 1
        runtime = published[0]
        assert runtime.get_template("TPL-0001").triangle_ids_by_rank == changed_ranks
        assert (tmp_path / "catalogue.json").exists()

        future = ScenarioAssemblage(
            "Après catalogue",
            source_type="manual",
            hypothesis=create_default_scenario_hypothesis(runtime),
        )
        assert future.hypothesis.triangle_ids_by_rank == changed_ranks
        assert existing.hypothesis.triangle_ids_by_rank == existing_ranks
        assert existing.hypothesis.triangle_ids_by_rank is not runtime.get_template("TPL-0001").triangle_ids_by_rank
    finally:
        window.destroy()
        root.destroy()
