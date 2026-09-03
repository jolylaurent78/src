from __future__ import annotations

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_core import ScenarioAssemblage
from src.assembleur_paths import ApplicationPaths
from src.assembleur_tk import TriangleViewerManual


def _catalogue_with_books() -> tuple[Catalogue, str, str, str]:
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    first = catalogue.add_book(name="Livre A", asset_file="books/a.txt")
    second = catalogue.add_book(name="Livre B", asset_file="books/b.txt")
    archived = catalogue.add_book(name="Livre archivé", asset_file="books/archive.txt")
    catalogue.archive_book(archived)
    return catalogue, first, second, archived


def _viewer(catalogue: Catalogue, scenario: ScenarioAssemblage):
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer.scenarios = [scenario]
    viewer.active_scenario_index = 0
    viewer._get_active_scenario = lambda: viewer.scenarios[viewer.active_scenario_index]
    return viewer


def test_book_choices_use_names_only_and_keep_the_current_archived_book() -> None:
    catalogue, first, second, archived = _catalogue_with_books()
    scenario = ScenarioAssemblage("Scénario")
    scenario.book_ref_id = archived
    viewer = _viewer(catalogue, scenario)

    choices = TriangleViewerManual._scenario_property_book_choices(viewer, scenario)

    assert choices == (
        (first, "Livre A"),
        (second, "Livre B"),
        (archived, "Livre archivé"),
    )
    assert all("BOOK-" not in name for _book_id, name in choices)


def test_missing_current_book_is_reported_without_default_fallback() -> None:
    catalogue, _first, _second, _archived = _catalogue_with_books()
    scenario = ScenarioAssemblage("Scénario")
    scenario.book_ref_id = "BOOK-SYS-999999"
    viewer = _viewer(catalogue, scenario)

    with pytest.raises(ValueError, match="absent du Catalogue"):
        TriangleViewerManual._scenario_property_book_choices(viewer, scenario)

    assert scenario.book_ref_id == "BOOK-SYS-999999"


def test_book_change_rebuilds_only_the_active_dictionary() -> None:
    catalogue, first, second, _archived = _catalogue_with_books()
    scenario = ScenarioAssemblage("Scénario")
    scenario.book_ref_id = first
    viewer = _viewer(catalogue, scenario)
    viewer.dicoPanel = object()
    viewer._dico_origin_cell = (4, 5)
    viewer._dico_ref_mode = "origin"
    initialized = []
    rebuilt = []
    viewer._getDicoTagExclure = lambda: "exclure"
    viewer._init_dictionary = lambda **kwargs: initialized.append(kwargs)
    viewer._build_dico_grid = lambda: rebuilt.append(True)

    changed = TriangleViewerManual._apply_scenario_book_selection(viewer, scenario, second)

    assert changed is True
    assert scenario.book_ref_id == second
    assert viewer._dico_origin_cell is None
    assert viewer._dico_ref_mode is None
    assert initialized == [{"tagExclure": "exclure"}]
    assert rebuilt == [True]


def test_unchanged_book_does_not_rebuild_the_dictionary() -> None:
    catalogue, first, _second, _archived = _catalogue_with_books()
    scenario = ScenarioAssemblage("Scénario")
    scenario.book_ref_id = first
    viewer = _viewer(catalogue, scenario)
    viewer._getDicoTagExclure = lambda: (_ for _ in ()).throw(AssertionError("Refresh inattendu"))

    assert TriangleViewerManual._apply_scenario_book_selection(viewer, scenario, first) is False


def test_archived_book_asset_remains_resolvable_for_an_existing_scenario(tmp_path) -> None:
    catalogue, _first, _second, archived = _catalogue_with_books()
    paths = ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user",
    )
    paths.default_catalogue_books_dir.mkdir(parents=True)
    asset = paths.default_catalogue_books_dir / "archive.txt"
    asset.write_text("530 mot[tag]\n", encoding="utf-8")
    scenario = ScenarioAssemblage("Scénario")
    scenario.book_ref_id = archived
    viewer = _viewer(catalogue, scenario)
    viewer.paths = paths

    assert viewer._resolve_active_scenario_book_path() == str(asset)


def test_runtime_uses_scenario_book_not_the_catalogue_default(tmp_path) -> None:
    catalogue, first, second, _archived = _catalogue_with_books()
    paths = ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user",
    )
    paths.default_catalogue_books_dir.mkdir(parents=True)
    first_asset = paths.default_catalogue_books_dir / "a.txt"
    second_asset = paths.default_catalogue_books_dir / "b.txt"
    first_asset.write_text("530 premier[tag]\n", encoding="utf-8")
    second_asset.write_text("530 second[tag]\n", encoding="utf-8")
    scenario = ScenarioAssemblage("Scénario")
    scenario.book_ref_id = second
    viewer = _viewer(catalogue, scenario)
    viewer.paths = paths

    assert catalogue.default_book_id == first
    assert viewer._resolve_active_scenario_book_path() == str(second_asset)


def test_missing_book_asset_keeps_the_resolver_error_explicit(tmp_path) -> None:
    catalogue, _first, second, _archived = _catalogue_with_books()
    paths = ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user",
    )
    scenario = ScenarioAssemblage("Scénario")
    scenario.book_ref_id = second
    viewer = _viewer(catalogue, scenario)
    viewer.paths = paths

    with pytest.raises(FileNotFoundError, match=second):
        viewer._resolve_active_scenario_book_path()
