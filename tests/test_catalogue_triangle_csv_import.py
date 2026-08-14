from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_window import CatalogueWindow


def _catalogue_with_cities() -> Catalogue:
    catalogue = Catalogue()
    for index, name in enumerate(("Ouverture", "Base", "Lumière", "Lumière 2")):
        catalogue.add_city(name, 45.0 + index, 2.0)
    return catalogue


def _import(catalogue: Catalogue, tmp_path, *rows: str):
    path = tmp_path / "triangles.csv"
    path.write_text("Note;Ouverture;Base;Lumiere\n" + "\n".join(rows) + "\n", encoding="utf-8")
    window = object.__new__(CatalogueWindow)
    window.catalogue = catalogue
    return CatalogueWindow._read_triangles_csv(window, str(path))


def test_triangle_csv_import_adds_new_triangles_and_skips_a_reimport(tmp_path):
    catalogue = _catalogue_with_cities()
    rows = ("Do;Ouverture;Base;Lumière", "Si;Ouverture;Base;Lumière 2")

    first = _import(catalogue, tmp_path, *rows)
    historical_ids = tuple(first.imported_triangle_ids)
    second = _import(catalogue, tmp_path, *rows)

    assert first.imported_count == 2
    assert first.already_present_count == 0
    assert first.errors == ()
    assert second.imported_count == 0
    assert second.already_present_count == 2
    assert second.errors == ()
    assert tuple(catalogue.triangles) == historical_ids


def test_triangle_csv_import_is_incremental_and_detects_file_duplicates(tmp_path):
    catalogue = _catalogue_with_cities()
    initial = _import(catalogue, tmp_path, "Do;Ouverture;Base;Lumière")
    historical_id = initial.imported_triangle_ids[0]

    result = _import(
        catalogue,
        tmp_path,
        "Do;Ouverture;Base;Lumière",
        "Si;Ouverture;Base;Lumière 2",
        "La;Ouverture;Base;Lumière 2",
    )

    assert result.imported_count == 1
    assert result.already_present_count == 2
    assert result.errors == ()
    assert catalogue.get_triangle(historical_id).triangle_id == historical_id
    assert len(catalogue.triangles) == 2


def test_triangle_csv_import_skips_an_archived_triangle_without_modifying_it(tmp_path):
    catalogue = _catalogue_with_cities()
    original = _import(catalogue, tmp_path, "Do;Ouverture;Base;Lumière")
    triangle_id = original.imported_triangle_ids[0]
    catalogue.update_triangle(triangle_id, archived=True)

    result = _import(catalogue, tmp_path, "Do;Ouverture;Base;Lumière")

    assert result.imported_count == 0
    assert result.already_present_count == 1
    assert result.errors == ()
    assert catalogue.get_triangle(triangle_id).archived is True


def test_triangle_csv_import_reports_only_genuine_errors(tmp_path):
    catalogue = _catalogue_with_cities()
    _import(catalogue, tmp_path, "Do;Ouverture;Base;Lumière")

    result = _import(
        catalogue,
        tmp_path,
        "Do;Ouverture;Base;Lumière",
        "Si;Ouverture;Base;Inconnue",
    )
    summary = CatalogueWindow._format_triangles_import_summary(result)

    assert result.already_present_count == 1
    assert len(result.errors) == 1
    assert result.errors[0].startswith("Ligne 3 :")
    assert "Inconnue" in result.errors[0]
    assert "déjà présent" in summary
    assert "Ligne 3" in summary
    assert "Ligne 2" not in summary
