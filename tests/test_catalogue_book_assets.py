from __future__ import annotations

from pathlib import Path

import pytest

from src.DictionnaireEnigmes import parse_book_file, parse_book_lines
from src.assembleur_catalogue import Catalogue, CatalogueBook
from src.assembleur_catalogue_book_asset_controller import CatalogueBookAssetController
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider, UserCatalogueIdProvider
from src.assembleur_catalogue_io import catalogue_from_dict, catalogue_to_dict
from src.assembleur_paths import ApplicationPaths


def _paths(tmp_path: Path, *, catalogue_mode: str = "USER") -> ApplicationPaths:
    return ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user",
        catalogue_mode=catalogue_mode,
    )


def _sys_book_catalogue() -> Catalogue:
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    book = CatalogueBook("BOOK-SYS-000001", "Livre SYS", "books/livre.txt", description="Référence")
    catalogue.books[book.book_id] = book
    catalogue.id_counters["book"] = 1
    catalogue.default_book_id = book.book_id
    return catalogue


def test_book_description_round_trip_clone_and_legacy_boundary() -> None:
    catalogue = _sys_book_catalogue()
    clone = catalogue.clone()
    assert clone.get_book("BOOK-SYS-000001").description == "Référence"
    assert clone.get_book("BOOK-SYS-000001") is not catalogue.get_book("BOOK-SYS-000001")

    serialized = catalogue_to_dict(catalogue)
    assert serialized["books"][0]["description"] == "Référence"
    serialized["books"][0].pop("description")
    restored = catalogue_from_dict(serialized)
    assert restored.get_book("BOOK-SYS-000001").description == ""
    assert "description" in catalogue_to_dict(restored)["books"][0]


def test_parse_book_source_exposes_all_tags_and_source_grid() -> None:
    lines = parse_book_lines([
        "530 cherche[localise] nord[direction] simple\n",
        "780 autre[direction] mot[exclure]\n",
    ])
    assert [line.title for line in lines] == ["530", "780"]
    assert [(token.text, token.tag) for token in lines[0].tokens] == [
        ("cherche", "localise"), ("nord", "direction"), ("simple", None),
    ]
    assert {token.tag for line in lines for token in line.tokens if token.tag} == {"localise", "direction", "exclure"}


def test_book_asset_controller_stages_new_and_cancel_removes_asset(tmp_path) -> None:
    paths = _paths(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("530 mot[tag]\n", encoding="utf-8")
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    controller = CatalogueBookAssetController(catalogue, paths)

    book_id = controller.stage_new_book(source, name="Nouveau", description="Essai")

    book = catalogue.get_book(book_id)
    assert book_id.startswith("BOOK-USR-")
    assert not (paths.user_catalogue_books_dir / f"{book_id}.txt").exists()
    assert controller.asset_path_for(book).read_text(encoding="utf-8") == "530 mot[tag]\n"
    controller.discard()
    assert not (paths.user_catalogue_books_dir / ".staging" / f"{book_id}.txt").exists()


def test_delete_new_staged_book_never_resolves_or_schedules_a_published_asset(tmp_path) -> None:
    paths = _paths(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("530 mot[tag]\n", encoding="utf-8")
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    catalogue.add_book(name="Défaut", asset_file="books/default.txt")
    controller = CatalogueBookAssetController(catalogue, paths)
    book_id = controller.stage_new_book(source, name="À supprimer", description="")
    book = catalogue.get_book(book_id)
    staged = controller.asset_path_for(book)

    controller.schedule_delete_asset(book)
    catalogue.delete_book(book_id)

    assert not staged.exists()
    assert book_id not in controller._staged
    assert controller._scheduled_deletions == set()
    assert book_id not in catalogue.books


def test_delete_published_user_book_removes_its_asset_on_commit(tmp_path) -> None:
    paths = _paths(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("530 mot[tag]\n", encoding="utf-8")
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    catalogue.add_book(name="Défaut", asset_file="books/default.txt")
    controller = CatalogueBookAssetController(catalogue, paths)
    book_id = controller.stage_new_book(source, name="Publié", description="")
    book = catalogue.get_book(book_id)
    controller.commit()
    controller.finalize_commit()
    published = paths.user_catalogue_books_dir / f"{book_id}.txt"
    assert published.exists()

    controller.schedule_delete_asset(book)
    catalogue.delete_book(book_id)
    controller.commit()

    assert not published.exists()
    assert book_id not in catalogue.books


def test_system_provider_creates_and_duplicates_system_books_transactionally(tmp_path) -> None:
    paths = _paths(tmp_path, catalogue_mode="SYS")
    source = tmp_path / "source.txt"
    source.write_text("530 mot[tag]\n", encoding="utf-8")
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    controller = CatalogueBookAssetController(catalogue, paths, allow_system_book_editing=True)

    first_id = controller.stage_new_book(source, name="Livre SYS", description="Original")
    first = catalogue.get_book(first_id)
    assert first_id == "BOOK-SYS-000001"
    assert not (paths.default_catalogue_books_dir / f"{first_id}.txt").exists()
    controller.discard()
    assert not (paths.default_catalogue_books_dir / f"{first_id}.txt").exists()

    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    controller = CatalogueBookAssetController(catalogue, paths, allow_system_book_editing=True)
    first_id = controller.stage_new_book(source, name="Livre SYS", description="Original")
    first = catalogue.get_book(first_id)
    controller.commit()
    controller.finalize_commit()
    copy_id = controller.stage_duplicate(first)
    copied = catalogue.get_book(copy_id)
    controller.commit()

    assert copy_id == "BOOK-SYS-000002"
    assert copied.description == "Original"
    assert (paths.default_catalogue_books_dir / f"{copy_id}.txt").read_bytes() == source.read_bytes()


def test_system_staged_book_delete_cleans_staging_without_resolving_final_asset(tmp_path) -> None:
    paths = _paths(tmp_path, catalogue_mode="SYS")
    source = tmp_path / "source.txt"
    source.write_text("530 mot[tag]\n", encoding="utf-8")
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    catalogue.add_book(name="Défaut", asset_file="books/default.txt")
    controller = CatalogueBookAssetController(catalogue, paths, allow_system_book_editing=True)
    book_id = controller.stage_new_book(source, name="À supprimer", description="")
    book = catalogue.get_book(book_id)
    staged = controller.asset_path_for(book)
    controller.resolver.resolve = lambda _book: (_ for _ in ()).throw(AssertionError("Resolver interdit pour un asset stagé"))

    controller.schedule_delete_asset(book)
    catalogue.delete_book(book_id)

    assert book_id == "BOOK-SYS-000002"
    assert not staged.exists()
    assert controller._scheduled_deletions == set()
    assert book_id not in catalogue.books


def test_system_published_book_delete_is_transactional_and_rollback_restores_bytes(tmp_path) -> None:
    paths = _paths(tmp_path, catalogue_mode="SYS")
    source = tmp_path / "source.txt"
    content = b"530 mot[tag]\r\n"
    source.write_bytes(content)
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    catalogue.add_book(name="Défaut", asset_file="books/default.txt")
    controller = CatalogueBookAssetController(catalogue, paths, allow_system_book_editing=True)
    book_id = controller.stage_new_book(source, name="Publié", description="")
    book = catalogue.get_book(book_id)
    controller.commit()
    controller.finalize_commit()
    published = paths.default_catalogue_books_dir / f"{book_id}.txt"
    assert published.read_bytes() == content

    controller.schedule_delete_asset(book)
    catalogue.delete_book(book_id)
    assert published.exists()
    assert published in controller._scheduled_deletions
    created = controller.commit()
    assert not published.exists()
    controller.rollback(created)

    assert published.read_bytes() == content
    controller.schedule_delete_asset(book)
    controller.commit()
    controller.finalize_commit()
    assert not published.exists()


def test_user_mode_refuses_system_book_asset_delete(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths.default_catalogue_books_dir.mkdir(parents=True)
    asset = paths.default_catalogue_books_dir / "livre.txt"
    asset.write_text("530 mot[tag]\n", encoding="utf-8")
    catalogue = _sys_book_catalogue()
    controller = CatalogueBookAssetController(catalogue, paths)

    with pytest.raises(ValueError, match="consultables uniquement"):
        controller.schedule_delete_asset(catalogue.get_book("BOOK-SYS-000001"))

    assert asset.exists()
    assert controller._scheduled_deletions == set()


def test_shared_book_asset_cannot_be_scheduled_for_deletion(tmp_path) -> None:
    paths = _paths(tmp_path, catalogue_mode="SYS")
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    first = catalogue.add_book(name="Défaut", asset_file="books/shared.txt")
    second = catalogue.add_book(name="Autre", asset_file="books/shared.txt")
    controller = CatalogueBookAssetController(catalogue, paths, allow_system_book_editing=True)

    with pytest.raises(ValueError, match="partagé"):
        controller.schedule_delete_asset(catalogue.get_book(second))

    assert controller._scheduled_deletions == set()


def test_duplicate_sys_book_creates_independent_user_asset_and_import_rolls_back(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths.user_catalogue_books_dir.mkdir(parents=True)
    (paths.user_catalogue_books_dir / "livre.txt").write_text("530 nord[direction]\n", encoding="utf-8")
    catalogue = _sys_book_catalogue()
    controller = CatalogueBookAssetController(catalogue, paths)

    copied_id = controller.stage_duplicate(catalogue.get_book("BOOK-SYS-000001"))
    copied = catalogue.get_book(copied_id)
    created = controller.commit()
    assert copied_id.startswith("BOOK-USR-")
    copied_path = paths.user_catalogue_books_dir / f"{copied_id}.txt"
    assert copied_path.read_text(encoding="utf-8") == "530 nord[direction]\n"
    controller.finalize_commit()

    replacement = tmp_path / "replacement.txt"
    replacement.write_text("780 cherche[localise]\n", encoding="utf-8")
    controller.stage_import(copied, replacement)
    committed = controller.commit()
    assert copied_path.read_text(encoding="utf-8") == "780 cherche[localise]\n"
    controller.rollback(committed)
    assert copied_path.read_text(encoding="utf-8") == "530 nord[direction]\n"


def test_sys_book_import_is_refused_in_user_mode_and_export_preserves_bytes(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths.user_catalogue_books_dir.mkdir(parents=True)
    source_asset = paths.user_catalogue_books_dir / "livre.txt"
    source_asset.write_bytes(b"530 mot[exclure]\r\n")
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("530 autre\n", encoding="utf-8")
    catalogue = _sys_book_catalogue()
    controller = CatalogueBookAssetController(catalogue, paths)
    book = catalogue.get_book("BOOK-SYS-000001")

    with pytest.raises(ValueError, match="SYS"):
        controller.stage_import(book, replacement)
    target = tmp_path / "export.txt"
    controller.export(book, target)
    assert target.read_bytes() == source_asset.read_bytes()


def test_archive_default_clears_default_and_delete_default_is_protected() -> None:
    catalogue = _sys_book_catalogue()
    catalogue.archive_book("BOOK-SYS-000001")
    assert catalogue.default_book_id is None
    catalogue.update_book("BOOK-SYS-000001", archived=False)
    catalogue.set_default_book("BOOK-SYS-000001")
    with pytest.raises(ValueError, match="par défaut"):
        catalogue.delete_book("BOOK-SYS-000001")
