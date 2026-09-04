from __future__ import annotations

import pytest

from src.assembleur_catalogue import Catalogue, CatalogueBook
from src.assembleur_catalogue_book_assets import CatalogueBookAssetResolver
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider, UserCatalogueIdProvider
from src.assembleur_catalogue_io import catalogue_from_dict, catalogue_to_dict
from src.assembleur_paths import ApplicationPaths


def test_catalogue_book_round_trip_and_default_validation(tmp_path) -> None:
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    book = CatalogueBook("BOOK-SYS-000001", "Livre", "books/livre.txt")
    catalogue.books[book.book_id] = book
    catalogue.id_counters["book"] = 1
    catalogue.default_book_id = book.book_id

    restored = catalogue_from_dict(catalogue_to_dict(catalogue), id_provider=SystemCatalogueIdProvider())

    assert restored.default_book_id == book.book_id
    assert restored.get_book(book.book_id).asset_file == "books/livre.txt"


def test_book_resolver_uses_the_active_catalogue_root(tmp_path) -> None:
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user", catalogue_mode="USER")
    sys_book = CatalogueBook("BOOK-SYS-000001", "SYS", "books/livre.txt")
    user_book = CatalogueBook("BOOK-USR-550e8400-e29b-41d4-a716-446655440000", "USER", "books/custom.txt")
    paths.user_catalogue_books_dir.mkdir(parents=True)
    (paths.user_catalogue_books_dir / "livre.txt").write_text("sys", encoding="utf-8")
    (paths.user_catalogue_books_dir / "custom.txt").write_text("user", encoding="utf-8")
    resolver = CatalogueBookAssetResolver(paths)

    assert resolver.resolve(sys_book) == paths.user_catalogue_books_dir / "livre.txt"
    assert resolver.resolve(user_book) == paths.user_catalogue_books_dir / "custom.txt"
    with pytest.raises(FileNotFoundError, match="BOOK-SYS-000001"):
        resolver.resolve(CatalogueBook("BOOK-SYS-000001", "Missing", "books/missing.txt"))


def test_legacy_catalogue_is_normalized_with_the_system_default_book() -> None:
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    raw = catalogue_to_dict(catalogue)
    raw.pop("defaultBookId")
    raw.pop("books")
    raw["idCounters"].pop("book")

    restored = catalogue_from_dict(raw)

    assert restored.default_book_id == "BOOK-SYS-000001"
    assert restored.get_book("BOOK-SYS-000001").asset_file == "books/livre.txt"
