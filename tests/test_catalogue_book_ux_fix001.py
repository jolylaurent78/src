from __future__ import annotations

from types import SimpleNamespace

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_book_asset_controller import CatalogueBookAssetController
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider, UserCatalogueIdProvider
from src.assembleur_catalogue_window import CatalogueWindow
from src.assembleur_paths import ApplicationPaths


def test_catalogue_csv_export_dialog_starts_in_exports_dir(monkeypatch, tmp_path) -> None:
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    window = object.__new__(CatalogueWindow)
    window._paths = paths
    captured = {}
    monkeypatch.setattr(
        "src.assembleur_catalogue_window.filedialog.asksaveasfilename",
        lambda **kwargs: captured.update(kwargs) or "",
    )

    assert CatalogueWindow._choose_export_path(window, "Exporter") == ""
    assert captured["initialdir"] == paths.exports_dir


def test_book_export_dialog_starts_in_exports_dir(monkeypatch, tmp_path) -> None:
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    window = object.__new__(CatalogueWindow)
    window._paths = paths
    window._get_selected_book = lambda: SimpleNamespace(name="Livre")
    captured = {}
    monkeypatch.setattr(
        "src.assembleur_catalogue_window.filedialog.asksaveasfilename",
        lambda **kwargs: captured.update(kwargs) or "",
    )

    CatalogueWindow._export_selected_book(window)
    assert captured["initialdir"] == paths.exports_dir
    assert captured["defaultextension"] == ".txt"


def test_new_book_name_is_numbered_case_insensitively() -> None:
    window = object.__new__(CatalogueWindow)
    window.catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    assert CatalogueWindow._next_new_book_name(window) == "Nouveau Livre"
    window.catalogue.add_book(name="nouveau livre", asset_file="books/one.txt")
    assert CatalogueWindow._next_new_book_name(window) == "Nouveau Livre 2"
    window.catalogue.add_book(name="Nouveau Livre 2", asset_file="books/two.txt")
    assert CatalogueWindow._next_new_book_name(window) == "Nouveau Livre 3"


def test_new_book_asks_only_for_file_and_uses_default_metadata(monkeypatch, tmp_path) -> None:
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    source = tmp_path / "source.txt"
    source.write_text("530 mot[tag]\n", encoding="utf-8")
    window = object.__new__(CatalogueWindow)
    window._paths = paths
    window.catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    window._book_assets = CatalogueBookAssetController(window.catalogue, paths)
    window._refresh_books = lambda: None
    window._mark_dirty = lambda: setattr(window, "dirty", True)
    captured = {}
    monkeypatch.setattr(
        "src.assembleur_catalogue_window.filedialog.askopenfilename",
        lambda **kwargs: captured.update(kwargs) or str(source),
    )
    monkeypatch.setattr(
        "src.assembleur_catalogue_window.simpledialog.askstring",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Aucun dialogue texte attendu")),
    )

    CatalogueWindow._add_book(window)

    created = window.catalogue.get_book(window._selected_book_id)
    assert captured["initialdir"] == paths.exports_dir
    assert created.name == "Nouveau Livre"
    assert created.description == ""
    assert created.book_id.startswith("BOOK-USR-")
    assert window.dirty is True


def test_new_book_uses_the_active_system_provider(monkeypatch, tmp_path) -> None:
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    source = tmp_path / "source.txt"
    source.write_text("530 mot[tag]\n", encoding="utf-8")
    window = object.__new__(CatalogueWindow)
    window._paths = paths
    window.catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    window._book_assets = CatalogueBookAssetController(
        window.catalogue,
        paths,
        allow_system_book_editing=True,
    )
    window._refresh_books = lambda: None
    window._mark_dirty = lambda: None
    monkeypatch.setattr(
        "src.assembleur_catalogue_window.filedialog.askopenfilename",
        lambda **_kwargs: str(source),
    )

    CatalogueWindow._add_book(window)

    created = window.catalogue.get_book(window._selected_book_id)
    assert created.book_id == "BOOK-SYS-000001"
    assert created.name == "Nouveau Livre"
    assert created.description == ""


def test_new_book_cancel_leaves_catalogue_and_staging_unchanged(monkeypatch, tmp_path) -> None:
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    window = object.__new__(CatalogueWindow)
    window._paths = paths
    window.catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    window._book_assets = CatalogueBookAssetController(window.catalogue, paths)
    window._refresh_books = lambda: (_ for _ in ()).throw(AssertionError("Aucun refresh attendu"))
    window._mark_dirty = lambda: (_ for _ in ()).throw(AssertionError("Aucune mutation attendue"))
    monkeypatch.setattr("src.assembleur_catalogue_window.filedialog.askopenfilename", lambda **_kwargs: "")

    CatalogueWindow._add_book(window)

    assert window.catalogue.books == {}
    assert not (paths.user_catalogue_books_dir / ".staging").exists()


def _book_delete_window(tmp_path, *, referenced=None):
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    window = object.__new__(CatalogueWindow)
    window.catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    window.catalogue.add_book(name="Défaut", asset_file="books/default.txt")
    book_id = window.catalogue.add_book(name="Cible", asset_file="books/target.txt")
    paths.user_catalogue_books_dir.mkdir(parents=True)
    (paths.user_catalogue_books_dir / "target.txt").write_text("530 mot[tag]\n", encoding="utf-8")
    window._selected_book_id = book_id
    window._book_assets = CatalogueBookAssetController(window.catalogue, paths)
    window._is_book_referenced = referenced
    window._refresh_books = lambda: None
    window._mark_dirty = lambda: setattr(window, "dirty", True)
    return window, book_id


def test_delete_unreferenced_book_without_callback_is_allowed(monkeypatch, tmp_path) -> None:
    window, book_id = _book_delete_window(tmp_path)
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.askyesno", lambda *_args, **_kwargs: True)

    CatalogueWindow._delete_selected_book(window)

    assert book_id not in window.catalogue.books
    assert window.dirty is True


def test_delete_book_referenced_by_loaded_scenario_is_refused(monkeypatch, tmp_path) -> None:
    window, book_id = _book_delete_window(tmp_path, referenced=lambda candidate: candidate == book_id)
    errors = []
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.showerror", lambda *_args, **_kwargs: errors.append(_args))
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.askyesno", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Confirmation inattendue")))

    CatalogueWindow._delete_selected_book(window)

    assert book_id in window.catalogue.books
    assert "scénario actuellement chargé" in errors[0][1]


def test_delete_default_book_is_refused_before_confirmation(monkeypatch, tmp_path) -> None:
    window, _book_id = _book_delete_window(tmp_path)
    window._selected_book_id = window.catalogue.default_book_id
    errors = []
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.showerror", lambda *_args, **_kwargs: errors.append(_args))
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.askyesno", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Confirmation inattendue")))

    CatalogueWindow._delete_selected_book(window)

    assert window.catalogue.default_book_id in window.catalogue.books
    assert "par défaut" in errors[0][1]
