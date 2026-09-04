"""Transaction des assets texte associes aux :class:`CatalogueBook`."""

from __future__ import annotations

from pathlib import Path
import shutil

from src.DictionnaireEnigmes import parse_book_file
from src.assembleur_catalogue import Catalogue, CatalogueBook
from src.assembleur_catalogue_book_assets import CatalogueBookAssetResolver
from src.assembleur_catalogue_identity import is_system_catalogue_id
from src.assembleur_paths import ApplicationPaths


class CatalogueBookAssetController:
    """Stage les modifications d'assets Books jusqu'au commit Catalogue."""

    def __init__(self, catalogue: Catalogue, paths: ApplicationPaths, *, allow_system_book_editing: bool = False) -> None:
        self.catalogue = catalogue
        self.paths = paths
        self.allow_system_book_editing = bool(allow_system_book_editing)
        self.resolver = CatalogueBookAssetResolver(paths)
        self._staged: dict[str, Path] = {}
        self._backups: dict[Path, bytes] = {}
        self._scheduled_deletions: set[Path] = set()

    def rebind_catalogue(self, catalogue: Catalogue) -> None:
        self.catalogue = catalogue

    def is_readonly(self, book: CatalogueBook) -> bool:
        return is_system_catalogue_id(book.book_id) and not self.allow_system_book_editing

    def stage_new_book(self, source: str | Path, *, name: str, description: str = "") -> str:
        source_path = self._validate_source(source)
        book_id = self.catalogue.add_book(
            name=name,
            asset_file="books/staging.txt",
            description=description,
        )
        asset_file = f"books/{book_id}.txt"
        self.catalogue.update_book(book_id, asset_file=asset_file)
        self._stage(book_id, source_path)
        return book_id

    def stage_duplicate(self, source_book: CatalogueBook, *, name: str | None = None) -> str:
        duplicate_name = name or self._copy_name(source_book.name)
        book_id = self.catalogue.add_book(
            name=duplicate_name,
            asset_file="books/staging.txt",
            description=source_book.description,
        )
        self.catalogue.update_book(book_id, asset_file=f"books/{book_id}.txt")
        self._stage(book_id, self.asset_path_for(source_book))
        return book_id

    def stage_import(self, book: CatalogueBook, source: str | Path) -> None:
        if self.is_readonly(book):
            raise ValueError("Les livres SYS sont consultables uniquement.")
        self._stage(book.book_id, self._validate_source(source))

    def export(self, book: CatalogueBook, destination: str | Path) -> None:
        source = self.asset_path_for(book)
        target = Path(destination)
        if target.resolve() == source.resolve():
            raise ValueError("Le fichier d'export doit être distinct de l'asset source.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def schedule_delete_asset(self, book: CatalogueBook) -> None:
        if self.is_readonly(book):
            raise ValueError("Les livres SYS sont consultables uniquement.")
        if any(item.book_id != book.book_id and item.asset_file == book.asset_file for item in self.catalogue.books.values()):
            raise ValueError("L'asset du livre est partagé par un autre livre.")
        staged = self._staged.pop(book.book_id, None)
        if staged is not None:
            staged.unlink(missing_ok=True)
            return
        self._scheduled_deletions.add(self.resolver.resolve(book))

    def asset_path_for(self, book: CatalogueBook) -> Path:
        staged = self._staged.get(book.book_id)
        return staged if staged is not None else self.resolver.resolve(book)

    def commit(self) -> list[Path]:
        created: list[Path] = []
        for book_id, staged in self._staged.items():
            book = self.catalogue.get_book(book_id)
            root = self.paths.active_catalogue_books_dir
            root.mkdir(parents=True, exist_ok=True)
            destination = root / book.asset_file.removeprefix("books/")
            self._publish(staged, destination, created)
        for destination in self._scheduled_deletions:
            if destination.exists():
                self._backups[destination] = destination.read_bytes()
                destination.unlink()
        return created

    def rollback(self, created: list[Path]) -> None:
        for destination, content in self._backups.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        for destination in created:
            destination.unlink(missing_ok=True)
        self._backups.clear()

    def finalize_commit(self) -> None:
        self.discard()

    def discard(self) -> None:
        for staged in self._staged.values():
            staged.unlink(missing_ok=True)
        self._staged.clear()
        self._backups.clear()
        self._scheduled_deletions.clear()

    def _stage(self, book_id: str, source: Path) -> None:
        staging_dir = self.paths.active_catalogue_books_dir / ".staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged = staging_dir / f"{book_id}.txt"
        shutil.copy2(source, staged)
        self._staged[book_id] = staged

    def _validate_source(self, source: str | Path) -> Path:
        path = Path(source)
        if path.suffix.casefold() != ".txt":
            raise ValueError("Le livre doit être un fichier .txt.")
        if not path.is_file():
            raise FileNotFoundError(f"Fichier livre introuvable : {path}")
        parse_book_file(path)
        return path

    def _publish(self, source: Path, destination: Path, created: list[Path]) -> None:
        if destination.exists():
            self._backups[destination] = destination.read_bytes()
        else:
            created.append(destination)
        shutil.copy2(source, destination)

    def _copy_name(self, source_name: str) -> str:
        base = f"{source_name} - Copie"
        candidate = base
        number = 2
        existing = {book.name.casefold() for book in self.catalogue.books.values()}
        while candidate.casefold() in existing:
            candidate = f"{base} {number}"
            number += 1
        return candidate
