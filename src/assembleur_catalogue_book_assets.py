"""Résolution physique des assets texte appartenant au Catalogue."""

from __future__ import annotations

from pathlib import Path

from src.assembleur_catalogue import CatalogueBook
from src.assembleur_catalogue_identity import is_catalogue_book_id
from src.assembleur_paths import ApplicationPaths


class CatalogueBookAssetResolver:
    def __init__(self, paths: ApplicationPaths) -> None:
        self._paths = paths

    def resolve(self, book: CatalogueBook) -> Path:
        if not is_catalogue_book_id(book.book_id):
            raise ValueError(f"Identifiant CatalogueBook invalide : {book.book_id!r}.")
        root = self._paths.active_catalogue_dir
        candidate = (root / book.asset_file).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Livre {book.book_id} : asset hors racine Catalogue.") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Livre {book.book_id} : asset introuvable : {candidate}")
        return candidate
