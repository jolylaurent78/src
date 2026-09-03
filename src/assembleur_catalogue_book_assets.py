"""Résolution physique des assets texte appartenant au Catalogue."""

from __future__ import annotations

from pathlib import Path

from src.assembleur_catalogue import CatalogueBook
from src.assembleur_catalogue_identity import is_system_catalogue_id, is_user_catalogue_id
from src.assembleur_paths import ApplicationPaths


class CatalogueBookAssetResolver:
    def __init__(self, paths: ApplicationPaths) -> None:
        self._paths = paths

    def resolve(self, book: CatalogueBook) -> Path:
        if is_system_catalogue_id(book.book_id):
            root = self._paths.default_catalogue_dir
        elif is_user_catalogue_id(book.book_id):
            root = self._paths.user_catalogue_dir
        else:
            raise ValueError(f"Identifiant CatalogueBook invalide : {book.book_id!r}.")
        candidate = (root / book.asset_file).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Livre {book.book_id} : asset hors racine Catalogue.") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Livre {book.book_id} : asset introuvable : {candidate}")
        return candidate
