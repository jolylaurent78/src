"""Résolution centralisée des ressources, defaults et données utilisateur."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping


_APPLICATION_NAME = "AssembleurTriangles"


@dataclass(frozen=True)
class ApplicationPaths:
    """Chemins runtime indépendants du répertoire courant."""

    installation_root: Path
    user_data_root: Path

    @classmethod
    def from_runtime(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        installation_root: str | Path | None = None,
        user_data_root: str | Path | None = None,
    ) -> "ApplicationPaths":
        source = os.environ if environ is None else environ
        resolved_installation_root = (
            Path(installation_root)
            if installation_root is not None
            else cls._runtime_installation_root()
        )
        local_app_data = source.get("LOCALAPPDATA")
        resolved_user_data_root = (
            Path(user_data_root)
            if user_data_root is not None
            else Path(local_app_data) / _APPLICATION_NAME
            if local_app_data
            else Path.home() / "AppData" / "Local" / _APPLICATION_NAME
        )
        return cls(resolved_installation_root.resolve(), resolved_user_data_root.resolve())

    @staticmethod
    def _runtime_installation_root() -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent.parent

    @property
    def resource_root(self) -> Path:
        return self.installation_root / "resources"

    @property
    def defaults_root(self) -> Path:
        return self.installation_root / "defaults"

    @property
    def images_dir(self) -> Path:
        return self.resource_root / "images"

    @property
    def resource_maps_dir(self) -> Path:
        return self.resource_root / "maps"

    @property
    def resource_texts_dir(self) -> Path:
        return self.resource_root / "texts"

    @property
    def dictionary_path(self) -> Path:
        return self.resource_texts_dir / "livre.txt"

    @property
    def default_catalogue_path(self) -> Path:
        return self.default_catalogue_dir / "catalogue.json"

    @property
    def default_catalogue_dir(self) -> Path:
        """Repertoire du Catalogue SYS distribue avec l'application."""
        return self.defaults_root / "catalogue"

    @property
    def default_catalogue_maps_dir(self) -> Path:
        """Assets physiques des cartes appartenant au Catalogue SYS."""
        return self.default_catalogue_dir / "maps"

    @property
    def default_scenarios_dir(self) -> Path:
        return self.defaults_root / "scenarios"

    @property
    def default_config_path(self) -> Path:
        return self.defaults_root / "config" / "assembleur_config.json"

    @property
    def user_catalogue_dir(self) -> Path:
        return self.user_data_root / "catalogue"

    @property
    def user_catalogue_path(self) -> Path:
        return self.user_catalogue_dir / "catalogue.json"

    @property
    def user_catalogue_maps_dir(self) -> Path:
        """Répertoire des assets physiques appartenant au Catalogue USER."""
        return self.user_catalogue_dir / "maps"

    @property
    def user_scenarios_dir(self) -> Path:
        return self.user_data_root / "scenarios"

    @property
    def config_dir(self) -> Path:
        return self.user_data_root / "config"

    @property
    def config_path(self) -> Path:
        return self.config_dir / "assembleur_config.json"

    @property
    def exports_dir(self) -> Path:
        return self.user_data_root / "exports"

    @property
    def logs_dir(self) -> Path:
        return self.user_data_root / "logs"

    @staticmethod
    def _merge_legacy_directory(source: Path, destination: Path) -> None:
        """Fusionne un répertoire legacy sans écraser de données existantes."""
        if not source.is_dir():
            raise NotADirectoryError(
                f"Répertoire legacy attendu, fichier trouvé : {source}"
            )
        if not destination.exists():
            shutil.move(str(source), str(destination))
            return
        if not destination.is_dir():
            raise FileExistsError(
                "Migration du layout utilisateur impossible : types incompatibles "
                f"entre {source} et {destination}"
            )

        for source_entry in source.iterdir():
            destination_entry = destination / source_entry.name
            if not destination_entry.exists():
                shutil.move(str(source_entry), str(destination_entry))
                continue
            if source_entry.is_dir() and destination_entry.is_dir():
                ApplicationPaths._merge_legacy_directory(source_entry, destination_entry)
                continue
            if source_entry.is_file() and destination_entry.is_file():
                if source_entry.read_bytes() != destination_entry.read_bytes():
                    raise FileExistsError(
                        "Migration du layout utilisateur impossible : fichiers "
                        f"différents : {source_entry} et {destination_entry}"
                    )
                source_entry.unlink()
                continue
            raise FileExistsError(
                "Migration du layout utilisateur impossible : types incompatibles "
                f"entre {source_entry} et {destination_entry}"
            )

        source.rmdir()

    @staticmethod
    def _legacy_collision_path(destination: Path) -> Path:
        """Retourne un nom ``.legacy`` libre en conservant l'extension finale."""
        suffix = destination.suffix
        stem = destination.name[:-len(suffix)] if suffix else destination.name
        candidate = destination.with_name(f"{stem}.legacy{suffix}")
        index = 2
        while candidate.exists():
            candidate = destination.with_name(f"{stem}.legacy-{index}{suffix}")
            index += 1
        return candidate

    @staticmethod
    def _merge_legacy_runtime_directory(source: Path, destination: Path) -> None:
        """Fusionne des données runtime mutables sans perdre de collision."""
        if not source.is_dir():
            raise NotADirectoryError(
                f"Répertoire legacy attendu, fichier trouvé : {source}"
            )
        if not destination.exists():
            shutil.move(str(source), str(destination))
            return
        if not destination.is_dir():
            raise FileExistsError(
                "Migration du layout utilisateur impossible : types incompatibles "
                f"entre {source} et {destination}"
            )

        for source_entry in source.iterdir():
            destination_entry = destination / source_entry.name
            if not destination_entry.exists():
                shutil.move(str(source_entry), str(destination_entry))
                continue
            if source_entry.is_dir() and destination_entry.is_dir():
                ApplicationPaths._merge_legacy_runtime_directory(
                    source_entry, destination_entry
                )
                continue
            if source_entry.is_file() and destination_entry.is_file():
                if source_entry.read_bytes() == destination_entry.read_bytes():
                    source_entry.unlink()
                else:
                    shutil.move(
                        str(source_entry),
                        str(ApplicationPaths._legacy_collision_path(destination_entry)),
                    )
                continue
            raise FileExistsError(
                "Migration du layout utilisateur impossible : types incompatibles "
                f"entre {source_entry} et {destination_entry}"
            )

        source.rmdir()

    def migrate_legacy_user_data_layout(self) -> None:
        """Migre de façon rejouable l'ancien layout utilisateur imbriqué."""
        legacy_roots = (
            self.user_data_root / "user_data",
            self.user_data_root / "user-data",
        )
        strict_directories = (
            "catalogue",
            "scenarios",
            "config",
            "exports",
            "reports",
        )
        runtime_directories = (
            "logs",
            "cache",
            "temp",
        )
        for legacy_root in legacy_roots:
            if not legacy_root.exists():
                continue
            if not legacy_root.is_dir():
                raise NotADirectoryError(
                    f"Racine utilisateur legacy invalide : {legacy_root}"
                )
            for name in strict_directories:
                source = legacy_root / name
                if not source.exists():
                    continue
                destination = self.user_data_root / name
                self._merge_legacy_directory(source, destination)
            for name in runtime_directories:
                source = legacy_root / name
                if not source.exists():
                    continue
                destination = self.user_data_root / name
                self._merge_legacy_runtime_directory(source, destination)
            try:
                legacy_root.rmdir()
            except OSError:
                # Des fichiers inconnus sont volontairement conservés dans le dossier legacy.
                pass

    def ensure_user_data_directories(self) -> None:
        self.migrate_legacy_user_data_layout()
        for directory in (
            self.user_data_root,
            self.user_catalogue_dir,
            self.user_catalogue_maps_dir,
            self.user_scenarios_dir,
            self.config_dir,
            self.exports_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def catalogue_path_for_mode(self, mode: str) -> Path:
        if mode == "SYS":
            if not self.default_catalogue_path.is_file():
                raise FileNotFoundError(
                    f"Catalogue de référence absent : {self.default_catalogue_path}"
                )
            return self.default_catalogue_path
        if mode != "USER":
            raise ValueError(f"Mode Assembleur inconnu : {mode!r}")
        self.ensure_user_data_directories()
        if self.user_catalogue_path.exists():
            return self.user_catalogue_path
        if not self.default_catalogue_path.is_file():
            raise FileNotFoundError(
                "Catalogue de référence absent pour initialiser la copie utilisateur : "
                f"{self.default_catalogue_path}"
            )
        shutil.copy2(self.default_catalogue_path, self.user_catalogue_path)
        return self.user_catalogue_path

    def config_path_for_runtime(self) -> Path:
        """Retourne la config mutable, initialisée une seule fois depuis le seed."""
        self.ensure_user_data_directories()
        if self.config_path.exists():
            return self.config_path
        if not self.default_config_path.is_file():
            raise FileNotFoundError(
                "Configuration par défaut absente pour initialiser la configuration utilisateur : "
                f"{self.default_config_path}"
            )
        shutil.copy2(self.default_config_path, self.config_path)
        return self.config_path
