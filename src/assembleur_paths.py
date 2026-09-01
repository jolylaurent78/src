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
            else Path(local_app_data) / _APPLICATION_NAME / "user-data"
            if local_app_data
            else Path.home() / "AppData" / "Local" / _APPLICATION_NAME / "user-data"
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
        return self.defaults_root / "catalogue.json"

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
    def user_scenarios_dir(self) -> Path:
        return self.user_data_root / "scenarios"

    @property
    def config_dir(self) -> Path:
        return self.user_data_root / "config"

    @property
    def config_path(self) -> Path:
        return self.config_dir / "assembleur_config.json"

    @property
    def calibrations_dir(self) -> Path:
        return self.user_data_root / "calibrations"

    @property
    def exports_dir(self) -> Path:
        return self.user_data_root / "exports"

    @property
    def logs_dir(self) -> Path:
        return self.user_data_root / "logs"

    def ensure_user_data_directories(self) -> None:
        for directory in (
            self.user_data_root,
            self.user_catalogue_dir,
            self.user_scenarios_dir,
            self.config_dir,
            self.calibrations_dir,
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
