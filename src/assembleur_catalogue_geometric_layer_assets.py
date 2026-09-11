"""Résolution et staging des assets physiques de calques géométriques."""

from __future__ import annotations

from pathlib import Path
import shutil

from src.assembleur_catalogue import Catalogue, CatalogueGeometricLayer
from src.assembleur_geometric_layer_io import GeometricLayerDocument, load_geometric_layer_document
from src.assembleur_paths import ApplicationPaths


class CatalogueGeometricLayerAssetResolver:
    """Résout un asset Catalogue sous le seul répertoire actif geometric-layers."""

    def __init__(self, paths: ApplicationPaths) -> None:
        self._paths = paths

    def resolve(self, asset_file: str) -> Path:
        prefix = "geometric-layers/"
        if not isinstance(asset_file, str) or not asset_file.startswith(prefix):
            raise ValueError(f"Référence d'asset de calque géométrique invalide : {asset_file!r}.")
        root = self._paths.active_catalogue_geometric_layers_dir.resolve()
        candidate = (root / asset_file.removeprefix(prefix)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Asset de calque géométrique hors de sa racine : {asset_file!r}.") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Asset du calque géométrique absent : {candidate}")
        return candidate

    def resolve_layer(self, base_city_id: str, catalogue: Catalogue) -> Path:
        layer = catalogue.get_geometric_layer(base_city_id)
        if layer is None:
            raise ValueError(f"La Base {base_city_id} ne possède aucun calque géométrique.")
        return self.resolve(layer.asset_file)


class CatalogueGeometricLayerAssetController:
    """Stage un import validé, puis publie l'asset et son association Catalogue."""

    def __init__(self, catalogue: Catalogue, paths: ApplicationPaths) -> None:
        self.catalogue = catalogue
        self.paths = paths
        self.resolver = CatalogueGeometricLayerAssetResolver(paths)
        self._staged: dict[str, Path] = {}
        self._backups: dict[Path, bytes] = {}
        self._scheduled_deletions: dict[str, tuple[CatalogueGeometricLayer, Path]] = {}
        self._previous_layers: dict[str, CatalogueGeometricLayer | None] = {}

    def rebind_catalogue(self, catalogue: Catalogue) -> None:
        self.catalogue = catalogue

    @staticmethod
    def asset_file_for(base_city_id: str) -> str:
        return f"geometric-layers/{base_city_id}.traces.json"

    def stage_geometric_layer(self, base_city_id: str, source_path: str | Path) -> GeometricLayerDocument:
        source = Path(source_path)
        document = load_geometric_layer_document(source)
        self.catalogue.get_city(base_city_id)
        if not self.catalogue.is_triangle_base_city(base_city_id):
            raise ValueError(f"Calque géométrique : la ville {base_city_id} n'est la Base d'aucun triangle.")
        staging_dir = self.paths.active_catalogue_geometric_layers_dir / ".staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged = staging_dir / f"{base_city_id}.traces.json"
        shutil.copy2(source, staged)
        previous = self._staged.get(base_city_id)
        if previous is not None and previous != staged:
            previous.unlink(missing_ok=True)
        self._staged[base_city_id] = staged
        self._scheduled_deletions.pop(base_city_id, None)
        return document

    def commit(self) -> list[Path]:
        created: list[Path] = []
        root = self.paths.active_catalogue_geometric_layers_dir
        root.mkdir(parents=True, exist_ok=True)
        self._previous_layers = {
            base_city_id: self.catalogue.get_geometric_layer(base_city_id)
            for base_city_id in self._staged
        }
        self._previous_layers.update({
            base_city_id: layer
            for base_city_id, (layer, _path) in self._scheduled_deletions.items()
        })
        try:
            for base_city_id, staged in self._staged.items():
                destination = root / f"{base_city_id}.traces.json"
                if destination.exists():
                    self._backups[destination] = destination.read_bytes()
                else:
                    created.append(destination)
                shutil.copy2(staged, destination)
                self.catalogue.set_geometric_layer(base_city_id, self.asset_file_for(base_city_id))
            for base_city_id, (_layer, destination) in self._scheduled_deletions.items():
                if destination.exists():
                    self._backups[destination] = destination.read_bytes()
                    destination.unlink()
                self.catalogue.remove_geometric_layer(base_city_id)
        except (OSError, ValueError, KeyError):
            self._restore_transaction(created)
            raise
        return created

    def delete_geometric_layer_asset(self, base_city_id: str) -> None:
        staged = self._staged.pop(base_city_id, None)
        if staged is not None:
            staged.unlink(missing_ok=True)
        layer = self.catalogue.get_geometric_layer(base_city_id)
        if layer is None:
            return
        self._scheduled_deletions[base_city_id] = (layer, self.resolver.resolve(layer.asset_file))

    def rollback(self, created: list[Path]) -> None:
        self._restore_transaction(created)

    def _restore_transaction(self, created: list[Path]) -> None:
        for destination, content in self._backups.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        for destination in created:
            destination.unlink(missing_ok=True)
        for base_city_id, previous in self._previous_layers.items():
            if previous is None:
                self.catalogue.remove_geometric_layer(base_city_id)
            else:
                self.catalogue.geometric_layers[base_city_id] = previous
        self._backups.clear()
        self._previous_layers.clear()

    def finalize_commit(self) -> None:
        self.discard()

    def discard(self) -> None:
        for staged in self._staged.values():
            staged.unlink(missing_ok=True)
        self._staged.clear()
        self._backups.clear()
        self._scheduled_deletions.clear()
        self._previous_layers.clear()
