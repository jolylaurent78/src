"""Etat technique et recalcul des calibrations de :class:`CatalogueMap`.

Le Catalogue ne conserve que les villes choisies.  Ce module conserve les
pixels et produit l'affine consommée par ``CalibratedGeoMap``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from src.assembleur_catalogue import Catalogue, CatalogueMap, centered_world_rect
from src.assembleur_catalogue_identity import is_system_catalogue_id
from src.assembleur_catalogue_map_assets import CatalogueMapAssetResolver
from src.assembleur_paths import ApplicationPaths


STATUS_UNCALIBRATED = "Non calibrée"
STATUS_INCOMPLETE = "Incomplète"
STATUS_VALID = "Valide"
STATUS_ARCHIVED = "Archivée"


@dataclass(frozen=True)
class CalibrationPoint:
    city_id: str
    pixel_x: float
    pixel_y: float


@dataclass(frozen=True)
class CalibrationResidual:
    """Diagnostic leave-one-out d'un point de calibration observe."""

    city_id: str
    observed_x: float
    observed_y: float
    predicted_x: float
    predicted_y: float
    dx: float
    dy: float
    error_px: float


def _points_from_payload(payload: object) -> dict[str, CalibrationPoint]:
    if not isinstance(payload, dict):
        return {}
    raw_points = payload.get("points", [])
    if not isinstance(raw_points, list):
        return {}
    points: dict[str, CalibrationPoint] = {}
    for raw in raw_points:
        if not isinstance(raw, dict):
            continue
        city_id, x, y = raw.get("cityId"), raw.get("pixelX"), raw.get("pixelY")
        if (
            isinstance(city_id, str)
            and not isinstance(x, bool)
            and not isinstance(y, bool)
            and isinstance(x, (int, float))
            and isinstance(y, (int, float))
            and np.isfinite(float(x))
            and np.isfinite(float(y))
        ):
            points[city_id] = CalibrationPoint(city_id, float(x), float(y))
    return points


def calibration_status(
    catalogue_map: CatalogueMap,
    points: dict[str, CalibrationPoint],
    payload: dict[str, Any] | None = None,
) -> str:
    """Retourne le statut dérivé, sans le persister."""
    if catalogue_map.archived:
        return STATUS_ARCHIVED
    if not catalogue_map.calibration_city_ids:
        return STATUS_UNCALIBRATED
    if catalogue_map.projection != "EPSG:2154" or len(points) < 3:
        return STATUS_INCOMPLETE
    if not set(catalogue_map.calibration_city_ids).issubset(points):
        return STATUS_INCOMPLETE
    if payload is not None:
        if {"affineWorldToLambertKm", "bgWorldRectAtCalibration"}.issubset(payload):
            return STATUS_VALID
        try:
            matrix = np.asarray(payload["A"], dtype=float)
            offset = np.asarray(payload["offset"], dtype=float)
            if matrix.shape != (2, 2) or offset.shape != (2,) or abs(float(np.linalg.det(matrix))) < 1e-15:
                return STATUS_INCOMPLETE
        except (KeyError, TypeError, ValueError):
            return STATUS_INCOMPLETE
    return STATUS_VALID


class CatalogueMapCalibrationController:
    """Contrôleur non-Tk : points, résolution affine et staging des assets USER."""

    def __init__(
        self,
        catalogue: Catalogue,
        paths: ApplicationPaths,
        *,
        allow_system_map_editing: bool = False,
    ) -> None:
        self.catalogue = catalogue
        self.paths = paths
        self._allow_system_map_editing = bool(allow_system_map_editing)
        self.resolver = CatalogueMapAssetResolver(paths)
        self._documents: dict[str, dict[str, Any]] = {}
        self._staged_images: dict[str, Path] = {}
        self._committed_backups: dict[Path, bytes] = {}

    def is_readonly(self, catalogue_map: CatalogueMap) -> bool:
        return is_system_catalogue_id(catalogue_map.map_id) and not self._allow_system_map_editing

    def rebind_catalogue(self, catalogue: Catalogue) -> None:
        """Change l'agrégat métier sans abandonner la transaction cartographique."""
        self.catalogue = catalogue

    def points_for(self, catalogue_map: CatalogueMap) -> dict[str, CalibrationPoint]:
        if catalogue_map.map_id not in self._documents:
            payload: dict[str, Any] = {}
            if catalogue_map.calibration_file is not None:
                try:
                    path = self.resolver.resolve(catalogue_map).calibration_path
                    if path is not None:
                        loaded = json.loads(path.read_text(encoding="utf-8"))
                        if isinstance(loaded, dict):
                            payload = loaded
                except (OSError, ValueError, json.JSONDecodeError):
                    payload = {}
            self._documents[catalogue_map.map_id] = payload
        return _points_from_payload(self._documents[catalogue_map.map_id])

    def status_for(self, catalogue_map: CatalogueMap) -> str:
        points = self.points_for(catalogue_map)
        return calibration_status(catalogue_map, points, self._documents[catalogue_map.map_id])

    def preview_map(self, catalogue_map: CatalogueMap):
        """Charge une carte affichable, même lorsqu'elle n'est pas encore calibrée."""
        from PIL import Image
        from src.assembleur_catalogue_map_assets import load_calibrated_catalogue_map
        from src.assembleur_geo_map_view import CalibratedGeoMap

        image_path = self._preview_image_path(catalogue_map)
        payload = self._documents.get(catalogue_map.map_id)
        if isinstance(payload, dict) and "A" in payload and "offset" in payload:
            with Image.open(image_path) as image:
                return CalibratedGeoMap(catalogue_map.map_id, image.copy(), payload)
        if catalogue_map.projection == "EPSG:2154":
            try:
                return load_calibrated_catalogue_map(catalogue_map, self.resolver)
            except (OSError, ValueError):
                pass
        with Image.open(image_path) as image:
            return CalibratedGeoMap(
                catalogue_map.map_id,
                image.copy(),
                {"projection": "EPSG:2154", "A": [[1.0, 0.0], [0.0, 1.0]], "offset": [0.0, 0.0]},
            )

    def leave_one_out_residuals(self, catalogue_map: CatalogueMap) -> dict[str, CalibrationResidual]:
        """Calcule les diagnostics sans modifier la calibration principale."""
        points = self.points_for(catalogue_map)
        selected = [points[city_id] for city_id in catalogue_map.calibration_city_ids if city_id in points]
        if len(selected) < 4:
            return {}
        residuals: dict[str, CalibrationResidual] = {}
        for observed in selected:
            others = [point for point in selected if point.city_id != observed.city_id]
            try:
                matrix, offset = self._solve_affine(others)
            except ValueError:
                continue
            lambert_x, lambert_y = self.catalogue.get_city_lambert(observed.city_id)
            predicted_x, predicted_y = matrix @ np.asarray((lambert_x, lambert_y), dtype=float) + offset
            dx = float(predicted_x - observed.pixel_x)
            dy = float(predicted_y - observed.pixel_y)
            residuals[observed.city_id] = CalibrationResidual(
                city_id=observed.city_id,
                observed_x=observed.pixel_x,
                observed_y=observed.pixel_y,
                predicted_x=float(predicted_x),
                predicted_y=float(predicted_y),
                dx=dx,
                dy=dy,
                error_px=float(np.hypot(dx, dy)),
            )
        return residuals

    def _preview_image_path(self, catalogue_map: CatalogueMap) -> Path:
        if catalogue_map.map_id in self._staged_images:
            return self._staged_images[catalogue_map.map_id]
        return self.resolver.resolve(catalogue_map).image_path

    def set_pixel(self, map_id: str, city_id: str, pixel_x: float, pixel_y: float) -> None:
        catalogue_map = self.catalogue.get_map(map_id)
        if self.is_readonly(catalogue_map):
            raise ValueError("Les cartes SYS sont consultables uniquement.")
        if city_id not in catalogue_map.calibration_city_ids:
            raise ValueError("La ville ne fait pas partie de la calibration de cette carte.")
        if not np.isfinite(pixel_x) or not np.isfinite(pixel_y):
            raise ValueError("Les coordonnées pixel doivent être finies.")
        payload = self._document_for(catalogue_map)
        points = self.points_for(catalogue_map)
        points[city_id] = CalibrationPoint(city_id, float(pixel_x), float(pixel_y))
        self._store_points(catalogue_map, payload, points)
        self._recalculate(catalogue_map, payload, points)

    def remove_city(self, map_id: str, city_id: str) -> None:
        catalogue_map = self.catalogue.get_map(map_id)
        if self.is_readonly(catalogue_map):
            raise ValueError("Les cartes SYS sont consultables uniquement.")
        city_ids = [item for item in catalogue_map.calibration_city_ids if item != city_id]
        self.catalogue.update_map(map_id, calibration_city_ids=city_ids)
        payload = self._document_for(catalogue_map)
        points = self.points_for(catalogue_map)
        points.pop(city_id, None)
        self._store_points(catalogue_map, payload, points)
        self._recalculate(catalogue_map, payload, points)

    def add_city(self, map_id: str, city_id: str) -> None:
        catalogue_map = self.catalogue.get_map(map_id)
        if self.is_readonly(catalogue_map):
            raise ValueError("Les cartes SYS sont consultables uniquement.")
        if city_id not in self.catalogue.cities or self.catalogue.get_city(city_id).archived:
            raise ValueError("La ville de calibration doit être active et appartenir au Catalogue.")
        if city_id in catalogue_map.calibration_city_ids:
            raise ValueError("Cette ville est déjà utilisée pour la calibration.")
        self.catalogue.update_map(map_id, calibration_city_ids=[*catalogue_map.calibration_city_ids, city_id])

    def stage_map(self, image_path: str | Path, *, name: str, description: str) -> str:
        source = Path(image_path)
        if not source.is_file():
            raise FileNotFoundError(f"Image de carte introuvable : {source}")
        suffix = source.suffix.lower() or ".png"
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("L'image de carte doit être JPG ou PNG.")
        provisional = self.catalogue.add_map(
            name=name,
            image_file="staging" + suffix,
            calibration_file=None,
            projection=None,
            default_world_rect=centered_world_rect(1, 1),
            default_scale_factor=1.0,
            description=description,
        )
        catalogue_map = self.catalogue.get_map(provisional)
        image_file = f"{provisional}{suffix}"
        calibration_file = f"{provisional}.json"
        self.catalogue.update_map(
            provisional,
            image_file=image_file,
            calibration_file=calibration_file,
            projection=None,
        )
        staging_dir = self.paths.active_catalogue_maps_dir / ".staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged = staging_dir / image_file
        shutil.copy2(source, staged)
        self._staged_images[provisional] = staged
        self._documents[provisional] = {"points": []}
        return catalogue_map.map_id

    # Compatibilite locale pour les appels existants pendant la transition de
    # nommage : la destination depend desormais de la racine active.
    stage_user_map = stage_map

    def commit(self) -> list[Path]:
        """Publie les fichiers différés ; les chemins retournés sont rollbackables."""
        created: list[Path] = []
        root = self.paths.active_catalogue_maps_dir
        root.mkdir(parents=True, exist_ok=True)
        for map_id, staged in self._staged_images.items():
            destination = root / self.catalogue.get_map(map_id).image_file
            if destination.exists():
                self._committed_backups[destination] = destination.read_bytes()
            else:
                created.append(destination)
            shutil.copy2(staged, destination)
        for map_id, payload in self._documents.items():
            catalogue_map = self.catalogue.get_map(map_id)
            if catalogue_map.calibration_file is None:
                continue
            if is_system_catalogue_id(map_id) and not self._allow_system_map_editing:
                continue
            destination = root / catalogue_map.calibration_file
            if destination.exists():
                self._committed_backups[destination] = destination.read_bytes()
            else:
                created.append(destination)
            self._write_json_atomic(destination, payload)
        return created

    def discard(self) -> None:
        """Abandonne l'état différé d'une session non appliquée."""
        for staged in self._staged_images.values():
            staged.unlink(missing_ok=True)
        self._staged_images.clear()
        self._documents.clear()
        self._committed_backups.clear()

    def finalize_commit(self) -> None:
        """Nettoie uniquement le staging après publication réussie des assets."""
        for staged in self._staged_images.values():
            staged.unlink(missing_ok=True)
        self._staged_images.clear()
        self._documents.clear()
        self._committed_backups.clear()

    def rollback(self, paths: list[Path]) -> None:
        for path, content in self._committed_backups.items():
            path.write_bytes(content)
        for path in paths:
            path.unlink(missing_ok=True)
        self._committed_backups.clear()

    def _document_for(self, catalogue_map: CatalogueMap) -> dict[str, Any]:
        self.points_for(catalogue_map)
        return self._documents[catalogue_map.map_id]

    def _store_points(
        self,
        catalogue_map: CatalogueMap,
        payload: dict[str, Any],
        points: dict[str, CalibrationPoint],
    ) -> None:
        payload["points"] = [
            {"cityId": item.city_id, "pixelX": item.pixel_x, "pixelY": item.pixel_y}
            for item in (points[city_id] for city_id in catalogue_map.calibration_city_ids if city_id in points)
        ]

    def _recalculate(
        self,
        catalogue_map: CatalogueMap,
        payload: dict[str, Any],
        points: dict[str, CalibrationPoint],
    ) -> None:
        selected = [points[city_id] for city_id in catalogue_map.calibration_city_ids if city_id in points]
        if len(selected) < 3:
            payload.pop("A", None)
            payload.pop("offset", None)
            payload.pop("projection", None)
            self.catalogue.update_map(catalogue_map.map_id, projection=None)
            return
        matrix, offset = self._solve_affine(selected)
        payload["projection"] = "EPSG:2154"
        payload["A"] = matrix.tolist()
        payload["offset"] = offset.tolist()
        self.catalogue.update_map(catalogue_map.map_id, projection="EPSG:2154")

    def _solve_affine(self, points: list[CalibrationPoint]) -> tuple[np.ndarray, np.ndarray]:
        """Resout l'affine Lambert-93 vers pixels, sans effet de bord."""
        if len(points) < 3:
            raise ValueError("Au moins trois villes de calibration sont requises.")
        source = np.asarray(
            [(*self.catalogue.get_city_lambert(point.city_id), 1.0) for point in points],
            dtype=float,
        )
        target = np.asarray([(point.pixel_x, point.pixel_y) for point in points], dtype=float)
        coefficients, _residuals, rank, _singular = np.linalg.lstsq(source, target, rcond=None)
        if rank < 3:
            raise ValueError("Les villes de calibration sont geometriquement degeneres.")
        matrix = coefficients[:2, :].T
        offset = coefficients[2, :]
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
            raise ValueError("La calibration calculee doit etre finie.")
        if abs(float(np.linalg.det(matrix))) < 1e-15:
            raise ValueError("La calibration calculee n'est pas inversible.")
        return matrix, offset

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
