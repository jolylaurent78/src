"""Résolution runtime des assets physiques d'une :class:`CatalogueMap`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.assembleur_catalogue import CatalogueMap
from src.assembleur_catalogue_identity import (
    is_catalogue_map_id,
    is_system_catalogue_id,
    is_user_catalogue_id,
)
from src.assembleur_paths import ApplicationPaths


@dataclass(frozen=True)
class ResolvedCatalogueMapAssets:
    """Assets physiques résolus et validés pour une carte Catalogue."""

    map_id: str
    image_path: Path
    calibration_points_path: Path | None
    calibration_path: Path | None


class CatalogueMapAssetResolver:
    """Résout les références logiques MAP-SYS/MAP-USR sans fallback inter-racines."""

    def __init__(self, paths: ApplicationPaths) -> None:
        self._paths = paths

    def resolve(self, catalogue_map: CatalogueMap) -> ResolvedCatalogueMapAssets:
        if not isinstance(catalogue_map, CatalogueMap):
            raise TypeError("La résolution d'assets exige un CatalogueMap.")
        if not is_catalogue_map_id(catalogue_map.map_id):
            raise ValueError(f"Identifiant CatalogueMap invalide : {catalogue_map.map_id!r}.")
        root = self._asset_root_for_map(catalogue_map.map_id)
        return ResolvedCatalogueMapAssets(
            map_id=catalogue_map.map_id,
            image_path=self._resolve_required_asset(
                root, catalogue_map.map_id, "image", catalogue_map.image_file
            ),
            calibration_points_path=self._resolve_optional_asset(
                root,
                catalogue_map.map_id,
                "calibration points",
                catalogue_map.calibration_points_file,
            ),
            calibration_path=self._resolve_optional_asset(
                root,
                catalogue_map.map_id,
                "calibration",
                catalogue_map.calibration_file,
            ),
        )

    def _asset_root_for_map(self, map_id: str) -> Path:
        if is_system_catalogue_id(map_id):
            return self._paths.default_catalogue_maps_dir
        if is_user_catalogue_id(map_id):
            return self._paths.user_catalogue_maps_dir
        raise ValueError(f"Namespace CatalogueMap non supporté : {map_id!r}.")

    @staticmethod
    def _resolve_required_asset(root: Path, map_id: str, role: str, reference: str) -> Path:
        return CatalogueMapAssetResolver._resolve_asset(root, map_id, role, reference)

    @staticmethod
    def _resolve_optional_asset(root: Path, map_id: str, role: str, reference: str | None) -> Path | None:
        if reference is None:
            return None
        return CatalogueMapAssetResolver._resolve_asset(root, map_id, role, reference)

    @staticmethod
    def _resolve_asset(root: Path, map_id: str, role: str, reference: str) -> Path:
        resolved_root = root.resolve()
        candidate = (resolved_root / reference).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"Carte {map_id} : référence {role} hors de sa racine d'assets : {reference!r}."
            ) from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Carte {map_id} : asset {role} introuvable : {candidate}")
        return candidate


def normalize_supported_projection(value: object, *, context: str) -> str:
    """Normalise uniquement les deux libellés historiques Lambert-93 autorisés."""
    if not isinstance(value, str):
        raise ValueError(f"{context} : projection absente ou invalide : {value!r}.")
    normalized = value.strip().casefold()
    if normalized in {"epsg:2154", "lambert93"}:
        return "EPSG:2154"
    raise ValueError(f"{context} : projection non supportée : {value!r}.")


def load_calibrated_catalogue_map(
    catalogue_map: CatalogueMap,
    resolver: CatalogueMapAssetResolver,
    *,
    max_image_dimension: int | None = None,
):
    """Charge une CatalogueMap calibrée et vérifie sa projection persistante."""
    if catalogue_map.calibration_file is None or catalogue_map.projection is None:
        raise ValueError(f"La carte {catalogue_map.map_id} n'est pas calibrée.")
    assets = resolver.resolve(catalogue_map)
    if assets.calibration_path is None:
        raise ValueError(f"La carte {catalogue_map.map_id} n'est pas calibrée.")
    from src.assembleur_geo_map_view import CalibratedGeoMap

    geo_map = CalibratedGeoMap.load_from_assets(
        map_id=catalogue_map.map_id,
        image_path=assets.image_path,
        calibration_path=assets.calibration_path,
        max_image_dimension=max_image_dimension,
    )
    expected_projection = normalize_supported_projection(
        catalogue_map.projection,
        context=f"Carte {catalogue_map.map_id}",
    )
    actual_projection = normalize_supported_projection(
        geo_map.projection,
        context=f"Calibration de la carte {catalogue_map.map_id}",
    )
    if actual_projection != expected_projection:
        raise ValueError(
            f"Carte {catalogue_map.map_id} : projection Catalogue {expected_projection} "
            f"incohérente avec la calibration {actual_projection}."
        )
    return geo_map
