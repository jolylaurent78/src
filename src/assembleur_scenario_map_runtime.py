"""Résolution pure de la carte effective d'un scénario."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.assembleur_catalogue import Catalogue, CatalogueMap, WorldRect
from src.assembleur_catalogue_map_assets import (
    CatalogueMapAssetResolver,
    load_calibrated_catalogue_map,
)
from src.assembleur_map_transform import (
    MapTransform,
    validate_world_rect,
    validate_world_rect_aspect,
    world_rect_for_scale,
)
from src.assembleur_scenario_map import ScenarioMapState

if TYPE_CHECKING:
    from src.assembleur_geo_map_view import CalibratedGeoMap


@dataclass(frozen=True)
class ResolvedScenarioMap:
    """Carte calibrée effectivement utilisable par un scénario."""

    map_id: str
    catalogue_map: CatalogueMap
    calibrated_map: "CalibratedGeoMap"
    world_rect: WorldRect
    scale_factor: float
    visible: bool
    opacity: float
    transform: MapTransform


class ScenarioMapResolver:
    """Résout des ``ScenarioMapState`` sans accéder au viewer ni au XML legacy."""

    def __init__(self, catalogue: Catalogue, asset_resolver: CatalogueMapAssetResolver) -> None:
        if not isinstance(catalogue, Catalogue):
            raise TypeError("ScenarioMapResolver exige un Catalogue.")
        if not isinstance(asset_resolver, CatalogueMapAssetResolver):
            raise TypeError("ScenarioMapResolver exige un CatalogueMapAssetResolver.")
        self._catalogue = catalogue
        self._asset_resolver = asset_resolver

    def resolve(
        self,
        scenario_map_state: ScenarioMapState,
        *,
        max_image_dimension: int | None = None,
    ) -> ResolvedScenarioMap | None:
        return resolve_scenario_map(
            self._catalogue,
            scenario_map_state,
            self._asset_resolver,
            max_image_dimension=max_image_dimension,
        )


def resolve_scenario_map(
    catalogue: Catalogue,
    scenario_map_state: ScenarioMapState,
    asset_resolver: CatalogueMapAssetResolver,
    *,
    max_image_dimension: int | None = None,
) -> ResolvedScenarioMap | None:
    """Résout une carte de scénario calibrée, ou ``None`` sans référence.

    Les cartes archivées sont volontairement chargées : une référence déjà
    enregistrée reste lisible. Les cartes non calibrées échouent explicitement
    via ``load_calibrated_catalogue_map`` ; leur support visuel est différé.
    """
    if not isinstance(catalogue, Catalogue):
        raise TypeError("resolve_scenario_map exige un Catalogue.")
    if not isinstance(scenario_map_state, ScenarioMapState):
        raise TypeError("resolve_scenario_map exige un ScenarioMapState.")
    if not isinstance(asset_resolver, CatalogueMapAssetResolver):
        raise TypeError("resolve_scenario_map exige un CatalogueMapAssetResolver.")
    if scenario_map_state.map_ref_id is None:
        return None

    catalogue_map = catalogue.get_map(scenario_map_state.map_ref_id)
    calibrated_map = load_calibrated_catalogue_map(
        catalogue_map,
        asset_resolver,
        max_image_dimension=max_image_dimension,
    )
    image_width, image_height = calibrated_map.image_size
    image_aspect = image_width / image_height
    default_rect = validate_world_rect(catalogue_map.default_world_rect, label="default_world_rect")
    validate_world_rect_aspect(default_rect, image_aspect)

    position = scenario_map_state.position_override
    if scenario_map_state.scale_factor_override is not None:
        scale_factor = scenario_map_state.scale_factor_override
        world_rect = world_rect_for_scale(
            default_rect,
            catalogue_map.default_scale_factor,
            scale_factor,
            image_aspect_ratio=image_aspect,
            x0=None if position is None else position.x0,
            y0=None if position is None else position.y0,
        )
    else:
        # Compatibilité : la pose livrée est normative même si son facteur
        # historique calculé (11.983520...) diffère du facteur métier 12.0.
        world_rect = (
            default_rect
            if position is None
            else WorldRect(position.x0, position.y0, default_rect.w, default_rect.h)
        )
        scale_factor = catalogue_map.default_scale_factor

    transform = MapTransform(calibrated_map, world_rect)
    return ResolvedScenarioMap(
        map_id=catalogue_map.map_id,
        catalogue_map=catalogue_map,
        calibrated_map=calibrated_map,
        world_rect=world_rect,
        scale_factor=scale_factor,
        visible=scenario_map_state.visible,
        opacity=scenario_map_state.opacity,
        transform=transform,
    )
