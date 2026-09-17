"""Runtime controller for the scenario background map."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

from src.assembleur_background_map_layer import BackgroundMapLayer, BackgroundMapWorldRect
from src.assembleur_catalogue import Catalogue, WorldRect
from src.assembleur_catalogue_map_assets import CatalogueMapAssetResolver
from src.assembleur_map_transform import MapTransform, scale_factor_for_world_rect
from src.assembleur_scenario_map import ScenarioMapPosition, ScenarioMapState
from src.assembleur_scenario_map_runtime import ResolvedScenarioMap, ScenarioMapResolver


class ScenarioMapController:
    """Adapts a scenario map state to the background map runtime layer."""

    def __init__(
        self,
        catalogue: Catalogue,
        paths,
        background_map_layer: BackgroundMapLayer,
        active_scenario_provider: Callable[[], object | None],
    ) -> None:
        self._paths = paths
        self._background_map_layer = background_map_layer
        self._active_scenario_provider = active_scenario_provider
        self._catalogue = catalogue
        self._assets = CatalogueMapAssetResolver(paths)
        self._resolver = ScenarioMapResolver(catalogue, self._assets)
        self._resolved_map: ResolvedScenarioMap | None = None

    @property
    def resolved_map(self) -> ResolvedScenarioMap | None:
        return self._resolved_map

    @property
    def scale_factor(self) -> float | None:
        return None if self._resolved_map is None else self._resolved_map.scale_factor

    def set_catalogue(self, catalogue: Catalogue) -> None:
        self._catalogue = catalogue
        self._assets = CatalogueMapAssetResolver(self._paths)
        self._resolver = ScenarioMapResolver(catalogue, self._assets)
        scenario = self._active_scenario_provider()
        if scenario is not None and isinstance(scenario.map_state, ScenarioMapState):
            self.apply_state(scenario.map_state)
        else:
            self._resolved_map = None
            self._background_map_layer.clear()

    def new_default_state(self) -> ScenarioMapState:
        return ScenarioMapState(map_ref_id=self._catalogue.default_map_id)

    def capture_active_state(self) -> ScenarioMapState:
        scenario = self._active_scenario_provider()
        if scenario is not None and isinstance(scenario.map_state, ScenarioMapState):
            return scenario.map_state
        return self.new_default_state()

    def apply_state(self, state: ScenarioMapState) -> None:
        if not isinstance(state, ScenarioMapState):
            raise TypeError("apply_state exige un ScenarioMapState.")
        resolved = self._resolver.resolve(state)
        self._resolved_map = resolved
        if resolved is None:
            self._background_map_layer.clear()
            return

        assets = self._assets.resolve(resolved.catalogue_map)
        rect = resolved.world_rect
        self._background_map_layer.set_map(
            resolved.calibrated_map.image.convert("RGBA"),
            BackgroundMapWorldRect(rect.x0, rect.y0, rect.w, rect.h),
            str(Path(assets.image_path)),
        )

    def lambert_to_world(
        self, lambert_x_m: float, lambert_y_m: float
    ) -> tuple[float, float]:
        if self._resolved_map is None:
            raise RuntimeError("Aucune carte calibrée active pour résoudre les balises Catalogue.")
        return self._resolved_map.transform.lambert_to_world(lambert_x_m, lambert_y_m)

    def sync_active_state_from_background(self) -> None:
        rect = self._background_map_layer.world_rect
        scenario = self._active_scenario_provider()
        if rect is None or scenario is None:
            return
        state = scenario.map_state
        if not isinstance(state, ScenarioMapState) or state.map_ref_id is None:
            return

        catalogue_map = self._catalogue.get_map(state.map_ref_id)
        default = catalogue_map.default_world_rect
        world_rect = WorldRect(rect.x0, rect.y0, rect.w, rect.h)
        scale = scale_factor_for_world_rect(
            world_rect, default, catalogue_map.default_scale_factor
        )
        same_position = (
            abs(world_rect.x0 - default.x0) < 1e-9
            and abs(world_rect.y0 - default.y0) < 1e-9
        )
        same_size = (
            abs(world_rect.w - default.w) < 1e-9
            and abs(world_rect.h - default.h) < 1e-9
        )
        updated = ScenarioMapState(
            map_ref_id=state.map_ref_id,
            position_override=(
                None if same_position else ScenarioMapPosition(world_rect.x0, world_rect.y0)
            ),
            scale_factor_override=None if same_size else scale,
            visible=state.visible,
        )
        scenario.map_state = updated
        if self._resolved_map is not None and self._resolved_map.map_id == updated.map_ref_id:
            self._resolved_map = replace(
                self._resolved_map,
                world_rect=world_rect,
                scale_factor=scale,
                transform=MapTransform(self._resolved_map.calibrated_map, world_rect),
            )

    def set_active_visibility(self, visible: bool) -> None:
        scenario = self._active_scenario_provider()
        if scenario is None or not isinstance(scenario.map_state, ScenarioMapState):
            return
        updated = replace(scenario.map_state, visible=bool(visible))
        scenario.map_state = updated
        if self._resolved_map is not None:
            self._resolved_map = replace(self._resolved_map, visible=updated.visible)
