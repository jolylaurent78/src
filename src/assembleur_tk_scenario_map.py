"""Adaptateur Tk entre l'état de carte d'un scénario et le renderer historique."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.assembleur_catalogue import WorldRect
from src.assembleur_catalogue_map_assets import CatalogueMapAssetResolver
from src.assembleur_map_transform import MapTransform, scale_factor_for_world_rect
from src.assembleur_scenario_map import ScenarioMapPosition, ScenarioMapState
from src.assembleur_scenario_map_runtime import ScenarioMapResolver
from src.assembleur_background_map_layer import BackgroundMapWorldRect


class TriangleViewerScenarioMapMixin:
    """Fait de ``ScenarioMapState`` la source métier de la carte Tk.

    Le rendu raster est détenu par ``background_map_layer``. Aucun calcul
    Lambert→monde ne lit cette projection runtime.
    """

    def _new_default_map_state(self) -> ScenarioMapState:
        return ScenarioMapState(map_ref_id=self.catalogue.default_map_id)

    def _scenario_map_resolver(self) -> ScenarioMapResolver:
        resolver = getattr(self, "_resolved_scenario_map_resolver", None)
        if resolver is None:
            assets = CatalogueMapAssetResolver(self.paths)
            resolver = ScenarioMapResolver(self.catalogue, assets)
            self._resolved_scenario_map_resolver = resolver
            self._resolved_scenario_map_assets = assets
        return resolver

    def _capture_map_state(self) -> ScenarioMapState:
        scenarios = getattr(self, "scenarios", ())
        index = getattr(self, "active_scenario_index", -1)
        if 0 <= index < len(scenarios):
            state = getattr(scenarios[index], "map_state", None)
            if isinstance(state, ScenarioMapState):
                return state
        return self._new_default_map_state()

    def _apply_map_state(
        self,
        state: ScenarioMapState,
        persist: bool = False,
        redraw: bool = True,
    ) -> None:
        if not isinstance(state, ScenarioMapState):
            raise TypeError("_apply_map_state exige un ScenarioMapState.")
        resolved = self._scenario_map_resolver().resolve(state)
        self._resolved_scenario_map = resolved
        if resolved is None:
            self.background_map_layer.clear()
        else:
            assets = self._resolved_scenario_map_assets.resolve(resolved.catalogue_map)
            rect = resolved.world_rect
            self.background_map_layer.set_map(
                resolved.calibrated_map.image.convert("RGBA"),
                BackgroundMapWorldRect(rect.x0, rect.y0, rect.w, rect.h),
                str(Path(assets.image_path)),
            )

        self.show_map_layer.set(state.visible)
        if redraw:
            self._redraw_from(self._last_drawn)

    def _catalogue_lambert_to_world(
        self, lambert_x_m: float, lambert_y_m: float
    ) -> tuple[float, float]:
        resolved = getattr(self, "_resolved_scenario_map", None)
        if resolved is None:
            raise RuntimeError("Aucune carte calibrée active pour résoudre les balises Catalogue.")
        return resolved.transform.lambert_to_world(lambert_x_m, lambert_y_m)

    def _background_map_scale_factor(self) -> float | None:
        resolved = getattr(self, "_resolved_scenario_map", None)
        return None if resolved is None else resolved.scale_factor

    def _on_background_map_geometry_changed(self) -> None:
        self._sync_active_map_state_from_rendered_rect()

    def _sync_active_map_state_from_rendered_rect(self) -> None:
        rect = self.background_map_layer.world_rect
        if rect is None:
            return
        scenarios = getattr(self, "scenarios", ())
        index = getattr(self, "active_scenario_index", -1)
        if not (0 <= index < len(scenarios)):
            return
        scenario = scenarios[index]
        state = getattr(scenario, "map_state", None)
        if not isinstance(state, ScenarioMapState) or state.map_ref_id is None:
            return
        catalogue_map = self.catalogue.get_map(state.map_ref_id)
        default = catalogue_map.default_world_rect
        rect = WorldRect(rect.x0, rect.y0, rect.w, rect.h)
        scale = scale_factor_for_world_rect(rect, default, catalogue_map.default_scale_factor)
        same_position = abs(rect.x0 - default.x0) < 1e-9 and abs(rect.y0 - default.y0) < 1e-9
        same_size = abs(rect.w - default.w) < 1e-9 and abs(rect.h - default.h) < 1e-9
        updated = ScenarioMapState(
            map_ref_id=state.map_ref_id,
            position_override=None if same_position else ScenarioMapPosition(rect.x0, rect.y0),
            scale_factor_override=None if same_size else scale,
            visible=state.visible,
        )
        scenario.map_state = updated
        resolved = getattr(self, "_resolved_scenario_map", None)
        if resolved is not None and resolved.map_id == updated.map_ref_id:
            self._resolved_scenario_map = replace(
                resolved,
                world_rect=rect,
                scale_factor=scale,
                transform=MapTransform(resolved.calibrated_map, rect),
            )

    def _set_active_map_visibility(self, visible: bool) -> None:
        self._replace_active_map_state(visible=bool(visible))

    def _replace_active_map_state(self, **changes: object) -> None:
        scenarios = getattr(self, "scenarios", ())
        index = getattr(self, "active_scenario_index", -1)
        if not (0 <= index < len(scenarios)):
            return
        state = getattr(scenarios[index], "map_state", None)
        if not isinstance(state, ScenarioMapState):
            return
        updated = replace(state, **changes)
        scenarios[index].map_state = updated
        resolved = getattr(self, "_resolved_scenario_map", None)
        if resolved is not None:
            self._resolved_scenario_map = replace(
                resolved, visible=updated.visible
            )
