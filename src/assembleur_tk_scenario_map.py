"""Adaptateur Tk entre l'état de carte d'un scénario et le renderer historique."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.assembleur_catalogue import WorldRect
from src.assembleur_catalogue_map_assets import CatalogueMapAssetResolver
from src.assembleur_map_transform import MapTransform, scale_factor_for_world_rect
from src.assembleur_scenario_map import ScenarioMapPosition, ScenarioMapState
from src.assembleur_scenario_map_runtime import ScenarioMapResolver


class TriangleViewerScenarioMapMixin:
    """Fait de ``ScenarioMapState`` la source métier de la carte Tk.

    ``_bg`` reste une projection de rendu transitoire, nécessaire au renderer
    raster historique. Aucun calcul Lambertâ†’monde ne lit cette projection.
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
            self._bg = None
            self._bg_base_pil = None
            self._bg_photo = None
            self._bg_resizing = None
        else:
            assets = self._resolved_scenario_map_assets.resolve(resolved.catalogue_map)
            rect = resolved.world_rect
            self._bg = {
                "path": str(Path(assets.image_path)),
                "x0": rect.x0,
                "y0": rect.y0,
                "w": rect.w,
                "h": rect.h,
                "aspect": rect.w / rect.h,
            }
            self._bg_base_pil = resolved.calibrated_map.image.convert("RGBA")
            self._bg_photo = None
            self._bg_resizing = None

        self.show_map_layer.set(state.visible)
        self.map_opacity.set(round(state.opacity * 100))
        if redraw:
            self._redraw_from(self._last_drawn)

    def _catalogue_lambert_to_world(
        self, lambert_x_m: float, lambert_y_m: float
    ) -> tuple[float, float]:
        resolved = getattr(self, "_resolved_scenario_map", None)
        if resolved is None:
            raise RuntimeError("Aucune carte calibrée active pour résoudre les balises Catalogue.")
        return resolved.transform.lambert_to_world(lambert_x_m, lambert_y_m)

    def _bg_compute_scale_factor(self) -> float | None:
        resolved = getattr(self, "_resolved_scenario_map", None)
        return None if resolved is None else resolved.scale_factor

    def _bg_update_move(self, sx: int, sy: int):
        super()._bg_update_move(sx, sy)
        self._sync_active_map_state_from_rendered_rect()

    def _bg_update_resize(self, sx: int, sy: int):
        super()._bg_update_resize(sx, sy)
        self._sync_active_map_state_from_rendered_rect()

    def _sync_active_map_state_from_rendered_rect(self) -> None:
        bg = getattr(self, "_bg", None)
        if not isinstance(bg, dict):
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
        rect = WorldRect(float(bg["x0"]), float(bg["y0"]), float(bg["w"]), float(bg["h"]))
        scale = scale_factor_for_world_rect(rect, default, catalogue_map.default_scale_factor)
        same_position = abs(rect.x0 - default.x0) < 1e-9 and abs(rect.y0 - default.y0) < 1e-9
        same_size = abs(rect.w - default.w) < 1e-9 and abs(rect.h - default.h) < 1e-9
        updated = ScenarioMapState(
            map_ref_id=state.map_ref_id,
            position_override=None if same_position else ScenarioMapPosition(rect.x0, rect.y0),
            scale_factor_override=None if same_size else scale,
            visible=state.visible,
            opacity=state.opacity,
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

    def _set_active_map_opacity(self, opacity: float) -> None:
        self._replace_active_map_state(opacity=float(opacity))

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
                resolved, visible=updated.visible, opacity=updated.opacity
            )
