"""Interactive canvas rendering and runtime placement for the compass."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from src.assembleur_compass_state import (
    CompassState,
    azimuth_world_deg,
    clock_angle_diff_deg,
    clock_arc_compute_angle_deg,
    clock_arc_compute_tk_arc,
    clock_theoretical_ref_azimuth_deg,
)


EPS_WORLD = 1e-6


class CompassController:
    """Owns compass rendering helpers while sharing the explicit runtime state."""

    def __init__(
        self,
        state: CompassState,
        world_to_screen: Callable[[object], tuple[float, float]],
        screen_to_world: Callable[[float, float], tuple[float, float]],
        active_scenario_provider: Callable[[], object | None],
        projection_entries_provider: Callable[[], list],
        decryptor_provider: Callable[[], object],
        visible_provider: Callable[[], bool],
        arc_filter_active_provider: Callable[[], bool],
        ctrl_down_provider: Callable[[], bool],
        status_callback: Callable[[str], None],
        projection_index_provider: Callable[[str], int | None],
        clear_arc_filter_callback: Callable[[], None],
        auto_arc_filter_callback: Callable[[], None],
        guide_color_provider: Callable[[], str],
        redraw_scene_callback: Callable[[], None],
        compass_menu_refresh_callback: Callable[[], None],
        on_ref_azimuth_changed: Callable[[float], None],
        auto_ref_enabled_provider: Callable[[], bool],
        selected_beacon_id_provider: Callable[[], str | None],
        beacon_world_provider: Callable[[str], tuple[float, float]],
        active_beacons_provider: Callable[[], list[dict]],
        active_preview_refresh_callback: Callable[[], None],
    ) -> None:
        self.state = state
        self._world_to_screen = world_to_screen
        self._screen_to_world = screen_to_world
        self._active_scenario_provider = active_scenario_provider
        self._projection_entries_provider = projection_entries_provider
        self._decryptor_provider = decryptor_provider
        self._visible_provider = visible_provider
        self._arc_filter_active_provider = arc_filter_active_provider
        self._ctrl_down_provider = ctrl_down_provider
        self._status_callback = status_callback
        self._projection_index_provider = projection_index_provider
        self._clear_arc_filter_callback = clear_arc_filter_callback
        self._auto_arc_filter_callback = auto_arc_filter_callback
        self._guide_color_provider = guide_color_provider
        self._redraw_scene_callback = redraw_scene_callback
        self._compass_menu_refresh_callback = compass_menu_refresh_callback
        self._on_ref_azimuth_changed = on_ref_azimuth_changed
        self._auto_ref_enabled_provider = auto_ref_enabled_provider
        self._selected_beacon_id_provider = selected_beacon_id_provider
        self._beacon_world_provider = beacon_world_provider
        self._active_beacons_provider = active_beacons_provider
        self._active_preview_refresh_callback = active_preview_refresh_callback
        self._canvas = None

    def attach_canvas(self, canvas) -> None:
        self._canvas = canvas

    def change_radius(self, delta: int) -> int:
        self.state.radius = max(50, int(self.state.radius) + int(delta))
        return self.state.radius

    def contains_point(self, sx: float, sy: float, *, pad: float = 0) -> bool:
        if self.state.cx is None or self.state.cy is None:
            return False
        return (
            (float(sx) - self.state.cx) ** 2 + (float(sy) - self.state.cy) ** 2
            <= (self.state.rendered_radius + float(pad)) ** 2
        )

    def point_on_circle(self, az_deg: float, radius: float) -> tuple[float, float]:
        return (
            float(self.state.cx) + float(radius) * math.sin(math.radians(float(az_deg) % 360.0)),
            float(self.state.cy) - float(radius) * math.cos(math.radians(float(az_deg) % 360.0)),
        )

    def compute_azimuth_deg(self, sx: float, sy: float) -> float:
        return math.degrees(math.atan2(float(sx) - float(self.state.cx), float(self.state.cy) - float(sy))) % 360.0

    def delta_display_deg(self, delta_az_deg: float) -> float:
        delta = float(delta_az_deg) % 360.0
        return delta if delta <= 180.0 else 360.0 - delta

    def delta_display_text(self, delta_az_deg: float) -> str:
        return f"{self.delta_display_deg(delta_az_deg):0.1f}{chr(176)}".replace(".", ",")

    def clip_ray_to_viewport(self, sx0: float, sy0: float, az_deg: float) -> tuple[float, float] | None:
        if self._canvas is None:
            return None
        width, height = int(self._canvas.winfo_width() or 0), int(self._canvas.winfo_height() or 0)
        if width <= 0 or height <= 0:
            return None
        dx, dy = math.sin(math.radians(float(az_deg) % 360.0)), -math.cos(math.radians(float(az_deg) % 360.0))
        hits = []
        if abs(dx) > 1e-12:
            for x in (0.0, width):
                t, y = (x - sx0) / dx, sy0 + (x - sx0) / dx * dy
                if t >= 0.0 and 0.0 <= y <= height:
                    hits.append((t, x, y))
        if abs(dy) > 1e-12:
            for y in (0.0, height):
                t, x = (y - sy0) / dy, sx0 + (y - sy0) / dy * dx
                if t >= 0.0 and 0.0 <= x <= width:
                    hits.append((t, x, y))
        if not hits:
            return None
        _, x, y = max(hits, key=lambda item: item[0])
        return float(x), float(y)

    def clear_snap_target(self) -> None:
        self.state.snap_target = None
        if self._canvas is not None:
            self._canvas.delete("clock_snap_target")

    def clear_anchor_binding(self) -> None:
        self.state.anchor_binding = None

    def bind_anchor_to_node(self, *, node_id: str, topo_group_id: str, idx: int | None = None, vkey: str | None = None, world_pos=None) -> None:
        binding = {"nodeId": node_id, "topoGroupId": topo_group_id, "idx": None if idx is None else int(idx), "vkey": None if vkey is None else str(vkey)}
        scenario = self._active_scenario_provider()
        if scenario is not None and (binding["idx"] is None or binding["vkey"] is None):
            reference = scenario.topoWorld.getElementVertexFromAnyNodeId(node_id, groupId=topo_group_id)
            if isinstance(reference, dict):
                if binding["idx"] is None:
                    binding["idx"] = next((index for index, entry in enumerate(self._projection_entries_provider()) if entry.get("topoElementId") == reference.get("elementId")), None)
                if binding["vkey"] is None and reference.get("vkey") is not None:
                    binding["vkey"] = str(reference["vkey"])
                if world_pos is None and reference.get("wbest") is not None:
                    world_pos = reference["wbest"]
        self.state.anchor_binding = binding
        if world_pos is not None:
            self.state.anchor_world = np.array(world_pos, dtype=float)

    def bind_anchor_from_snap_target(self, snap_target: dict | None) -> None:
        if not isinstance(snap_target, dict) or not snap_target.get("nodeId") or not snap_target.get("topoGroupId"):
            self.clear_anchor_binding()
            return
        self.bind_anchor_to_node(
            node_id=snap_target["nodeId"], topo_group_id=snap_target["topoGroupId"],
            idx=snap_target.get("idx"), vkey=snap_target.get("vkey"), world_pos=snap_target.get("world"),
        )

    def refresh_anchor_world_from_binding(self) -> None:
        binding = self.state.anchor_binding
        if not isinstance(binding, dict):
            return
        idx, vkey = binding.get("idx"), binding.get("vkey")
        entries = self._projection_entries_provider()
        if isinstance(idx, int) and vkey in ("O", "B", "L") and 0 <= idx < len(entries):
            points = entries[idx].get("pts") or {}
            if vkey in points:
                self.state.anchor_world = np.array(points[vkey], dtype=float)
                return
        scenario = self._active_scenario_provider()
        if scenario is None or not binding.get("nodeId") or not binding.get("topoGroupId"):
            return
        self.state.anchor_world = np.array(scenario.topoWorld.getConceptNodeWorldXY(binding["nodeId"], binding["topoGroupId"]), dtype=float)

    def get_center_world(self):
        if self.state.dragging and self.state.cx is not None and self.state.cy is not None:
            return np.array(self._screen_to_world(self.state.cx, self.state.cy), dtype=float)
        self.refresh_anchor_world_from_binding()
        if self.state.anchor_world is not None:
            return np.array(self.state.anchor_world, dtype=float)
        if self.state.cx is None or self.state.cy is None:
            return None
        return np.array(self._screen_to_world(self.state.cx, self.state.cy), dtype=float)

    def get_anchor_node_hit(self) -> dict | None:
        scenario = self._active_scenario_provider()
        if scenario is None:
            return None
        binding = self.state.anchor_binding
        if isinstance(binding, dict) and binding.get("nodeId") and binding.get("topoGroupId"):
            return {"nodeId": str(binding["nodeId"]), "groupId": str(binding["topoGroupId"])}
        center_world = self.get_center_world()
        if center_world is None:
            return None
        hit = scenario.topoWorld.findNearestBoundaryNode(None, center_world)
        if hit is None:
            return None
        node_world = np.array(
            scenario.topoWorld.getConceptNodeWorldXY(str(hit["nodeId"]), str(hit["groupId"])),
            dtype=float,
        )
        return hit if float(np.linalg.norm(node_world - center_world)) <= EPS_WORLD else None

    def clamp_preview_text_xy(self, sx: int, sy: int) -> tuple[int, int]:
        if self._canvas is None:
            return int(sx) + 14, int(sy) + 10
        width, height = int(self._canvas.winfo_width() or 0), int(self._canvas.winfo_height() or 0)
        tx, ty = int(sx) + 14, int(sy) + 10
        pad, estimated_width, estimated_height = 6, 60, 20
        if width > 0:
            if tx + estimated_width > width - pad:
                tx = int(sx) - estimated_width - 14
            tx = max(pad, min(tx, width - estimated_width - pad))
        if height > 0:
            if ty + estimated_height > height - pad:
                ty = int(sy) - estimated_height - 10
            ty = max(pad, min(ty, height - estimated_height - pad))
        return int(tx), int(ty)

    def update_azimuth_preview(
        self,
        sx: int,
        sy: int,
        *,
        line_id: int | None,
        text_id: int | None,
        preview_tag: str,
        relative_to_ref: bool,
        enable_snap: bool,
        draw_line: bool = True,
        label_text: str | None = None,
        line_fill: str = "#202020",
        line_dash: tuple[int, int] | None = (4, 3),
        text_fill: str = "#202020",
    ) -> tuple[int | None, int | None, tuple[int, int, float, float] | None]:
        if self._canvas is None or self.state.cx is None or self.state.cy is None:
            return line_id, text_id, None
        sx2, sy2 = int(sx), int(sy)
        if enable_snap:
            if self._ctrl_down_provider():
                self.clear_snap_target()
            else:
                sx2, sy2 = self.apply_optional_snap(sx2, sy2, enable_snap=True)
        canvas = self._canvas
        if draw_line:
            if line_id is None:
                options = {"width": 2, "fill": line_fill, "tags": ("clock_overlay", preview_tag)}
                if line_dash is not None:
                    options["dash"] = line_dash
                line_id = canvas.create_line(float(self.state.cx), float(self.state.cy), sx2, sy2, **options)
            else:
                canvas.coords(line_id, float(self.state.cx), float(self.state.cy), sx2, sy2)
                canvas.itemconfig(line_id, fill=line_fill, width=2, dash=(() if line_dash is None else line_dash))
        elif line_id is not None:
            canvas.delete(line_id)
            line_id = None
        az_abs = self.compute_azimuth_deg(sx2, sy2)
        az_value = (az_abs - float(self.state.ref_azimuth_deg)) % 360.0 if relative_to_ref else az_abs
        label = str(label_text) if label_text is not None else f"{az_value:0.0f}{chr(176)}"
        tx, ty = self.clamp_preview_text_xy(sx2, sy2)
        if text_id is None:
            text_id = canvas.create_text(tx, ty, text=label, anchor="nw", fill=text_fill, font=("Arial", 12, "bold"), tags=("clock_overlay", preview_tag))
        else:
            canvas.itemconfig(text_id, text=label, fill=text_fill)
            canvas.coords(text_id, tx, ty)
        return line_id, text_id, (sx2, sy2, float(az_abs), float(az_value))

    def start_measure(self, sx: int, sy: int) -> None:
        self.cancel_trace(silent=True)
        self.cancel_measure(silent=True)
        self.cancel_set_ref(silent=True)
        if self._canvas is None:
            return
        if not self._visible_provider():
            self._status_callback("Compas masqu\u00e9 : affiche-le pour mesurer un azimut.")
            return
        self.state.measure.active, self.state.measure.last = True, None
        self._canvas.focus_set()
        self.update_measure(sx, sy)
        self._status_callback("Mesurer un azimut : clic gauche pour valider, ESC pour annuler. (Snap noeuds, CTRL = d\u00e9sactiver snap)")

    def update_measure(self, sx: int, sy: int) -> None:
        if not self.state.measure.active:
            return
        measure = self.state.measure
        measure.line_id, measure.text_id, result = self.update_azimuth_preview(
            sx, sy, line_id=measure.line_id, text_id=measure.text_id,
            preview_tag="clock_measure_preview", relative_to_ref=True, enable_snap=True,
        )
        if result is not None:
            measure.last = result

    def confirm_measure(self) -> None:
        if not self.state.measure.active:
            return
        last = self.state.measure.last
        if last is None:
            self.cancel_measure(silent=True)
            return
        _, _, az_abs, az_rel = last
        self.cancel_measure(silent=True)
        self._status_callback(
            f"Azimut mesur\u00e9 : {az_rel:0.0f}{chr(176)} (ref={float(self.state.ref_azimuth_deg) % 360.0:0.0f}{chr(176)}, abs={az_abs:0.0f}{chr(176)})"
        )

    def cancel_measure(self, silent: bool = False) -> None:
        measure = self.state.measure
        if not measure.active:
            return
        measure.active = False
        if self._canvas is not None:
            if measure.line_id is not None:
                self._canvas.delete(measure.line_id)
            if measure.text_id is not None:
                self._canvas.delete(measure.text_id)
        measure.line_id, measure.text_id, measure.last = None, None, None
        self.clear_snap_target()
        if not silent:
            self._status_callback("Mesure d'azimut annul\u00e9e.")

    def start_trace(self, sx: int, sy: int) -> None:
        self.cancel_set_ref(silent=True)
        self.cancel_measure(silent=True)
        if self._canvas is None:
            return
        if not self._visible_provider():
            self._status_callback("Compas masqu\u00e9 : affiche-le pour tracer un azimut.")
            return
        if self.get_anchor_node_hit() is None:
            self._status_callback("Accroche le compas \u00e0 un n\u0153ud pour tracer un azimut.")
            return
        trace = self.state.trace
        trace.active = True
        trace.preview_az_abs = trace.preview_node_id = trace.preview_topo_group_id = trace.preview_delta_az = None
        trace.line_id = trace.text_id = None
        self._canvas.delete("clock_trace_preview")
        self._canvas.focus_set()
        self._status_callback("Tracer un azimut : d\u00e9placer la souris, clic gauche pour valider, ESC pour annuler.")
        self.update_trace(sx, sy)

    def update_trace(self, sx: int, sy: int) -> None:
        trace = self.state.trace
        if not trace.active:
            return
        if not self._visible_provider():
            self.cancel_trace(silent=True)
            self._status_callback("Compas masqu\u00e9 : affiche-le pour tracer un azimut.")
            return
        scenario = self._active_scenario_provider()
        if scenario is None:
            self.cancel_trace(silent=True)
            return
        hit = self.get_anchor_node_hit()
        if hit is None:
            self._status_callback("Accroche le compas \u00e0 un n\u0153ud pour tracer un azimut.")
            self.cancel_trace(silent=True)
            return
        node_id, group_id = hit["nodeId"], hit["groupId"]
        node_world = np.array(scenario.topoWorld.getConceptNodeWorldXY(node_id, group_id), dtype=float)
        az_abs = self.compute_azimuth_deg(sx, sy)
        delta = (az_abs - float(self.state.ref_azimuth_deg) + 360.0) % 360.0
        trace.preview_az_abs, trace.preview_node_id = float(az_abs), node_id
        trace.preview_topo_group_id, trace.preview_delta_az = group_id, float(delta)
        self._canvas.delete("clock_trace_preview")
        trace.line_id = trace.text_id = None
        sx0, sy0 = self._world_to_screen(node_world)
        clipped = self.clip_ray_to_viewport(float(sx0), float(sy0), az_abs)
        if clipped is not None:
            trace.line_id = self._canvas.create_line(float(sx0), float(sy0), *clipped, dash=(4, 3), fill="#202020", width=2, tags=("clock_trace_preview",))
        _, trace.text_id, _ = self.update_azimuth_preview(
            sx, sy, line_id=None, text_id=None, preview_tag="clock_trace_preview",
            relative_to_ref=False, enable_snap=False, draw_line=False,
            label_text=self.delta_display_text(delta), text_fill="#202020",
        )

    def confirm_trace(self) -> None:
        trace = self.state.trace
        if not trace.active:
            return
        if trace.preview_node_id is None or trace.preview_topo_group_id is None or trace.preview_delta_az is None:
            self.cancel_trace(silent=True)
            return
        scenario = self._active_scenario_provider()
        if scenario is None:
            self.cancel_trace(silent=True)
            return
        delta = float(trace.preview_delta_az) % 360.0
        scenario.clockAzimuthTraits.append({
            "nodeId": trace.preview_node_id, "topoGroupId": trace.preview_topo_group_id,
            "deltaAzDeg": delta, "colorHex": str(self._guide_color_provider()),
        })
        self.cancel_trace(silent=True)
        self._compass_menu_refresh_callback()
        self._redraw_scene_callback()
        self._status_callback(f"Trait azimut ajout\u00e9 ({chr(916)}={delta:0.0f}{chr(176)}).")

    def cancel_trace(self, silent: bool = False) -> None:
        trace = self.state.trace
        trace.active = False
        if self._canvas is not None:
            self._canvas.delete("clock_trace_preview")
        trace.line_id = trace.text_id = None
        trace.preview_az_abs = trace.preview_node_id = trace.preview_topo_group_id = trace.preview_delta_az = None
        if not silent:
            self._status_callback("Tra\u00e7age d'azimut annul\u00e9.")

    def collect_set_ref_candidates(self) -> list[dict]:
        scenario = self._active_scenario_provider()
        center_world = self.get_center_world()
        if scenario is None or center_world is None:
            return []
        world = scenario.topoWorld
        candidates: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        anchor_hit = self.get_anchor_node_hit()
        if anchor_hit is not None:
            anchor_node_id, anchor_group_id = str(anchor_hit["nodeId"]), str(anchor_hit["groupId"])
            for neighbor_id in world.getConceptNeighborNodes(anchor_node_id, anchor_group_id):
                neighbor_world = np.array(world.getConceptNodeWorldXY(str(neighbor_id), anchor_group_id), dtype=float)
                if float(np.linalg.norm(neighbor_world - center_world)) <= EPS_WORLD:
                    continue
                key = ("node", anchor_group_id, str(neighbor_id))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "type": "node", "id": str(neighbor_id), "nodeId": str(neighbor_id),
                    "topoGroupId": anchor_group_id, "world": neighbor_world,
                    "azAbsDeg": float(azimuth_world_deg(center_world, neighbor_world)),
                    "label": str(world.getNodeLabel(str(neighbor_id))),
                })
        for beacon in self._active_beacons_provider():
            beacon_id = str(beacon["beaconId"])
            beacon_world = np.array(self._beacon_world_provider(beacon_id), dtype=float)
            if float(np.linalg.norm(beacon_world - center_world)) <= EPS_WORLD:
                continue
            key = ("balise", "", beacon_id)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "type": "balise", "id": beacon_id, "beaconId": beacon_id, "world": beacon_world,
                "azAbsDeg": float(azimuth_world_deg(center_world, beacon_world)),
                "label": str(beacon["label"]),
            })
        return candidates

    def pick_set_ref_snap_candidate(self, mouse_az_abs: float, candidates: list[dict]) -> dict | None:
        best, best_diff = None, None
        for candidate in candidates:
            if "azAbsDeg" not in candidate:
                continue
            diff = clock_angle_diff_deg(float(candidate["azAbsDeg"]), float(mouse_az_abs))
            if best_diff is None or diff < best_diff:
                best, best_diff = candidate, diff
        return best

    def clear_set_ref_snap_target(self) -> None:
        if self._canvas is not None:
            self._canvas.delete("clock_setref_snap_target")

    def draw_set_ref_snap_target(self, snap_target: dict | None) -> None:
        self.clear_set_ref_snap_target()
        if self._canvas is None or not isinstance(snap_target, dict) or snap_target.get("world") is None:
            return
        px, py = self._world_to_screen(snap_target["world"])
        radius = 10 if snap_target.get("type") == "node" else 8
        self._canvas.create_oval(px-radius, py-radius, px+radius, py+radius, outline="#FF0000", width=3, fill="", tags=("clock_overlay", "clock_setref_snap_target"))
        self._canvas.tag_raise("clock_setref_snap_target")

    def build_set_ref_preview(self, sx: int, sy: int) -> dict | None:
        if self.state.cx is None or self.state.cy is None:
            return None
        mouse_az_abs = self.compute_azimuth_deg(sx, sy)
        preview_az_abs, preview_sx, preview_sy, snap_target = mouse_az_abs, int(sx), int(sy), None
        if not self._ctrl_down_provider():
            snap_target = self.pick_set_ref_snap_candidate(mouse_az_abs, self.collect_set_ref_candidates())
            if isinstance(snap_target, dict) and snap_target.get("world") is not None:
                preview_az_abs = float(snap_target["azAbsDeg"]) % 360.0
                preview_sx, preview_sy = map(int, self._world_to_screen(snap_target["world"]))
        return {
            "mouseAzAbs": float(mouse_az_abs) % 360.0,
            "previewAzAbs": float(preview_az_abs) % 360.0,
            "previewSx": preview_sx, "previewSy": preview_sy, "snapTarget": snap_target,
        }

    def start_set_ref(self, sx: int, sy: int) -> None:
        self.cancel_trace(silent=True)
        self.cancel_set_ref(silent=True)
        if self._canvas is None:
            return
        if not self._visible_provider():
            self._status_callback("Compas masqu\u00e9 : affiche-le pour d\u00e9finir l'azimut de r\u00e9f\u00e9rence.")
            return
        self.state.set_ref.active, self.state.set_ref.last = True, None
        self.clear_set_ref_snap_target()
        self._canvas.focus_set()
        self.update_set_ref(sx, sy)
        self._status_callback("D\u00e9finir azimut de r\u00e9f\u00e9rence : d\u00e9placer la souris, clic gauche pour valider, CTRL = azimut libre, ESC pour annuler.")

    def update_set_ref(self, sx: int, sy: int) -> None:
        set_ref = self.state.set_ref
        if not set_ref.active:
            return
        preview = self.build_set_ref_preview(sx, sy)
        if preview is None:
            return
        self.draw_set_ref_snap_target(preview["snapTarget"])
        set_ref.line_id, set_ref.text_id, result = self.update_azimuth_preview(
            int(preview["previewSx"]), int(preview["previewSy"]),
            line_id=set_ref.line_id, text_id=set_ref.text_id, preview_tag="clock_ref_preview",
            relative_to_ref=False, enable_snap=False,
            label_text=f"{float(preview['previewAzAbs']):0.0f}{chr(176)}",
        )
        if result is not None:
            set_ref.last = dict(preview)

    def confirm_set_ref(self, sx: int, sy: int) -> None:
        set_ref = self.state.set_ref
        if not set_ref.active:
            return
        preview = self.build_set_ref_preview(sx, sy)
        if preview is None:
            self.cancel_set_ref(silent=True)
            return
        azimuth = float(preview["mouseAzAbs"] if self._ctrl_down_provider() else preview["previewAzAbs"]) % 360.0
        self.state.ref_azimuth_deg = azimuth
        self._on_ref_azimuth_changed(azimuth)
        scenario = self._active_scenario_provider()
        if scenario is not None:
            scenario.clockRefEdgeId = scenario.clockRefNodeId = scenario.clockRefTopoGroupId = None
        self.cancel_set_ref(silent=True)
        self._status_callback(f"Azimut de r\u00e9f\u00e9rence d\u00e9fini : {azimuth:0.0f}{chr(176)}")
        self._compass_menu_refresh_callback()
        self.redraw()

    def cancel_set_ref(self, silent: bool = False) -> None:
        set_ref = self.state.set_ref
        if not set_ref.active:
            return
        set_ref.active = False
        if self._canvas is not None:
            if set_ref.line_id is not None:
                self._canvas.delete(set_ref.line_id)
            if set_ref.text_id is not None:
                self._canvas.delete(set_ref.text_id)
        set_ref.line_id = set_ref.text_id = None
        set_ref.last = None
        self.clear_set_ref_snap_target()
        if not silent:
            self._status_callback("D\u00e9finition d'azimut annul\u00e9e.")

    def compute_ref_azimuth_from_selected_beacon(self) -> float | None:
        center_world = self.get_center_world()
        beacon_id = self._selected_beacon_id_provider()
        if center_world is None or not beacon_id:
            return None
        beacon_world = self._beacon_world_provider(beacon_id)
        return float(azimuth_world_deg(center_world, beacon_world))

    def apply_auto_ref_sync(self) -> None:
        if self.state.auto_ref_sync_in_progress or not self._auto_ref_enabled_provider() or self._canvas is None or not self._visible_provider():
            return
        azimuth = self.compute_ref_azimuth_from_selected_beacon()
        if azimuth is None:
            return
        self.state.auto_ref_sync_in_progress = True
        try:
            self.state.ref_azimuth_deg = float(azimuth) % 360.0
            self.redraw()
            self._active_preview_refresh_callback()
        finally:
            self.state.auto_ref_sync_in_progress = False

    def update_snap_target(self, sx: float, sy: float) -> None:
        previous = self.state.snap_target
        scenario = self._active_scenario_provider()
        world = scenario.topoWorld
        hit = world.findNearestBoundaryNode(None, np.array(self._screen_to_world(sx, sy), dtype=float))
        if hit is None:
            self.clear_snap_target()
            if previous is not None and self.arc_is_available():
                self.clear_arc_last()
            self.apply_auto_ref_sync()
            return
        node_dsu, group_id = hit["nodeId"], hit["groupId"]
        reference = world.getElementVertexFromAnyNodeId(node_dsu, groupId=group_id)
        if reference is None:
            raise RuntimeError(f"[ClockSnap] Topo hit but cannot resolve nodeId to element/vertex: {hit}")
        index = self._projection_index_provider(reference["elementId"])
        if index is None:
            raise RuntimeError(f"[ClockSnap] Topo elementId '{reference['elementId']}' not found in GUI last_drawn")
        previous_key = (previous.get("topoGroupId"), previous.get("nodeDsu")) if isinstance(previous, dict) else None
        new_key = (group_id, node_dsu)
        if previous_key is not None and new_key != previous_key and self.arc_is_available():
            self.clear_arc_last()
        target = {"idx": int(index), "vkey": str(reference["vkey"]), "world": np.array(reference["wbest"], dtype=float), "nodeId": node_dsu, "nodeDsu": node_dsu, "topoGroupId": group_id}
        self.state.snap_target = target
        if new_key != previous_key:
            self.auto_arc_from_snap_target(target, drag=True)
        if self._canvas is not None:
            self._canvas.delete("clock_snap_target")
            px, py = self._world_to_screen(reference["wbest"])
            self._canvas.create_oval(px-10, py-10, px+10, py+10, outline="#FF0000", width=3, fill="", tags="clock_snap_target")
            self._canvas.tag_raise("clock_snap_target")
        self.apply_auto_ref_sync()

    def apply_optional_snap(self, sx: int, sy: int, *, enable_snap: bool) -> tuple[int, int]:
        if not enable_snap:
            return int(sx), int(sy)
        if self._ctrl_down_provider():
            self.clear_snap_target()
            return int(sx), int(sy)
        self.update_snap_target(sx, sy)
        target = self.state.snap_target
        if isinstance(target, dict) and target.get("world") is not None:
            return tuple(map(int, self._world_to_screen(target["world"])))
        return int(sx), int(sy)

    def arc_is_available(self) -> bool:
        return self.state.arc.last_angle_deg is not None and isinstance(self.state.arc.last, dict)

    def clear_arc_last(self) -> None:
        if self._arc_filter_active_provider():
            self._clear_arc_filter_callback()
        self.state.arc.last = None
        self.state.arc.last_angle_deg = None
        self._compass_menu_refresh_callback()

    def start_arc(self) -> None:
        self.cancel_trace(silent=True)
        self.cancel_arc(silent=True)
        self.cancel_measure(silent=True)
        self.cancel_set_ref(silent=True)
        if self._canvas is None:
            return
        if not self._visible_provider():
            self._status_callback("Compas masqu\u00e9 : affiche-le pour mesurer un arc d'angle.")
            return
        arc = self.state.arc
        arc.active, arc.step, arc.p1, arc.p2 = True, 0, None, None
        self.clear_snap_target()
        self._canvas.focus_set()
        self._status_callback("Mesurer un arc d'angle : clic gauche P1 puis P2, ESC pour annuler. (Snap noeuds, CTRL = d\u00e9sactiver snap)")

    def handle_arc_click(self, sx: int, sy: int) -> None:
        arc = self.state.arc
        if not arc.active:
            return
        sx2, sy2 = self.apply_optional_snap(sx, sy, enable_snap=True)
        azimuth = self.compute_azimuth_deg(sx2, sy2)
        if arc.step == 0:
            arc.p1, arc.step = (sx2, sy2, azimuth), 1
            self._status_callback("Mesurer un arc d'angle : sélectionne le point P2 (clic gauche), ESC pour annuler. (Snap noeuds, CTRL = désactiver snap)")
            self.update_arc_preview(sx2, sy2)
            return
        arc.p2 = (sx2, sy2, azimuth)
        angle = clock_arc_compute_angle_deg(arc.p1[2], arc.p2[2])
        arc.last, arc.last_angle_deg = {"az1": arc.p1[2], "az2": arc.p2[2], "angle": angle}, angle
        self._compass_menu_refresh_callback()
        self.cancel_arc(silent=True)
        self.redraw()
        self._status_callback(f"Arc mesur\u00e9 : {angle:0.0f}{chr(176)}")

    def update_arc_preview(self, sx: int, sy: int) -> None:
        arc = self.state.arc
        if not arc.active or arc.step != 1 or arc.p1 is None or self._canvas is None:
            return
        sx2, sy2 = self.apply_optional_snap(sx, sy, enable_snap=True)
        az1, az2 = arc.p1[2], self.compute_azimuth_deg(sx2, sy2)
        cx, cy, radius = self.state.cx, self.state.cy, self.state.rendered_radius
        x1, y1 = arc.p1[0], arc.p1[1]
        if arc.line1_id is None:
            arc.line1_id = self._canvas.create_line(cx, cy, x1, y1, width=2, dash=(4,3), fill="#202020", tags=("clock_overlay", "clock_arc_preview"))
        else: self._canvas.coords(arc.line1_id, cx, cy, x1, y1)
        if arc.line2_id is None:
            arc.line2_id = self._canvas.create_line(cx, cy, sx2, sy2, width=2, dash=(4,3), fill="#202020", tags=("clock_overlay", "clock_arc_preview"))
        else: self._canvas.coords(arc.line2_id, cx, cy, sx2, sy2)
        start, extent, angle, middle = clock_arc_compute_tk_arc(az1, az2)
        bbox = (cx-radius, cy-radius, cx+radius, cy+radius)
        if arc.arc_id is None:
            arc.arc_id = self._canvas.create_arc(*bbox, start=start, extent=extent, style="arc", width=2, outline="#202020", tags=("clock_overlay", "clock_arc_preview"))
        else:
            self._canvas.coords(arc.arc_id, *bbox); self._canvas.itemconfig(arc.arc_id, start=start, extent=extent)
        tx, ty = self.point_on_circle(middle, radius * 1.08)
        if arc.text_id is None:
            arc.text_id = self._canvas.create_text(tx, ty, text=f"{angle:0.0f}{chr(176)}", anchor="center", fill="#202020", font=("Arial",12,"bold"), tags=("clock_overlay", "clock_arc_preview"))
        else:
            self._canvas.itemconfig(arc.text_id, text=f"{angle:0.0f}{chr(176)}"); self._canvas.coords(arc.text_id, tx, ty)

    def cancel_arc(self, silent: bool = False) -> None:
        arc = self.state.arc
        if not arc.active:
            return
        arc.active, arc.step, arc.p1, arc.p2 = False, 0, None, None
        for attribute in ("line1_id", "line2_id", "arc_id", "text_id"):
            item_id = getattr(arc, attribute)
            if item_id is not None and self._canvas is not None: self._canvas.delete(item_id)
            setattr(arc, attribute, None)
        self.clear_snap_target()
        if not silent: self._status_callback("Mesure d'arc annul\u00e9e.")

    def auto_arc_from_snap_target(self, target: dict, drag: bool, prev_node_dsu: str | None = None, next_node_dsu: str | None = None) -> bool:
        if not self._visible_provider() or not isinstance(target, dict) or not target.get("topoGroupId") or not target.get("nodeDsu"):
            return False
        scenario = self._active_scenario_provider(); world = scenario.topoWorld
        node_id, group_id, node = target["nodeId"], world.getGroupIdFromConceptNode(target["nodeId"]), target["nodeDsu"]
        if (prev_node_dsu is None) ^ (next_node_dsu is None): raise RuntimeError("[ClockArc] prevNodeDsu and nextNodeDsu must be provided together")
        if prev_node_dsu is None:
            previous, following = world.getBoundaryNeighbors(group_id, node)
            if previous is None or following is None: raise RuntimeError(f"[ClockArc] Boundary neighbors not found for node '{node}' in group '{group_id}'")
        else: previous, following = prev_node_dsu, next_node_dsu
        center = np.array(world.getConceptNodeWorldXY(node, group_id), dtype=float)
        first, second = np.array(world.getConceptNodeWorldXY(previous, group_id), dtype=float), np.array(world.getConceptNodeWorldXY(following, group_id), dtype=float)
        self.state.cx, self.state.cy = map(float, self._world_to_screen(center))
        az1, az2 = azimuth_world_deg(center, first), azimuth_world_deg(center, second); angle = clock_arc_compute_angle_deg(az1, az2)
        self.state.arc.last, self.state.arc.last_angle_deg = {"az1": az1, "az2": az2, "angle": angle}, angle
        self.redraw()
        if not drag: self._auto_arc_filter_callback()
        return True

    def redraw(self) -> None:
        if self._canvas is None:
            return
        self._canvas.delete("clock_overlay")
        self.draw_overlay()

    def draw_arc_last(self, cx: float, cy: float, radius: float) -> None:
        if self._canvas is None or not isinstance(self.state.arc.last, dict):
            return
        last = self.state.arc.last
        az1, az2 = float(last["az1"]), float(last["az2"])
        start, extent, angle, middle = clock_arc_compute_tk_arc(az1, az2)
        x1, y1 = self.point_on_circle(az1, radius)
        x2, y2 = self.point_on_circle(az2, radius)
        canvas = self._canvas
        canvas.create_line(cx, cy, x1, y1, width=2, dash=(4, 3), fill="#202020", tags=("clock_overlay", "clock_arc_persist"))
        canvas.create_line(cx, cy, x2, y2, width=2, dash=(4, 3), fill="#202020", tags=("clock_overlay", "clock_arc_persist"))
        canvas.create_arc(cx-radius, cy-radius, cx+radius, cy+radius, start=start, extent=extent, style="arc", width=2, outline="#202020", tags=("clock_overlay", "clock_arc_persist"))
        tx, ty = self.point_on_circle(middle, radius * 1.08)
        canvas.create_text(tx, ty, text=f"{angle:0.0f}{chr(176)}", anchor="center", fill="#202020", font=("Arial", 12, "bold"), tags=("clock_overlay", "clock_arc_persist"))

    def draw_overlay(self) -> None:
        if self._canvas is None:
            return
        canvas = self._canvas
        canvas.delete("clock_overlay")
        if not self._visible_provider():
            return
        radius = max(50, int(self.state.radius))
        self.refresh_anchor_world_from_binding()
        if self.state.anchor_world is not None and not self.state.dragging:
            self.state.cx, self.state.cy = map(float, self._world_to_screen(self.state.anchor_world))
        if self.state.anchor_world is None and self.state.cx is not None and self.state.cy is not None and not self.state.dragging:
            self.state.anchor_world = np.array(self._screen_to_world(self.state.cx, self.state.cy), dtype=float)
        if self.state.cx is None or self.state.cy is None:
            self.state.cx = self.state.cy = 12 + radius
        cx, cy = float(self.state.cx), float(self.state.cy)
        self.state.rendered_radius = radius
        ref = float(self.state.ref_azimuth_deg) % 360.0
        decryptor = self._decryptor_provider()
        hours = max(1, int(getattr(decryptor, "getHoursBase", lambda: getattr(decryptor, "hoursBase", 12))()))
        minutes = max(1, int(getattr(decryptor, "getMinutesBase", lambda: getattr(decryptor, "minutesBase", 60))()))
        show_hour = bool(getattr(decryptor, "shouldShowHourHand", lambda: True)())
        show_minute = bool(getattr(decryptor, "shouldShowMinuteHand", lambda: True)())
        show_labels = bool(getattr(decryptor, "shouldShowHourLabels", lambda: True)())
        show_ticks = bool(getattr(decryptor, "shouldShowHourTicks", lambda: True)())
        canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius, outline="#b0b0b0", width=2, tags="clock_overlay")
        ref_end = self.point_on_circle(ref, radius * 0.92)
        canvas.create_line(cx, cy, *ref_end, width=2, fill="#404040", tags=("clock_overlay",))
        for count, is_hour in ((minutes, False), (hours, True)):
            if is_hour and not show_ticks:
                continue
            for mark in range(count):
                angle = math.radians(ref + mark * 360.0 / count)
                inner = radius - (14 if is_hour and ((hours % 4 == 0 and mark % max(1, hours // 4) == 0) or (hours == 12 and mark % 3 == 0)) else (8 if is_hour else (10 if mark % 5 == 0 else 6)))
                x1, y1 = cx + inner * math.sin(angle), cy - inner * math.cos(angle)
                x2, y2 = cx + radius * math.sin(angle), cy - radius * math.cos(angle)
                canvas.create_line(x1, y1, x2, y2, width=2 if is_hour and inner == radius - 14 else 1, fill="#707070", tags="clock_overlay")
        if show_labels:
            for value, angle in ((hours, ref), (hours / 4.0, ref + 90), (hours / 2.0, ref + 180), (3 * hours / 4.0, ref + 270)):
                text = str(int(value)) if float(value).is_integer() else f"{value:.1f}".rstrip("0").rstrip(".")
                canvas.create_text(*self.point_on_circle(angle, radius - 18), text=text, font=("Arial", 11, "bold"), fill="#707070", tags="clock_overlay")
        hour = float(self.state.clock.get("hour", 5.0)) % hours
        minute = int(self.state.clock.get("minute", 9)) % minutes
        label = str(self.state.clock.get("label", ""))
        hour_base = minute_base = delta = None
        if show_hour or show_minute:
            hour_base, minute_base = decryptor.anglesFromClock(hour=float(hour), minute=int(minute))
            if show_hour and show_minute:
                delta = clock_arc_compute_angle_deg((ref + hour_base) % 360.0, (ref + minute_base) % 360.0)
            if show_hour:
                canvas.create_line(cx, cy, *self.point_on_circle(ref + hour_base, radius * .58), width=3, fill="#0b3d91", tags="clock_overlay")
            if show_minute:
                canvas.create_line(cx, cy, *self.point_on_circle(ref + minute_base, radius * .86), width=2, fill="#000000", tags="clock_overlay")
            canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill="#000000", outline="#000000", tags="clock_overlay")
        if label:
            display = f"{label} — Δ={delta:0.0f}{chr(176)}" if delta is not None else label
            last = self.state.arc.last
            if self._arc_filter_active_provider() and delta is not None and isinstance(last, dict) and "az1" in last and "az2" in last:
                theory = clock_theoretical_ref_azimuth_deg(az1=last["az1"], az2=last["az2"], ang_hour_0=hour_base, ang_min_0=minute_base)
                display = f"{display} — Ref={theory:0.1f}{chr(176)}"
            canvas.create_text(cx, cy + radius + 20, text=display, font=("Arial", 11, "bold"), fill="#000000", anchor="n", tags="clock_overlay")
        self.draw_arc_last(cx, cy, radius)
