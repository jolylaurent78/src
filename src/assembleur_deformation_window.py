"""Fenetre geographique temporaire pour le mode DEFORM.

La fenetre ne connait ni le scenario ni le moteur TopologyWorld : elle
affiche des sommets Lambert et remonte les gestes de drag a son proprietaire.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import tkinter as tk
from tkinter import ttk

from src.assembleur_geo_map_view import (
    CalibratedGeoMap,
    GeoMapMarker,
    GeoMapPolyline,
    GeoMapView,
)


@dataclass(frozen=True)
class DeformationVertex:
    role: str
    name: str
    lambert_xy: tuple[float, float]


def derive_assembly_view_rotation_deg(
    opening_lambert_xy: tuple[float, float],
    base_lambert_xy: tuple[float, float],
    opening_world_xy: tuple[float, float],
    base_world_xy: tuple[float, float],
) -> float:
    """Aligne O->B geographique courant sur O->B du preview TopologyWorld."""
    geo_dx = float(base_lambert_xy[0]) - float(opening_lambert_xy[0])
    geo_dy = float(base_lambert_xy[1]) - float(opening_lambert_xy[1])
    world_dx = float(base_world_xy[0]) - float(opening_world_xy[0])
    world_dy = float(base_world_xy[1]) - float(opening_world_xy[1])
    if math.hypot(geo_dx, geo_dy) <= 1e-9 or math.hypot(world_dx, world_dy) <= 1e-9:
        raise ValueError("Vecteur O-B nul pour la vue Assemblage")
    geo_angle = math.atan2(geo_dy, geo_dx)
    world_angle = math.atan2(world_dy, world_dx)
    return math.degrees(math.atan2(
        math.sin(world_angle - geo_angle),
        math.cos(world_angle - geo_angle),
    ))


class DeformationWindow(tk.Toplevel):
    """Editeur geographique non modal d'un unique triangle temporaire."""

    def __init__(
        self,
        parent,
        *,
        calibrated_map: CalibratedGeoMap,
        on_vertex_drag_started: Callable[[str], None],
        on_vertex_dragged: Callable[[str, tuple[float, float]], None],
        on_vertex_drag_released: Callable[[str], None],
        on_view_mode_changed: Callable[[str], None],
        on_closed: Callable[[], None],
    ):
        super().__init__(parent)
        self.title("Deformation")
        self.geometry("900x700")
        self.minsize(600, 450)
        self._on_vertex_drag_started = on_vertex_drag_started
        self._on_vertex_dragged = on_vertex_dragged
        self._on_vertex_drag_released = on_vertex_drag_released
        self._on_view_mode_changed = on_view_mode_changed
        self._on_closed = on_closed
        self._element_id: str | None = None
        self._assembly_rotation_deg = 0.0
        self._view_mode = tk.StringVar(value="north")

        toolbar = ttk.Frame(self, padding=(8, 8, 8, 4))
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Vue :").pack(side=tk.LEFT)
        ttk.Radiobutton(
            toolbar,
            text="Nord",
            value="north",
            variable=self._view_mode,
            command=self._view_mode_changed,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(
            toolbar,
            text="Assemblage",
            value="assembly",
            variable=self._view_mode,
            command=self._view_mode_changed,
        ).pack(side=tk.LEFT, padx=(6, 0))
        self._status = ttk.Label(toolbar, text="")
        self._status.pack(side=tk.RIGHT)

        self.map_view = GeoMapView(
            self,
            on_marker_drag_started=self._marker_drag_started,
            on_marker_dragged=self._marker_dragged,
            on_marker_drag_released=self._marker_drag_released,
        )
        self.map_view.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.map_view.set_map(calibrated_map)
        self.protocol("WM_DELETE_WINDOW", self._close)

    @property
    def view_mode(self) -> str:
        return self._view_mode.get()

    def set_triangle(
        self,
        *,
        element_id: str,
        vertices: Mapping[str, DeformationVertex],
        assembly_rotation_deg: float,
        status_text: str = "",
    ) -> None:
        if set(vertices) != {"O", "B", "L"}:
            raise ValueError("La fenetre DEFORM exige exactement les roles O/B/L")
        changed_element = element_id != self._element_id
        self._element_id = element_id
        self._assembly_rotation_deg = float(assembly_rotation_deg)
        self._apply_view_rotation()
        ordered_vertices = tuple(vertices[role] for role in ("O", "B", "L"))
        colors = {"O": "#000000", "B": "#1565c0", "L": "#f6d32d"}
        outline_colors = {"O": "#000000", "B": "#1565c0", "L": "#202020"}

        def geographic(vertex: DeformationVertex) -> tuple[float, float]:
            return self.map_view.map.lambert_to_geographic(*vertex.lambert_xy)

        coordinates = tuple(geographic(vertex) for vertex in ordered_vertices)
        self.map_view.set_markers(
            GeoMapMarker(
                vertex.role,
                latitude,
                longitude,
                f"{vertex.role} - {vertex.name}",
                always_show_label=True,
                fill_color=colors[vertex.role],
                outline_color=outline_colors[vertex.role],
                tooltip=f"{vertex.role} - {vertex.name}",
            )
            for vertex, (latitude, longitude) in zip(ordered_vertices, coordinates)
        )
        self.map_view.set_polylines((
            GeoMapPolyline(coordinates, color="#d00000", width=3, closed=True),
        ))
        self._status.configure(text=status_text)
        if changed_element:
            self.after_idle(
                lambda points=coordinates: self.map_view.fit_to_bounds(
                    points,
                    margin=0.18,
                )
            )

    def _apply_view_rotation(self) -> None:
        rotation = self._assembly_rotation_deg if self.view_mode == "assembly" else 0.0
        self.map_view.set_view_rotation_deg(rotation)

    def _view_mode_changed(self) -> None:
        self._apply_view_rotation()
        self._on_view_mode_changed(self.view_mode)

    def _marker_drag_started(self, marker_id: object) -> None:
        self._on_vertex_drag_started(str(marker_id))

    def _marker_dragged(self, marker_id: object, lambert_xy: tuple[float, float]) -> None:
        self._on_vertex_dragged(str(marker_id), lambert_xy)

    def _marker_drag_released(self, marker_id: object) -> None:
        self._on_vertex_drag_released(str(marker_id))

    def _close(self) -> None:
        self._on_closed()
