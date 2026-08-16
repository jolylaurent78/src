"""Fenetre geographique temporaire pour le mode DEFORM.

La fenetre ne connait ni le scenario ni le moteur TopologyWorld : elle
affiche des sommets Lambert et remonte les gestes de drag a son proprietaire.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
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
        on_vertex_selected: Callable[[str], None],
        on_occurrence_selected: Callable[[str, str], None],
        on_delete_selected: Callable[[], None],
        on_map_pin_selected: Callable[[], None],
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
        self._on_vertex_selected = on_vertex_selected
        self._on_occurrence_selected = on_occurrence_selected
        self._on_delete_selected = on_delete_selected
        self._on_map_pin_selected = on_map_pin_selected
        self._on_view_mode_changed = on_view_mode_changed
        self._on_closed = on_closed
        self._element_id: str | None = None
        self._assembly_rotation_deg = 0.0
        self._view_mode = tk.StringVar(value="north")
        self._closed = False
        self._occurrence_by_iid: dict[str, tuple[str, str]] = {}
        self._updating_occurrences = False

        self._icon_compass = tk.PhotoImage(
            file=str(Path(__file__).resolve().parent.parent / "images" / "compass.png")
        )
        self._icon_geometry = tk.PhotoImage(
            file=str(Path(__file__).resolve().parent.parent / "images" / "geometry.png")
        )
        self._icon_map_pin = tk.PhotoImage(
            file=str(Path(__file__).resolve().parent.parent / "images" / "map-pin.png")
        )
        self._icon_delete = tk.PhotoImage(
            file=str(Path(__file__).resolve().parent.parent / "images" / "scenario_delete.png")
        )

        content = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))
        self._content = content

        occurrence_panel = ttk.Frame(content, padding=(0, 0, 6, 0), width=250)
        ttk.Label(occurrence_panel, text="Points déformés").pack(anchor=tk.W, pady=(0, 4))
        list_toolbar = ttk.Frame(occurrence_panel)
        list_toolbar.pack(fill=tk.X, pady=(0, 4))
        self._map_pin_button = tk.Button(
            list_toolbar, image=self._icon_map_pin, command=self._on_map_pin_selected,
            state=tk.DISABLED, relief=tk.FLAT, bd=1,
        )
        self._map_pin_button.pack(side=tk.LEFT)
        self._delete_button = tk.Button(
            list_toolbar, image=self._icon_delete, command=self._on_delete_selected,
            state=tk.DISABLED, relief=tk.FLAT, bd=1,
        )
        self._delete_button.pack(side=tk.LEFT, padx=(4, 0))
        self._map_pin_button.bind("<Enter>", lambda _event: self._status.configure(text="Déplacer vers une ville..."))
        self._delete_button.bind("<Enter>", lambda _event: self._status.configure(text="Supprimer la déformation"))

        occurrence_scrollbar = ttk.Scrollbar(occurrence_panel, orient=tk.VERTICAL)
        self._occurrence_tree = ttk.Treeview(occurrence_panel, show="tree", selectmode="browse", yscrollcommand=occurrence_scrollbar.set, height=12)
        occurrence_scrollbar.configure(command=self._occurrence_tree.yview)
        occurrence_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._occurrence_tree.pack(fill=tk.BOTH, expand=True)
        self._occurrence_tree.bind("<<TreeviewSelect>>", self._occurrence_tree_selected)

        map_panel = ttk.Frame(content)
        content.add(occurrence_panel, weight=0)
        content.add(map_panel, weight=1)
        self._initial_sash_pending = True
        content.bind("<Configure>", self._place_initial_sash)
        view_toolbar = ttk.Frame(map_panel)
        view_toolbar.pack(fill=tk.X, pady=(0, 4))
        self._north_button = tk.Button(
            view_toolbar,
            image=self._icon_compass,
            command=lambda: self._set_view_mode("north"),
            relief=tk.SUNKEN,
            bd=1,
        )
        self._north_button.pack(side=tk.LEFT)
        self._assembly_button = tk.Button(
            view_toolbar,
            image=self._icon_geometry,
            command=lambda: self._set_view_mode("assembly"),
            relief=tk.FLAT,
            bd=1,
        )
        self._assembly_button.pack(side=tk.LEFT, padx=(4, 0))
        self._status = ttk.Label(view_toolbar, text="")
        self._status.pack(side=tk.RIGHT)

        self.map_view = GeoMapView(
            map_panel,
            on_marker_drag_started=self._marker_drag_started,
            on_marker_dragged=self._marker_dragged,
            on_marker_drag_released=self._marker_drag_released,
            on_marker_selected=self._marker_selected,
        )
        self.map_view.pack(fill=tk.BOTH, expand=True)
        self.map_view.set_map(calibrated_map)

        footer = ttk.Frame(self, padding=(8, 0, 8, 8))
        footer.pack(fill=tk.X)
        ttk.Button(footer, text="Fermer", command=self._close).pack(side=tk.RIGHT)
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
        selected_role: str | None = None,
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
        self.map_view.set_selected_marker(selected_role, recenter=False)
        self._status.configure(text=status_text)
        if changed_element:
            self.after_idle(
                lambda points=coordinates: self.map_view.fit_to_bounds(
                    points,
                    margin=0.18,
                )
            )

    def set_occurrences(
        self,
        occurrences: tuple[tuple[str, str, str], ...],
        selected_occurrence: tuple[str, str] | None,
    ) -> None:
        self._updating_occurrences = True
        try:
            self._occurrence_tree.delete(*self._occurrence_tree.get_children())
            self._occurrence_by_iid.clear()
            selected_iid = None
            for index, (element_id, role, label) in enumerate(occurrences):
                iid = f"occurrence-{index}"
                self._occurrence_tree.insert("", tk.END, iid=iid, text=label)
                occurrence = (element_id, role)
                self._occurrence_by_iid[iid] = occurrence
                if occurrence == selected_occurrence:
                    selected_iid = iid
            if selected_iid is not None:
                self._occurrence_tree.selection_set(selected_iid)
                self._occurrence_tree.focus(selected_iid)
                self._occurrence_tree.see(selected_iid)
            state = tk.NORMAL if selected_occurrence is not None else tk.DISABLED
            self._map_pin_button.configure(state=state)
            self._delete_button.configure(state=state)
        finally:
            self._updating_occurrences = False

    def _place_initial_sash(self, _event=None) -> None:
        if not self._initial_sash_pending or self._content.winfo_width() <= 250:
            return
        self._initial_sash_pending = False
        self._content.sashpos(0, 250)

    def _apply_view_rotation(self) -> None:
        rotation = self._assembly_rotation_deg if self.view_mode == "assembly" else 0.0
        self.map_view.set_view_rotation_deg(rotation)

    def _set_view_mode(self, mode: str) -> None:
        if mode not in {"north", "assembly"}:
            raise ValueError(f"Mode de vue DEFORM invalide: {mode!r}")
        if mode == self.view_mode:
            return
        self._view_mode.set(mode)
        self._update_view_mode_buttons()
        self._apply_view_rotation()
        self._on_view_mode_changed(self.view_mode)

    def _update_view_mode_buttons(self) -> None:
        north_active = self.view_mode == "north"
        self._north_button.configure(relief=tk.SUNKEN if north_active else tk.FLAT)
        self._assembly_button.configure(relief=tk.FLAT if north_active else tk.SUNKEN)

    def _marker_drag_started(self, marker_id: object) -> None:
        self._on_vertex_drag_started(str(marker_id))

    def _marker_dragged(self, marker_id: object, lambert_xy: tuple[float, float]) -> None:
        self._on_vertex_dragged(str(marker_id), lambert_xy)

    def _marker_drag_released(self, marker_id: object) -> None:
        self._on_vertex_drag_released(str(marker_id))

    def _marker_selected(self, marker_id: object | None) -> None:
        if marker_id is not None:
            self._on_vertex_selected(str(marker_id))

    def _occurrence_tree_selected(self, _event=None) -> None:
        if self._updating_occurrences:
            return
        selection = self._occurrence_tree.selection()
        if not selection:
            return
        occurrence = self._occurrence_by_iid.get(selection[0])
        if occurrence is not None:
            self._on_occurrence_selected(*occurrence)

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_closed()
