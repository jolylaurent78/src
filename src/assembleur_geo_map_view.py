"""Composants cartographiques génériques, indépendants de tout domaine métier."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Iterable
import warnings

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from pyproj import Transformer

from src.assembleur_paths import ApplicationPaths


@dataclass(frozen=True)
class GeoMapMarker:
    """Marqueur géographique générique, identifié par le client du composant."""

    marker_id: object
    latitude: float
    longitude: float
    label: str = ""
    always_show_label: bool = False
    fill_color: str | None = None
    outline_color: str | None = None
    label_color: str | None = None
    tooltip: str | None = None


@dataclass(frozen=True)
class GeoMapPolyline:
    """Polyligne géographique générique, sans sémantique métier."""

    points: tuple[tuple[float, float], ...]
    color: str | None = None
    width: int = 2
    closed: bool = False


class CalibratedGeoMap:
    """Image de carte et transformation bidirectionnelle Lambert-93 / pixels."""

    def __init__(self, map_id: str, image: Image.Image, calibration: dict, *, image_size: tuple[int, int] | None = None):
        self.map_id = map_id
        self.image = image
        self.image_size = image_size or image.size
        self.calibration = calibration
        self.projection = str(calibration.get("projection", "lambert93"))
        self._matrix = np.asarray(calibration["A"], dtype=float)
        self._offset = np.asarray(calibration["offset"], dtype=float)
        if self._matrix.shape != (2, 2) or self._offset.shape != (2,):
            raise ValueError("Calibration cartographique invalide (A ou offset).")
        if abs(float(np.linalg.det(self._matrix))) < 1e-15:
            raise ValueError("Calibration cartographique non inversible.")
        self._inverse_matrix = np.linalg.inv(self._matrix)
        self._to_lambert = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
        self._from_lambert = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    @classmethod
    def load_map(
        cls,
        map_id: str,
        maps_dir: str | Path | None = None,
        max_image_dimension: int | None = None,
    ) -> "CalibratedGeoMap":
        """Charge une carte à sa résolution native, sauf limite explicitement demandée."""
        root = (
            Path(maps_dir)
            if maps_dir is not None
            else ApplicationPaths.from_runtime().resource_maps_dir
        )
        image_path = root / f"{map_id}.jpg"
        calibration_path = root / f"{map_id}.json"
        if not image_path.is_file():
            raise FileNotFoundError(f"Image de carte introuvable : {image_path}")
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Calibration de carte introuvable : {calibration_path}")
        if max_image_dimension is not None and int(max_image_dimension) <= 0:
            raise ValueError("max_image_dimension doit être un entier strictement positif ou None.")
        with calibration_path.open(encoding="utf-8") as calibration_file:
            calibration = json.load(calibration_file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(image_path) as source:
                image_size = source.size
                if max_image_dimension is not None and max(image_size) > int(max_image_dimension):
                    source.thumbnail(
                        (int(max_image_dimension), int(max_image_dimension)),
                        Image.Resampling.LANCZOS,
                    )
                image = source.copy()
        return cls(map_id, image, calibration, image_size=image_size)

    def lambert_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        pixel = self._matrix @ np.asarray((x_m, y_m), dtype=float) + self._offset
        return float(pixel[0]), float(pixel[1])

    def pixel_to_lambert(self, x_px: float, y_px: float) -> tuple[float, float]:
        lambert = self._inverse_matrix @ (np.asarray((x_px, y_px), dtype=float) - self._offset)
        return float(lambert[0]), float(lambert[1])

    def geographic_to_pixel(self, latitude: float, longitude: float) -> tuple[float, float]:
        x_m, y_m = self._to_lambert.transform(float(longitude), float(latitude))
        return self.lambert_to_pixel(x_m, y_m)

    def pixel_to_geographic(self, x_px: float, y_px: float) -> tuple[float, float]:
        x_m, y_m = self.pixel_to_lambert(x_px, y_px)
        return self.lambert_to_geographic(x_m, y_m)

    def lambert_to_geographic(self, x_m: float, y_m: float) -> tuple[float, float]:
        longitude, latitude = self._from_lambert.transform(x_m, y_m)
        return float(latitude), float(longitude)


class GeoMapView(tk.Frame):
    """Vue cartographique réutilisable : image, zoom, pan et marqueurs sélectionnables."""

    _MARKER_HIT_RADIUS = 11.0
    _CLICK_DRAG_THRESHOLD = 4.0
    _TOOLTIP_DELAY_MS = 350

    def __init__(
        self,
        parent,
        *,
        on_marker_selected: Callable[[object | None], None] | None = None,
        on_marker_drag_started: Callable[[object], None] | None = None,
        on_marker_dragged: Callable[[object, tuple[float, float]], None] | None = None,
        on_marker_drag_released: Callable[[object], None] | None = None,
        initial_fit_zoom: float = 1.0,
        minimum_fit_zoom: float = 1.0,
        maximum_zoom: float = 0.5,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#e9e9e9")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.on_marker_selected = on_marker_selected
        self.on_marker_drag_started = on_marker_drag_started
        self.on_marker_dragged = on_marker_dragged
        self.on_marker_drag_released = on_marker_drag_released
        self.map: CalibratedGeoMap | None = None
        self._markers: list[GeoMapMarker] = []
        self._selected_marker_id: object | None = None
        self._marker_screen_positions: dict[object, tuple[float, float]] = {}
        self._source_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._view_scale = 1.0
        self._view_rotation_deg = 0.0
        self._fit_scale = 0.001
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._initial_fit_zoom = max(0.01, float(initial_fit_zoom))
        self._minimum_fit_zoom = max(0.01, float(minimum_fit_zoom))
        self._maximum_zoom = max(0.01, float(maximum_zoom))
        self._initial_fit_applied = False
        self._press_position: tuple[int, int] | None = None
        self._pan_last_position: tuple[int, int] | None = None
        self._drag_distance = 0.0
        self._dragging_marker_id: object | None = None
        self._fit_pending = False
        self._redraw_after_id: str | None = None
        self._tooltip: tk.Toplevel | None = None
        self._tooltip_label: tk.Label | None = None
        self._tooltip_after_id: str | None = None
        self._tooltip_marker_id: object | None = None
        self._hover_marker_id: object | None = None
        self._hover_root_position: tuple[int, int] | None = None
        self._polylines: list[GeoMapPolyline] = []
        self._polygons: list[object] = []   # Extension prévue : géométries surfaciques.
        self._overlays: list[object] = []   # Extension prévue : overlays applicatifs.

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(event.x, event.y, 1.15))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(event.x, event.y, 1 / 1.15))
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _event: self._hide_tooltip())

    def load_map(self, map_id: str, maps_dir: str | Path | None = None) -> None:
        self.set_map(CalibratedGeoMap.load_map(map_id, maps_dir))

    def set_map(self, calibrated_map: CalibratedGeoMap) -> None:
        self.map = calibrated_map
        self._source_image = calibrated_map.image
        self._initial_fit_applied = False
        self.fit_to_view()

    def set_view_rotation_deg(self, angle_deg: float) -> None:
        """Tourne uniquement le repere visuel de la carte autour du viewport."""
        self._view_rotation_deg = float(angle_deg) % 360.0
        self._constrain_view_offsets()
        self._request_redraw()

    def screen_to_lambert(self, x_screen: float, y_screen: float) -> tuple[float, float]:
        if self.map is None:
            raise RuntimeError("Aucune carte geographique chargee.")
        x_map, y_map = self._screen_to_map(x_screen, y_screen)
        return self.map.pixel_to_lambert(x_map, y_map)

    def lambert_to_screen(self, x_m: float, y_m: float) -> tuple[float, float]:
        if self.map is None:
            raise RuntimeError("Aucune carte geographique chargee.")
        x_map, y_map = self.map.lambert_to_pixel(x_m, y_m)
        return self._map_to_screen(x_map, y_map)

    def set_markers(self, markers: Iterable[GeoMapMarker]) -> None:
        self._markers = list(markers)
        self._request_redraw()

    def set_polylines(self, polylines: Iterable[GeoMapPolyline]) -> None:
        """Définit des polylignes géographiques à dessiner."""
        self._polylines = list(polylines)
        self._request_redraw()

    def set_polygons(self, polygons: Iterable[object]) -> None:
        """Réserve l'API pour de futures géométries surfaciques."""
        self._polygons = list(polygons)
        self._request_redraw()

    def set_overlays(self, overlays: Iterable[object]) -> None:
        """Réserve l'API pour de futurs overlays applicatifs."""
        self._overlays = list(overlays)
        self._request_redraw()

    def set_selected_marker(self, marker_id: object | None, *, recenter: bool = False) -> None:
        self._selected_marker_id = marker_id
        if recenter and marker_id is not None:
            self.recenter_on_marker(marker_id)
        else:
            self._request_redraw()

    def recenter_on_marker(self, marker_id: object) -> None:
        if self.map is None:
            return
        marker = next((item for item in self._markers if item.marker_id == marker_id), None)
        if marker is None:
            return
        x_px, y_px = self.map.geographic_to_pixel(marker.latitude, marker.longitude)
        width, height = self._canvas_size()
        self._offset_x = width / 2 - x_px * self._view_scale
        self._offset_y = height / 2 - y_px * self._view_scale
        self._constrain_view_offsets()
        self._request_redraw()

    def fit_to_bounds(self, coordinates: Iterable[tuple[float, float]], *, margin: float = 0.12) -> None:
        """Ajuste la vue à des coordonnées (latitude, longitude) avec une marge relative."""
        if self.map is None:
            return
        points = [self.map.geographic_to_pixel(latitude, longitude) for latitude, longitude in coordinates]
        if not points:
            return
        canvas_width, canvas_height = self._canvas_size()
        margin = min(max(float(margin), 0.0), 0.45)
        available_width = max(1.0, canvas_width * (1.0 - 2.0 * margin))
        available_height = max(1.0, canvas_height * (1.0 - 2.0 * margin))
        min_x, max_x = min(point[0] for point in points), max(point[0] for point in points)
        min_y, max_y = min(point[1] for point in points), max(point[1] for point in points)
        bounds_width, bounds_height = max_x - min_x, max_y - min_y
        if bounds_width < 1.0:
            bounds_width = max(20.0, self.map.image_size[0] * 0.01)
        if bounds_height < 1.0:
            bounds_height = max(20.0, self.map.image_size[1] * 0.01)
        requested_scale = min(available_width / bounds_width, available_height / bounds_height)
        self._view_scale = min(
            max(requested_scale, self._minimum_scale()),
            max(self._maximum_zoom, self._minimum_scale()),
        )
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        self._offset_x = canvas_width / 2 - center_x * self._view_scale
        self._offset_y = canvas_height / 2 - center_y * self._view_scale
        self._constrain_view_offsets()
        self._request_redraw()

    def fit_to_view(self) -> None:
        if self.map is None:
            return
        width, height = self._canvas_size()
        if width <= 1 or height <= 1:
            self._fit_pending = True
            return
        image_width, image_height = self.map.image_size
        self._fit_scale = max(0.001, min((width - 12) / image_width, (height - 12) / image_height))
        initial_factor = self._initial_fit_zoom if not self._initial_fit_applied else 1.0
        self._view_scale = min(
            max(self._fit_scale * initial_factor, self._minimum_scale()),
            max(self._maximum_zoom, self._minimum_scale()),
        )
        if not self._initial_fit_applied:
            self._initial_fit_applied = True
        self._offset_x = (width - image_width * self._view_scale) / 2
        self._offset_y = (height - image_height * self._view_scale) / 2
        self._constrain_view_offsets()
        self._fit_pending = False
        self._request_redraw()

    def _canvas_size(self) -> tuple[int, int]:
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _rotation_radians(self) -> float:
        return math.radians(self._view_rotation_deg)

    def _rotate_forward(self, dx: float, dy: float) -> tuple[float, float]:
        """Applique la rotation visuelle Pillow/Canvas (angle positif anti-horaire)."""
        cosine = math.cos(self._rotation_radians())
        sine = math.sin(self._rotation_radians())
        return cosine * dx + sine * dy, -sine * dx + cosine * dy

    def _rotate_inverse(self, dx: float, dy: float) -> tuple[float, float]:
        cosine = math.cos(self._rotation_radians())
        sine = math.sin(self._rotation_radians())
        return cosine * dx - sine * dy, sine * dx + cosine * dy

    def _map_to_screen(self, x_map: float, y_map: float) -> tuple[float, float]:
        base_x = self._offset_x + float(x_map) * self._view_scale
        base_y = self._offset_y + float(y_map) * self._view_scale
        canvas_width, canvas_height = self._canvas_size()
        rotated_x, rotated_y = self._rotate_forward(
            base_x - canvas_width / 2,
            base_y - canvas_height / 2,
        )
        return rotated_x + canvas_width / 2, rotated_y + canvas_height / 2

    def _screen_to_map(self, x_screen: float, y_screen: float) -> tuple[float, float]:
        canvas_width, canvas_height = self._canvas_size()
        base_dx, base_dy = self._rotate_inverse(
            float(x_screen) - canvas_width / 2,
            float(y_screen) - canvas_height / 2,
        )
        base_x = base_dx + canvas_width / 2
        base_y = base_dy + canvas_height / 2
        return (
            (base_x - self._offset_x) / self._view_scale,
            (base_y - self._offset_y) / self._view_scale,
        )

    def _minimum_scale(self) -> float:
        return max(0.0001, self._fit_scale * self._minimum_fit_zoom)

    def _request_redraw(self) -> None:
        """Regroupe les rafraîchissements successifs produits par les interactions."""
        if self._redraw_after_id is None:
            self._redraw_after_id = self.after_idle(self._redraw)

    def _on_resize(self, _event):
        if self._fit_pending:
            self.fit_to_view()
        else:
            self._constrain_view_offsets()
            self._request_redraw()

    def _on_press(self, event):
        self._hide_tooltip()
        marker_id = self._marker_at(event.x, event.y)
        if marker_id is not None and self.on_marker_dragged is not None:
            self._dragging_marker_id = marker_id
            if self.on_marker_drag_started is not None:
                self.on_marker_drag_started(marker_id)
            return
        self._press_position = (event.x, event.y)
        self._pan_last_position = (event.x, event.y)
        self._drag_distance = 0.0

    def _on_drag(self, event):
        if self._dragging_marker_id is not None:
            self.on_marker_dragged(
                self._dragging_marker_id,
                self.screen_to_lambert(event.x, event.y),
            )
            return
        if self._pan_last_position is None:
            return
        previous_x, previous_y = self._pan_last_position
        delta_x, delta_y = self._rotate_inverse(
            event.x - previous_x,
            event.y - previous_y,
        )
        self._offset_x += delta_x
        self._offset_y += delta_y
        self._pan_last_position = (event.x, event.y)
        if self._press_position is not None:
            press_x, press_y = self._press_position
            self._drag_distance = max(self._drag_distance, ((event.x - press_x) ** 2 + (event.y - press_y) ** 2) ** 0.5)
        self._constrain_view_offsets()
        self._request_redraw()

    def _on_release(self, event):
        if self._dragging_marker_id is not None:
            marker_id = self._dragging_marker_id
            self._dragging_marker_id = None
            if self.on_marker_drag_released is not None:
                self.on_marker_drag_released(marker_id)
            return
        was_click = self._press_position is not None and self._drag_distance <= self._CLICK_DRAG_THRESHOLD
        self._press_position = None
        self._pan_last_position = None
        self._drag_distance = 0.0
        if was_click:
            self._select_marker_at(event.x, event.y)

    def _on_zoom(self, event):
        self._zoom_at(event.x, event.y, 1.15 if event.delta > 0 else 1 / 1.15)

    def _zoom_at(self, x_screen: float, y_screen: float, factor: float) -> None:
        if self.map is None:
            return
        self._hide_tooltip()
        old_scale = self._view_scale
        new_scale = min(
            max(old_scale * factor, self._minimum_scale()),
            max(self._maximum_zoom, self._minimum_scale()),
        )
        map_x, map_y = self._screen_to_map(x_screen, y_screen)
        self._view_scale = new_scale
        canvas_width, canvas_height = self._canvas_size()
        base_dx, base_dy = self._rotate_inverse(
            x_screen - canvas_width / 2,
            y_screen - canvas_height / 2,
        )
        self._offset_x = base_dx + canvas_width / 2 - map_x * new_scale
        self._offset_y = base_dy + canvas_height / 2 - map_y * new_scale
        self._constrain_view_offsets()
        self._request_redraw()

    def _select_marker_at(self, x_screen: float, y_screen: float) -> None:
        closest_id = self._marker_at(x_screen, y_screen)
        self._selected_marker_id = closest_id
        self._request_redraw()
        if self.on_marker_selected is not None:
            self.on_marker_selected(closest_id)

    def _marker_at(self, x_screen: float, y_screen: float) -> object | None:
        closest_id = None
        closest_distance = self._MARKER_HIT_RADIUS
        for marker_id, (x_marker, y_marker) in self._marker_screen_positions.items():
            distance = ((x_screen - x_marker) ** 2 + (y_screen - y_marker) ** 2) ** 0.5
            if distance <= closest_distance:
                closest_id, closest_distance = marker_id, distance
        return closest_id

    def _constrain_view_offsets(self) -> None:
        """Borne le viewport dans l'image, ou centre celle-ci si elle est plus petite."""
        if self.map is None:
            return
        canvas_width, canvas_height = self._canvas_size()
        image_width, image_height = self.map.image_size
        rendered_width = image_width * self._view_scale
        rendered_height = image_height * self._view_scale
        if rendered_width <= canvas_width:
            self._offset_x = (canvas_width - rendered_width) / 2
        else:
            self._offset_x = min(0.0, max(canvas_width - rendered_width, self._offset_x))
        if rendered_height <= canvas_height:
            self._offset_y = (canvas_height - rendered_height) / 2
        else:
            self._offset_y = min(0.0, max(canvas_height - rendered_height, self._offset_y))

    def _on_motion(self, event) -> None:
        if self._pan_last_position is not None:
            return
        marker_id = self._marker_at(event.x, event.y)
        self._hover_root_position = (event.x_root, event.y_root)
        if marker_id == self._hover_marker_id:
            if marker_id is not None and self._tooltip_marker_id == marker_id:
                self._position_tooltip()
            return
        self._hide_tooltip()
        self._hover_marker_id = marker_id
        if marker_id is not None:
            self._tooltip_after_id = self.after(self._TOOLTIP_DELAY_MS, self._show_tooltip)

    def _show_tooltip(self) -> None:
        self._tooltip_after_id = None
        marker_id = self._hover_marker_id
        marker = next((item for item in self._markers if item.marker_id == marker_id), None)
        if marker is None:
            return
        tooltip_text = marker.tooltip if marker.tooltip is not None else marker.label
        if not tooltip_text:
            return
        if self._tooltip is None:
            self._tooltip = tk.Toplevel(self)
            self._tooltip.overrideredirect(True)
            try:
                self._tooltip.attributes("-disabled", True)
            except tk.TclError:
                pass
            self._tooltip_label = tk.Label(self._tooltip, relief=tk.SOLID, borderwidth=1, padx=5, pady=2)
            self._tooltip_label.pack()
        self._tooltip_label.config(text=tooltip_text)
        self._tooltip_marker_id = marker_id
        self._position_tooltip()
        self._tooltip.deiconify()

    def _position_tooltip(self) -> None:
        if self._tooltip is None or self._hover_root_position is None:
            return
        x_root, y_root = self._hover_root_position
        self._tooltip.update_idletasks()
        width, height = self._tooltip.winfo_reqwidth(), self._tooltip.winfo_reqheight()
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        x_root = min(x_root + 14, screen_width - width - 4)
        y_root = min(y_root + 16, screen_height - height - 4)
        self._tooltip.geometry(f"+{max(0, x_root)}+{max(0, y_root)}")

    def _hide_tooltip(self) -> None:
        if self._tooltip_after_id is not None:
            self.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None
        self._hover_marker_id = None
        self._tooltip_marker_id = None
        if self._tooltip is not None:
            self._tooltip.withdraw()

    def _redraw(self) -> None:
        self._redraw_after_id = None
        self._hide_tooltip()
        self.canvas.delete("all")
        self._marker_screen_positions = {}
        if self.map is None or self._source_image is None:
            return
        image_width, image_height = self.map.image_size
        canvas_width, canvas_height = self._canvas_size()
        source_width, source_height = self._source_image.size
        source_scale_x = source_width / image_width
        source_scale_y = source_height / image_height

        # Coordonnées du rectangle visible dans le repère de l'image calibrée.
        viewport_corners = (
            self._screen_to_map(0.0, 0.0),
            self._screen_to_map(float(canvas_width), 0.0),
            self._screen_to_map(float(canvas_width), float(canvas_height)),
            self._screen_to_map(0.0, float(canvas_height)),
        )
        visible_left = max(0.0, min(point[0] for point in viewport_corners))
        visible_top = max(0.0, min(point[1] for point in viewport_corners))
        visible_right = min(image_width, max(point[0] for point in viewport_corners))
        visible_bottom = min(image_height, max(point[1] for point in viewport_corners))
        if visible_right <= visible_left or visible_bottom <= visible_top:
            return

        # Conversion en pixels de l'image native, puis rendu de ce seul crop.
        source_left = max(0, math.floor(visible_left * source_scale_x))
        source_top = max(0, math.floor(visible_top * source_scale_y))
        source_right = min(source_width, math.ceil(visible_right * source_scale_x))
        source_bottom = min(source_height, math.ceil(visible_bottom * source_scale_y))
        if source_right <= source_left or source_bottom <= source_top:
            return
        # Le viewport initial peut couvrir la carte Michelin entière (ressource interne
        # déjà validée à son chargement). Pillow contrôle aussi la taille du résultat
        # de crop et émet alors un DecompressionBombWarning malgré cette provenance sûre.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            crop = self._source_image.crop((source_left, source_top, source_right, source_bottom))
        crop_map_width = (source_right - source_left) / source_scale_x
        crop_map_height = (source_bottom - source_top) / source_scale_y
        rendered = crop.resize(
            (max(1, round(crop_map_width * self._view_scale)), max(1, round(crop_map_height * self._view_scale))),
            Image.Resampling.LANCZOS,
        )
        if self._view_rotation_deg:
            rendered = rendered.rotate(
                self._view_rotation_deg,
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor="white",
            )
        self._photo = ImageTk.PhotoImage(rendered)
        crop_map_left = source_left / source_scale_x
        crop_map_top = source_top / source_scale_y
        crop_center_x = crop_map_left + crop_map_width / 2
        crop_center_y = crop_map_top + crop_map_height / 2
        crop_screen_x, crop_screen_y = self._map_to_screen(crop_center_x, crop_center_y)
        self.canvas.create_image(
            crop_screen_x,
            crop_screen_y,
            image=self._photo,
            anchor="center",
        )
        for polyline in self._polylines:
            screen_points = []
            for latitude, longitude in polyline.points:
                x_map, y_map = self.map.geographic_to_pixel(latitude, longitude)
                screen_points.extend(self._map_to_screen(x_map, y_map))
            if len(screen_points) >= 4:
                if polyline.closed:
                    screen_points.extend(screen_points[:2])
                self.canvas.create_line(
                    *screen_points,
                    fill=polyline.color if polyline.color is not None else "#2e7d32",
                    width=polyline.width,
                    joinstyle=tk.ROUND,
                )
        for marker in self._markers:
            x_map, y_map = self.map.geographic_to_pixel(marker.latitude, marker.longitude)
            x_screen, y_screen = self._map_to_screen(x_map, y_map)
            self._marker_screen_positions[marker.marker_id] = (x_screen, y_screen)
            selected = marker.marker_id == self._selected_marker_id
            radius = 8 if selected else 6
            color = marker.fill_color if marker.fill_color is not None else ("#d52b1e" if selected else "#1565c0")
            outline = marker.outline_color if marker.outline_color is not None else "white"
            self.canvas.create_oval(x_screen - radius, y_screen - radius, x_screen + radius, y_screen + radius,
                                    fill=color, outline=outline, width=2)
            if (selected or marker.always_show_label) and marker.label:
                self.canvas.create_text(x_screen + 10, y_screen - 10, text=marker.label, anchor="sw",
                                        fill=marker.label_color or "#202020", font=("TkDefaultFont", 9, "bold"))
