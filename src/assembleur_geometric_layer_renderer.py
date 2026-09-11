"""Rendu Canvas indépendant des fenêtres pour les documents Traces V2 validés."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping

from src.assembleur_geometric_layer_io import GeometricLayerDocument, GeometricLayerTrace
from src.assembleur_geometric_layer_display import GeometricLayerModuleDisplayOverride


_CIRCLE_SEGMENTS = 48
_POINT_RADIUS_PX = 4
_STYLE_OPTIONS: dict[str, tuple[int, int] | None] = {"plein": None, "dash": (6, 4), "Arrow": None}
_LAYER_TAG = "geometric-layer"


@dataclass(frozen=True)
class GeometricLayerRenderContext:
    """Conversions runtime ; le document Traces ne reçoit aucun cache écran."""

    image_size: tuple[float, float]
    lambert_to_image: Callable[[float, float], tuple[float, float]]
    image_to_canvas: Callable[[float, float], tuple[float, float]]

    def lambert_to_canvas(self, x_l93: float, y_l93: float) -> tuple[float, float]:
        return self.image_to_canvas(*self.lambert_to_image(x_l93, y_l93))


@dataclass(frozen=True)
class RenderedTrace:
    module_id: str
    trace: GeometricLayerTrace
    item_ids: tuple[int, ...]


@dataclass(frozen=True)
class GeometricLayerRenderResult:
    traces_drawn: int
    rendered_traces: tuple[RenderedTrace, ...]


def bgr_to_tk_color(color_bgr: tuple[float, float, float]) -> str:
    """Convertit explicitement la couleur BGR portable de Traces en #RRGGBB Tk."""
    if len(color_bgr) != 3 or any(value != int(value) or not 0 <= value <= 255 for value in color_bgr):
        raise ValueError("color_bgr doit contenir trois entiers entre 0 et 255.")
    blue, green, red = (int(value) for value in color_bgr)
    return f"#{red:02X}{green:02X}{blue:02X}"


def clip_infinite_line_to_image(
    anchor: tuple[float, float], direction: tuple[float, float], image_size: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clippe une droite infinie au rectangle image [0,w]×[0,h], sans constante arbitraire."""
    x0, y0 = anchor
    dx, dy = direction
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size doit être strictement positive.")
    if dx == 0 and dy == 0:
        raise ValueError("Une droite à clipper doit avoir un vecteur non nul.")
    intersections: list[tuple[float, float, float]] = []
    if dx != 0:
        for x in (0.0, float(width)):
            t = (x - x0) / dx
            y = y0 + t * dy
            if 0.0 <= y <= height:
                intersections.append((t, x, y))
    if dy != 0:
        for y in (0.0, float(height)):
            t = (y - y0) / dy
            x = x0 + t * dx
            if 0.0 <= x <= width:
                intersections.append((t, x, y))
    unique: list[tuple[float, float, float]] = []
    for item in sorted(intersections):
        if not unique or not math.isclose(item[1], unique[-1][1]) or not math.isclose(item[2], unique[-1][2]):
            unique.append(item)
    if len(unique) < 2:
        return None
    return (unique[0][1], unique[0][2]), (unique[-1][1], unique[-1][2])


class GeometricLayerRenderer:
    """Dessine par clear/redraw les traces validées dans un Canvas compatible Tk."""

    def __init__(self, canvas, context: GeometricLayerRenderContext) -> None:
        self.canvas = canvas
        self.context = context

    def clear(self) -> None:
        self.canvas.delete(_LAYER_TAG)

    def render_document(
        self,
        document: GeometricLayerDocument,
        *,
        module_ids: set[str] | None = None,
        display_overrides: Mapping[str, GeometricLayerModuleDisplayOverride] | None = None,
        clear: bool = False,
    ) -> GeometricLayerRenderResult:
        if clear:
            self.clear()
        rendered: list[RenderedTrace] = []
        for module in document.modules:
            if module_ids is not None and module.module_id not in module_ids:
                continue
            display_override = display_overrides.get(module.module_id) if display_overrides is not None else None
            for trace in module.traces:
                if not trace.graphics.visible:
                    continue
                item_ids, label_anchor = self._draw_trace(trace, module.module_id, display_override)
                if not item_ids:
                    continue
                if trace.graphics.show_name and trace.graphics.name:
                    item_ids.append(self.canvas.create_text(
                        *label_anchor, text=trace.graphics.name, anchor="sw", fill=bgr_to_tk_color(
                            display_override.color_bgr if display_override is not None and display_override.color_bgr is not None else trace.graphics.color_bgr
                        ),
                        tags=(_LAYER_TAG, f"{_LAYER_TAG}:{module.module_id}"),
                    ))
                rendered.append(RenderedTrace(module.module_id, trace, tuple(item_ids)))
        return GeometricLayerRenderResult(len(rendered), tuple(rendered))

    def _canvas_options(
        self,
        trace: GeometricLayerTrace,
        module_id: str,
        display_override: GeometricLayerModuleDisplayOverride | None,
    ) -> dict:
        dash = _STYLE_OPTIONS.get(trace.graphics.style)
        if trace.graphics.style not in _STYLE_OPTIONS:
            raise ValueError(f"Style Traces non supporté par le renderer : {trace.graphics.style!r}.")
        options = {
            "fill": bgr_to_tk_color(
                display_override.color_bgr if display_override is not None and display_override.color_bgr is not None else trace.graphics.color_bgr
            ),
            "width": display_override.width if display_override is not None and display_override.width is not None else trace.graphics.width,
            "tags": (_LAYER_TAG, f"{_LAYER_TAG}:{module_id}"),
        }
        if dash is not None:
            options["dash"] = dash
        return options

    def _draw_trace(
        self,
        trace: GeometricLayerTrace,
        module_id: str,
        display_override: GeometricLayerModuleDisplayOverride | None,
    ) -> tuple[list[int], tuple[float, float]]:
        geometry = trace.geometry
        kind = trace.geometry_type
        options = self._canvas_options(trace, module_id, display_override)
        if kind == "point":
            x, y = self.context.lambert_to_canvas(geometry["x_l93"], geometry["y_l93"])
            return [self.canvas.create_oval(x - _POINT_RADIUS_PX, y - _POINT_RADIUS_PX, x + _POINT_RADIUS_PX, y + _POINT_RADIUS_PX, outline=options["fill"], fill=options["fill"], width=options["width"], tags=options["tags"])], (x, y)
        if kind == "symbol":
            x, y = self.context.lambert_to_canvas(geometry["x_l93"], geometry["y_l93"])
            radius = _POINT_RADIUS_PX + 2
            return [self.canvas.create_polygon(x, y - radius, x + radius, y, x, y + radius, x - radius, y, outline=options["fill"], fill="", width=options["width"], tags=options["tags"])], (x, y)
        if kind in {"circle", "arc"}:
            image_points = self._sample_round_geometry_image(geometry, kind)
            canvas_points = [self.context.image_to_canvas(x, y) for x, y in image_points]
            item_ids = [self._create_polyline(canvas_points, options)]
            if kind == "arc" and trace.graphics.style == "Arrow":
                item_ids.append(self._create_polyline(canvas_points[-2:], {**options, "arrow": "last"}))
            return item_ids, canvas_points[len(canvas_points) // 2]
        if kind == "segment":
            points = [self.context.lambert_to_canvas(geometry["x1_l93"], geometry["y1_l93"]), self.context.lambert_to_canvas(geometry["x2_l93"], geometry["y2_l93"])]
            return [self._create_polyline(points, options)], points[0]
        if kind == "line_between_points":
            first = self.context.lambert_to_image(geometry["x1_l93"], geometry["y1_l93"])
            second = self.context.lambert_to_image(geometry["x2_l93"], geometry["y2_l93"])
            clipped = clip_infinite_line_to_image(first, (second[0] - first[0], second[1] - first[1]), self.context.image_size)
            return self._draw_clipped_line(clipped, options)
        if kind == "line_image_azimuth":
            anchor = self.context.lambert_to_image(geometry["x_l93"], geometry["y_l93"])
            angle = math.radians(geometry["azimuth_deg"])
            return self._draw_clipped_line(clip_infinite_line_to_image(anchor, (math.sin(angle), -math.cos(angle)), self.context.image_size), options)
        if kind == "vertical_image_line":
            anchor = self.context.lambert_to_image(geometry["x_l93"], geometry["y_l93"])
            return self._draw_clipped_line(clip_infinite_line_to_image(anchor, (0.0, 1.0), self.context.image_size), options)
        if kind == "horizontal_image_line":
            anchor = self.context.lambert_to_image(geometry["x_l93"], geometry["y_l93"])
            return self._draw_clipped_line(clip_infinite_line_to_image(anchor, (1.0, 0.0), self.context.image_size), options)
        raise RuntimeError(f"Type de géométrie Traces impossible pour le renderer : {kind!r}.")

    def _draw_clipped_line(self, clipped, options: dict) -> tuple[list[int], tuple[float, float]]:
        if clipped is None:
            return [], (0.0, 0.0)
        points = [self.context.image_to_canvas(*point) for point in clipped]
        return [self._create_polyline(points, options)], points[0]

    def _create_polyline(self, points: Iterable[tuple[float, float]], options: dict) -> int:
        coordinates = [coordinate for point in points for coordinate in point]
        return self.canvas.create_line(*coordinates, **options)

    def _sample_round_geometry_image(self, geometry: Mapping[str, object], kind: str) -> list[tuple[float, float]]:
        center_x, center_y = geometry["center_x_l93"], geometry["center_y_l93"]
        radius_m = geometry["radius_km"] * 1000.0
        image_center = self.context.lambert_to_image(center_x, center_y)
        image_edge = self.context.lambert_to_image(center_x + radius_m, center_y)
        radius_image = math.dist(image_center, image_edge)
        if kind == "circle":
            start, rotation, count = 0.0, 360.0, _CIRCLE_SEGMENTS
        else:
            start, rotation = geometry["start_azimuth_deg"], geometry["rotation_deg"]
            count = max(1, math.ceil(abs(rotation) / 360.0 * _CIRCLE_SEGMENTS))
        return [
            (
                image_center[0] + radius_image * math.sin(math.radians(start + rotation * index / count)),
                image_center[1] - radius_image * math.cos(math.radians(start + rotation * index / count)),
            )
            for index in range(count + 1)
        ]


def render_document(
    document: GeometricLayerDocument,
    canvas,
    context: GeometricLayerRenderContext,
    *,
    module_ids: set[str] | None = None,
    display_overrides: Mapping[str, GeometricLayerModuleDisplayOverride] | None = None,
    clear: bool = False,
) -> GeometricLayerRenderResult:
    """Raccourci sans état pour un rendu ponctuel du document validé."""
    return GeometricLayerRenderer(canvas, context).render_document(
        document,
        module_ids=module_ids,
        display_overrides=display_overrides,
        clear=clear,
    )
