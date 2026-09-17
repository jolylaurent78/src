"""Headless print model and raster renderer for AssembleurTriangles.

This module deliberately knows nothing about Tk or the live viewer.  A print
dialog owns its settings and viewport; the viewer supplies one immutable-ish
business snapshot when the dialog is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PAPER_SIZES_MM = {"A4": (210.0, 297.0)}
DEFAULT_TITLE_HEIGHT_MM = 8.0
DEFAULT_TITLE_GAP_MM = 0.9
DEFAULT_TITLE_FONT_MM = 5.0
MIN_TITLE_FONT_MM = 2.5
MARGIN_PRESETS_MM = {"Minimales": 3.0, "Étroites": 5.0, "Standard": 10.0}
_LAYERS = ("map", "assembly", "beacons")


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} ne peut pas être booléen.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} doit être numérique.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} doit être fini.")
    return result


@dataclass(frozen=True)
class AssembleurPrintSettings:
    paper_format: str = "A4"
    orientation: str = "portrait"
    title: str = "Assemblage"
    margin_mm: float = 5.0
    selected_layers: tuple[str, ...] = _LAYERS
    map_opacity: int = 70

    def __post_init__(self) -> None:
        if self.paper_format not in PAPER_SIZES_MM:
            raise ValueError(f"Format papier inconnu : {self.paper_format!r}")
        if self.orientation not in {"portrait", "landscape"}:
            raise ValueError("orientation doit être 'portrait' ou 'landscape'.")
        object.__setattr__(self, "title", str(self.title or "Assemblage").strip() or "Assemblage")
        margin = _finite(self.margin_mm, "margin_mm")
        if margin < 0:
            raise ValueError("margin_mm doit être >= 0.")
        object.__setattr__(self, "margin_mm", margin)
        layers = tuple(dict.fromkeys(str(layer) for layer in self.selected_layers))
        if any(layer not in _LAYERS for layer in layers):
            raise ValueError(f"selected_layers invalide : {layers!r}")
        object.__setattr__(self, "selected_layers", layers)
        if isinstance(self.map_opacity, bool) or not isinstance(self.map_opacity, int):
            raise TypeError("map_opacity doit être un int (bool interdit).")
        if not 0 <= self.map_opacity <= 100:
            raise ValueError("map_opacity doit être compris entre 0 et 100.")

    @property
    def page_size_mm(self) -> tuple[float, float]:
        width, height = PAPER_SIZES_MM[self.paper_format]
        return (width, height) if self.orientation == "portrait" else (height, width)

    def has_layer(self, layer: str) -> bool:
        return layer in self.selected_layers


@dataclass(frozen=True)
class AssembleurPrintViewport:
    min_x: float
    min_y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name in ("min_x", "min_y", "width", "height"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width et height doivent être strictement positifs.")

    @property
    def max_x(self) -> float:
        return self.min_x + self.width

    @property
    def max_y(self) -> float:
        return self.min_y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.min_x + self.width / 2, self.min_y + self.height / 2)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def expanded_to_aspect(self, aspect: float) -> "AssembleurPrintViewport":
        aspect = _finite(aspect, "aspect")
        if aspect <= 0:
            raise ValueError("aspect doit être strictement positif.")
        cx, cy = self.center
        if self.aspect_ratio < aspect:
            width, height = self.height * aspect, self.height
        else:
            width, height = self.width, self.width / aspect
        return AssembleurPrintViewport(cx - width / 2, cy - height / 2, width, height)

    def zoom_at(self, x_fraction: float, y_fraction: float, factor: float) -> "AssembleurPrintViewport":
        x_fraction = _finite(x_fraction, "x_fraction")
        y_fraction = _finite(y_fraction, "y_fraction")
        factor = _finite(factor, "factor")
        if not factor > 0:
            raise ValueError("factor doit être strictement positif.")
        anchor_x = self.min_x + self.width * x_fraction
        # preview y goes down, World y goes up
        anchor_y = self.max_y - self.height * y_fraction
        width, height = self.width / factor, self.height / factor
        return AssembleurPrintViewport(
            anchor_x - width * x_fraction,
            anchor_y - height * (1 - y_fraction),
            width,
            height,
        )

    def panned(self, dx_fraction: float, dy_fraction: float) -> "AssembleurPrintViewport":
        return AssembleurPrintViewport(
            self.min_x - _finite(dx_fraction, "dx_fraction") * self.width,
            self.min_y + _finite(dy_fraction, "dy_fraction") * self.height,
            self.width,
            self.height,
        )


@dataclass(frozen=True)
class AssembleurPrintMap:
    image: Image.Image
    x0: float
    y0: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not isinstance(self.image, Image.Image):
            raise TypeError("image doit être une image PIL.")
        for name in ("x0", "y0", "width", "height"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Les dimensions de la carte doivent être positives.")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x0 + self.width, self.y0 + self.height)


@dataclass(frozen=True)
class AssembleurPrintTriangle:
    element_id: str
    o: tuple[float, float]
    b: tuple[float, float]
    l: tuple[float, float]
    labels: tuple[str, str, str]
    display_label: str

    def __post_init__(self) -> None:
        for name in ("o", "b", "l"):
            point = getattr(self, name)
            if len(point) != 2:
                raise ValueError(f"{name} doit contenir deux coordonnées.")
            object.__setattr__(self, name, (_finite(point[0], name), _finite(point[1], name)))
        if len(self.labels) != 3:
            raise ValueError("Un triangle doit posséder trois labels.")

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        return (self.o, self.b, self.l)


@dataclass(frozen=True)
class AssembleurPrintBeacon:
    beacon_id: str
    position: tuple[float, float]
    name: str

    def __post_init__(self) -> None:
        if len(self.position) != 2:
            raise ValueError("position doit contenir deux coordonnées.")
        object.__setattr__(self, "position", (_finite(self.position[0], "x"), _finite(self.position[1], "y")))


@dataclass(frozen=True)
class AssembleurPrintSnapshot:
    scenario_name: str
    map_snapshot: AssembleurPrintMap | None
    triangles: tuple[AssembleurPrintTriangle, ...]
    beacons: tuple[AssembleurPrintBeacon, ...]
    contour_only: bool = False
    boundary_segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()


@dataclass(frozen=True)
class PrintLayout:
    page_width_mm: float
    page_height_mm: float
    title_rect_mm: tuple[float, float, float, float]
    map_rect_mm: tuple[float, float, float, float]


def calculate_print_layout(settings: AssembleurPrintSettings) -> PrintLayout:
    """Return the single shared title/map layout, with rectangles x/y/w/h."""
    page_w, page_h = settings.page_size_mm
    margin = settings.margin_mm
    usable_w = max(1.0, page_w - 2 * margin)
    title_y = margin
    title_h = min(_compute_title_height_mm(settings), max(1.0, page_h - 2 * margin))
    map_y = title_y + title_h + _compute_title_gap_mm(settings)
    map_h = max(1.0, page_h - margin - map_y)
    return PrintLayout(page_w, page_h, (margin, title_y, usable_w, title_h), (margin, map_y, usable_w, map_h))


def _compute_title_height_mm(settings: AssembleurPrintSettings) -> float:
    """Keep the title band compact while giving Standard margins modest breathing room."""
    return min(9.4, max(DEFAULT_TITLE_HEIGHT_MM, DEFAULT_TITLE_HEIGHT_MM + .2 * (settings.margin_mm - 3.0)))


def _compute_title_gap_mm(settings: AssembleurPrintSettings) -> float:
    """Scale the visual separation with the selected margin without ever removing it."""
    return min(3.0, max(DEFAULT_TITLE_GAP_MM, settings.margin_mm * .3))


def fit_initial_viewport(
    snapshot: AssembleurPrintSnapshot,
    settings: AssembleurPrintSettings,
    fallback: AssembleurPrintViewport | None = None,
) -> AssembleurPrintViewport:
    points: list[tuple[float, float]] = []
    if settings.has_layer("assembly"):
        points.extend(point for triangle in snapshot.triangles for point in triangle.points)
    if settings.has_layer("beacons"):
        points.extend(beacon.position for beacon in snapshot.beacons)
    if not points and settings.has_layer("map") and snapshot.map_snapshot is not None:
        x0, y0, x1, y1 = snapshot.map_snapshot.bounds
        points.extend(((x0, y0), (x1, y1)))
    if not points:
        if fallback is not None:
            return fallback.expanded_to_aspect(_layout_aspect(settings))
        points = [(-50.0, -50.0), (50.0, 50.0)]
    xs, ys = zip(*points)
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    extent = max(width, height, 1.0)
    pad = extent * 0.05
    viewport = AssembleurPrintViewport(min(xs) - pad, min(ys) - pad, max(width + 2 * pad, 1.0), max(height + 2 * pad, 1.0))
    return viewport.expanded_to_aspect(_layout_aspect(settings))


def _layout_aspect(settings: AssembleurPrintSettings) -> float:
    _x, _y, width, height = calculate_print_layout(settings).map_rect_mm
    return width / height


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf", "Arial Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf", "Arial.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, max(1, size))
        except OSError:
            continue
    return ImageFont.load_default()


def _world_to_raster(viewport: AssembleurPrintViewport, width: int, height: int, point: tuple[float, float]) -> tuple[float, float]:
    return ((point[0] - viewport.min_x) * width / viewport.width, (viewport.max_y - point[1]) * height / viewport.height)


def render_print_raster(snapshot: AssembleurPrintSnapshot, settings: AssembleurPrintSettings, viewport: AssembleurPrintViewport, output_width_px: int, output_height_px: int) -> Image.Image:
    """Render only the map area in RGB, using the shared World-to-raster transform."""
    width, height = int(output_width_px), int(output_height_px)
    if width <= 0 or height <= 0:
        raise ValueError("La taille de raster doit être positive.")
    result = Image.new("RGB", (width, height), "white")
    if settings.has_layer("map") and settings.map_opacity > 0 and snapshot.map_snapshot is not None:
        _render_map(result, snapshot.map_snapshot, viewport, settings.map_opacity)
    draw = ImageDraw.Draw(result)
    scale = max(1.0, min(width, height) / 650.0)
    if settings.has_layer("assembly"):
        _render_assembly(draw, snapshot, viewport, width, height, scale)
    if settings.has_layer("beacons"):
        _render_beacons(draw, snapshot.beacons, viewport, width, height, scale)
    return result


def _render_map(result: Image.Image, map_snapshot: AssembleurPrintMap, viewport: AssembleurPrintViewport, opacity: int) -> None:
    mx0, my0, mx1, my1 = map_snapshot.bounds
    ix0, iy0 = max(mx0, viewport.min_x), max(my0, viewport.min_y)
    ix1, iy1 = min(mx1, viewport.max_x), min(my1, viewport.max_y)
    if ix0 >= ix1 or iy0 >= iy1:
        return
    source = map_snapshot.image
    sw, sh = source.size
    left = max(0, min(sw - 1, int((ix0 - mx0) / map_snapshot.width * sw)))
    upper = max(0, min(sh - 1, int((my1 - iy1) / map_snapshot.height * sh)))
    right = max(left + 1, min(sw, math.ceil((ix1 - mx0) / map_snapshot.width * sw)))
    lower = max(upper + 1, min(sh, math.ceil((my1 - iy0) / map_snapshot.height * sh)))
    crop = source.crop((left, upper, right, lower)).convert("RGBA")
    x0, y0 = _world_to_raster(viewport, result.width, result.height, (ix0, iy1))
    x1, y1 = _world_to_raster(viewport, result.width, result.height, (ix1, iy0))
    box = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    crop = crop.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    if opacity < 100:
        alpha = crop.getchannel("A").point(lambda value: value * opacity // 100)
        crop.putalpha(alpha)
    result.paste(crop, box[:2], crop)


def _render_assembly(draw: ImageDraw.ImageDraw, snapshot: AssembleurPrintSnapshot, viewport: AssembleurPrintViewport, width: int, height: int, scale: float) -> None:
    line_width, marker = max(1, round(2 * scale)), max(3, round(6 * scale))
    label_font, tri_font = _font(round(8 * scale)), _font(round(10 * scale), bold=True)
    for triangle in snapshot.triangles:
        o, b, l = (_world_to_raster(viewport, width, height, point) for point in triangle.points)
        if not snapshot.contour_only:
            draw.line((o, l), fill="black", width=line_width)
            draw.line((b, l), fill="#00008b", width=line_width)
            draw.line((b, o), fill="#808080", width=line_width)
        for point, color in ((o, "black"), (b, "#0000ff"), (l, "#ffd700")):
            draw.ellipse((point[0] - marker, point[1] - marker, point[0] + marker, point[1] + marker), fill=color, outline="black", width=max(1, round(scale)))
        cx, cy = ((o[0] + b[0] + l[0]) / 3, (o[1] + b[1] + l[1]) / 3)
        for point, label in zip((o, b, l), triangle.labels):
            if point is o or not str(label).strip():
                continue
            x, y = point[0] * .65 + cx * .35, point[1] * .65 + cy * .35
            _centered_text(draw, (x, y), str(label), label_font, "black")
        _centered_text(draw, (cx, cy), triangle.display_label, tri_font, "red")
    if snapshot.contour_only:
        for p0, p1 in snapshot.boundary_segments:
            draw.line((_world_to_raster(viewport, width, height, p0), _world_to_raster(viewport, width, height, p1)), fill="black", width=max(1, round(3 * scale)))


def _render_beacons(draw: ImageDraw.ImageDraw, beacons: Iterable[AssembleurPrintBeacon], viewport: AssembleurPrintViewport, width: int, height: int, scale: float) -> None:
    radius, font = max(3, round(5 * scale)), _font(round(8 * scale))
    for beacon in beacons:
        x, y = _world_to_raster(viewport, width, height, beacon.position)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="black")
        _centered_text(draw, (x, y + radius + 2), beacon.name, font, "black", anchor="mt")


def _centered_text(draw: ImageDraw.ImageDraw, position: tuple[float, float], text: str, font: ImageFont.ImageFont, fill: str, anchor: str = "mm") -> None:
    draw.text(position, text, font=font, fill=fill, anchor=anchor)


def render_print_page_raster(snapshot: AssembleurPrintSnapshot, settings: AssembleurPrintSettings, viewport: AssembleurPrintViewport, dpi: int) -> Image.Image:
    """Render the complete page; preview and PDF both call this exact function."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi doit être un entier strictement positif.")
    layout = calculate_print_layout(settings)
    factor = dpi / 25.4
    page = Image.new("RGB", (round(layout.page_width_mm * factor), round(layout.page_height_mm * factor)), "white")
    x, y, map_w, map_h = layout.map_rect_mm
    map_size = (max(1, round(map_w * factor)), max(1, round(map_h * factor)))
    page.paste(render_print_raster(snapshot, settings, viewport, *map_size), (round(x * factor), round(y * factor)))
    _render_title(page, settings.title, layout.title_rect_mm, factor)
    return page


def _render_title(page: Image.Image, title: str, rect_mm: tuple[float, float, float, float], factor: float) -> None:
    x, y, width, height = (round(value * factor) for value in rect_mm)
    draw = ImageDraw.Draw(page)
    max_size = max(1, round(DEFAULT_TITLE_FONT_MM * factor))
    min_size = max(1, round(MIN_TITLE_FONT_MM * factor))
    sizes = list(range(max_size, min_size - 1, -1))
    # Very long titles may use a smaller fallback only when two lines at the
    # minimum nominal size still cannot fit.
    sizes.extend(range(min_size - 1, 0, -1))
    for size in sizes:
        font = _font(size, bold=True)
        lines = _wrap_title(draw, title, font, width)
        if len(lines) <= 2:
            line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
            if len(lines) * line_height <= height:
                start_y = y + (height - len(lines) * line_height) / 2 + line_height / 2
                for index, line in enumerate(lines):
                    _centered_text(draw, (x + width / 2, start_y + index * line_height), line, font, "black")
                return


def _wrap_title(draw: ImageDraw.ImageDraw, title: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    """Wrap by words when possible and by glyph only for an overlong token."""
    lines: list[str] = []
    current = ""
    for char in title:
        candidate = current + char
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > width:
            lines.append(current.rstrip())
            current = char.lstrip()
        else:
            current = candidate
    if current:
        lines.append(current.rstrip())
    return lines or [""]
