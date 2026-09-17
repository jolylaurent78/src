from __future__ import annotations

import pytest
from PIL import Image

from src.assembleur_map_print import (
    AssembleurPrintBeacon,
    AssembleurPrintMap,
    AssembleurPrintSettings,
    AssembleurPrintSnapshot,
    AssembleurPrintTriangle,
    AssembleurPrintViewport,
    DEFAULT_TITLE_FONT_MM,
    MIN_TITLE_FONT_MM,
    calculate_print_layout,
    fit_initial_viewport,
    render_print_page_raster,
    render_print_raster,
)


def _snapshot(*, with_map: bool = True) -> AssembleurPrintSnapshot:
    map_snapshot = AssembleurPrintMap(Image.new("RGB", (20, 20), "red"), 0, 0, 20, 20) if with_map else None
    return AssembleurPrintSnapshot(
        "Scénario", map_snapshot,
        (AssembleurPrintTriangle("E1", (2, 2), (8, 2), (2, 8), ("O", "B", "L"), "T1"),),
        (AssembleurPrintBeacon("B1", (15, 15), "Paris"),),
    )


def test_settings_validates_and_normalizes_layers():
    settings = AssembleurPrintSettings(selected_layers=("map", "map", "beacons"))
    assert settings.selected_layers == ("map", "beacons")
    with pytest.raises(TypeError):
        AssembleurPrintSettings(map_opacity=True)
    with pytest.raises(ValueError):
        AssembleurPrintSettings(orientation="square")


def test_viewport_aspect_zoom_pan_and_no_map_clamp():
    viewport = AssembleurPrintViewport(-10, -10, 20, 10)
    resized = viewport.expanded_to_aspect(1)
    assert resized.center == viewport.center
    zoomed = resized.zoom_at(.25, .75, 2)
    assert zoomed.width == pytest.approx(resized.width / 2)
    assert zoomed.height == pytest.approx(resized.height / 2)
    assert zoomed.panned(.5, -.5).min_x < -10
    with pytest.raises(ValueError):
        AssembleurPrintViewport(0, 0, float("inf"), 1)


def test_auto_fit_prefers_geometry_then_map_then_fallback():
    settings = AssembleurPrintSettings(selected_layers=("map", "assembly"))
    fitted = fit_initial_viewport(_snapshot(), settings)
    assert fitted.min_x < 2 and fitted.max_x < 20  # map has not forced a full-map fit
    only_map = fit_initial_viewport(_snapshot(), AssembleurPrintSettings(selected_layers=("map",)))
    assert only_map.min_x <= 0 and only_map.max_x >= 20
    fallback = AssembleurPrintViewport(100, 200, 10, 10)
    empty = AssembleurPrintSnapshot("", None, (), ())
    assert fit_initial_viewport(empty, AssembleurPrintSettings(selected_layers=()), fallback).center == fallback.center


def test_layout_changes_with_orientation_and_margins():
    minimal = calculate_print_layout(AssembleurPrintSettings(margin_mm=3))
    standard = calculate_print_layout(AssembleurPrintSettings(margin_mm=10))
    portrait = minimal
    landscape = calculate_print_layout(AssembleurPrintSettings(orientation="landscape", margin_mm=10))
    assert portrait.page_height_mm > portrait.page_width_mm
    assert landscape.page_width_mm > landscape.page_height_mm
    assert landscape.map_rect_mm[2] == landscape.page_width_mm - 20
    minimal_gap = minimal.map_rect_mm[1] - (minimal.title_rect_mm[1] + minimal.title_rect_mm[3])
    standard_gap = standard.map_rect_mm[1] - (standard.title_rect_mm[1] + standard.title_rect_mm[3])
    assert minimal_gap == pytest.approx(.9)
    assert standard_gap == pytest.approx(3)
    assert minimal.map_rect_mm[3] > standard.map_rect_mm[3]


def test_title_is_rendered_and_long_title_stays_inside_its_shared_rect():
    settings = AssembleurPrintSettings(title="Titre très long " * 12)
    page = render_print_page_raster(AssembleurPrintSnapshot("", None, (), ()), settings, AssembleurPrintViewport(0, 0, 1, 1), 96)
    layout = calculate_print_layout(settings)
    factor = 96 / 25.4
    x, y, width, height = (round(value * factor) for value in layout.title_rect_mm)
    title_pixels = page.crop((x, y, x + width, y + height))
    assert any(pixel != (255, 255, 255) for pixel in title_pixels.getdata())


def test_title_nominal_size_uses_a_document_scale_in_physical_mm():
    assert DEFAULT_TITLE_FONT_MM == 5.0
    assert MIN_TITLE_FONT_MM == 2.5
    assert round(DEFAULT_TITLE_FONT_MM * (96 / 25.4)) == 19


def test_map_opacity_and_outside_map_are_composed_on_white():
    snapshot = AssembleurPrintSnapshot("", AssembleurPrintMap(Image.new("RGB", (10, 10), "black"), 0, 0, 10, 10), (), ())
    viewport = AssembleurPrintViewport(-10, 0, 20, 10)
    half = render_print_raster(snapshot, AssembleurPrintSettings(map_opacity=50, selected_layers=("map",)), viewport, 200, 100)
    assert half.getpixel((0, 50)) == (255, 255, 255)
    assert all(120 <= channel <= 135 for channel in half.getpixel((150, 50)))
    off = render_print_raster(snapshot, AssembleurPrintSettings(map_opacity=100, selected_layers=()), viewport, 200, 100)
    assert off.getpixel((150, 50)) == (255, 255, 255)


def test_assembly_and_beacons_follow_world_y_inversion_and_layer_selection():
    snapshot = _snapshot(with_map=False)
    viewport = AssembleurPrintViewport(0, 0, 20, 20)
    assembly = render_print_raster(snapshot, AssembleurPrintSettings(selected_layers=("assembly",)), viewport, 200, 200)
    assert assembly.getpixel((20, 180)) != (255, 255, 255)
    no_assembly = render_print_raster(snapshot, AssembleurPrintSettings(selected_layers=()), viewport, 200, 200)
    assert no_assembly.getpixel((20, 180)) == (255, 255, 255)
    beacons = render_print_raster(snapshot, AssembleurPrintSettings(selected_layers=("beacons",)), viewport, 200, 200)
    assert beacons.getpixel((150, 50)) == (0, 0, 0)
