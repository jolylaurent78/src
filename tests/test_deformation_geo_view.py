import inspect
from types import SimpleNamespace

import pytest

from src.assembleur_deformation_window import (
    DEFORMATION_MAXIMUM_ZOOM,
    derive_assembly_view_rotation_deg,
)
from src.assembleur_geo_map_view import GeoMapView
from src.assembleur_geo_map_view import GeoMapPixelMarker


def _rotated_view(rotation_deg: float) -> GeoMapView:
    view = object.__new__(GeoMapView)
    view.canvas = SimpleNamespace(
        winfo_width=lambda: 800,
        winfo_height=lambda: 600,
    )
    view._view_rotation_deg = rotation_deg
    view._view_scale = 0.25
    view._offset_x = 120.0
    view._offset_y = -35.0
    return view


@pytest.mark.parametrize("rotation_deg", (0.0, 37.5, 90.0, 181.0, 270.0))
def test_rotated_geo_view_map_screen_round_trip(rotation_deg):
    view = _rotated_view(rotation_deg)
    for point in ((0.0, 0.0), (125.5, 87.25), (999.0, -120.0)):
        screen = view._map_to_screen(*point)
        assert view._screen_to_map(*screen) == pytest.approx(point)


def test_rotated_geo_view_screen_to_lambert_uses_inverse_view_rotation():
    view = _rotated_view(63.0)
    view.map = SimpleNamespace(pixel_to_lambert=lambda x, y: (x * 10.0, y * 10.0))
    point = (42.0, 81.0)
    assert view.screen_to_lambert(*view._map_to_screen(*point)) == pytest.approx((420.0, 810.0))


def test_assembly_view_rotation_aligns_ob_vectors():
    rotation = derive_assembly_view_rotation_deg(
        (0.0, 0.0),
        (10.0, 0.0),
        (100.0, 100.0),
        (100.0, 110.0),
    )
    assert rotation == pytest.approx(90.0)


def test_assembly_view_rotation_uses_current_overridden_base_position():
    original = derive_assembly_view_rotation_deg(
        (0.0, 0.0),
        (10.0, 0.0),
        (0.0, 0.0),
        (0.0, 10.0),
    )
    overridden = derive_assembly_view_rotation_deg(
        (0.0, 0.0),
        (0.0, 10.0),
        (0.0, 0.0),
        (0.0, 10.0),
    )
    assert original == pytest.approx(90.0)
    assert overridden == pytest.approx(0.0)


def _zoom_limited_view(maximum_zoom: float) -> GeoMapView:
    view = _rotated_view(0.0)
    view.map = SimpleNamespace(
        image_size=(100.0, 100.0),
        geographic_to_pixel=lambda latitude, longitude: (longitude, latitude),
    )
    view._maximum_zoom = maximum_zoom
    view._minimum_scale = lambda: 0.01
    view._constrain_view_offsets = lambda: None
    view._request_redraw = lambda: None
    view._hide_tooltip = lambda: None
    return view


def test_geo_map_view_maximum_zoom_keeps_half_as_the_constructor_default():
    assert inspect.signature(GeoMapView).parameters["maximum_zoom"].default == 0.5


@pytest.mark.parametrize("method_name", ("_zoom_at", "fit_to_bounds", "fit_to_view"))
def test_geo_map_view_uses_configured_maximum_zoom_for_all_zoom_paths(method_name):
    view = _zoom_limited_view(1.0)
    if method_name == "_zoom_at":
        view._zoom_at(400.0, 300.0, 100.0)
    elif method_name == "fit_to_bounds":
        view.fit_to_bounds([(0.0, 0.0), (1.0, 1.0)])
    else:
        view._fit_scale = 1.0
        view._initial_fit_zoom = 2.0
        view._initial_fit_applied = False
        view.fit_to_view()

    assert view._view_scale == pytest.approx(1.0)


def test_geo_map_view_default_maximum_zoom_remains_half_for_zoom_interaction():
    view = _zoom_limited_view(0.5)

    view._zoom_at(400.0, 300.0, 100.0)

    assert view._view_scale == pytest.approx(0.5)


def test_geo_map_view_replaces_calibration_without_refitting_when_requested():
    view = _rotated_view(0.0)
    view.map = SimpleNamespace(map_id="MAP-SYS-000001", image="old")
    view._source_image = "old"
    view._constrain_view_offsets = lambda: None
    redraws = []
    view._request_redraw = lambda: redraws.append(True)

    GeoMapView.set_map(
        view,
        SimpleNamespace(map_id="MAP-SYS-000001", image="new"),
        preserve_view=True,
    )

    assert view.map.image == "new"
    assert view._source_image == "new"
    assert (view._view_scale, view._offset_x, view._offset_y) == (0.25, 120.0, -35.0)
    assert redraws == [True]


def test_geo_map_view_fits_normally_when_replacing_a_map():
    view = _rotated_view(0.0)
    view._initial_fit_applied = True
    fit_calls = []
    view.fit_to_view = lambda: fit_calls.append(True)

    GeoMapView.set_map(view, SimpleNamespace(map_id="MAP-SYS-000002", image="new"))

    assert view._initial_fit_applied is False
    assert fit_calls == [True]


@pytest.mark.parametrize("scale", (0.5, 1.0, 3.0, 8.0))
def test_pixel_marker_coordinates_follow_the_map_at_every_zoom(scale):
    view = _rotated_view(37.5)
    view._view_scale = scale
    pixel = (1234.5, 678.25)

    assert view._screen_to_map(*view._map_to_screen(*pixel)) == pytest.approx(pixel)


def test_recenter_on_pixel_marker_uses_its_observed_position():
    view = _rotated_view(0.0)
    view.map = SimpleNamespace(image_size=(2000.0, 1000.0))
    view._markers = []
    view._pixel_markers = [GeoMapPixelMarker("CITY", 1234.5, 678.25)]
    view._constrain_view_offsets = lambda: None
    view._request_redraw = lambda: None

    view.recenter_on_marker("CITY")

    assert view._offset_x == pytest.approx(400.0 - 1234.5 * 0.25)
    assert view._offset_y == pytest.approx(300.0 - 678.25 * 0.25)


def test_deformation_window_allows_four_times_the_standard_maximum_zoom():
    standard = inspect.signature(GeoMapView).parameters["maximum_zoom"].default
    assert DEFORMATION_MAXIMUM_ZOOM == pytest.approx(standard * 4)
