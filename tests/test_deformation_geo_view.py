from types import SimpleNamespace

import pytest

from src.assembleur_deformation_window import derive_assembly_view_rotation_deg
from src.assembleur_geo_map_view import GeoMapView


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
