import pytest

from src.assembleur_compass_state import (
    CompassState,
    azimuth_world_deg,
    clock_angle_diff_deg,
    clock_arc_compute_angle_deg,
    clock_arc_compute_tk_arc,
)


def test_compass_state_defaults_are_independent() -> None:
    first = CompassState()
    second = CompassState()

    assert first.clock == {"hour": 5.0, "minute": 9, "label": "Trouver — (5h, 9')"}
    assert first.radius == 69
    assert first.rendered_radius == 69
    assert first.arc.active is False
    first.arc.last = {"angle": 42}
    first.clock["hour"] = 8.0
    assert second.arc.last is None
    assert second.clock["hour"] == 5.0


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [(0, 0, 0), (0, 90, 90), (0, 180, 180), (350, 10, 20)],
)
def test_smallest_clock_arc_angle(first, second, expected) -> None:
    assert clock_arc_compute_angle_deg(first, second) == expected


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (0, 0, (90, 0, 0, 0)),
        (0, 90, (90, -90, 90, 45)),
        (0, 180, (90, -180, 180, 90)),
        (350, 10, (100, -20, 20, 0)),
    ],
)
def test_clock_arc_tk_parameters(first, second, expected) -> None:
    assert clock_arc_compute_tk_arc(first, second) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("target", "expected"),
    [((0, 1), 0), ((1, 0), 90), ((0, -1), 180), ((-1, 0), 270)],
)
def test_world_azimuth_cardinals(target, expected) -> None:
    assert azimuth_world_deg((0, 0), target) == expected


def test_clock_angle_difference_wraps() -> None:
    assert clock_angle_diff_deg(350, 10) == 20
