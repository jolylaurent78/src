"""Etat runtime du compas, independant de Tk et du viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class CompassArcState:
    active: bool = False
    step: int = 0
    p1: tuple[int, int, float] | None = None
    p2: tuple[int, int, float] | None = None
    line1_id: int | None = None
    line2_id: int | None = None
    arc_id: int | None = None
    text_id: int | None = None
    last: dict | None = None
    last_angle_deg: float | None = None


@dataclass
class CompassMeasureState:
    active: bool = False
    line_id: int | None = None
    text_id: int | None = None
    last: tuple[int, int, float, float] | None = None


@dataclass
class CompassSetRefState:
    active: bool = False
    line_id: int | None = None
    text_id: int | None = None
    last: dict | None = None


@dataclass
class CompassTraceState:
    active: bool = False
    preview_az_abs: float | None = None
    preview_node_id: str | None = None
    preview_topo_group_id: str | None = None
    preview_delta_az: float | None = None
    line_id: int | None = None
    text_id: int | None = None


@dataclass
class CompassState:
    clock: dict = field(
        default_factory=lambda: {"hour": 5.0, "minute": 9, "label": "Trouver — (5h, 9')"}
    )
    cx: float | None = None
    cy: float | None = None
    rendered_radius: int = 69
    radius: int = 69
    ref_azimuth_deg: float = 0.0
    anchor_world: object | None = None
    anchor_binding: dict | None = None
    dragging: bool = False
    drag_dx: int = 0
    drag_dy: int = 0
    snap_target: dict | None = None
    auto_ref_sync_in_progress: bool = False
    set_ref: CompassSetRefState = field(default_factory=CompassSetRefState)
    measure: CompassMeasureState = field(default_factory=CompassMeasureState)
    trace: CompassTraceState = field(default_factory=CompassTraceState)
    arc: CompassArcState = field(default_factory=CompassArcState)


def clock_arc_compute_angle_deg(az1: float, az2: float) -> float:
    """Retourne le plus petit angle, dans [0, 180], entre deux azimuts."""
    delta = (float(az2) - float(az1)) % 360.0
    return float(360.0 - delta if delta > 180.0 else delta)


def clock_arc_compute_tk_arc(az1: float, az2: float) -> tuple[float, float, float, float]:
    """Retourne ``start``, ``extent``, angle et azimut milieu pour Tk."""
    first = float(az1) % 360.0
    second = float(az2) % 360.0
    clockwise = (second - first) % 360.0
    counterclockwise = (first - second) % 360.0
    if clockwise <= counterclockwise:
        angle = clockwise
        mid_az = (first + angle * 0.5) % 360.0
        return float((90.0 - first) % 360.0), float(-angle), float(angle), float(mid_az)
    angle = counterclockwise
    mid_az = (first - angle * 0.5) % 360.0
    return float((90.0 - first) % 360.0), float(angle), float(angle), float(mid_az)


def azimuth_world_deg(a, b) -> float:
    """Azimut absolu monde : Nord=0, Est=90, sens horaire."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return float(math.degrees(math.atan2(bx - ax, by - ay)) % 360.0)


def clock_angle_diff_deg(a: float, b: float) -> float:
    """Différence angulaire minimale dans [0, 180]."""
    return clock_arc_compute_angle_deg(a, b)


def clock_theoretical_ref_azimuth_deg(
    *, az1: float, az2: float, ang_hour_0: float, ang_min_0: float
) -> float:
    """Calcule l'azimut de référence alignant les aiguilles sur deux droites."""
    first, second = float(az1) % 360.0, float(az2) % 360.0
    hour, minute = float(ang_hour_0) % 360.0, float(ang_min_0) % 360.0
    ref1 = (first - hour) % 360.0
    ref2 = (second - hour) % 360.0
    error1 = clock_angle_diff_deg((ref1 + minute) % 360.0, second)
    error2 = clock_angle_diff_deg((ref2 + minute) % 360.0, first)
    return float(ref1 if error1 <= error2 else ref2)
