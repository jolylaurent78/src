import numpy as np

from src.assembleur_compass_state import CompassState
from src.assembleur_tk import TriangleViewerManual


class _Canvas:
    def __init__(self):
        self.cursor = None

    def configure(self, **kwargs):
        self.cursor = kwargs.get("cursor")


class _Status:
    def __init__(self):
        self.text = ""

    def config(self, *, text):
        self.text = text


class _BackgroundMap:
    is_resizing = False
    is_moving = False


class _Viewer:
    def __init__(self):
        self.compass_state = CompassState(
            cx=120.0,
            cy=80.0,
            anchor_binding={"nodeId": "A", "topoGroupId": "G"},
            anchor_world=np.array((1.0, 2.0)),
        )
        self.compass_state.dragging = True
        self.compass_state.snap_target = {"nodeId": "B", "topoGroupId": "G", "world": np.array((9.0, 9.0))}
        self.compass_state.arc.last = {"angle": 42.0}
        self.compass_state.arc.last_angle_deg = 42.0
        self.canvas = _Canvas()
        self.background_map_layer = _BackgroundMap()
        self.status = _Status()
        self.bind_calls = self.auto_arc_calls = 0
        self.redraw_calls = self.menu_calls = 0
        self._drag = self._sel = None
        self._pan_anchor = None

    def _screen_to_world(self, x, y):
        return x / 10.0, y / 10.0

    def _clock_clear_anchor_binding(self):
        self.compass_state.anchor_binding = None

    def _clock_clear_snap_target(self):
        self.compass_state.snap_target = None

    def _clock_arc_clear_last(self):
        self.compass_state.arc.last = None
        self.compass_state.arc.last_angle_deg = None

    def _redraw_overlay_only(self):
        self.redraw_calls += 1

    def _update_compass_ctx_menu_and_dico_state(self):
        self.menu_calls += 1


def test_escape_during_compass_drag_keeps_current_free_position():
    viewer = _Viewer()

    assert TriangleViewerManual._on_escape_key(viewer, None) == "break"

    assert not viewer.compass_state.dragging
    assert viewer.compass_state.anchor_binding is None
    assert np.array_equal(viewer.compass_state.anchor_world, np.array((12.0, 8.0)))
    assert viewer.compass_state.snap_target is None
    assert viewer.compass_state.cx == 120.0 and viewer.compass_state.cy == 80.0
    assert viewer.compass_state.arc.last is None
    assert viewer.canvas.cursor == ""
    assert viewer.redraw_calls == 1 and viewer.menu_calls == 1

    TriangleViewerManual._on_canvas_left_up(viewer, None)
    assert viewer.bind_calls == 0 and viewer.auto_arc_calls == 0
    assert np.array_equal(viewer.compass_state.anchor_world, np.array((12.0, 8.0)))
