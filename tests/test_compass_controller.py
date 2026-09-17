import math

import numpy as np
import pytest

from src.assembleur_compass_controller import CompassController
from src.assembleur_compass_state import CompassState


class FakeCanvas:
    def __init__(self, width=200, height=100):
        self.width = width
        self.height = height
        self.calls = []
        self._next_id = 1

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height

    def delete(self, *args):
        self.calls.append(("delete", args, {}))

    def _create(self, kind, *args, **kwargs):
        self.calls.append((kind, args, kwargs))
        item_id = self._next_id
        self._next_id += 1
        return item_id

    def create_line(self, *args, **kwargs):
        return self._create("line", *args, **kwargs)

    def create_oval(self, *args, **kwargs):
        return self._create("oval", *args, **kwargs)

    def create_arc(self, *args, **kwargs):
        return self._create("arc", *args, **kwargs)

    def create_text(self, *args, **kwargs):
        return self._create("text", *args, **kwargs)

    def coords(self, *args):
        self.calls.append(("coords", args, {}))

    def itemconfig(self, *args, **kwargs):
        self.calls.append(("itemconfig", args, kwargs))

    def tag_raise(self, *args):
        self.calls.append(("tag_raise", args, {}))

    def focus_set(self):
        self.calls.append(("focus_set", (), {}))


class FakeDecryptor:
    def getHoursBase(self):
        return 12

    def getMinutesBase(self):
        return 60

    def shouldShowHourHand(self):
        return True

    def shouldShowMinuteHand(self):
        return True

    def shouldShowHourLabels(self):
        return True

    def shouldShowHourTicks(self):
        return True

    def anglesFromClock(self, *, hour, minute):
        return hour * 30.0, minute * 6.0


class FakeTopoWorld:
    def getElementVertexFromAnyNodeId(self, node_id, groupId):
        assert (node_id, groupId) == ("N1", "G1")
        return {"elementId": "T1", "vkey": "O", "wbest": (3.0, 4.0)}

    def getConceptNodeWorldXY(self, node_id, group_id):
        return {("N1", "G1"): (30.0, 40.0), ("N2", "G1"): (50.0, 40.0)}[(node_id, group_id)]

    def findNearestBoundaryNode(self, unused, center_world):
        return {"nodeId": "N1", "groupId": "G1"}

    def getConceptNeighborNodes(self, node_id, group_id):
        assert (node_id, group_id) == ("N1", "G1")
        return ("N2",)

    def getNodeLabel(self, node_id):
        return f"Node {node_id}"

    def getGroupIdFromConceptNode(self, node_id):
        return "G1"

    def getBoundaryNeighbors(self, group_id, node_id):
        return "N1", "N2"


class FakeScenario:
    def __init__(self):
        self.topoWorld = FakeTopoWorld()
        self.clockAzimuthTraits = []
        self.clockRefEdgeId = "E1"
        self.clockRefNodeId = "N1"
        self.clockRefTopoGroupId = "G1"


def make_controller():
    state = CompassState()
    entries = [{"topoElementId": "T1", "pts": {"O": (10.0, 20.0)}}]
    scenario = FakeScenario()
    events = []
    ctrl_down = {"value": False}
    auto_ref_enabled = {"value": True}
    controller = CompassController(
        state,
        lambda point: (float(point[0]) + 1.0, float(point[1]) + 2.0),
        lambda x, y: (float(x) - 1.0, float(y) - 2.0),
        lambda: scenario,
        lambda: entries,
        lambda: FakeDecryptor(),
        lambda: True,
        lambda: False,
        lambda: ctrl_down["value"],
        lambda text: events.append(("status", text)),
        lambda element_id: 0 if element_id == "T1" else None,
        lambda: events.append(("clear_arc_filter",)),
        lambda: events.append(("auto_arc_filter",)),
        lambda: "#123456",
        lambda: events.append(("redraw_scene",)),
        lambda: events.append(("menu_refresh",)),
        lambda value: events.append(("persist_ref", value)),
        lambda: auto_ref_enabled["value"],
        lambda: "B1",
        lambda beacon_id: {"B1": (30.0, 60.0), "B2": (30.0, 80.0)}[beacon_id],
        lambda: [{"beaconId": "B2", "label": "Beacon 2"}],
        lambda: events.append(("refresh_preview",)),
    )
    controller._test_events = events
    controller._test_scenario = scenario
    controller._test_ctrl_down = ctrl_down
    controller._test_auto_ref_enabled = auto_ref_enabled
    return controller, state


def test_canvas_helpers_radius_hit_test_azimuth_and_clip():
    controller, state = make_controller()
    canvas = FakeCanvas()
    controller.attach_canvas(canvas)
    state.cx, state.cy, state.rendered_radius = 50.0, 50.0, 25

    assert controller.change_radius(-1000) == 50
    assert controller.contains_point(75, 50)
    assert not controller.contains_point(76, 50)
    assert controller.point_on_circle(90, 10) == (60.0, 50.0)
    assert controller.compute_azimuth_deg(50, 40) == 0.0
    assert controller.compute_azimuth_deg(60, 50) == 90.0
    assert controller.delta_display_text(270) == f"90,0{chr(176)}"
    endpoint = controller.clip_ray_to_viewport(50, 50, 90)
    assert endpoint is not None
    assert math.isclose(endpoint[0], 200.0) and math.isclose(endpoint[1], 50.0)


def test_snap_and_anchor_binding_follow_projected_vertex():
    controller, state = make_controller()
    canvas = FakeCanvas()
    controller.attach_canvas(canvas)
    state.snap_target = {"nodeId": "N1"}
    controller.clear_snap_target()
    assert state.snap_target is None
    assert ("delete", ("clock_snap_target",), {}) in canvas.calls

    controller.bind_anchor_from_snap_target({"nodeId": "N1", "topoGroupId": "G1"})
    assert state.anchor_binding == {"nodeId": "N1", "topoGroupId": "G1", "idx": 0, "vkey": "O"}
    assert np.array_equal(state.anchor_world, np.array((3.0, 4.0)))
    controller.refresh_anchor_world_from_binding()
    assert np.array_equal(state.anchor_world, np.array((10.0, 20.0)))
    assert np.array_equal(controller.get_center_world(), np.array((10.0, 20.0)))
    controller.clear_anchor_binding()
    assert state.anchor_binding is None


def test_draw_overlay_and_persisted_arc_without_tk():
    controller, state = make_controller()
    canvas = FakeCanvas()
    controller.attach_canvas(canvas)
    state.arc.last = {"az1": 0.0, "az2": 90.0, "angle": 90.0}

    controller.draw_overlay()

    assert state.cx == 81.0 and state.cy == 81.0
    assert state.rendered_radius == 69
    assert any(kind == "arc" for kind, _, _ in canvas.calls)
    assert any(kind == "text" and kwargs.get("text") == f"90{chr(176)}" for kind, _, kwargs in canvas.calls)


def test_unexpected_core_errors_are_not_swallowed():
    class BrokenTopoWorld:
        def getElementVertexFromAnyNodeId(self, node_id, groupId):
            raise KeyError("broken reference")

        def getConceptNodeWorldXY(self, node_id, group_id):
            raise KeyError("broken world")

    class BrokenScenario:
        topoWorld = BrokenTopoWorld()

    controller, state = make_controller()
    controller._active_scenario_provider = lambda: BrokenScenario()
    with pytest.raises(KeyError, match="broken reference"):
        controller.bind_anchor_to_node(node_id="N1", topo_group_id="G1")

    state.anchor_binding = {"nodeId": "N1", "topoGroupId": "G1", "idx": None, "vkey": None}
    with pytest.raises(KeyError, match="broken world"):
        controller.refresh_anchor_world_from_binding()


def test_measure_preview_confirm_cancel_and_ctrl_snap():
    controller, state = make_controller()
    canvas = FakeCanvas()
    controller.attach_canvas(canvas)
    state.cx, state.cy, state.ref_azimuth_deg = 50.0, 50.0, 30.0

    controller.start_measure(60, 50)
    assert state.measure.active
    assert state.measure.last is not None
    assert state.measure.last[:2] == (4, 6)
    assert state.measure.last[3] == pytest.approx((controller.compute_azimuth_deg(4, 6) - state.ref_azimuth_deg) % 360.0)
    controller.confirm_measure()
    assert not state.measure.active
    assert any(event[0] == "status" and "Azimut" in event[1] for event in controller._test_events)

    controller._test_ctrl_down["value"] = True
    controller.start_measure(60, 50)
    assert state.measure.last[:2] == (60, 50)
    controller.cancel_measure()
    assert state.measure.last is None
    assert not state.measure.active


def test_trace_requires_anchor_and_confirms_trait_with_current_color():
    controller, state = make_controller()
    canvas = FakeCanvas()
    controller.attach_canvas(canvas)
    state.cx, state.cy = 50.0, 50.0

    controller.start_trace(100, 50)
    assert not state.trace.active
    assert not controller._test_scenario.clockAzimuthTraits

    state.anchor_binding = {"nodeId": "N1", "topoGroupId": "G1", "idx": 0, "vkey": "O"}
    controller.start_trace(100, 50)
    assert state.trace.active
    assert state.trace.preview_delta_az == pytest.approx(90.0)
    controller.confirm_trace()
    assert not state.trace.active
    assert controller._test_scenario.clockAzimuthTraits == [{
        "nodeId": "N1", "topoGroupId": "G1", "deltaAzDeg": 90.0, "colorHex": "#123456",
    }]
    assert ("redraw_scene",) in controller._test_events
    controller.cancel_trace()
    assert state.trace.preview_az_abs is None


def test_set_ref_candidates_snap_ctrl_confirm_and_auto_ref():
    controller, state = make_controller()
    canvas = FakeCanvas()
    controller.attach_canvas(canvas)
    state.anchor_binding = None
    state.anchor_world = np.array((30.0, 40.0))
    state.cx, state.cy = 31.0, 42.0

    candidates = controller.collect_set_ref_candidates()
    assert [candidate["type"] for candidate in candidates] == ["node", "balise"]
    assert controller.pick_set_ref_snap_candidate(88.0, candidates)["nodeId"] == "N2"
    assert controller.pick_set_ref_snap_candidate(2.0, candidates)["beaconId"] == "B2"
    controller.start_set_ref(61, 42)
    assert state.set_ref.active
    assert state.set_ref.last["snapTarget"]["nodeId"] == "N2"
    controller.confirm_set_ref(61, 42)
    assert state.ref_azimuth_deg == 90.0
    assert ("persist_ref", 90.0) in controller._test_events
    assert controller._test_scenario.clockRefEdgeId is None
    assert controller._test_scenario.clockRefNodeId is None
    assert controller._test_scenario.clockRefTopoGroupId is None

    controller._test_ctrl_down["value"] = True
    controller.start_set_ref(31, 72)
    assert state.set_ref.last["snapTarget"] is None
    controller.cancel_set_ref()
    assert state.set_ref.last is None

    state.ref_azimuth_deg = 12.0
    controller._test_auto_ref_enabled["value"] = False
    controller.apply_auto_ref_sync()
    assert state.ref_azimuth_deg == 12.0
    controller._test_auto_ref_enabled["value"] = True
    controller.apply_auto_ref_sync()
    assert state.ref_azimuth_deg == pytest.approx(0.0)


def test_snap_and_manual_or_automatic_arc_lifecycle():
    controller, state = make_controller()
    canvas = FakeCanvas()
    controller.attach_canvas(canvas)
    state.cx, state.cy, state.rendered_radius = 50.0, 50.0, 30
    controller.update_snap_target(60, 50)
    assert state.snap_target["nodeDsu"] == "N1"
    assert any(kind == "oval" for kind, _, _ in canvas.calls)

    controller._test_ctrl_down["value"] = True
    state.cx, state.cy = 50.0, 50.0
    controller.start_arc()
    controller.handle_arc_click(50, 0)
    assert state.arc.step == 1
    controller.handle_arc_click(100, 50)
    assert controller.arc_is_available()
    assert state.arc.last_angle_deg == pytest.approx(90.0)
    controller.clear_arc_last()
    assert state.arc.last is None and state.arc.last_angle_deg is None

    target = {"nodeId": "N1", "nodeDsu": "N1", "topoGroupId": "G1"}
    controller._test_ctrl_down["value"] = False
    assert controller.auto_arc_from_snap_target(target, drag=True)
    assert ("auto_arc_filter",) not in controller._test_events
    assert controller.auto_arc_from_snap_target(target, drag=False)
    assert ("auto_arc_filter",) in controller._test_events
    with pytest.raises(RuntimeError, match="provided together"):
        controller.auto_arc_from_snap_target(target, drag=False, prev_node_dsu="N1")


def test_snap_projection_provider_is_resolved_late_bound():
    controller, state = make_controller()
    canvas = FakeCanvas()
    controller.attach_canvas(canvas)
    collections = {"current": {}}
    controller._projection_index_provider = lambda element_id: collections["current"].get(element_id)

    collections["current"] = {"T1": 7}
    controller.update_snap_target(60, 50)

    assert state.snap_target is not None
    assert state.snap_target["idx"] == 7
    assert collections["current"] == {"T1": 7}
    state.auto_ref_sync_in_progress = True
    controller.apply_auto_ref_sync()
    assert state.ref_azimuth_deg == pytest.approx(0.0)
