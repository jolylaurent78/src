from types import SimpleNamespace

import numpy as np
import pytest

from src.assembleur_core import ScenarioAssemblage, TopologyElement, TopologyNodeType
from src.assembleur_edgechoice import ManualAttachmentIntent, previewManualAttachment
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


class _StatusStub:
    def config(self, **_kwargs):
        pass


def _make_viewer_with_one_core_group():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._last_drawn = [{
        "pts": {"O": (0.0, 0.0), "B": (3.0, 0.0), "L": (0.0, 4.0)},
        "topoElementId": "T01",
    }]
    viewer.canvas_objects = CanvasObjectsCollection(viewer._last_drawn)
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.active_scenario_index = 0
    viewer._ctrl_down = False

    scenario = ScenarioAssemblage(name="test")
    scenario.last_drawn = viewer._last_drawn
    element = TopologyElement(
        element_id="T01",
        name="Triangle 01",
        vertex_labels=["O", "B", "L"],
        vertex_types=[TopologyNodeType.OUVERTURE, TopologyNodeType.BASE, TopologyNodeType.LUMIERE],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )
    core_group_id = scenario.topoWorld.add_element_as_new_group(element)
    viewer.scenarios = [scenario]
    return viewer, core_group_id


def _make_ctrl_move_group_viewer(*, include_second_member=True):
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    first = {
        "topoElementId": "T01",
        "pts": {"O": [0.0, 0.0], "B": [3.0, 0.0], "L": [0.0, 4.0]},
    }
    second = {
        "topoElementId": "T02",
        "pts": {"O": [5.0, 0.0], "B": [8.0, 0.0], "L": [5.0, 4.0]},
    }
    outsider = {
        "topoElementId": "T03",
        "pts": {"O": [10.0, 0.0], "B": [13.0, 0.0], "L": [10.0, 4.0]},
    }
    projected_entries = [first] + ([second] if include_second_member else []) + [outsider]
    viewer.canvas_objects = CanvasObjectsCollection(projected_entries)
    viewer._last_drawn = viewer.canvas_objects.entries

    scenario = ScenarioAssemblage(name="ctrl-group")
    first_group_id = scenario.topoWorld.add_element_as_new_group(TopologyElement(
        element_id="T01",
        name="Triangle 01",
        vertex_labels=["O", "B", "L"],
        vertex_types=[TopologyNodeType.OUVERTURE, TopologyNodeType.BASE, TopologyNodeType.LUMIERE],
        edge_lengths_km=[3.0, 5.0, 4.0],
    ))
    second_group_id = scenario.topoWorld.add_element_as_new_group(TopologyElement(
        element_id="T02",
        name="Triangle 02",
        vertex_labels=["O", "B", "L"],
        vertex_types=[TopologyNodeType.OUVERTURE, TopologyNodeType.BASE, TopologyNodeType.LUMIERE],
        edge_lengths_km=[3.0, 5.0, 4.0],
    ))
    scenario.topoWorld.add_element_as_new_group(TopologyElement(
        element_id="T03",
        name="Triangle 03",
        vertex_labels=["O", "B", "L"],
        vertex_types=[TopologyNodeType.OUVERTURE, TopologyNodeType.BASE, TopologyNodeType.LUMIERE],
        edge_lengths_km=[3.0, 5.0, 4.0],
    ))
    core_group_id = scenario.topoWorld.union_groups(first_group_id, second_group_id)
    viewer.scenarios = [scenario]
    viewer.active_scenario_index = 0
    viewer._sel = {
        "mode": "move_group",
        "core_group_id": core_group_id,
        "anchor": {"type": "vertex", "tid": 0, "vkey": "O"},
    }
    viewer._edge_choice = (0, "O", len(projected_entries) - 1, "O", (
        [0.0, 0.0], [3.0, 0.0], [10.0, 0.0], [10.0, 3.0],
    ))
    viewer._ctrl_down = False
    viewer._clock_measure_active = False
    viewer._clock_arc_active = False
    viewer._clock_setref_active = False
    viewer._clock_trace_active = False
    viewer._hide_tooltip = lambda: None

    class _Canvas:
        def configure(self, **_kwargs):
            pass

    viewer.canvas = _Canvas()
    viewer._ang_of_vec = lambda x, y: float(np.arctan2(y, x))
    return viewer, first, second, outsider, core_group_id


def _make_attachment_preview_viewer():
    def element(element_id, light_xy):
        return TopologyElement(
            element_id=element_id,
            name=element_id,
            vertex_labels=["O", "B", "L"],
            vertex_types=[
                TopologyNodeType.OUVERTURE,
                TopologyNodeType.BASE,
                TopologyNodeType.LUMIERE,
            ],
            edge_lengths_km=[10.0, float(np.hypot(10.0 - light_xy[0], light_xy[1])), float(np.hypot(*light_xy))],
            vertex_local_xy={0: (0.0, 0.0), 1: (10.0, 0.0), 2: light_xy},
        )

    scenario = ScenarioAssemblage(name="attachment-preview")
    world = scenario.topoWorld
    mobile = element("T01", (3.0, 4.0))
    target = element("T02", (6.0, 8.0))
    core_group_id = world.add_element_as_new_group(mobile)
    world.add_element_as_new_group(target)
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
    entries = []
    for element_id in ("T01", "T02"):
        core_element = world.elements[element_id]
        entries.append({
            "topoElementId": element_id,
            "pts": {
                vertex: core_element.localToWorld(core_element.vertex_local_xy[index])
                for index, vertex in enumerate(("O", "B", "L"))
            },
        })
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.canvas_objects = CanvasObjectsCollection(entries)
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.scenarios = [scenario]
    viewer.active_scenario_index = 0
    viewer._sel = {"mode": "move_group", "core_group_id": core_group_id}
    viewer._invalidate_pick_cache = lambda: None
    viewer._edge_highlights = None
    return viewer, world, core_group_id


def test_attachment_v2_preview_projects_temporary_rotation_without_mutating_core():
    viewer, world, core_group_id = _make_attachment_preview_viewer()
    first, second = viewer._last_drawn
    intent = ManualAttachmentIntent("vertex-edge", "T01", "O", "OB", "T02", "B", "OB")
    preview = previewManualAttachment(world, intent)
    assert preview.accepted
    real_pose = world.getElementPose("T01")
    free_mobile_points = {
        vertex: np.array(point, copy=True) for vertex, point in first["pts"].items()
    }
    viewer._sel["anchor"] = {"type": "vertex", "tid": 0, "vkey": "O"}
    viewer._ctrl_down = False
    viewer._clock_measure_active = False
    viewer._clock_arc_active = False
    viewer._clock_setref_active = False
    viewer._clock_trace_active = False
    viewer._hide_tooltip = lambda: None
    viewer.canvas = SimpleNamespace(configure=lambda **_kwargs: None)
    viewer._update_group_drag_snap_assist = lambda *_args: (
        setattr(viewer, "_attachment_intent", intent),
        setattr(viewer, "_attachment_preview", preview),
    )
    redraw_calls = []
    viewer._redraw_from = lambda entries: redraw_calls.append(entries)

    viewer._on_ctrl_down()

    np.testing.assert_allclose(first["pts"]["O"], free_mobile_points["O"])
    np.testing.assert_allclose(world.getElementPose("T01")[0], real_pose[0])
    np.testing.assert_allclose(world.getElementPose("T01")[1], real_pose[1])
    assert world.getElementPose("T01")[2] is real_pose[2]
    target_before = viewer._get_core_triangle_world_points(world, "T02")
    for vertex in ("O", "B", "L"):
        np.testing.assert_allclose(second["pts"][vertex], target_before[vertex])
    mobile_edge = np.asarray(first["pts"]["B"]) - np.asarray(first["pts"]["O"])
    preview_mobile = viewer._get_core_triangle_world_points(preview.world, "T01")
    target_edge = np.asarray(preview_mobile["B"]) - np.asarray(preview_mobile["O"])
    cross_product = mobile_edge[0] * target_edge[1] - mobile_edge[1] * target_edge[0]
    assert np.isclose(cross_product, 0.0)
    assert viewer._sel["core_group_id"] == core_group_id
    assert viewer._ctrl_down is True
    assert redraw_calls == [viewer._last_drawn]


def test_attachment_v2_preview_rejects_incomplete_projection_without_partial_rotation():
    viewer, first, _second, outsider, _core_group_id = _make_ctrl_move_group_viewer(
        include_second_member=False,
    )
    world = viewer.scenarios[0].topoWorld
    first_before = {key: list(value) for key, value in first["pts"].items()}
    outsider_before = {key: list(value) for key, value in outsider["pts"].items()}
    viewer._invalidate_pick_cache = lambda: None
    viewer._attachment_intent = SimpleNamespace(mob_element_id="T01", mob_edge="OB")

    with pytest.raises(KeyError, match="T02"):
        viewer._preview_attachment_rotation_to_last_drawn(
            SimpleNamespace(accepted=True, world=world.clonePhysicalState())
        )

    assert first["pts"] == first_before
    assert outsider["pts"] == outsider_before


def test_attachment_v2_preview_requires_core_group_id_without_legacy_fallback():
    viewer, first, _second, outsider, _core_group_id = _make_ctrl_move_group_viewer()
    viewer._sel.pop("core_group_id")
    first_before = {key: list(value) for key, value in first["pts"].items()}
    outsider_before = {key: list(value) for key, value in outsider["pts"].items()}

    with pytest.raises(RuntimeError, match="core_group_id absent"):
        viewer._preview_attachment_rotation_to_last_drawn(
            SimpleNamespace(accepted=True, world=viewer.scenarios[0].topoWorld.clonePhysicalState())
        )

    assert first["pts"] == first_before
    assert outsider["pts"] == outsider_before


def test_canvas_objects_lookups_use_active_index_and_core_group_projection():
    viewer, core_group_id = _make_viewer_with_one_core_group()

    entry = viewer.canvas_objects.get_by_topology_id("T01")

    assert entry is viewer._last_drawn[0]
    assert viewer.canvas_objects.get_index_by_topology_id("T01") == 0
    assert viewer.canvas_objects.get_many_by_topology_ids(["unknown", "T01"]) == (entry,)
    assert viewer.get_last_drawn_entries_for_core_group(core_group_id) == [entry]


def test_projected_elements_are_resolved_from_core_members_and_collection_index():
    viewer, core_group_id = _make_viewer_with_one_core_group()
    group = viewer.scenarios[0].topoWorld.groups[core_group_id]

    projected = tuple(
        entry
        for entry in (
            viewer.canvas_objects.get_by_topology_id(element_id)
            for element_id in group.element_ids
        )
        if entry is not None
    )

    assert projected == (viewer._last_drawn[0],)


def test_draw_group_outlines_enumerates_canonical_core_groups_without_ui_groups():
    class _Canvas:
        def __init__(self):
            self.deleted_tags = []
            self.lines = []

        def delete(self, tag):
            self.deleted_tags.append(tag)

        def create_line(self, *args, **kwargs):
            self.lines.append((args, kwargs))

    viewer, core_group_id = _make_viewer_with_one_core_group()
    viewer.canvas = _Canvas()
    viewer._world_to_screen = lambda point: point
    requested_core_group_ids = []
    world = viewer.scenarios[0].topoWorld
    live_group_calls = []
    original_get_live_group_ids = world.getLiveGroupIds
    world.getLiveGroupIds = lambda: (
        live_group_calls.append(True) or original_get_live_group_ids()
    )
    original_get_boundary_segments = world.getBoundarySegments
    world.getBoundarySegments = lambda group_id: (
        requested_core_group_ids.append(group_id) or original_get_boundary_segments(group_id)
    )
    viewer._project_boundary_segments = lambda group_id, _segments: [((0.0, 0.0), (1.0, 0.0))]

    viewer._draw_group_outlines()

    assert requested_core_group_ids == [core_group_id]
    assert live_group_calls == [True]
    assert viewer.canvas.deleted_tags == ["group_outline"]
    assert len(viewer.canvas.lines) == 1


def test_chemin_context_resolves_core_group_without_ui_groups():
    class _BoundarySegment:
        fromNodeId = "T01:N0"
        toNodeId = "T01:N1"

    class _World:
        elements = {"T01": object()}

        def get_group_of_element(self, element_id):
            assert element_id == "T01"
            return "G-CORE"

        def getBoundarySegments(self, group_id):
            assert group_id == "G-CORE"
            return [_BoundarySegment()]

        def getConceptNodeWorldXY(self, node_id, group_id):
            assert group_id == "G-CORE"
            return {"T01:N0": (0.0, 0.0), "T01:N1": (5.0, 0.0)}[node_id]

        def find_node(self, node_id):
            return "CANON:" + node_id

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    world = _World()
    viewer.scenarios = [SimpleNamespace(topoWorld=world)]
    viewer.active_scenario_index = 0
    viewer._screen_to_world = lambda x, y: (x, y)

    viewer._ctx_capture_chemin_context("T01", 4.0, 0.0)

    assert viewer.ctxGroupId == "G-CORE"
    assert viewer.ctxStartNodeId == "CANON:T01:N1"


def test_triangle_core_group_id_uses_only_public_element_lookup():
    class _World:
        def get_group_of_element(self, element_id):
            assert element_id == "T01"
            return "G-CORE"

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._last_drawn = [{"topoElementId": "T01"}]
    viewer.scenarios = [SimpleNamespace(topoWorld=_World())]
    viewer.active_scenario_index = 0

    assert viewer._get_core_group_id_for_triangle_index(0) == "G-CORE"


def test_projected_core_members_use_public_element_ids_api_only():
    class _World:
        def getGroupElementIds(self, core_group_id):
            assert core_group_id == "G-CORE"
            return ["T01", "T02"]

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._last_drawn = [
        {"topoElementId": "T01"},
        {"topoElementId": "T02"},
    ]
    viewer.canvas_objects = CanvasObjectsCollection(viewer._last_drawn)
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer.scenarios = [SimpleNamespace(topoWorld=_World())]
    viewer.active_scenario_index = 0

    assert viewer._get_projected_elements_for_core_group("G-CORE") == tuple(viewer._last_drawn)


def test_topology_lookup_tracks_in_place_relink():
    viewer, _core_group_id = _make_viewer_with_one_core_group()
    assert viewer.canvas_objects.get_by_topology_id("T01") is viewer._last_drawn[0]

    viewer._last_drawn[0]["topoElementId"] = "T02"

    assert viewer.canvas_objects.get_by_topology_id("T01") is None
    assert viewer.canvas_objects.get_by_topology_id("T02") is viewer._last_drawn[0]


def test_move_members_are_resolved_from_core_group():
    viewer, core_group_id = _make_viewer_with_one_core_group()

    prepared = viewer._prepare_core_group_operation_members("MOVE", 0)

    assert prepared["core_group_id"] == core_group_id
    assert prepared["entries"] == [viewer._last_drawn[0]]

    viewer._move_group_world(core_group_id, 2.0, -1.0, prepared["entries"])

    assert tuple(viewer._last_drawn[0]["pts"]["O"]) == (2.0, -1.0)
    assert tuple(viewer._last_drawn[0]["pts"]["B"]) == (5.0, -1.0)


def test_center_move_initializes_a_core_only_selection_state():
    viewer, core_group_id = _make_viewer_with_one_core_group()
    viewer.status = _StatusStub()
    viewer.offset = (0.0, 0.0)
    viewer.zoom = 1.0
    viewer._sel = None
    viewer._drag = None
    viewer._clock_arc_active = False
    viewer._clock_trace_active = False
    viewer._clock_measure_active = False
    viewer._clock_setref_active = False
    viewer._bg_calib_active = False
    viewer._bg = None
    viewer.bg_resize_mode = SimpleNamespace(get=lambda: False)
    viewer._ensure_pick_cache = lambda: None
    viewer._screen_to_world = lambda x, y: (x, -y)
    viewer._is_in_clock = lambda _x, _y: False
    viewer._hide_tooltip = lambda: None
    viewer._reset_assist = lambda: None
    viewer._hit_test = lambda _x, _y: ("center", 0, None)

    viewer._on_canvas_left_down(SimpleNamespace(x=1.0, y=1.0))

    assert viewer._sel["core_group_id"] == core_group_id
    assert "move_member_entries" not in viewer._sel
    assert "orig_group_pts" not in viewer._sel
    assert "mouse_world_prev" not in viewer._sel
    assert tuple(viewer._sel["mouse_world_start"]) == (1.0, -1.0)
    assert set(viewer._sel["move_preview_initial_pts"]) == {"T01"}
    assert "gid" not in viewer._sel
    assert "ui_group_id" not in viewer._sel

    viewer._clock_dragging = False
    viewer._bg_moving = False
    viewer._bg_resizing = False
    viewer._redraw_from = lambda _entries: None
    viewer._clock_apply_auto_ref_sync = lambda: None
    viewer._on_canvas_left_move(SimpleNamespace(x=3.0, y=2.0))

    assert tuple(viewer._last_drawn[0]["pts"]["O"]) == (2.0, -1.0)


def test_rotate_and_flip_prepare_members_from_core_group():
    viewer, core_group_id = _make_viewer_with_one_core_group()
    viewer.status = _StatusStub()
    viewer._redraw_from = lambda _entries: None
    viewer.offset = (0.0, 0.0)
    viewer.zoom = 1.0
    viewer._ctx_last_rclick = (0.0, 0.0)

    viewer._ctx_target_element_id = "T01"
    viewer._ctx_rotate_selected()

    assert viewer._sel["core_group_id"] == core_group_id
    assert "rotate_member_entries" not in viewer._sel
    assert "orig_group_pts" not in viewer._sel
    assert set(viewer._sel["rotate_preview_initial_pts"]) == {"T01"}

    viewer._ctx_target_element_id = "T01"
    viewer._ctx_flip_selected()

    assert viewer._get_core_element_mirrored("T01") is True


def test_nearest_line_forwards_the_core_group_id_to_snapping():
    """MIG-GROUP-019: le pipeline de snap ne reçoit plus de gid UI."""
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    calls = []
    viewer._find_nearest_vertex = lambda *args, **kwargs: calls.append((args, kwargs)) or None
    viewer._clear_nearest_line = lambda: None

    viewer._update_nearest_line(
        (1.0, 2.0),
        exclude_idx=4,
        exclude_core_group_id="G-CANONICAL",
    )

    assert calls == [
        (
            ((1.0, 2.0),),
            {"exclude_idx": 4, "exclude_core_group_id": "G-CANONICAL"},
        )
    ]
