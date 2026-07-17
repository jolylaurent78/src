from types import SimpleNamespace

from src.assembleur_core import ScenarioAssemblage, TopologyElement, TopologyNodeType
from src.assembleur_tk import (
    TriangleViewerManual,
    build_last_drawn_index,
    get_projected_elements,
)


class _StatusStub:
    def config(self, **_kwargs):
        pass


def _make_viewer_with_one_core_group():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._last_drawn = [{
        "id": 1,
        "labels": ("O", "B", "L"),
        "pts": {"O": (0.0, 0.0), "B": (3.0, 0.0), "L": (0.0, 4.0)},
        "topoElementId": "T01",
        "topoGroupId": "G1",
        "group_id": 10,
    }]
    viewer._last_drawn_topo_index = {}
    viewer._last_drawn_topo_index_source = None
    viewer._last_drawn_topo_index_length = -1
    viewer.groups = {10: {"id": 10, "topoGroupId": "G1", "nodes": [{"tid": 0}]}}
    viewer.active_scenario_index = 0

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
    # The Core controls the canonical id; the UI metadata mirrors it for this test.
    viewer._last_drawn[0]["topoGroupId"] = core_group_id
    viewer.groups[10]["topoGroupId"] = core_group_id
    viewer.scenarios = [scenario]
    return viewer, core_group_id


def test_group_fields_keep_only_group_identifier_cache():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    triangle = {}

    viewer._ensure_group_fields(triangle)

    assert len(triangle) == 1
    assert triangle.get("group_id") is None


def test_core_to_last_drawn_helpers_use_active_index_and_core_group():
    viewer, core_group_id = _make_viewer_with_one_core_group()

    entry = viewer.get_last_drawn_entry_by_topo_id("T01")

    assert entry is viewer._last_drawn[0]
    assert viewer.getTidForTopoElementId("T01") == 0
    assert viewer.get_last_drawn_entries_by_topo_ids(["unknown", "T01"]) == [entry]
    assert viewer.get_last_drawn_entries_for_core_group(core_group_id) == [entry]
    assert viewer.debug_validate_core_ui_group_linking() == []


def test_projected_elements_are_resolved_from_core_members_and_index():
    viewer, core_group_id = _make_viewer_with_one_core_group()
    group = viewer.scenarios[0].topoWorld.groups[core_group_id]

    index = build_last_drawn_index(viewer._last_drawn)

    assert get_projected_elements(group, index) == (viewer._last_drawn[0],)


def test_pose_sync_uses_core_members_without_ui_groups():
    viewer, core_group_id = _make_viewer_with_one_core_group()
    world = viewer.scenarios[0].topoWorld
    viewer.groups = {}
    calls = []
    original_set_pose = world.setElementPose

    def _record_pose(element_id, **kwargs):
        calls.append((element_id, kwargs["mirrored"]))
        return original_set_pose(element_id, **kwargs)

    world.setElementPose = _record_pose
    viewer._sync_group_elements_pose_to_core(core_group_id)

    assert calls == [("T01", False)]


def test_passive_group_geometry_uses_core_members_not_legacy_nodes():
    viewer, core_group_id = _make_viewer_with_one_core_group()
    world = viewer.scenarios[0].topoWorld
    other_group_id = world.add_element_as_new_group(TopologyElement(
        element_id="T02",
        name="Triangle 02",
        vertex_labels=["O", "B", "L"],
        vertex_types=[TopologyNodeType.OUVERTURE, TopologyNodeType.BASE, TopologyNodeType.LUMIERE],
        edge_lengths_km=[3.0, 5.0, 4.0],
    ))
    canonical_group_id = world.union_groups(core_group_id, other_group_id)
    viewer._last_drawn.append({
        "id": 2,
        "labels": ("O", "B", "L"),
        "pts": {"O": (10.0, 0.0), "B": (13.0, 0.0), "L": (10.0, 4.0)},
        "topoElementId": "T02",
        "topoGroupId": canonical_group_id,
        "group_id": 10,
    })
    viewer._last_drawn[0]["topoGroupId"] = canonical_group_id
    viewer.groups[10]["topoGroupId"] = canonical_group_id
    # La chaine historique reste volontairement incomplete : elle ne contient
    # que T01. Les lecteurs passifs doivent tout de meme voir T01 + T02.
    viewer._invalidate_last_drawn_topo_index()

    assert tuple(viewer._group_centroid(canonical_group_id)) == (6.0, 4.0 / 3.0)

    # T02 porte volontairement un autre group_id UI : l'exclusion doit
    # neanmoins suivre son groupe Core commun avec T01.
    viewer._last_drawn[1]["group_id"] = 999
    assert viewer._find_nearest_vertex((10.0, 0.0), exclude_idx=0, exclude_gid=10) is None


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
    viewer.groups = {}
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
    viewer._last_drawn = [{"topoElementId": "T01"}]
    viewer.scenarios = [SimpleNamespace(topoWorld=world)]
    viewer.active_scenario_index = 0
    viewer._screen_to_world = lambda x, y: (x, y)
    # Pas de self.groups : le test garantit l'absence de dependance UI.

    viewer._ctx_capture_chemin_context(0, 4.0, 0.0)

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
    viewer._last_drawn_topo_index = {}
    viewer._last_drawn_topo_index_source = None
    viewer._last_drawn_topo_index_length = -1
    viewer.scenarios = [SimpleNamespace(topoWorld=_World())]
    viewer.active_scenario_index = 0

    assert viewer._get_projected_elements_for_core_group("G-CORE") == tuple(viewer._last_drawn)


def test_topo_index_can_be_invalidated_after_in_place_relink():
    viewer, _core_group_id = _make_viewer_with_one_core_group()
    assert viewer.get_last_drawn_entry_by_topo_id("T01") is viewer._last_drawn[0]

    viewer._last_drawn[0]["topoElementId"] = "T02"
    viewer._invalidate_last_drawn_topo_index()

    assert viewer.get_last_drawn_entry_by_topo_id("T01") is None
    assert viewer.get_last_drawn_entry_by_topo_id("T02") is viewer._last_drawn[0]


def test_move_members_are_resolved_from_core_group():
    viewer, core_group_id = _make_viewer_with_one_core_group()

    prepared = viewer._prepare_core_group_operation_members("MOVE", 0)

    assert prepared["core_group_id"] == core_group_id
    assert prepared["entries"] == [viewer._last_drawn[0]]

    # Le MOVE ne consulte pas le cache de groupes UI pour ses membres.
    viewer.groups = {}
    viewer._move_group_world(core_group_id, 2.0, -1.0, prepared["entries"])

    assert tuple(viewer._last_drawn[0]["pts"]["O"]) == (2.0, -1.0)
    assert tuple(viewer._last_drawn[0]["pts"]["B"]) == (5.0, -1.0)


def test_center_move_initializes_a_core_only_selection_state():
    viewer, core_group_id = _make_viewer_with_one_core_group()
    viewer.groups = {}
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
    assert viewer._sel["move_member_entries"] == [viewer._last_drawn[0]]
    assert "gid" not in viewer._sel
    assert "ui_group_id" not in viewer._sel

    viewer._clock_dragging = False
    viewer._bg_moving = False
    viewer._bg_resizing = False
    viewer._redraw_from = lambda _entries: None
    viewer._clock_apply_auto_ref_sync = lambda: None
    viewer._on_canvas_left_move(SimpleNamespace(x=3.0, y=2.0))

    assert tuple(viewer._last_drawn[0]["pts"]["O"]) == (2.0, -1.0)


def test_delete_group_keeps_remaining_ui_tid_mapping():
    class _World:
        def __init__(self):
            self.groups = {"G-REMOVED": object(), "G-KEEP": object(), "G-ALIAS": object()}
            self.removed_element_ids = None

        def removeElementsAndRebuild(self, element_ids):
            self.removed_element_ids = list(element_ids)
            del self.groups["G-REMOVED"]

        def find_group(self, group_id):
            return "G-KEEP" if group_id == "G-ALIAS" else group_id

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    world = _World()
    viewer._ctx_target_idx = 0
    viewer._last_drawn = [
        {"topoElementId": "T01", "group_id": 10},
        {"topoElementId": "T02", "group_id": 20},
    ]
    viewer.groups = {
        10: {"id": 10, "nodes": [{"tid": 0}]},
        20: {"id": 20, "nodes": [{"tid": 1}]},
    }
    viewer.scenarios = [SimpleNamespace(topoWorld=world)]
    viewer.active_scenario_index = 0
    viewer._reinsert_triangle_to_list = lambda _triangle: None
    viewer._reset_assist = lambda: None
    viewer._redraw_from = lambda _entries: None
    viewer.status = _StatusStub()
    viewer._ctx_delete_group()

    assert world.removed_element_ids == ["T01"]
    assert viewer.groups[20]["nodes"] == [{"tid": 0}]
    assert viewer._last_drawn[0]["group_id"] == 20


def test_rotate_and_flip_prepare_members_from_core_group():
    viewer, core_group_id = _make_viewer_with_one_core_group()
    viewer.status = _StatusStub()
    viewer._redraw_from = lambda _entries: None
    viewer.offset = (0.0, 0.0)
    viewer.zoom = 1.0
    viewer._ctx_last_rclick = (0.0, 0.0)

    viewer._ctx_target_idx = 0
    viewer._ctx_rotate_selected()

    assert viewer._sel["core_group_id"] == core_group_id
    assert viewer._sel["rotate_member_entries"] == [viewer._last_drawn[0]]

    viewer._ctx_target_idx = 0
    viewer._ctx_flip_selected()

    assert viewer._last_drawn[0]["mirrored"] is True


def test_f8_audit_exports_ok_report(tmp_path):
    viewer, _core_group_id = _make_viewer_with_one_core_group()
    viewer.exports_dir = str(tmp_path)
    viewer.excel_path = None

    path = viewer.export_mig_geo001_audit()

    report = (tmp_path / "audits" / path.split("\\")[-1]).read_text(encoding="utf-8")
    assert "Statut global : OK" in report
    assert "Triangles _last_drawn                    : 1" in report
    assert "RÉSULTAT FINAL" in report


def test_f8_audit_detects_duplicate_or_missing_core_and_missing_projection():
    viewer, core_group_id = _make_viewer_with_one_core_group()
    world = viewer.scenarios[0].topoWorld
    live_group_calls = []
    original_get_live_group_ids = world.getLiveGroupIds
    world.getLiveGroupIds = lambda: (
        live_group_calls.append(True) or original_get_live_group_ids()
    )
    world.add_element_as_new_group(TopologyElement(
        element_id="T02",
        name="Triangle 02",
        vertex_labels=["O", "B", "L"],
        vertex_types=[TopologyNodeType.OUVERTURE, TopologyNodeType.BASE, TopologyNodeType.LUMIERE],
        edge_lengths_km=[3.0, 5.0, 4.0],
    ))
    viewer._last_drawn.append(dict(viewer._last_drawn[0]))
    viewer._last_drawn.append(dict(viewer._last_drawn[0]))
    viewer._last_drawn[-1]["topoElementId"] = "UNKNOWN"
    viewer._last_drawn[-1]["group_id"] = 10

    audit = viewer._collect_mig_geo001_audit(viewer.scenarios[0], world)

    assert audit["status"] == "ERROR"
    assert audit["ui_orphans"][0]["topoElementId"] == "UNKNOWN"
    assert audit["duplicate_ids"] == {"T01": [0, 1]}
    assert any("UNKNOWN" in error for error in audit["errors"])
    assert "T02" in audit["core_missing_ui_ids"]
    assert core_group_id in {row["core_group_id"] for row in audit["group_rows"]}
    assert live_group_calls == [True]


def test_f8_audit_marks_ui_group_composition_mismatch_as_error():
    viewer, core_group_id = _make_viewer_with_one_core_group()
    world = viewer.scenarios[0].topoWorld
    element = TopologyElement(
        element_id="T02",
        name="Triangle 02",
        vertex_labels=["O", "B", "L"],
        vertex_types=[TopologyNodeType.OUVERTURE, TopologyNodeType.BASE, TopologyNodeType.LUMIERE],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )
    second_group_id = world.add_element_as_new_group(element)
    canonical_group_id = world.union_groups(core_group_id, second_group_id)
    viewer._last_drawn.append({
        "id": 2,
        "labels": ("O", "B", "L"),
        "pts": {"O": (10.0, 0.0), "B": (13.0, 0.0), "L": (10.0, 4.0)},
        "topoElementId": "T02",
        "topoGroupId": canonical_group_id,
        "group_id": 20,
    })
    viewer._last_drawn[0]["topoGroupId"] = canonical_group_id
    viewer.groups[10]["topoGroupId"] = canonical_group_id
    viewer.groups[20] = {"id": 20, "topoGroupId": canonical_group_id, "nodes": [{"tid": 1}]}

    audit = viewer._collect_mig_geo001_audit(viewer.scenarios[0], world)

    assert audit["status"] == "ERROR"
    assert any(row["result"] == "INCOHÉRENT" for row in audit["group_rows"])


def test_f8_audit_without_active_scenario_is_safe():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = []
    viewer.active_scenario_index = 0

    assert viewer.export_mig_geo001_audit() is None
