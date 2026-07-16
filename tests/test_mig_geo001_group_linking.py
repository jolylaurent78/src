from src.assembleur_core import ScenarioAssemblage, TopologyElement, TopologyNodeType
from src.assembleur_tk import TriangleViewerManual


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


def test_topo_index_can_be_invalidated_after_in_place_relink():
    viewer, _core_group_id = _make_viewer_with_one_core_group()
    assert viewer.get_last_drawn_entry_by_topo_id("T01") is viewer._last_drawn[0]

    viewer._last_drawn[0]["topoElementId"] = "T02"
    viewer._invalidate_last_drawn_topo_index()

    assert viewer.get_last_drawn_entry_by_topo_id("T01") is None
    assert viewer.get_last_drawn_entry_by_topo_id("T02") is viewer._last_drawn[0]


def test_move_members_are_resolved_from_core_group():
    viewer, core_group_id = _make_viewer_with_one_core_group()

    prepared = viewer._prepare_mig_geo_move_members(0, 10)

    assert prepared["source"] == "core"
    assert prepared["core_group_id"] == core_group_id
    assert prepared["entries"] == [viewer._last_drawn[0]]

    viewer._move_group_world(10, 2.0, -1.0, prepared["entries"])

    assert tuple(viewer._last_drawn[0]["pts"]["O"]) == (2.0, -1.0)
    assert tuple(viewer._last_drawn[0]["pts"]["B"]) == (5.0, -1.0)


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
