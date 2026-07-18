import ast
from pathlib import Path

import pytest
import pandas as pd

import numpy as np

from src.assembleur_core import TopologyElement, TopologyWorld
from src.assembleur_sim import createTopoQuadrilateral
from src.assembleur_tk import TriangleViewerManual


def _element(*, element_id=None, tri_rank=1):
    return TopologyElement(
        element_id=element_id,
        name=f"Triangle {tri_rank:02d}",
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[3.0, 5.0, 4.0],
        meta={"triRank": tri_rank},
    )


def test_world_allocates_monotone_element_ids_without_reuse_after_removal(tmp_path):
    world = TopologyWorld()
    first = _element(tri_rank=1)
    second = _element(tri_rank=2)

    world.add_element_as_new_group(first)
    world.add_element_as_new_group(second)
    assert (first.element_id, second.element_id) == ("T01", "T02")

    world.removeElementsAndRebuild(["T01", "T02"])
    third = _element(tri_rank=1)
    fourth = _element(tri_rank=2)
    world.add_element_as_new_group(third)
    world.add_element_as_new_group(fourth)

    assert (third.element_id, fourth.element_id) == ("T03", "T04")
    assert third.meta["triRank"] == 1
    assert fourth.meta["triRank"] == 2

    dump_path = world.export_topo_dump_xml(str(tmp_path / "TopoDump.xml"))
    dump = open(dump_path, encoding="utf-8").read()
    assert "T03" in dump and "T04" in dump
    assert "T03:N0" in dump and "T04:N0" in dump


def test_explicit_import_advances_element_counter_and_rejects_collisions():
    world = TopologyWorld()
    imported = _element(element_id="T07", tri_rank=1)
    world.add_element_as_new_group(imported)

    created = _element(tri_rank=2)
    world.add_element_as_new_group(created)
    assert created.element_id == "T08"

    with pytest.raises(ValueError, match="d.j. pr.sent"):
        world.add_element_as_new_group(_element(element_id="T07", tri_rank=7))


def test_non_ordered_explicit_imports_keep_highest_sequence_for_next_id():
    world = TopologyWorld()
    for element_id in ("T07", "T03", "T12"):
        world.add_element_as_new_group(_element(element_id=element_id))

    created = _element()
    world.add_element_as_new_group(created)
    assert created.element_id == "T13"


def test_clone_preserves_element_ids_and_advances_its_own_counter():
    world = TopologyWorld()
    for _ in range(3):
        world.add_element_as_new_group(_element())

    clone = world.clonePhysicalState()
    created_in_clone = _element()
    clone.add_element_as_new_group(created_in_clone)

    assert created_in_clone.element_id == "T04"
    assert sorted(world.elements) == ["T01", "T02", "T03"]
    assert sorted(clone.elements) == ["T01", "T02", "T03", "T04"]


def test_explicit_element_id_must_use_the_topology_instance_format():
    world = TopologyWorld()
    with pytest.raises(ValueError, match="elementId invalide"):
        world.add_element_as_new_group(_element(element_id="triangle-1"))


def test_simulation_uses_core_ids_instead_of_catalog_triangle_ids():
    world = TopologyWorld()
    first_entry, second_entry = {}, {}
    first_pts = {
        "O": np.array((0.0, 0.0)),
        "B": np.array((3.0, 0.0)),
        "L": np.array((0.0, 4.0)),
    }
    second_pts = {
        "O": np.array((0.0, 0.0)),
        "B": np.array((3.0, 0.0)),
        "L": np.array((3.0, -4.0)),
    }

    _group_id, first_id, second_id, _src_edge, _dst_edge = createTopoQuadrilateral(
        world=world,
        triangleMobFromId=7,
        triangleMobToId=12,
        triangleMobFrom={"labels": ("O", "B", "L"), "orient": "N"},
        triangleMobTo={"labels": ("O", "B", "L"), "orient": "N"},
        triangleMobFrom_PtsLocal=first_pts,
        triangleMobTo_PtsLocal=second_pts,
        triangleMobFromPts=first_pts,
        triangleMobToPts=second_pts,
        entryOdd=first_entry,
        entryEven=second_entry,
    )

    assert (first_id, second_id) == ("T01", "T02")
    assert (first_entry["topoElementId"], second_entry["topoElementId"]) == ("T01", "T02")
    assert world.elements["T01"].meta["triRank"] == 7
    assert world.elements["T02"].meta["triRank"] == 12


def test_manual_placement_keeps_catalog_id_separate_from_core_instance_id():
    class _Status:
        def config(self, **_kwargs):
            pass

    class _Listbox:
        def selection_clear(self, *_args):
            pass

    world = TopologyWorld()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._last_drawn = []
    viewer._last_drawn_topo_index = {}
    viewer._last_drawn_topo_index_source = None
    viewer._last_drawn_topo_index_length = -1
    viewer.groups = {}
    viewer._next_group_id = 1
    viewer._placed_ids = set()
    viewer._get_active_scenario = lambda: type("Scenario", (), {"topoWorld": world})()
    viewer._sync_group_elements_pose_to_core = lambda _core_group_id: None
    viewer._redraw_from = lambda _entries: None
    viewer._reset_assist = lambda: None
    viewer._reinsert_triangle_to_list = lambda _entry: None
    viewer._update_triangle_listbox_colors = lambda: None
    viewer.status = _Status()
    viewer.listbox = _Listbox()
    viewer.df = pd.DataFrame([
        {"len_OB": 3.0, "len_OL": 4.0, "len_BL": 5.0, "orient": "N", "B": "B1", "L": "L1"},
        {"len_OB": 3.0, "len_OL": 4.0, "len_BL": 5.0, "orient": "S", "B": "B2", "L": "L2"},
    ])

    def place(tri_id):
        viewer._drag = {
            "triangle": {"id": tri_id, "labels": ("O", "B", "L")},
            "world_pts": {"O": (0.0, 0.0), "B": (3.0, 0.0), "L": (0.0, 4.0)},
        }
        viewer._place_dragged_triangle()

    place(1)
    place(2)
    assert [entry["topoElementId"] for entry in viewer._last_drawn] == ["T01", "T02"]

    for _ in range(2):
        viewer._ctx_target_idx = 0
        viewer._ctx_delete_group()

    place(1)
    place(2)
    assert [entry["id"] for entry in viewer._last_drawn] == [1, 2]
    assert [entry["topoElementId"] for entry in viewer._last_drawn] == ["T03", "T04"]
    assert world.elements["T03"].name == "Triangle 01"
    assert world.elements["T04"].name == "Triangle 02"
    assert world.elements["T03"].meta["triRank"] == 1
    assert world.elements["T04"].meta["triRank"] == 2
    for vertex_key in ("O", "B", "L"):
        assert viewer._resolve_hover_vertex_node_id(
            entry=viewer._last_drawn[0],
            vertex_key=vertex_key,
            world=world,
        ) == world.get_element_vertex_node_id_by_type("T03", vertex_key)


def test_core_resolves_vertex_nodes_without_exposing_id_construction_to_ui():
    world = TopologyWorld()
    element = TopologyElement(
        element_id="T27",
        name="Triangle catalogue 01",
        vertex_labels=["B", "O", "L"],
        vertex_types=["B", "O", "L"],
        edge_lengths_km=[3.0, 5.0, 4.0],
        meta={"triRank": 1},
    )
    world.add_element_as_new_group(element)

    assert world.get_element_vertex_node_id("T27", 1) == element.vertexes[1].node_id
    assert world.get_element_vertex_node_id_by_type("T27", "O") == element.vertexes[1].node_id
    assert world.get_element_vertex_node_id_by_type("T27", "B") == element.vertexes[0].node_id
    assert world.get_element_edge_id("T27", 2) == element.edges[2].edge_id()

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    assert viewer._resolve_hover_vertex_node_id(
        entry={"id": 1, "topoElementId": "T27"},
        vertex_key="O",
        world=world,
    ) == element.vertexes[1].node_id
    assert viewer._resolve_hover_vertex_node_id(
        entry={"id": 1, "topoElementId": "T999"},
        vertex_key="O",
        world=world,
    ) is None

    with pytest.raises(KeyError, match="element inconnu"):
        world.get_element_vertex_node_id_by_type("T999", "O")


def test_non_core_modules_do_not_construct_or_parse_topology_ids():
    source_root = Path(__file__).resolve().parents[1] / "src"
    forbidden_calls = {
        "format_element_id",
        "format_node_id",
        "format_edge_id",
        "parse_tri_rank_from_element_id",
        "parse_element_sequence_from_id",
    }
    violations = []

    for source_path in source_root.glob("*.py"):
        if source_path.name == "assembleur_core.py":
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in forbidden_calls:
                    violations.append(f"{source_path.name}:{node.lineno}:{node.func.attr}")
            if isinstance(node, ast.JoinedStr):
                literals = "".join(
                    value.value for value in node.values
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                )
                if literals == "T" and any(
                    isinstance(value, ast.FormattedValue) for value in node.values
                ):
                    violations.append(f"{source_path.name}:{node.lineno}:f-string T")
                if ":N" in literals or ":E" in literals:
                    violations.append(f"{source_path.name}:{node.lineno}:feature-id f-string")

    assert violations == []
