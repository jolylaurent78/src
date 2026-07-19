import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

import numpy as np
import src.assembleur_sim as assembleur_sim

from src.assembleur_core import (
    ScenarioAssemblage,
    TopologyElement,
    TopologyWorld,
)
from src.assembleur_edgechoice import buildEdgeChoiceEptsForAutoChain
from src.assembleur_sim import (
    AlgoQuadrisParPaires,
    BranchState,
    MoteurSimulationAssemblage,
    PlacedTriangle,
    PlacedTriangles,
    buildLegacyLastDrawnFromTopology,
    _BranchNode,
    createTopoQuadrilateral,
)
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


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
    first_entry = PlacedTriangle(triangleId=7, points={})
    second_entry = PlacedTriangle(triangleId=12, points={})
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
    assert (first_entry.topologyElementId, second_entry.topologyElementId) == ("T01", "T02")
    assert not hasattr(first_entry, "topologyGroupId")
    assert not hasattr(second_entry, "topologyGroupId")
    assert world.elements["T01"].meta["triRank"] == 7
    assert world.elements["T02"].meta["triRank"] == 12


def test_auto_scenarios_keep_an_independent_ordered_element_history_per_branch():
    viewer = SimpleNamespace(df=pd.DataFrame([
        {
            "id": triangle_id,
            "len_OB": 3.0,
            "len_OL": 4.0,
            "len_BL": 5.0,
            "orient": "CCW",
            "B": f"B{triangle_id}",
            "L": f"L{triangle_id}",
        }
        for triangle_id in range(1, 5)
    ]))

    scenarios = AlgoQuadrisParPaires(MoteurSimulationAssemblage(viewer)).run([1, 2, 3, 4])

    assert len(scenarios) >= 2
    assert all(
        "_chain_edge_in" not in entry and "_chain_edge_out" not in entry
        for scenario in scenarios
        for entry in scenario.last_drawn
    )
    for scenario in scenarios:
        assert scenario.groups == {}
        assert all("group_id" not in entry for entry in scenario.last_drawn)
        assert len({entry["topoGroupId"] for entry in scenario.last_drawn}) == 1
        assert all(
            entry["topoGroupId"] == str(
                scenario.topoWorld.get_group_of_element(entry["topoElementId"])
            )
            for entry in scenario.last_drawn
        )
    for scenario in scenarios:
        assert scenario.orderedElementIds == [
            entry["topoElementId"] for entry in scenario.last_drawn
        ]
    assert scenarios[0].orderedElementIds is not scenarios[1].orderedElementIds

    scenarios[0].orderedElementIds.append("test-only")
    assert "test-only" not in scenarios[1].orderedElementIds


def test_auto_scenarios_reconstruct_connections_from_core_attachments():
    """Les scénarios automatiques restent construits depuis les attachments Core."""
    viewer = SimpleNamespace(df=pd.DataFrame([
        {
            "id": triangle_id,
            "len_OB": 3.0,
            "len_OL": 4.0,
            "len_BL": 5.0,
            "orient": "CCW",
            "B": f"B{triangle_id}",
            "L": f"L{triangle_id}",
        }
        for triangle_id in range(1, 5)
    ]))

    scenarios = AlgoQuadrisParPaires(MoteurSimulationAssemblage(viewer)).run([1, 2, 3, 4])

    assert len(scenarios) >= 2


def test_auto_two_triangle_scenario_exposes_legacy_projection_dicts():
    viewer = SimpleNamespace(df=pd.DataFrame([
        {"id": 1, "len_OB": 3.0, "len_OL": 4.0, "len_BL": 5.0, "orient": "CCW", "B": "B1", "L": "L1"},
        {"id": 2, "len_OB": 3.0, "len_OL": 4.0, "len_BL": 5.0, "orient": "CCW", "B": "B2", "L": "L2"},
    ]))

    scenarios = AlgoQuadrisParPaires(MoteurSimulationAssemblage(viewer)).run([1, 2])

    assert len(scenarios) == 1
    assert len(scenarios[0].last_drawn) == 2
    assert all("group_id" not in entry for entry in scenarios[0].last_drawn)
    assert scenarios[0].groups == {}
    for entry in scenarios[0].last_drawn:
        assert isinstance(entry, dict)
        assert {"labels", "pts", "id", "mirrored", "topoElementId", "topoGroupId"} <= entry.keys()
        assert set(entry["pts"]) == {"O", "B", "L"}


def test_simulator_exposes_no_legacy_group_projection_helpers():
    assert not hasattr(assembleur_sim, "buildLegacyGroupIdMappingFromTopology")
    assert not hasattr(assembleur_sim, "buildLegacyGroupsFromTopology")


def test_scenario_assembly_defaults_to_an_empty_construction_order():
    assert ScenarioAssemblage("manual").orderedElementIds == []


def test_branch_state_derives_tail_from_its_own_ordered_element_ids():
    original_order = ["T01", "T02"]
    state = BranchState(
        node=_BranchNode(parent=None, children=[]),
        topoWorld=TopologyWorld(),
        placedTriangles=PlacedTriangles(),
        orderedElementIds=original_order,
        poly_occ=set(),
    )

    original_order.append("T99")
    assert state.orderedElementIds == ["T01", "T02"]
    assert state.tailElementId == "T02"

    extended = BranchState(
        node=_BranchNode(parent=state.node, children=[]),
        topoWorld=TopologyWorld(),
        placedTriangles=PlacedTriangles(),
        orderedElementIds=[*state.orderedElementIds, "T03"],
        poly_occ=set(),
    )
    assert extended.tailElementId == "T03"


def test_placed_triangles_preserves_entry_format_and_clones_branch_data():
    first = PlacedTriangle(
        triangleId=1,
        topologyElementId="T01",
        points={"O": [0.0, 0.0]},
    )
    placed = PlacedTriangles([first])
    second = PlacedTriangle(
        triangleId=2,
        topologyElementId="T02",
        points={"O": [1.0, 0.0]},
    )
    placed.append(second)

    assert len(placed) == placed.count() == 2
    assert list(placed) == placed.items()
    assert placed[0] is first
    assert placed.last().triangleId == 2
    assert placed.findByTopologyElementId("T02") is second
    assert placed.findByTopologyElementId("T99") is None
    assert placed.findByTriangleId(1) is first
    assert placed.findByTriangleId(99) is None
    assert placed.toLegacyList() == [
        {
            "labels": None,
            "pts": {"O": [0.0, 0.0]},
            "id": 1,
            "mirrored": False,
            "topoElementId": "T01",
        },
        {
            "labels": None,
            "pts": {"O": [1.0, 0.0]},
            "id": 2,
            "mirrored": False,
            "topoElementId": "T02",
        },
    ]

    clone = placed.clone()
    clone[0].points["O"][0] = 99.0
    assert placed[0].points["O"][0] == 0.0


def test_placed_triangle_ignores_obsolete_chain_keys_on_import_and_export():
    triangle = PlacedTriangle.fromLegacyDict({
        "id": 1,
        "pts": {"O": [0.0, 0.0]},
        "_chain_edge_in": "LO",
        "_chain_edge_out": "BL",
    })

    assert triangle.toLegacyDict() == {
        "labels": None,
        "pts": {"O": [0.0, 0.0]},
        "id": 1,
        "mirrored": False,
    }


def test_placed_triangle_ignores_legacy_group_id_on_import_and_export():
    triangle = PlacedTriangle.fromLegacyDict({
        "id": 1,
        "pts": {"O": [0.0, 0.0]},
        "topoElementId": "T01",
        "group_id": 42,
    })

    assert not hasattr(triangle, "groupId")
    assert "group_id" not in triangle.toLegacyDict()

    world = TopologyWorld()
    world.add_element_as_new_group(_element(element_id="T01"))
    last_drawn = buildLegacyLastDrawnFromTopology(
        topologyWorld=world,
        orderedElementIds=["T01"],
        placedTriangles=PlacedTriangles([triangle]),
    )
    assert "group_id" not in last_drawn[0]
    assert last_drawn[0]["topoGroupId"] == str(world.get_group_of_element("T01"))


def test_placed_triangle_ignores_legacy_topology_group_id_on_import_and_export():
    triangle = PlacedTriangle.fromLegacyDict({
        "id": 1,
        "pts": {"O": [0.0, 0.0]},
        "topoGroupId": "G-LEGACY",
    })

    assert not hasattr(triangle, "topologyGroupId")
    assert "topoGroupId" not in triangle.toLegacyDict()


def test_last_drawn_topology_group_id_is_projected_from_current_core(monkeypatch):
    world = TopologyWorld()
    world.add_element_as_new_group(_element(element_id="T01"))
    placed = PlacedTriangles([
        PlacedTriangle(triangleId=1, topologyElementId="T01", points={}),
    ])

    monkeypatch.setattr(world, "get_group_of_element", lambda _element_id: "G-PROJECTED")
    last_drawn = buildLegacyLastDrawnFromTopology(
        topologyWorld=world,
        orderedElementIds=["T01"],
        placedTriangles=placed,
    )

    assert "group_id" not in last_drawn[0]
    assert last_drawn[0]["topoGroupId"] == "G-PROJECTED"


def test_auto_edgechoice_receives_explicit_projection_entries():
    world = TopologyWorld()
    world.add_element_as_new_group(_element(element_id="T01"))
    world.add_element_as_new_group(_element(element_id="T02"))
    mobile = PlacedTriangle(
        triangleId=1,
        topologyElementId="T01",
        points={"O": (0.0, 0.0), "B": (3.0, 0.0), "L": (0.0, 4.0)},
    )
    destination = PlacedTriangle(
        triangleId=2,
        topologyElementId="T02",
        points={"O": (10.0, 0.0), "B": (13.0, 0.0), "L": (10.0, 4.0)},
    )

    epts, meta = buildEdgeChoiceEptsForAutoChain(
        world=world,
        mobile_entry=mobile,
        destination_entry=destination,
        src_edge="LO",
        dst_edge="LO",
    )

    assert (epts.elementIdSrc, epts.elementIdDst) == ("T01", "T02")
    assert (meta["src_owner_tid"], meta["dst_owner_tid"]) == (1, 2)


@pytest.mark.parametrize(
    ("mobile_entry", "destination_entry", "role"),
    [
        (None, {"id": 2, "topoElementId": "T02", "pts": {}}, "mobile"),
        (PlacedTriangle(triangleId=1, topologyElementId="T01", points={}), {}, "destination"),
    ],
)
def test_auto_edgechoice_reports_invalid_projection_entry(mobile_entry, destination_entry, role):
    with pytest.raises(ValueError, match=role):
        buildEdgeChoiceEptsForAutoChain(
            world=TopologyWorld(),
            mobile_entry=mobile_entry,
            destination_entry=destination_entry,
            src_edge="LO",
            dst_edge="LO",
        )


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
    viewer.canvas_objects = CanvasObjectsCollection(viewer._last_drawn)
    viewer._last_drawn = viewer.canvas_objects.entries
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
    assert [entry["id"] for entry in viewer._last_drawn] == [1, 2]
    assert all("group_id" not in entry for entry in viewer._last_drawn)
    assert viewer.groups == {}
    for entry in viewer._last_drawn:
        core_group_id = entry["topoGroupId"]
        assert world.getGroupElementIds(core_group_id) == [entry["topoElementId"]]

    assert world.elements["T01"].name == "Triangle 01"
    assert world.elements["T02"].name == "Triangle 02"
    assert world.elements["T01"].meta["triRank"] == 1
    assert world.elements["T02"].meta["triRank"] == 2
    for vertex_key in ("O", "B", "L"):
        assert viewer._resolve_hover_vertex_node_id(
            entry=viewer._last_drawn[0],
            vertex_key=vertex_key,
            world=world,
        ) == world.get_element_vertex_node_id_by_type("T01", vertex_key)


def test_manual_placement_syncs_core_pose_without_creating_a_ui_group():
    class _Status:
        def config(self, **_kwargs):
            pass

    class _Listbox:
        def selection_clear(self, *_args):
            pass

    world = TopologyWorld()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.canvas_objects = CanvasObjectsCollection()
    viewer._last_drawn = viewer.canvas_objects.entries
    scenario = SimpleNamespace(topoWorld=world, last_drawn=viewer._last_drawn)
    viewer._get_active_scenario = lambda: scenario
    viewer._placed_ids = set()
    viewer._redraw_from = lambda _entries: None
    viewer._update_triangle_listbox_colors = lambda: None
    viewer.status = _Status()
    viewer.listbox = _Listbox()
    viewer.df = pd.DataFrame([
        {"len_OB": 3.0, "len_OL": 4.0, "len_BL": 5.0, "orient": "N", "B": "B1", "L": "L1"},
    ])
    world_pts = {"O": (10.0, 20.0), "B": (13.0, 20.0), "L": (10.0, 24.0)}
    viewer._drag = {
        "triangle": {"id": 1, "labels": ("O", "B", "L")},
        "world_pts": world_pts,
    }

    viewer._place_dragged_triangle()

    entry = viewer._last_drawn[0]
    assert {"topoElementId", "topoGroupId"} <= entry.keys()
    assert "group_id" not in entry
    element_id = entry["topoElementId"]
    assert world.getGroupElementIds(entry["topoGroupId"]) == [element_id]

    element = world.elements[element_id]
    for vertex_index, vertex_name in enumerate(("O", "B", "L")):
        actual = world.elementLocalToWorld(element_id, element.vertex_local_xy[vertex_index])
        np.testing.assert_allclose(actual, world_pts[vertex_name], atol=1e-9)


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
