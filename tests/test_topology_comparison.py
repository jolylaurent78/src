from src.assembleur_core import (
    ScenarioAssemblage,
    TopologyAttachment,
    TopologyFeatureRef,
    TopologyFeatureType,
)
from src.assembleur_topology_comparison import (
    build_attachment_signature,
    build_topology_prefix_steps,
    differing_attachment_element_ids,
)
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


class _CoreGroup:
    def __init__(self, element_ids):
        self.element_ids = list(element_ids)


class _VertexWorld:
    def __init__(self):
        self.attachments = {"A001": object()}
        self.groups = {"G001": _CoreGroup(["T01", "T02"])}

    def get_element_vertex_node_id_by_type(self, element_id, node_type):
        vertex_index = {"O": 0, "B": 1, "L": 2}[node_type]
        return f"{element_id}:N{vertex_index}"

    def find_node(self, node_id):
        return f"CN:{node_id}"

    def get_group_of_element(self, element_id):
        assert element_id in {"T01", "T02"}
        return "G001"

    def getGroupElementIds(self, core_group_id):
        assert core_group_id == "G001"
        return ["T01", "T02"]


def _attachment(kind, a_type, a_element, a_index, b_type, b_element, b_index, params=None):
    return TopologyAttachment(
        None,
        kind,
        TopologyFeatureRef(a_type, a_element, a_index),
        TopologyFeatureRef(b_type, b_element, b_index),
        params or {},
    )


def test_attachment_signatures_are_endpoint_order_independent_for_all_kinds():
    cases = [
        ("edge-edge", TopologyFeatureType.EDGE, "T01", 1, TopologyFeatureType.EDGE, "T02", 2, {"mapping": "reverse"}),
        ("vertex-edge", TopologyFeatureType.VERTEX, "T01", 0, TopologyFeatureType.EDGE, "T02", 1, {"t": 0.25, "edgeFrom": "T02:N1"}),
        ("vertex-vertex", TopologyFeatureType.VERTEX, "T01", 0, TopologyFeatureType.VERTEX, "T02", 2, {}),
    ]
    for kind, at, ae, ai, bt, be, bi, params in cases:
        direct = _attachment(kind, at, ae, ai, bt, be, bi, params)
        reverse = _attachment(kind, bt, be, bi, at, ae, ai, params)
        assert build_attachment_signature(direct) == build_attachment_signature(reverse)


def test_attachment_difference_returns_endpoint_elements():
    ref = ScenarioAssemblage("ref").topoWorld
    cur = ScenarioAssemblage("cur").topoWorld
    shared = _attachment("vertex-vertex", TopologyFeatureType.VERTEX, "T01", 0, TopologyFeatureType.VERTEX, "T02", 0)
    changed = _attachment("edge-edge", TopologyFeatureType.EDGE, "T01", 1, TopologyFeatureType.EDGE, "T03", 2, {"mapping": "direct"})
    ref.attachments = {"A001": shared}
    cur.attachments = {"A009": shared, "A010": changed}

    assert differing_attachment_element_ids(ref, cur) == {"T01", "T03"}


def test_same_scenario_has_no_core_comparison_difference():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    scenario = ScenarioAssemblage("auto", source_type="auto")
    scenario.last_drawn = [{"topoElementId": "T01"}]
    viewer.scenarios = [scenario]
    viewer.active_scenario_index = 0
    viewer.ref_scenario_token = id(scenario)
    viewer._comparison_diff_indices = {99}

    viewer._update_current_scenario_differences()

    assert viewer._comparison_diff_indices == set()


def test_core_comparison_marks_attachment_endpoint_triangles_for_all_kinds_and_params():
    def marked_indices(reference_attachment, current_attachment):
        viewer = TriangleViewerManual.__new__(TriangleViewerManual)
        reference = ScenarioAssemblage("reference", source_type="auto")
        current = ScenarioAssemblage("current", source_type="auto")
        reference.last_drawn = [
            {"id": 1, "topoElementId": "T01"},
            {"id": 2, "topoElementId": "T02"},
        ]
        current.last_drawn = [
            {"id": 1, "topoElementId": "T01"},
            {"id": 2, "topoElementId": "T02"},
        ]
        reference.topoWorld.attachments = {"A001": reference_attachment}
        current.topoWorld.attachments = {"A001": current_attachment}
        viewer.scenarios = [reference, current]
        viewer.active_scenario_index = 1
        viewer.ref_scenario_token = id(reference)

        viewer._update_current_scenario_differences()
        return viewer._comparison_diff_indices

    cases = [
        (
            _attachment("edge-edge", TopologyFeatureType.EDGE, "T01", 0,
                        TopologyFeatureType.EDGE, "T02", 1, {"mapping": "direct"}),
            _attachment("edge-edge", TopologyFeatureType.EDGE, "T01", 0,
                        TopologyFeatureType.EDGE, "T02", 1, {"mapping": "reverse"}),
        ),
        (
            _attachment("vertex-edge", TopologyFeatureType.VERTEX, "T01", 1,
                        TopologyFeatureType.EDGE, "T02", 2, {"edgeFrom": "T02:N1", "t": 0.25}),
            _attachment("vertex-edge", TopologyFeatureType.VERTEX, "T01", 1,
                        TopologyFeatureType.EDGE, "T02", 2, {"edgeFrom": "T02:N1", "t": 0.75}),
        ),
        (
            _attachment("vertex-vertex", TopologyFeatureType.VERTEX, "T01", 0,
                        TopologyFeatureType.VERTEX, "T02", 1),
            _attachment("vertex-vertex", TopologyFeatureType.VERTEX, "T01", 2,
                        TopologyFeatureType.VERTEX, "T02", 1),
        ),
    ]

    for reference_attachment, current_attachment in cases:
        assert marked_indices(reference_attachment, current_attachment) == {0, 1}


def test_prefix_step_edge_edge_is_oriented_by_the_traversal_order():
    world = ScenarioAssemblage("step").topoWorld
    world.attachments = {
        "A001": _attachment(
            "edge-edge", TopologyFeatureType.EDGE, "T02", 1,
            TopologyFeatureType.EDGE, "T01", 2, {"mapping": "reverse"},
        ),
    }

    steps = build_topology_prefix_steps(world, ["T01", "T02"], 1)

    assert steps == [(
        ("edge-edge", ("edge", "T01", 2), ("edge", "T02", 1), (("mapping", "reverse"),)),
    )]


def test_prefix_step_collects_all_attachments_between_two_triangles_independently_of_storage_order():
    world = ScenarioAssemblage("step").topoWorld
    vertex_vertex = _attachment(
        "vertex-vertex", TopologyFeatureType.VERTEX, "T03", 0,
        TopologyFeatureType.VERTEX, "T04", 1,
    )
    vertex_edge = _attachment(
        "vertex-edge", TopologyFeatureType.EDGE, "T04", 2,
        TopologyFeatureType.VERTEX, "T03", 2,
        {"t": 0.5, "edgeFrom": "T04:N2"},
    )
    irrelevant = _attachment(
        "edge-edge", TopologyFeatureType.EDGE, "T03", 1,
        TopologyFeatureType.EDGE, "T05", 1, {"mapping": "direct"},
    )
    world.attachments = {"A003": vertex_edge, "A001": irrelevant, "A002": vertex_vertex}

    first = build_topology_prefix_steps(world, ["T03", "T04"], 1)
    world.attachments = {"A002": vertex_vertex, "A003": vertex_edge, "A001": irrelevant}
    second = build_topology_prefix_steps(world, ["T03", "T04"], 1)

    assert first == second
    assert len(first[0]) == 2
    assert {signature[0] for signature in first[0]} == {"vertex-vertex", "vertex-edge"}


def test_prefix_steps_have_no_link_for_zero_or_one_triangle_and_none_when_missing():
    world = ScenarioAssemblage("step").topoWorld

    assert build_topology_prefix_steps(world, [], 0) == []
    assert build_topology_prefix_steps(world, ["T01"], 0) == []
    assert build_topology_prefix_steps(world, ["T01", "T02"], 1) is None


def test_viewer_core_prefix_steps_uses_ordered_element_ids_and_attachments_without_groups():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    scenario = ScenarioAssemblage("auto", source_type="auto", tri_ids=[1, 2])
    scenario.groups = None
    scenario.last_drawn = [
        {"id": 1, "topoElementId": "T01"},
        {"id": 2, "topoElementId": "T02"},
    ]
    scenario.orderedElementIds = ["T01", "T02"]
    scenario.topoWorld.elements.update({"T01": object(), "T02": object()})
    scenario.topoWorld.attachments = {
        "A001": _attachment(
            "edge-edge", TopologyFeatureType.EDGE, "T01", 0,
            TopologyFeatureType.EDGE, "T02", 1, {"mapping": "direct"},
        ),
    }

    assert viewer._scenario_prefix_edge_steps(scenario, 1) == [(
        ("edge-edge", ("edge", "T01", 0), ("edge", "T02", 1), (("mapping", "direct"),)),
    )]


def test_prefix_filter_keeps_only_candidates_with_same_ordered_core_path():
    def scenario(name, ordered_element_ids, attachment):
        item = ScenarioAssemblage(name, source_type="auto", tri_ids=[1, 2])
        item.groups = None
        item.last_drawn = [
            {"id": 1, "topoElementId": "T01"},
            {"id": 2, "topoElementId": "T02"},
        ]
        item.orderedElementIds = list(ordered_element_ids)
        item.topoWorld.elements.update({"T01": object(), "T02": object()})
        item.topoWorld.attachments = {"A001": attachment}
        return item

    shared = _attachment(
        "edge-edge", TopologyFeatureType.EDGE, "T01", 0,
        TopologyFeatureType.EDGE, "T02", 1, {"mapping": "direct"},
    )
    active = scenario("active", ["T01", "T02"], shared)
    same_path = scenario("same", ["T01", "T02"], shared)
    different_path = scenario("different", ["T02", "T01"], shared)
    manual = ScenarioAssemblage("manual", source_type="manual", tri_ids=[])

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [active, same_path, different_path, manual]
    viewer.active_scenario_index = 0
    viewer._refresh_scenario_listbox = lambda: None
    viewer._set_active_scenario = lambda index: None

    viewer._filter_auto_scenarios_by_prefix_edges("T02")

    assert viewer.scenarios == [active, same_path, manual]


def test_prefix_filter_accepts_the_first_ordered_element_without_attachment_step():
    def scenario(name, ordered_element_ids):
        item = ScenarioAssemblage(name, source_type="auto", tri_ids=[1, 2])
        item.orderedElementIds = list(ordered_element_ids)
        item.topoWorld.elements.update({"T01": object(), "T02": object()})
        return item

    active = scenario("active", ["T01", "T02"])
    same_prefix = scenario("same", ["T01", "T02"])
    other_prefix = scenario("other", ["T02", "T01"])
    manual = ScenarioAssemblage("manual", source_type="manual")
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [active, same_prefix, other_prefix, manual]
    viewer.active_scenario_index = 0
    viewer._refresh_scenario_listbox = lambda: None
    viewer._set_active_scenario = lambda _index: None

    viewer._filter_auto_scenarios_by_prefix_edges("T01")

    assert viewer.scenarios == [active, same_prefix, manual]


def test_prefix_steps_reject_invalid_ordered_element_ids():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    scenario = ScenarioAssemblage("auto", source_type="auto")
    scenario.topoWorld.elements.update({"T01": object()})

    scenario.orderedElementIds = ["T01", "T01"]
    assert viewer._scenario_prefix_edge_steps(scenario, 0) is None

    scenario.orderedElementIds = ["T01", "T02"]
    assert viewer._scenario_prefix_edge_steps(scenario, 1) is None

    scenario.orderedElementIds = ["T01"]
    assert viewer._scenario_prefix_edge_steps(scenario, 1) is None


def test_vertex_move_resolution_uses_dsu_node_and_existing_core_group_without_mutation():
    world = _VertexWorld()
    scenario = ScenarioAssemblage("manual", source_type="manual")
    scenario.topoWorld = world
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._last_drawn = [
        {"topoElementId": "T01", "pts": {}},
        {"topoElementId": "T02", "pts": {}},
    ]
    viewer.canvas_objects = CanvasObjectsCollection(viewer._last_drawn)
    viewer._last_drawn = viewer.canvas_objects.entries
    viewer._get_active_scenario = lambda: scenario
    attachments_before = dict(world.attachments)
    groups_before = dict(world.groups)

    resolved = viewer._resolve_core_vertex_move_members(0, "B")

    assert resolved["element_id"] == "T01"
    assert resolved["node_id"] == "T01:N1"
    assert resolved["node_canon"] == "CN:T01:N1"
    assert resolved["core_group_id"] == "G001"
    assert [entry["topoElementId"] for entry in resolved["entries"]] == ["T01", "T02"]
    assert world.attachments == attachments_before
    assert world.groups == groups_before
