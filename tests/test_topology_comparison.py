from src.assembleur_core import (
    ScenarioAssemblage,
    TopologyAttachment,
    TopologyFeatureRef,
    TopologyFeatureType,
)
from src.assembleur_topology_comparison import (
    build_attachment_signature,
    differing_attachment_element_ids,
)
from src.assembleur_tk import TriangleViewerManual


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
    scenario.last_drawn = [{"id": 1, "topoElementId": "T01"}]
    viewer.scenarios = [scenario]
    viewer.active_scenario_index = 0
    viewer.ref_scenario_token = id(scenario)
    viewer._comparison_diff_indices = {99}

    viewer._update_current_scenario_differences()

    assert viewer._comparison_diff_indices == set()
