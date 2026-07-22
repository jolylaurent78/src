from types import SimpleNamespace

import numpy as np
import pytest

from src.assembleur_balises import Beacon, BeaconCatalog
from src.assembleur_core import TopologyElement, TopologyNodeType, TopologyWorld
from src.assembleur_tk import TriangleViewerManual


def _catalog() -> BeaconCatalog:
    catalog = BeaconCatalog()
    catalog._by_id = {
        "BAL-1": Beacon("BAL-1", "Balise 1", 0, 0, 0, 0, 10.0, 5.0),
    }
    return catalog


def _auto_scenario(element_id: str, catalog: BeaconCatalog):
    world = TopologyWorld(beacon_catalog=catalog)
    element = TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=[
            TopologyNodeType.OUVERTURE,
            TopologyNodeType.BASE,
            TopologyNodeType.LUMIERE,
        ],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )
    group_id = world.add_element_as_new_group(element)
    node_id = world.get_element_vertex_node_id_by_type(element_id, "L")
    anchor = world.createGroupAnchor(group_id, "BAL-1", node_id)
    world.applyGroupAnchor(anchor.anchor_id)
    return SimpleNamespace(
        source_type="auto",
        topoWorld=world,
        orderedElementIds=[element_id],
    ), group_id, anchor


def test_collective_auto_rotation_uses_each_scenario_anchor_and_updates_theta():
    catalog = _catalog()
    first, first_group, first_anchor = _auto_scenario("T01", catalog)
    second, second_group, second_anchor = _auto_scenario("T02", catalog)
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [first, second]
    viewer.auto_rotation_state = {"thetaDeg": 0.0}
    viewer._project_auto_scenario_from_core = lambda _scen: None

    viewer._rotate_all_auto_scenarios_around_anchors(np.pi / 2.0)

    assert viewer.auto_rotation_state == {"thetaDeg": 90.0}
    for scenario, group_id, anchor in (
        (first, first_group, first_anchor),
        (second, second_group, second_anchor),
    ):
        world = scenario.topoWorld
        assert world.getAnchorForGroup(group_id) is anchor
        assert world.getConceptNodeWorldXY(anchor.node_id, group_id) == pytest.approx(
            (10.0, 5.0)
        )


def test_collective_auto_rotation_rejects_missing_anchor():
    catalog = _catalog()
    scenario, group_id, anchor = _auto_scenario("T01", catalog)
    scenario.topoWorld.removeGroupAnchor(anchor.anchor_id)
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [scenario]
    viewer.auto_rotation_state = {"thetaDeg": 0.0}

    with pytest.raises(RuntimeError, match="ancre de groupe absente"):
        viewer._rotate_all_auto_scenarios_around_anchors(np.pi / 2.0)

    assert scenario.topoWorld.getAnchorForGroup(group_id) is None
