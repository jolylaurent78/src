from types import SimpleNamespace

import numpy as np
import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage, TopologyElement, TopologyWorld
from src.assembleur_scenario import ScenarioHypothesis
from src.assembleur_tk import TriangleViewerManual


class _BeaconResolver:
    def __init__(self, world_by_id):
        self.world_by_id = world_by_id

    def contains(self, beacon_id):
        return beacon_id in self.world_by_id

    def get_world(self, beacon_id):
        return self.world_by_id[beacon_id]


def _catalogue_with_beacons():
    catalogue = Catalogue()
    first = catalogue.add_city("Balise 1", 45.0, 2.0)
    second = catalogue.add_city("Balise 2", 46.0, 3.0)
    catalogue.add_beacon(first.city_id)
    catalogue.add_beacon(second.city_id)
    resolver = _BeaconResolver({"BEA-0001": (10.0, 5.0), "BEA-0002": (20.0, 0.0)})
    return catalogue, resolver


def _catalogue_triangle_ids(catalogue, count):
    base = catalogue.add_city("Base", 44.0, 1.0)
    triangle_ids = []
    for index in range(count):
        opening = catalogue.add_city(f"Ouverture {index}", 43.0, 1.0 + index / 10)
        light = catalogue.add_city(f"Lumiere {index}", 45.0, 1.0 + index / 10)
        triangle_ids.append(
            catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id).triangle_id
        )
    return triangle_ids


def _element(element_id="T01", source_triangle_id=None):
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[3.0, 4.0, 5.0],
        source_triangle_id=source_triangle_id,
    )


def _viewer(catalogue, resolver, world):
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._beacon_world_resolver = resolver
    viewer.scenarios = [SimpleNamespace(topoWorld=world, source_type="manual")]
    viewer.active_scenario_index = 0
    return viewer


def test_orientation_reference_resolves_the_rank_from_scenario_hypothesis():
    catalogue, resolver = _catalogue_with_beacons()
    triangle_ids = _catalogue_triangle_ids(catalogue, 3)
    world = TopologyWorld(beacon_resolver=resolver)
    first = world.add_element_as_new_group(_element("T01", triangle_ids[1]))
    second = world.add_element_as_new_group(_element("T02", triangle_ids[2]))
    world.createGroupAnchor(first, "BEA-0001", world.get_element_vertex_node_id_by_type("T01", "O"))
    world.createGroupAnchor(second, "BEA-0001", world.get_element_vertex_node_id_by_type("T02", "L"))
    theta = np.pi / 3.0
    world.setElementPose("T02", np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))), np.zeros(2))
    scenario = ScenarioAssemblage("Scenario", hypothesis=ScenarioHypothesis(triangle_ids))
    scenario.topoWorld = world

    reference = _viewer(catalogue, resolver, world)._find_orientation_reference_for_beacon(scenario, "BEA-0001")

    assert reference.element_id == "T02"
    assert reference.tri_rank == 3
    assert reference.theta_rad == pytest.approx(theta)


def test_orientation_reference_keeps_the_smallest_hypothesis_rank():
    catalogue, resolver = _catalogue_with_beacons()
    triangle_ids = _catalogue_triangle_ids(catalogue, 2)
    world = TopologyWorld(beacon_resolver=resolver)
    for element_id, triangle_id in (("T01", triangle_ids[1]), ("T02", triangle_ids[0])):
        group_id = world.add_element_as_new_group(_element(element_id, triangle_id))
        world.createGroupAnchor(group_id, "BEA-0001", world.get_element_vertex_node_id_by_type(element_id, "L"))
    scenario = ScenarioAssemblage("Scenario", hypothesis=ScenarioHypothesis(triangle_ids))
    scenario.topoWorld = world

    reference = _viewer(catalogue, resolver, world)._find_orientation_reference_for_beacon(scenario, "BEA-0001")

    assert reference.element_id == "T02"
    assert reference.tri_rank == 1


def test_orientation_reference_resolves_a_local_triangle_to_its_catalogue_rank():
    catalogue, resolver = _catalogue_with_beacons()
    triangle_ids = _catalogue_triangle_ids(catalogue, 25)
    source = catalogue.get_triangle(triangle_ids[24])
    scenario = ScenarioAssemblage("Scenario", hypothesis=ScenarioHypothesis(triangle_ids))
    local_light = scenario.reference.create_city("Lumiere locale", 49.0, 3.0)
    local_triangle = scenario.reference.create_triangle(
        "Do local", source.opening_city_id, source.base_city_id, local_light.city_ref_id,
        catalogue_source_triangle_id=source.triangle_id,
    )
    scenario.hypothesis.triangle_ids_by_rank[24] = local_triangle.triangle_ref_id
    world = TopologyWorld(beacon_resolver=resolver)
    group_id = world.add_element_as_new_group(_element("T25", local_triangle.triangle_ref_id))
    world.createGroupAnchor(group_id, "BEA-0001", world.get_element_vertex_node_id_by_type("T25", "L"))
    scenario.topoWorld = world

    reference = _viewer(catalogue, resolver, world)._find_orientation_reference_for_beacon(scenario, "BEA-0001")

    assert reference.element_id == "T25"
    assert reference.tri_rank == 25


def test_orientation_reference_rejects_missing_source_triangle_id():
    catalogue, resolver = _catalogue_with_beacons()
    world = TopologyWorld(beacon_resolver=resolver)
    group_id = world.add_element_as_new_group(_element("T01", None))
    world.createGroupAnchor(group_id, "BEA-0001", world.get_element_vertex_node_id_by_type("T01", "L"))
    scenario = ScenarioAssemblage("Scenario", hypothesis=ScenarioHypothesis([]))
    scenario.topoWorld = world

    with pytest.raises(ValueError, match="source_triangle_id absent"):
        _viewer(catalogue, resolver, world)._find_orientation_reference_for_beacon(scenario, "BEA-0001")


def test_orientation_reference_rejects_catalogue_triangle_absent_from_hypothesis():
    catalogue, resolver = _catalogue_with_beacons()
    triangle_ids = _catalogue_triangle_ids(catalogue, 2)
    world = TopologyWorld(beacon_resolver=resolver)
    group_id = world.add_element_as_new_group(_element("T01", triangle_ids[0]))
    world.createGroupAnchor(group_id, "BEA-0001", world.get_element_vertex_node_id_by_type("T01", "L"))
    scenario = ScenarioAssemblage("Scenario", hypothesis=ScenarioHypothesis([triangle_ids[1]]))
    scenario.topoWorld = world

    with pytest.raises(ValueError, match=r"absent de l.hypoth"):
        _viewer(catalogue, resolver, world)._find_orientation_reference_for_beacon(scenario, "BEA-0001")


def test_nearest_beacon_uses_active_catalogue_beacons_and_runtime_resolver():
    catalogue, resolver = _catalogue_with_beacons()
    world = TopologyWorld(beacon_resolver=resolver)
    viewer = _viewer(catalogue, resolver, world)

    candidate = viewer._find_nearest_beacon_candidate(np.array((-10.0, 0.0)))

    assert candidate["beacon_id"] == "BEA-0001"
    assert candidate["distance2"] == 425.0
    catalogue.update_beacon("BEA-0001", archived=True)
    assert viewer._find_nearest_beacon_candidate(np.array((-10.0, 0.0)))["beacon_id"] == "BEA-0002"


def test_auto_scenario_anchor_uses_catalogue_beacon_id():
    catalogue, resolver = _catalogue_with_beacons()
    world = TopologyWorld(beacon_resolver=resolver)
    group_id = world.add_element_as_new_group(_element())
    scenario = ScenarioAssemblage(name="Auto", source_type="auto")
    scenario.topoWorld = world
    scenario.orderedElementIds = ["T01"]
    viewer = _viewer(catalogue, resolver, world)
    viewer._project_auto_scenario_from_core = lambda _scenario: None

    viewer._anchor_auto_scenario_to_beacon(scenario, "BEA-0001")

    anchor = world.getAnchorForGroup(group_id)
    assert anchor.beacon_id == "BEA-0001"
    assert world.getConceptNodeWorldXY(anchor.node_id, group_id) == pytest.approx((10.0, 5.0))


def test_auto_scenario_rejects_archived_beacon_for_a_new_anchor():
    catalogue, resolver = _catalogue_with_beacons()
    catalogue.update_beacon("BEA-0001", archived=True)
    world = TopologyWorld(beacon_resolver=resolver)
    world.add_element_as_new_group(_element())
    scenario = ScenarioAssemblage(name="Auto", source_type="auto")
    scenario.topoWorld = world
    scenario.orderedElementIds = ["T01"]
    viewer = _viewer(catalogue, resolver, world)

    with pytest.raises(ValueError, match="archivÃ©e"):
        viewer._anchor_auto_scenario_to_beacon(scenario, "BEA-0001")
