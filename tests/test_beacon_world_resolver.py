import pytest

from src.assembleur_beacon_runtime import BeaconWorldResolver
from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage, TopologyElement, TopologyWorld
from src.assembleur_tk import TriangleViewerManual


def _catalogue_with_beacon() -> tuple[Catalogue, str]:
    catalogue = Catalogue()
    city = catalogue.add_city("Donon", 48.5133, 7.165)
    beacon = catalogue.add_beacon(city.city_id)
    return catalogue, beacon.beacon_id


def test_resolver_uses_catalogue_city_lambert_and_keeps_archived_beacons_resolvable():
    catalogue, beacon_id = _catalogue_with_beacon()
    catalogue.get_city_lambert = lambda city_id: (1000.0, 2000.0)
    resolver = BeaconWorldResolver(catalogue, lambda x_m, y_m: (x_m / 100.0, y_m / 100.0))

    catalogue.update_beacon(beacon_id, archived=True)

    assert resolver.contains(beacon_id)
    assert resolver.get_world(beacon_id) == pytest.approx((10.0, 20.0))


def test_resolver_immediately_uses_the_published_catalogue_and_current_projection():
    first, beacon_id = _catalogue_with_beacon()
    first.get_city_lambert = lambda _city_id: (1000.0, 2000.0)
    projection_state = {"offset": 0.0}
    projection = lambda x_m, y_m: (x_m + projection_state["offset"], y_m)
    resolver = BeaconWorldResolver(first, projection)

    second = first.clone()
    second.get_city_lambert = lambda _city_id: (3000.0, 4000.0)
    resolver.set_catalogue(second)

    assert resolver.get_world(beacon_id) == pytest.approx((3000.0, 4000.0))
    projection_state["offset"] = 500.0
    assert resolver.get_world(beacon_id) == pytest.approx((3500.0, 4000.0))


def test_topology_world_uses_only_the_injected_resolver_contract():
    catalogue, beacon_id = _catalogue_with_beacon()
    resolver = BeaconWorldResolver(catalogue, lambda _x_m, _y_m: (12.5, -3.0))
    world = TopologyWorld(beacon_resolver=resolver)

    assert world.hasBeacon(beacon_id)
    assert world.getBeaconWorldXY(beacon_id) == pytest.approx((12.5, -3.0))
    assert world.clonePhysicalState()._beacon_resolver is resolver


def test_catalogue_publication_repositions_existing_anchored_groups():
    first, beacon_id = _catalogue_with_beacon()
    first.get_city_lambert = lambda city_id: (first.get_city(city_id).latitude, first.get_city(city_id).longitude)
    resolver = BeaconWorldResolver(first, lambda x_m, y_m: (x_m, y_m))
    world = TopologyWorld(beacon_resolver=resolver)
    group_id = world.add_element_as_new_group(TopologyElement(
        element_id="T01", name="T01", vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"], edge_lengths_km=[3.0, 4.0, 5.0],
    ))
    node_id = world.get_element_vertex_node_id_by_type("T01", "L")
    anchor = world.createGroupAnchor(group_id, beacon_id, node_id)
    world.applyGroupAnchor(anchor.anchor_id)

    published = first.clone()
    published.update_city("CITY-0001", latitude=49.0, longitude=8.0)
    published.get_city_lambert = lambda city_id: (published.get_city(city_id).latitude, published.get_city(city_id).longitude)
    scenario = ScenarioAssemblage("Manuel", source_type="manual")
    scenario.topoWorld = world
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = first
    viewer._beacon_world_resolver = resolver
    viewer.scenarios = [scenario]
    viewer.active_scenario_index = 0
    viewer._rebuild_active_projection_from_core = lambda: None
    viewer._refreshCheminsBaliseRefCombo = lambda: None
    viewer._redraw_from = lambda _entries: None
    viewer._last_drawn = []
    viewer._deformation_window = None

    viewer._publish_catalogue(published)

    assert resolver.get_world(beacon_id) == pytest.approx((49.0, 8.0))
    assert world.getConceptNodeWorldXY(node_id, group_id) == pytest.approx((49.0, 8.0))


def test_viewer_detects_a_beacon_referenced_by_a_runtime_anchor():
    catalogue, beacon_id = _catalogue_with_beacon()
    resolver = BeaconWorldResolver(catalogue, lambda _x_m, _y_m: (0.0, 0.0))
    world = TopologyWorld(beacon_resolver=resolver)
    group_id = world.add_element_as_new_group(TopologyElement(
        element_id="T01", name="T01", vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"], edge_lengths_km=[3.0, 4.0, 5.0],
    ))
    world.createGroupAnchor(group_id, beacon_id, world.get_element_vertex_node_id_by_type("T01", "L"))
    scenario = ScenarioAssemblage("Manuel", source_type="manual")
    scenario.topoWorld = world
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [scenario]

    assert viewer._is_beacon_referenced_by_anchor(beacon_id)
    assert not viewer._is_beacon_referenced_by_anchor("BEA-9999")
