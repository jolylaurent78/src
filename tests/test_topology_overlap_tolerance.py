from src.assembleur_core import (
    TopologyAttachment,
    TopologyElement,
    TopologyFeatureRef,
    TopologyFeatureType,
    TopologyWorld,
)


def _triangle(element_id: str) -> TopologyElement:
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )


def test_overlap_simulation_initializes_its_core_tolerances_for_two_groups():
    world = TopologyWorld()
    destination_group = world.add_element_as_new_group(_triangle("T01"))
    mobile_group = world.add_element_as_new_group(_triangle("T02"))
    world.recomputeConceptAndBoundary(destination_group)
    world.recomputeConceptAndBoundary(mobile_group)
    attachment = TopologyAttachment(
        "A01",
        "edge-edge",
        TopologyFeatureRef(TopologyFeatureType.EDGE, "T01", 0),
        TopologyFeatureRef(TopologyFeatureType.EDGE, "T02", 0),
        {"mapping": "reverse"},
    )

    overlap = world.simulateOverlapTopologique(
        destination_group,
        mobile_group,
        [attachment],
    )

    assert isinstance(overlap, bool)
    assert world.overlap_eps_world == 1e-9
    assert world.overlap_eps_area == 1e-12
