import pytest

from src.assembleur_core import TopologyElement, TopologyWorld


def _element(element_id, labels):
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=labels,
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[10.0, 10.0, 10.0],
        vertex_local_xy={0: (0.0, 0.0), 1: (10.0, 0.0), 2: (0.0, 10.0)},
    )


def test_business_edge_key_is_canonical_and_independent_of_edge_code():
    world = TopologyWorld()
    world.add_element_as_new_group(_element("T01", ["Bourges", "Rocamadour", "Loches"]))
    world.add_element_as_new_group(_element("T02", ["Loches", "Rocamadour", "Bourges"]))

    assert world.get_element_edge_business_key("T01", " ob ") == (
        "Bourges",
        "Rocamadour",
    )
    assert world.are_same_business_edge("T01", "OB", "T02", "BL")


def test_business_edge_requires_the_same_two_cities_not_the_same_length():
    world = TopologyWorld()
    world.add_element_as_new_group(_element("T01", ["Bourges", "Rocamadour", "Loches"]))
    world.add_element_as_new_group(_element("T02", ["Rocamadour", "Vierzon", "Bourges"]))
    world.add_element_as_new_group(_element("T03", ["A", "B", "C"]))

    assert not world.are_same_business_edge("T01", "OB", "T02", "OB")
    assert not world.are_same_business_edge("T01", "OB", "T03", "OB")


@pytest.mark.parametrize(
    ("element_id", "edge", "exception"),
    [("UNKNOWN", "OB", KeyError), ("T01", "OX", ValueError)],
)
def test_business_edge_key_rejects_invalid_arguments(element_id, edge, exception):
    world = TopologyWorld()
    world.add_element_as_new_group(_element("T01", ["Bourges", "Rocamadour", "Loches"]))

    with pytest.raises(exception):
        world.get_element_edge_business_key(element_id, edge)


def test_business_edge_key_rejects_missing_labels():
    world = TopologyWorld()
    world.add_element_as_new_group(_element("T01", ["Bourges", "", "Loches"]))

    with pytest.raises(ValueError, match="libellé de ville absent"):
        world.get_element_edge_business_key("T01", "OB")
