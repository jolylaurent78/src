"""Catalogue-to-Core triangle materialisation."""

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import TopologyWorld, build_topology_element_from_catalogue_triangle
from src.assembleur_scenario import materialize_catalogue_triangle


def _element_for(light_y):
    return build_topology_element_from_catalogue_triangle(
        triangle_id="TRI-0042",
        opening_name="Ouverture",
        base_name="Base",
        light_name="Lumière",
        opening_lambert_xy=(0.0, 0.0),
        base_lambert_xy=(3000.0, 0.0),
        light_lambert_xy=(0.0, light_y),
    )


def test_factory_derives_geometry_and_catalogue_identity():
    element = _element_for(4000.0)

    assert element.source_triangle_id == "TRI-0042"
    assert element.vertex_labels == ["Ouverture", "Base", "Lumière"]
    assert element.edge_lengths_km == pytest.approx([3.0, 5.0, 4.0])
    assert element.vertex_local_xy[2] == pytest.approx((0.0, 4.0))
    assert element.meta["orient"] == "CCW"
    assert "triRank" not in element.meta
    assert "modelId" not in element.meta


def test_materialisation_resolves_catalogue_cities():
    catalogue = Catalogue()
    opening = catalogue.add_city("O", 45.0, 2.0)
    base = catalogue.add_city("B", 45.1, 2.1)
    light = catalogue.add_city("L", 45.2, 2.2)
    triangle = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
    points = {
        opening.city_id: (0.0, 0.0),
        base.city_id: (3000.0, 0.0),
        light.city_id: (0.0, 4000.0),
    }
    catalogue.get_city_lambert = lambda city_id: points[city_id]

    element = materialize_catalogue_triangle(catalogue, triangle.triangle_id)

    assert element.source_triangle_id == triangle.triangle_id
    assert element.vertex_labels == ["O", "B", "L"]


def test_source_triangle_identity_survives_a_core_snapshot():
    world = TopologyWorld()
    world.add_element_as_new_group(_element_for(4000.0))

    restored = TopologyWorld()
    restored._importPhysicalSnapshot(world._exportPhysicalSnapshot())

    assert next(iter(restored.elements.values())).source_triangle_id == "TRI-0042"
