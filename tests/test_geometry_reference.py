"""REF-001A: referentiel local de scenario et resolver geometrique."""

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage
from src.assembleur_geometry_reference import GeometryReferenceResolver, ScenarioReference
from src.assembleur_scenario import (
    ScenarioHypothesis,
    materialize_catalogue_triangle,
    materialize_triangle,
)


def _catalogue_with_triangle():
    catalogue = Catalogue()
    opening = catalogue.add_city("Ouverture", 45.0, 2.0)
    base = catalogue.add_city("Base", 45.1, 2.1)
    light = catalogue.add_city("Lumiere", 45.2, 2.2)
    triangle = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
    return catalogue, opening, base, light, triangle


def _catalogue_with_complete_hypothesis():
    catalogue = Catalogue()
    triangle_ids = []
    for pair_index in range(16):
        base = catalogue.add_city(f"Base {pair_index}", 42.0 + pair_index / 10, 1.0)
        for member_index in range(2):
            opening = catalogue.add_city(
                f"Ouverture {pair_index}-{member_index}",
                44.0 + pair_index / 10,
                2.0 + member_index / 10,
            )
            light = catalogue.add_city(
                f"Lumiere {pair_index}-{member_index}",
                46.0 + pair_index / 10,
                4.0 + member_index / 10,
            )
            triangle = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
            triangle_ids.append(triangle.triangle_id)
    return catalogue, triangle_ids


def test_empty_reference_falls_back_to_catalogue_definitions():
    catalogue, opening, base, light, triangle = _catalogue_with_triangle()
    resolver = GeometryReferenceResolver(catalogue, ScenarioReference())

    city = resolver.resolve_city(opening.city_id)
    resolved = resolver.resolve_triangle(triangle.triangle_id)

    assert city.ref_id == opening.city_id
    assert city.origin == "catalogue"
    assert city.catalogue_source_city_id == opening.city_id
    assert resolved.ref_id == triangle.triangle_id
    assert resolved.origin == "catalogue"
    assert resolver.city_ref_ids_by_role(triangle.triangle_id) == {
        "O": opening.city_id,
        "B": base.city_id,
        "L": light.city_id,
    }


def test_local_city_is_independent_from_catalogue_and_has_lambert_coordinates():
    catalogue, *_ = _catalogue_with_triangle()
    reference = ScenarioReference()
    local = reference.create_city(
        "Ville locale", 48.8566, 2.3522, catalogue_source_city_id="CITY-0001"
    )
    resolver = GeometryReferenceResolver(catalogue, reference)

    resolved = resolver.resolve_city(local.city_ref_id)
    lambert = resolver.get_city_lambert(local.city_ref_id)

    assert local.city_ref_id == "SCITY-0001"
    assert resolved.origin == "scenario"
    assert resolved.catalogue_source_city_id == "CITY-0001"
    assert len(catalogue.cities) == 3
    assert all(abs(value) > 1.0 for value in lambert)


def test_local_triangle_can_mix_catalogue_and_local_city_references():
    catalogue, opening, base, _light, triangle = _catalogue_with_triangle()
    reference = ScenarioReference()
    local_light = reference.create_city("Lumiere locale", 46.0, 3.0)
    local_triangle = reference.create_triangle(
        "Do local",
        opening.city_id,
        base.city_id,
        local_light.city_ref_id,
        catalogue_source_triangle_id=triangle.triangle_id,
    )
    resolver = GeometryReferenceResolver(catalogue, reference)

    resolved = resolver.resolve_triangle(local_triangle.triangle_ref_id)

    assert local_triangle.triangle_ref_id == "STRI-0001"
    assert resolved.origin == "scenario"
    assert resolver.city_ref_ids_by_role(local_triangle.triangle_ref_id)["L"] == local_light.city_ref_id
    assert resolver.resolve_city(local_light.city_ref_id).name == "Lumiere locale"


def test_resolver_reports_unknown_references_explicitly():
    catalogue, *_ = _catalogue_with_triangle()
    resolver = GeometryReferenceResolver(catalogue, ScenarioReference())

    with pytest.raises(KeyError, match="Triangle inconnu"):
        resolver.resolve_triangle("STRI-9999")
    with pytest.raises(KeyError, match="Ville inconnue"):
        resolver.resolve_city("SCITY-9999")


def test_catalogue_materialization_remains_compatible_via_resolver():
    catalogue, opening, base, light, triangle = _catalogue_with_triangle()
    points = {
        opening.city_id: (0.0, 0.0),
        base.city_id: (3000.0, 0.0),
        light.city_id: (0.0, 4000.0),
    }
    catalogue.get_city_lambert = lambda city_id: points[city_id]
    resolver = GeometryReferenceResolver(catalogue, ScenarioReference())

    old = materialize_catalogue_triangle(catalogue, triangle.triangle_id)
    new = materialize_triangle(resolver, triangle.triangle_id)

    assert new.source_triangle_id == old.source_triangle_id == triangle.triangle_id
    assert new.vertex_labels == old.vertex_labels
    assert new.edge_lengths_km == pytest.approx(old.edge_lengths_km)
    assert new.vertex_local_xy == pytest.approx(old.vertex_local_xy)


def test_local_triangle_materializes_with_its_effective_reference_id():
    catalogue, opening, base, _light, triangle = _catalogue_with_triangle()
    reference = ScenarioReference()
    local_light = reference.create_city("Lumiere locale", 46.0, 3.0)
    local_triangle = reference.create_triangle(
        "Do local", opening.city_id, base.city_id, local_light.city_ref_id,
        catalogue_source_triangle_id=triangle.triangle_id,
    )

    element = materialize_triangle(
        GeometryReferenceResolver(catalogue, reference), local_triangle.triangle_ref_id
    )

    assert element.source_triangle_id == "STRI-0001"
    assert element.vertex_labels == ["Ouverture", "Base", "Lumiere locale"]
    assert all(length > 0.0 for length in element.edge_lengths_km)


def test_reference_clone_is_independent_and_keeps_next_ids_coherent():
    reference = ScenarioReference()
    city = reference.create_city("Locale", 48.0, 2.0)
    triangle = reference.create_triangle("Local", "CITY-0001", "CITY-0002", city.city_ref_id)

    cloned = reference.clone()
    cloned.cities[city.city_ref_id].name = "Locale modifiee"
    cloned.triangles[triangle.triangle_ref_id].note = "Local modifie"

    assert reference.cities[city.city_ref_id].name == "Locale"
    assert reference.triangles[triangle.triangle_ref_id].note == "Local"
    assert reference.create_city("Deuxieme", 47.0, 3.0).city_ref_id == "SCITY-0002"
    assert cloned.create_city("Deuxieme clone", 47.0, 3.0).city_ref_id == "SCITY-0002"
    assert reference.create_triangle("Second", "CITY-0003", "CITY-0004", "CITY-0005").triangle_ref_id == "STRI-0002"
    assert cloned.create_triangle("Second clone", "CITY-0003", "CITY-0004", "CITY-0005").triangle_ref_id == "STRI-0002"


def test_hypothesis_accepts_local_triangle_when_validated_with_resolver():
    catalogue, triangle_ids = _catalogue_with_complete_hypothesis()
    reference = ScenarioReference()
    second = catalogue.get_triangle(triangle_ids[1])
    local_light = reference.create_city("Lumiere locale", 49.0, 3.0)
    local_triangle = reference.create_triangle(
        "Do local",
        second.opening_city_id,
        second.base_city_id,
        local_light.city_ref_id,
        catalogue_source_triangle_id=triangle_ids[0],
    )
    ranks = list(triangle_ids)
    ranks[0] = local_triangle.triangle_ref_id
    hypothesis = ScenarioHypothesis(ranks)

    hypothesis.validate(GeometryReferenceResolver(catalogue, reference))
    with pytest.raises(ValueError, match="Triangle inconnu"):
        hypothesis.validate(catalogue)


def test_rank_resolution_uses_the_effective_local_triangle_reference():
    catalogue, triangle_ids = _catalogue_with_complete_hypothesis()
    source_triangle_id = triangle_ids[24]
    source = catalogue.get_triangle(source_triangle_id)
    reference = ScenarioReference()
    local_light = reference.create_city("Lumiere locale", 49.0, 3.0)
    local_triangle = reference.create_triangle(
        "Do local", source.opening_city_id, source.base_city_id, local_light.city_ref_id,
        catalogue_source_triangle_id=source_triangle_id,
    )
    resolver = GeometryReferenceResolver(catalogue, reference)
    ranks = list(triangle_ids)
    ranks[24] = local_triangle.triangle_ref_id
    hypothesis = ScenarioHypothesis(ranks)

    assert resolver.get_catalogue_source_triangle_id(source_triangle_id) == source_triangle_id
    assert resolver.get_catalogue_source_triangle_id(local_triangle.triangle_ref_id) == source_triangle_id
    assert ScenarioHypothesis(triangle_ids).get_rank_for_triangle_ref(source_triangle_id) == 25
    assert hypothesis.get_rank_for_triangle_ref(local_triangle.triangle_ref_id) == 25
    with pytest.raises(ValueError, match="Référence effective absente"):
        hypothesis.get_rank_for_triangle_ref(source_triangle_id)


def test_rank_resolution_rejects_a_local_triangle_without_catalogue_provenance():
    catalogue, opening, base, light, _triangle = _catalogue_with_triangle()
    reference = ScenarioReference()
    local_triangle = reference.create_triangle(
        "Sans provenance", opening.city_id, base.city_id, light.city_id,
    )
    resolver = GeometryReferenceResolver(catalogue, reference)

    with pytest.raises(ValueError, match="sans provenance Catalogue"):
        resolver.get_catalogue_source_triangle_id(local_triangle.triangle_ref_id)


def test_scenario_assemblage_owns_an_independent_empty_reference():
    first = ScenarioAssemblage("Premier")
    second = ScenarioAssemblage("Second")

    assert first.reference.cities == {}
    assert first.reference.triangles == {}
    assert second.reference is not first.reference
    first.reference.create_city("Locale", 48.0, 2.0)
    assert second.reference.cities == {}
