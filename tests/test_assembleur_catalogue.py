import math

import pytest

from src.assembleur_catalogue import (
    Catalogue,
    CatalogueBeacon,
    CatalogueCity,
    CatalogueTriangle,
    HypothesisTemplate,
)
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider


def _three_cities(catalogue: Catalogue):
    return (
        catalogue.add_city("Ouverture", 47.0, 2.0),
        catalogue.add_city("Base", 46.0, 3.0),
        catalogue.add_city("Lumière", 48.0, 4.0),
    )


def _triangle(catalogue: Catalogue):
    opening, base, light = _three_cities(catalogue)
    return catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)


def test_empty_catalogue_uses_initial_system_counters():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    assert catalogue.cities == {}
    assert catalogue.default_template_id is None
    assert catalogue.id_counters == {"city": 0, "beacon": 0, "triangle": 0, "template": 0, "map": 0, "book": 0}
    assert catalogue.add_city("A", 0, 0).city_id == "CITY-SYS-000001"


def test_beacons_have_stable_ids_and_reference_one_catalogue_city_each():
    catalogue = Catalogue()
    first_city = catalogue.add_city("Ville A", 47.0, 2.0)
    second_city = catalogue.add_city("Ville B", 46.0, 3.0)

    first = catalogue.add_beacon(first_city.city_id)
    second = catalogue.add_beacon(second_city.city_id)

    assert first == CatalogueBeacon(first.beacon_id, first_city.city_id)
    assert tuple(CatalogueBeacon.__dataclass_fields__) == ("beacon_id", "city_id", "archived")
    assert second == CatalogueBeacon(second.beacon_id, second_city.city_id)
    assert catalogue.get_beacon(first.beacon_id) is first
    assert {beacon.beacon_id for beacon in catalogue.iter_beacons()} == {first.beacon_id, second.beacon_id}
    with pytest.raises(ValueError, match="introuvable"):
        catalogue.add_beacon("CITY-9999")
    with pytest.raises(ValueError, match="déjà une balise"):
        catalogue.add_beacon(first_city.city_id)
    assert tuple(catalogue.beacons) == (first.beacon_id, second.beacon_id)


def test_beacon_update_is_atomic_and_archived_beacons_still_protect_their_city():
    catalogue = Catalogue()
    first_city = catalogue.add_city("Ville A", 47.0, 2.0)
    second_city = catalogue.add_city("Ville B", 46.0, 3.0)
    first = catalogue.add_beacon(first_city.city_id)
    second = catalogue.add_beacon(second_city.city_id)

    with pytest.raises(ValueError, match="déjà une balise"):
        catalogue.update_beacon(first.beacon_id, city_id=second_city.city_id)
    assert first.city_id == first_city.city_id
    assert second.city_id == second_city.city_id

    catalogue.update_beacon(first.beacon_id, archived=True)
    assert catalogue.get_beacon(first.beacon_id).archived is True
    with pytest.raises(ValueError, match="balise"):
        catalogue.delete_city(first_city.city_id)

    catalogue.delete_beacon(first.beacon_id)
    catalogue.delete_city(first_city.city_id)
    assert first_city.city_id not in catalogue.cities


def test_beacon_clone_is_independent_and_validate_rejects_corrupted_references():
    catalogue = Catalogue()
    city = catalogue.add_city("Ville A", 47.0, 2.0)
    beacon = catalogue.add_beacon(city.city_id)
    clone = catalogue.clone()

    clone.update_beacon(beacon.beacon_id, archived=True)
    assert clone.get_beacon(beacon.beacon_id) is not beacon
    assert catalogue.get_beacon(beacon.beacon_id).archived is False

    invalid_id = "BEA-SYS-000002"
    catalogue.beacons[invalid_id] = CatalogueBeacon(invalid_id, "CITY-SYS-999999")
    with pytest.raises(ValueError, match="introuvable"):
        catalogue.validate()


def test_cities_validate_names_coordinates_and_invalidate_only_changed_coordinate_cache():
    catalogue = Catalogue()
    city = catalogue.add_city("Paris", 48.8, 2.3)
    cached = catalogue.get_city_lambert(city.city_id)
    catalogue.update_city(city.city_id, name="Paris Nord", archived=True)
    assert catalogue._city_lambert_cache[city.city_id] == cached
    catalogue.update_city(city.city_id, latitude=48.9)
    assert city.city_id not in catalogue._city_lambert_cache
    with pytest.raises(ValueError, match="déjà"):
        catalogue.add_city("paris nord", 47, 2)
    with pytest.raises(ValueError):
        catalogue.add_city(" ", 0, 0)
    with pytest.raises(ValueError):
        catalogue.add_city("Trop loin", 91, 0)


def test_triangle_creation_references_and_deletion_rules():
    catalogue = Catalogue()
    triangle = _triangle(catalogue)
    with pytest.raises(ValueError, match="déjà"):
        catalogue.add_triangle("Si", triangle.opening_city_id, triangle.base_city_id, triangle.light_city_id)
    with pytest.raises(ValueError, match="référencée"):
        catalogue.delete_city(triangle.opening_city_id)
    catalogue.update_city(triangle.light_city_id, archived=True)
    catalogue.update_triangle(triangle.triangle_id, note="Si")
    new_city = catalogue.add_city("Archivée", 45, 1, archived=True)
    with pytest.raises(ValueError, match="archivée"):
        catalogue.update_triangle(triangle.triangle_id, light_city_id=new_city.city_id)
    assert catalogue.get_triangles_referencing_city(triangle.base_city_id) == (triangle,)


@pytest.mark.parametrize("field", ["opening_city_id", "base_city_id", "light_city_id"])
def test_update_triangle_rejects_an_explicit_empty_city_id_without_mutation(field):
    catalogue = Catalogue()
    triangle = _triangle(catalogue)
    before = (triangle.opening_city_id, triangle.base_city_id, triangle.light_city_id)
    with pytest.raises(KeyError):
        catalogue.update_triangle(triangle.triangle_id, **{field: ""})
    assert (triangle.opening_city_id, triangle.base_city_id, triangle.light_city_id) == before


def test_geometry_is_lambert_based_lazy_cached_and_rejects_degenerate_triangles():
    catalogue = Catalogue()
    triangle = _triangle(catalogue)
    geometry = catalogue.get_triangle_geometry(triangle.triangle_id)
    assert geometry.distance_ob_km > 0
    assert geometry.orientation in {"CW", "CCW"}
    assert math.isclose(geometry.angle_o_deg + geometry.angle_b_deg + geometry.angle_l_deg, 180.0, abs_tol=1e-7)
    assert len(catalogue._city_lambert_cache) == 3
    collinear = Catalogue()
    a = collinear.add_city("A", 45.0, 2.0)
    b = collinear.add_city("B", 46.0, 2.0)
    c = collinear.add_city("C", 47.0, 2.0)
    tri = collinear.add_triangle("Do", a.city_id, b.city_id, c.city_id)
    collinear._city_lambert_cache = {
        a.city_id: (0.0, 0.0),
        b.city_id: (1.0, 1.0),
        c.city_id: (2.0, 2.0),
    }
    with pytest.raises(ValueError, match="géométrie dégénérée"):
        collinear.get_triangle_geometry(tri.triangle_id)


def test_templates_default_and_rank_rules_are_atomic():
    catalogue = Catalogue()
    triangle = _triangle(catalogue)
    template = catalogue.add_template("Principal")
    second = catalogue.add_template("Second")
    assert catalogue.default_template_id == template.template_id
    catalogue.set_default_template(second.template_id)
    assert catalogue.get_default_template() is second
    with pytest.raises(ValueError, match="plusieurs rangs"):
        catalogue.set_template_rank(template.template_id, 1, triangle.triangle_id)
        catalogue.set_template_rank(template.template_id, 2, triangle.triangle_id)
    assert template.triangle_ids_by_rank[1] is None
    assert catalogue.get_template_validation_status(template.template_id).filled_ranks == 1
    catalogue.delete_template(second.template_id)
    assert catalogue.default_template_id == template.template_id


def test_template_pair_bases_and_references_are_enforced():
    catalogue = Catalogue()
    opening, base, light = _three_cities(catalogue)
    first = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
    other_base = catalogue.add_city("Autre base", 44, 5)
    other_light = catalogue.add_city("Autre lumière", 43, 6)
    second = catalogue.add_triangle("Si", opening.city_id, other_base.city_id, other_light.city_id)
    template = catalogue.add_template("T")
    catalogue.set_template_rank(template.template_id, 1, first.triangle_id)
    with pytest.raises(ValueError, match="même base"):
        catalogue.set_template_rank(template.template_id, 2, second.triangle_id)
    assert template.triangle_ids_by_rank[1] is None
    with pytest.raises(ValueError, match="référencé"):
        catalogue.delete_triangle(first.triangle_id)


def test_set_template_ranks_supports_atomic_move_swap_and_rejects_invalid_preview():
    catalogue = Catalogue()
    opening, base, light = _three_cities(catalogue)
    first = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
    other_light = catalogue.add_city("Lumière deux", 49.0, 5.0)
    second = catalogue.add_triangle("Si", opening.city_id, base.city_id, other_light.city_id)
    template = catalogue.add_template("T")
    catalogue.set_template_rank(template.template_id, 1, first.triangle_id)
    catalogue.set_template_rank(template.template_id, 7, second.triangle_id)
    preview = list(template.triangle_ids_by_rank)
    preview[0], preview[6] = preview[6], preview[0]
    catalogue.set_template_ranks(template.template_id, preview)
    assert template.triangle_ids_by_rank[0] == second.triangle_id
    assert template.triangle_ids_by_rank[6] == first.triangle_id
    preview = list(template.triangle_ids_by_rank)
    preview[0], preview[2] = None, second.triangle_id
    catalogue.set_template_ranks(template.template_id, preview)
    assert template.triangle_ids_by_rank[0] is None
    assert template.triangle_ids_by_rank[2] == second.triangle_id
    before = list(template.triangle_ids_by_rank)
    invalid = list(before)
    invalid[1] = first.triangle_id
    with pytest.raises(ValueError, match="plusieurs rangs"):
        catalogue.set_template_ranks(template.template_id, invalid)
    assert template.triangle_ids_by_rank == before


def test_set_template_ranks_keeps_historical_archived_triangles_but_rejects_new_ones():
    catalogue = Catalogue()
    first = _triangle(catalogue)
    template = catalogue.add_template("T")
    catalogue.set_template_rank(template.template_id, 1, first.triangle_id)
    catalogue.update_triangle(first.triangle_id, archived=True)
    catalogue.set_template_ranks(template.template_id, list(template.triangle_ids_by_rank))
    moved = list(template.triangle_ids_by_rank)
    moved[0], moved[6] = None, first.triangle_id
    catalogue.set_template_ranks(template.template_id, moved)
    second_opening = catalogue.add_city("Autre ouverture", 42.0, 1.0)
    second_base = catalogue.add_city("Autre base", 43.0, 2.0)
    second_light = catalogue.add_city("Autre lumière", 44.0, 3.0)
    second = catalogue.add_triangle("Si", second_opening.city_id, second_base.city_id, second_light.city_id)
    catalogue.update_triangle(second.triangle_id, archived=True)
    before = list(template.triangle_ids_by_rank)
    rejected = list(before)
    rejected[1] = second.triangle_id
    with pytest.raises(ValueError, match="archivé"):
        catalogue.set_template_ranks(template.template_id, rejected)
    assert template.triangle_ids_by_rank == before


def test_validate_template_ranks_is_pure_for_drag_previews():
    catalogue = Catalogue()
    triangle = _triangle(catalogue)
    template = catalogue.add_template("T")
    preview = [None] * 32
    preview[0] = triangle.triangle_id

    assert catalogue.validate_template_ranks(template.template_id, preview) is None
    assert template.triangle_ids_by_rank == [None] * 32

    invalid_preview = list(preview)
    invalid_preview[1] = triangle.triangle_id
    assert catalogue.validate_template_ranks(template.template_id, invalid_preview) is not None
    assert template.triangle_ids_by_rank == [None] * 32


def test_validate_template_ranks_handles_archived_triangles_without_mutation():
    catalogue = Catalogue()
    historical = _triangle(catalogue)
    template = catalogue.add_template("T")
    catalogue.set_template_rank(template.template_id, 1, historical.triangle_id)
    catalogue.update_triangle(historical.triangle_id, archived=True)

    moved = list(template.triangle_ids_by_rank)
    moved[0], moved[2] = None, historical.triangle_id
    assert catalogue.validate_template_ranks(template.template_id, moved) is None
    assert template.triangle_ids_by_rank[0] == historical.triangle_id

    opening = catalogue.add_city("Autre ouverture", 42.0, 1.0)
    base = catalogue.add_city("Autre base", 43.0, 2.0)
    light = catalogue.add_city("Autre lumière", 44.0, 3.0)
    fresh = catalogue.add_triangle("Si", opening.city_id, base.city_id, light.city_id)
    catalogue.update_triangle(fresh.triangle_id, archived=True)
    rejected = list(template.triangle_ids_by_rank)
    rejected[1] = fresh.triangle_id
    assert catalogue.validate_template_ranks(template.template_id, rejected) is not None
    assert template.triangle_ids_by_rank[1] is None


def test_template_status_default_requirement_and_global_validation():
    catalogue = Catalogue()
    template = catalogue.add_template("T")
    assert catalogue.get_template_validation_status(template.template_id).state == "Incomplet"
    assert not catalogue.can_create_scenario()
    with pytest.raises(ValueError, match="incomplet"):
        catalogue.require_valid_default_template()
    catalogue.validate()
    catalogue.default_template_id = "TPL-9999"
    with pytest.raises(ValueError, match="défaut"):
        catalogue.validate()


def test_complete_template_is_valid_and_can_be_required_as_default():
    catalogue = Catalogue()
    opening = catalogue.add_city("Ouverture commune", 47.0, 2.0)
    template = catalogue.add_template("Complet")
    rank = 1
    for pair in range(16):
        base = catalogue.add_city(f"Base {pair}", 43.0 + pair * 0.1, 3.0)
        for member in range(2):
            light = catalogue.add_city(f"Lumière {pair}-{member}", 45.0 + pair * 0.1, 4.0 + member * 0.1)
            triangle = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
            catalogue.set_template_rank(template.template_id, rank, triangle.triangle_id)
            rank += 1
    status = catalogue.get_template_validation_status(template.template_id)
    assert status.state == "Valide"
    assert status.filled_ranks == 32
    assert catalogue.require_valid_default_template() is template
    assert catalogue.can_create_scenario()


def test_validate_detects_directly_corrupted_state():
    catalogue = Catalogue()
    city = catalogue.add_city("A", 0, 0)
    catalogue.cities["CITY-0002"] = city
    with pytest.raises(ValueError, match="Clé ville"):
        catalogue.validate()


def test_clone_copies_the_aggregate_without_copying_runtime_lambert_cache():
    catalogue = Catalogue()
    city = catalogue.add_city("Paris", 48.8566, 2.3522)
    template = catalogue.add_template("T")
    catalogue.get_city_lambert(city.city_id)
    clone = catalogue.clone()
    assert clone is not catalogue
    assert clone.cities[city.city_id] is not city
    assert clone.cities[city.city_id] == city
    assert clone.default_template_id == catalogue.default_template_id
    assert clone.templates[template.template_id].triangle_ids_by_rank is not template.triangle_ids_by_rank
    assert clone._city_lambert_cache == {}
    clone.update_city(city.city_id, name="Paris clone")
    clone.set_template_rank(template.template_id, 1, None)
    assert catalogue.get_city(city.city_id).name == "Paris"
