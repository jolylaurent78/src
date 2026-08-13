import json

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_io import catalogue_from_dict, load_catalogue, save_catalogue


def _catalogue() -> Catalogue:
    catalogue = Catalogue()
    opening = catalogue.add_city("Bourges", 47.081, 2.398)
    base = catalogue.add_city("Rocamadour", 44.799, 1.619)
    light = catalogue.add_city("Loches", 47.128, 0.995)
    triangle = catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id)
    template = catalogue.add_template("Ordre principal", "Description")
    catalogue.set_template_rank(template.template_id, 1, triangle.triangle_id)
    return catalogue


def test_round_trip_keeps_ids_data_references_and_runtime_cache_empty(tmp_path):
    catalogue = _catalogue()
    catalogue.update_city("CITY-0002", archived=True)
    catalogue.update_triangle("TRI-0001", archived=True)
    catalogue.get_city_lambert("CITY-0001")
    path = tmp_path / "nested" / "catalogue.json"

    save_catalogue(catalogue, path)
    loaded = load_catalogue(path)

    assert set(loaded.cities) == set(catalogue.cities)
    assert set(loaded.triangles) == set(catalogue.triangles)
    assert set(loaded.templates) == set(catalogue.templates)
    assert loaded.default_template_id == catalogue.default_template_id
    assert loaded.templates["TPL-0001"].triangle_ids_by_rank == catalogue.templates["TPL-0001"].triangle_ids_by_rank
    assert loaded._city_lambert_cache == {}
    assert loaded.get_city_lambert("CITY-0001")
    assert not path.with_suffix(".json.tmp").exists()
    assert "Bourges" in path.read_text(encoding="utf-8")


def test_incomplete_template_round_trips_and_ids_keep_max_plus_one(tmp_path):
    catalogue = _catalogue()
    path = tmp_path / "catalogue.json"
    save_catalogue(catalogue, path)
    loaded = load_catalogue(path)

    assert loaded.get_template_validation_status("TPL-0001").state == "Incomplet"
    unused = loaded.add_city("Ville isolée", 42.0, 0.0)
    unused.city_id = "CITY-0005"
    loaded.cities["CITY-0005"] = loaded.cities.pop("CITY-0004")
    reloaded_path = tmp_path / "sparse.json"
    save_catalogue(loaded, reloaded_path)
    sparse = load_catalogue(reloaded_path)
    assert sparse.add_city("Nouvelle ville", 40.0, 0.0).city_id == "CITY-0006"


def test_invalid_json_version_missing_field_and_invalid_reference_are_rejected(tmp_path):
    syntax_path = tmp_path / "syntax.json"
    syntax_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON catalogue invalide"):
        load_catalogue(syntax_path)

    with pytest.raises(ValueError, match="Version de catalogue non supportée : 2"):
        catalogue_from_dict({"version": 2})
    with pytest.raises(KeyError):
        catalogue_from_dict({"version": 1, "cities": []})

    data = {
        "version": 1,
        "defaultTemplateId": None,
        "cities": [],
        "triangles": [{
            "triangleId": "TRI-0001", "note": "Do", "openingCityId": "CITY-0001",
            "baseCityId": "CITY-0002", "lightCityId": "CITY-0003", "archived": False,
        }],
        "templates": [],
    }
    with pytest.raises(KeyError, match="Ville inconnue"):
        catalogue_from_dict(data)


def test_save_replaces_existing_file_atomically(tmp_path):
    path = tmp_path / "catalogue.json"
    first = _catalogue()
    save_catalogue(first, path)
    second = _catalogue()
    second.update_template("TPL-0001", description="Nouvelle description")
    save_catalogue(second, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["templates"][0]["description"] == "Nouvelle description"
    assert not path.with_suffix(".json.tmp").exists()
