import json

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_catalogue_io import save_catalogue
from src.assembleur_tk import _load_application_catalogue


def test_startup_catalogue_loader_creates_empty_catalogue_only_when_absent(tmp_path):
    provider = SystemCatalogueIdProvider()
    catalogue = _load_application_catalogue(str(tmp_path / "absent.json"), provider)
    assert catalogue.cities == {}
    assert catalogue.id_provider is provider


def test_startup_catalogue_loader_loads_a_valid_catalogue(tmp_path):
    provider = SystemCatalogueIdProvider()
    source = Catalogue(id_provider=provider)
    source.add_city("Paris", 48.8, 2.3)
    path = tmp_path / "catalogue.json"
    save_catalogue(source, path)

    loaded = _load_application_catalogue(str(path), provider)
    assert set(loaded.cities) == {"CITY-SYS-000001"}
    assert loaded.id_provider is provider


def test_startup_catalogue_loader_fails_fast_when_an_existing_catalogue_is_invalid(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps({
        "version": 1,
        "defaultTemplateId": None,
        "cities": [{
            "cityId": "CITY-0001", "name": "Paris", "latitude": 48.8,
            "longitude": 2.3, "archived": False,
        }],
        "beacons": [], "triangles": [], "templates": [],
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Impossible de charger le catalogue") as exc_info:
        _load_application_catalogue(str(path), SystemCatalogueIdProvider())
    assert "Version de catalogue non supportée : 1" in str(exc_info.value)
