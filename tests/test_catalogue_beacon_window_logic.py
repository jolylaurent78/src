from types import SimpleNamespace

import pytest
from openpyxl import Workbook

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_window import CatalogueWindow


class _Value:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _catalogue_with_cities() -> Catalogue:
    catalogue = Catalogue()
    catalogue.add_city("Grand Ballon", 47.9, 7.1)
    catalogue.add_city("Donon", 48.5, 7.1)
    catalogue.add_city("Frontiere Nord", 50.0, 2.0)
    return catalogue


def _window_logic(catalogue: Catalogue):
    return SimpleNamespace(
        catalogue=catalogue,
        _BEACON_XLSX_HEADER=CatalogueWindow._BEACON_XLSX_HEADER,
        _beacon_search_var=_Value(""),
        _show_archived_beacons_var=_Value(False),
    )


def _write_beacon_workbook(path, names):
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(("Nom",))
    for name in names:
        worksheet.append((name,))
    workbook.save(path)
    workbook.close()


def test_beacon_list_uses_city_names_search_and_archive_filter():
    catalogue = _catalogue_with_cities()
    first = catalogue.add_beacon("CITY-0001")
    second = catalogue.add_beacon("CITY-0002")
    catalogue.update_beacon(second.beacon_id, archived=True)
    window = _window_logic(catalogue)

    visible = CatalogueWindow._visible_beacons(window)
    assert [catalogue.get_city(beacon.city_id).name for beacon in visible] == ["Grand Ballon"]

    window._show_archived_beacons_var = _Value(True)
    window._beacon_search_var = _Value("don")
    visible = CatalogueWindow._visible_beacons(window)
    assert [beacon.beacon_id for beacon in visible] == [second.beacon_id]
    assert CatalogueWindow._beacon_export_names(window) == ("Grand Ballon", "Donon")
    assert first.city_id not in {city.city_id for city in CatalogueWindow._available_beacon_cities(window)}


def test_beacon_xlsx_import_resolves_city_ids_and_is_atomic(tmp_path):
    catalogue = _catalogue_with_cities()
    window = _window_logic(catalogue)
    source = tmp_path / "balises.xlsx"
    _write_beacon_workbook(source, (" Grand Ballon ", "donon", "Frontiere Nord"))

    rows = CatalogueWindow._read_beacons_xlsx(window, str(source))
    CatalogueWindow._import_beacon_rows(window, rows)

    assert [(beacon.beacon_id, beacon.city_id) for beacon in window.catalogue.iter_beacons()] == [
        ("BEA-0001", "CITY-0001"),
        ("BEA-0002", "CITY-0002"),
        ("BEA-0003", "CITY-0003"),
    ]

    invalid = tmp_path / "invalid.xlsx"
    _write_beacon_workbook(invalid, ("Donon", "Ville inconnue", "Grand Ballon"))
    with pytest.raises(ValueError, match="ville inconnue"):
        CatalogueWindow._read_beacons_xlsx(window, str(invalid))
    assert tuple(window.catalogue.beacons) == ("BEA-0001", "BEA-0002", "BEA-0003")


def test_beacon_xlsx_duplicate_rolls_back_without_mutating_the_catalogue(tmp_path):
    catalogue = _catalogue_with_cities()
    window = _window_logic(catalogue)
    source = tmp_path / "duplicate.xlsx"
    _write_beacon_workbook(source, ("Donon", "Donon"))

    rows = CatalogueWindow._read_beacons_xlsx(window, str(source))
    with pytest.raises(ValueError, match="déjà une balise"):
        CatalogueWindow._import_beacon_rows(window, rows)
    assert window.catalogue.beacons == {}
