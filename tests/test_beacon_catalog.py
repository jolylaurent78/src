import pytest

from src.assembleur_balises import Beacon, BeaconCatalog
from src.assembleur_core import TopologyWorld

pytest.importorskip("pyproj")


def _write_csv(path, rows):
    path.write_text("Id,Nom,Latitude,Longitude\n" + rows, encoding="utf-8")


def test_beacon_is_immutable_and_catalog_resolves_by_id(tmp_path):
    csv_path = tmp_path / "beacons.csv"
    _write_csv(csv_path, "B02,Donon,48 30 48 N,7 09 54 E\nB01,Grand Ballon,47.9000,7.1000\n")
    catalog = BeaconCatalog()
    catalog.load_from_csv(csv_path)

    beacon = catalog.get("B02")
    assert isinstance(beacon, Beacon)
    assert beacon.beacon_id == "B02"
    assert beacon.name == "Donon"
    assert beacon.latitude > 48.0
    assert isinstance(beacon.lambert_x, float)
    assert catalog.contains("B01")
    assert [item.beacon_id for item in catalog.iter_beacons()] == ["B01", "B02"]


def test_duplicate_names_remain_distinct_and_find_by_name_is_secondary(tmp_path):
    csv_path = tmp_path / "beacons.csv"
    _write_csv(csv_path, "B01,Donon,48.0,7.0\nB02,Donon,48.1,7.1\n")
    catalog = BeaconCatalog()
    catalog.load_from_csv(csv_path)
    assert [item.beacon_id for item in catalog.find_by_name("Donon")] == ["B01", "B02"]


@pytest.mark.parametrize("rows, message", [
    ("B01,Paris,48.0,2.0\nB01,Lyon,45.0,4.0\n", "duplicate beacon_id"),
    (",Paris,48.0,2.0\n", "missing beacon_id"),
    ("B01,Paris,nope,2.0\n", "Invalid latitude"),
    ("B01,Paris,48.0,nope\n", "Invalid longitude"),
])
def test_catalog_rejects_invalid_rows(tmp_path, rows, message):
    csv_path = tmp_path / "beacons.csv"
    _write_csv(csv_path, rows)
    with pytest.raises(ValueError, match=message):
        BeaconCatalog().load_from_csv(csv_path)


def test_unknown_beacon_id_is_rejected():
    with pytest.raises(KeyError):
        BeaconCatalog().get("missing")


def test_world_coordinates_are_rebuilt_in_the_shared_catalog() -> None:
    catalog = BeaconCatalog()
    original = Beacon("B01", "Repère", 48.0, 2.0, 700.0, 6600.0)
    catalog._by_id = {original.beacon_id: original}

    catalog.reproject_world(lambda x, y: (x + 10.0, y - 20.0))

    projected = catalog.get("B01")
    assert projected is not original
    assert original.world_x is None
    assert catalog.get_world("B01") == pytest.approx((710.0, 6580.0))


def test_topology_world_keeps_only_a_catalog_reference() -> None:
    catalog = BeaconCatalog()
    catalog._by_id = {
        "B01": Beacon("B01", "Repère", 48.0, 2.0, 700.0, 6600.0, 12.5, -3.0)
    }
    world = TopologyWorld(beacon_catalog=catalog)

    assert world.hasBeacon("B01")
    assert world.getBeaconWorldXY("B01") == pytest.approx((12.5, -3.0))
    assert not hasattr(world, "beaconsWorld")
    assert not hasattr(world, "setBeaconWorldXY")
    assert not hasattr(world, "clearBeacons")
    assert world.clonePhysicalState()._beacon_catalog is catalog
