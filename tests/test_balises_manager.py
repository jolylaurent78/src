import pytest

from src.assembleur_balises import BalisesManager

pytest.importorskip("pyproj")


def test_balises_load_csv_ok(tmp_path):
    csv_path = tmp_path / "balises_ok.csv"
    csv_path.write_text(
        "Nom,Latitude,Longitude\n"
        "Zeta,48 50 11 N,2 20 11 E\n"
        "Alpha,47 28 25 N,0 33 15 W\n",
        encoding="utf-8",
    )

    manager = BalisesManager()
    manager.loadFromCsv(str(csv_path))

    assert manager.listNoms() == ["Alpha", "Zeta"]
    x_km, y_km = manager.getLambertKm("Zeta")
    assert isinstance(x_km, float)
    assert isinstance(y_km, float)


def test_balises_header_invalide(tmp_path):
    csv_path = tmp_path / "balises_bad_header.csv"
    csv_path.write_text(
        "Nom;Latitude;Longitude\n"
        "Paris,48 50 11 N,2 20 11 E\n",
        encoding="utf-8",
    )

    manager = BalisesManager()
    with pytest.raises(ValueError):
        manager.loadFromCsv(str(csv_path))


def test_balises_dms_invalide(tmp_path):
    csv_path = tmp_path / "balises_bad_dms.csv"
    csv_path.write_text(
        "Nom,Latitude,Longitude\n"
        "Paris,48.5,2 20 11 E\n",
        encoding="utf-8",
    )

    manager = BalisesManager()
    with pytest.raises(ValueError):
        manager.loadFromCsv(str(csv_path))


def test_balises_nom_duplique(tmp_path):
    csv_path = tmp_path / "balises_dup.csv"
    csv_path.write_text(
        "Nom,Latitude,Longitude\n"
        "Paris,48 50 11 N,2 20 11 E\n"
        " Paris ,48 51 11 N,2 21 11 E\n",
        encoding="utf-8",
    )

    manager = BalisesManager()
    with pytest.raises(ValueError):
        manager.loadFromCsv(str(csv_path))


def test_balises_get_lambert_inconnu(tmp_path):
    csv_path = tmp_path / "balises_ok.csv"
    csv_path.write_text(
        "Nom,Latitude,Longitude\n"
        "Paris,48 50 11 N,2 20 11 E\n",
        encoding="utf-8",
    )

    manager = BalisesManager()
    manager.loadFromCsv(str(csv_path))

    with pytest.raises(KeyError):
        manager.getLambertKm("Inconnue")
