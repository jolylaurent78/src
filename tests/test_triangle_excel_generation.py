import os
from pathlib import Path

import pandas as pd
import pytest

from src.assembleur_io import (
    TriangleFileService,
)


class _ViewerStub:
    def __init__(self, data_dir: Path):
        self.data_dir = str(data_dir)
        self.appConfig = {}
        self.save_calls = 0

    def getAppConfigValue(self, key, default=None):
        return self.appConfig.get(key, default)

    def setAppConfigValue(self, key, value):
        self.appConfig[key] = value
        self.saveAppConfig()

    def saveAppConfig(self):
        self.save_calls += 1


def test_parse_dms_valid_examples():
    service = TriangleFileService(_ViewerStub(Path(".")))
    assert service._parseDms("47 28 25 N") == pytest.approx(47.4736111111, rel=0, abs=1e-9)
    assert service._parseDms("0 33 15 W") == pytest.approx(-0.5541666667, rel=0, abs=1e-9)


@pytest.mark.parametrize(
    "raw",
    [
        "47 28 N",
        "47 61 25 N",
        "47 28 61 N",
        "47 28 25 X",
        "47 28 25N",
        "",
    ],
)
def test_parse_dms_invalid_formats(raw):
    service = TriangleFileService(_ViewerStub(Path(".")))
    with pytest.raises(ValueError):
        service._parseDms(raw)


def test_create_triangle_excel_generates_expected_file(tmp_path):
    pytest.importorskip("pyproj")
    openpyxl = pytest.importorskip("openpyxl")
    load_workbook = openpyxl.load_workbook
    service = TriangleFileService(_ViewerStub(tmp_path))

    tri_csv = tmp_path / "triangles.csv"
    villes_csv = tmp_path / "villes.csv"
    out_xlsx = tmp_path / "triangles.xlsx"

    tri_csv.write_text(
        "Ouverture;Base;Lumiere\n"
        "Bourges;Rocamadour;Paris\n"
        "Paris;Bourges;Rocamadour\n",
        encoding="utf-8",
    )
    villes_csv.write_text(
        "Nom,Latitude,Longitude\n"
        "Bourges,47 05 00 N,2 24 00 E\n"
        "Rocamadour,44 48 00 N,1 37 00 E\n"
        "Paris,48 51 24 N,2 21 03 E\n",
        encoding="utf-8",
    )

    returned_path = service.createExcelFromCsv(str(tri_csv), str(villes_csv), str(out_xlsx))
    assert os.path.normpath(returned_path) == os.path.normpath(str(out_xlsx))
    assert out_xlsx.exists()

    wb = load_workbook(out_xlsx, read_only=True)
    assert "Triangles" in wb.sheetnames
    ws = wb["Triangles"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    assert headers == [
        "Rang",
        "Ouverture",
        "Base",
        "Lumiere",
        "Ouverture",
        "Base",
        "Lumiere",
        "Ouverture-Base",
        "Ouverture-Lumiere",
        "Lumiere-Base",
        "Orientation",
    ]
    wb.close()

    df = pd.read_excel(out_xlsx, sheet_name="Triangles")
    assert list(df.columns) == [
        "Rang",
        "Ouverture",
        "Base",
        "Lumiere",
        "Ouverture.1",
        "Base.1",
        "Lumiere.1",
        "Ouverture-Base",
        "Ouverture-Lumiere",
        "Lumiere-Base",
        "Orientation",
    ]

    for col in ["Ouverture-Base", "Ouverture-Lumiere", "Lumiere-Base"]:
        assert pd.api.types.is_numeric_dtype(df[col])
        assert df[col].apply(lambda v: round(float(v), 2) == float(v)).all()

    orientations = set(df["Orientation"].dropna().astype(str))
    assert orientations.issubset({"CW", "CCW"})


def test_triangle_file_service_list_files(tmp_path):
    viewer = _ViewerStub(tmp_path)
    service = TriangleFileService(viewer)
    (tmp_path / "b.xlsx").write_text("", encoding="utf-8")
    (tmp_path / "A.xlsx").write_text("", encoding="utf-8")
    (tmp_path / "~$ignored.xlsx").write_text("", encoding="utf-8")
    (tmp_path / "note.txt").write_text("", encoding="utf-8")

    assert service.listTriangleExcelFiles() == ["A.xlsx", "b.xlsx"]


def test_triangle_file_service_get_set_last_paths(tmp_path):
    viewer = _ViewerStub(tmp_path)
    service = TriangleFileService(viewer)

    assert service.getLastPaths() == {
        "lastTriangleCsvIn": "",
        "lastVillesCsvIn": "",
        "lastTriangleExcelOut": "",
        "lastTriangleExcel": "",
    }

    service.setLastPaths(
        lastTriangleCsvIn="a.csv",
        lastVillesCsvIn="v.csv",
        lastTriangleExcelOut="o.xlsx",
        lastTriangleExcel="t.xlsx",
    )
    assert service.getLastPaths() == {
        "lastTriangleCsvIn": "a.csv",
        "lastVillesCsvIn": "v.csv",
        "lastTriangleExcelOut": "o.xlsx",
        "lastTriangleExcel": "t.xlsx",
    }
    assert viewer.save_calls >= 1


def test_triangle_file_service_load_excel_canon(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    Workbook = openpyxl.Workbook
    viewer = _ViewerStub(tmp_path)
    service = TriangleFileService(viewer)

    wb = Workbook()
    ws = wb.active
    ws.title = "Triangles"
    ws.append(["preface"])
    ws.append([
        "Rang", "Ouverture", "Base", "Lumiere",
        "Ouverture", "Base", "Lumiere",
        "Ouverture-Base", "Ouverture-Lumiere", "Lumiere-Base", "Orientation",
    ])
    ws.append([2, "Bourges", "Rocamadour", "Paris", 10.0, 20.0, 150.0, 1.1, 2.2, 3.3, "cw"])
    ws.append([1, "Bourges", "B", "L", 11.0, 22.0, 147.0, 4.4, 5.5, 6.6, "CCW"])
    path = tmp_path / "tri.xlsx"
    wb.save(path)

    df_canon, path_norm = service.loadExcel(str(path))

    assert path_norm == os.path.normpath(os.path.abspath(str(path)))
    assert list(df_canon.columns) == ["id", "B", "L", "len_OB", "len_OL", "len_BL", "orient"]
    assert list(df_canon["id"]) == [1, 2]
    assert set(df_canon["orient"]) == {"CCW", "CW"}


def test_triangle_file_service_create_from_csv(tmp_path):
    pytest.importorskip("pyproj")
    viewer = _ViewerStub(tmp_path)
    service = TriangleFileService(viewer)

    tri_csv = tmp_path / "triangles.csv"
    villes_csv = tmp_path / "villes.csv"
    out_xlsx = tmp_path / "triangles_service.xlsx"
    tri_csv.write_text(
        "Ouverture;Base;Lumiere\n"
        "Bourges;Rocamadour;Paris\n",
        encoding="utf-8",
    )
    villes_csv.write_text(
        "Nom,Latitude,Longitude\n"
        "Bourges,47 05 00 N,2 24 00 E\n"
        "Rocamadour,44 48 00 N,1 37 00 E\n"
        "Paris,48 51 24 N,2 21 03 E\n",
        encoding="utf-8",
    )

    path = service.createExcelFromCsv(str(tri_csv), str(villes_csv), str(out_xlsx))
    assert os.path.normpath(path) == os.path.normpath(str(out_xlsx))
    assert out_xlsx.exists()
