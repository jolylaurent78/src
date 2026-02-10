import pytest
from pathlib import Path

from src.DictionnaireEnigmes import DictionnaireEnigmes, DicoScope


ROW_SIZES = [73, 21, 100, 41, 21, 31, 65, 89, 27, 75]
NB_LIGNES = 10
PLAGE_MAX = 100
TOTAL_STRICT = 543


@pytest.fixture(scope="module")
def dico():
    data_path = Path(__file__).resolve().parents[1] / "data" / "livre.txt"
    if not data_path.is_file():
        raise FileNotFoundError(f"Fichier dictionnaire introuvable: {data_path}")
    return DictionnaireEnigmes(str(data_path))


# -------------------------
# A) Tests structure
# -------------------------


def test_tc_a1_getNbLignes(dico):
    assert dico.getNbLignes() == NB_LIGNES


def test_tc_a2_getRowSize(dico):
    for row_ext, expected in enumerate(ROW_SIZES, start=1):
        assert dico.getRowSize(row_ext) == expected


def test_tc_a3_getPlageMax(dico):
    assert dico.getPlageMax() == PLAGE_MAX


# -------------------------
# B) Tests normalizeRow
# -------------------------


def test_tc_b1_normalisation_simple(dico):
    assert dico.normalizeRow(1) == 1
    assert dico.normalizeRow(10) == 10


def test_tc_b2_wrap_positif(dico):
    assert dico.normalizeRow(11) == 1
    assert dico.normalizeRow(12) == 2


def test_tc_b3_wrap_negatif(dico):
    assert dico.normalizeRow(0) == 10
    assert dico.normalizeRow(-1) == 9


# -------------------------
# C) Tests iterCoords(scope)
# -------------------------


def test_tc_c1_itercoords_strict_cardinalite(dico):
    coords = list(dico.iterCoords(DicoScope.STRICT))
    assert len(coords) == TOTAL_STRICT


def test_tc_c2_itercoords_mirroring_cardinalite(dico):
    coords = list(dico.iterCoords(DicoScope.MIRRORING))
    assert len(coords) == 2 * TOTAL_STRICT


def test_tc_c3_itercoords_extended_cardinalite(dico):
    coords = list(dico.iterCoords(DicoScope.EXTENDED))
    assert len(coords) == NB_LIGNES * 2 * PLAGE_MAX


def test_tc_c4_itercoords_strict_domaine(dico):
    for row, col in dico.iterCoords(DicoScope.STRICT):
        assert 1 <= row <= NB_LIGNES
        assert 1 <= col <= ROW_SIZES[row - 1]


def test_tc_c5_itercoords_mirroring_domaine(dico):
    for row, col in dico.iterCoords(DicoScope.MIRRORING):
        assert 1 <= row <= NB_LIGNES
        assert col != 0
        assert abs(col) <= ROW_SIZES[row - 1]


def test_tc_c6_itercoords_extended_domaine(dico):
    for row, col in dico.iterCoords(DicoScope.EXTENDED):
        assert 1 <= row <= NB_LIGNES
        assert col != 0
        assert abs(col) <= PLAGE_MAX


# -------------------------
# D) Tests getMotExtended
# -------------------------


def test_tc_d1_getMotExtended_row_out_of_range(dico):
    with pytest.raises(IndexError):
        dico.getMotExtended(0, 1)
    with pytest.raises(IndexError):
        dico.getMotExtended(11, 1)


def test_tc_d2_getMotExtended_col_zero(dico):
    with pytest.raises(ValueError):
        dico.getMotExtended(3, 0)


def test_tc_d3_getMotExtended_col_out_of_range(dico):
    with pytest.raises(IndexError):
        dico.getMotExtended(3, 101)
    with pytest.raises(IndexError):
        dico.getMotExtended(3, -101)


def test_tc_d4_getMotExtended_examples(dico):
    assert dico.getMotExtended(5, -8) == "CLEF"
    assert dico.getMotExtended(5, 77) == "CLEF"
    assert dico.getMotExtended(3, 55) == "QU"


# -------------------------
# E) Tests recalageAbs
# -------------------------


def test_tc_e1_recalage_exemple(dico):
    assert dico.recalageAbs(5, 77) == (5, 14)


def test_tc_e2_recalage_col_dans_bornes(dico):
    sample_cols = [-100, -99, -1, 1, 2, 50, 99, 100]
    for row_ext in (1, 5, 10):
        row_real = dico.normalizeRow(row_ext)
        nrow = dico.getRowSize(row_real)
        for col_ext in sample_cols:
            r, c = dico.recalageAbs(row_ext, col_ext)
            assert r == row_real
            assert 1 <= c <= nrow


def test_tc_e3_recalage_normalise_row(dico):
    assert dico.recalageAbs(15, 77) == (5, 14)


# -------------------------
# F) Tests helpers ABS/REL
# -------------------------


def test_tc_f1_computeDeltaAbs(dico):
    delta_row, delta_col = dico.computeDeltaAbs((5, 14), (5, 77))
    assert delta_row == 0
    assert delta_col == 63


def test_tc_f2_applyDeltaAbs(dico):
    assert dico.applyDeltaAbs((5, 14), (0, 63)) == (5, 77)


def test_tc_f3_deltaRow_cyclique(dico):
    delta_row, delta_col = dico.computeDeltaAbs((10, 1), (1, 1))
    assert delta_row == 1
    assert delta_col == 0
