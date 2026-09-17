import pytest

from src.assembleur_dictionary_panel import (
    DICO_TAG_EXCLURE,
    DictionaryPanel,
    exclusion_tag_from_bool,
    tk_to_ext_abs,
    tk_to_rel,
)


@pytest.mark.parametrize("value, expected", [(False, None), (True, DICO_TAG_EXCLURE)])
def test_dico_exclude_bool_to_tag_mapping(value, expected):
    assert exclusion_tag_from_bool(value) == expected


def test_tk_to_extended_absolute_coordinates_skip_zero_column():
    assert tk_to_ext_abs(0, 0, 4) == (1, -4)
    assert tk_to_ext_abs(2, 4, 4) == (3, 1)
    assert tk_to_ext_abs(12, 4, 4) == (3, 1)


def test_tk_to_relative_coordinates_support_origin_and_inverted_target():
    assert tk_to_rel(3, 7, origin_row=1, origin_col=5, ref_mode="origin", nb_lignes=10) == (2, 2)
    assert tk_to_rel(3, 7, origin_row=1, origin_col=5, ref_mode="target", nb_lignes=10) == (8, -2)


@pytest.mark.parametrize("row, col, nbm", [(0, 0, 4), (12, 4, 4), (9, 7, 3)])
def test_runtime_absolute_conversion_delegates_to_pure_helper(row, col, nbm):
    panel = object.__new__(DictionaryPanel)
    assert panel._tkToExtAbs(row, col, nbm=nbm) == tk_to_ext_abs(row, col, nbm)


@pytest.mark.parametrize("row, col, r0, c0, mode", [(3, 7, 1, 5, "origin"), (3, 7, 1, 5, "target"), (0, 0, 9, 4, "target")])
def test_runtime_relative_conversion_delegates_to_pure_helper(row, col, r0, c0, mode):
    panel = object.__new__(DictionaryPanel)
    assert panel._tkToRel(row, col, r0=r0, c0=c0, refMode=mode) == tk_to_rel(
        row, col, origin_row=r0, origin_col=c0, ref_mode=mode, nb_lignes=10
    )
