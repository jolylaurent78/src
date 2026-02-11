import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent))

from helpers_triplets import triplets_smoke

from src.DictionnaireEnigmes import DictionnaireEnigmes
from src.assembleur_decryptor import ClockDicoDecryptor, DecryptorConfig


@pytest.fixture(scope="module")
def dico():
    data_path = Path(__file__).resolve().parents[1] / "data" / "livre.txt"
    if not data_path.is_file():
        raise FileNotFoundError(f"Fichier dictionnaire introuvable: {data_path}")
    return DictionnaireEnigmes(str(data_path))


@pytest.fixture
def decryptorConfigAngle180():
    return DecryptorConfig(
        decryptor=ClockDicoDecryptor(),
        useAzA=False,
        useAzB=False,
        useAngle180=True,
        toleranceDeg=4.0,
    )


def _clock_from_cell(dico, decryptorConfigAngle180, row_ext: int, col_ext: int):
    word = dico.getMotExtended(row_ext, col_ext)
    return decryptorConfigAngle180.decryptor.clockStateFromDicoCell(
        row=row_ext,
        col=col_ext,
        word=word,
        mode="abs",
    )


def test_angle180_matching_positive(dico, decryptorConfigAngle180, triplets_smoke):
    triplet1, triplet2, triplet3, _ = triplets_smoke

    for (row_ext, col_ext) in [(9, 19), (7, 63), (9, -47), (3, -10)]:
        clock = _clock_from_cell(dico, decryptorConfigAngle180, row_ext, col_ext)
        assert decryptorConfigAngle180.match(triplet1, clock) is True

    for (row_ext, col_ext) in [(1, 21), (1, -40)]:
        clock = _clock_from_cell(dico, decryptorConfigAngle180, row_ext, col_ext)
        assert decryptorConfigAngle180.match(triplet2, clock) is True

    clock = _clock_from_cell(dico, decryptorConfigAngle180, 9, 37)
    assert decryptorConfigAngle180.match(triplet3, clock) is True


def test_angle180_matching_negative(dico, decryptorConfigAngle180, triplets_smoke):
    _, triplet2, _, _ = triplets_smoke
    clock = _clock_from_cell(dico, decryptorConfigAngle180, 9, 19)
    assert decryptorConfigAngle180.match(triplet2, clock) is False
