from pathlib import Path

import pytest

from src.DictionnaireEnigmes import DictionnaireEnigmes


@pytest.fixture(scope="module")
def dico():
    data_path = Path(__file__).resolve().parents[1] / "data" / "livre.txt"
    if not data_path.is_file():
        raise FileNotFoundError(f"Fichier dictionnaire introuvable: {data_path}")
    return DictionnaireEnigmes(str(data_path))


def test_engine_dataset_words(dico):
    assert dico.getMotExtended(9, 19) == "TROUVE"
    assert dico.getMotExtended(7, 63) == "TROUVER"
    assert dico.getMotExtended(9, -47) == "CHERCHE"
    assert dico.getMotExtended(3, -10) == "MON"
    assert dico.getMotExtended(1, 21) == "QUATRIEME"
    assert dico.getMotExtended(1, -40) == "QUATRIEME"
    assert dico.getMotExtended(9, 37) == "SENTINELLES"
