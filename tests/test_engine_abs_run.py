import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent))

from helpers_triplets import CheminsStub, triplets_smoke

from src.DictionnaireEnigmes import DicoScope, DictionnaireEnigmes, ListePatterns
from src.assembleur_decryptor import ClockDicoDecryptor, DecryptorConfig
from src.assembleur_decryptor_engine import DecryptorEngine


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


@pytest.fixture
def listePatterns_engine(dico):
    return ListePatterns(
        dico,
        [
            "[LOCALISE] [ORDRE] SENTINELLES",
            "MON [ORDRE]",
        ],
    )


@pytest.fixture
def listePatterns_engine_joker(dico):
    return ListePatterns(
        dico,
        [
            "[*] [ORDRE] SENTINELLES",
            "MON [ORDRE]",
        ],
    )


@pytest.fixture
def engine(dico, triplets_smoke):
    chemins = CheminsStub(triplets_smoke[:3])
    return DecryptorEngine(chemins, dico)


def test_engine_abs_strict_total_2_solutions(engine, listePatterns_engine, decryptorConfigAngle180):
    results = engine.runAbs(
        scope=DicoScope.STRICT,
        listePatterns=listePatterns_engine,
        decryptorConfig=decryptorConfigAngle180,
        patternMode="last",
    )
    assert len(results) == 2


def test_engine_abs_mirroring_total_6_solutions(engine, listePatterns_engine, decryptorConfigAngle180):
    results = engine.runAbs(
        scope=DicoScope.MIRRORING,
        listePatterns=listePatterns_engine,
        decryptorConfig=decryptorConfigAngle180,
        patternMode="last",
    )
    assert len(results) == 6


def test_engine_abs_extended_total_15_solutions(engine, listePatterns_engine, decryptorConfigAngle180):
    results = engine.runAbs(
        scope=DicoScope.EXTENDED,
        listePatterns=listePatterns_engine,
        decryptorConfig=decryptorConfigAngle180,
        patternMode="last",
    )
    assert len(results) == 15

    counts = {0: 0, 1: 0}
    for sol in results:
        counts[sol.patternIndex] = counts.get(sol.patternIndex, 0) + 1
        if sol.patternIndex == 0:
            assert sol.words[-1] == "SENTINELLES"
        if sol.patternIndex == 1:
            assert sol.words[0] == "MON"

    assert counts[0] == 9
    assert counts[1] == 6


def test_engine_abs_extended_total_joker_solutions(engine, listePatterns_engine_joker, decryptorConfigAngle180):
    results = engine.runAbs(
        scope=DicoScope.EXTENDED,
        listePatterns=listePatterns_engine_joker,
        decryptorConfig=decryptorConfigAngle180,
        patternMode="last",
    )
    assert len(results) == 9

    counts = {0: 0, 1: 0}
    for sol in results:
        counts[sol.patternIndex] = counts.get(sol.patternIndex, 0) + 1
        if sol.patternIndex == 0:
            assert sol.words[-1] == "SENTINELLES"
        if sol.patternIndex == 1:
            assert sol.words[0] == "MON"

    assert counts[0] == 3
    assert counts[1] == 6