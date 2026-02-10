from pathlib import Path

import pytest

from src.DictionnaireEnigmes import DictionnaireEnigmes, ListePatterns


@pytest.fixture(scope="module")
def dico():
    data_path = Path(__file__).resolve().parents[1] / "data" / "livre.txt"
    return DictionnaireEnigmes(str(data_path))


@pytest.fixture(scope="module")
def patterns_syntax():
    return ["[*] [ORDRE] SENTINELLES", "[ORDRE] SENTINELLES", "SENTINELLES"]


def _solution_from_yes(patterns, yes_indexes):
    if not yes_indexes:
        return None
    return patterns[min(yes_indexes)]


def test_sequence_s1(dico, patterns_syntax):
    lp = ListePatterns(dico, patterns_syntax)
    continue_explore, yes_indexes, _ = lp.validateSequence(["SEPTIEME"], mode="safe")
    assert continue_explore is True
    assert yes_indexes == []
    assert _solution_from_yes(patterns_syntax, yes_indexes) is None


def test_sequence_s2(dico, patterns_syntax):
    lp = ListePatterns(dico, patterns_syntax)
    continue_explore, yes_indexes, _ = lp.validateSequence(
        ["LA", "SEPTIEME", "SENTINELLES"], mode="safe"
    )
    assert continue_explore is False
    assert yes_indexes == [0]
    assert _solution_from_yes(patterns_syntax, yes_indexes) == "[*] [ORDRE] SENTINELLES"


def test_sequence_s3_invalid_words(dico, patterns_syntax):
    lp = ListePatterns(dico, patterns_syntax)
    continue_explore, yes_indexes, _ = lp.validateSequence(["TEST", "TEST"], mode="safe")
    assert continue_explore is False
    assert yes_indexes == []
    assert _solution_from_yes(patterns_syntax, yes_indexes) is None


def test_sequence_s3(dico, patterns_syntax):
    lp = ListePatterns(dico, patterns_syntax)
    continue_explore, yes_indexes, _ = lp.validateSequence(["SENTINELLES", "BLA"], mode="safe")
    assert continue_explore is False
    assert yes_indexes == []
