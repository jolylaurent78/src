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


def test_sequence_incremental_equivalence_safe(dico, patterns_syntax):
    lp = ListePatterns(dico, patterns_syntax)
    seq = ["LA", "SEPTIEME", "SENTINELLES"]

    scratch = lp.validateSequence(seq, mode="safe")

    _, _, packed = lp.validateSequence([seq[0]], mode="safe")
    incremental = None
    for k in range(2, len(seq) + 1):
        incremental = lp.validateSequence(seq[:k], mode="safe", packedState=packed)
        _, _, packed = incremental

    assert incremental == scratch


def test_sequence_incremental_equivalence_last(dico, patterns_syntax):
    lp = ListePatterns(dico, patterns_syntax)
    seq = ["LA", "SEPTIEME", "SENTINELLES"]

    scratch = lp.validateSequence(seq, mode="last")

    _, _, packed = lp.validateSequence([seq[0]], mode="last")
    incremental = None
    for k in range(2, len(seq) + 1):
        incremental = lp.validateSequence(seq[:k], mode="last", packedState=packed)
        _, _, packed = incremental

    assert incremental == scratch


def test_validateSequence_rejects_packed_out_of_range(dico, patterns_syntax):
    lp = ListePatterns(dico, patterns_syntax)
    pattern_count = lp.getPatternCount()
    bad = 1 << (2 * pattern_count)
    with pytest.raises(RuntimeError):
        lp.validateSequence(["SEPTIEME"], mode="safe", packedState=bad)


def test_validateSequence_rejects_reserved_0b11(dico, patterns_syntax):
    lp = ListePatterns(dico, patterns_syntax)
    with pytest.raises(RuntimeError):
        lp.validateSequence(["SEPTIEME"], mode="safe", packedState=0b11)
