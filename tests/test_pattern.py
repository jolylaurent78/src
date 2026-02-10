from pathlib import Path

import pytest

from src.DictionnaireEnigmes import (
    DictionnaireEnigmes,
    PATTERN_MAX_TOKENS,
    PATTERN_MIN_TOKENS,
    Pattern,
)


@pytest.fixture(scope="module")
def dico():
    data_path = Path(__file__).resolve().parents[1] / "data" / "livre.txt"
    return DictionnaireEnigmes(str(data_path))


def _cat_words(dico_obj, categorie: str) -> set[str]:
    return {mot for (mot, _, _) in dico_obj.getListeCategorie(categorie)}


def test_setSyntax_pattern_vide(dico):
    p = Pattern(dico)
    ok, msg = p.setSyntax(" ")
    assert ok is False
    assert msg == "Pattern vide"


def test_setSyntax_trop_court(dico):
    p = Pattern(dico)
    ok, msg = p.setSyntax("QUAND")
    assert ok is False
    assert msg == f"Pattern trop court (min = {PATTERN_MIN_TOKENS})"


def test_setSyntax_trop_long(dico):
    p = Pattern(dico)
    ok, msg = p.setSyntax("[*] [*] [*] [localise] [ordre] QUAND")
    assert ok is False
    assert msg == f"Pattern trop long (max = {PATTERN_MAX_TOKENS})"


def test_setSyntax_token_etoile_interdit(dico):
    p = Pattern(dico)
    ok, msg = p.setSyntax("* QUAND")
    assert ok is False
    assert msg == "Token '*' interdit, utiliser '[*]'"


def test_setSyntax_uniquement_jokers(dico):
    p = Pattern(dico)
    ok, msg = p.setSyntax("[*] [*]")
    assert ok is False
    assert msg == "Pattern non discriminant (uniquement des jokers)"


def test_setSyntax_categorie_valide(dico):
    p = Pattern(dico)
    ok, msg = p.setSyntax("[localise] QUAND")
    assert ok is True
    assert msg == ""


def test_setSyntax_categorie_invalide(dico):
    p = Pattern(dico)
    ok, msg = p.setSyntax("[faux] QUAND")
    assert ok is False
    assert msg == "Categorie inconnue: FAUX"


def test_setSyntax_mot_valide(dico):
    p = Pattern(dico)
    ok, msg = p.setSyntax("[ordre] AUGUSTA")
    assert ok is True
    assert msg == ""


def test_setSyntax_mot_invalide(dico):
    p = Pattern(dico)
    ok, msg = p.setSyntax("[ordre] BLABLA")
    assert ok is False
    assert msg == "Mot inconnu du dictionnaire: BLABLA"


def test_donnees_localise_contient_et_pas(dico):
    words = _cat_words(dico, "localise")
    assert "CHERCHE" in words
    assert "TROUVER" in words
    assert "DEPLACE" not in words


def test_donnees_ordre_contient_et_pas(dico):
    words = _cat_words(dico, "ordre")
    assert "SEPTIEME" in words
    assert "DEVANT" not in words


def test_loadCache_ne_leve_pas(dico):
    p = Pattern(dico)
    ok, _ = p.setSyntax("[localise] QUAND")
    assert ok is True
    p.loadCache()


def test_validateSequence_localise_safe(dico):
    p = Pattern(dico)
    ok, _ = p.setSyntax("[localise] [*]")
    assert ok is True
    p.loadCache()
    assert p.validateSequence(["CHERCHE"], "safe") == "MAYBE"
    assert p.validateSequence(["DEPLACE"], "safe") == "NO"
    assert p.validateSequence(["CHERCHE", "QUAND"], "safe") == "YES"


def test_validateSequence_ordre_safe(dico):
    p = Pattern(dico)
    ok, _ = p.setSyntax("[ordre] [*]")
    assert ok is True
    p.loadCache()
    assert p.validateSequence(["SEPTIEME"], "safe") == "MAYBE"
    assert p.validateSequence(["DEVANT"], "safe") == "NO"
    assert p.validateSequence(["SEPTIEME", "AUGUSTA"], "safe") == "YES"


def test_validateSequence_mode_last(dico):
    p = Pattern(dico)
    ok, _ = p.setSyntax("[localise] QUAND")
    assert ok is True
    p.loadCache()
    assert p.validateSequence(["DEPLACE", "QUAND"], "last") == "YES"
