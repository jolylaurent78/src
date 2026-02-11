import pytest

from src.assembleur_decryptor import DecryptorBase, DecryptorConfig


class StubTriplet:
    def __init__(self, azOA=0, azOB=0, angleDeg=0):
        self.azOA = azOA
        self.azOB = azOB
        self.angleDeg = angleDeg


class StubClock:
    def __init__(self, azHourDeg=0, azMinDeg=0, deltaDeg180=0):
        self.azHourDeg = azHourDeg
        self.azMinDeg = azMinDeg
        self.deltaDeg180 = deltaDeg180


def test_tc1_construction_valide():
    decryptor = DecryptorBase()
    cfg = DecryptorConfig(
        decryptor=decryptor,
        useAzA=False,
        useAzB=True,
        useAngle180=False,
        toleranceDeg=0.0,
    )
    assert cfg.decryptor is decryptor
    assert cfg.useAzA is False
    assert cfg.useAzB is True
    assert cfg.useAngle180 is False
    assert cfg.toleranceDeg == 0.0


def test_tc2_erreur_aucune_mesure_active():
    decryptor = DecryptorBase()
    with pytest.raises(ValueError):
        DecryptorConfig(
            decryptor=decryptor,
            useAzA=False,
            useAzB=False,
            useAngle180=False,
            toleranceDeg=1.0,
        )


def test_tc3_erreur_tolerance_negative():
    decryptor = DecryptorBase()
    with pytest.raises(ValueError):
        DecryptorConfig(
            decryptor=decryptor,
            useAzA=True,
            useAzB=False,
            useAngle180=False,
            toleranceDeg=-1.0,
        )


def test_tc4_aza_match_parfait():
    cfg = DecryptorConfig(DecryptorBase(), True, False, False, 0.0)
    triplet = StubTriplet(azOA=10)
    clock = StubClock(azHourDeg=10)
    ok, score = cfg.match(triplet, clock)
    assert ok is True
    assert score == 0.0


def test_tc5_aza_hors_tolerance():
    cfg = DecryptorConfig(DecryptorBase(), True, False, False, 2.0)
    triplet = StubTriplet(azOA=10)
    clock = StubClock(azHourDeg=15)
    ok, score = cfg.match(triplet, clock)
    assert ok is False
    assert score == 5.0


def test_tc6_azb_wrap360_dans_tolerance():
    cfg = DecryptorConfig(DecryptorBase(), False, True, False, 5.0)
    triplet = StubTriplet(azOB=358)
    clock = StubClock(azMinDeg=2)
    ok, score = cfg.match(triplet, clock)
    assert ok is True
    assert score == 4.0


def test_tc7_angle180_match_parfait():
    cfg = DecryptorConfig(DecryptorBase(), False, False, True, 0.0)
    triplet = StubTriplet(angleDeg=200)
    clock = StubClock(deltaDeg180=160)
    ok, score = cfg.match(triplet, clock)
    assert ok is True
    assert score == 0.0


def test_tc8_angle180_hors_tolerance():
    cfg = DecryptorConfig(DecryptorBase(), False, False, True, 1.0)
    triplet = StubTriplet(angleDeg=170)
    clock = StubClock(deltaDeg180=180)
    ok, score = cfg.match(triplet, clock)
    assert ok is False
    assert score == 10.0


def test_tc9_combinatoire_trois_mesures_actives():
    cfg = DecryptorConfig(DecryptorBase(), True, True, True, 3.0)
    triplet = StubTriplet(azOA=30, azOB=150, angleDeg=120)
    clock = StubClock(azHourDeg=29, azMinDeg=153, deltaDeg180=121)
    ok, score = cfg.match(triplet, clock)
    assert ok is True
    assert score == pytest.approx(3.0)


def test_tc10_combinatoire_un_seul_echoue():
    cfg = DecryptorConfig(DecryptorBase(), True, True, True, 3.0)
    triplet = StubTriplet(azOA=30, azOB=150, angleDeg=120)
    clock = StubClock(azHourDeg=29, azMinDeg=157, deltaDeg180=121)
    ok, score = cfg.match(triplet, clock)
    assert ok is False
    assert score == 7.0


def test_tc11_wrap360_limite_exacte():
    cfg = DecryptorConfig(DecryptorBase(), True, False, False, 0.0)
    triplet = StubTriplet(azOA=359)
    clock = StubClock(azHourDeg=359)
    ok, score = cfg.match(triplet, clock)
    assert ok is True
    assert score == 0.0


def test_tc12_wrap360_bord_extreme():
    cfg = DecryptorConfig(DecryptorBase(), True, False, False, 1.0)
    triplet = StubTriplet(azOA=0)
    clock = StubClock(azHourDeg=359)
    ok, score = cfg.match(triplet, clock)
    assert ok is True
    assert score == 1.0


def test_tc13_valeurs_manquantes():
    cfg = DecryptorConfig(DecryptorBase(), True, False, False, 1.0)
    triplet = StubTriplet(azOA=None)
    clock = StubClock(azHourDeg=10)
    with pytest.raises(ValueError):
        cfg.match(triplet, clock)
