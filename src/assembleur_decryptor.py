"""assembleur_decryptor.py
Moteur + algorithmes d'assemblage automatique (sans dépendance Tk).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.assembleur_core import TopologyCheminTriplet


# ============================================================
# Décryptage (générique) — sans dépendance UI
# ============================================================

@dataclass
class ClockState:
    """État minimal du compas/horloge pour le décryptage."""
    hour: float = 0.0
    minute: int = 0
    label: str = ""
    # La géométrie en azimut
    azHourDeg: float | None = None
    azMinDeg: float | None = None
    deltaDeg180: float | None = None
    # Conserver la provenance (utile pour debug / synchro)
    dicoRow: Optional[int] = None
    dicoCol: Optional[int] = None
    word: str = ""


class DecryptorBase:
    """Contrat minimal pour brancher différents types de décryptage."""
    id: str = "decrypt_base"
    label: str = "Décryptage (base)"

    def __init__(self):
        super().__init__()
        # Paramètres génériques (communs pour l’instant)
        self.hourMovesWithMinutes = True
        # Bases du cadran
        self.minutesBase: int = 60
        self.hoursBase: int = 12

    def getMinutesBase(self) -> int:
        """Base minutes du cadran (ex: 60 ou 100)."""
        return self.minutesBase

    def getHoursBase(self) -> int:
        """Base heures du cadran (ex: 12 ou 10)."""
        return self.hoursBase

    def setMinutesBase(self, base: int):
        b = int(base)
        if b not in (60, 100):
            raise ValueError(f"minutesBase invalide: {base}")
        self.minutesBase = b

    def setHoursBase(self, base: int):
        b = int(base)
        if b not in (12, 10):
            raise ValueError(f"hoursBase invalide: {base}")
        self.hoursBase = b

    def degreesPerMinute(self) -> float:
        base = max(1, int(self.getMinutesBase()))
        return 360.0 / float(base)

    def degreesPerHour(self) -> float:
        base = max(1, int(self.getHoursBase()))
        return 360.0 / float(base)

    def anglesFromClock(self, *, hour: float, minute: int) -> Tuple[float, float]:
        """Retourne (angleHourDeg, angleMinuteDeg) dans [0..360).

        Convention:
          - 0° = 12h
          - sens horaire
        """
        hBase = max(1, int(self.getHoursBase()))
        mBase = max(1, int(self.getMinutesBase()))
        h = float(hour) % float(hBase)
        m = int(minute) % int(mBase)
        ang_min = (m * self.degreesPerMinute()) % 360.0
        ang_hour = (h * self.degreesPerHour()) % 360.0
        return (ang_hour, ang_min)

    def computeDeltaDeg180(self, az1: float, az2: float) -> float:
        """Retourne le plus petit angle (0..180) entre deux azimuts."""
        d = (az2 - az1) % 360.0
        if d > 180.0:
            d = 360.0 - d
        return d

    def clockStateFromDicoCell(self, *, row: int, col: int, word: str = "", mode: str = "delta") -> ClockState:
        """Convertit une cellule (row,col) en état d'horloge.
        Par défaut: non supporté.
        """
        raise NotImplementedError


class ClockDicoDecryptor(DecryptorBase):
    """Décryptage Horloge ↔ Dictionnaire.

    Modes supportés:
      - ABS:
        * row = 1..10 (pas de 0)
        * col = … -2, -1, 1, 2, … (pas de 0)
        * mapping compas: hour=row (1..10), minute=col en base 1
          (col=-1 => 60, col=-2 => 59, etc.)
        * Row 1 (premiere) = 1 heure ; Row 10 (derniere) = 10 heures

      - DELTA:
        * row = 0..9 (0 autorisé), col = delta signé (0 autorisé)
        * mapping compas: hour=row, minute=col mod 60  (ex: -5 => 55')
        * Pour le filtrage d'angle, la ligne a 2 interprétations horaires
          possibles (h et h+10 mod 12) — géré dans deltaAngleFromDicoCell.
    """
    id = "clock_dico_v1"
    label = "Horloge ↔ Dictionnaire (v1)"

    def clockStateFromDicoCell(self, *, row: int, col: int, word: str = "", mode: str = "delta") -> ClockState:
        m = str(mode or "delta").strip().lower()

        hBase = max(1, self.getHoursBase())
        mBase = max(1, self.getMinutesBase())

        if m.startswith("abs"):
            # --- ABS ---
            # Le référentiel est (1, 1) row = [1.10] ligne =  [-100.. 1] ou [1..100] ] la cellule [0,0] n'existe pas
            # On associe des heures de 1 à 10 ==> OUVERTURE = 1H
            rowDisp = ((row - 1) % 10) + 1
            hourDisp = rowDisp
            hour = hourDisp % hBase

            # col: pas de 0. Convertir en minute 1..60.
            # On suppose que la 1ere colonne = 1mn ==> OUVERTURE = 1mn
            # Pour le dictionnaire symétrique col = -1 ==> DEVIN = 60mn
            if col > 0:
                minute = col
            else:
                minute = mBase + col + 1
            minute = minute % mBase
            if minute == 0:
                minute = mBase

        else:
            # --- DELTA ---
            # Le référentiel est (0,0) donc la cellule [0,0] veut dire même mot
            hour = row % hBase
            hourDisp = hour
            # En DELTA, une colonne négative se lit comme une minute "avant":
            #   -5 => 55' (comme 10h - 5' = 9h55)
            # Donc: minute = col mod 60 (0..59)
            minute = col % mBase

        # Option : l’aiguille des heures avance avec les minutes
        if self.hourMovesWithMinutes:
            minuteFloat = 0.0 if minute == mBase else float(minute)
            hourFloat = (float(hour) + minuteFloat / float(mBase)) % float(hBase)
        else:
            hourFloat = float(hour)

        # Label
        w = str(word or "").strip()
        if w:
            label = f"{w} — ({hourDisp}h, {minute}')"
        else:
            label = f"({hourDisp}h, {minute}')"

        # On calule les azimuts et delta
        azHourDeg, azMinDeg = self.anglesFromClock(hour=hourFloat, minute=minute)
        deltaDeg180 = self.computeDeltaDeg180(azHourDeg, azMinDeg)
        return ClockState(
            hour=float(hourFloat),
            minute=minute,
            label=label,
            dicoRow=row,
            dicoCol=col,
            word=w,
            azHourDeg=azHourDeg,
            azMinDeg=azMinDeg,
            deltaDeg180=deltaDeg180,
        )


@dataclass(frozen=True)
class DecryptorConfig:
    """Immutable configuration for matching geometry measures."""

    decryptor: DecryptorBase
    useAzA: bool
    useAzB: bool
    useAngle180: bool
    toleranceDeg: float

    def __post_init__(self) -> None:
        if not isinstance(self.decryptor, DecryptorBase):
            raise TypeError("decryptor must be a DecryptorBase instance")
        for name, value in (
            ("useAzA", self.useAzA),
            ("useAzB", self.useAzB),
            ("useAngle180", self.useAngle180),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        if isinstance(self.toleranceDeg, bool) or not isinstance(self.toleranceDeg, (int, float)):
            raise TypeError("toleranceDeg must be a number")

        tol = float(self.toleranceDeg)
        object.__setattr__(self, "toleranceDeg", tol)

        if not (self.useAzA or self.useAzB or self.useAngle180):
            raise ValueError("at least one measure must be active")
        if tol < 0.0:
            raise ValueError("toleranceDeg must be >= 0")

    def match(self, triplet: "TopologyCheminTriplet", clock: ClockState) -> bool:
        if triplet is None or clock is None:
            raise TypeError("triplet and clock must be provided")

        if self.useAzA:
            azA_target = triplet.azOA
            azA_cand = clock.azHourDeg
            if azA_target is None or azA_cand is None:
                raise ValueError("azOA and azHourDeg must be non-null when useAzA is True")
            diff = abs(float(azA_target) - float(azA_cand)) % 360.0
            diff = min(diff, 360.0 - diff)
            if diff > self.toleranceDeg:
                return False

        if self.useAzB:
            azB_target = triplet.azOB
            azB_cand = clock.azMinDeg
            if azB_target is None or azB_cand is None:
                raise ValueError("azOB and azMinDeg must be non-null when useAzB is True")
            diff = abs(float(azB_target) - float(azB_cand)) % 360.0
            diff = min(diff, 360.0 - diff)
            if diff > self.toleranceDeg:
                return False

        if self.useAngle180:
            target180 = float(triplet.angleDeg) % 180.0
            cand180 = clock.deltaDeg180
            if cand180 is None:
                raise ValueError("deltaDeg180 must be non-null when useAngle180 is True")
            diff = abs(target180 - float(cand180))
            if diff > self.toleranceDeg:
                return False

        return True


# Petit registre (optionnel) pour brancher d?autres d?cryptages
DECRYPTORS: Dict[str, Type[DecryptorBase]] = {
    ClockDicoDecryptor.id: ClockDicoDecryptor,
}
