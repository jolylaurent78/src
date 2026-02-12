# src/assembleur_engine_runtime.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Any, Optional


# ============================================================
# EngineControl : télécommande passive (partagée UX <-> moteur)
# ============================================================

class EngineControl:
    """
    Télécommande passive :
    - Mutable par l'UX
    - Lue par le moteur aux checkpoints (pas à chaque cellule)
    """
    def __init__(self) -> None:
        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()

    # --- UX side ---
    def requestStop(self) -> None:
        self._stop_evt.set()

    def requestPause(self) -> None:
        self._pause_evt.set()

    def resume(self) -> None:
        self._pause_evt.clear()

    # --- Engine side (read-only) ---
    def isStopRequested(self) -> bool:
        return self._stop_evt.is_set()

    def isPauseRequested(self) -> bool:
        return self._pause_evt.is_set()


# ============================================================
# EventQueue : canal d'événements thread-safe moteur -> UX
# ============================================================

@dataclass(frozen=True)
class EngineEvent:
    """
    type: "STATUS" | "SOLUTION" | "PROGRESS" | "DONE" | "STOPPED" | "ERROR"
    payload: dépend du type
    """
    type: str
    payload: Any = None


class EventQueue:
    """
    Wrapper simple autour de queue.Queue pour la V1.
    La queue est considérée non bornée en pratique (volume limité par design).
    """
    def __init__(self) -> None:
        self._q: Queue[EngineEvent] = Queue()

    def put(self, eventType: str, payload: Any = None) -> None:
        self._q.put(EngineEvent(str(eventType), payload))

    def getNowait(self) -> Optional[EngineEvent]:
        try:
            return self._q.get_nowait()
        except Empty:
            return None

    def empty(self) -> bool:
        return self._q.empty()


# ============================================================
# RunControlConfig : paramètres de pilotage d'un run
# ============================================================

@dataclass(frozen=True)
class RunControlConfig:
    """
    Paramètres de contrôle d'exécution (copiés au démarrage du run).
    """
    maxSolutions: int = 50

    # Batch adaptatif (unité = cellules testées)
    minBatchCells: int = 500
    maxBatchCells: int = 200_000

    # Cible de réactivité UI (en secondes)
    targetBatchSec: float = 0.05  # 50 ms

    # Cadence minimale entre 2 events PROGRESS (en secondes)
    progressMinIntervalSec: float = 0.2


# ============================================================
# CheckpointPolicy : logique batch/timer adaptative
# ============================================================

class CheckpointPolicy:
    """
    Gère :
    - le comptage des cellules testées
    - le timing des batches
    - l'ajustement adaptatif de la taille de batch
    - la cadence des events PROGRESS
    """

    def __init__(self, cfg: RunControlConfig, enabled: bool = True) -> None:
        self.cfg = cfg
        self.enabled = bool(enabled)

        # taille de batch courante (en cellules testées)
        self._batchCellsTarget = int(cfg.minBatchCells)

        # compteurs
        self._testedCellsSinceLastCheck = 0

        # timing
        self._lastBatchStartTime = time.perf_counter()
        self._lastProgressTime = 0.0

    # --- appelé par le moteur à chaque cellule testée ---
    def onCellTested(self, n: int = 1) -> None:
        self._testedCellsSinceLastCheck += int(n)

    # --- le moteur appelle ceci pour savoir s'il faut faire un checkpoint ---
    def shouldCheckpoint(self) -> bool:
        if not self.enabled:
            return False
        return self._testedCellsSinceLastCheck >= self._batchCellsTarget

    # --- appelé par le moteur AU MOMENT du checkpoint ---
    def onCheckpoint(self) -> dict:
        """
        Retourne un dict d'infos utiles au moteur :
        {
          "emitProgress": bool
        }
        Et met à jour la policy (batch adaptatif, reset compteurs, timers).
        """
        now = time.perf_counter()
        elapsed = now - self._lastBatchStartTime

        # --- Ajustement adaptatif de la taille de batch ---
        # Si trop lent -> on divise par 2
        if elapsed > self.cfg.targetBatchSec * 2.0:
            newTarget = max(self.cfg.minBatchCells, self._batchCellsTarget // 2)
        # Si trop rapide -> on multiplie par 2
        elif elapsed < self.cfg.targetBatchSec * 0.5:
            newTarget = min(self.cfg.maxBatchCells, self._batchCellsTarget * 2)
        else:
            newTarget = self._batchCellsTarget

        self._batchCellsTarget = int(newTarget)

        # --- Décider si on peut émettre un PROGRESS ---
        emitProgress = False
        if (now - self._lastProgressTime) >= float(self.cfg.progressMinIntervalSec):
            emitProgress = True
            self._lastProgressTime = now

        # --- Reset batch ---
        self._testedCellsSinceLastCheck = 0
        self._lastBatchStartTime = now

        return {
            "emitProgress": emitProgress,
            "batchCellsTarget": self._batchCellsTarget,
            "lastBatchElapsed": elapsed,
        }


# ============================================================
# Mini test manuel
# ============================================================

def _demo():
    ctrl = EngineControl()
    q = EventQueue()
    cfg = RunControlConfig()
    policy = CheckpointPolicy(cfg, enabled=True)

    q.put("STATUS", "RUNNING")
    ctrl.requestPause()
    print("pause?", ctrl.isPauseRequested())  # True
    ctrl.resume()
    print("pause?", ctrl.isPauseRequested())  # False

    # Simuler des cellules testées
    for i in range(2000):
        policy.onCellTested()
        if policy.shouldCheckpoint():
            info = policy.onCheckpoint()
            print("Checkpoint:", info)

    ev = q.getNowait()
    print(ev)


if __name__ == "__main__":
    _demo()
