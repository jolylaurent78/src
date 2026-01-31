"""assembleur_sim.py
Moteur + algorithmes d'assemblage automatique (sans dépendance Tk).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Type, Optional, Any
import numpy as np
import math
import copy

from src.assembleur_core import (
    _overlap_shrink,
    _tri_shape,
    _group_shape_from_nodes,
    _build_local_triangle,
    _pose_params,
    _apply_R_T_on_P,
    ScenarioAssemblage,
    TopologyWorld,
    TopologyElement,
    TopologyNodeType,
)
from src.assembleur_edgechoice import (
    EdgeChoiceEpts,
    buildEdgeChoiceEptsForAutoChain,
)

EPS_WORLD = 1e-6

# ============================================================
# Décryptage (générique) — sans dépendance UI
# ============================================================
 
@dataclass
class ClockState:
    """État minimal du compas/horloge pour le décryptage."""
    hour: float = 0.0
    minute: int = 0
    label: str = ""
    # Conserver la provenance (utile pour debug / synchro)
    dicoRow: Optional[int] = None
    dicoCol: Optional[int] = None
    word: str = ""


class DecryptorBase:
    """Contrat minimal pour brancher différents types de décryptage."""
    id: str = "decrypt_base"
    label: str = "Décryptage (base)"

    # ----------------------------
    #  Helpers "angles horloge"
    # ----------------------------
    def getMinutesBase(self) -> int:
        """Base minutes du cadran (ex: 60 ou 100)."""
        return int(getattr(self, "minutesBase", 60) or 60)

    def getHoursBase(self) -> int:
        """Base heures du cadran (ex: 12 ou 10)."""
        return int(getattr(self, "hoursBase", 12) or 12)

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

    def deltaAngleBetweenHands(self, *, hour: float, minute: int) -> float:
        """Angle non orienté entre aiguilles, ramené dans [0..180]."""
        ang_hour, ang_min = self.anglesFromClock(hour=hour, minute=minute)
        d = abs(float(ang_hour) - float(ang_min)) % 360.0
        if d > 180.0:
            d = 360.0 - d
        return float(d)

    def deltaAngleFromDicoCell(self,*, row: int, col: int, nbMotsMax: int, rowTitles: Optional[List[Any]] = None, word: str = "", mode: str = "delta") -> float:
        """Calcul complet: cellule dico -> hour/minute -> delta angle (0..180).

        Note:
        - En DELTA (10 énigmes projetées sur un cadran 12h), une même ligne peut
          correspondre à 2 heures possibles (ex: 1h ou 11h). Dans ce cas, on
          renvoie l'angle MIN (utile pour le filtrage par angle).
        - En ABS, la projection est unique.
        """
        m = str(mode or "delta").strip().lower()
        st = self.clockStateFromDicoCell(
            row=int(row),
            col=int(col),
            nbMotsMax=int(nbMotsMax),
            rowTitles=rowTitles,
            word=word,
            mode=m,
        )

        if m.startswith("del"):
            hBase = max(1, int(self.getHoursBase()))
            hour0 = float(st.hour) % float(hBase)
            minute0 = int(st.minute)
            angA = self.deltaAngleBetweenHands(hour=hour0, minute=minute0)
            if hBase == 12:
                # Cas historique : 10 lignes projetées sur 12h -> ambiguité h vs h+10
                angB = self.deltaAngleBetweenHands(hour=(hour0 + 10.0) % 12.0, minute=minute0)
                return float(min(angA, angB))
            return float(angA)

        return self.deltaAngleBetweenHands(hour=float(st.hour), minute=int(st.minute))

    def clockStateFromDicoCell(self, *, row: int, col: int, nbMotsMax: int, rowTitles: Optional[List[Any]] = None, word: str = "", mode: str = "delta") -> ClockState:
        """Convertit une cellule (row,col) en état d'horloge.
        Par défaut: non supporté.
        """
        raise NotImplementedError

    def dicoCellFromClock(self, *, hour: int, minute: int, nbMotsMax: int, rowTitles: Optional[List[Any]] = None) -> Optional[Tuple[int, int]]:
        """Convertit un état (hour,minute) en (row,col) si possible.

        Par défaut: non supporté.
        """
        return None

class ClockDicoDecryptor(DecryptorBase):
    """Décryptage Horloge ↔ Dictionnaire.

    Modes supportés:
      - ABS:
        * row = 1..10 (pas de 0)
        * col = … -2, -1, 1, 2, … (pas de 0)
        * mapping compas: hour=row (1..10), minute=col en base 1
          (col=-1 => 60, col=-2 => 59, etc.)

      - DELTA:
        * row = 0..9 (0 autorisé), col = delta signé (0 autorisé)
        * mapping compas: hour=row, minute=col mod 60  (ex: -5 => 55')
        * Pour le filtrage d'angle, la ligne a 2 interprétations horaires
          possibles (h et h+10 mod 12) — géré dans deltaAngleFromDicoCell.
    """
    id = "clock_dico_v1"
    label = "Horloge ↔ Dictionnaire (v1)"
 
    def __init__(self):
        super().__init__()
        # Paramètres génériques (communs pour l’instant)
        self.hourMovesWithMinutes = True
        # Bases du cadran
        self.minutesBase: int = 60
        self.hoursBase: int = 12

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

    def clockStateFromDicoCell(self, *, row: int, col: int, nbMotsMax: int, rowTitles: Optional[List[Any]] = None, word: str = "", mode: str = "delta") -> ClockState:
        r = int(row)
        c = int(col)
        nbm = max(0, int(nbMotsMax))
        m = str(mode or "delta").strip().lower()

        hBase = max(1, int(self.getHoursBase()))
        mBase = max(1, int(self.getMinutesBase()))

        if m.startswith("abs"):
            # --- ABS ---
            rowDisp = ((int(r) - 1) % 10) + 1
            hourDisp = int(rowDisp)
            hour = int(hourDisp) % int(hBase)

            # col: pas de 0. Convertir en minute 1..60.
            if int(c) > 0:
                minute = int(c)
            else:
                minute = int(mBase) + int(c) + 1
            minute = int(minute) % int(mBase)
            if minute == 0:
                minute = int(mBase)

        else:
            # --- DELTA ---
            hour = int(r) % int(hBase)
            hourDisp = int(hour)
            # En DELTA, une colonne négative se lit comme une minute "avant":
            #   -5 => 55' (comme 10h - 5' = 9h55)
            # Donc: minute = col mod 60 (0..59)
            minute = int(c) % int(mBase)

        # Option : l’aiguille des heures avance avec les minutes
        if self.hourMovesWithMinutes:
            minuteFloat = 0.0 if int(minute) == int(mBase) else float(minute)
            hourFloat = (float(hour) + minuteFloat / float(mBase)) % float(hBase)
        else:
            hourFloat = float(hour)
 
        # Label
        w = str(word or "").strip()
        if w:
            label = f"{w} — ({hourDisp}h, {minute}')"
        else:
            label = f"({hourDisp}h, {minute}')"
 
        return ClockState(
            hour=float(hourFloat),
            minute=int(minute),
            label=label,
            dicoRow=r,
            dicoCol=c,
            word=w,
        )
 
    def dicoCellFromClock(
        self,
        *,
        hour: int,
        minute: int,
        nbMotsMax: int,
        rowTitles: Optional[List[Any]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Version simple (WIP):
 
        - row: on cherche la première ligne dont le titre commence par {hour}
        - col: nbMotsMax ± minute (2 candidats)
 
        Renvoie None si impossible.
        """
        h = int(hour) % 12
        m = max(0, int(minute))
        nbm = max(0, int(nbMotsMax))
 
        # 1) résoudre row via rowTitles si dispo
        if not rowTitles:
            return None
 
        targetRow = None
        for i, t in enumerate(rowTitles):
            s = str(t).strip()
            if s.isdigit() and int(str(int(s))[:1]) % 12 == h:
                targetRow = i
                break

        if targetRow is None:
            return None
 
        # 2) col: deux possibilités symétriques
        #    (à trancher via la présence du mot / heuristique plus tard)
        colA = nbm + m
        colB = nbm - m
        # Par défaut, renvoyer colA (même côté que l’implémentation habituelle “+”)
        return (int(targetRow), int(colA))
 
 
# Petit registre (optionnel) pour brancher d’autres décryptages
DECRYPTORS: Dict[str, Type[DecryptorBase]] = {
    ClockDicoDecryptor.id: ClockDicoDecryptor,
}
 

class AlgorithmeAssemblage:
    """Contrat minimal pour un algo d'assemblage automatique."""
    id: str = "algo_base"
    label: str = "Algorithme (base)"

    def __init__(self, engine: "MoteurSimulationAssemblage"):
        self.engine = engine

    def run(self, tri_ids: List[int]) -> List["ScenarioAssemblage"]:
        """Lance la simulation et retourne une liste de scénarios."""
        raise NotImplementedError

def createTopoQuadrilateral(
    *,
    world: TopologyWorld,
    topoScenarioId: str,
    triOddId: int,
    triEvenId: int,
    tOdd: dict,
    tEven: dict,
    Podd_local: Dict[str, np.ndarray],
    Peven_local: Dict[str, np.ndarray],
    Podd_world: Dict[str, np.ndarray],
    Peven_world: Dict[str, np.ndarray],
    entryOdd: dict | None = None,
    entryEven: dict | None = None,
    tol_rel: float = 1e-3,
    eps_world: float = 1e-6,
) -> tuple[str, str, str, str, str]:
    """
    Crée un quadrilatère topologique cohérent (toujours) :
      - 2 TopologyElement (odd/even) dans `world`
      - pose des deux éléments via setElementPoseFromWorldPts()
      - attachement interne edge-edge (arête commune détectée en local)
      - commit topo
      - retourne (topoGroupId, elementIdOdd, elementIdEven, src_edge, dst_edge)

    entryOdd/entryEven (si fournis) reçoivent:
      - topoElementId
      - topoGroupId (après commit)
    """

    if world is None:
        raise ValueError("createTopoQuadrilateral: world manquant")
    topoScenarioId = str(topoScenarioId or "").strip()
    if not topoScenarioId:
        raise ValueError("createTopoQuadrilateral: topoScenarioId manquant")

    # --- helpers locaux ---
    def _edge_len(P: Dict[str, np.ndarray], a: str, b: str) -> float:
        v = np.array(P[b], float) - np.array(P[a], float)
        return float(np.hypot(v[0], v[1]))

    def _edge_code(a: str, b: str) -> str | None:
        s = {a, b}
        if s == {"O", "B"}: return "OB"
        if s == {"B", "L"}: return "BL"
        if s == {"L", "O"}: return "LO"
        return None

    def _ensure_element_from_local(
        *,
        elementId: str,
        triId: int,
        pts_local: Dict[str, np.ndarray],
        labels: tuple | list | None,
        mirrored: bool,
    ) -> None:
        if str(elementId) in world.elements:
            return
        v_labels = list(labels or ("O", "B", "L"))
        v_types = [TopologyNodeType.OUVERTURE, TopologyNodeType.BASE, TopologyNodeType.LUMIERE]
        edge_lengths = [
            _edge_len(pts_local, "O", "B"),
            _edge_len(pts_local, "B", "L"),
            _edge_len(pts_local, "L", "O"),
        ]
        orient = "CW" if bool(mirrored) else "CCW"
        el = TopologyElement(
            element_id=str(elementId),
            name=f"Triangle {int(triId):02d}",
            vertex_labels=v_labels,
            vertex_types=v_types,
            edge_lengths_km=edge_lengths,
            meta={"orient": orient},
        )
        # IMPORTANT: crée un nouveau groupe topo pour cet élément
        world.add_element_as_new_group(el)

    # --- 1) IDs éléments (déterministes) ---
    elementIdOdd = TopologyWorld.format_element_id(topoScenarioId, int(triOddId))
    elementIdEven = TopologyWorld.format_element_id(topoScenarioId, int(triEvenId))

    # --- 2) Créer les 2 éléments (si absents) ---
    _ensure_element_from_local(
        elementId=elementIdOdd,
        triId=int(triOddId),
        pts_local={k: np.array(Podd_local[k], dtype=float) for k in ("O", "B", "L")},
        labels=tOdd.get("labels"),
        mirrored=bool(tOdd.get("mirrored", False)),
    )
    _ensure_element_from_local(
        elementId=elementIdEven,
        triId=int(triEvenId),
        pts_local={k: np.array(Peven_local[k], dtype=float) for k in ("O", "B", "L")},
        labels=tEven.get("labels"),
        mirrored=bool(tEven.get("mirrored", False)),
    )

    # --- 3) Poser les 2 éléments (monde) ---
    setElementPoseFromWorldPts(world, elementIdOdd, Podd_world, mirrored=False)
    setElementPoseFromWorldPts(world, elementIdEven, Peven_world, mirrored=False)

    # Injecter topoElementId dans les entrées graphiques si fournies
    if entryOdd is not None:
        entryOdd["topoElementId"] = elementIdOdd
    if entryEven is not None:
        entryEven["topoElementId"] = elementIdEven

    # --- 4) Détecter l’arête commune (en local) ---
    edges = [("O", "B"), ("O", "L"), ("B", "L")]
    best = None  # (rel, (aO,bO), (aE,bE))
    for aO, bO in edges:
        lO = _edge_len(Podd_local, aO, bO)
        for aE, bE in edges:
            lE = _edge_len(Peven_local, aE, bE)
            rel = abs(lO - lE) / max(lO, lE, 1e-9)
            if rel <= float(tol_rel) and (best is None or rel < best[0]):
                best = (rel, (aO, bO), (aE, bE))

    if best is None:
        raise ValueError("createTopoQuadrilateral: aucune arête commune détectée (tol_rel)")

    _, (aO, bO), (aE, bE) = best
    src_edge = _edge_code(aO, bO)
    dst_edge = _edge_code(aE, bE)
    if not src_edge or not dst_edge:
        raise ValueError("createTopoQuadrilateral: edge code invalide")

    # --- 5) Déterminer l’ordre des endpoints côté EVEN via les positions monde ---
    # Objectif: que Peven_world[dst_a] corresponde à Podd_world[aO] (à eps_world près)
    dst_a = aE
    dst_b = bE
    if np.linalg.norm(np.array(Peven_world[dst_a], float) - np.array(Podd_world[aO], float)) <= float(eps_world):
        pass
    elif np.linalg.norm(np.array(Peven_world[dst_b], float) - np.array(Podd_world[aO], float)) <= float(eps_world):
        dst_a, dst_b = dst_b, dst_a
    # sinon: on garde l'ordre, mais l'attachment edge-edge décidera mapping via vkeys (direct/reverse)

    # --- 6) Créer + commit l’attachement interne edge-edge ---
    epts = EdgeChoiceEpts(
        Podd_world[aO], Podd_world[bO],
        Peven_world[dst_a], Peven_world[dst_b],
        src_owner_tid=int(triOddId), src_edge=src_edge,
        dst_owner_tid=int(triEvenId), dst_edge=dst_edge,
        src_vkey_at_mA=aO, src_vkey_at_mB=bO,
        dst_vkey_at_tA=dst_a, dst_vkey_at_tB=dst_b,
        kind="edge-edge",
        elementIdSrc=elementIdOdd,
        elementIdDst=elementIdEven,
    )

    atts = epts.createTopologyAttachments(world=world)
    if not atts:
        raise ValueError("createTopoQuadrilateral: attachments edge-edge introuvables")

    world.beginTopoTransaction()
    world.apply_attachments(atts)
    world.commitTopoTransaction()

    # --- 7) Groupe topo résultant (odd/even doivent maintenant être dans le même groupe) ---
    topoGroupId = world.get_group_of_element(elementIdOdd)

    if entryOdd is not None:
        entryOdd["topoGroupId"] = topoGroupId
    if entryEven is not None:
        entryEven["topoGroupId"] = topoGroupId

    return (str(topoGroupId), str(elementIdOdd), str(elementIdEven), str(src_edge), str(dst_edge))

def setElementPoseFromWorldPts(
    world: TopologyWorld,
    elementId: str,
    Pw: dict,
    mirrored: bool = False,
) -> None:
    eps = 1e-12
    if world is None or str(elementId) not in world.elements:
        raise ValueError(f"setElementPoseFromWorldPts: elementId inconnu: {elementId}")
    if not isinstance(Pw, dict):
        raise ValueError("setElementPoseFromWorldPts: Pw invalide")
    for k in ("O", "B", "L"):
        if k not in Pw:
            raise ValueError("setElementPoseFromWorldPts: Pw incomplet")

    el = world.elements[str(elementId)]
    pO = np.array(el.vertex_local_xy.get(0, (0.0, 0.0)), dtype=float)
    pB = np.array(el.vertex_local_xy.get(1, (0.0, 0.0)), dtype=float)
    pL = np.array(el.vertex_local_xy.get(2, (0.0, 0.0)), dtype=float)

    Ow = np.array(Pw["O"], dtype=float)
    Bw = np.array(Pw["B"], dtype=float)
    Lw = np.array(Pw["L"], dtype=float)
    if Ow.shape != (2,) or Bw.shape != (2,) or Lw.shape != (2,):
        raise ValueError("setElementPoseFromWorldPts: Pw dimension invalide")
    if not np.isfinite(Ow).all() or not np.isfinite(Bw).all() or not np.isfinite(Lw).all():
        raise ValueError("setElementPoseFromWorldPts: Pw non fini")

    if mirrored:
        M = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
        pO = (M @ pO)
        pB = (M @ pB)
        pL = (M @ pL)

    X = np.stack([pO, pB, pL], axis=0)
    Y = np.stack([Ow, Bw, Lw], axis=0)
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    if np.linalg.norm(Xc) <= eps or np.linalg.norm(Yc) <= eps:
        raise ValueError("setElementPoseFromWorldPts: points degeneres")
    H = Xc.T @ Yc
    U, _S, Vt = np.linalg.svd(H)
    R = (Vt.T @ U.T)
    if np.linalg.det(R) < 0.0:
        Vt[1, :] *= -1.0
        R = (Vt.T @ U.T)
    T = Y.mean(axis=0) - (R @ X.mean(axis=0))

    world.setElementPose(str(elementId), R=R, T=T, mirrored=bool(mirrored))

class AlgoQuadrisParPaires(AlgorithmeAssemblage):
    id = "quadris_par_paires"
    label = "Quadrilatères par paires (bases communes) [WIP]"

    def _fill_group_vkeys_from_geometry(last_drawn: list, groups: dict, Q: float = 1e-6):
        """
        Enrichit groups[*]["nodes"] en remplissant :
        - edge_in / edge_out : l'arête réellement partagée ("OB","BL","LO")
        # (legacy vkey_in/out supprimé : on ne stocke plus que edge_in/out)
        """
        def edgeFromVkeys(a, b):
            if not a or not b or a == b:
                return None
            s = {a, b}
            if s == {"O", "B"}: return "OB"
            if s == {"B", "L"}: return "BL"
            if s == {"L", "O"}: return "LO"
            return None

        def oppVertex(a, b):
            if not a or not b:
                return None
            for k in ("O", "B", "L"):
                if k != a and k != b:
                    return k
            return None

        def qpt(p):
            return (round(float(p[0]) / Q) * Q, round(float(p[1]) / Q) * Q)

        def seg_key(a, b):
            a2 = qpt(a); b2 = qpt(b)
            return (a2, b2) if a2 <= b2 else (b2, a2)

        def edges_of(P):
            return {
                "OB": seg_key(P["O"], P["B"]),
                "BL": seg_key(P["B"], P["L"]),
                "LO": seg_key(P["L"], P["O"]),
            }

        # arête -> sommet opposé
        for g in (groups or {}).values():
            nodes = (g or {}).get("nodes") or []
            if len(nodes) < 2:
                continue

            # reset soft (optionnel mais évite les restes)
            for nd in nodes:
                nd["edge_in"] = nd.get("edge_in") if nd.get("edge_in") is not None else None
                nd["edge_out"] = nd.get("edge_out") if nd.get("edge_out") is not None else None

            for i in range(len(nodes) - 1):
                a = nodes[i]
                b = nodes[i + 1]
                ia = int(a.get("tid"))
                ib = int(b.get("tid"))

                if not (0 <= ia < len(last_drawn) and 0 <= ib < len(last_drawn)):
                    continue

                Pa = (last_drawn[ia] or {}).get("pts") or {}
                Pb = (last_drawn[ib] or {}).get("pts") or {}
                if not all(k in Pa for k in ("O", "B", "L")):
                    continue
                if not all(k in Pb for k in ("O", "B", "L")):
                    continue

                ea = edges_of(Pa)
                eb = edges_of(Pb)

                shared_a = shared_b = None
                for name_a, seg_a in ea.items():
                    for name_b, seg_b in eb.items():
                        if seg_a == seg_b:
                            shared_a = name_a
                            shared_b = name_b
                            break
                    if shared_a:
                        break

                if not shared_a:
                    # pas d'arête partagée -> on ne remplit pas (normal pour des triangles juste "proches")
                    continue

                # --- NOUVEAU : stocker explicitement l'arête connectée (non ambigu : BL vs LO, etc.) ---
                a["edge_out"] = shared_a
                b["edge_in"] = shared_b


    def run(self, tri_ids: List[int]) -> List["ScenarioAssemblage"]:
        """Étape 1+2 :
        - Si n=2 : assemble uniquement le 1er couple (2 triangles) en un quadrilatère.
        - Si n>=4 : assemble le 1er couple (tri1,tri2), puis assemble le 2e couple (tri3,tri4)
          et pose le quad2 en le connectant au quad1 via L2↔L3 avec EXACTEMENT 2 essais :
            - aligner (L3→O3) sur (L2→O2)
            - aligner (L3→O3) sur (L2→B2)

        - Applique la convention: le triangle 1 est orienté pour que l'azimut O→L soit 0° (Nord, axe +Y).
        - Détecte automatiquement la longueur commune entre les 2 triangles (OB / OL / BL).
        - Tente 2 poses (direction directe / inversée) et conserve les poses sans chevauchement (shrink-only).
        """
        if not tri_ids or len(tri_ids) < 2:
            return []
        if (len(tri_ids) % 2) != 0:
            # On ne gère que des n pairs pour l'instant
            return []

        # --- DEBUG init ---
        engine = self.engine
        engine.debugReset(tri_ids)

        tri1_id = int(tri_ids[0])
        tri2_id = int(tri_ids[1])

        v = engine.viewer

        topoScenarioId = "SA_AUTO"
        topoWorld0 = TopologyWorld(topoScenarioId)

        t1 = engine.build_local_triangle(tri1_id)
        t2 = engine.build_local_triangle(tri2_id)

        # ---- 1) Orientation : OL ou BL au Nord (+Y) pour le 1er triangle
        P1 = {k: np.array(t1["pts"][k], dtype=float) for k in ("O","B","L")}

        edge = str(getattr(engine, "firstTriangleEdge", "OL") or "OL").upper().strip()
        src = "B" if edge == "BL" else "O"

        vOL = P1["L"] - P1[src]
        if float(np.hypot(vOL[0], vOL[1])) > 1e-12:
            cur = math.atan2(vOL[1], vOL[0])
            target = math.pi / 2.0  # Nord = +Y
            dtheta = target - cur
            c, s = math.cos(dtheta), math.sin(dtheta)
            R = np.array([[c, -s],[s, c]], dtype=float)
            pivot = P1[src]
            for k in ("O","B","L"):
                P1[k] = (R @ (P1[k] - pivot)) + pivot


        # ---- 2) Détection de l'arête commune (OB / OL / BL)
        def _edge_len(P: Dict[str,np.ndarray], a: str, b: str) -> float:
            vv = np.array(P[b], float) - np.array(P[a], float)
            return float(np.hypot(vv[0], vv[1]))

        edges = [("O","B"), ("O","L"), ("B","L")]
        e1 = [(a, b, _edge_len(P1, a, b)) for a, b in edges]

        P2_local = {k: np.array(t2["pts"][k], dtype=float) for k in ("O","B","L")}
        e2 = [(a, b, _edge_len(P2_local, a, b)) for a, b in edges]

        tol_rel = 1e-3
        best = None  # (rel, (a1,b1),(a2,b2))
        for a1, b1, l1 in e1:
            for a2, b2, l2 in e2:
                rel = abs(l1 - l2) / max(l1, l2, 1e-9)
                if rel <= tol_rel and (best is None or rel < best[0]):
                    best = (rel, (a1, b1), (a2, b2))

        if best is None:
            raise ValueError("Aucune arête commune détectée entre les 2 triangles (tolérance 0.1%).")

        (_, (a1, b1), (a2, b2)) = best

        def _edge_code(a: str, b: str) -> str | None:
            s = {a, b}
            if s == {"O", "B"}: return "OB"
            if s == {"B", "L"}: return "BL"
            if s == {"L", "O"}: return "LO"
            return None

        def _ensure_element_from_local(
            *,
            world: TopologyWorld,
            element_id: str,
            tri_id: int,
            pts_local: Dict[str, np.ndarray],
            labels: tuple | list,
            mirrored: bool,
        ) -> None:
            if element_id in world.elements:
                return
            v_labels = list(labels or ("O", "B", "L"))
            v_types = [TopologyNodeType.OUVERTURE, TopologyNodeType.BASE, TopologyNodeType.LUMIERE]
            edge_lengths = [
                _edge_len(pts_local, "O", "B"),
                _edge_len(pts_local, "B", "L"),
                _edge_len(pts_local, "L", "O"),
            ]
            orient = "CW" if bool(mirrored) else "CCW"
            el = TopologyElement(
                element_id=element_id,
                name=f"Triangle {int(tri_id):02d}",
                vertex_labels=v_labels,
                vertex_types=v_types,
                edge_lengths_km=edge_lengths,
                meta={"orient": orient},
            )
            world.add_element_as_new_group(el)

        def _assign_topo_element_id_to_last_drawn(
            *,
            topo_scenario_id: str,
            entry: dict,
        ) -> str:
            tri_id = int(entry.get("id"))
            elem_id = TopologyWorld.format_element_id(topo_scenario_id, tri_id)
            entry["topoElementId"] = elem_id
            return elem_id

        def _bootstrap_topo_first_pair(
            *,
            world: TopologyWorld,
            topoScenarioId: str,
            tri1_id: int,
            tri2_id: int,
            t1: dict,
            t2: dict,
            P1_local: dict,
            P2_local: dict,
            P1_world: dict,
            P2_world: dict,
            base_list: list,
        ):
            """
            Bootstrap du premier quadrilatère topo.
            Délègue intégralement à createTopoQuadrilateral().
            """

            # base_list[0] ↔ tri1, base_list[1] ↔ tri2
            entryOdd = base_list[0]
            entryEven = base_list[1]

            topoGroupId, elementIdOdd, elementIdEven, _, _ = createTopoQuadrilateral(
                world=world,
                topoScenarioId=topoScenarioId,
                triOddId=int(tri1_id),
                triEvenId=int(tri2_id),
                tOdd=t1,
                tEven=t2,
                Podd_local=P1_local,
                Peven_local=P2_local,
                Podd_world=P1_world,
                Peven_world=P2_world,
                entryOdd=entryOdd,
                entryEven=entryEven,
            )

            return topoGroupId


        # ---- 3) Pose du triangle 2 : 2 essais (direct / inversé)
        poly1 = _tri_shape(P1)

        poses: List[Dict[str,np.ndarray]] = []
        dbg_try = 0
        dbg_pose_exc = 0
        dbg_overlap = 0

        # Si on enchaîne (n>=4), on fige le 1er quad à la première pose valide
        max_poses_first_pair = 2 if len(tri_ids) <= 2 else 1        
        for (at, bt) in [(a1, b1), (b1, a1)]:  # direct puis inversé
            dbg_try += 1
            P2w = engine.pose_points_on_edge(
                Pm=P2_local, am=a2, bm=b2,
                Pt=P1, at=at, bt=bt,
                Vm=P2_local[a2], Vt=P1[at],
            )

            # Chevauchement : utiliser EXACTEMENT la règle partagée "shrink-only"
            poly2 = _tri_shape(P2w)
            if _overlap_shrink(
                poly2, poly1,
                getattr(v, "stroke_px", 2),
                engine.getOverlapZoomRef(),
            ):
                dbg_overlap += 1
                continue

            poses.append(P2w)

            if len(poses) >= max_poses_first_pair:
                break
        if not poses:
            engine.debugFail(
                step="pair1",
                pair=(tri1_id, tri2_id),
                reason="Aucune pose valide pour la première paire",
                detail=f"essais={dbg_try}, exceptions_pose={dbg_pose_exc}, prunes_overlap={dbg_overlap}",
            )
            return []

        # ---- 4) Si n=2 : même comportement qu'avant (1 ou 2 scénarios possibles)
        if len(tri_ids) <= 2:
            out: List[ScenarioAssemblage] = []
            poses_short = list(poses[:2])
            for i, P2 in enumerate(poses_short):
                scen = ScenarioAssemblage(
                    name=(f"#1" if i==0 else f"#{i+1}=#1+({tri2_id})"),
                    source_type="auto",
                    algo_id=self.id,
                    tri_ids=[tri1_id, tri2_id],
                )
                scen.status = "complete"
                scen.topoScenarioId = topoScenarioId

                last_drawn = []
                last_drawn.append({
                    "labels": t1.get("labels"),
                    "pts": P1,
                    "id": tri1_id,
                    "mirrored": bool(t1.get("mirrored", False)),
                    "group_id": 1,
                    "group_pos": 0,
                })
                last_drawn.append({
                    "labels": t2.get("labels"),
                    "pts": P2,
                    "id": tri2_id,
                    "mirrored": bool(t2.get("mirrored", False)),
                    "group_id": 1,
                    "group_pos": 1,
                })
                topoWorld_scen = TopologyWorld(topoScenarioId)
                _bootstrap_topo_first_pair(
                    world=topoWorld_scen,
                    topoScenarioId=topoScenarioId,
                    tri1_id=tri1_id,
                    tri2_id=tri2_id,
                    t1=t1,
                    t2=t2,
                    P1_local={k: np.array(t1["pts"][k], dtype=float) for k in ("O","B","L")},
                    P2_local={k: np.array(t2["pts"][k], dtype=float) for k in ("O","B","L")},
                    P1_world=P1,
                    P2_world=P2,
                    base_list=last_drawn,
                )
                scen.topoWorld = topoWorld_scen

                xs, ys = [], []
                for t in last_drawn:
                    for k in ("O","B","L"):
                        xs.append(float(t["pts"][k][0]))
                        ys.append(float(t["pts"][k][1]))
                bbox = (min(xs), min(ys), max(xs), max(ys))
                groups = {
                    1: {
                        "id": 1,
                        "nodes": [
                            {"tid": 0, "edge_in": None, "edge_out": None},
                            {"tid": 1, "edge_in": None, "edge_out": None},
                        ],
                        "bbox": bbox,
                    }
                }
                groups[1]["topoGroupId"] = last_drawn[0].get("topoGroupId")
                scen.last_drawn = last_drawn
                # Enrichit les nodes (vkey_in / vkey_out) en déduisant l’arête partagée via la géométrie
                AlgoQuadrisParPaires._fill_group_vkeys_from_geometry(last_drawn, groups, Q=1e-6)

            scen.groups = groups
            out.append(scen)
            return out

        # ---- 5) Étape 2 : chaîner les quadrilatères (tri3,tri4), (tri5,tri6), ... via les sommets Lumière
        # Convention : on connecte L(2) ↔ L(3), puis L(4) ↔ L(5), etc.
        # À chaque connexion : EXACTEMENT 2 essais (aligner (Lodd→Oodd) sur (Leven→Oeven) puis sur (Leven→Beven)).
        #
        # IMPORTANT :
        # - On fige l'assemblage interne de chaque paire (A,B) à la 1ère pose valide (shrink-only)
        #   pour éviter l'explosion combinatoire.
        # - Tous les triangles restent dans un SEUL groupe (group_id=1) afin que les tests de chevauchement
        #   et la manipulation de groupe restent cohérents.

        if len(tri_ids) < 4:
            return []

        # On fige le 1er quad sur la pose retenue (déjà pruné par overlap shrink-only)
        P2 = poses[0]

        # Fabrique un last_drawn "base" (quad1)
        base_last = [
            {
                "labels": t1.get("labels"),
                "pts": P1,
                "id": tri1_id,
                "mirrored": bool(t1.get("mirrored", False)),
                "group_id": 1,
                "group_pos": 0,
            },
            {
                "labels": t2.get("labels"),
                "pts": P2,
                "id": tri2_id,
                "mirrored": bool(t2.get("mirrored", False)),
                "group_id": 1,
                "group_pos": 1,
            },
        ]
        _assign_topo_element_id_to_last_drawn(topo_scenario_id=topoScenarioId, entry=base_last[0])
        _assign_topo_element_id_to_last_drawn(topo_scenario_id=topoScenarioId, entry=base_last[1])

        def _orient_O_to_L_north(P: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
            P = {k: np.array(P[k], dtype=float) for k in ("O","B","L")}
            vOL = P["L"] - P["O"]
            if float(np.hypot(vOL[0], vOL[1])) > 1e-12:
                cur = math.atan2(vOL[1], vOL[0])
                target = math.pi / 2.0
                dtheta = target - cur
                c, s = math.cos(dtheta), math.sin(dtheta)
                R = np.array([[c, -s],[s, c]], dtype=float)
                pivot = P["O"]
                for k in ("O","B","L"):
                    P[k] = (R @ (P[k] - pivot)) + pivot
            return P

        def _edge_len(P: Dict[str,np.ndarray], a: str, b: str) -> float:
            vv = np.array(P[b], float) - np.array(P[a], float)
            return float(np.hypot(vv[0], vv[1]))

        def _build_quad_local(triA_id: int, triB_id: int) -> Tuple[Dict, Dict, Dict[str,np.ndarray], Dict[str,np.ndarray]]:
            """Construit un quad (A,B) en repère local, en figeant B à la 1ère pose valide."""
            tA = engine.build_local_triangle(triA_id)
            tB = engine.build_local_triangle(triB_id)

            PA = _orient_O_to_L_north({k: np.array(tA["pts"][k], dtype=float) for k in ("O","B","L")})
            PB_local = {k: np.array(tB["pts"][k], dtype=float) for k in ("O","B","L")}

            edges = [("O","B"), ("O","L"), ("B","L")]
            eA = [(a, b, _edge_len(PA, a, b)) for a, b in edges]
            eB = [(a, b, _edge_len(PB_local, a, b)) for a, b in edges]

            tol_rel = 1e-3
            best = None  # (rel, (aA,bA),(aB,bB))
            for aA, bA, lA in eA:
                for aB, bB, lB in eB:
                    rel = abs(lA - lB) / max(lA, lB, 1e-9)
                    if rel <= tol_rel and (best is None or rel < best[0]):
                        best = (rel, (aA, bA), (aB, bB))

            if best is None:
                raise ValueError("Aucune arête commune détectée entre les 2 triangles (tolérance 0.1%).")

            (_, (aA, bA), (aB, bB)) = best

            polyA = _tri_shape(PA)
            PB = None
            # 2 essais internes (direct/inversé) MAIS on fige à la 1ère pose valide
            for (at, bt) in [(aA, bA), (bA, aA)]:
                PBw = engine.pose_points_on_edge(
                    Pm=PB_local, am=aB, bm=bB,
                    Pt=PA, at=at, bt=bt,
                    Vm=PB_local[aB], Vt=PA[at],
                )

                polyB = _tri_shape(PBw)
                if _overlap_shrink(
                    polyB, polyA,
                    getattr(v, "stroke_px", 2),
                    engine.getOverlapZoomRef(),
                ):
                    continue

                PB = PBw
                break

            if PB is None:
                raise ValueError("Aucune pose valide trouvée pour assembler le quadrilatère (A,B).")

            return tA, tB, PA, PB

        # État de recherche : liste de branches (scénarios partiels)
        # Chaque état = (last_drawn, poly_occ)
        poly_occ0 = _group_shape_from_nodes(
            [{"tid": 0}, {"tid": 1}],
            base_last
        )

        _bootstrap_topo_first_pair(
            world=topoWorld0,
            topoScenarioId=topoScenarioId,
            tri1_id=tri1_id,
            tri2_id=tri2_id,
            t1=t1,
            t2=t2,
            P1_local={k: np.array(t1["pts"][k], dtype=float) for k in ("O","B","L")},
            P2_local={k: np.array(t2["pts"][k], dtype=float) for k in ("O","B","L")},
            P1_world=P1,
            P2_world=P2,
            base_list=base_last,
        )
        @dataclass(eq=False)
        class _BranchNode:
            parent: Optional["_BranchNode"]
            children: List["_BranchNode"]
            branchTriId: Optional[int] = None

        rootNode = _BranchNode(parent=None, children=[], branchTriId=None)

        # État de recherche : liste de branches (scénarios partiels)
        # Chaque état = (node, last_drawn, poly_occ, topoWorld)
        states = [(rootNode, base_last, poly_occ0, topoWorld0)]

        # Boucle sur les paires suivantes : (tri3,tri4), (tri5,tri6), ...
        for pair_start in range(2, len(tri_ids), 2):
            if pair_start + 1 >= len(tri_ids):
                break

            tri_odd_id = int(tri_ids[pair_start])       # tri3, tri5, ...
            tri_even_id = int(tri_ids[pair_start + 1])  # tri4, tri6, ...

            # On construit le quad local dans l'ordre courant (odd->even)
            tOdd, tEven, Podd, Peven = _build_quad_local(tri_odd_id, tri_even_id)

            def tryChainConnect(tOdd, tEven, Podd, Peven, triOddId, triEvenId):
                new_states = []
                dbg_try = 0
                dbg_overlap = 0
                dbg_added = 0

                for (node_prev, last_drawn_prev, poly_occ_prev, topoWorld_prev) in states:
                    baseKey = getattr(node_prev, "debugKey", "")    # Une cle de debug pour tracer les scénarios    
                    candidates = []
                    mob_keys = ("O", "B")

                    # --- Phase 3 : création du quadrilatère topo mobile (UNE FOIS)
                    createTopoQuadrilateral(
                        world=topoWorld_prev,
                        topoScenarioId=topoScenarioId,
                        triOddId=tri_odd_id,
                        triEvenId=tri_even_id,
                        tOdd=tOdd,
                        tEven=tEven,
                        Podd_local={k: np.array(tOdd["pts"][k], float) for k in ("O","B","L")},
                        Peven_local={k: np.array(tEven["pts"][k], float) for k in ("O","B","L")},
                        Podd_world=Podd,
                        Peven_world=Peven,
                    )

                    # clone du last_drawn de la branche pour référencer les triangles temporaires
                    last_drawn_base = copy.deepcopy(last_drawn_prev)

                    pos0 = len(last_drawn_base)
                    last_drawn_base.append({
                        "labels": tOdd.get("labels"),
                        "pts": Podd,          # pose INITIALE (pas Podd_w)
                        "id": triOddId,
                        "mirrored": bool(tOdd.get("mirrored", False)),
                        "group_id": 1,
                        "group_pos": pos0,
                    })
                    _assign_topo_element_id_to_last_drawn(
                        topo_scenario_id=topoScenarioId,
                        entry=last_drawn_base[pos0],
                    )

                    pos1 = len(last_drawn_base)
                    last_drawn_base.append({
                        "labels": tEven.get("labels"),
                        "pts": Peven,         # pose INITIALE
                        "id": triEvenId,
                        "mirrored": bool(tEven.get("mirrored", False)),
                        "group_id": 1,
                        "group_pos": pos1,
                    })
                    _assign_topo_element_id_to_last_drawn(
                        topo_scenario_id=topoScenarioId,
                        entry=last_drawn_base[pos1],
                    )

                    # On teste sur les 4 cas LB-LB, LB-LO, LO-LB, LO - Lo
                    for mob_key in mob_keys:
                        for bt_key in ("O", "B"):
                            dbg_try += 1

                            # --- Phase 3.1 : construire EdgeChoiceEpts (sans décision topo, juste préparation)
                            # Convention auto actuelle : raccord via L, arêtes LO/BL.
                            # Détermination explicite des arêtes testées
                            anchor_edge = "LO" if bt_key == "O" else "BL"
                            odd_edge    = "LO" if mob_key == "O" else "BL"


                            edgeChoiceEpts, edgeChoiceMeta = buildEdgeChoiceEptsForAutoChain(
                                world=topoWorld_prev,
                                last_drawn_base=last_drawn_base,
                                pos_mobile=pos0,                        # mobile = triangle odd
                                pos_dest=len(last_drawn_prev) - 1,      # destination = dernier triangle du groupe existant
                                src_edge=odd_edge,                      # arêtes explicites
                                dst_edge=anchor_edge,
                                src_vkey="L",                           # sommet d’ancrage (toujours L en phase 3)
                                dst_vkey="L",
                                kind="vertex-edge",
                                debug=False,
                            )

                            topoAttachments = edgeChoiceEpts.createTopologyAttachments(world=topoWorld_prev)

                            # On teste l'overlap . 
                            # group ids depuis les elementIds
                            gidDest = topoWorld_prev.get_group_of_element(last_drawn_prev[-1]["topoElementId"])
                            gidMob  = topoWorld_prev.get_group_of_element(last_drawn_base[pos0]["topoElementId"])              
                            # simulateOverlapTopologique attend des TopologyGroup (pas des str)
                            gDest = topoWorld_prev.groups[topoWorld_prev.find_group(gidDest)]
                            gMob  = topoWorld_prev.groups[topoWorld_prev.find_group(gidMob)]

                            overlap = topoWorld_prev.simulateOverlapTopologique(gDest, gMob, topoAttachments, debug=False)  
                            if overlap:
                                dbg_overlap += 1
                                continue                                

                            # Pour les cas validés, on récupère la transformation R et T 
                            # que l'on applique au 2 triangles graphiques
                            def applyRT(P, R, T):
                                return {k: (R @ np.array(P[k], float) + T) for k in ("O","B","L")}
  
                            R, T = edgeChoiceEpts.computeRigidTransform()
                            Podd_w  = applyRT(Podd,  R, T)
                            Peven_w = applyRT(Peven, R, T)

                            poly_new = _group_shape_from_nodes(
                                [{"tid": 0}, {"tid": 1}],
                                [{"pts": Podd_w}, {"pts": Peven_w}]
                            )

                            last_drawn_new = copy.deepcopy(last_drawn_prev)

                            # --- mémoriser la jonction entre quads (connexion via L)
                            # bt_key = côté choisi sur l'ancre (L->O ou L->B)
                            # mob_key = côté choisi sur le triangle odd (L->O ou L->B)
                            # On encode l'arête incidente à L : "LO" ou "BL"
                            anchor_edge = "LO" if bt_key == "O" else "BL"
                            odd_edge = "LO" if mob_key == "O" else "BL"
                            last_drawn_new[-1]["_chain_edge_out"] = anchor_edge

                            pos0 = len(last_drawn_new)
                            last_drawn_new.append({
                                "labels": tOdd.get("labels"),
                                "pts": Podd_w,
                                "id": triOddId,
                                "mirrored": bool(tOdd.get("mirrored", False)),
                                "group_id": 1,
                                "group_pos": pos0,
                            })
                            elem_id_odd = _assign_topo_element_id_to_last_drawn(
                                topo_scenario_id=topoScenarioId,
                                entry=last_drawn_new[pos0],
                            )
                            last_drawn_new[pos0]["_chain_edge_in"] = odd_edge
                            pos1 = len(last_drawn_new)
                            last_drawn_new.append({
                                "labels": tEven.get("labels"),
                                "pts": Peven_w,
                                "id": triEvenId,
                                "mirrored": bool(tEven.get("mirrored", False)),
                                "group_id": 1,
                                "group_pos": pos1,
                            })
                            elem_id_even = _assign_topo_element_id_to_last_drawn(
                                topo_scenario_id=topoScenarioId,
                                entry=last_drawn_new[pos1],
                            )
                            # décision de raccord: anchor_edge (côté ancre), odd_edge (côté mobile odd)
                            candKey = f"{baseKey}|{int(triOddId)}:{odd_edge}->{anchor_edge}"

                            candidates.append((
                                last_drawn_new,
                                poly_occ_prev.union(poly_new),
                                {
                                    "odd": {
                                        "element_id": elem_id_odd,
                                        "tri_id": int(triOddId),
                                        "pts_world": Podd_w,
                                        "labels": tOdd.get("labels"),
                                        "mirrored": bool(tOdd.get("mirrored", False)),
                                        "pts_local": Podd,
                                    },
                                    "even": {
                                        "element_id": elem_id_even,
                                        "tri_id": int(triEvenId),
                                        "pts_world": Peven_w,
                                        "labels": tEven.get("labels"),
                                        "mirrored": bool(tEven.get("mirrored", False)),
                                        "pts_local": Peven,
                                    },
                                    "edgeChoice": {
                                        "ok": bool(edgeChoiceEpts),
                                        "meta": edgeChoiceMeta,
                                        "topoAccepted": True,
                                    },
                                    "topoAttachments": topoAttachments,
                                    "topoAnchorElemId": last_drawn_prev[-1]["topoElementId"],
                                    "topoMobileElemId": last_drawn_base[pos0]["topoElementId"],
                                    "debugKey": candKey,
                                },
                            ))
                            dbg_added += 1

                    # Si au moins 2 candidats existent *à cette étape*, on enregistre une bifurcation.
                    # La bifurcation ne devient "réelle" que si les 2 sous-branches mènent à des feuilles survivantes,
                    # ce qui sera résolu après pruning (sur l'arbre survivant).
                    if candidates:
                        node_prev.children = []
                        if len(candidates) >= 2:
                            # IMPORTANT (naming): la bifurcation correspond au triangle
                            # connecté au bloc précédent via le point de Lumière,
                            # c'est triOddId (triEvenId est "collé" au triOddId via BO).
                            node_prev.branchTriId = int(triOddId)
                        else:
                            node_prev.branchTriId = None

                        if len(candidates) >= 2:
                            topo_candidates = [topoWorld_prev.clonePhysicalState() for _ in candidates]
                        else:
                            topo_candidates = [topoWorld_prev for _ in candidates]

                        for (ld_new, poly_u, topo_meta), topo_new in zip(candidates, topo_candidates):
                            for _key in ("odd", "even"):
                                _info = topo_meta.get(_key, {})
                                _elem_id = _info.get("element_id")
                                _tri_id = _info.get("tri_id")
                                _pts_local = _info.get("pts_local")
                                if _elem_id and _tri_id and _pts_local is not None:
                                    _ensure_element_from_local(
                                        world=topo_new,
                                        element_id=str(_elem_id),
                                        tri_id=int(_tri_id),
                                        pts_local={k: np.array(_pts_local[k], dtype=float) for k in ("O", "B", "L")},
                                        labels=_info.get("labels"),
                                        mirrored=bool(_info.get("mirrored", False)),
                                    )
                                if _elem_id and _info.get("pts_world") is not None:
                                    # Pose from world points in simulation uses non-mirrored fit.
                                    setElementPoseFromWorldPts(
                                        topo_new,
                                        str(_elem_id),
                                        _info.get("pts_world"),
                                        mirrored=False,
                                    )

                            # On raccroche les 2 groupes via l'attachement
                            atts = topo_meta.get("topoAttachments")
                            if atts:
                                topo_new.beginTopoTransaction()
                                topo_new.apply_attachments(atts)
                                topo_new.commitTopoTransaction()                            

                            child = _BranchNode(parent=node_prev, children=[], branchTriId=None)
                            node_prev.children.append(child)
                            child.debugKey = topo_meta.get("debugKey", baseKey)  # <= AJOUT
                            new_states.append((child, ld_new, poly_u, topo_new))

                return new_states, dbg_try, dbg_overlap, dbg_added

            # 1) Tentative standard : mobile = odd (tri_odd_id)
            new_states, dbg_try, dbg_overlap, dbg_added = tryChainConnect(
                tOdd, tEven, Podd, Peven, tri_odd_id, tri_even_id
            )

            # 2) Fallback minimal : si échec, on retente en inversant odd/even
            #    (mobile = tri_even_id). Ça corrige la dépendance implicite à l'ordre.
            if not new_states:
                tOdd2, tEven2, Podd2, Peven2 = _build_quad_local(tri_even_id, tri_odd_id)
                new_states2, dbg_try2, dbg_overlap2, dbg_added2 = tryChainConnect(
                    tOdd2, tEven2, Podd2, Peven2, tri_even_id, tri_odd_id
                )
                if new_states2:
                    new_states = new_states2
                    dbg_try += dbg_try2
                    dbg_overlap += dbg_overlap2
                    dbg_added += dbg_added2

            states = new_states
            if not states:
                engine.debugFail(
                    step="chain_connect",
                    pair=(tri_odd_id, tri_even_id),
                    reason="Aucune connexion valide (chaînage)",
                    detail=f"essais={dbg_try}, prunes_overlap={dbg_overlap}, ajoutés={dbg_added}",
                )
                return []

        # --- Post-traitement : construire une numérotation COHÉRENTE sur l'arbre survivant ---
        # Objectif : pouvoir "pruner mentalement" par plages (#1..#96 / #97..#117, etc.).
        leafData = {node: (last_drawn, _poly_occ, topoWorld) for (node, last_drawn, _poly_occ, topoWorld) in states}

        kept = set()
        for leaf in leafData.keys():
            n = leaf
            while n is not None and n not in kept:
                kept.add(n)
                n = n.parent

        def _keptChildren(n):
            return [c for c in (n.children or []) if c in kept]

        # Collecte des feuilles survivantes dans l'ordre gauche→droite (DFS)
        leaves = []
        def _collectLeaves(n):
            if n not in kept:
                return
            ch = _keptChildren(n)
            if not ch:
                leaves.append(n)
                return
            for c in ch:
                _collectLeaves(c)

        _collectLeaves(rootNode)

        leafIndex = {leaf: (i + 1) for i, leaf in enumerate(leaves)}

        def _leftMostLeaf(n):
            cur = n
            while True:
                ch = _keptChildren(cur)
                if not ch:
                    return cur
                cur = ch[0]

        # Par défaut : "#k"
        labels = {leaf: f"#{leafIndex[leaf]}" for leaf in leaves}

        # Pour chaque bifurcation survivante, on étiquette UNIQUEMENT le "start" du sous-arbre droit :
        #   #startR = #startL + (triEvenId)
        for n in list(kept):
            ch = _keptChildren(n)
            if n.branchTriId is None:
                continue
            if len(ch) < 2:
                continue
            leftLeaf = _leftMostLeaf(ch[0])
            idxL = leafIndex.get(leftLeaf)
            if idxL is None:
                continue
            for j in range(1, len(ch)):
                rightLeaf = _leftMostLeaf(ch[j])
                idxR = leafIndex.get(rightLeaf)
                if idxR is None:
                    continue
                labels[rightLeaf] = f"#{idxR}=#{idxL}+({int(n.branchTriId)})"

        # Finalisation : créer les scénarios complets
        out: List[ScenarioAssemblage] = []
        for leaf in leaves:
            (last_drawn, _poly_occ, topoWorld_leaf) = leafData[leaf]
            idx = int(leafIndex.get(leaf, 0) or 0)
            scen = ScenarioAssemblage(
                name=labels.get(leaf, f"#{idx}"),
                source_type="auto",
                algo_id=self.id,
                tri_ids=[int(x) for x in tri_ids[:len(last_drawn)]],
            )
            scen.status = "complete"
            scen.last_drawn = last_drawn
            scen.topoWorld = topoWorld_leaf
            scen.topoScenarioId = topoScenarioId

            # Groupe unique
            idxs = list(range(len(last_drawn)))
            xs, ys = [], []
            for k in idxs:
                t = last_drawn[k]
                for vkey in ("O", "B", "L"):
                    xs.append(float(t["pts"][vkey][0]))
                    ys.append(float(t["pts"][vkey][1]))
            bbox = (min(xs), min(ys), max(xs), max(ys))
            groups = {
                1: {"id": 1, "nodes": [{"tid": k, "edge_in": None, "edge_out": None} for k in idxs], "bbox": bbox},
            }
            # Enrichit les nodes (vkey_in / vkey_out) en déduisant l’arête partagée via la géométrie
            AlgoQuadrisParPaires._fill_group_vkeys_from_geometry(last_drawn, groups, Q=1e-6)

            # --- NOUVEAU : injecter la jonction "chaîne" (L->O vs L->B) dans les nodes
            # Ici, ce n'est PAS une arête partagée géométriquement, donc _fill_group... ne peut pas la trouver.
            vkey_from_edge = {"OB": "L", "BL": "O", "LO": "B"}
            nodes = groups[1]["nodes"]
            for k in idxs:
                t = last_drawn[k]
                if "_chain_edge_out" in t:
                    eout = t.get("_chain_edge_out")
                    if eout:
                        nodes[k]["edge_out"] = eout
                        nodes[k]["vkey_out"] = vkey_from_edge.get(eout)
                if "_chain_edge_in" in t:
                    ein = t.get("_chain_edge_in")
                    if ein:
                        nodes[k]["edge_in"] = ein
                        nodes[k]["vkey_in"] = vkey_from_edge.get(ein)

            scen.groups = groups
            out.append(scen)

        return out


# Registre des algorithmes disponibles pour la boîte de dialogue.
# IMPORTANT : conserver l'ordre pour un affichage stable dans la combo.
ALGOS: Dict[str, Type[AlgorithmeAssemblage]] = {
    AlgoQuadrisParPaires.id: AlgoQuadrisParPaires,
}


class MoteurSimulationAssemblage:
    """Wrapper 'pur data' autour des briques géométriques du viewer (manuel)."""

    def __init__(self, viewer: "TriangleViewerManual"):
        self.viewer = viewer
        self.firstTriangleEdge = "OL"
        # --- DEBUG (instrumentation minimaliste, sans console spam) ---
        # Rempli par les algos si un run() échoue et retourne [].
        self.debug_last: Dict | None = None

        # IMPORTANT: la simulation ne doit pas dépendre du zoom écran courant.
        # On fige un zoom de référence, clampé, utilisé uniquement pour _overlap_shrink().
        z = float(getattr(viewer, "simulationOverlapZoomRef", getattr(viewer, "zoom", 1.0)))

        # borne haute: évite shrink trop petit => faux chevauchement sur triangles collés
        self.overlapZoomRef = min(max(z, 0.25), 8.0)

    # --- DEBUG helpers ---
    def debugReset(self, tri_ids: List[int] | None = None):
        self.debug_last = {
            "tri_ids": list(tri_ids or []),
            "step": None,
            "pair": None,
            "anchor": None,
            "reason": None,
            "detail": None,
        }

    def debugFail(self, step: str, pair=None, reason: str | None = None, detail: str | None = None, anchor=None):
        if self.debug_last is None:
            self.debugReset([])
        self.debug_last.update({
            "step": step,
            "pair": pair,
            "anchor": anchor,
            "reason": reason,
            "detail": detail,
        })

    def getOverlapZoomRef(self) -> float:
        return float(getattr(self, "overlapZoomRef", 1.0))

    def build_local_triangle(self, tri_id: int) -> Dict:
        """Construit un triangle en repère local à partir du DF (longueurs OB/OL/BL + orientation)."""
        v = self.viewer
        if getattr(v, "df", None) is None or v.df is None or v.df.empty:
            raise RuntimeError("Aucun fichier Triangle chargé (df vide).")

        row = v.df[v.df["id"] == int(tri_id)]
        if row.empty:
            raise ValueError(f"Triangle id={tri_id} introuvable dans la source.")
        r = row.iloc[0]
        P = _build_local_triangle(float(r["len_OB"]), float(r["len_OL"]), float(r["len_BL"]))

        # Orientation : si CW → symétrie verticale (y -> -y)
        ori = str(r.get("orient", "CCW")).upper().strip()

        mirrored = (ori == "CW")
        if mirrored:
            P = {
                "O": np.array([P["O"][0], -P["O"][1]], dtype=float),
                "B": np.array([P["B"][0], -P["B"][1]], dtype=float),
                "L": np.array([P["L"][0], -P["L"][1]], dtype=float),
            }

        return {
            "labels": ("Bourges", str(r["B"]), str(r["L"])),  # (O,B,L) labels
            "id": int(tri_id),
            "mirrored": mirrored,
            "pts": P,
        }

    def pose_points_on_edge(
        self,
        Pm: Dict[str, np.ndarray], am: str, bm: str,
        Pt: Dict[str, np.ndarray], at: str, bt: str,
        Vm: np.ndarray, Vt: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Pose Pm (mobile) : aligne am→bm sur at→bt, en collant le point Vm sur Vt."""
        R, T, pivot = _pose_params(Pm, am, bm, Vm, Pt, at, bt, Vt)
        return _apply_R_T_on_P(Pm, R, T, pivot)

    def check_overlap(self, tri_or_group_nodes: List[Dict], occupied_nodes: List[Dict]) -> bool:
        """Retourne True si chevauchement (overlap) après shrink, sinon False."""
        v = self.viewer
        if not occupied_nodes:
            return False
        last_drawn = getattr(v, "_last_drawn", []) or []
        poly_new = _group_shape_from_nodes(tri_or_group_nodes, last_drawn)
        poly_occ = _group_shape_from_nodes(occupied_nodes, last_drawn)
        if (poly_new is None) or (poly_occ is None):
            return False
        # même règle que le mode manuel : shrink-only basé sur stroke_px et zoom
        return _overlap_shrink(
            poly_new,
            poly_occ,
            getattr(v, "stroke_px", 2),
            max(self.getOverlapZoomRef(), 1e-9),
        )
