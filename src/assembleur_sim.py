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
)

 
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
    def anglesFromClock(self, *, hour: float, minute: int) -> Tuple[float, float]:
        """Retourne (angleHourDeg, angleMinuteDeg) dans [0..360).

        Convention:
          - 0° = 12h
          - sens horaire
        """
        h = float(hour) % 12.0
        m = int(minute) % 60
        ang_min = (m * 6.0) % 360.0
        ang_hour = (h * 30.0) % 360.0
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
            hour0 = float(st.hour) % 12.0
            minute0 = int(st.minute)
            angA = self.deltaAngleBetweenHands(hour=hour0, minute=minute0)
            angB = self.deltaAngleBetweenHands(hour=(hour0 + 10.0) % 12.0, minute=minute0)
            return float(min(angA, angB))

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

    def clockStateFromDicoCell(self, *, row: int, col: int, nbMotsMax: int, rowTitles: Optional[List[Any]] = None, word: str = "", mode: str = "delta") -> ClockState:
        r = int(row)
        c = int(col)
        nbm = max(0, int(nbMotsMax))
        m = str(mode or "delta").strip().lower()

        if m.startswith("abs"):
            # --- ABS ---
            rowDisp = ((int(r) - 1) % 10) + 1
            hourDisp = int(rowDisp)
            hour = int(hourDisp) % 12  # 12 -> 0 (position "12h")

            # col: pas de 0. Convertir en minute 1..60.
            if int(c) > 0:
                minute = int(c)
            else:
                minute = 60 + int(c) + 1
            minute = int(minute) % 60
            if minute == 0:
                minute = 60

        else:
            # --- DELTA ---
            hour = int(r) % 12
            hourDisp = int(hour)
            # En DELTA, une colonne négative se lit comme une minute "avant":
            #   -5 => 55' (comme 10h - 5' = 9h55)
            # Donc: minute = col mod 60 (0..59)
            minute = int(c) % 60

        # Option : l’aiguille des heures avance avec les minutes
        if self.hourMovesWithMinutes:
            minuteFloat = 0.0 if int(minute) == 60 else float(minute)
            hourFloat = (float(hour) + minuteFloat / 60.0) % 12.0
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
            try:
                s = str(t).strip()
                if s.isdigit() and int(str(int(s))[:1]) % 12 == h:
                    targetRow = i
                    break
            except Exception:
                continue
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

class AlgoQuadrisParPaires(AlgorithmeAssemblage):
    id = "quadris_par_paires"
    label = "Quadrilatères par paires (bases communes) [WIP]"

    def _fill_group_vkeys_from_geometry(last_drawn: list, groups: dict, Q: float = 1e-6):
        """
        Enrichit groups[*]["nodes"] en remplissant :
        - edge_in / edge_out : l'arête réellement partagée ("OB","BL","LO")
        - vkey_in / vkey_out : le SOMMET OPPOSÉ à l'arête partagée
          (car ton code de signature fait: O->BL, B->LO, L->OB).
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
        vkey_from_edge = {"OB": "L", "BL": "O", "LO": "B"}

        for g in (groups or {}).values():
            nodes = (g or {}).get("nodes") or []
            if len(nodes) < 2:
                continue

            # reset soft (optionnel mais évite les restes)
            for nd in nodes:
                nd["vkey_in"] = nd.get("vkey_in") if nd.get("vkey_in") is not None else None
                nd["vkey_out"] = nd.get("vkey_out") if nd.get("vkey_out") is not None else None
                nd["edge_in"] = nd.get("edge_in") if nd.get("edge_in") is not None else None
                nd["edge_out"] = nd.get("edge_out") if nd.get("edge_out") is not None else None

            for i in range(len(nodes) - 1):
                a = nodes[i]
                b = nodes[i + 1]
                try:
                    ia = int(a.get("tid"))
                    ib = int(b.get("tid"))
                except Exception:
                    continue
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

                # --- vkey = sommet opposé à l'arête partagée (pour compat avec l'ancien code) ---
                a["vkey_out"] = vkey_from_edge.get(shared_a)
                b["vkey_in"] = vkey_from_edge.get(shared_b)

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
            try:
                P2w = engine.pose_points_on_edge(
                    Pm=P2_local, am=a2, bm=b2,
                    Pt=P1, at=at, bt=bt,
                    Vm=P2_local[a2], Vt=P1[at],
                )
            except Exception:
                dbg_pose_exc += 1
                continue

            # Chevauchement : utiliser EXACTEMENT la règle partagée "shrink-only"
            try:
                poly2 = _tri_shape(P2w)
                if _overlap_shrink(
                    poly2, poly1,
                    getattr(v, "stroke_px", 2),
                    engine.getOverlapZoomRef(),
                ):
                    dbg_overlap += 1
                    continue
            except Exception:
                # En auto, si doute → on prune
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
            for i, P2 in enumerate(poses[:2]):
                scen = ScenarioAssemblage(
                    name=f"Auto quad (2 triangles){'' if len(poses)==1 else f' #{i+1}'}",
                    source_type="auto",
                    algo_id=self.id,
                    tri_ids=[tri1_id, tri2_id],
                )
                scen.status = "complete"

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
                            {"tid": 0, "vkey_in": None, "vkey_out": None},
                            {"tid": 1, "vkey_in": None, "vkey_out": None},
                        ],
                        "bbox": bbox,
                    }
                }
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
                try:
                    PBw = engine.pose_points_on_edge(
                        Pm=PB_local, am=aB, bm=bB,
                        Pt=PA, at=at, bt=bt,
                        Vm=PB_local[aB], Vt=PA[at],
                    )
                except Exception:
                    continue
                try:
                    polyB = _tri_shape(PBw)
                    if _overlap_shrink(
                        polyB, polyA,
                        getattr(v, "stroke_px", 2),
                        engine.getOverlapZoomRef(),
                    ):
                        continue
                except Exception:
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
        states = [(base_last, poly_occ0)]

        MAX_SCENARIOS = 200

        # Boucle sur les paires suivantes : (tri3,tri4), (tri5,tri6), ...
        for pair_start in range(2, len(tri_ids), 2):
            if pair_start + 1 >= len(tri_ids):
                break

            tri_odd_id = int(tri_ids[pair_start])       # tri3, tri5, ...
            tri_even_id = int(tri_ids[pair_start + 1])  # tri4, tri6, ...

            # On construit le quad local dans l'ordre courant (odd->even)
            try:
                tOdd, tEven, Podd, Peven = _build_quad_local(tri_odd_id, tri_even_id)
            except Exception:
                return []

            def tryChainConnect(tOdd, tEven, Podd, Peven, triOddId, triEvenId):
                new_states = []
                dbg_try = 0
                dbg_overlap = 0
                dbg_added = 0

                for (last_drawn_prev, poly_occ_prev) in states:
                    anchor = last_drawn_prev[-1]["pts"]
                    mob_keys = ("O", "B")

                    for mob_key in mob_keys:
                        for bt_key in ("O", "B"):
                            dbg_try += 1
                            R, T, pivot = _pose_params(
                                Podd, "L", mob_key, Podd["L"],
                                anchor, "L", bt_key, anchor["L"]
                            )
                            Podd_w = _apply_R_T_on_P(Podd, R, T, pivot)
                            Peven_w = _apply_R_T_on_P(Peven, R, T, pivot)

                            poly_new = _group_shape_from_nodes(
                                [{"tid": 0}, {"tid": 1}],
                                [{"pts": Podd_w}, {"pts": Peven_w}]
                            )
                            if _overlap_shrink(
                                poly_new, poly_occ_prev,
                                getattr(v, "stroke_px", 2),
                                engine.getOverlapZoomRef(),
                            ):
                                dbg_overlap += 1
                                continue

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

                            new_states.append((last_drawn_new, poly_occ_prev.union(poly_new)))
                            dbg_added += 1

                return new_states, dbg_try, dbg_overlap, dbg_added

            # 1) Tentative standard : mobile = odd (tri_odd_id)
            new_states, dbg_try, dbg_overlap, dbg_added = tryChainConnect(
                tOdd, tEven, Podd, Peven, tri_odd_id, tri_even_id
            )

            # 2) Fallback minimal : si échec, on retente en inversant odd/even
            #    (mobile = tri_even_id). Ça corrige la dépendance implicite à l'ordre.
            if not new_states:
                try:
                    tOdd2, tEven2, Podd2, Peven2 = _build_quad_local(tri_even_id, tri_odd_id)
                    new_states2, dbg_try2, dbg_overlap2, dbg_added2 = tryChainConnect(
                        tOdd2, tEven2, Podd2, Peven2, tri_even_id, tri_odd_id
                    )
                    if new_states2:
                        new_states = new_states2
                        dbg_try += dbg_try2
                        dbg_overlap += dbg_overlap2
                        dbg_added += dbg_added2
                except Exception:
                    pass

            states = new_states
            if not states:
                engine.debugFail(
                    step="chain_connect",
                    pair=(tri_odd_id, tri_even_id),
                    reason="Aucune connexion valide (chaînage)",
                    detail=f"essais={dbg_try}, prunes_overlap={dbg_overlap}, ajoutés={dbg_added}",
                )
                return []

        # Finalisation : créer les scénarios complets
        out: List[ScenarioAssemblage] = []
        for i, (last_drawn, _poly_occ) in enumerate(states):
            scen = ScenarioAssemblage(
                name=f"Auto quadris ({len(last_drawn)} triangles)#{i+1}",
                source_type="auto",
                algo_id=self.id,
                tri_ids=[int(x) for x in tri_ids[:len(last_drawn)]],
            )
            scen.status = "complete"
            scen.last_drawn = last_drawn

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
                1: {"id": 1, "nodes": [{"tid": k, "vkey_in": None, "vkey_out": None} for k in idxs], "bbox": bbox},
            }
            # Enrichit les nodes (vkey_in / vkey_out) en déduisant l’arête partagée via la géométrie
            AlgoQuadrisParPaires._fill_group_vkeys_from_geometry(last_drawn, groups, Q=1e-6)

            # --- NOUVEAU : injecter la jonction "chaîne" (L->O vs L->B) dans les nodes
            # Ici, ce n'est PAS une arête partagée géométriquement, donc _fill_group... ne peut pas la trouver.
            vkey_from_edge = {"OB": "L", "BL": "O", "LO": "B"}
            try:
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
            except Exception:
                pass
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
        try:
            ori = str(r.get("orient", "CCW")).upper().strip()
        except Exception:
            ori = "CCW"
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
        try:
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
        except Exception:
            # En auto, en cas de doute on préfère "pruner" la branche plutôt que d'accepter une pose invalide
            return True
