
import os
import json
import datetime as _dt
import math
import re
import copy
import io
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Type

from tkinter import filedialog, messagebox, simpledialog
import tkinter as tk
from tksheet import Sheet


from PIL import Image, ImageTk

# --- SVG rasterization (svglib/reportlab) ---
# IMPORTANT:
# - On utilise svglib + reportlab (renderPDF) + pdfium pour rasteriser le SVG.
# - On évite volontairement renderPM ici : sur certains environnements (notamment Windows/Python 3.13),
#   reportlab peut tenter d'utiliser un backend Cairo absent et planter dès l'import.
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas as _rl_canvas
import pypdfium2 as pdfium

from shapely.geometry import Polygon as _ShPoly, LineString as _ShLine, Point as _ShPoint
from shapely.ops import unary_union as _sh_union
from shapely.ops import unary_union
from shapely.affinity import rotate, translate 

import xml.etree.ElementTree as ET

# --- Dictionnaire (chargement livre.txt) ---
from src.DictionnaireEnigmes import DictionnaireEnigmes

EPS_WORLD = 1e-6

# ---------- Utils géométrie ----------
def _overlap_shrink(S_m, S_t, stroke_px: float, zoom: float) -> bool:
    """
    Anti-chevauchement unique et partagé : SHRINK-ONLY sur les deux solides.
    - Rétrécit S_m et S_t de 2× (stroke_px / zoom) en unités monde,
      puis teste l'intersection des solides réduits.
    - Retourne True si chevauchement détecté, False sinon.
    """
    shrink_world = 2.0 * (float(stroke_px) / max(zoom, 1e-9))
    S_t_shrunk = S_t.buffer(-shrink_world).buffer(0)
    S_m_shrunk = S_m.buffer(-shrink_world).buffer(0)
    if S_t_shrunk.is_empty or S_m_shrunk.is_empty:
        return False
    return S_m_shrunk.intersects(S_t_shrunk)


def _tri_shape(P: Dict[str, np.ndarray]):
    """Triangle → Polygon (brut, sans rétrécissement)."""
    poly = _ShPoly([tuple(map(float, P["O"])),
                    tuple(map(float, P["B"])),
                    tuple(map(float, P["L"]))])
    # 'buffer(0)' nettoie les très petites imprécisions numériques
    return poly.buffer(0)

def _group_shape_from_nodes(nodes, last_drawn):
    """Union BRUTE des triangles listés dans 'nodes' (sans rétrécissement)."""
    polys = []
    for nd in nodes:
        tid = nd.get("tid")
        if tid is None or not (0 <= tid < len(last_drawn)):
            continue
        polys.append(_tri_shape(last_drawn[tid]["pts"]))
    if not polys:
        return None
    # nettoie et unionne
    return unary_union([p.buffer(0) for p in polys])


def _build_local_triangle(OB: float, OL: float, BL: float) -> dict:
    """
    Construit un triangle local avec O=(0,0), B=(OB,0), L=(x,y) à partir des 3 longueurs.
    """
    x = (OL*OL - BL*BL + OB*OB) / (2*OB)
    y2 = max(0.0, OL*OL - x*x)
    y = math.sqrt(y2)
    return {
        "O": np.array([0.0,     0.0], dtype=float),
        "B": np.array([float(OB), 0.0], dtype=float),
        "L": np.array([float(x),  float(y)], dtype=float),
    }

# --- Simulation d'une pose (pour filtrer le chevauchement après collage) ---
def _apply_R_T_on_P(P: Dict[str,np.ndarray], R: np.ndarray, T: np.ndarray, pivot: np.ndarray) -> Dict[str,np.ndarray]:
    """Applique une rotation R autour de 'pivot' puis une translation T aux points P."""
    out = {}
    for k in ("O","B","L"):
        v = np.array(P[k], dtype=float)
        out[k] = (R @ (v - pivot)) + pivot + T
    return out

def _pose_params(Pm: Dict[str,np.ndarray], am: str, bm: str, Vm: np.ndarray,
                 Pt: Dict[str,np.ndarray], at: str, bt: str, Vt: np.ndarray):
    """Retourne (R, T, pivot) pour la pose 'am->bm' alignée sur 'at->bt' avec Vm→Vt."""
    vm_dir = np.array(Pm[bm], float) - np.array(Pm[am], float)
    vt_dir = np.array(Pt[bt], float) - np.array(Pt[at], float)
    ang_m = math.atan2(vm_dir[1], vm_dir[0])
    ang_t = math.atan2(vt_dir[1], vt_dir[0])
    dtheta = ang_t - ang_m
    c, s = math.cos(dtheta), math.sin(dtheta)
    R = np.array([[c, -s],[s, c]], float)
    pivot = np.array(Vm, float)
    Vm_rot = (R @ (np.array(Pm[am], float) - pivot)) + pivot
    T = np.array(Vt, float) - Vm_rot
    return R, T, pivot


class ScenarioAssemblage:
    """
    Représente un scénario d'assemblage (manuel ou automatique).
    Un scénario contient :
      - un nom lisible (ex. 'Scénario manuel', 'Auto #1'),
      - un type de source ('manual' / 'auto'),
      - éventuellement l'identifiant de l'algorithme qui l'a généré,
      - la liste des triangles d'origine (ids),
      - l'état géométrique : triangles dessinés + groupes.

    Pour le scénario manuel, last_drawn / groups sont des références
    directes vers les structures runtime du viewer (pas une copie).
    Pour les scénarios automatiques, on utilisera plutôt des copies.
    """
    def __init__(self, name: str, source_type: str = "manual", algo_id: Optional[str] = None, tri_ids: Optional[List[int]] = None):
        self.name: str = name
        self.source_type: str = source_type      # "manual" ou "auto"
        self.algo_id: Optional[str] = algo_id    # id de l'algo (pour les auto)
        self.tri_ids: List[int] = list(tri_ids) if tri_ids is not None else []

        # État géométrique associé au scénario
        self.last_drawn: List[Dict] = []         # même structure que viewer._last_drawn
        self.groups: Dict[int, Dict] = {}        # même structure que viewer.groups

        # Statut global du scénario (utile pour les simulations auto)
        self.status: str = "complete"            # "complete", "pruned", etc.
        self.created_at: _dt.datetime = _dt.datetime.now()


# =========================
#  SIMULATION: moteur + algos
# =========================

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

        tri1_id = int(tri_ids[0])
        tri2_id = int(tri_ids[1])

        engine = self.engine
        v = engine.viewer

        t1 = engine.build_local_triangle(tri1_id)
        t2 = engine.build_local_triangle(tri2_id)

        # ---- 1) Orientation : O→L au Nord (+Y) pour le 1er triangle
        P1 = {k: np.array(t1["pts"][k], dtype=float) for k in ("O","B","L")}
        vOL = P1["L"] - P1["O"]
        if float(np.hypot(vOL[0], vOL[1])) > 1e-12:
            cur = math.atan2(vOL[1], vOL[0])
            target = math.pi / 2.0  # Nord = +Y
            dtheta = target - cur
            c, s = math.cos(dtheta), math.sin(dtheta)
            R = np.array([[c, -s],[s, c]], dtype=float)
            pivot = P1["O"]
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
        # Si on enchaîne (n>=4), on fige le 1er quad à la première pose valide
        max_poses_first_pair = 2 if len(tri_ids) <= 2 else 1        
        for (at, bt) in [(a1, b1), (b1, a1)]:  # direct puis inversé
            try:
                P2w = engine.pose_points_on_edge(
                    Pm=P2_local, am=a2, bm=b2,
                    Pt=P1, at=at, bt=bt,
                    Vm=P2_local[a2], Vt=P1[at],
                )
            except Exception:
                continue

            # Chevauchement : utiliser EXACTEMENT la règle partagée "shrink-only"
            try:
                poly2 = _tri_shape(P2w)
                if _overlap_shrink(
                    poly2, poly1,
                    getattr(v, "stroke_px", 2),
                    engine.getOverlapZoomRef(),
                ):
                    continue
            except Exception:
                # En auto, si doute → on prune
                continue

            poses.append(P2w)

            if len(poses) >= max_poses_first_pair:
                break
        if not poses:
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
        poly_occ0 = unary_union([_tri_shape(base_last[0]["pts"]), _tri_shape(base_last[1]["pts"])])
        states = [(base_last, poly_occ0)]

        MAX_SCENARIOS = 200

        # Boucle sur les paires suivantes : (tri3,tri4), (tri5,tri6), ...
        for pair_start in range(2, len(tri_ids), 2):
            if pair_start + 1 >= len(tri_ids):
                break

            tri_odd_id = int(tri_ids[pair_start])       # tri3, tri5, ...
            tri_even_id = int(tri_ids[pair_start + 1])  # tri4, tri6, ...

            # Construit le quad courant en repère local (indépendant de la branche)
            try:
                tOdd, tEven, Podd, Peven = _build_quad_local(tri_odd_id, tri_even_id)
            except Exception:
                return []

            new_states = []

            for (last_drawn_prev, poly_occ_prev) in states:
                # Triangle "ancre" = le dernier triangle posé (tri2, puis tri4, puis tri6, ...)
                anchor = last_drawn_prev[-1]["pts"]

                # Côté cible (anchor) : 2 arêtes issues de L (L->O et L->B)
                # Côté mobile (odd)   : pour n=4 on garde la convention (L->O) uniquement,
                #                       mais à partir de n>=6 on doit aussi tester (L->B),
                #                       sinon on peut pruner des solutions valides.
                mob_keys = ("O",) if pair_start == 2 else ("O", "B")
                for mob_key in mob_keys:
                    for bt_key in ("O", "B"):
                        try:
                            R, T, pivot = _pose_params(
                                Podd, "L", mob_key, Podd["L"],
                                anchor, "L", bt_key, anchor["L"]
                            )
                            Podd_w = _apply_R_T_on_P(Podd, R, T, pivot)
                            Peven_w = _apply_R_T_on_P(Peven, R, T, pivot)
                        except Exception:
                            continue

                        # Overlap : le quad courant ne doit pas chevaucher l'occupation déjà posée
                        try:
                            poly_new = unary_union([_tri_shape(Podd_w), _tri_shape(Peven_w)])
                            if _overlap_shrink(
                                poly_new, poly_occ_prev,
                                getattr(v, "stroke_px", 2),
                                engine.getOverlapZoomRef(),
                            ):
                                continue
                        except Exception:
                            continue

                        # Étend le scénario partiel
                        last_drawn_new = copy.deepcopy(last_drawn_prev)

                        pos0 = len(last_drawn_new)
                        last_drawn_new.append({
                            "labels": tOdd.get("labels"),
                            "pts": Podd_w,
                            "id": tri_odd_id,
                            "mirrored": bool(tOdd.get("mirrored", False)),
                            "group_id": 1,
                            "group_pos": pos0,
                        })
                        last_drawn_new.append({
                            "labels": tEven.get("labels"),
                            "pts": Peven_w,
                            "id": tri_even_id,
                            "mirrored": bool(tEven.get("mirrored", False)),
                            "group_id": 1,
                            "group_pos": pos0 + 1,
                        })

                        poly_occ_new = unary_union([poly_occ_prev, poly_new])
                        new_states.append((last_drawn_new, poly_occ_new))

                        if len(new_states) >= MAX_SCENARIOS:
                            break

                    if len(new_states) >= MAX_SCENARIOS:
                        break

            states = new_states
            if not states:
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
        # IMPORTANT: la simulation ne doit pas dépendre du zoom écran courant.
        # On fige un zoom de référence, clampé, utilisé uniquement pour _overlap_shrink().
        try:
            z = float(getattr(viewer, "simulationOverlapZoomRef", getattr(viewer, "zoom", 1.0)))
        except Exception:
            z = 1.0
        # borne haute: évite shrink trop petit => faux chevauchement sur triangles collés
        self.overlapZoomRef = min(max(z, 0.25), 8.0)

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

class DialogSimulationAssembler(tk.Toplevel):
    """Boîte de dialogue 'Simulation > Assembler…'"""

    def __init__(self, parent, algo_items: List[Tuple[str, str]], n_max: int, default_algo_id: str, default_n: int):
        super().__init__(parent)
        self.title("Assembler (simulation)")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result = None  # (algo_id, n, clear_auto)

        # Imports locaux (évite d'imposer ttk partout)
        from tkinter import ttk

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Algorithme :").grid(row=0, column=0, sticky="w")
        self.algo_var = tk.StringVar(value=default_algo_id or (algo_items[0][0] if algo_items else ""))
        self.algo_combo = ttk.Combobox(
            frm,
            textvariable=self.algo_var,
            state="readonly",
            values=[f"{aid} - {label}" for aid, label in algo_items],
            width=48
        )
        self._algo_items = list(algo_items)
        sel_index = 0
        for i, (aid, _lbl) in enumerate(self._algo_items):
            if aid == self.algo_var.get():
                sel_index = i
                break
        if algo_items:
            self.algo_combo.current(sel_index)
        self.algo_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        ttk.Label(frm, text="Nombre de triangles (n premiers) :").grid(row=2, column=0, sticky="w")
        self.n_var = tk.IntVar(value=int(default_n))
        self.n_spin = ttk.Spinbox(frm, from_=2, to=max(2, int(n_max)), textvariable=self.n_var, width=8)
        self.n_spin.grid(row=2, column=1, sticky="e")

        self.clear_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Supprimer les scénarios auto existants", variable=self.clear_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        ttk.Label(frm, text="Note : pour l’instant, n doit être pair (on arrondit à l’inférieur).").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Annuler", command=self._on_cancel).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="OK", command=self._on_ok).grid(row=0, column=1)

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        try:
            self.n_spin.focus_set()
            self.n_spin.selection_range(0, tk.END)
        except Exception:
            pass

    def _on_cancel(self):
        self.result = None
        self.destroy()

    def _on_ok(self):
        raw = str(self.algo_combo.get() or "")
        algo_id = raw.split(" - ", 1)[0].strip() if " - " in raw else raw.strip()

        try:
            n = int(self.n_var.get())
        except Exception:
            n = 0

        if n <= 0:
            messagebox.showerror("Assembler", "Nombre de triangles invalide.")
            return

        if n % 2 == 1:
            n2 = n - 1
            if n2 < 2:
                messagebox.showerror("Assembler", "n doit être pair (minimum 2).")
                return
            messagebox.showinfo("Assembler", f"n doit être pair : on utilise n={n2}.")
            n = n2
            try:
                self.n_var.set(n)
            except Exception:
                pass

        self.result = (algo_id, int(n), bool(self.clear_var.get()))
        self.destroy()

# ---------- Application (MANUEL — sans algorithmes) ----------
class TriangleViewerManual(tk.Tk):
    """
    Version épurée pour travail manuel :
      - Chargement Excel
      - Liste des triangles
      - Affichage "brut" (sans assemblage) avec mise en page simple en ligne(s)
      - Fit à l'écran, pan/zoom
      - Impression PDF des triangles bruts (même échelle)
    """
    def __init__(self):
        super().__init__()
        self.title("Assembleur de Triangles — Mode Manuel")
        self.geometry("1200x700")

        # état de vue
        # Référence STABLE pour l'anti-chevauchement en simulation.
        # IMPORTANT : ne doit pas bouger quand on fait un "fit à l'écran" (qui modifie self.zoom
        self.simulationOverlapZoomRef = 1.0
        self.zoom = 1.0
        self.offset = np.array([400.0, 350.0], dtype=float)
        self._drag = None             # état de drag & drop depuis la liste
        self._drag_preview_id = None  # id du polygone "fantôme" sur le canvas
        self._sel = None              # sélection sur canvas: {'mode': 'move'|'vertex', 'idx': int}
        self._hit_px = 12             # tolérance de hit (pixels) pour les sommets
        self._marker_px = 6           # rayon des marqueurs (cercles) dessinés aux sommets
        self._ctx_target_idx = None   # index du triangle visé par clic droit (menu contextuel)
        self._placed_ids = set()      # ids déjà posés dans le scénario actif
        self._ctx_last_rclick = None  # dernière position écran du clic droit (pour pivoter)
        self._nearest_line_id = None  # trait d'aide "sommet le plus proche"
        self._edge_highlight_ids = [] # surlignage des 2 arêtes (mobile/cible)
        self._edge_choice = None      # (i_mob, key_mob, edge_m, i_tgt, key_tgt, edge_t)
        self._edge_highlights = None  # données brutes des aides (candidates + best)
        self._tooltip = None          # tk.Toplevel
        self._tooltip_label = None    # tk.Label

        # --- cache pick (écran) régénéré après load / zoom / pan ---
        self._pick_cache_valid = False
        # chaque item de self._last_drawn recevra:
        #   t["_pick_poly"] : liste de points écran [(x,y),...]
        #   t["_pick_pts"]  : dict {'O':(x,y), 'B':(x,y), 'L':(x,y)}

        # Association triangle -> mot du dictionnaire: { tri_id: {"word": str, "row": int, "col": int} }
        self._tri_words: Dict[int, Dict[str, int | str]] = {}

        # distance écran supplémentaire pour l'ancrage du tooltip (px)
        self._tooltip_cushion_px = 14

        # Épaisseur de trait (px) — utilisée pour le test de chevauchement "shrink seul"
        self.stroke_px = 2

        # Hauteur fixe du panneau "Dico" (en pixels) sous le canvas
        self.dico_panel_height = 290
        # État d'affichage du panneau dictionnaire (toggle via menu Visualisation)
        self.show_dico_panel = tk.BooleanVar(value=True)
        # État d'affichage du compas horaire (overlay horloge)
        self.show_clock_overlay = tk.BooleanVar(value=True)

        # --- Fond SVG (coordonnées monde) ---
        self.bg_resize_mode = tk.BooleanVar(value=False)
        self._bg = None  # dict: {path,x0,y0,w,h,aspect}
        self._bg_base_pil = None  # PIL.Image RGBA (base, normalisée)
        self._bg_photo = None     # ImageTk.PhotoImage (vue écran)
        self._bg_resizing = None  # dict état drag poignée

        # mode "déconnexion" activé par CTRL
        self._ctrl_down = False        

        # --- DEBUG: permettre de SKIP le test de chevauchement durant le highlight (F9) ---
        self.debug_skip_overlap_highlight = False

        # === Groupes (Étape A) ===
        self.groups: Dict[int, Dict] = {}
        self._next_group_id: int = 1

        # === Scénarios d'assemblage ===
        # Liste de scénarios (1 scénario manuel actif + futurs scénarios auto).
        self.scenarios: List[ScenarioAssemblage] = []
        self.active_scenario_index: int = 0

        # état IHM
        self.triangle_file = tk.StringVar(value="")
        self.start_index = tk.IntVar(value=1)
        self.num_triangles = tk.IntVar(value=8)

        self.excel_path = None
        # Répertoires par défaut
        self.data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
        self.scenario_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "scenario"))
        # Répertoire des icônes
        self.images_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "images"))
        os.makedirs(self.scenario_dir, exist_ok=True)
        # === Config (persistance des paramètres) ===
        self.config_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "config"))
        self.config_path = os.path.join(self.config_dir, "assembleur_config.json")
        self.appConfig: Dict = {}
        self.loadAppConfig()

        self.df = None
        self._last_drawn = []   # liste d'items: (labels, P_world)


        # Crée le scénario manuel "par défaut" qui pointe sur l'état runtime.
        # last_drawn et groups sont partagés par référence : toute modification
        # manuelle met automatiquement à jour ce scénario.
        manual = ScenarioAssemblage(
            name="Scénario manuel",
            source_type="manual",
            algo_id=None,
            tri_ids=[],
        )
        manual.last_drawn = self._last_drawn
        manual.groups = self.groups
        self.scenarios.append(manual)

        # --- Horloge (overlay fixe) : état par défaut ---
        self._clock_state = {"hour": 5, "minute": 9, "label": "Trouver — (5h, 9')"}
        # Position & état de drag de l'horloge (coords CANVAS)
        self._clock_cx = None
        self._clock_cy = None
        self._clock_R  = 69    # rayon px (mis à jour dans le draw)
        self._clock_dragging = False
        self._clock_drag_dx = 0
        self._clock_drag_dy = 0

        self._build_ui()
        # Centraliser / garantir les bindings (création initiale)
        self._bind_canvas_handlers()      

        # Bind pour annuler avec ESC (drag ou sélection)
        self.bind("<Escape>", self._on_escape_key)

        # === Dictionnaire : chargement automatique de ../data/livre.txt ===
        self.dico = None
        self._init_dictionary()
        self._build_dico_grid()


    # ---------- Dictionnaire : init ----------
    def _init_dictionary(self):
        """Construit le dictionnaire en lisant ../data/livre.txt (si présent)."""
        try:
            if DictionnaireEnigmes is None:
                raise ImportError("Module DictionnaireEnigmes introuvable")
            livre_path = os.path.join(self.data_dir, "livre.txt")
            if not os.path.isfile(livre_path):
                self.status.config(text="Dico: fichier 'livre.txt' non trouvé dans ../data")
                return
            self.dico = DictionnaireEnigmes(livre_path)  # charge tout le livre
            nb_lignes = len(self.dico)
            self.status.config(text=f"Dico chargé: {nb_lignes} lignes depuis {livre_path}")
        except Exception as e:
            try:
                self.status.config(text=f"Dico: échec de chargement — {e}")
            except Exception:
                pass

    # ---------- Dictionnaire : affichage dans le panneau bas ----------
    def _build_dico_grid(self):
        """
        Construit/affiche la grille tksheet du dictionnaire dans self.dicoPanel.
        N’opère que si self.dico est chargé et tksheet disponible.
        """
        # Le panneau bas doit exister (créé dans _build_canvas)
        if not hasattr(self, "dicoPanel"):
            return
        # Nettoyer le contenu existant (placeholder, ancienne grille…)
        for child in list(self.dicoPanel.children.values()):
            try:
                child.destroy()
            except Exception:
                pass
        # Vérifs
        if self.dico is None:
            tk.Label(self.dicoPanel, text="Dictionnaire non chargé",
                     bg="#f3f3f3", anchor="w").pack(fill="x", padx=8, pady=6)
            return

        # --- Layout du panneau bas : [sidebar catégories] | [grille] ---
        from tkinter import ttk
        container = tk.Frame(self.dicoPanel, bg="#f3f3f3")
        container.pack(fill="both", expand=True)
        # colonne gauche (catégories + liste)
        left = tk.Frame(container, width=180, bg="#f3f3f3", bd=1, relief=tk.GROOVE)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        # colonne droite (grille)
        right = tk.Frame(container, bg="#f3f3f3")
        right.pack(side="left", fill="both", expand=True)

        # ===== Barre "catégories" =====
        tk.Label(left, text="Catégorie :", anchor="w", bg="#f3f3f3").pack(anchor="w", padx=8, pady=(8,2))
        cats = []
        try:
            cats = list(self.dico.getCategories())
        except Exception:
            cats = []
        self._dico_cat_var = tk.StringVar(value=cats[0] if cats else "")
        self._dico_cat_combo = ttk.Combobox(left, state="readonly", values=cats, textvariable=self._dico_cat_var)
        self._dico_cat_combo.pack(fill="x", padx=8, pady=(0,6))

        # Liste des mots de la catégorie sélectionnée
        lb_frame = tk.Frame(left, bg="#f3f3f3"); lb_frame.pack(fill="both", expand=True, padx=6, pady=(0,8))
        self._dico_cat_list = tk.Listbox(lb_frame, exportselection=False)
        self._dico_cat_list.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(lb_frame, orient="vertical", command=self._dico_cat_list.yview)
        sb.pack(side="right", fill="y")
        self._dico_cat_list.configure(yscrollcommand=sb.set)

        # Remplissage initial + binding de la combo
        self._dico_cat_items = []
        def _refresh_cat_list(*_):
            cat = self._dico_cat_var.get()
            self._dico_cat_list.delete(0, tk.END)
            if not cat:
                return
            try:
                # getListeCategorie -> [(mot, enigme, indexMot), ...]
                items = self.dico.getListeCategorie(cat)
            except Exception:
                items = []
            # mémoriser pour la synchro avec la grille
            self._dico_cat_items = list(items)
            # Afficher: "mot — (e, m)"
            for mot, e, m in items:
                self._dico_cat_list.insert(tk.END, f"{mot} — ({e}, {m})")
        self._dico_cat_combo.bind("<<ComboboxSelected>>", _refresh_cat_list)
        _refresh_cat_list()


        # Synchronisation: clic/double-clic dans la liste -> centrer/sélectionner le mot dans la grille
        def _goto_selected_word(event=None):
            if not getattr(self, "dicoSheet", None):
                return
            try:
                sel = self._dico_cat_list.curselection()
                if not sel:
                    return
                i = int(sel[0])
                mot, enigme, indexMot = self._dico_cat_items[i]
            except Exception:
                return
            # convertir indexMot [-N..N) -> colonne [0..2N)
            try:
                nb_mots_max = self._dico_nb_mots_max if hasattr(self, "_dico_nb_mots_max") else self.dico.nbMotMax()
            except Exception:
                nb_mots_max = 50
            col = int(indexMot) + int(nb_mots_max)
            row = int(enigme)
            # sélectionner et faire voir la cellule ; see() centre autant que possible
            try:
                self.dicoSheet.select_cell(row, col, redraw=False)
            except Exception:
                pass
            try:
                self.dicoSheet.see(row=row, column=col)
            except Exception:
                pass
            # MAJ horloge (sélection indirecte via la liste)
            try:
                self._update_clock_from_cell(row, col)
            except Exception:
                pass

        # Clic et double-clic compatibles
        self._dico_cat_list.bind("<<ListboxSelect>>", _goto_selected_word)
        self._dico_cat_list.bind("<Double-Button-1>", _goto_selected_word)

        # ===== Grille (tksheet) =====
        # Construire la matrice
        try:
            nb_mots_max = self.dico.nbMotMax()
        except Exception:
            # fallback conservateur
            nb_mots_max = 50
        # mémoriser pour conversions (colonne → minute)
        self._dico_nb_mots_max = int(nb_mots_max)
        headers = list(range(-nb_mots_max, nb_mots_max))
        data = []
        for i in range(len(self.dico)):
            try:
                row = [self.dico[i][j] for j in range(-nb_mots_max, nb_mots_max)]
            except Exception:
                # en cas de dépassement, remplir de chaînes vides
                row = ["" for _ in range(2 * nb_mots_max)]
            data.append(row)
        try:
            row_index = self.dico.getTitres()
        except Exception:
            row_index = [str(i) for i in range(len(self.dico))]
        # mémoriser les titres (ligne → "530", "780", …)
        self._dico_row_index = list(row_index)

        # Créer la grille
        self.dicoSheet = Sheet(
            right,
            data=data,
            headers=headers,
            row_index=row_index,
            show_row_index=True,
            height=max(120, getattr(self, "dico_panel_height", 220) - 10),
            empty_vertical=0,
        )
        self.dicoSheet.enable_bindings((
            "single_select","row_select","column_select",
            "arrowkeys","copy","rc_select","right_click_popup_menu"
        ))
        self.dicoSheet.align_columns(columns="all", align="center")
        self.dicoSheet.set_options(cell_align="center")
        self.dicoSheet.pack(expand=True, fill="both")

        # --- MAJ horloge sur sélection de cellule (événement unique & propre) ---
        def _on_dico_cell_select(event=None):
            # Source unique de vérité : cellule actuellement sélectionnée
            sel = self.dicoSheet.get_selected_cells()
            r = c = None
            if sel:
                # tksheet peut renvoyer un set({(r,c), ...}) ou une liste([(r,c), ...])
                if isinstance(sel, set):
                    r, c = next(iter(sel))
                elif isinstance(sel, (list, tuple)):
                    r, c = sel[0]
            if r is None or c is None:
                # Secours léger : cellule "courante" si aucune sélection renvoyée
                cur = self.dicoSheet.get_currently_selected()
                if isinstance(cur, tuple) and len(cur) == 2 and cur[0] == "cell":
                    r, c = cur[1]
            if r is None or c is None:
                return
            self._update_clock_from_cell(int(r), int(c))

        # Un seul binding tksheet, propre : "cell_select"
        self.dicoSheet.extra_bindings([("cell_select", _on_dico_cell_select)])

        # --- Centrer l’affichage par défaut sur la colonne 0 ---
        try:
            zero_col = int(nb_mots_max)              # colonne "0" du dico
            self.dicoSheet.select_cell(0, zero_col, redraw=False)
            self.dicoSheet.see(row=0, column=zero_col)  # amène la colonne 0 dans la vue (le plus centré possible)
        except Exception:
            pass

        try:
            self.status.config(text="Dico affiché dans le panneau bas")
        except Exception:
            pass


    # ---------- DICO → Horloge ----------
    def _update_clock_from_cell(self, row: int, col: int):
        """
        Met à jour l'horloge à partir d'une cellule de la grille Dico.
          - Heure : dérivée du titre de ligne (ex: '530' -> 5h).
          - Minute : distance à la colonne 0 (col absolue) -> abs(col - nb_mots_max).
          - Libellé : contenu texte de la cellule.
        """
        if not getattr(self, "dicoSheet", None):
            return
        # Sécurités
        nbm = int(getattr(self, "_dico_nb_mots_max", 50))
        row_titles = getattr(self, "_dico_row_index", [])
        # Mot (texte de cellule)
        try:
            word = str(self.dicoSheet.get_cell_data(row, col)).strip()
        except Exception:
            word = ""
        # Heure -> prendre les centaines (530 -> 5)
        try:
            titre = str(row_titles[row]) if row_titles and 0 <= row < len(row_titles) else ""
            hour = int(str(int(titre))[:1]) if titre and titre.isdigit() else (int(titre)//100)
        except Exception:
            # fallback: si pas de titre exploitable, approx à partir de l'index
            hour = max(0, int(row) % 12)
        # Normaliser heure 0..11 (affichage 0h..11h ; si tu veux 12h remapper 0->12 à l'affichage)
        try:
            hour = int(hour) % 12
        except Exception:
            hour = 0
        # Minute -> distance à la colonne centrale (colonne 0)
        try:
            minute = abs(int(col) - nbm)
        except Exception:
            minute = 0
        # Libellé
        label = word if word else ""
        if label:
            label = f"{label} — ({hour}h, {minute}')"
        else:
            label = f"({hour}h, {minute}')"
        # Appliquer et redessiner
        self._clock_state.update({"hour": hour, "minute": minute, "label": label})
        self._redraw_overlay_only()


    # ---------- DICO : lecture de la sélection courante ----------
    def _get_selected_dico_word(self) -> Optional[Tuple[str, int, int]]:
        """
        Retourne (word, row, col) depuis la sélection de tksheet, ou None si rien.
        """
        if not getattr(self, "dicoSheet", None):
            return None
        sel = self.dicoSheet.get_selected_cells()
        r = c = None
        if sel:
            if isinstance(sel, set):
                r, c = next(iter(sel))
            elif isinstance(sel, (list, tuple)):
                r, c = sel[0]
        if r is None or c is None:
            cur = self.dicoSheet.get_currently_selected()
            if isinstance(cur, tuple) and len(cur) == 2 and cur[0] == "cell":
                r, c = cur[1]
        if r is None or c is None:
            return None
        try:
            word = str(self.dicoSheet.get_cell_data(int(r), int(c))).strip()
        except Exception:
            word = ""
        if not word:
            return None
        return (word, int(r), int(c))

    # ---------- Contexte : actions mot <-> triangle ----------
    def _ctx_add_or_replace_word(self):
        """Ajoute/remplace le mot sélectionné du dico sur le triangle ciblé."""
        if self._ctx_target_idx is None or not (0 <= self._ctx_target_idx < len(self._last_drawn)):
            return
        tri = self._last_drawn[self._ctx_target_idx]
        tri_id = int(tri.get("id"))
        sel = self._get_selected_dico_word()
        if not sel:
            messagebox.showinfo("Association mot", "Aucun mot sélectionné dans le dictionnaire.")
            return
        word, row, col = sel
        self._tri_words[tri_id] = {"word": word, "row": row, "col": col}
        self._redraw_from(self._last_drawn)

    def _ctx_clear_word(self):
        """Efface l'association de mot du triangle ciblé, si présente."""
        if self._ctx_target_idx is None or not (0 <= self._ctx_target_idx < len(self._last_drawn)):
            return
        tri = self._last_drawn[self._ctx_target_idx]
        tri_id = int(tri.get("id"))
        if tri_id in self._tri_words:
            del self._tri_words[tri_id]
            self._redraw_from(self._last_drawn)

    def _rebuild_ctx_word_entries(self):
        """Reconstruit la partie 'mot' du menu contextuel en fonction du triangle visé + sélection dico."""
        # Nettoyer les deux entrées dynamiques existantes
        try:
            # On supprime depuis la fin pour conserver l'index d'ancrage
            end = self._ctx_menu.index("end")
            while end > self._ctx_idx_words_start:
                self._ctx_menu.delete(end)
                end = self._ctx_menu.index("end")
        except Exception:
            pass
        # Recréer deux entrées selon contexte
        label_add = "Ajouter…"
        cmd_add = None
        sel = self._get_selected_dico_word()
        if sel:
            label_add = f"Ajouter « {sel[0]} »"
            cmd_add = self._ctx_add_or_replace_word
        # Triangle ciblé ?
        has_target = (self._ctx_target_idx is not None) and (0 <= self._ctx_target_idx < len(self._last_drawn))
        exists = False
        label_del = "Effacer…"
        if has_target:
            tri = self._last_drawn[self._ctx_target_idx]
            tri_id = int(tri.get("id"))
            if tri_id in self._tri_words:
                exists = True
                cur_word = self._tri_words[tri_id]["word"]
                # si on a aussi une sélection, on préfère le verbe "Remplacer"
                if sel:
                    label_add = f"Remplacer par « {sel[0]} »"
                label_del = f"Effacer « {cur_word} »"
        # Ajouter les entrées (activées/désactivées selon contexte)
        self._ctx_menu.add_command(label=label_add, command=cmd_add, state=("normal" if cmd_add and has_target else "disabled"))
        self._ctx_menu.add_command(label=label_del, command=(self._ctx_clear_word if exists else None),
                                   state=("normal" if exists else "disabled"))


    # ---------- DEBUG: toggle du filtre d'intersection au highlight ----------
    def _toggle_skip_overlap_highlight(self, event=None):
        self.debug_skip_overlap_highlight = not self.debug_skip_overlap_highlight
        state = "IGNORE" if self.debug_skip_overlap_highlight else "APPLIQUE"
        try:
            self.status.config(text=f"[DEBUG] Chevauchement (highlight): {state} — F9 pour basculer")
        except Exception:
            pass

    # ---------- UI ----------
    def _build_ui(self):
        # --- Barre de menus ---
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # --- Menu Scénario (save/load XML) ---
        self.menu_scenario = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Scénario", menu=self.menu_scenario)
        self.menu_scenario.add_command(label="Nouveau", command=self._new_empty_scenario)
        self.menu_scenario.add_command(label="Charger…", command=self._scenario_load_dialog)
        self.menu_scenario.add_command(label="Enregistrer…", command=self._scenario_save_dialog)
        self.menu_scenario.add_separator()
        # placeholder pour la liste des .xml du dossier 'scenario'
        self._menu_scenario_files_anchor = self.menu_scenario.index("end")
        self.menu_scenario.add_command(label="(scan des scénarios…)", state="disabled")
        self._rebuild_scenario_file_list_menu()

        # --- Menu Triangle (save/load XML) ---
        self.menu_triangle = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Triangle", menu=self.menu_triangle)

        # --- Menu Simulation (assemblage automatique) ---
        self.menu_simulation = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulation", menu=self.menu_simulation)
        self.menu_simulation.add_command(label="Assembler…", command=self._simulation_assemble_dialog)
        self.menu_simulation.add_separator()
        self.menu_simulation.add_command(label="Supprimer les scénarios automatiques", command=self._simulation_clear_auto_scenarios)

        # --- Menu Visualisation ---
        self.menu_visual = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualisation", menu=self.menu_visual)
        self.menu_visual.add_command(
            label="Fit à l'écran",
            command=lambda: self._fit_to_view(self._last_drawn)
        )
        # Effacer l'affichage depuis le menu
        self.menu_visual.add_command(
            label="Effacer l'affichage…",
            command=self.clear_canvas
        )

        # Toggle d'affichage du dictionnaire (panneau bas + combo/liste)
        self.menu_visual.add_checkbutton(
            label="Afficher le dictionnaire",
            variable=self.show_dico_panel,
            command=self._toggle_dico_panel
        )

        self.menu_visual.add_checkbutton(
            label="Afficher le compas horaire",
            variable=self.show_clock_overlay,
            command=self._toggle_clock_overlay
        )

        self.menu_visual.add_separator()
        self.menu_visual.add_command(label="Charger fond SVG…", command=self._bg_load_svg_dialog)
        self.menu_visual.add_checkbutton(
            label="Mode redimensionnement fond (ratio)",
            variable=self.bg_resize_mode,
            command=lambda: self._redraw_from(self._last_drawn)
        )
        self.menu_visual.add_command(label="Supprimer fond", command=self._bg_clear)

        # Items fixes
        self.menu_triangle.add_command(label="Ouvrir un fichier Triangle…", command=self.open_excel)
        self.menu_triangle.add_separator()
        # Espace réservé: liste de fichiers du répertoire data (reconstruite dynamiquement)
        # On pose un placeholder que l'on remplace juste après.
        self._menu_triangle_files_start_index = self.menu_triangle.index("end") + 1 if self.menu_triangle.index("end") is not None else 2
        self.menu_triangle.add_command(label="(scan en cours)")
        self.menu_triangle.add_separator()
        self.menu_triangle.add_command(label="Imprimer", command=self.print_triangles_dialog)
        # Construit la liste de fichiers disponibles
        self._rebuild_triangle_file_list_menu()

        main = tk.Frame(self); main.pack(fill=tk.BOTH, expand=True)
        self._build_left_pane(main)
        self._build_canvas(main)

        self.status = tk.Label(self, text="Prêt", bd=1, relief=tk.SUNKEN, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Auto-load: recharger le dernier fichier triangles ouvert (config)
        # sinon fallback sur data/triangle.xlsx si présent.
        self.autoLoadTrianglesFileAtStartup()

    # =========================
    #  SIMULATION (AUTO ASSEMBLAGE)
    # =========================
    def _simulation_get_tri_ids_first_n(self, n: int) -> List[int]:
        """Retourne les IDs logiques des n premiers triangles (ordre de la listbox)."""
        ids: List[int] = []
        if not hasattr(self, "listbox"):
            return ids
        max_n = int(self.listbox.size())
        n2 = min(int(n), max_n)
        for i in range(n2):
            s = str(self.listbox.get(i))
            m = re.match(r"\s*(\d+)\.", s)
            if m:
                ids.append(int(m.group(1)))
        return ids

    def _simulation_clear_auto_scenarios(self):
        """Supprime tous les scénarios 'auto' (conserve les manuels)."""
        if not getattr(self, "scenarios", None):
            return
        kept = [s for s in self.scenarios if getattr(s, "source_type", "manual") == "manual"]
        removed = len(self.scenarios) - len(kept)
        self.scenarios = kept
        if not self.scenarios:
            self.scenarios = [ScenarioAssemblage(name="Scénario manuel", source_type="manual")]
        self.active_scenario_index = min(self.active_scenario_index, len(self.scenarios) - 1)
        self._set_active_scenario(self.active_scenario_index)
        self._refresh_scenario_listbox()
        try:
            self.status.config(text=f"Scénarios auto supprimés : {removed}")
        except Exception:
            pass

    def _simulation_assemble_dialog(self):
        """Ouvre la boîte de dialogue 'Assembler…' et lance l'algo choisi."""
        if getattr(self, "df", None) is None:
            messagebox.showwarning("Assembler", "Charge d'abord un fichier Triangle.")
            return
        if not hasattr(self, "listbox"):
            messagebox.showwarning("Assembler", "Listbox des triangles introuvable.")
            return

        n_max = int(self.listbox.size())
        if n_max < 2:
            messagebox.showwarning("Assembler", "Il faut au moins 2 triangles dans la liste.")
            return

        algo_items = [(aid, cls.label) for aid, cls in ALGOS.items()]
        default_algo_id = getattr(self, "_simulation_last_algo_id", algo_items[0][0] if algo_items else "")
        default_n = getattr(self, "_simulation_last_n", n_max)
        default_n = min(int(default_n), n_max)
        if default_n < 2:
            default_n = 2

        dlg = DialogSimulationAssembler(self, algo_items, n_max=n_max, default_algo_id=default_algo_id, default_n=default_n)
        self.wait_window(dlg)
        if not getattr(dlg, "result", None):
            return

        algo_id, n, clear_auto = dlg.result
        self._simulation_last_algo_id = algo_id
        self._simulation_last_n = int(n)

        if clear_auto:
            self._simulation_clear_auto_scenarios()

        tri_ids = self._simulation_get_tri_ids_first_n(n)
        if len(tri_ids) < 2:
            messagebox.showwarning("Assembler", "Impossible de construire la liste des IDs de triangles.")
            return

        if len(tri_ids) % 2 == 1:
            tri_ids = tri_ids[:-1]

        try:
            engine = MoteurSimulationAssemblage(self)
            algo_cls = ALGOS.get(algo_id)
            if algo_cls is None:
                raise ValueError(f"Algo inconnu: {algo_id}")
            algo = algo_cls(engine)
            scenarios = algo.run(tri_ids)
        except NotImplementedError as e:
            messagebox.showinfo("Assembler", str(e))
            return
        except Exception as e:
            messagebox.showerror("Assembler", str(e))
            return

        if not scenarios:
            messagebox.showinfo("Assembler", "Aucun scénario généré.")
            return

        base_idx = len(self.scenarios)
        count_auto = sum(1 for s in self.scenarios if s.source_type == "auto")
        for k, scen in enumerate(scenarios):
            if not isinstance(scen, ScenarioAssemblage):
                continue
            scen.source_type = "auto"
            scen.algo_id = scen.algo_id or algo_id
            scen.tri_ids = scen.tri_ids or list(tri_ids)
            if not scen.name:
                scen.name = f"Auto #{count_auto + k + 1}"
            self.scenarios.append(scen)

        self._refresh_scenario_listbox()
        try:
            self._set_active_scenario(base_idx)
        except Exception:
            pass
        try:
            self.status.config(text=f"Simulation: {len(scenarios)} scénario(s) généré(s) (algo={algo_id}, n={n})")
        except Exception:
            pass

    def _rebuild_triangle_file_list_menu(self):
        """
        Reconstruit la portion du menu 'Triangle' listant les fichiers disponibles
        dans self.data_dir. Charge direct au clic.
        """
        m = self.menu_triangle
        # Supprimer les anciens items listés entre la 1re séparatrice et la 2e.
        # Organisation actuelle: [Ouvrir][sep][FICHIERS...][sep][Imprimer]
        # On repère la 2e séparatrice en partant de la fin.
        last_index = m.index("end")
        if last_index is None:
            return
        # Trouver l'index de la 2ème séparatrice (celle avant "Imprimer")
        # Simplification: on sait que l’avant-dernier item est une séparatrice.
        second_sep_index = last_index - 1
        # On efface tous les items entre la 1re séparatrice (index 1) exclue et second_sep_index exclu
        # Indices actuels: 0:"Ouvrir", 1:"sep", [2..n-2]=fichiers, n-1:"sep", n:"Imprimer"
        # On retire de n-2 jusqu’à 2 pour éviter le décalage pendant la suppression
        for i in range(second_sep_index - 1, 1, -1):
            try:
                m.delete(i)
            except Exception:
                pass
        # Repeupler
        try:
            files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".xlsx")]
        except Exception:
            files = []
        if not files:
            m.insert_command(2, label="(aucun fichier trouvé dans data)", state="disabled")
        else:
            # Trier par nom lisible
            files.sort(key=lambda s: s.lower())
            for idx, fname in enumerate(files):
                full = os.path.join(self.data_dir, fname)
                # Insérer à partir de l’index 2 (après la 1re séparatrice)
                m.insert_command(2 + idx, label=fname, command=lambda p=full: self.load_excel(p))

    # ---------- Icônes ----------
    def _load_icon(self, filename: str):
        """
        Charge une icône depuis le dossier images.
        Retourne un tk.PhotoImage ou None si échec (fichier manquant, etc.).
        """
        try:
            base = getattr(self, "images_dir", None)
            if not base:
                return None
            path = os.path.join(base, filename)
            if not os.path.isfile(path):
                return None
            return tk.PhotoImage(file=path)
        except Exception:
            return None

    def _build_left_pane(self, parent):
        left = tk.Frame(parent, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        # PanedWindow vertical : haut = triangles, bas = scénarios
        pw = tk.PanedWindow(left, orient=tk.VERTICAL)
        pw.pack(fill=tk.BOTH, expand=True)

        # --- Panneau haut : liste des triangles ---
        tri_frame = tk.Frame(pw)
        tk.Label(tri_frame, text="Triangles (ordre)").pack(anchor="w", padx=6, pady=(6, 0))

        lb_frame = tk.Frame(tri_frame)
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.listbox = tk.Listbox(lb_frame, width=34, exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_scroll = tk.Scrollbar(lb_frame, orient="vertical", command=self.listbox.yview)
        lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.configure(yscrollcommand=lb_scroll.set)
        # Gestion de la sélection / triangles déjà utilisés
        self._last_triangle_selection = None
        self._in_triangle_select_guard = False
        self.listbox.bind("<<ListboxSelect>>", self._on_triangle_list_select)
        # Démarrer le drag dès qu'on clique sur un item de triangle
        self.listbox.bind("<ButtonPress-1>", self._on_list_mouse_down)

        pw.add(tri_frame, minsize=150)  # hauteur mini raisonnable pour les triangles

        # --- Panneau bas : scénarios + barre d'icônes ---
        scen_frame = tk.Frame(pw)
        tk.Label(scen_frame, text="Scénarios").pack(anchor="w", padx=6, pady=(6, 0))

        # Barre d'icônes (Nouveau, Charger, Propriétés, Sauver, Dupliquer, Supprimer)
        toolbar = tk.Frame(scen_frame)
        toolbar.pack(anchor="w", padx=4, pady=(0, 2), fill="x")

        # Chargement des icônes (tu peux adapter les noms de fichiers PNG)
        self.icon_scen_new   = self._load_icon("scenario_new.png")
        self.icon_scen_open  = self._load_icon("scenario_open.png")
        self.icon_scen_props = self._load_icon("scenario_props.png")
        self.icon_scen_save  = self._load_icon("scenario_save.png")
        self.icon_scen_dup   = self._load_icon("scenario_duplicate.png")
        self.icon_scen_del   = self._load_icon("scenario_delete.png")

        def _make_btn(parent, icon, text, cmd):
            if icon is not None:
                return tk.Button(parent, image=icon, command=cmd, relief=tk.FLAT)
            # fallback texte si l'icône n'est pas trouvée
            return tk.Button(parent, text=text, command=cmd, width=2, relief=tk.FLAT)

        _make_btn(toolbar, self.icon_scen_new,   "N", self._new_empty_scenario).pack(side=tk.LEFT, padx=1)
        _make_btn(toolbar, self.icon_scen_open,  "O", self._scenario_load_dialog).pack(side=tk.LEFT, padx=1)
        _make_btn(toolbar, self.icon_scen_props, "P", self._scenario_edit_properties).pack(side=tk.LEFT, padx=1)
        _make_btn(toolbar, self.icon_scen_save,  "S", self._scenario_save_dialog).pack(side=tk.LEFT, padx=1)
        _make_btn(toolbar, self.icon_scen_dup,   "D", self._scenario_duplicate).pack(side=tk.LEFT, padx=1)
        _make_btn(toolbar, self.icon_scen_del,   "X", self._scenario_delete).pack(side=tk.LEFT, padx=1)

        scen_lb_frame = tk.Frame(scen_frame)
        scen_lb_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.scenario_listbox = tk.Listbox(scen_lb_frame, width=34, exportselection=False, height=6)
        self.scenario_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scen_scroll = tk.Scrollbar(scen_lb_frame, orient="vertical", command=self.scenario_listbox.yview)
        scen_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.scenario_listbox.configure(yscrollcommand=scen_scroll.set)
        self.scenario_listbox.bind("<<ListboxSelect>>", self._on_scenario_select)

        pw.add(scen_frame, minsize=80)  # hauteur mini pour la liste des scénarios

        # Remplir la liste des scénarios existants (pour l'instant : le manuel)
        self._refresh_scenario_listbox()

    def _refresh_scenario_listbox(self):
        """Met à jour la liste visible des scénarios dans le panneau de gauche."""
        if not hasattr(self, "scenario_listbox"):
            return
        lb = self.scenario_listbox
        lb.delete(0, tk.END)

        for i, scen in enumerate(self.scenarios):
            prefix = "[M]" if scen.source_type == "manual" else "[A]"
            label = f"{prefix} {scen.name}"
            if scen.status != "complete":
                label += f" ({scen.status})"
            lb.insert(tk.END, label)

        # Sélectionner le scénario actif si possible
        if 0 <= self.active_scenario_index < len(self.scenarios):
            lb.selection_clear(0, tk.END)
            lb.selection_set(self.active_scenario_index)
            lb.see(self.active_scenario_index)

    def _update_triangle_listbox_colors(self):
        """
        Met à jour la couleur des entrées de la listbox des triangles
        en fonction de leur utilisation dans le scénario actif.
        Triangles utilisés → grisés, triangles disponibles → noir.
        """
        if not hasattr(self, "listbox"):
            return
        lb = self.listbox
        try:
            size = lb.size()
        except Exception:
            return

        for idx in range(size):
            try:
                txt = lb.get(idx)
            except Exception:
                continue

            tri_id = None
            try:
                m = re.match(r"\s*(\d+)\.", str(txt))
                if m:
                    tri_id = int(m.group(1))
            except Exception:
                tri_id = None

            if tri_id is not None and tri_id in self._placed_ids:
                # Triangle déjà posé dans le scénario → grisé
                try:
                    lb.itemconfig(idx, fg="gray50")
                except Exception:
                    pass
            else:
                # Triangle disponible
                try:
                    lb.itemconfig(idx, fg="black")
                except Exception:
                    pass

    def _on_scenario_select(self, event=None):
        """Callback quand l'utilisateur sélectionne un scénario dans la liste."""
        if not hasattr(self, "scenario_listbox"):
            return
        sel = self.scenario_listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        self._set_active_scenario(idx)

    def _set_active_scenario(self, index: int):
        """
        Bascule vers le scénario d'index donné.
        Pour l'instant, on n'a qu'un scénario manuel qui partage les mêmes
        structures _last_drawn / groups, mais cette méthode sera utilisée
        plus tard pour les scénarios automatiques (copies séparées).
        """
        if index < 0 or index >= len(self.scenarios):
            return
        if index == self.active_scenario_index:
            return

        scen = self.scenarios[index]
        self.active_scenario_index = index

        # Rattacher les structures géométriques du scénario courant
        self._last_drawn = scen.last_drawn
        self.groups = scen.groups

        # Recalibrer _next_group_id si besoin
        try:
            self._next_group_id = (max(self.groups.keys()) + 1) if self.groups else 1
        except Exception:
            self._next_group_id = 1

        # Recalcule la liste des triangles déjà utilisés pour ce scénario
        try:
            self._placed_ids = {
                int(t["id"]) for t in self._last_drawn
                if t.get("id") is not None
            }
        except Exception:
            self._placed_ids = set()
        self._update_triangle_listbox_colors()

        # Invalider le cache de pick et redessiner
        self._invalidate_pick_cache()

        # Fit à l'écran systématique lors de la sélection d'un scénario
        # (utile pour voir immédiatement l’ensemble de l’assemblage)
        if self._last_drawn:
            # _fit_to_view redessine déjà via _redraw_from()
            self._fit_to_view(self._last_drawn)
        else:
            self._redraw_from(self._last_drawn)
        self._redraw_overlay_only()

        # Mettre à jour la sélection visuelle dans la liste (au cas d'appel programmatique)
        try:
            if hasattr(self, "scenario_listbox"):
                self.scenario_listbox.selection_clear(0, tk.END)
                self.scenario_listbox.selection_set(self.active_scenario_index)
                self.scenario_listbox.see(self.active_scenario_index)
        except Exception:
            pass

        try:
            self.status.config(text=f"Scénario actif : {scen.name}")
        except Exception:
            pass

    def _new_empty_scenario(self):
        """
        Crée un nouveau scénario *vide* (sans triangles assemblés),
        l'ajoute à la liste et le rend actif.
        Les triangles sources (dans la listbox) restent évidemment disponibles.
        """
        # Nom par défaut : "Scénario N" (N = nombre total de scénarios après ajout)
        new_index = len(self.scenarios)  # l'index qu'il prendra une fois append
        name = f"Scénario {new_index + 1}"

        scen = ScenarioAssemblage(
            name=name,
            source_type="manual",
            algo_id=None,
            tri_ids=[],
        )
        # Scénario vide : nouvelles structures indépendantes
        scen.last_drawn = []
        scen.groups = {}

        self.scenarios.append(scen)
        # Bascule sur ce nouveau scénario
        self._set_active_scenario(new_index)
        # Rafraîchir la liste visible
        self._refresh_scenario_listbox()

        try:
            self.status.config(text=f"Nouveau scénario créé : {scen.name}")
        except Exception:
            pass

    def _scenario_edit_properties(self):
        """
        Modifie les propriétés du scénario actif (pour l'instant : uniquement le nom).
        """
        if not self.scenarios:
            return
        idx = self.active_scenario_index
        if idx < 0 or idx >= len(self.scenarios):
            return
        scen = self.scenarios[idx]
        new_name = simpledialog.askstring(
            "Propriétés du scénario",
            "Nom du scénario :",
            initialvalue=scen.name,
            parent=self,
        )
        if new_name is None:
            return  # annulé
        new_name = new_name.strip()
        if not new_name:
            return
        scen.name = new_name
        self._refresh_scenario_listbox()
        try:
            self.status.config(text=f"Nom du scénario mis à jour : {scen.name}")
        except Exception:
            pass

    def _scenario_duplicate(self):
        """
        Duplique le scénario actif dans un nouveau scénario indépendant.
        Les triangles et groupes sont copiés (deepcopy).
        """
        if not self.scenarios:
            return
        idx = self.active_scenario_index
        if idx < 0 or idx >= len(self.scenarios):
            return
        src = self.scenarios[idx]

        base_name = src.name or "Scénario"
        new_name = f"{base_name} (copie)"
        n = 2
        # garantir un nom unique
        while any(s.name == new_name for s in self.scenarios):
            new_name = f"{base_name} (copie {n})"
            n += 1

        new_index = len(self.scenarios)
        dup = ScenarioAssemblage(
            name=new_name,
            source_type=src.source_type,
            algo_id=src.algo_id,
            tri_ids=list(src.tri_ids),
        )
        # copies indépendantes
        dup.last_drawn = copy.deepcopy(src.last_drawn)
        dup.groups = copy.deepcopy(src.groups)
        dup.status = src.status

        self.scenarios.append(dup)
        self._set_active_scenario(new_index)
        self._refresh_scenario_listbox()
        try:
            self.status.config(text=f"Scénario dupliqué : {dup.name}")
        except Exception:
            pass

    def _scenario_delete(self):
        """
        Supprime le scénario actif.
        On interdit la suppression du scénario manuel de base pour garder un point d'appui.
        """
        if len(self.scenarios) <= 1:
            messagebox.showinfo("Supprimer le scénario",
                                "Impossible de supprimer le dernier scénario.")
            return
        idx = self.active_scenario_index
        if idx < 0 or idx >= len(self.scenarios):
            return

        scen = self.scenarios[idx]
        if idx == 0 and scen.source_type == "manual":
            messagebox.showinfo("Supprimer le scénario",
                                "Le scénario manuel ne peut pas être supprimé.")
            return

        if not messagebox.askyesno(
            "Supprimer le scénario",
            f"Supprimer le scénario « {scen.name} » ?",
            parent=self,
        ):
            return

        # Retirer le scénario
        self.scenarios.pop(idx)
        # Choisir le nouveau scénario actif (celui d'avant si possible)
        if idx >= len(self.scenarios):
            idx = len(self.scenarios) - 1

        self._set_active_scenario(idx)
        self._refresh_scenario_listbox()
        try:
            self.status.config(text=f"Scénario supprimé : {scen.name}")
        except Exception:
            pass

    def _build_canvas(self, parent):
        # Conteneur de droite : canvas (haut, expansible) + panel dico (bas, hauteur fixe)
        self.rightPane = tk.Frame(parent)
        self.rightPane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas d’affichage des triangles
        self.canvas = tk.Canvas(self.rightPane, bg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Redessiner l'overlay si la taille du canvas change
        self.canvas.bind("<Configure>", lambda e: (self._redraw_overlay_only(), self._invalidate_pick_cache()))

        # Panel dico (placeholder à hauteur fixe, prêt pour intégrer la grille)
        self.dicoPanel = tk.Frame(self.rightPane, height=self.dico_panel_height,
                                  bd=1, relief=tk.SUNKEN, bg="#f3f3f3")
        # empêcher le panel de rétrécir sur le contenu
        self.dicoPanel.pack_propagate(False)
        # Pack initial seulement si le toggle est actif
        if self.show_dico_panel.get():
            self.dicoPanel.pack(side=tk.BOTTOM, fill=tk.X)
            # Placeholder visuel (sera remplacé par _build_dico_grid)
            tk.Label(self.dicoPanel, text="Dictionnaire — (grille à intégrer)",
                     bg="#f3f3f3", anchor="w").pack(fill=tk.X, padx=8, pady=4)

        # Menu contextuel
        self._ctx_menu = tk.Menu(self, tearoff=0)
        self._ctx_menu.add_command(label="Supprimer", command=self._ctx_delete_group)
        self._ctx_menu.add_command(label="Pivoter", command=self._ctx_rotate_selected)
        self._ctx_menu.add_command(label="Inverser", command=self._ctx_flip_selected)
        self._ctx_menu.add_command(label="OL=0°", command=self._ctx_orient_OL_north)

        # Mémoriser l'index de l'entrée "OL=0°" pour pouvoir la (dés)activer au vol
        self._ctx_idx_ol0 = self._ctx_menu.index("end")

        # Séparateur et zone dynamique pour les actions "mot"
        self._ctx_menu.add_separator()
        self._ctx_idx_words_start = self._ctx_menu.index("end")  # point d'ancrage
        # entrées placeholders (seront remplacées dynamiquement)
        self._ctx_menu.add_command(label="Ajouter …", state="disabled")
        self._ctx_menu.add_command(label="Effacer …", state="disabled")

        # DEBUG: F9 pour activer/désactiver le filtrage de chevauchement au highlight
        # (bind_all pour capter même si le focus clavier n'est pas explicitement sur le canvas)
        self.bind_all("<F9>", self._toggle_skip_overlap_highlight)
        # Premier rendu de l'horloge (overlay)
        self._draw_clock_overlay()

    def _toggle_dico_panel(self):
        """
        Affiche / cache le panneau dictionnaire (combo + liste + grille)
        en fonction de self.show_dico_panel (toggle du menu Visualisation).
        """
        if not hasattr(self, "dicoPanel"):
            return

        show = bool(self.show_dico_panel.get())
        if show:
            # Si le panneau n'est pas déjà packé, on le repack en bas
            if not self.dicoPanel.winfo_ismapped():
                self.dicoPanel.pack(side=tk.BOTTOM, fill=tk.X)
                # Si la grille n'a jamais été construite, on la (re)construit
                if not getattr(self, "dicoSheet", None):
                    try:
                        self._build_dico_grid()
                    except Exception:
                        pass
        else:
            # Cacher le panneau (sans le détruire, pour pouvoir le réafficher)
            try:
                self.dicoPanel.pack_forget()
            except Exception:
                pass

    def _toggle_clock_overlay(self):
        """Affiche ou cache le compas horaire (overlay horloge)."""
        if not getattr(self, "canvas", None):
            return
        # Effacer systématiquement l'overlay courant
        try:
            self.canvas.delete("clock_overlay")
        except Exception:
            pass
        # Si l'option est active, on redessine l'horloge (sans toucher aux triangles)
        if self.show_clock_overlay.get():
            self._draw_clock_overlay()

    # -- pick cache helpers ---------------------------------------------------
    def _invalidate_pick_cache(self):
        """À appeler dès que zoom/offset ou _last_drawn peuvent changer."""
        self._pick_cache_valid = False

    def _ensure_pick_cache(self):
        """Reconstruit le pick-cache si nécessaire (appel paresseux côté input)."""
        if not getattr(self, "_pick_cache_valid", False):
            self._rebuild_pick_cache()

    def _rebuild_pick_cache(self):
        """Reconstruit les polygones écran utilisés pour le hit-test."""
        if not self._last_drawn:
            self._pick_cache_valid = True
            return
        Z = float(getattr(self, "zoom", 1.0))
        Ox, Oy = (float(self.offset[0]), float(self.offset[1]))
        def W2S(p):
            # monde -> écran (canvas)
            return (Ox + p[0] * Z, Oy - p[1] * Z)
        for t in self._last_drawn:
            P = t.get("pts", {})
            O = P.get("O"); B = P.get("B"); L = P.get("L")
            if O is None or B is None or L is None:
                t["_pick_poly"] = None
                t["_pick_pts"] = {}
                continue
            Os = W2S(O); Bs = W2S(B); Ls = W2S(L)
            t["_pick_pts"]  = {"O": Os, "B": Bs, "L": Ls}
            t["_pick_poly"] = [Os, Bs, Ls]
        self._pick_cache_valid = True


    # centralisation des bindings canvas/clavier --
    def _bind_canvas_handlers(self):
        """(Ré)applique tous les bindings nécessaires au canvas et au clavier.
        À appeler après création du canvas ET après un chargement de scénario."""
        if not getattr(self, "canvas", None):
            return
        # Purge défensive pour éviter les doublons (Tk ignore les doublons, mais on nettoie)
        self.canvas.unbind("<MouseWheel>")
        self.canvas.unbind("<Button-4>")
        self.canvas.unbind("<Button-5>")
        self.canvas.unbind("<ButtonPress-2>")
        self.canvas.unbind("<B2-Motion>")
        self.canvas.unbind("<ButtonRelease-2>")
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.unbind("<Motion>")
        self.canvas.unbind("<Button-3>")

        # Zoom (wheel)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)   # Linux
        self.canvas.bind("<Button-5>", self._on_mousewheel)   # Linux
        # Pan (middle)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        # Drag/Move (left)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_left_down)
        self.canvas.bind("<B1-Motion>", self._on_canvas_left_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_left_up)
        # Suivi hover / drag preview
        self.canvas.bind("<Motion>", self._on_canvas_motion_update_drag)
        # Menu contextuel
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)

        # Clavier (CTRL pour déconnexion ; F9 pour debug)
        self.bind_all("<KeyPress-Control_L>", self._on_ctrl_down)
        self.bind_all("<KeyPress-Control_R>", self._on_ctrl_down)
        self.bind_all("<KeyRelease-Control_L>", self._on_ctrl_up)
        self.bind_all("<KeyRelease-Control_R>", self._on_ctrl_up)
        self.bind_all("<F9>", self._toggle_skip_overlap_highlight)

    # --- conversions utilitaires (utilisées par le pick) ---
    def _screen_to_world(self, x, y):
        Z = float(getattr(self, "zoom", 1.0))
        Ox, Oy = (float(self.offset[0]), float(self.offset[1]))
        return ((x - Ox) / Z, (Oy - y) / Z)


    # =========================
    #  SCENARIO: SAVE / LOAD
    # =========================
    def _scenario_save_dialog(self):
        """Boîte de dialogue pour enregistrer un scénario en XML."""
        # nom par défaut daté dans le dossier 'scenario'
        ts = _dt.datetime.now().strftime("scenario-%Y%m%d-%H%M.xml")
        initial = os.path.join(self.scenario_dir, ts)
        path = filedialog.asksaveasfilename(
            title="Enregistrer le scénario",
            defaultextension=".xml",
            initialfile=os.path.basename(initial),
            initialdir=self.scenario_dir,
            filetypes=[("Scenario XML", "*.xml"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return
        try:
            self.save_scenario_xml(path)
            self.status.config(text=f"Scénario enregistré dans {path}")
        except Exception as e:
            messagebox.showerror("Enregistrer le scénario", str(e))


    def _load_scenario_into_new_scenario(self, path: str):
        """
        Charge un fichier scénario XML dans un *nouveau* scénario.
        - Le scénario courant n'est pas modifié.
        - Le nouveau scénario porte le nom du fichier XML (sans extension).
        """
        # Nom lisible dérivé du fichier
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        if not name:
            name = base

        prev_index = self.active_scenario_index
        new_index = len(self.scenarios)

        scen = ScenarioAssemblage(
            name=name,
            source_type="manual",
            algo_id=None,
            tri_ids=[],
        )
        # Structures indépendantes pour ce scénario
        scen.last_drawn = []
        scen.groups = {}

        self.scenarios.append(scen)

        try:
            # On bascule sur ce nouveau scénario AVANT de charger,
            # ainsi load_scenario_xml() écrit bien dans ses structures.
            self._set_active_scenario(new_index)
            self.load_scenario_xml(path)
        except Exception:
            # En cas d'erreur, on nettoie le scénario incomplet
            try:
                self.scenarios.pop(new_index)
            except Exception:
                pass
            # On revient au scénario précédent (s'il existe encore)
            try:
                if self.scenarios:
                    if 0 <= prev_index < len(self.scenarios):
                        self._set_active_scenario(prev_index)
                    else:
                        self._set_active_scenario(len(self.scenarios) - 1)
            except Exception:
                pass
            self._refresh_scenario_listbox()
            raise

        # Succès : rafraîchir la liste et le statut
        self._refresh_scenario_listbox()
        try:
            self.status.config(text=f"Scénario importé : {scen.name}")
        except Exception:
            pass

    def _scenario_load_dialog(self):
        """Boîte de dialogue pour charger un scénario XML."""
        path = filedialog.askopenfilename(
            title="Charger un scénario",
            initialdir=self.scenario_dir,
            filetypes=[("Scenario XML", "*.xml"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return
        try:
            self._load_scenario_into_new_scenario(path)
        except Exception as e:
            messagebox.showerror("Charger le scénario", str(e))
    
    def _rebuild_scenario_file_list_menu(self):
        """Recrée la liste des scénarios (XML) disponibles dans le menu Scénario."""
        m = self.menu_scenario
        if m is None:
            return
        # supprimer tout ce qui suit l’ancre
        end = m.index("end")
        while end is not None and end > self._menu_scenario_files_anchor:
            m.delete(end)
            end = m.index("end")
        # re-remplir
        try:
            files = [f for f in os.listdir(self.scenario_dir) if f.lower().endswith(".xml")]
        except Exception:
            files = []
        if not files:
            m.add_command(label="(aucun scénario dans 'scenario')", state="disabled")
            return
        files.sort(key=str.lower)
        for fname in files:
            full = os.path.join(self.scenario_dir, fname)
            # Chaque fichier XML crée désormais un *nouveau* scénario
            m.add_command(label=fname, command=lambda p=full: self._load_scenario_into_new_scenario(p))

    def _pt_to_xml(self, p):
        return f"{float(p[0]):.9g},{float(p[1]):.9g}"

    def _xml_to_pt(self, s):
        x, y = str(s).split(",")
        return np.array([float(x), float(y)], dtype=float)

    def save_scenario_xml(self, path: str):
        """
        Sauvegarde 'robuste' (v1) :
          - source excel, view (zoom/offset), clock (pos + hm),
          - ids restants (listbox),
          - triangles affichés (id, mirrored, points monde O/B/L, group),
          - groupes (membres best-effort),
          - associations triangle→mot (word,row,col).
        """
        root = ET.Element("scenario", {
            "version": "1",
            "saved_at": _dt.datetime.now().isoformat(timespec="seconds")
        })
        # source
        ET.SubElement(root, "source", {
            "excel": os.path.abspath(self.excel_path) if getattr(self, "excel_path", None) else ""
        })
        # view
        view = ET.SubElement(root, "view", {
            "zoom": f"{float(getattr(self, 'zoom', 1.0)):.6g}",
            "offset_x": f"{float(self.offset[0]) if hasattr(self,'offset') else 0.0:.6g}",
            "offset_y": f"{float(self.offset[1]) if hasattr(self,'offset') else 0.0:.6g}",
        })
        # clock
        ET.SubElement(root, "clock", {
            "x": f"{float(self._clock_cx) if self._clock_cx is not None else 0.0:.6g}",
            "y": f"{float(self._clock_cy) if self._clock_cy is not None else 0.0:.6g}",
            "hour": f"{int(self._clock_state.get('hour', 0))}",
            "minute": f"{int(self._clock_state.get('minute', 0))}",
            "label": str(self._clock_state.get("label","")),
        })
        # listbox: ids restants
        lb = ET.SubElement(root, "listbox")
        try:
            for i in range(self.listbox.size()):
                txt = self.listbox.get(i)
                m = re.match(r"\s*(\d+)\.", str(txt))
                if m:
                    ET.SubElement(lb, "tri", {"id": m.group(1)})
        except Exception:
            pass
        # groupes (best-effort)
        groups_xml = ET.SubElement(root, "groups")
        try:
            for gid, gdata in (self.groups or {}).items():
                g_el = ET.SubElement(groups_xml, "group", {"id": str(gid)})
                # si une API _group_nodes existe, on l’utilise
                members = []
                try:
                    members = [nd.get("tid") for nd in self._group_nodes(gid) if nd.get("tid") is not None]
                except Exception:
                    # fallback: scanner _last_drawn via le champ runtime correct 'group_id'
                    for idx, t in enumerate(self._last_drawn or []):
                        if int(t.get("group_id", 0)) == int(gid):
                            members.append(idx)
                for tid in members:
                    if tid is None:
                        continue
                    ET.SubElement(g_el, "member", {"tid": str(tid)})
        except Exception:
            pass
        # triangles posés
        tris_xml = ET.SubElement(root, "triangles")
        for t in (self._last_drawn or []):
            tri_el = ET.SubElement(tris_xml, "triangle", {
                "id": str(t.get("id", "")),
                "mirrored": "1" if t.get("mirrored", False) else "0",
                # on sérialise la valeur runtime correcte
                "group": str(t.get("group_id", 0)),
            })
            P = t.get("pts", {})
            for key in ("O","B","L"):
                if key in P:
                    ET.SubElement(tri_el, key).text = self._pt_to_xml(P[key])
        # mots associés
        words_xml = ET.SubElement(root, "words")
        for tri_id, info in (self._tri_words or {}).items():
            ET.SubElement(words_xml, "w", {
                "tri_id": str(tri_id),
                "row": str(info.get("row","")),
                "col": str(info.get("col","")),
                "text": str(info.get("word","")),
            })
        # écrire
        tree = ET.ElementTree(root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tree.write(path, encoding="utf-8", xml_declaration=True)

    def load_scenario_xml(self, path: str):
        """
        Recharge un scénario v1 :
          - tente de recharger le fichier Excel source,
          - restaure vue, horloge, listbox, triangles posés (+mots), groupes (best-effort),
          - redessine.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        if root.tag != "scenario":
            raise ValueError("Fichier scénario invalide (balise racine).")
        # 1) Excel source
        src = root.find("source")
        excel = src.get("excel") if src is not None else ""
        if excel and os.path.isfile(excel):
            self.load_excel(excel)
        # 2) vue (restaurer AVANT toute reconstruction pour que les conversions monde<->écran soient cohérentes)
        view = root.find("view")
        if view is not None:
            try:
                self.zoom = float(view.get("zoom", self.zoom))
            except Exception: pass
            try:
                ox = float(view.get("offset_x", self.offset[0] if hasattr(self,"offset") else 0.0))
                oy = float(view.get("offset_y", self.offset[1] if hasattr(self,"offset") else 0.0))
                self.offset = np.array([ox, oy], dtype=float)
            except Exception:
                pass
        # État d’interaction propre (purge complète UI)
        self._sel = {"mode": None}
        self._clear_nearest_line()
        self._clear_edge_highlights()
        self._hide_tooltip()
        self._ctx_target_idx = None
        self._edge_choice = None
        self._drag_preview_id = None
        self.canvas.delete("preview")

        # 3) horloge
        clock = root.find("clock")
        if clock is not None:
            try: self._clock_cx = float(clock.get("x", "0"))
            except Exception: pass
            try: self._clock_cy = float(clock.get("y", "0"))
            except Exception: pass
            try:
                h = int(clock.get("hour", "0")); m = int(clock.get("minute", "0"))
                lbl = clock.get("label","")
                self._clock_state.update({"hour": h, "minute": m, "label": lbl})
            except Exception:
                pass
        # 4) listbox (ids restants)
        lb = root.find("listbox")
        if lb is not None:
            try:
                # ids des triangles encore présents dans la listbox au moment de la sauvegarde
                remain_ids = [int(e.get("id")) for e in lb.findall("tri") if e.get("id")]
                self.listbox.delete(0, tk.END)
                # reconstruire la liste depuis df en conservant l'ordre original
                if getattr(self, "df", None) is not None and not self.df.empty:
                    if remain_ids:
                        wanted = set(remain_ids)
                        for _, r in self.df.iterrows():
                            tid = int(r["id"])
                            if tid in wanted:
                                self.listbox.insert(
                                    tk.END, f"{tid:02d}. B:{r['B']}  L:{r['L']}"
                                )
                    else:
                        # listbox vide dans le XML (ancien scénario ou bug) :
                        # on retombe sur le comportement par défaut = tous les triangles.
                        for _, r in self.df.iterrows():
                            tid = int(r["id"])
                            self.listbox.insert(
                                tk.END, f"{tid:02d}. B:{r['B']}  L:{r['L']}"
                            )
            except Exception:
                pass
        # 5) triangles posés (monde)
        # On reconstruit la liste en vidant d'abord celle du scénario actif,
        # sans casser la référence partagée avec scen.last_drawn.
        self._last_drawn.clear()
        tris_xml = root.find("triangles")
        if tris_xml is not None:
            for t_el in tris_xml.findall("triangle"):
                try:
                    tid = int(t_el.get("id"))
                except Exception:
                    continue
                mirrored = (t_el.get("mirrored","0") == "1")
                group_id = int(t_el.get("group","0"))
                P = {}
                for k in ("O","B","L"):
                    n = t_el.find(k)
                    if n is not None and n.text:
                        P[k] = self._xml_to_pt(n.text)
                # Ne plus écrire 'group' (inconnu du runtime) ; initialiser les champs officiels
                item = {"id": tid, "pts": P, "mirrored": mirrored}
                item["group_id"] = None
                item["group_pos"] = None
                self._last_drawn.append(item)

        # 5bis) compléter les 'labels' manquants (compat v1: non stockés dans le XML)
        try:
            if getattr(self, "df", None) is not None and not self.df.empty:
                # dictionnaire id -> (B, L) sous forme de chaînes
                _by_id = {int(r["id"]): (str(r["B"]), str(r["L"])) for _, r in self.df.iterrows()}
                for t in self._last_drawn:
                    if "labels" not in t or not t["labels"]:
                        b, l = _by_id.get(int(t.get("id", -1)), ("", ""))
                        t["labels"] = ("Bourges", b, l)
            else:
                # DF absent : fallback neutre
                for t in self._last_drawn:
                    if "labels" not in t or not t["labels"]:
                        t["labels"] = ("Bourges", "", "")
        except Exception:
            # sécurité : ne jamais laisser 'labels' absent
            for t in self._last_drawn:
                if "labels" not in t or not t["labels"]:
                    t["labels"] = ("Bourges", "", "")

        # 5ter) tenir à jour les ids déjà posés (pour cohérence listbox / drag)
        try:
            self._placed_ids = {int(t["id"]) for t in self._last_drawn}
        except Exception:
            self._placed_ids = set()
        # Mettre à jour l'affichage de la listbox en fonction des triangles déjà posés
        self._update_triangle_listbox_colors()

        # 6) mots associés
        self._tri_words = {}
        words_xml = root.find("words")
        if words_xml is not None:
            for w in words_xml.findall("w"):
                try:
                    tid = int(w.get("tri_id"))
                except Exception:
                    continue
                self._tri_words[tid] = {
                    "word": w.get("text",""),
                    "row": int(w.get("row","0")) if w.get("row") else 0,
                    "col": int(w.get("col","0")) if w.get("col") else 0,
                }

        # 7) groupes (reconstruction complète: nodes + bboxes)
        # Même logique : on réutilise le dict existant pour préserver
        # le lien avec manual.groups.
        self.groups.clear()
        groups_xml = root.find("groups")
        if groups_xml is not None:
            for g_el in groups_xml.findall("group"):
                try:
                    gid = int(g_el.get("id"))
                except Exception:
                    continue
                nodes = []
                for mem in g_el.findall("member"):
                    # compat: on accepte 'tid' (index dans _last_drawn) OU 'id' (id triangle)
                    tidx = None
                    if mem.get("tid") is not None:
                        try:
                            tidx = int(mem.get("tid"))
                        except Exception:
                            tidx = None
                    elif mem.get("id") is not None:
                        # recherche de l’index correspondant à l’id
                        try:
                            tri_id = int(mem.get("id"))
                            for k, t in enumerate(self._last_drawn):
                                if int(t.get("id")) == tri_id:
                                    tidx = k
                                    break
                        except Exception:
                            tidx = None
                    if tidx is None or not (0 <= tidx < len(self._last_drawn)):
                        continue
                    # Nodes au format runtime + marquage triangle: group_id/group_pos
                    nodes.append({"tid": tidx, "vkey_in": None, "vkey_out": None})
                    self._last_drawn[tidx]["group_id"]  = gid
                    self._last_drawn[tidx]["group_pos"] = len(nodes) - 1
                    # Hygiène: supprimer l’ancien champ 'group' s’il subsiste
                    if "group" in self._last_drawn[tidx]:
                        del self._last_drawn[tidx]["group"]
                self.groups[gid] = {"id": gid, "nodes": nodes, "bbox": None}
                self._recompute_group_bbox(gid)

        # Nettoyage global de compatibilité : purger toute trace résiduelle de 'group'
        for _t in self._last_drawn:
            if "group" in _t:
                del _t["group"]
        # sécurité: prochain id de groupe
        try:
            self._next_group_id = (max(self.groups.keys()) + 1) if self.groups else 1
        except Exception:
            self._next_group_id = 1

        # 8) sélection et aides reset
        self._sel = {"mode": None}
        self._clear_nearest_line()
        self._clear_edge_highlights()

        # 9) réappliquer les bindings (utile si le canvas a été recréé ou si Tk a perdu des liaisons)
        self._bind_canvas_handlers()

        # 10) redraw complet avec la vue restaurée + overlay
        self._redraw_from(self._last_drawn)
        self._redraw_overlay_only()
        # [H6] reconstruire le pick-cache avec la vue effectivement restaurée
        try:
            self._rebuild_pick_cache()
            self._pick_cache_valid = True
        except Exception:
            self._pick_cache_valid = False

        # focus + rebind défensif (tooltips/drag)
        self.canvas.focus_set()
        self._bind_canvas_handlers()

        # DEBUG D1 — vérifier les champs de groupe après load
        for i, t in enumerate(self._last_drawn):
            gid = t.get("group_id", None)
            g   = t.get("group", None)
            print(f"[D1] tid={i} id={t.get('id')} group_id={gid} group={g}")

    # ---------- Horloge : test de hit ----------
    def _is_in_clock(self, x, y, pad=10):
        """Vrai si (x,y) (coords canvas) est à l'intérieur du disque de l'horloge (+marge)."""
        if self._clock_cx is None or self._clock_cy is None:
            return False
        dx = x - self._clock_cx
        dy = y - self._clock_cy
        return (dx*dx + dy*dy) <= (self._clock_R + pad) ** 2

    # ---------- Tooltip helpers ----------
    def _show_tooltip_at_center(self, text: str, cx_canvas: float, cy_canvas: float):
        """Affiche/MAJ le tooltip en le **centrant** sur (cx_canvas, cy_canvas) (coords CANVAS)."""
        if not text:
            self._hide_tooltip(); return
        if self._tooltip is None or not self._tooltip.winfo_exists():
            self._tooltip = tk.Toplevel(self)
            self._tooltip.wm_overrideredirect(True)
            try:
                self._tooltip.attributes("-topmost", True)
            except Exception:
                pass
            self._tooltip_label = tk.Label(self._tooltip, text=text, bg="#ffffe0",
                                           relief="solid", borderwidth=1, font=("Arial", 9),
                                           justify="left", anchor="w")
            self._tooltip_label.pack(ipadx=4, ipady=2)
        else:
            self._tooltip_label.config(text=text, justify="left", anchor="w")
        # Mesurer la taille réelle du tooltip
        self._tooltip.update_idletasks()
        tw = max(1, int(self._tooltip.winfo_width()))
        th = max(1, int(self._tooltip.winfo_height()))
        # Convertir coords CANVAS -> écran et centrer
        base_x = self.canvas.winfo_rootx()
        base_y = self.canvas.winfo_rooty()
        x = int(base_x + cx_canvas - tw/2)
        y = int(base_y + cy_canvas - th/2)
        c_w = int(self.canvas.winfo_width())
        c_h = int(self.canvas.winfo_height())
        min_x = base_x
        max_x = base_x + c_w - tw
        min_y = base_y
        max_y = base_y + c_h - th
        if max_x < min_x:
            max_x = min_x
        if max_y < min_y:
            max_y = min_y
        x = max(min_x, min(x, max_x))
        y = max(min_y, min(y, max_y))
        self._tooltip.wm_geometry(f"+{x}+{y}")

    def _hide_tooltip(self):
        if self._tooltip is not None and self._tooltip.winfo_exists():
            try: self._tooltip.destroy()
            except Exception: pass
        self._tooltip = None
        self._tooltip_label = None

    # ---------- Config (JSON) ----------
    def loadAppConfig(self):
        """Charge la config JSON (best-effort)."""
        self.appConfig = {}
        try:
            path = getattr(self, "config_path", "")
            if not path:
                return
            if not os.path.isfile(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.appConfig = data
        except Exception:
            # Jamais bloquant : si la config est corrompue on repart de zéro.
            self.appConfig = {}

    def saveAppConfig(self):
        """Sauvegarde la config JSON (best-effort)."""
        try:
            path = getattr(self, "config_path", "")
            if not path:
                return
            cfg_dir = os.path.dirname(path)
            if cfg_dir:
                os.makedirs(cfg_dir, exist_ok=True)
            # écriture atomique (évite un fichier vide si un souci survient)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(getattr(self, "appConfig", {}) or {}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            pass

    def getAppConfigValue(self, key: str, default=None):
        try:
            return (getattr(self, "appConfig", {}) or {}).get(key, default)
        except Exception:
            return default

    def setAppConfigValue(self, key: str, value):
        try:
            if not hasattr(self, "appConfig") or self.appConfig is None:
                self.appConfig = {}
            self.appConfig[key] = value
            self.saveAppConfig()
        except Exception:
            pass

    def autoLoadTrianglesFileAtStartup(self):
        """Recharge au démarrage le dernier fichier triangles ouvert."""
        # 1) Dernier fichier explicitement chargé
        last = self.getAppConfigValue("lastTriangleExcel", "")
        if last and os.path.isfile(str(last)):
            try:
                self.load_excel(str(last))
                return
            except Exception:
                # On tente un fallback si le fichier est devenu invalide
                pass
        # 2) Fallback : data/triangle.xlsx
        try:
            default = os.path.join(getattr(self, "data_dir", ""), "triangle.xlsx")
            if default and os.path.isfile(default):
                self.load_excel(default)
        except Exception:
            pass

    # ---------- Chargement Excel ----------
    def open_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if path:
            self.load_excel(path)

    @staticmethod
    def _norm(s: str) -> str:
        import unicodedata, re
        s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()
        return re.sub(r"[^a-z0-9]+", "", s)

    @staticmethod
    def _find_header_row(df0: pd.DataFrame) -> int:
        for i in range(min(12, len(df0))):
            row_norm = [TriangleViewerManual._norm(x) for x in df0.iloc[i].tolist()]
            if any("ouverture" in x for x in row_norm) and \
               any("base"      in x for x in row_norm) and \
               any("lumiere"   in x for x in row_norm):
                return i
        raise KeyError("Impossible de détecter l'entête ('Ouverture', 'Base', 'Lumière').")

    @staticmethod
    def _build_df(df: pd.DataFrame) -> pd.DataFrame:
        cmap = {TriangleViewerManual._norm(c): c for c in df.columns}
        col_id = cmap.get("rang") or cmap.get("id")
        col_B  = cmap.get("base")
        col_L  = cmap.get("lumiere")
        col_OB = cmap.get("ouverturebase")
        col_OL = cmap.get("ouverturelumiere")
        col_BL = cmap.get("lumierebase")
        col_OR = cmap.get("orientation")  # colonne Orientation (CCW/CW/COL)  
        missing = [n for n,c in {
            "Base":col_B, "Lumière":col_L, "Ouverture-Base":col_OB,
            "Ouverture-Lumière":col_OL, "Lumière-Base":col_BL
        }.items() if c is None]
        if missing:
            raise KeyError("Colonnes manquantes: " + ", ".join(missing))
        out = pd.DataFrame({
            "id": df[col_id] if col_id else range(1, len(df)+1),
            "B":  df[col_B],
            "L":  df[col_L],
            "len_OB": pd.to_numeric(df[col_OB], errors="coerce"),
            "len_OL": pd.to_numeric(df[col_OL], errors="coerce"),
            "len_BL": pd.to_numeric(df[col_BL], errors="coerce"),
            # Orientation normalisée (par défaut CCW si absent)
            "orient": (
                df[col_OR].astype(str).str.upper().str.strip()
                if col_OR else pd.Series(["CCW"] * len(df))
            ),
        }).dropna(subset=["len_OB","len_OL","len_BL"]).sort_values("id")
        return out.reset_index(drop=True)

    def load_excel(self, path: str):
        df0 = pd.read_excel(path, header=None)
        header_row = self._find_header_row(df0)
        df = pd.read_excel(path, header=header_row)
        self.df = self._build_df(df)
        path_norm = os.path.normpath(os.path.abspath(path))
        self.excel_path = path_norm
        self.setAppConfigValue("lastTriangleExcel", path_norm)
        self.triangle_file.set(os.path.basename(path_norm))
        print(f"[Triangles] Fichier chargé: {os.path.basename(path_norm)} ({path_norm})")
        # MAJ du menu fichiers si un nouveau fichier arrive dans data
        self._rebuild_triangle_file_list_menu()

        self.listbox.delete(0, tk.END)
        for _, r in self.df.iterrows():
            self.listbox.insert(tk.END, f"{int(r['id']):02d}. B:{r['B']}  L:{r['L']}")
        self.status.config(text=f"{len(self.df)} triangles chargés depuis {path_norm}")
        # Réinitialiser les triangles posés du scénario actif
        # IMPORTANT : on vide la liste en place pour préserver le lien
        # avec scen.last_drawn du scénario manuel.
        self._last_drawn.clear()
        # Aucun triangle n'est encore utilisé dans le scénario actif        self._placed_ids = set()
        self._update_triangle_listbox_colors()

    # ---------- Triangles (données locales) ----------
    def _triangles_local(self, start, n):
        """
        Construit les triangles (coord. locales) à partir du DF pour [start:start+n].
        Retour: liste de dict { 'labels':(O,B,L), 'pts':{'O','B','L'} }
        """
        if getattr(self, "df", None) is None or self.df.empty:
            raise RuntimeError("Pas de données — ouvre d'abord l’Excel.")
        sub = self.df.iloc[start:start+n]
        out = []
        for _, r in sub.iterrows():
            P = _build_local_triangle(float(r["len_OB"]), float(r["len_OL"]), float(r["len_BL"]))
            # Si orientation = CW, on applique une symétrie verticale (y -> -y)
            try:
                ori = str(r.get("orient", "CCW")).upper()
            except Exception:
                ori = "CCW"
            if ori == "CW":
                P = {"O": np.array([P["O"][0], -P["O"][1]]),
                     "B": np.array([P["B"][0], -P["B"][1]]),
                     "L": np.array([P["L"][0], -P["L"][1]])}
            out.append({
                "labels": ( "Bourges", str(r["B"]), str(r["L"]) ),  # O,B,L
                "pts": P,
                "id": int(r["id"]),
                "mirrored": (ori == "CW"),
            })
        return out

    # ---------- Mise en page simple (aperçu brut) ----------
    def _triangle_from_index(self, idx):
        """Construit un triangle 'local' depuis l’élément sélectionné de la listbox.
        IMPORTANT: on parse l'ID affiché (NN.) au lieu d'utiliser df.iloc[idx],
        car la listbox peut avoir des éléments retirés → indices décalés.
        """
        if getattr(self, "df", None) is None or self.df.empty:
            raise RuntimeError("Pas de données — ouvre d'abord l’Excel.")
        # Récupérer le texte de la listbox et extraire l'id (NN. ...)
        lb_txt = ""
        try:
            if 0 <= idx < self.listbox.size():
                lb_txt = self.listbox.get(idx)
        except Exception:
            pass
        tri_id = None
        m = re.match(r"\s*(\d+)\.", str(lb_txt))
        if m:
            tri_id = int(m.group(1))
            row = self.df[self.df["id"] == tri_id]
            if not row.empty:
                r = row.iloc[0]
            else:
                # secours si pas trouvé (ne devrait pas arriver)
                r = self.df.iloc[min(max(idx, 0), len(self.df)-1)]
                tri_id = int(r["id"])
        else:
            # si le libellé n'est pas conforme, on retombe sur l'index
            r = self.df.iloc[min(max(idx, 0), len(self.df)-1)]
            tri_id = int(r["id"])

        P = _build_local_triangle(float(r["len_OB"]), float(r["len_OL"]), float(r["len_BL"]))
        # Normalisation de l’orientation
        try:
            ori = str(r.get("orient", "CCW")).upper()
        except Exception:
            ori = "CCW"
        if ori == "CW":
            P = {"O": np.array([P["O"][0], -P["O"][1]]),
                 "B": np.array([P["B"][0], -P["B"][1]]),
                 "L": np.array([P["L"][0], -P["L"][1]])}
        return {
            "labels": ("Bourges", str(r["B"]), str(r["L"])),
            "pts": P,
            "id": tri_id,
            "mirrored": (ori == "CW"),   # reflet initial si orientation horaire
        }

    def _layout_tris_horizontal(self, tris, gap=0.5, wrap_width=None):
        """
        Place les triangles les uns à côté des autres (dans le repère "monde"),
        avec un espacement 'gap' (unités mêmes que les longueurs).
        Si wrap_width est fourni (en unités monde), on passe à la ligne quand la largeur est dépassée.
        Retour: liste de dict { 'labels', 'pts' (coordonnées MONDE après translation) }
        """
        placed = []
        x_cursor = 0.0
        y_cursor = 0.0
        line_height = 0.0

        if wrap_width is None:
            # largeur de ligne par défaut : environ 20 unités de longueur
            wrap_width = 20.0

        for t in tris:
            P = t["pts"]
            xs = [float(P["O"][0]), float(P["B"][0]), float(P["L"][0])]
            ys = [float(P["O"][1]), float(P["B"][1]), float(P["L"][1])]
            mnx, mny, mxx, mxy = min(xs), min(ys), max(xs), max(ys)
            w = (mxx - mnx)
            h = (mxy - mny)

            # saut de ligne si nécessaire
            if x_cursor > 0.0 and (x_cursor + w) > wrap_width:
                x_cursor = 0.0
                y_cursor -= (line_height + gap)  # vers le bas (coord monde Y+ en haut -> inversé en écran)
                line_height = 0.0

            # translation pour placer ce triangle
            dx = x_cursor - mnx
            dy = y_cursor - mny
            Pw = {
                "O": np.array([P["O"][0] + dx, P["O"][1] + dy]),
                "B": np.array([P["B"][0] + dx, P["B"][1] + dy]),
                "L": np.array([P["L"][0] + dx, P["L"][1] + dy]),
            }
            placed.append({"labels": t["labels"], "pts": Pw, "id": t.get("id"), "mirrored": t.get("mirrored", False)})

            x_cursor += w + gap
            line_height = max(line_height, h)

        return placed

    # ====== FRONTIER GRAPH HELPERS (factorisation) ===============================================
    def _edge_dir(self, a, b):
        return (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))

    def _ang_wrap(self, x):
        import math
        # wrap sur [-pi, pi]
        x = (x + math.pi) % (2*math.pi) - math.pi
        return x

    def _ang_of_vec(self, vx, vy):
        import math
        return math.atan2(vy, vx)

    def _ang_diff(self, a, b):
        # plus petit écart absolu d’angle
        return abs(self._ang_wrap(a - b))

    def _pt_key_eps(self, p, eps=1e-9):
        return (round(float(p[0])/eps)*eps, round(float(p[1])/eps)*eps)

    def _build_boundary_graph(self, outline):
        """
        Construire un graphe de frontière léger (half-edges) à partir de 'outline'
        outline: liste de segments [(a,b), ...]
        Retour: dict {
            "adj": {key(point)-> [point_voisin,...]},
            "pts": {key(point)-> point_float_tuple}
        }
        """
        adj = {}
        pts = {}
        for a,b in (outline or []):
            a = (float(a[0]), float(a[1])); b = (float(b[0]), float(b[1]))
            ka, kb = self._pt_key_eps(a), self._pt_key_eps(b)
            pts.setdefault(ka, a); pts.setdefault(kb, b)
            adj.setdefault(ka, []).append(b)
            adj.setdefault(kb, []).append(a)
        return {"adj": adj, "pts": pts}

    def _incident_half_edges_at_vertex(self, graph, v, eps=EPS_WORLD):
        """
        Renvoie les deux demi-arêtes (si existantes) qui partent de 'v' le long de la frontière.
        Retour: liste de 0..2 éléments, chaque élément est ((ax,ay),(bx,by))
        """
        key = self._pt_key_eps(v)
        neighs = graph["adj"].get(key, [])
        out = []
        for w in neighs:
            out.append(( (float(v[0]),float(v[1])), (float(w[0]),float(w[1])) ))
        # garder au maximum 2 (enveloppe standard) ; si plus, on trie par angle autour de v et on prend 2 extrêmes
        if len(out) > 2:
            import math
            def ang(e):
                (a,b)=e; vx,vy=self._edge_dir(a,b); return math.atan2(vy,vx)
            out = sorted(out, key=ang)
            # choisir deux qui maximisent l'écart angulaire (les bords de l'enveloppe)
            # heuristique simple: prendre le premier et celui qui maximise |Δ|
            a0 = out[0]; ang0 = ang(a0)
            a1 = max(out[1:], key=lambda e: self._ang_diff(ang(e), ang0))
            out = [a0,a1]
        return out

    def _normalize_to_outline_granularity(self, outline, edges, eps=EPS_WORLD):
        """
        Décompose chaque segment incident en chaîne de micro-segments collés à l'outline (granularité identique),
        en s'appuyant sur l'adjacence du graphe de frontière. Retourne une liste de segments.
        """
        g = self._build_boundary_graph(outline)
        def almost(p,q):
            return abs(p[0]-q[0])<=eps and abs(p[1]-q[1])<=eps
        def dir_forward(u,v,w):
            uvx,uvy = v[0]-u[0], v[1]-u[1]
            uwx,uwy = w[0]-u[0], w[1]-u[1]
            cross = abs(uvx*uwy - uvy*uwx)
            dot   = (uvx*uwx + uvy*uwy)
            return cross <= 1e-9 and dot > 0.0
        out=[]
        for (a,b) in (edges or []):
            a=(float(a[0]),float(a[1])); b=(float(b[0]),float(b[1]))
            cur=a; guard=0; chain=[]
            while not almost(cur,b) and guard<2048:
                neigh = g["adj"].get(self._pt_key_eps(cur), [])
                nxt=None
                for w in neigh:
                    if dir_forward(cur,b,w):
                        nxt=w; break
                if nxt is None: break
                chain.append((cur,nxt))
                cur=nxt; guard+=1
            if chain and almost(cur,b):
                out.extend(chain)
            else:
                # fallback: garder le segment brut si pas décomposable finement
                out.append((a,b))
        return out



    # ------------------------------
    # Géométrie / utils divers
    # ------------------------------
    def _ang_of_vec(self, vx, vy):
        import math
        return math.atan2(vy, vx)
 
    def _ang_diff(self, a, b):
        # plus petit écart absolu d’angle
        return abs(self._ang_wrap(a - b))
         

    def _refresh_listbox_from_df(self):
        """Recharge la listbox des triangles depuis self.df (sans filtrage)."""
        try:
            self.listbox.delete(0, tk.END)
            if getattr(self, "df", None) is not None and not self.df.empty:
                for _, r in self.df.iterrows():
                    self.listbox.insert(tk.END, f"{int(r['id']):02d}. B:{r['B']}  L:{r['L']}")
        except Exception:
            pass

    def clear_canvas(self):
        """Efface l'affichage après confirmation, et remet à jour la liste des triangles."""
        if not messagebox.askyesno(
            "Effacer l'affichage",
            "Voulez-vous effacer l'affichage et réinitialiser la liste des triangles ?"
        ):
            return
        try:
            self.canvas.delete("all")
        except Exception:
            pass
        # Réinitialiser l'état d'affichage du scénario actif
        # (vidage en place pour garder le lien scen.last_drawn)
        self._last_drawn.clear()
        self._nearest_line_id = None
        self._clear_edge_highlights()
        self._edge_choice = None
        self._edge_highlights = None
        # Reconstruire la listbox à partir de la DF courante
        if hasattr(self, "triangle_df") and self.triangle_df is not None:
            self._refresh_listbox_from_df()
        # Après effacement, plus aucun triangle n'est considéré comme "déjà utilisé"
        # → on peut à nouveau les sélectionner pour un drag & drop.
        self._placed_ids.clear()
        self._update_triangle_listbox_colors()

        self.status.config(text="Affichage effacé")
        self._hide_tooltip()

        # L’horloge reste visible (overlay)
        self._draw_clock_overlay()

    # ---------- Overlay Horloge (indépendant du zoom/pan) ----------
    def _redraw_overlay_only(self):
        """Efface/redessine uniquement l'overlay (horloge)."""
        if not getattr(self, "canvas", None):
            return
        try:
            self.canvas.delete("clock_overlay")
        except Exception:
            pass
        self._draw_clock_overlay()

    def _draw_clock_overlay(self):
        """
        Dessine une horloge en haut-gauche du canvas, à taille FIXE (px),
        indépendante du zoom/pan (coordonnées canvas).
        Utilise self._clock_state = {'hour':h, 'minute':m, 'label':str}.
        """
        if not getattr(self, "canvas", None):
            return
        # Nettoyer l'ancien overlay
        try:
            self.canvas.delete("clock_overlay")
        except Exception:
            pass
        # Si le compas est masqué via le menu, ne rien dessiner
        if hasattr(self, "show_clock_overlay") and not self.show_clock_overlay.get():
            return
        # Paramètres d'aspect
        margin = 12              # marge par rapport aux bords du canvas (px)
        R      = 69              # rayon (px) — +50% (46 -> 69)
        # Si première fois, placer en haut-gauche ; sinon garder la position utilisateur
        if self._clock_cx is None or self._clock_cy is None:
            cx = margin + R
            cy = margin + R
            self._clock_cx, self._clock_cy = cx, cy
        else:
            cx, cy = float(self._clock_cx), float(self._clock_cy)
        self._clock_R = R
        # Couleurs
        col_circle = "#b0b0b0"   # gris cercle
        col_ticks  = "#707070"
        col_hour   = "#0b3d91"   # bleue (petite aiguille)
        col_min    = "#000000"   # noire  (grande aiguille)
        # Données
        h = int(self._clock_state.get("hour", 5)) % 12
        m = int(self._clock_state.get("minute", 9)) % 60
        label = str(self._clock_state.get("label", ""))
        # Cercle
        self.canvas.create_oval(cx-R, cy-R, cx+R, cy+R,
                                outline=col_circle, width=2, tags="clock_overlay")
        # Graduations heures (12 traits) : quarts plus longs/épais
        for hmark in range(12):
            ang = math.radians(hmark * 30.0)  # 360/12
            # longueur du trait
            if hmark % 3 == 0:     # 12/3/6/9
                inner = R - 14
                w = 2
            else:
                inner = R - 8
                w = 1
            outer = R
            x1 = cx + inner * math.sin(ang)
            y1 = cy - inner * math.cos(ang)
            x2 = cx + outer * math.sin(ang)
            y2 = cy - outer * math.cos(ang)
            self.canvas.create_line(x1, y1, x2, y2, width=w, fill=col_ticks, tags="clock_overlay")

        # Repères 12/3/6/9
        font_marks = ("Arial", 11, "bold")
        self.canvas.create_text(cx,     cy-R+14, text="12", font=font_marks,
                                fill=col_ticks, tags="clock_overlay")
        self.canvas.create_text(cx+R-14, cy,     text="3",  font=font_marks,
                                fill=col_ticks, tags="clock_overlay")
        self.canvas.create_text(cx,     cy+R-14, text="6",  font=font_marks,
                                fill=col_ticks, tags="clock_overlay")
        self.canvas.create_text(cx-R+14, cy,     text="9",  font=font_marks,
                                fill=col_ticks, tags="clock_overlay")
        # Aiguilles
        # Convention : angle 0° = 12h, sens horaire ; conversion vers coords canvas:
        #   x = cx + R * sin(theta), y = cy - R * cos(theta)
        def _end_point(angle_deg, length):
            import math
            a = math.radians(angle_deg)
            return (cx + length * math.sin(a), cy - length * math.cos(a))
        # Minutes -> 6° par minute
        ang_min = m * 6.0
        # Heures -> 30° par heure + 0,5° par minute
        ang_hour = (h % 12) * 30.0 + (m * 0.5)
        # Longueurs des aiguilles
        L_min  = R * 0.86
        L_hour = R * 0.58
        x2m, y2m = _end_point(ang_min,  L_min)
        x2h, y2h = _end_point(ang_hour, L_hour)
        # traits
        self.canvas.create_line(cx, cy, x2h, y2h, width=3, fill=col_hour, tags="clock_overlay")
        self.canvas.create_line(cx, cy, x2m, y2m, width=2, fill=col_min,  tags="clock_overlay")
        # axe central
        self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill=col_min, outline=col_min, tags="clock_overlay")
        # Libellé sous l'horloge
        if label:
            self.canvas.create_text(cx, cy + R + 20, text=label,
                                    font=("Arial", 11), fill="#000000",
                                    anchor="n", tags="clock_overlay")

    # =========================
    # Fond SVG en coordonnées monde
    # =========================

    def _bg_clear(self):
        self._bg = None
        self._bg_base_pil = None
        self._bg_photo = None
        self._bg_resizing = None
        self._redraw_from(self._last_drawn)

    def _bg_load_svg_dialog(self):
        path = filedialog.askopenfilename(
            title="Choisir un fond SVG",
            filetypes=[("SVG", "*.svg"), ("Tous fichiers", "*.*")]
        )
        if not path:
            return
        self._bg_set_svg(path)

    def _bg_set_svg(self, svg_path: str):
        if (Image is None or ImageTk is None or svg2rlg is None
                or renderPDF is None or _rl_canvas is None or pdfium is None):
            messagebox.showerror(
                "Dépendances manquantes",
                "Pour afficher un fond SVG (svglib/reportlab, sans Cairo), installe :\n"
                "  - pillow\n  - svglib\n  - reportlab\n  - pypdfium2\n\n"
                "pip install pillow svglib reportlab pypdfium2"
            )
            return

        aspect = self._bg_parse_aspect(svg_path)

        # Position/taille initiale : calée sur bbox des triangles si dispo, sinon vue écran
        if self._last_drawn:
            xs, ys = [], []
            for t in self._last_drawn:
                P = t["pts"]
                for k in ("O", "B", "L"):
                    xs.append(float(P[k][0])); ys.append(float(P[k][1]))
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
        else:
            # bbox monde visible
            cw = max(2, int(self.canvas.winfo_width() or 2))
            ch = max(2, int(self.canvas.winfo_height() or 2))
            x0, yTop = self._screen_to_world(0, 0)
            x1, yBot = self._screen_to_world(cw, ch)
            xmin, xmax = (min(x0, x1), max(x0, x1))
            ymin, ymax = (min(yBot, yTop), max(yBot, yTop))

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        bw = max(1e-6, (xmax - xmin) * 1.10)
        bh = max(1e-6, (ymax - ymin) * 1.10)

        # Ajuster pour conserver le ratio du SVG
        if (bw / bh) > aspect:
            w = bw
            h = w / aspect
        else:
            h = bh
            w = h * aspect

        self._bg = {"path": svg_path, "x0": cx - w/2, "y0": cy - h/2, "w": w, "h": h, "aspect": aspect}

        # Raster base "normalisée" (taille fixe, ratio respecté)
        self._bg_base_pil = self._bg_render_base(svg_path, aspect, max_dim=4096)

        self._redraw_from(self._last_drawn)

    def _bg_parse_aspect(self, svg_path: str) -> float:
        try:
            s = open(svg_path, "r", encoding="utf-8", errors="ignore").read(8192)
        except Exception:
            return 1.0

        # viewBox="minx miny width height"
        m = re.search(r'viewBox\s*=\s*["\']\s*([-\d\.eE]+)\s+([-\d\.eE]+)\s+([-\d\.eE]+)\s+([-\d\.eE]+)\s*["\']', s)
        if m:
            vw = float(m.group(3)); vh = float(m.group(4))
            if abs(vh) > 1e-12:
                return max(1e-6, vw / vh)

        # width="xxx" height="yyy" (units possible)
        mw = re.search(r'width\s*=\s*["\']\s*([-\d\.eE]+)', s)
        mh = re.search(r'height\s*=\s*["\']\s*([-\d\.eE]+)', s)
        if mw and mh:
            w = float(mw.group(1)); h = float(mh.group(1))
            if abs(h) > 1e-12:
                return max(1e-6, w / h)

        return 1.0

    def _bg_render_base(self, svg_path: str, aspect: float, max_dim: int = 4096):
        # base normalisée : max_dim sur le plus grand côté
        if aspect >= 1.0:
            W = int(max_dim)
            H = max(1, int(round(W / aspect)))
        else:
            H = int(max_dim)
            W = max(1, int(round(H * aspect)))

        try:
            drawing = svg2rlg(svg_path) if svg2rlg is not None else None
            if drawing is None:
                raise RuntimeError("svg2rlg() a retourné None")

            # --- IMPORTANT : éviter le SVG tronqué ---
            # svglib peut produire un Drawing dont le contenu a un bounding-box décalé (xmin/ymin != 0),
            # ou des width/height non représentatifs. On normalise sur getBounds() quand c'est possible.
            xmin = ymin = 0.0
            bw = bh = 0.0
            try:
                if hasattr(drawing, "getBounds"):
                    b = drawing.getBounds()
                    if b and len(b) == 4:
                        xmin, ymin, xmax, ymax = map(float, b)
                        bw = max(0.0, xmax - xmin)
                        bh = max(0.0, ymax - ymin)
            except Exception:
                bw = bh = 0.0

            # Dimensions de référence : d'abord le bounding-box, sinon width/height svglib
            dw = float(bw) if bw > 1e-9 else float(getattr(drawing, "width", 0) or 0)
            dh = float(bh) if bh > 1e-9 else float(getattr(drawing, "height", 0) or 0)
            if dw <= 0 or dh <= 0:
                # fallback: utiliser le ratio calculé, avec une base arbitraire
                dw = float(W)
                dh = float(H)

            # Si le contenu est décalé (xmin/ymin), on le ramène à l'origine avant le scale.
            if bw > 1e-9 and bh > 1e-9:
                try:
                    if hasattr(drawing, "translate"):
                        drawing.translate(-xmin, -ymin)
                except Exception:
                    pass

            # svglib exprime en "points" (1/72 inch). En rasterisant à 72dpi,
            # 1 point ~= 1 pixel. On scale le dessin puis on force le canvas aux dims voulues.
            sx = float(W) / dw
            sy = float(H) / dh
            # On applique les deux échelles pour respecter EXACTEMENT W/H (le ratio vient déjà de 'aspect')
            try:
                drawing.scale(sx, sy)
            except Exception:
                # certains objets retournés par svg2rlg peuvent ne pas exposer scale()
                pass
            try:
                drawing.width = float(W)
                drawing.height = float(H)
            except Exception:
                pass

            # Rasterisation sans Cairo : SVG -> PDF (reportlab) -> bitmap (pypdfium2)
            if renderPDF is None or _rl_canvas is None or pdfium is None:
                raise RuntimeError("Rasterisation SVG indisponible : pip install svglib reportlab pypdfium2")

            buf = io.BytesIO()
            c = _rl_canvas.Canvas(buf, pagesize=(W, H))
            renderPDF.draw(drawing, c, 0, 0)
            c.showPage()
            c.save()
            pdf_bytes = buf.getvalue()

            doc = pdfium.PdfDocument(pdf_bytes)
            try:
                page = doc[0]
                bitmap = page.render(scale=1)  # ~72 dpi : 1 point ≈ 1 pixel
                pil = bitmap.to_pil().convert("RGBA")
            finally:
                try:
                    doc.close()
                except Exception:
                    pass

            if pil.size != (W, H):
                pil = pil.resize((W, H), Image.LANCZOS)
            return pil
        except Exception as e:
            messagebox.showerror("Erreur SVG", f"Impossible de rasteriser le SVG (svglib/reportlab):\n{e}")
            return None

    def _bg_draw_world_layer(self):
        """Dessine le fond en 'monde' : on recadre la base en fonction pan/zoom et on l'affiche en plein canvas."""
        if not self._bg or self._bg_base_pil is None or Image is None or ImageTk is None:
            return

        cw = int(self.canvas.winfo_width() or 0)
        ch = int(self.canvas.winfo_height() or 0)
        if cw <= 2 or ch <= 2:
            try:
                self.update_idletasks()
                cw = int(self.canvas.winfo_width() or 0)
                ch = int(self.canvas.winfo_height() or 0)
            except Exception:
                return
        if cw <= 2 or ch <= 2:
            return

        bx0 = float(self._bg["x0"]); by0 = float(self._bg["y0"])
        bw = float(self._bg["w"]);  bh = float(self._bg["h"])
        bx1 = bx0 + bw; by1 = by0 + bh

        # Vue monde actuelle (canvas entier)
        xA, yTop = self._screen_to_world(0, 0)
        xB, yBot = self._screen_to_world(cw, ch)
        vx0, vx1 = (min(xA, xB), max(xA, xB))
        vy0, vy1 = (min(yBot, yTop), max(yBot, yTop))

        # Intersection vue <-> fond
        ix0 = max(vx0, bx0); ix1 = min(vx1, bx1)
        iy0 = max(vy0, by0); iy1 = min(vy1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            return

        baseW, baseH = self._bg_base_pil.size

        # Crop dans l'image base
        left  = int((ix0 - bx0) / bw * baseW)
        right = int((ix1 - bx0) / bw * baseW)
        # y base : 0 en haut, donc on inverse
        upper = int((by1 - iy1) / bh * baseH)  # iy1 = top
        lower = int((by1 - iy0) / bh * baseH)  # iy0 = bottom

        # clamp
        left = max(0, min(baseW-1, left))
        right = max(left+1, min(baseW, right))
        upper = max(0, min(baseH-1, upper))
        lower = max(upper+1, min(baseH, lower))

        crop = self._bg_base_pil.crop((left, upper, right, lower))

        # Où coller sur l'écran
        sx0, syTop = self._world_to_screen((ix0, iy1))
        sx1, syBot = self._world_to_screen((ix1, iy0))
        wpx = int(round(sx1 - sx0))
        hpx = int(round(syBot - syTop))
        if wpx <= 1 or hpx <= 1:
            return

        crop = crop.resize((wpx, hpx), Image.LANCZOS)

        out = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        px = int(round(sx0)); py = int(round(syTop))

        # clip paste
        paste_x0 = max(0, px)
        paste_y0 = max(0, py)
        paste_x1 = min(cw, px + wpx)
        paste_y1 = min(ch, py + hpx)
        if paste_x1 <= paste_x0 or paste_y1 <= paste_y0:
            return

        src_x0 = paste_x0 - px
        src_y0 = paste_y0 - py
        src_x1 = src_x0 + (paste_x1 - paste_x0)
        src_y1 = src_y0 + (paste_y1 - paste_y0)

        crop2 = crop.crop((src_x0, src_y0, src_x1, src_y1))
        out.paste(crop2, (paste_x0, paste_y0), crop2)

        self._bg_photo = ImageTk.PhotoImage(out)
        self.canvas.create_image(0, 0, anchor="nw", image=self._bg_photo, tags=("bg_world",))
        self.canvas.tag_lower("bg_world")

    def _bg_corners_world(self):
        if not self._bg:
            return None
        x0 = float(self._bg["x0"]); y0 = float(self._bg["y0"])
        w = float(self._bg["w"]);  h = float(self._bg["h"])
        return {
            "bl": (x0,     y0),
            "br": (x0 + w, y0),
            "tl": (x0,     y0 + h),
            "tr": (x0 + w, y0 + h),
        }

    def _bg_corners_screen(self):
        c = self._bg_corners_world()
        if not c:
            return None
        return {k: self._world_to_screen(v) for k, v in c.items()}

    def _bg_draw_resize_handles(self):
        if not self._bg or not self.bg_resize_mode.get():
            return
        c = self._bg_corners_screen()
        if not c:
            return
        tl = c["tl"]; br = c["br"]

        self.canvas.create_rectangle(tl[0], tl[1], br[0], br[1], outline="gray30", dash=(3, 2), width=1, tags=("bg_ui",))

        r = 6
        for k in ("tl", "tr", "bl", "br"):
            x, y = c[k]
            self.canvas.create_rectangle(x-r, y-r, x+r, y+r, outline="gray10", fill="white", width=1, tags=("bg_ui",))

    def _bg_hit_test_handle(self, sx: float, sy: float):
        c = self._bg_corners_screen()
        if not c:
            return None
        r = 8
        for k in ("tl", "tr", "bl", "br"):
            x, y = c[k]
            if (sx - x)*(sx - x) + (sy - y)*(sy - y) <= r*r:
                return k
        return None

    def _bg_start_resize(self, handle: str, sx: int, sy: int):
        # handle: tl,tr,bl,br ; corner opposée fixe
        opp = {"tl": "br", "br": "tl", "tr": "bl", "bl": "tr"}[handle]
        corners = self._bg_corners_world()
        fx, fy = corners[opp]
        mx, my = self._screen_to_world(sx, sy)
        self._bg_resizing = {
            "handle": handle,
            "fixed": (fx, fy),
            "start_mouse": (mx, my),
            "start_rect": (float(self._bg["x0"]), float(self._bg["y0"]), float(self._bg["w"]), float(self._bg["h"])),
        }

    def _bg_update_resize(self, sx: int, sy: int):
        if not self._bg_resizing or not self._bg:
            return
        aspect = float(self._bg["aspect"])
        fx, fy = self._bg_resizing["fixed"]
        mx, my = self._screen_to_world(sx, sy)

        dx = mx - fx
        dy = my - fy
        w0 = abs(dx)
        h0 = abs(dy)
        if w0 < 1e-6 or h0 < 1e-6:
            return

        # Conserver ratio : choisir la dominante (horizontal vs vertical)
        if (w0 / h0) > aspect:
            w = w0
            h = w / aspect
        else:
            h = h0
            w = h * aspect

        w = max(1e-3, w)
        h = max(1e-3, h)

        # Recomposer x0/y0 selon le quadrant (fixed est la corner opposée)
        # On place le rectangle de sorte que fixed reste fixe
        x0 = fx if dx >= 0 else (fx - w)
        y0 = fy if dy >= 0 else (fy - h)

        self._bg["x0"] = float(x0)
        self._bg["y0"] = float(y0)
        self._bg["w"]  = float(w)
        self._bg["h"]  = float(h)

    def _world_to_screen(self, p):
        x = self.offset[0] + float(p[0]) * self.zoom
        y = self.offset[1] - float(p[1]) * self.zoom
        return x, y

    def _fit_to_view(self, placed):
        if not placed:
            return
        xs, ys = [], []
        for t in placed:
            P = t["pts"]
            for k in ("O", "B", "L"):
                xs.append(float(P[k][0]))
                ys.append(float(P[k][1]))
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        w, h = maxx - minx, maxy - miny
        if w <= 0 or h <= 0:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        margin = 40
        zx = (cw - 2 * margin) / w
        zy = (ch - 2 * margin) / h
        self.zoom = max(0.1, min(zx, zy))
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        self.offset = np.array(
            [cw / 2.0 - cx * self.zoom, ch / 2.0 + cy * self.zoom],
            dtype=float
        )
        # redraw for fit
        self._redraw_from(self._last_drawn)

    def _redraw_from(self, placed):
        self.canvas.delete("all")
        # Fond SVG monde (toujours derrière)
        self._bg_draw_world_layer()        

        # l'ID de la ligne n'est plus valide après delete("all")
        self._nearest_line_id = None
        # on efface les IDs de surlignage déjà dessinés (mais on conserve le choix et les données)
        self._clear_edge_highlights()
        for t in placed:
            labels = t["labels"]
            P = t["pts"]
            tri_id = t.get("id")
            self._draw_triangle_screen(
                P,
                labels=[f"O:{labels[0]}", f"B:{labels[1]}", f"L:{labels[2]}"],
                tri_id=tri_id,
                tri_mirrored=t.get("mirrored", False)
            )
        # — Recrée l'aide visuelle UNIQUEMENT si une sélection 'vertex' est encore active —
        try:
            if self._sel and self._sel.get("mode") == "vertex":
                # redessine candidates + best si on a des données
                if getattr(self, "_edge_highlights", None):
                    self._redraw_edge_highlights()
                # remet la ligne grise
                idx = self._sel["idx"]; vkey = self._sel["vkey"]
                P = self._last_drawn[idx]["pts"]
                v_world = np.array(P[vkey], dtype=float)
                self._update_nearest_line(v_world, exclude_idx=idx)
            else:
                # pas de sélection active -> pas d'aides persistantes
                self._edge_highlights = None
                self._edge_choice = None
        except Exception:
            pass
        # Poignées de redimensionnement fond (overlay UI)
        self._bg_draw_resize_handles()
        # Redessiner l'horloge (overlay indépendant)
        self._draw_clock_overlay()
        # Après tout redraw, le cache de pick n'est plus valide
        self._invalidate_pick_cache()

    def _draw_triangle_screen(self, P, outline="black", width=2, labels=None, inset=0.35, tri_id=None, tri_mirrored=False):
        """
        P : dict {'O','B','L'} en coordonnées monde (np.array 2D)
        labels : liste de 3 strings pour O,B,L (facultatif)
        inset : 0..1, fraction du chemin du sommet vers le barycentre pour placer le texte
        """
        # 1) coords monde -> écran
        pts_world = [P["O"], P["B"], P["L"]]
        coords = []
        for pt in pts_world:
            sx, sy = self._world_to_screen(pt)
            coords += [sx, sy]

        # 2) tracé du triangle (arêtes colorées selon les sommets)
        #    - Lumière (L) -> Ouverture (O) : noir
        #    - Lumière (L) -> Base (B)      : bleu foncé
        #    - Base (B) -> Ouverture (O)    : gris
        Ox, Oy, Bx, By, Lx, Ly = coords
        self.canvas.create_line(Ox, Oy, Lx, Ly, fill="#000000", width=width)
        self.canvas.create_line(Bx, By, Lx, Ly, fill="#00008B", width=width)
        self.canvas.create_line(Bx, By, Ox, Oy, fill="#808080", width=width)

        # 2b) marqueurs colorés par type de bord
        # O = Ouverture (noir), B = Base (bleu), L = Lumière (jaune)
        marker_px = 6  # rayon en pixels (indépendant du zoom)
        def _dot(x, y, fill, outline="black"):
            r = marker_px
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=1)

        # Ouverture / O (noir)
        Ox, Oy = self._world_to_screen(P["O"])
        _dot(Ox, Oy, fill="#000000", outline="#000000")
        # Base / B (bleu)
        Bx, By = self._world_to_screen(P["B"])
        _dot(Bx, By, fill="#0000FF", outline="#000000")
        # Lumière / L (jaune)
        Lx, Ly = self._world_to_screen(P["L"])
        _dot(Lx, Ly, fill="#FFD700", outline="#000000")

        # 3) barycentre (monde)
        cx = (P["O"][0] + P["B"][0] + P["L"][0]) / 3.0
        cy = (P["O"][1] + P["B"][1] + P["L"][1]) / 3.0

        # 4) labels (sans "O:" et sans préfixes "B:" / "L:")
        if labels:
            for pt, txt in zip(pts_world, labels):
                # Supprimer les préfixes "O:", "B:", "L:" si présents
                if ":" in txt:
                    prefix, value = txt.split(":", 1)
                else:
                    prefix, value = "", txt
                prefix = prefix.strip().lower()
                value  = value.strip()

                # Ne rien afficher pour l'ouverture (on a le point noir)
                if prefix in ("o", "ouverture", "ouv"):
                    continue

                # Pour Base/Lumière, n'afficher que la valeur (sans codes/lettres)
                display = value
                if not display:
                    continue

                lx = (1.0 - inset) * pt[0] + inset * cx
                ly = (1.0 - inset) * pt[1] + inset * cy
                sx, sy = self._world_to_screen((lx, ly))
                self.canvas.create_text(sx, sy, text=display, anchor="center", font=("Arial", 8), tags="tri_label")

        # 5) numéro du triangle (toujours affiché après les labels, en avant-plan)
        if tri_id is not None:
            sx, sy = self._world_to_screen((cx, cy))
            num_txt = f"{tri_id}{'S' if tri_mirrored else ''}"
            # Bloc ID + mot recentré verticalement autour de sy, avec un écart réduit
            gap = 8  # écart vertical (px) entre ID et mot
            id_y   = sy - (gap // 2)
            word_y = sy + (gap - gap // 2)

            self.canvas.create_text(
                sx, id_y, text=num_txt,
                anchor="center", font=("Arial", 10, "bold"),
                fill="red", tags="tri_num"
            )
            self.canvas.tag_raise("tri_num")

            # 5b) si un mot est associé à ce triangle, l’afficher sous l’ID (italique)
            winfo = self._tri_words.get(int(tri_id))
            if winfo and isinstance(winfo.get("word"), str) and winfo["word"].strip():
                self.canvas.create_text(
                    sx, word_y, text=winfo["word"],
                    anchor="n", font=("Arial", 9, "italic"),
                    fill="#222", tags="tri_word"
                )

    # --- helpers: mode déconnexion (CTRL) + curseur ---
    def _on_ctrl_down(self, event=None):
        if not self._ctrl_down:
            self._ctrl_down = True
            # Masquer tout tooltip en mode déconnexion
            self._hide_tooltip()
            try:
                self.canvas.configure(cursor="X_cursor")
            except tk.TclError:
                self.canvas.configure(cursor="crosshair")


            # --- NOUVEAU (étape 2) ---
            # Si CTRL est pressé *après* avoir sélectionné un sommet (assist ON),
           # on applique immédiatement la ROTATION d'alignement (comme au release),
            # mais SANS translation et SANS collage (CTRL empêchera le snap au relâchement).
            try:
                if self._sel and (not self._sel.get("suppress_assist")):
                    mode = self._sel.get("mode")
                    choice = getattr(self, "_edge_choice", None)
                    if choice:
                        import numpy as np

                        # -------- Cas 1 : triangle seul (mode vertex) --------
                        if mode == "vertex":
                            idx = self._sel.get("idx")
                            vkey = self._sel.get("vkey")
                            if choice[0] == idx and choice[1] == vkey:
                                (_, _, idx_t, vkey_t, (m_a, m_b, t_a, t_b)) = choice
                                tri_m = self._last_drawn[idx]
                                Pm = tri_m["pts"]

                                A = np.array(m_a, dtype=float)  # mobile: sommet saisi
                                B = np.array(m_b, dtype=float)  # mobile: voisin arête
                                U = np.array(t_a, dtype=float)  # cible: sommet (pas utilisé ici)
                                V = np.array(t_b, dtype=float)  # cible: voisin arête

                                ang_m = self._ang_of_vec(B[0] - A[0], B[1] - A[1])
                                ang_t = self._ang_of_vec(V[0] - U[0], V[1] - U[1])
                                dtheta = ang_t - ang_m

                                if abs(dtheta) > 1e-12:
                                    R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                                                  [np.sin(dtheta),  np.cos(dtheta)]], dtype=float)
                                    for k in ("O", "B", "L"):
                                        p = np.array(Pm[k], dtype=float)
                                        p_rot = A + (R @ (p - A))
                                        Pm[k][0] = float(p_rot[0])
                                        Pm[k][1] = float(p_rot[1])

                                    # Redessiner + remettre les aides (rouge/gris)
                                    self._redraw_from(self._last_drawn)
                                    v_world = np.array(self._last_drawn[idx]["pts"][vkey], dtype=float)
                                    self._update_nearest_line(v_world, exclude_idx=idx)
                                    self._update_edge_highlights(idx, vkey, idx_t, vkey_t)

                        # -------- Cas 2 : déplacement de groupe ancré sur sommet --------
                        elif mode == "move_group":
                            anchor = self._sel.get("anchor")
                            gid = self._sel.get("gid")
                            if anchor and anchor.get("type") == "vertex":
                                anchor_tid = anchor.get("tid")
                                anchor_vkey = anchor.get("vkey")
                                if choice[0] == anchor_tid and choice[1] == anchor_vkey:
                                    (_, _, idx_t, vkey_t, (m_a, m_b, t_a, t_b)) = choice

                                    A = np.array(m_a, dtype=float)
                                    B = np.array(m_b, dtype=float)
                                    U = np.array(t_a, dtype=float)
                                    V = np.array(t_b, dtype=float)

                                    ang_m = self._ang_of_vec(B[0] - A[0], B[1] - A[1])
                                    ang_t = self._ang_of_vec(V[0] - U[0], V[1] - U[1])
                                    dtheta = ang_t - ang_m

                                    if abs(dtheta) > 1e-12:
                                        R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                                                      [np.sin(dtheta),  np.cos(dtheta)]], dtype=float)

                                        g = self.groups.get(gid)
                                        if g:
                                            for node in g["nodes"]:
                                                tid = node.get("tid")
                                                if 0 <= tid < len(self._last_drawn):
                                                    P = self._last_drawn[tid]["pts"]
                                                    for k in ("O", "B", "L"):
                                                        p = np.array(P[k], dtype=float)
                                                        p_rot = A + (R @ (p - A))
                                                        P[k][0] = float(p_rot[0])
                                                        P[k][1] = float(p_rot[1])

                                        # Redessiner + remettre les aides (en excluant le groupe mobile)
                                        try:
                                            self._recompute_group_bbox(gid)
                                        except Exception:
                                            pass
                                        self._redraw_from(self._last_drawn)
                                        v_world = np.array(self._last_drawn[anchor_tid]["pts"][anchor_vkey], dtype=float)
                                        self._update_nearest_line(v_world, exclude_idx=anchor_tid, exclude_gid=gid)
                                        self._update_edge_highlights(anchor_tid, anchor_vkey, idx_t, vkey_t)
            except Exception:
                pass

    def _on_ctrl_up(self, event=None):
        if self._ctrl_down:
            self._ctrl_down = False
            try:
                self.canvas.configure(cursor="")
            except tk.TclError:
                pass


    # ---------- Mouse navigation ----------
    # Drag depuis la liste
    def _on_triangle_list_select(self, event=None):
        """Empêche la sélection d'un triangle déjà utilisé dans le scénario courant."""
        # éviter la récursion quand on modifie la sélection nous-mêmes
        if getattr(self, "_in_triangle_select_guard", False):
            return
        if not hasattr(self, "listbox"):
            return

        sel = self.listbox.curselection()
        if not sel:
            self._last_triangle_selection = None
            return

        idx = int(sel[0])
        try:
            tri = self._triangle_from_index(idx)
            tid = int(tri.get("id"))
        except Exception:
            tid = None

        placed = getattr(self, "_placed_ids", set()) or set()
        if tid is not None and tid in placed:
            # Triangle déjà posé : on annule la sélection visuelle
            self._in_triangle_select_guard = True
            try:
                self.listbox.selection_clear(0, tk.END)
                if (self._last_triangle_selection is not None and
                        0 <= self._last_triangle_selection < self.listbox.size()):
                    self.listbox.selection_set(self._last_triangle_selection)
            finally:
                self._in_triangle_select_guard = False
            if hasattr(self, "status"):
                self.status.config(text=f"Triangle {tid} déjà utilisé dans ce scénario.")
            return

        # Sélection valide
        self._last_triangle_selection = idx

    def _on_ctrl_up(self, event=None):
        if self._ctrl_down:
            self._ctrl_down = False
            try:
                self.canvas.configure(cursor="")
            except tk.TclError:
                pass

    # ---------- Mouse navigation ----------
    # Drag depuis la liste
    def _on_list_mouse_down(self, event):
        """
        Démarre un drag & drop depuis la listbox,
        sauf si le triangle est déjà utilisé dans le scénario courant.
        """
        # Pas de DF → rien à faire
        if getattr(self, "df", None) is None or self.df.empty:
            return

        # Index de la ligne cliquée dans la listbox
        i = self.listbox.nearest(event.y)
        if i < 0:
            return

        # Sélection visuelle
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(i)

        # Récupération de la définition du triangle
        try:
            tri = self._triangle_from_index(i)
        except Exception as e:
            try:
                self.status.config(text=f"Erreur: {e}")
            except Exception:
                pass
            return

        tri_id = tri.get("id")
        # Si le triangle est déjà posé dans ce scénario, on bloque le drag
        if tri_id is not None and int(tri_id) in getattr(self, "_placed_ids", set()):
            try:
                self.status.config(text=f"Triangle {tri_id} déjà utilisé dans ce scénario.")
            except Exception:
                pass
            self._drag = None
            return

        # Préparation de la structure de drag pour le moteur commun
        start_screen = np.array([event.x_root, event.y_root], dtype=float)
        # NOTE : last_canvas n'est plus utilisé dans la nouvelle logique de drag & drop
        # (tout est géré en coords monde/écran via _world_to_screen / _screen_to_world).
        self._drag = {
            "from": "list",
            "triangle": tri,
            "list_index": i,
            "start_screen": start_screen,
            "mirrored": tri.get("mirrored", False),
        }

        # Réinitialiser un éventuel fantôme existant
        if self._drag_preview_id is not None:
            self.canvas.delete(self._drag_preview_id)
            self._drag_preview_id = None

        # Forcer le focus sur le canvas pour recevoir les mouvements
        self.canvas.focus_set()

        # Curseur "main" pendant le drag
        try:
            self.canvas.configure(cursor="hand2")
        except tk.TclError:
            pass

        try:
            self.status.config(text="Glissez le triangle sur le canvas puis relâchez pour le déposer.")
        except Exception:
            pass

    # ---------- Neighbours for tooltip ----------
    def _point_on_segment(self, P, A, B, eps):
        """Vrai si P appartient au segment [A,B] (colinéarité + projection dans [0,1])."""
        Ax,Ay = float(A[0]), float(A[1]); Bx,By = float(B[0]), float(B[1])
        Px,Py = float(P[0]), float(P[1])
        ABx, ABy = Bx-Ax, By-Ay
        APx, APy = Px-Ax, Py-Ay
        cross = abs(ABx*APy - ABy*APx)
        if cross > eps: 
            return False
        dot  = ABx*APx + ABy*APy
        ab2  = ABx*ABx + ABy*ABy
        if ab2 <= eps: 
            return False
        t = dot/ab2
        return -eps <= t <= 1.0+eps

    def _display_name(self, key, labels_tuple):
        """Retourne le libellé affiché (sans préfixe) pour O/B/L."""
        try:
            if   key == "O": raw = labels_tuple[0]
            elif key == "B": raw = labels_tuple[1]
            else:            raw = labels_tuple[2]
        except Exception:
            raw = ""
        s = str(raw or "").strip()
        return s
 
    # --- Azimut Nord->horaire (repère monde: Y vers le haut).
    #     Horloge : 360° = 12h et 360° = 60' (minutes d'horloge)
    def _azimutDegEtMinute(self, pSrc, pDst):
        """
        Renvoie (degInt, minFloat, hourFloat) pour le vecteur pSrc->pDst :
          - degInt    : angle depuis le Nord (sens horaire) en degrés entiers 0..359
          - minFloat  : minutes d’horloge au dixième (0.0 .. ≤60.0)   [360° = 60' → 1' = 6°]
          - hourFloat : heures d’horloge au dixième (0.0 .. ≤12.0)    [360° = 12h → 1h = 30°]
        """
        import math
        x1, y1 = float(pSrc[0]), float(pSrc[1])
        x2, y2 = float(pDst[0]), float(pDst[1])
        dx, dy = (x2 - x1), (y2 - y1)
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return None
        # Mesure depuis l'axe +Y (Nord) en tournant horaire
        angRad = math.atan2(dx, dy)
        angDeg = (math.degrees(angRad) + 360.0) % 360.0
        degInt   = int(round(angDeg)) % 360
        # Heures: 360° = 12h → 1h = 30° ; Minutes: 360° = 60' → 1' = 6°
        hourFloat = round(angDeg / 30.0, 1)        # h = deg / 30
        minFloat  = round(angDeg / 6.0, 1)         # ' = deg / 6  (0..60)
        return (degInt, minFloat, hourFloat)

    def _connected_vertices_from(self, v_world, gid, eps):
        """
        Depuis un sommet (ou un point tombant sur une arête), trouver toutes les
        extrémités connectées à ce point dans le GROUPE `gid`.
         Déduplique par direction et garde, sur une même direction, le point le plus éloigné.
         Retourne une liste d'objets { "name": str, "pt": (x, y) }.
        """
        nodes = self._group_nodes(gid) if gid else []
        out_candidates = []   # [(name, dx, dy, dist, neigh_xy)]
        for nd in nodes:
            tid = nd.get("tid")
            if tid is None or not (0 <= tid < len(self._last_drawn)):
                continue
            tri = self._last_drawn[tid]
            Pt  = tri["pts"]
            lab = tri.get("labels", ("Bourges","",""))
            # 3 arêtes du triangle
            edges = (("O","B"), ("B","L"), ("L","O"))
            for a,b in edges:
                A, B = Pt[a], Pt[b]
                # 1) si v_world == A ➜ connecté à B ; si v_world == B ➜ connecté à A
                if (abs(v_world[0]-A[0]) <= eps and abs(v_world[1]-A[1]) <= eps):
                    name = self._display_name(b, lab)
                    dx, dy = float(B[0]-v_world[0]), float(B[1]-v_world[1])
                    dist = (dx*dx+dy*dy)**0.5
                    if name:
                        out_candidates.append((name, dx, dy, dist, (float(B[0]), float(B[1]))))
                    continue
                if (abs(v_world[0]-B[0]) <= eps and abs(v_world[1]-B[1]) <= eps):
                    name = self._display_name(a, lab)
                    dx, dy = float(A[0]-v_world[0]), float(A[1]-v_world[1])
                    dist = (dx*dx+dy*dy)**0.5
                    if name:
                        out_candidates.append((name, dx, dy, dist, (float(A[0]), float(A[1]))))
                    continue
                # 2) sinon : le point est-il sur le SEGMENT [A,B] ? ➜ connecté aux deux extrémités
                if self._point_on_segment(v_world, A, B, eps):
                    name_a = self._display_name(a, lab)
                    name_b = self._display_name(b, lab)
                    dx, dy = float(A[0]-v_world[0]), float(A[1]-v_world[1])
                    dist = (dx*dx+dy*dy)**0.5
                    if name_a:
                        out_candidates.append((name_a, dx, dy, dist, (float(A[0]), float(A[1]))))
                    dx, dy = float(B[0]-v_world[0]), float(B[1]-v_world[1])
                    dist = (dx*dx+dy*dy)**0.5
                    if name_b:
                        out_candidates.append((name_b, dx, dy, dist, (float(B[0]), float(B[1]))))
        if not out_candidates:
            return []
        # Déduplication UNIQUE par (nom + position voisine quantifiée)
        by_key = {}  # key = (name, qx, qy) -> (dist, pt)
        def q(p):
            return (round(neigh_xy[0]/eps)*eps, round(neigh_xy[1]/eps)*eps)
        for (name, dx, dy, dist, neigh_xy) in out_candidates:
            key = (name, *q(neigh_xy))
            if (key not in by_key) or (dist > by_key[key][0]):
                by_key[key] = (dist, neigh_xy)
        # Restituer [{name, pt}] triés par distance décroissante (optionnel)
        out = []
        for (name, qx, qy), (dist, pt) in sorted(by_key.items(), key=lambda kv: -kv[1][0]):
            out.append({"name": name, "pt": (float(pt[0]), float(pt[1]))})
        return out

    def _on_canvas_motion_update_drag(self, event):
        # Toujours garantir un pick-cache à jour avant tout hit/tooltip
        self._ensure_pick_cache()

        # 1) Drag & drop depuis la liste → fantôme
        if self._drag:
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            tri = self._drag["triangle"]
            P = tri["pts"]
            dx = wx - float(P["O"][0])
            dy = wy - float(P["O"][1])
            O = np.array([P["O"][0] + dx, P["O"][1] + dy])
            B = np.array([P["B"][0] + dx, P["B"][1] + dy])
            L = np.array([P["L"][0] + dx, P["L"][1] + dy])
            self._drag["world_pts"] = {"O": O, "B": B, "L": L}
            coords = []
            for pt in (O, B, L):
                sx, sy = self._world_to_screen(pt)
                coords += [sx, sy]
            if self._drag_preview_id is None:
                self._drag_preview_id = self.canvas.create_polygon(*coords, outline="gray50", dash=(4,2), fill="", width=2)
            else:
                self.canvas.coords(self._drag_preview_id, *coords)
            return

        # 2) Mode rotation : suivre la souris sans bouton appuyé
        if self._sel and self._sel.get("mode") == "rotate":
            idx = self._sel["idx"]
            sel = self._sel
            pivot = sel["pivot"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            cur_angle = math.atan2(wy - pivot[1], wx - pivot[0])
            dtheta = cur_angle - sel["start_angle"]
            c, s = math.cos(dtheta), math.sin(dtheta)
            R = np.array([[c, -s], [s, c]], dtype=float)
            P = self._last_drawn[idx]["pts"]
            for k in ("O", "B", "L"):
                v = sel["orig_pts"][k] - pivot
                P[k] = (R @ v) + pivot
            self._redraw_from(self._last_drawn)
            self._sel["last_angle"] = cur_angle
            return
        # 2b) Mode rotation de GROUPE : suivre la souris (sans bouton appuyé)
        if self._sel and self._sel.get("mode") == "rotate_group":
            sel = self._sel
            gid = sel["gid"]
            pivot = sel["pivot"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            cur_angle = math.atan2(wy - pivot[1], wx - pivot[0])
            dtheta = cur_angle - sel["start_angle"]
            c, s = math.cos(dtheta), math.sin(dtheta)
            R = np.array([[c, -s], [s, c]], dtype=float)
            g = self.groups.get(gid)
            if not g:
                return
            # repartir du snapshot pour éviter l'accumulation d'erreurs
            for node in g["nodes"]:
                tid = node["tid"]
                if 0 <= tid < len(self._last_drawn) and tid in sel["orig_group_pts"]:
                    Pt = self._last_drawn[tid]["pts"]
                    Orig = sel["orig_group_pts"][tid]
                    for k in ("O","B","L"):
                        v = np.array(Orig[k], dtype=float) - pivot
                        Pt[k] = (R @ v) + pivot
            self._recompute_group_bbox(gid)
            self._redraw_from(self._last_drawn)
            self._sel["last_angle"] = cur_angle
            return

        # 3) Pas de drag/rotation : gestion du TOOLTIP (survol de sommet)
        try:
            mode, idx, extra = self._hit_test(event.x, event.y)
        except Exception:
            mode, idx, extra = (None, None, None)
        # En mode déconnexion (CTRL), ne pas afficher de tooltip
        if getattr(self, "_ctrl_down", False):
            self._hide_tooltip()
            return

        if mode == "vertex" and idx is not None:
            vkey = extra if isinstance(extra, str) else None
            # position monde du sommet visé
            try:
                P0 = self._last_drawn[idx]["pts"]
                v_world = np.array(P0[vkey], dtype=float) if vkey in ("O","B","L") else None
            except Exception:
                v_world = None
            lines = []
            if v_world is not None:
                # tolérance monde proportionnelle à l'affichage (hit_px/zoom)
                tol_world = max(1e-9, float(getattr(self, "_hit_px", 12)) / max(self.zoom, 1e-9))
                def same_pos(a, b):
                    return (abs(float(a[0]) - float(b[0])) <= tol_world) and \
                           (abs(float(a[1]) - float(b[1])) <= tol_world)
                # parcourir le GROUPE uniquement
                gid = self._get_group_of_triangle(idx)
                nodes = self._group_nodes(gid) if gid else []
                seen = set()  # éviter doublons texte
                for nd in nodes:
                    tid = nd.get("tid")
                    if tid is None or not (0 <= tid < len(self._last_drawn)):
                        continue
                    tri = self._last_drawn[tid]
                    Pt = tri["pts"]
                    labels = tri.get("labels", ("Bourges","",""))
                    # Préfixes comme dans la liste : O:..., B:..., L:...
                    for key, lbl in (("O", labels[0] if len(labels) > 0 else ""),
                                     ("B", labels[1] if len(labels) > 1 else ""),
                                     ("L", labels[2] if len(labels) > 2 else "")):
                        try:
                            if same_pos(Pt[key], v_world):
                                txt = f"{key}:{str(lbl).strip()}"
                                if txt and txt not in seen:
                                    lines.append(txt); seen.add(txt)
                        except Exception:
                            continue
                # ---- LIGNE(S) DU DESSOUS : arêtes connectées ----
                try:
                    connected = self._connected_vertices_from(v_world, gid, tol_world)
                except Exception:
                    connected = []
                if connected:
                    # séparateur visuel entre “sommet(s)” et “connectés”
                    lines.append("")  # ligne vide
                    # chaque voisin: "-> Nom — ddd° / mm.m' / h.h"
                    for item in connected:
                        try:
                            name = item.get("name", "")
                            pt   = item.get("pt", None)
                        except Exception:
                            name, pt = (str(item), None)
                        if not name:
                            continue
                        if pt is None:
                            lines.append(f"-> {name}")
                            continue
                        az = self._azimutDegEtMinute(v_world, pt)
                        if az is None:
                            lines.append(f"-> {name} — azimut indéterminé")
                        else:
                            degInt, minFloat, hourFloat = az
                            # Alignements lisibles: degrés sur 3, minutes & heures avec 1 décimale
                            lines.append(f"-> {name} — {degInt:03d}° / {minFloat:04.1f}' / {hourFloat:0.1f}h")
            tooltip_txt = "\n".join(lines)
            if tooltip_txt:
                # === Placement robuste par CENTRE du tooltip ===
                # 1) Coordonnées CANVAS du sommet et du barycentre
                sx_v, sy_v = self._world_to_screen(v_world)
                try:
                    Cx = (P0["O"][0] + P0["B"][0] + P0["L"][0]) / 3.0
                    Cy = (P0["O"][1] + P0["B"][1] + P0["L"][1]) / 3.0
                except Exception:
                    Cx, Cy = float(v_world[0]), float(v_world[1])
                sx_c, sy_c = self._world_to_screen((Cx, Cy))
                # 2) Direction (centroïde -> sommet) en PIXELS CANVAS
                vx = float(sx_v) - float(sx_c)
                vy = float(sy_v) - float(sy_c)
                n  = (vx*vx + vy*vy) ** 0.5
                if n <= 1e-6:
                    # secours : pousser vers le haut écran
                    ux, uy = 0.0, -1.0
                else:
                    ux, uy = (vx / n), (vy / n)

                # 3) Mesurer le tooltip pour connaître sa diagonale
                #    (on l'affiche/MAJ en mémoire, sans se soucier de la position pour l'instant)
                if self._tooltip is None or not self._tooltip.winfo_exists():
                    # créer/mesurer via helper centre, posé provisoirement au sommet
                    self._show_tooltip_at_center(tooltip_txt, sx_v, sy_v)
                else:
                    self._tooltip_label.config(text=tooltip_txt)
                    self._tooltip.update_idletasks()
                tw = max(1, int(self._tooltip.winfo_width()))
                th = max(1, int(self._tooltip.winfo_height()))

                # 4) Distance à partir du SOMMET jusqu'au CENTRE du tooltip
                #    Utiliser le support d'un rectangle axis-aligné : r(θ)=|rx·cosθ|+|ry·sinθ|
                #    pour projeter la demi-taille du tooltip le long de (centroïde→sommet).
                rx = 0.5 * tw
                ry = 0.5 * th
                ang = math.atan2(uy, ux)
                r_eff = abs(rx * math.cos(ang)) + abs(ry * math.sin(ang))
                cushion_px  = float(getattr(self, "_tooltip_cushion_px", 14))
                dist = r_eff + cushion_px
                cx_tip = sx_v + ux * dist
                cy_tip = sy_v + uy * dist
    
                # 5) Placement centré (convertit canvas->écran en interne)
                self._show_tooltip_at_center(tooltip_txt, cx_tip, cy_tip)
            else:
                self._hide_tooltip()
        else:
            # pas de sommet → masquer tooltip
            self._hide_tooltip()

        # sécurité : si le cache n’est pas prêt (post-load), le régénérer pour les tooltips
        if not self._pick_cache_valid:
            self._rebuild_pick_cache()

    # ---------- Lien + surlignage faces candidates ----------
    def _find_nearest_vertex(self, v_world, exclude_idx=None, exclude_gid=None):
        """Retourne (idx_triangle, key('O'|'B'|'L'), pos_world) du sommet d'un AUTRE triangle le plus proche.
        On peut exclure un triangle précis (exclude_idx) et/ou tout un groupe (exclude_gid)."""
        best = None
        best_d2 = None
        for j, t in enumerate(self._last_drawn):
            if j == exclude_idx:
                continue
            if exclude_gid is not None and t.get("group_id", None) == exclude_gid:
                continue            
            P = t["pts"]
            for k in ("O","B","L"):
                w = np.array(P[k], dtype=float)
                d2 = float((w[0]-v_world[0])**2 + (w[1]-v_world[1])**2)
                if (best_d2 is None) or (d2 < best_d2):
                    best_d2 = d2
                    best = (j, k, w)
        return best

    def _update_nearest_line(self, v_world, exclude_idx=None, exclude_gid=None):
        """Dessine (ou MAJ) un trait fin entre v_world et le sommet le plus proche d'un AUTRE triangle."""
        found = self._find_nearest_vertex(v_world, exclude_idx=exclude_idx, exclude_gid=exclude_gid)

        if found is None:
            self._clear_nearest_line()
            return
        _, _, best = found
        # tracer/mettre à jour
        x1, y1 = self._world_to_screen(v_world)
        x2, y2 = self._world_to_screen(best)
        if self._nearest_line_id is None:
            self._nearest_line_id = self.canvas.create_line(x1, y1, x2, y2, fill="#888888", width=1)
        else:
            self.canvas.coords(self._nearest_line_id, x1, y1, x2, y2)

    def _clear_nearest_line(self):
        if self._nearest_line_id is not None:
            try:
                self.canvas.delete(self._nearest_line_id)
            except Exception:
                pass
            self._nearest_line_id = None

    def _reset_assist(self):
        """Nettoie TOUTES les aides visuelles et l'état associé."""
        self._clear_nearest_line()
        self._clear_edge_highlights()
        self._edge_highlights = None
        self._edge_choice = None

    def _draw_temp_edge_world(self, p1, p2, color="#ff7f00", width=3):
        """
        Trace une ligne temporaire en coordonnées MONDE entre p1 et p2.
        p1/p2 : np.array([x, y]) ou tuple (x, y)
        """
        x1, y1 = self._world_to_screen(p1)
        x2, y2 = self._world_to_screen(p2)
        try:
            _id = self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
        except Exception:
            _id = None
        return _id

    # ---------- utilitaires contour de groupe ----------
    def _group_outline_segments(self, gid: int, eps: float = EPS_WORLD):
        """
        Retourne une liste de sous-segments (P1,P2) qui appartiennent au
        **contour extérieur** de l’union des triangles du groupe `gid`.
        (On retire donc toutes les arêtes internes.)
        """
        g = self.groups.get(gid) or {}
        nodes = g.get("nodes", [])
        if not nodes:
            return []

        # 1) collecter triangles (monde) + tous les sommets
        tris = []
        vertices = []
        for nd in nodes:
            tid = nd.get("tid")
            if tid is None or not (0 <= tid < len(self._last_drawn)):
                continue
            P = self._last_drawn[tid]["pts"]
            tris.append(_ShPoly([tuple(P["O"]), tuple(P["B"]), tuple(P["L"])]))
            vertices.extend([np.array(P["O"], float), np.array(P["B"], float), np.array(P["L"], float)])

        if not tris:
            return []

        # 2) union → frontière (LineString/MultiLineString)
        poly = _sh_union([p for p in tris if p.is_valid and p.area > 0]).buffer(0)
        boundary = poly.boundary  # peut être MultiLineString
        if boundary.is_empty:
            return []

        # helper
        def _split_by_vertices(A, B):
            """Découpe [A,B] par les sommets colinéaires situés dessus."""
            A = np.array(A, float); B = np.array(B, float)
            AB = B - A
            ts = [0.0, 1.0]
            for V in vertices:
                AV = V - A
                cross = abs(AB[0]*AV[1] - AB[1]*AV[0])
                if cross > eps:
                    continue
                dot = AV[0]*AB[0] + AV[1]*AB[1]
                ab2 = AB[0]*AB[0] + AB[1]*AB[1]
                if -eps <= dot <= ab2 + eps:
                    t = 0.0 if ab2 == 0 else dot / ab2
                    if 1e-9 < t < 1 - 1e-9:
                        ts.append(float(t))
            ts = sorted(set(ts))
            out = []
            for i in range(len(ts) - 1):
                q0 = A + (B - A) * ts[i]
                q1 = A + (B - A) * ts[i+1]
                if np.linalg.norm(q1 - q0) > eps:
                    out.append((q0, q1))
            return out

        # 3) prendre toutes les arêtes des triangles, les découper,
        #    puis NE GARDER que les sous-segments dont le **milieu** est sur la frontière
        outline = []
        # Frontière Shapely du solide de groupe
        boundary = poly.boundary if hasattr(poly, "boundary") else _ShLine([])
        # Tolérance en unités "monde" (liée à l'épaisseur d'affichage)
        tol = max(1e-9, float(getattr(self, "stroke_px", 2)) / max(getattr(self, "zoom", 1.0), 1e-9))
        bmask = boundary.buffer(tol) if (not boundary.is_empty) else boundary
        def _push(P):
            for a, b in ((P["O"], P["B"]), (P["B"], P["L"]), (P["L"], P["O"])):
                for s in _split_by_vertices(a, b):
                    seg = _ShLine([
                        (float(s[0][0]), float(s[0][1])),
                        (float(s[1][0]), float(s[1][1]))
                    ])
                    # Garder uniquement les sous-segments véritablement sur le périmètre
                    if (not seg.is_empty) and seg.within(bmask):
                        outline.append(s)

        for nd in nodes:
            tid = nd.get("tid")
            if 0 <= tid < len(self._last_drawn):
                _push(self._last_drawn[tid]["pts"])

        return outline

    # --- util: segments d'enveloppe (groupe ou triangle seul) ---
    def _outline_for_item(self, idx: int):
        """
        Retourne les segments (p1,p2) de l'enveloppe **du groupe** auquel appartient idx.
        Hypothèse d'architecture : tout item a un group_id (singleton possible).
        """
        gid = self._last_drawn[idx].get("group_id")
        assert gid is not None, "Invariant brisé: item sans group_id"
        try:
            return self._group_outline_segments(gid, eps=EPS_WORLD)
        except Exception:
            return []


    def _update_edge_highlights(self, mob_idx: int, vkey_m: str, tgt_idx: int, tgt_vkey: str):
        """Version factorisée 'graphe de frontière' (symétrique) :
        1) On récupère l'outline des deux groupes
        2) On construit un graphe de frontière (half-edges) côté mobile & cible
        3) Au sommet cliqué de chaque côté, on prend **les 2 demi-arêtes incidentes**
        4) On sélectionne la paire (mobile×cible) qui **minimise globalement** Δ-angle
        5) Affichage : toujours bleues (incidentes), rouge pour la paire choisie
        """

        import numpy as np
        from math import atan2, pi

        def _to_np(p): return np.array([float(p[0]), float(p[1])], dtype=float)
        def _azim(a, b):
            # tuple-safe: accepte tuples ou np.array
            from math import atan2
            return atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))
        def _ang_dist(a, b):
             from math import pi
             d = abs(a - b) % (2*pi)
             return d if d <= pi else (2*pi - d)
        def _almost_eq(a,b,eps=EPS_WORLD): return abs(a[0]-b[0])<=eps and abs(a[1]-b[1])<=eps

        # Sommets & groupes
        tri_m = self._last_drawn[mob_idx];  Pm = tri_m["pts"]; vm = _to_np(Pm[vkey_m])
        tri_t = self._last_drawn[tgt_idx];  Pt = tri_t["pts"]; vt = _to_np(Pt[tgt_vkey])
        gid_m = tri_m.get("group_id"); gid_t = tri_t.get("group_id")
        if gid_m is None or gid_t is None:
            self._clear_edge_highlights(); self._edge_choice = None; return
        
        # Tolérance (utile à d'autres endroits si besoin)
        tol_world = max(1e-9, float(getattr(self, "stroke_px", 2)) / max(getattr(self, "zoom", 1.0), 1e-9))

        # === Collecte via graphe de frontière puis argmin global (Δ-angle) ===
        # 1) outlines des deux groupes
        mob_outline = self._outline_for_item(mob_idx) or []
        tgt_outline = self._outline_for_item(tgt_idx) or []
        # 2) half-edges incidentes au sommet cliqué, côté mobile et côté cible
        g_m = self._build_boundary_graph(mob_outline)
        g_t = self._build_boundary_graph(tgt_outline)
        m_inc_raw = self._incident_half_edges_at_vertex(g_m, vm)
        t_inc_raw = self._incident_half_edges_at_vertex(g_t, vt)
        # 3) normalisation (granularité identique à l’outline) pour l’affichage
        m_inc = self._normalize_to_outline_granularity(mob_outline, m_inc_raw, eps=EPS_WORLD)
        t_inc = self._normalize_to_outline_granularity(tgt_outline, t_inc_raw, eps=EPS_WORLD)

        # 4) sélection globale : minimiser l’écart d’azimut + anti-chevauchement
        best = None  # (score, m_edge, t_edge)
        # Unions géométriques brutes des groupes (avant pose)
        S_t = _group_shape_from_nodes(self._group_nodes(gid_t), self._last_drawn)
        S_m_base = _group_shape_from_nodes(self._group_nodes(gid_m), self._last_drawn)
        # Helper de pose rigide (vm->vt, alignement (vm→mB) // (vt→tB))
        from math import atan2, pi
        def _place_union_on_pair(S_base, vm_pt, mB_pt, vt_pt, tB_pt):
            try:
                from shapely.affinity import translate, rotate
            except Exception:
                return None  # si Shapely indispo → on ne filtre pas
            vmx, vmy = float(vm_pt[0]), float(vm_pt[1])
            mBx, mBy = float(mB_pt[0]), float(mB_pt[1])
            vtx, vty = float(vt_pt[0]), float(vt_pt[1])
            tBx, tBy = float(tB_pt[0]), float(tB_pt[1])
            ang_m = atan2(mBy - vmy, mBx - vmx)
            ang_t = atan2(tBy - vty, tBx - vtx)
            dtheta = (ang_t - ang_m + pi) % (2*pi) - pi  # wrap [-pi,pi]
            # T =  translate(vt) ∘ rotate(dtheta around (0,0)) ∘ translate(-vm)
            S = translate(S_base, xoff=-vmx, yoff=-vmy)
            S = rotate(S, dtheta * 180.0 / pi, origin=(0.0, 0.0), use_radians=False)
            S = translate(S, xoff=vtx, yoff=vty)
            return S
        m_oriented = [ (a,b) if _almost_eq(a, vm) else (b,a) for (a,b) in (m_inc_raw or []) ]
        t_oriented = [ (a,b) if _almost_eq(a, vt) else (b,a) for (a,b) in (t_inc_raw or []) ]
        for me in m_oriented:
            azm = _azim(*me)
            for te in t_oriented:
                azt = _azim(*te)
                score = _ang_dist(azm, azt)
                # --- Anti-chevauchement (shrink-only) remis en service ---
                if not getattr(self, "debug_skip_overlap_highlight", False):
                    S_m_pose = _place_union_on_pair(S_m_base, me[0], me[1], te[0], te[1])
                    if S_m_pose is not None:
                        if _overlap_shrink(S_m_pose, S_t,
                                           getattr(self, "stroke_px", 2),
                                           max(getattr(self, "zoom", 1.0), 1e-9)):
                            continue  # paire rejetée pour chevauchement
                # argmin global (parmi les paires admissibles)
                if (best is None) or (score < best[0]):
                    best = (score, me, te)
        # 5) sorties visuelles
        mo = [(tuple(a), tuple(b)) for (a, b) in (mob_outline or [])]
        self._edge_highlights = {
            "all":        [(tuple(a), tuple(b)) for (a, b) in (t_inc or [])],
            "mob_inc":    [(tuple(a), tuple(b)) for (a, b) in (m_inc or [])],
            "tgt_inc":    [(tuple(a), tuple(b)) for (a, b) in (t_inc or [])],
            "best":       (tuple(best[1][0]), tuple(best[1][1]),
                           tuple(best[2][0]), tuple(best[2][1])) if best else None,
            "mob_outline": mo,
            "tgt_outline": [(tuple(a), tuple(b)) for (a, b) in (tgt_outline or [])],
        }

        self._edge_choice = None
        if best:
            (mA, mB), (tA, tB) = best[1], best[2]
            self._edge_choice = (mob_idx, vkey_m, tgt_idx, tgt_vkey, (tuple(mA), tuple(mB), tuple(tA), tuple(tB)))

        # Ajout des contours pour le debug (bleu)
        try:  mob_outline = [(tuple(a), tuple(b)) for (a,b) in self._outline_for_item(mob_idx)]
        except Exception: mob_outline = []
        try:  tgt_outline = [(tuple(a), tuple(b)) for (a,b) in self._outline_for_item(tgt_idx)]
        except Exception: tgt_outline = []
        self._redraw_edge_highlights()


    def _clear_edge_highlights(self):
        """Efface du canvas les lignes d'aide déjà dessinées.
        N'efface PAS self._edge_choice ni self._edge_highlights (elles peuvent être réutilisées)."""
        if self._edge_highlight_ids:
            for _id in self._edge_highlight_ids:
                try:
                    self.canvas.delete(_id)
                except Exception:
                    pass
        self._edge_highlight_ids = []

    def _redraw_edge_highlights(self):
        """Redessine les aides à partir de self._edge_highlights :
        - tout le périmètre (segments EXTÉRIEURS) des 2 groupes en BLEU (fin),
        - uniquement les segments INCIDENTS (candidats possibles) en BLEU **épais**,
        - meilleure paire (mobile & cible) en **ROUGE** par-dessus."""
        self._clear_edge_highlights()
        data = getattr(self, "_edge_highlights", None)
        if not data:
            return
        # 0) Contours (bleu, fin)
        blue = "#0B3D91"
        for key in ("mob_outline", "tgt_outline"):
            for (a, b) in data.get(key, []):
                _id = self._draw_temp_edge_world(np.array(a, float), np.array(b, float), color=blue, width=2)
                if _id:
                    self._edge_highlight_ids.append(_id)
        # 1) Segments INCIDENTS (candidats possibles connectés au sommet) — plus épais
        for key in ("mob_inc", "tgt_inc"):
            for (a, b) in data.get(key, []):
                _id = self._draw_temp_edge_world(np.array(a, float), np.array(b, float), color=blue, width=4)
                if _id:
                    self._edge_highlight_ids.append(_id)
        # 2) (optionnel) autres candidates calculées — en gris fin si présentes
        for (a, b) in data.get("all", []):
            _id = self._draw_temp_edge_world(np.array(a, float), np.array(b, float), color="#BBBBBB", width=1)
            if _id:
                self._edge_highlight_ids.append(_id)
        # 3) Meilleure (ROUGE, cible + mobile) — toujours par-dessus
        best = data.get("best")
        if best:
            # best = (m_a, m_b, t_a, t_b)
            m_a, m_b, t_a, t_b = best
            # Côté cible (épais)
            _id = self._draw_temp_edge_world(np.array(t_a, float), np.array(t_b, float), color="#FF0000", width=4)
            if _id:
                self._edge_highlight_ids.append(_id)
            # Côté mobile (épais également)
            _id = self._draw_temp_edge_world(np.array(m_a, float), np.array(m_b, float), color="#FF0000", width=4)
            if _id:
                self._edge_highlight_ids.append(_id)

            # côté mobile (fin) — l'arête entrée en contact
            _id2 = self._draw_temp_edge_world(np.array(m_a, float), np.array(m_b, float), color="#FF0000", width=3)
            if _id2:
                self._edge_highlight_ids.append(_id2)

    def _place_dragged_triangle(self):
        """Dépose un triangle ET crée immédiatement un groupe singleton cohérent."""
        if not self._drag or "world_pts" not in self._drag:
            return
        tri = self._drag["triangle"]
        Pw = self._drag["world_pts"]
        # 1) Ajout de l'item dans le document
        self._last_drawn.append({
            "labels": tri["labels"],
            "pts": Pw,
            "id": tri.get("id"),
            "mirrored": tri.get("mirrored", False),
        })
        new_tid = len(self._last_drawn) - 1
        # 2) Création d'un groupe singleton
        self._ensure_group_fields(self._last_drawn[new_tid])
        gid = self._new_group_id()
        self.groups[gid] = {
            "id": gid,
            "nodes": [ {"tid": new_tid, "vkey_in": None, "vkey_out": None} ],
            "bbox": None,
        }
        self._last_drawn[new_tid]["group_id"]  = gid
        self._last_drawn[new_tid]["group_pos"] = 0
        self._recompute_group_bbox(gid)
        # 3) UI
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Triangle déposé → Groupe #{gid} créé.")
        li = self._drag.get("list_index")  # conservé pour compat, même si on ne supprime plus
        # On ne supprime plus l'entrée de la listbox : on marque le triangle comme utilisé
        if tri.get("id") is not None:
            self._placed_ids.add(int(tri["id"]))
            self._update_triangle_listbox_colors()

        # Après le dépôt, on désélectionne le triangle dans la listbox
        # pour ne pas laisser un item actif alors qu'il est déjà utilisé
        self._in_triangle_select_guard = True
        if hasattr(self, "listbox"):
            self.listbox.selection_clear(0, tk.END)
        self._last_triangle_selection = None
        self._in_triangle_select_guard = False

    # =============   Groupes : helpers   ====================
    def _new_group_id(self) -> int:
        gid = self._next_group_id
        self._next_group_id += 1
        return gid

    def _ensure_group_fields(self, tri_dict: Dict) -> None:
        """Idempotent: garantit la présence des champs de groupe sur un triangle."""
        if "group_id" not in tri_dict:
            tri_dict["group_id"] = None
        if "group_pos" not in tri_dict:
            tri_dict["group_pos"] = None

    def _get_group_of_triangle(self, idx: int) -> Optional[int]:
        try:
            return self._last_drawn[idx].get("group_id", None)
        except Exception:
            return None

    def _group_nodes(self, gid: int) -> List[Dict]:
        g = self.groups.get(gid)
        return [] if not g else g["nodes"]

    def _recompute_group_bbox(self, gid: int):
        g = self.groups.get(gid)
        if not g or not g["nodes"]:
            return None
        xs, ys = [], []
        for node in g["nodes"]:
            tid = node["tid"]
            P = self._last_drawn[tid]["pts"]
            for k in ("O", "B", "L"):
                xs.append(float(P[k][0])); ys.append(float(P[k][1]))
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        g["bbox"] = (xmin, ymin, xmax, ymax)
        return g["bbox"]

    def _group_centroid(self, gid: int) -> Optional[np.ndarray]:
        """Barycentre de tous les sommets (O,B,L) du groupe gid."""
        g = self.groups.get(gid)
        if not g or not g.get("nodes"):
            return None
        sx = sy = 0.0
        n = 0
        for node in g["nodes"]:
            tid = node["tid"]
            if not (0 <= tid < len(self._last_drawn)):
                continue
            P = self._last_drawn[tid]["pts"]
            for k in ("O","B","L"):
                sx += float(P[k][0]); sy += float(P[k][1]); n += 1
        if n == 0:
            return None
        return np.array([sx/n, sy/n], dtype=float)

    # utilitaires de déconnexion/scission ---
    def _find_group_link_for_vertex(self, idx: int, vkey: str):
        """
        Si (idx,vkey) est un sommet de LIAISON dans son groupe, retourne
        (gid, pos, link_type) où link_type ∈ {"in","out"}.
        Sinon retourne (None, None, None).
        """
        gid = self._get_group_of_triangle(idx)
        if not gid:
            return (None, None, None)
        nodes = self._group_nodes(gid)
        pos = None
        for i, nd in enumerate(nodes):
            if nd.get("tid") == idx:
                pos = i
                break
        if pos is None:
            return (None, None, None)

        nd = nodes[pos]

        if nd.get("vkey_in") == vkey and pos > 0:
            return (gid, pos, "in")   # lien avec le précédent
        if nd.get("vkey_out") == vkey and pos < len(nodes)-1:
            return (gid, pos, "out")  # lien avec le suivant
        return (None, None, None)

    def _apply_group_meta_after_split_(self, gid: int):
        """Recalcule bbox et group_pos de tous les noeuds du groupe."""
        g = self.groups.get(gid)
        if not g:
            return
        for i, nd in enumerate(g["nodes"]):
            tid = nd["tid"]
            if 0 <= tid < len(self._last_drawn):
                self._last_drawn[tid]["group_id"]  = gid
                self._last_drawn[tid]["group_pos"] = i
        self._recompute_group_bbox(gid)


    def _split_group_at(self, gid: int, split_after_pos: int):
        """
        Scinde le groupe 'gid' ENTRE split_after_pos et split_after_pos+1.
        Retourne (left_gid, right_gid).
        Simplification : on conserve l'invariant « tout triangle appartient à un groupe ».
        - Le groupe de gauche réutilise 'gid' (même singleton).
        - Le groupe de droite reçoit toujours un nouveau gid (même singleton).
        """
        g = self.groups.get(gid)
        if not g:
            return (None, None)
        nodes = g["nodes"]
        if not (0 <= split_after_pos < len(nodes)-1):
            return (gid, None)  # rien à couper

        left_nodes  = [dict(n) for n in nodes[:split_after_pos+1]]
        right_nodes = [dict(n) for n in nodes[split_after_pos+1:]]

        # Corriger les vkey aux frontières : tête/queue
        if left_nodes:
            left_nodes[0]["vkey_in"] = None
            left_nodes[-1]["vkey_out"] = None
        if right_nodes:
            right_nodes[0]["vkey_in"] = None
            right_nodes[-1]["vkey_out"] = None

        # Appliquer côté gauche : réutilise gid (même singleton)
        left_gid = None
        if left_nodes:
            self.groups[gid] = {"id": gid, "nodes": left_nodes, "bbox": None}
            self._apply_group_meta_after_split_(gid)
            left_gid = gid

        # Appliquer côté droit : nouveau gid (même singleton)
        right_gid = None
        if right_nodes:
            new_gid = self._new_group_id()
            self.groups[new_gid] = {"id": new_gid, "nodes": right_nodes, "bbox": None}
            self._apply_group_meta_after_split_(new_gid)
            right_gid = new_gid

        return (left_gid, right_gid)

    def _link_triangles_into_group(self, idx_mob: int, vkey_mob: str, idx_tgt: int, vkey_tgt: str) -> None:
        try:
            tri_m = self._last_drawn[idx_mob]
            tri_t = self._last_drawn[idx_tgt]
        except Exception:
            return
        self._ensure_group_fields(tri_m); self._ensure_group_fields(tri_t)
        if tri_m["group_id"] is not None or tri_t["group_id"] is not None:
            return
        gid = self._new_group_id()
        nodes = [
            {"tid": idx_mob, "vkey_in": None,     "vkey_out": vkey_mob},
            {"tid": idx_tgt, "vkey_in": vkey_tgt, "vkey_out": None},
        ]
        self.groups[gid] = {"id": gid, "nodes": nodes, "bbox": None}
        tri_m["group_id"] = gid; tri_m["group_pos"] = 0
        tri_t["group_id"] = gid; tri_t["group_pos"] = 1
        self._recompute_group_bbox(gid)
        self.status.config(text=f"Groupe #{gid} créé : "
                                f"t0=tid{nodes[0]['tid']}({nodes[0]['vkey_out']}) "
                                f"→ t1=tid{nodes[1]['tid']}({nodes[1]['vkey_in']})")

    # --- Ajout d'un triangle isolé sur une extrémité de groupe ---
    def _append_triangle_to_group(self, gid: int, pos_tgt: int,
                                  idx_new: int, key_new: str, key_tgt: str) -> None:
        """
        Ajoute le triangle 'idx_new' (isolé) au groupe 'gid'.
        - Si pos_tgt == 0 -> PREPEND (le nouveau devient tête, relié à l'ancienne tête)
        - Si pos_tgt == len(nodes)-1 -> APPEND (le nouveau devient queue, relié à l'ancienne queue)
        key_new  : sommet utilisé côté 'nouveau'
        key_tgt  : sommet utilisé côté 'cible' (triangle d'extrémité)
        """
        g = self.groups.get(gid)
        if not g: return
        nodes = g["nodes"]
        n = len(nodes)
        if n == 0: return

        # sécurité : le mobile doit être isolé
        self._ensure_group_fields(self._last_drawn[idx_new])
        if self._last_drawn[idx_new]["group_id"] is not None:
            return  # on ne gère pas l'insertion groupe↔groupe ici

        if pos_tgt == 0:
            # PREPEND : new -> head
            head = nodes[0]
            head["vkey_in"] = key_tgt
            new_node = {"tid": idx_new, "vkey_in": None, "vkey_out": key_new}
            nodes.insert(0, new_node)
        elif pos_tgt == n-1:
            # APPEND : tail -> new
            tail = nodes[-1]
            tail["vkey_out"] = key_tgt
            new_node = {"tid": idx_new, "vkey_in": key_new, "vkey_out": None}
            nodes.append(new_node)
        else:
            # pas une extrémité -> on ne gère pas encore l'insertion au milieu
            return

        # MAJ meta group_id/group_pos et bbox
        for i, nd in enumerate(nodes):
            tid = nd["tid"]
            if 0 <= tid < len(self._last_drawn):
                self._ensure_group_fields(self._last_drawn[tid])
                self._last_drawn[tid]["group_id"]  = gid
                self._last_drawn[tid]["group_pos"] = i
        self._recompute_group_bbox(gid)
        self.status.config(text=f"Triangle ajouté au groupe #{gid}.")

    def _cancel_drag(self):
        # Mémoriser la source du drag avant de le remettre à zéro
        drag_info = self._drag
        self._drag = None

        if self._drag_preview_id is not None:
            self.canvas.delete(self._drag_preview_id)
            self._drag_preview_id = None

        # Toujours remettre le curseur normal quand on annule un drag (ESC ou autre)
        self.canvas.configure(cursor="")

        # Si le drag venait de la liste, on annule aussi la sélection du triangle
        if drag_info and drag_info.get("from") == "list" and hasattr(self, "listbox"):
            self._in_triangle_select_guard = True
            self.listbox.selection_clear(0, tk.END)
            self._in_triangle_select_guard = False
            self._last_triangle_selection = None

    def _on_escape_key(self, event):
        """Annuler un drag&drop (liste) ou un déplacement/selection de triangle (avec rollback)."""
        if self._drag:
            self._cancel_drag()
            self.status.config(text="Drag annulé (ESC).")
            return
        if self._sel:
            # rollback rotation de GROUPE
            if self._sel.get("mode") == "rotate_group":
                gid = self._sel.get("gid")
                orig = self._sel.get("orig_group_pts")
                if gid is not None and isinstance(orig, dict):
                    for tid, pts in orig.items():
                        if 0 <= tid < len(self._last_drawn):
                            self._last_drawn[tid]["pts"] = {k: np.array(pts[k].copy()) for k in ("O","B","L")}
                    self._recompute_group_bbox(gid)
                    self._redraw_from(self._last_drawn)
                self._sel = None
                self.status.config(text="Rotation de groupe annulée (ESC).")
                self._clear_nearest_line()
                self._clear_edge_highlights()
                return
            # rollback déplacement de GROUPE
            if self._sel.get("mode") == "move_group":
                gid = self._sel.get("gid")
                orig = self._sel.get("orig_group_pts")
                if gid is not None and isinstance(orig, dict):
                    for tid, pts in orig.items():
                        if 0 <= tid < len(self._last_drawn):
                            self._last_drawn[tid]["pts"] = {k: np.array(pts[k].copy()) for k in ("O","B","L")}
                    self._recompute_group_bbox(gid)
                    self._redraw_from(self._last_drawn)
                self._sel = None
                self.status.config(text="Déplacement de groupe annulé (ESC).")
                self._clear_nearest_line()
                self._clear_edge_highlights()
                return
            # rollback déplacement/édition d'un triangle seul
            idx = self._sel.get("idx")
            orig = self._sel.get("orig_pts")
            if idx is not None and orig is not None and 0 <= idx < len(self._last_drawn):
                self._last_drawn[idx]["pts"] = {k: np.array(orig[k].copy()) for k in ("O","B","L")}
                self._redraw_from(self._last_drawn)
            self._sel = None
            self.status.config(text="Action annulée (rollback).")
            self._clear_nearest_line()
            self._clear_edge_highlights()            

#
# ---------- Sélection / déplacement sur canvas ----------
    def _tri_centroid(self, P):
        return np.array([
            (P["O"][0] + P["B"][0] + P["L"][0]) / 3.0,
            (P["O"][1] + P["B"][1] + P["L"][1]) / 3.0
        ], dtype=float)

    def _hit_test(self, x, y):
        """Retourne ('center'|'vertex'|None, idx, extra) selon la zone cliquée.
        - 'vertex' si clic dans un disque autour d'un sommet
        - 'center' si clic à l'intérieur du triangle (hors disques sommets)
        """
        if not self._last_drawn:
            return (None, None, None)
        tol2 = float(self._hit_px) ** 2
        center_tol2 = float(getattr(self, "_center_hit_px", self._hit_px)) ** 2
        # Parcourt les triangles dans l'ordre inverse de dessin (avant-plan d'abord)
        for i in reversed(range(len(self._last_drawn))):
            t = self._last_drawn[i]
            P = t["pts"]
            # 1) tester d'abord les SOMMETS (priorité au mode "vertex")
            for key in ("O", "B", "L"):
                v = P[key]
                vs = np.array(self._world_to_screen(v))
                dv2 = (x - vs[0])**2 + (y - vs[1])**2
                if dv2 <= tol2:
                    return ("vertex", i, key)
            # 2) puis le CENTRE (tolérance plus petite)
            Cw = (P["O"] + P["B"] + P["L"]) / 3.0
            Cs = np.array(self._world_to_screen(Cw))
            ds2 = (x - Cs[0])**2 + (y - Cs[1])**2
            if ds2 <= center_tol2:
                return ("center", i, None)
        return (None, None, None)

    # ---------- Gestion liste déroulante : retrait / réinsertion ----------
    def _reinsert_triangle_to_list(self, tri):
        """Réinsère un triangle supprimé du canvas dans la listbox, trié par id."""
        tri_id = tri.get("id")
        if tri_id is None:
            return
        # Construire le libellé comme au chargement
        labels = tri.get("labels", ("", "", ""))
        b_val = labels[1] if len(labels) > 1 else ""
        l_val = labels[2] if len(labels) > 2 else ""
        entry = f"{int(tri_id):02d}. B:{b_val}  L:{l_val}"

        # Éviter les doublons si déjà présent
        for idx in range(self.listbox.size()):
            try:
                existing = self.listbox.get(idx)
                # on parse l'id au début (2 chiffres)
                ex_id = int(str(existing)[:2])
                if ex_id == int(tri_id):
                    break
            except Exception:
                continue
        else:
            # Trouver la position triée par id
            insert_at = self.listbox.size()
            for idx in range(self.listbox.size()):
                try:
                    ex = self.listbox.get(idx)
                    ex_id = int(str(ex)[:2])
                    if int(tri_id) < ex_id:
                        insert_at = idx
                        break
                except Exception:
                    continue
            self.listbox.insert(insert_at, entry)
        # Dé-marquer l'id comme posé
        if tri_id in self._placed_ids:
            self._placed_ids.discard(int(tri_id))
        # Mettre à jour la couleur des entrées (ce triangle redevient disponible)
        self._update_triangle_listbox_colors()

    # ---------- Clic droit / menu contextuel ----------
    def _on_canvas_right_click(self, event):
        """Affiche le menu contextuel si un triangle est cliqué (intérieur ou sommet)."""
        # Pas de menu si on est en train de drag depuis la liste
        if self._drag:
            return
        mode, idx, extra = self._hit_test(event.x, event.y)
        if idx is None:
            return
        # On ne propose Supprimer que si on est sur un triangle
        if mode in ("center", "vertex"):
            self._ctx_target_idx = idx
            self._ctx_last_rclick = (event.x, event.y)

            # Activer "OL=0°" seulement si le groupe est un singleton
            try:
                gid = self._get_group_of_triangle(idx)
                n_nodes = len(self._group_nodes(gid)) if gid else 0
                state = (tk.NORMAL if n_nodes == 1 else tk.DISABLED)
                # sécurité: si l'index n'existe pas, ne rien faire
                if hasattr(self, "_ctx_idx_ol0") and self._ctx_idx_ol0 is not None:
                    self._ctx_menu.entryconfig(self._ctx_idx_ol0, state=state)
            except Exception:
                # en cas d'erreur inattendue, mieux vaut désactiver
                try:
                    if hasattr(self, "_ctx_idx_ol0") and self._ctx_idx_ol0 is not None:
                        self._ctx_menu.entryconfig(self._ctx_idx_ol0, state=tk.DISABLED)
                except Exception:
                    pass
            # (ré)construire la section "mot" du menu selon le triangle visé + selection dico
            self._rebuild_ctx_word_entries()
            try:
                self._ctx_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self._ctx_menu.grab_release()

    def _ctx_delete_group(self):
        """Supprime **tout le groupe** du triangle ciblé, réinsère les triangles dans la liste,
        remappe les tids restants et recalcule les bbox de chaque groupe."""
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None or not (0 <= idx < len(self._last_drawn)):
            return
        gid = self._get_group_of_triangle(idx)
        if not gid:
            return
        g = self.groups.get(gid)
        if not g or not g.get("nodes"):
            return

        # --- Confirmation si le groupe comporte au moins 2 triangles ---
        try:
            n_nodes = len(g.get("nodes", []))
        except Exception:
            n_nodes = 0
        if n_nodes >= 2:
            confirm = messagebox.askyesno(
                "Supprimer le groupe",
                "Voulez-vous supprimer le groupe ?"
            )
            if not confirm:
                return

        # 1) Réinsertion listbox (avant de modifier _last_drawn)
        removed_tids = sorted({nd["tid"] for nd in g["nodes"] if "tid" in nd}, reverse=True)
        removed_set  = set(removed_tids)
        for tid in removed_tids:
            if 0 <= tid < len(self._last_drawn):
                self._reinsert_triangle_to_list(self._last_drawn[tid])

        # 2) Reconstruire _last_drawn et table de remap old->new
        keep, old2new = [], {}
        for old_i, tri in enumerate(self._last_drawn):
            if old_i in removed_set:
                continue
            new_i = len(keep)
            old2new[old_i] = new_i
            keep.append(tri)
        # IMPORTANT : on remplace le contenu de la liste en place
        # pour ne pas casser la référence scen.last_drawn.
        self._last_drawn[:] = keep

        # 3) Supprimer le groupe et remapper les autres
        if gid in self.groups:
            del self.groups[gid]
        for g2 in list(self.groups.values()):
            new_nodes = []
            for nd in g2.get("nodes", []):
                otid = nd.get("tid")
                if otid in removed_set:
                    continue
                nd2 = dict(nd)
                nd2["tid"] = old2new.get(otid, otid)
                new_nodes.append(nd2)
            g2["nodes"] = new_nodes
            # Reposer group_id/group_pos cohérents
            for i, nd in enumerate(g2["nodes"]):
                tid = nd["tid"]
                if 0 <= tid < len(self._last_drawn):
                    self._last_drawn[tid]["group_id"]  = g2["id"]
                    self._last_drawn[tid]["group_pos"] = i
            self._recompute_group_bbox(g2["id"])

        # 4) Fin : purge sélection/assist et redraw
        self._sel = None
        self._reset_assist()
        self._redraw_from(self._last_drawn)
        try:
            self.status.config(text=f"Groupe supprimé (gid={gid}, {len(removed_tids)} triangle(s)).")
        except Exception:
            pass

    def _ctx_rotate_selected(self):
        """Passe en mode rotation autour du barycentre pour le triangle ciblé."""
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None or not (0 <= idx < len(self._last_drawn)):
            return
        # Si le triangle fait partie d'un groupe : PIVOTER LE GROUPE
        gid = self._get_group_of_triangle(idx)
        if gid:
            pivot = self._group_centroid(gid)
            if pivot is None:
                return
            # snapshot complet du groupe pour rollback
            orig_group_pts = {}
            for node in self._group_nodes(gid):
                tid = node["tid"]
                if 0 <= tid < len(self._last_drawn):
                    Pt = self._last_drawn[tid]["pts"]
                    orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O","B","L")}
            # angle de départ = angle (pivot -> curseur au clic droit)
            if self._ctx_last_rclick:
                sx, sy = self._ctx_last_rclick
            else:
                sx, sy = self._world_to_screen(pivot)
            wx = (sx - self.offset[0]) / self.zoom
            wy = (self.offset[1] - sy) / self.zoom
            start_angle = math.atan2(wy - pivot[1], wx - pivot[0])
            self._sel = {
                "mode": "rotate_group",
                "gid": gid,
                "orig_group_pts": orig_group_pts,
                "pivot": np.array(pivot, dtype=float),
                "start_angle": start_angle,
            }
            self.status.config(text=f"Mode pivoter GROUPE #{gid} : bouge la souris pour tourner, clic gauche pour valider, ESC pour annuler.")
            return
        # Sinon : rotation TRIANGLE seul (comportement existant)
        P = self._last_drawn[idx]["pts"]
        pivot = self._tri_centroid(P)
        orig_pts = {k: np.array(P[k].copy()) for k in ("O","B","L")}
        if self._ctx_last_rclick:
            sx, sy = self._ctx_last_rclick
        else:
            sx, sy = self._world_to_screen(pivot)
        wx = (sx - self.offset[0]) / self.zoom
        wy = (self.offset[1] - sy) / self.zoom
        start_angle = math.atan2(wy - pivot[1], wx - pivot[0])
        self._sel = {
            "mode": "rotate",
            "idx": idx,
            "orig_pts": orig_pts,
            "pivot": np.array(pivot, dtype=float),
            "start_angle": start_angle,
        }
        self.status.config(text="Mode pivoter : bouge la souris pour tourner, clic gauche pour valider, ESC pour annuler.")
 
    def _ctx_orient_OL_north(self):
        """
        Oriente automatiquement le triangle pour que l'azimut du segment
        Ouverture->Lumière soit 0° = vers le Nord (axe +Y en coords monde).
        Rotation autour du barycentre (comme le mode Pivoter).
        """
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None or not (0 <= idx < len(self._last_drawn)):
            return
        P = self._last_drawn[idx]["pts"]
        # vecteur O->L en monde
        v = np.array([P["L"][0] - P["O"][0], P["L"][1] - P["O"][1]], dtype=float)
        if float(np.hypot(v[0], v[1])) < 1e-12:
            return  # triangle dégénéré, rien à faire
        cur = math.atan2(v[1], v[0])     # angle standard (x->y)
        target = math.pi / 2.0           # Nord = +Y
        dtheta = target - cur
        c, s = math.cos(dtheta), math.sin(dtheta)
        R = np.array([[c, -s], [s, c]], dtype=float)
        # pivot : barycentre (cohérent avec le mode Pivoter)
        pivot = self._tri_centroid(P)
        for k in ("O", "B", "L"):
            pt = np.array(P[k], dtype=float)
            P[k] = (R @ (pt - pivot)) + pivot
        self._redraw_from(self._last_drawn)
        self.status.config(text="Orientation appliquée : O→L au Nord (0°).")

    def _ctx_flip_selected(self):
        """
        Inverse **tout le GROUPE** par symétrie axiale rigide.
        Axe = direction (O→L) du triangle ciblé ; la droite passe par le **barycentre du groupe**.
        Chaque triangle du groupe garde sa forme; on bascule le flag 'mirrored' de tous.
        """
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None or not (0 <= idx < len(self._last_drawn)):
            return
        # --- Étendue : GROUPE du triangle ciblé ---
        gid = self._get_group_of_triangle(idx)
        if not gid:
            return
        nodes = self._group_nodes(gid)  # liste des triangles du groupe
        if not nodes:
            return

        # Axe = O→L du triangle ciblé (en monde)
        t0 = self._last_drawn[idx]
        P0 = t0["pts"]
        axis = np.array([P0["L"][0] - P0["O"][0], P0["L"][1] - P0["O"][1]], dtype=float)
        nrm = float(np.hypot(axis[0], axis[1]))
        if nrm < 1e-12:
            return  # triangle dégénéré
        u = axis / nrm

        # Pivot = barycentre du GROUPE (rigide pour tous les triangles)
        pivot = self._group_centroid(gid)
        if pivot is None:
            # secours : barycentre du triangle ciblé
            pivot = self._tri_centroid(P0)

        # Matrice de réflexion autour de la droite passant par 'pivot' et dirigée par u
        R = np.array([[2*u[0]*u[0] - 1, 2*u[0]*u[1]],
                      [2*u[0]*u[1],     2*u[1]*u[1] - 1]], dtype=float)

        # Appliquer la réflexion à TOUS les triangles du groupe
        for nd in nodes:
            tid = nd.get("tid")
            if tid is None or not (0 <= tid < len(self._last_drawn)):
                continue
            Pt = self._last_drawn[tid]["pts"]
            for k in ("O", "B", "L"):
                v = np.array([Pt[k][0] - pivot[0], Pt[k][1] - pivot[1]], dtype=float)
                Pt[k] = (R @ v) + pivot
            # Toggle du flag 'mirrored' pour l'affichage "S"
            self._last_drawn[tid]["mirrored"] = not self._last_drawn[tid].get("mirrored", False)

        # BBox groupe puis redraw
        try:
            self._recompute_group_bbox(gid)
        except Exception:
            pass
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Inversion appliquée au groupe #{gid}.")

    def _move_group_world(self, gid, dx_w, dy_w):
        """Translate rigide de tout le groupe en coordonnées monde."""
        g = self.groups.get(gid)
        if not g:
            return
        d = np.array([dx_w, dy_w], dtype=float)
        for node in g["nodes"]:
            tid = node["tid"]
            if 0 <= tid < len(self._last_drawn):
                P = self._last_drawn[tid]["pts"]
                for k in ("O","B","L"):
                    P[k] = np.array([P[k][0] + d[0], P[k][1] + d[1]], dtype=float)
        self._recompute_group_bbox(gid)

    def _on_canvas_left_down(self, event):
        # garantir un cache pick à jour avant tout hit-test
        self._ensure_pick_cache()
        # mémoriser l'ancre monde de la souris pour des déplacements "delta"
        try:
            self._mouse_world_prev = self._screen_to_world(event.x, event.y)
        except Exception:
            self._mouse_world_prev = None

        # Horloge : démarrer drag si clic dans le disque (marge 10px)
        if self._is_in_clock(event.x, event.y):
            # ne pas intercepter si un drag de triangle est en cours
            if not getattr(self, "_drag", None):
                self._clock_dragging = True
                self._clock_drag_dx = event.x - (self._clock_cx or event.x)
                self._clock_drag_dy = event.y - (self._clock_cy or event.y)
                try: self.canvas.configure(cursor="fleur")
                except Exception: pass
                return "break"  # on court-circuite la logique des triangles

        # Fond SVG : si mode resize et clic sur poignée -> on capture et on court-circuite le reste
        if self.bg_resize_mode.get() and self._bg:
            h = self._bg_hit_test_handle(event.x, event.y)
            if h:
                self._bg_start_resize(h, event.x, event.y)
                try: self.canvas.configure(cursor="sizing")
                except Exception: pass
                return "break"

        # Nouveau clic gauche : purge l'éventuelle aide précédente (évite les fantômes)
        # et masque le tooltip s'il est visible.
        self._hide_tooltip()
        self._reset_assist()
        # priorité au drag & drop depuis la liste
        if self._drag:
            return
        mode, idx, extra = self._hit_test(event.x, event.y)
        if mode == "center":
            P = self._last_drawn[idx]["pts"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            # Groupe obligatoire : on démarre toujours un move_group
            gid = self._get_group_of_triangle(idx)
            # En théorie, gid existe toujours (même un triangle seul est un groupe).
            # Si jamais non (état inattendu), on refuse de créer un mode 'move' triangle.
            if not gid:
                return
            Gc = self._group_centroid(gid)
            if Gc is None:
                return
            # snapshot complet du groupe pour rollback
            orig_group_pts = {}
            for node in self._group_nodes(gid):
                tid = node["tid"]
                if 0 <= tid < len(self._last_drawn):
                    Pt = self._last_drawn[tid]["pts"]
                    orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O","B","L")}
            self._sel = {
                "mode": "move_group",
                "gid": gid,
                "grab_offset": np.array([wx, wy]) - Gc,
                "orig_group_pts": orig_group_pts,
                "mouse_world_prev": np.array([wx, wy], dtype=float),
            }
            self.status.config(text=f"Déplacement du groupe #{gid} (clic sur tri {self._last_drawn[idx].get('id','?')})")
        elif mode == "vertex":
            # déplacement par sommet (translation comme 'center', mais calée sur le sommet choisi)
            vkey = extra or "O"
            P = self._last_drawn[idx]["pts"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            # NOTE: si on est en déconnexion, on désactivera toute aide au collage

            # ----- DEBUG étape 3 : clic sur sommet -----
            gid0 = self._get_group_of_triangle(idx)
            tri_id = self._last_drawn[idx].get("id","?")

            # si ce sommet est un LIEN -> on DECONNECTE et on démarre un move_group par sommet ---
            gid_link, pos, link_type = self._find_group_link_for_vertex(idx, vkey)
            if gid_link and self._ctrl_down:
                # Déterminer où couper :
                # - si click sur vkey_in (lien vers le précédent), on coupe AVANT 'pos' -> le morceau MOBILE = suffixe [pos..]
                # - si click sur vkey_out (lien vers le suivant), on coupe APRÈS  'pos' -> le morceau MOBILE = préfixe [..pos]
                nodes = self._group_nodes(gid_link)
                if link_type == "in":
                    cut_after = pos - 1
                    if cut_after < 0:  # sécurité (ne devrait pas arriver car vkey_in implies pos>0)
                        cut_after = 0
                else:  # "out"
                    cut_after = pos

                left_gid, right_gid = self._split_group_at(gid_link, cut_after)

                # Quel groupe contient le triangle cliqué après split ?
                mobile_gid = None
                # si pos <= cut_after : le tri cliqué est dans la PARTIE GAUCHE
                if pos <= cut_after and left_gid:
                    mobile_gid = left_gid
                # sinon dans la partie droite
                elif pos > cut_after and right_gid:
                    mobile_gid = right_gid

                # Invariant "100% groupes" : même singleton => on force le move_group (assist OFF)
                if mobile_gid is None:
                    mobile_gid = gid_link
                # --- Démarre un déplacement de GROUPE par le SOMMET cliqué ---
                orig_group_pts = {}
                for node in self._group_nodes(mobile_gid):
                    tid = node["tid"]
                    if 0 <= tid < len(self._last_drawn):
                        Pt = self._last_drawn[tid]["pts"]
                        orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O","B","L")}
                anchor_world = np.array(P[vkey], dtype=float)
                self._sel = {
                    "mode": "move_group",
                    "gid": mobile_gid,
                    "orig_group_pts": orig_group_pts,
                    "anchor": {"type": "vertex", "tid": idx, "vkey": vkey},
                    "grab_offset": np.array([wx, wy]) - anchor_world,
                    "suppress_assist": True,
                }
                self.status.config(text=f"Lien cassé. Déplacement du groupe #{mobile_gid} par sommet {vkey}.")
                self._clear_nearest_line()
                self._clear_edge_highlights()
                self._redraw_from(self._last_drawn)
                return

                # --- Démarre un déplacement de GROUPE par le SOMMET cliqué ---
                # Snapshot du groupe mobile pour rollback
                orig_group_pts = {}
                for node in self._group_nodes(mobile_gid):
                    tid = node["tid"]
                    if 0 <= tid < len(self._last_drawn):
                        Pt = self._last_drawn[tid]["pts"]
                        orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O","B","L")}

                # Ancre = sommet (du triangle cliqué) — on translate le groupe pour suivre ce sommet
                anchor_world = np.array(P[vkey], dtype=float)
                self._sel = {
                    "mode": "move_group",
                    "gid": mobile_gid,
                    "orig_group_pts": orig_group_pts,
                    "anchor": {"type": "vertex", "tid": idx, "vkey": vkey},
                    "grab_offset": np.array([wx, wy]) - anchor_world,
                    "suppress_assist": True,
                }
                self.status.config(text=f"Lien cassé. Déplacement du groupe #{mobile_gid} par sommet {vkey}.")
                # purge immédiate de toute aide existante
                self._clear_nearest_line()
                self._clear_edge_highlights()
                self._redraw_from(self._last_drawn)
                return

            # --- NOUVEAU (CTRL sans lien) : si CTRL est enfoncé mais que le sommet cliqué n'est PAS un lien,
            #                                on déplace le GROUPE entier ancré sur ce sommet, SANS assist.
            if self._ctrl_down:
                gid0 = self._get_group_of_triangle(idx)
                if gid0:
                    # snapshot du groupe (rollback sûr pendant le drag)
                    orig_group_pts = {}
                    for nd in self._group_nodes(gid0):
                        tid = nd["tid"]
                        if 0 <= tid < len(self._last_drawn):
                            Pt = self._last_drawn[tid]["pts"]
                            orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O", "B", "L")}

                    anchor_world = np.array(P[vkey], dtype=float)
                    self._sel = {
                        "mode": "move_group",
                        "gid": gid0,
                        "orig_group_pts": orig_group_pts,
                        "anchor": {"type": "vertex", "tid": idx, "vkey": vkey},
                        "grab_offset": np.array([wx, wy]) - anchor_world,
                        "suppress_assist": True,  # pas d'aides visuelles en CTRL
                    }
                    # nettoyer toute aide existante
                    self._reset_assist()
                    try:
                        self.status.config(text=f"CTRL sans lien : déplacement du groupe #{gid0} par sommet {vkey}.")
                    except Exception:
                        pass
                    self._redraw_from(self._last_drawn)
                    return

            # --- NOUVEAU : si le triangle appartient à un groupe et qu'on NE tient PAS CTRL,            #               on déplace le **groupe entier** ancré sur ce sommet.
            gid0 = self._get_group_of_triangle(idx)
            if gid0 and not self._ctrl_down:
                # snapshot du groupe (rollback sûr pendant le drag)
                orig_group_pts = {}
                for nd in self._group_nodes(gid0):
                    tid = nd["tid"]
                    if 0 <= tid < len(self._last_drawn):
                        Pt = self._last_drawn[tid]["pts"]
                        orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O","B","L")}

                anchor_world = np.array(P[vkey], dtype=float)
                self._sel = {
                    "mode": "move_group",
                    "gid": gid0,
                    "orig_group_pts": orig_group_pts,
                    "anchor": {"type": "vertex", "tid": idx, "vkey": vkey},
                    "grab_offset": np.array([wx, wy]) - anchor_world,
                    # on veut l'aide de collage active
                    "suppress_assist": False,
                }
                self.status.config(text=f"Déplacement du groupe #{gid0} par sommet {vkey}.")
                # Aide immédiate : viser un sommet d'un AUTRE triangle (exclure le groupe lui-même)
                v_world = np.array(P[vkey], dtype=float)
                tgt = self._find_nearest_vertex(v_world, exclude_idx=idx, exclude_gid=gid0)
                if tgt is not None:
                    j, tgt_key, _ = tgt
                    self._update_nearest_line(v_world, exclude_idx=idx, exclude_gid=gid0)
                    self._update_edge_highlights(idx, vkey, j, tgt_key)
                else:
                    self._clear_nearest_line()
                    self._clear_edge_highlights()
                return

            orig_pts = {k: np.array(P[k].copy()) for k in ("O","B","L")}
            self._sel = {
                "mode": "vertex",
                "idx": idx,
                "vkey": vkey,
                "grab_offset": np.array([wx, wy]) - np.array(P[vkey], dtype=float),
                "orig_pts": orig_pts,
            }
            # Affiche immédiatement la liaison + surlignage des arêtes candidates
            v_world = np.array(P[vkey], dtype=float)
            tgt = self._find_nearest_vertex(v_world, exclude_idx=idx)
            if tgt is not None:
                j, tgt_key, _ = tgt
                self._update_nearest_line(v_world, exclude_idx=idx)
                self._update_edge_highlights(idx, vkey, j, tgt_key)
            else:
                self._clear_nearest_line()
                self._clear_edge_highlights()
            self.status.config(text=f"Déplacement par sommet {vkey}.")
            return
        else:
            # clic ailleurs : pan au clic gauche
            self._on_pan_start(event)

    def _record_group_after_last_snap(self, idx_m: int, vkey_m: str, idx_t: int, vkey_t: str):
        """
        Version simplifiée : on ne gère que des **GROUPES** (les triangles sont
        toujours en groupe singleton). On fusionne **gid_m** et **gid_t** UNIQUEMENT
        en extrémité↔extrémité :
          - tail(m) → head(t) : tail_m.vkey_out = vkey_m ; head_t.vkey_in = vkey_t
          - head(m) → tail(t) : head_m.vkey_in = vkey_m ; tail_t.vkey_out = vkey_t
        Invariants :
          head.vkey_in is None ; tail.vkey_out is None ; liens internes complets.
        """
        tri_m = self._last_drawn[idx_m]
        tri_t = self._last_drawn[idx_t]
        gid_m = tri_m.get("group_id")
        gid_t = tri_t.get("group_id")

        # Rien à faire si mêmes groupes (collage interne non géré ici)
        if not gid_m or not gid_t or gid_m == gid_t:
            return

        nodes_m = self.groups[gid_m]["nodes"]
        nodes_t = self.groups[gid_t]["nodes"]
        head_m, tail_m = nodes_m[0], nodes_m[-1]
        head_t, tail_t = nodes_t[0], nodes_t[-1]

        # Cas A : queue(m) -> tête(t)
        if tri_m["group_pos"] == len(nodes_m) - 1 and tri_t["group_pos"] == 0:
            tail_m["vkey_out"] = vkey_m
            head_t["vkey_in"]  = vkey_t
            # concat t à la suite de m
            nodes_m.extend(nodes_t)
            for i, nd in enumerate(nodes_m):
                tid = nd["tid"]
                self._last_drawn[tid]["group_id"]  = gid_m
                self._last_drawn[tid]["group_pos"] = i
            # supprimer l'ancien groupe t
            del self.groups[gid_t]
            self._recompute_group_bbox(gid_m)
            return

        # Cas B : tête(m) -> queue(t)
        if tri_m["group_pos"] == 0 and tri_t["group_pos"] == len(nodes_t) - 1:
            head_m["vkey_in"]   = vkey_m
            tail_t["vkey_out"]  = vkey_t
            # concat m après t
            nodes_t.extend(nodes_m)
            for i, nd in enumerate(nodes_t):
                tid = nd["tid"]
                self._last_drawn[tid]["group_id"]  = gid_t
                self._last_drawn[tid]["group_pos"] = i
            del self.groups[gid_m]
            self._recompute_group_bbox(gid_t)
            return

        # Autres combinaisons (tête↔tête, queue↔queue, insertion interne) non gérées
        # dans cette version simplifiée. On ne modifie rien.
        return

    def _on_canvas_left_move(self, event):
        # Horloge : drag en cours -> on déplace le centre et on redessine l’overlay
        if getattr(self, "_clock_dragging", False):
            self._clock_cx = event.x - self._clock_drag_dx
            self._clock_cy = event.y - self._clock_drag_dy
            self._redraw_overlay_only()
            return "break"

        # Mode resize fond d'écran
        if self._bg_resizing:
            self._bg_update_resize(event.x, event.y)
            self._redraw_from(self._last_drawn)
            return "break"

        if self._drag:
            return  # le drag liste gère déjà le mouvement
        if not self._sel:
            self._on_pan_move(event)
            return

        # --- Déplacement de GROUPE ---
        if self._sel["mode"] == "move_group":
            # Pendant une déconnexion, ne jamais montrer l'aide de collage
            if self._sel.get("suppress_assist"):
                self._clear_nearest_line()
                self._clear_edge_highlights()
            gid = self._sel["gid"]
            # déplacement CALÉ SUR DELTA MONDE: w -> w_prev
            wx, wy = self._screen_to_world(event.x, event.y)
            prev = self._sel.get("mouse_world_prev")
            if prev is None:
                self._sel["mouse_world_prev"] = np.array([wx, wy], dtype=float)
                return
            dx, dy = float(wx - prev[0]), float(wy - prev[1])
            if dx != 0.0 or dy != 0.0:
                self._move_group_world(gid, dx, dy)
                # avancer l'ancre
                self._sel["mouse_world_prev"] = np.array([wx, wy], dtype=float)
            self._redraw_from(self._last_drawn)
            # --- NOUVEAU : pendant un move_group ancré sur SOMMET, afficher l'aide de collage ---
            if not self._sel.get("suppress_assist"):
                anchor = self._sel.get("anchor")
                if anchor and anchor.get("type") == "vertex":
                    anchor_tid = anchor.get("tid")
                    anchor_vkey = anchor.get("vkey")
                    if 0 <= anchor_tid < len(self._last_drawn):
                        Panchor = self._last_drawn[anchor_tid]["pts"]
                        v_world = np.array(Panchor[anchor_vkey], dtype=float)
                        # chercher une cible HORS du groupe mobile
                        tgt = self._find_nearest_vertex(v_world, exclude_idx=anchor_tid, exclude_gid=gid)
                        if tgt is not None:
                            j, tgt_key, _ = tgt
                            self._update_nearest_line(v_world, exclude_idx=anchor_tid, exclude_gid=gid)
                            self._update_edge_highlights(anchor_tid, anchor_vkey, j, tgt_key)
                        else:
                            self._clear_nearest_line(); self._clear_edge_highlights()            
            return
        elif self._sel["mode"] == "move":
            idx = self._sel["idx"]
            P = self._last_drawn[idx]["pts"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            target_c = np.array([wx, wy]) - self._sel["grab_offset"]
            cur_c = self._tri_centroid(P)
            d = target_c - cur_c
            for k in ("O", "B", "L"):
                P[k] = np.array([P[k][0] + d[0], P[k][1] + d[1]])
            self._redraw_from(self._last_drawn)
        elif self._sel["mode"] == "vertex":
            # translation calée sur un sommet précis
            idx = self._sel["idx"]
            vkey = self._sel["vkey"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            target_v = np.array([wx, wy]) - self._sel["grab_offset"]
            P = self._last_drawn[idx]["pts"]
            cur_v = np.array(P[vkey], dtype=float)
            d = target_v - cur_v
            for k in ("O", "B", "L"):
                P[k] = np.array([P[k][0] + d[0], P[k][1] + d[1]])
            # Redessine d'abord les triangles (efface tout)
            self._redraw_from(self._last_drawn)
            # En déconnexion, NE RIEN AFFICHER (pas de ligne grise, pas d'arêtes orange)
            if not self._sel.get("suppress_assist"):
                v_world = np.array(P[vkey], dtype=float)
                tgt = self._find_nearest_vertex(v_world, exclude_idx=idx)
                if tgt is not None:
                    (j, tgt_key, w) = tgt
                    self._update_nearest_line(v_world, exclude_idx=idx)
                    self._update_edge_highlights(idx, vkey, j, tgt_key)
            else:
                self._clear_nearest_line()
                self._clear_edge_highlights()
        elif self._sel["mode"] == "rotate":
            # désormais la rotation est gérée dans <Motion> (pas besoin de maintenir le clic)
            pass

    def _on_canvas_left_up(self, event):
        # Horloge : fin de drag
        if getattr(self, "_clock_dragging", False):
            self._clock_dragging = False
            try: self.canvas.configure(cursor="")
            except Exception: pass
            return "break"

        if self._bg_resizing:
            self._bg_resizing = None
            try: self.canvas.configure(cursor="")
            except Exception: pass
            self._redraw_from(self._last_drawn)
            return "break"

        """Fin du drag au clic gauche : dépôt de triangle (drag liste) OU fin d'une édition."""
        import numpy as np
        from math import atan2

        # 0) Dépôt d'un triangle glissé depuis la liste
        if self._drag:
            # Si le fantôme existe, on le supprime
            if self._drag_preview_id is not None:
                try:
                    self.canvas.delete(self._drag_preview_id)
                except Exception:
                    pass
                self._drag_preview_id = None

            # Sécurité : si world_pts n'a pas été posé (pas de <Motion>), le calculer ici
            if "world_pts" not in self._drag:
                tri = self._drag["triangle"]
                P = tri["pts"]
                wx = (event.x - self.offset[0]) / self.zoom
                wy = (self.offset[1] - event.y) / self.zoom
                dx = wx - float(P["O"][0])
                dy = wy - float(P["O"][1])
                O = np.array([P["O"][0] + dx, P["O"][1] + dy])
                B = np.array([P["B"][0] + dx, P["B"][1] + dy])
                L = np.array([P["L"][0] + dx, P["L"][1] + dy])
                self._drag["world_pts"] = {"O": O, "B": B, "L": L}

            # Dépose réellement le triangle dans le document
            self._place_dragged_triangle()
            # Reset état de drag
            self._drag = None
            # Remettre le curseur normal en fin de drag
            self.canvas.configure(cursor="")
            return

        # 1) Le reste est ton comportement existant (fin de drag/rotation/snap)
        if not self._sel:
            return

        mode = self._sel.get("mode")
        if mode == "vertex":
            idx = self._sel["idx"]
            vkey = self._sel["vkey"]

            # On a nécessairement choisi la meilleure arête dans _update_edge_highlights
            choice = getattr(self, "_edge_choice", None)

            # Nettoie correctement l'aide visuelle (liste d'ids + choix)
            self._clear_edge_highlights()

            if not choice or choice[0] != idx or choice[1] != vkey:
                # rien à coller -> simple fin de drag
                self._sel = None
                self._redraw_from(self._last_drawn)
                return

            # si CTRL est enfoncé AU RELÂCHEMENT, on ne colle pas.
            # (On garde l'aide visuelle pendant le drag si CTRL a été pressé après le clic.)
            if getattr(self, "_ctrl_down", False):
                self._sel = None
                self._reset_assist()
                self._redraw_from(self._last_drawn)
                try: self.status.config(text="Dépôt sans collage (CTRL).")
                except Exception: pass
                return


            # Déroulé du collage (rotation+translation) sur le TRIANGLE seul
            (_, _, idx_t, vkey_t, (m_a, m_b, t_a, t_b)) = choice
            tri_m = self._last_drawn[idx]
            Pm = tri_m["pts"]

            A = np.array(m_a, dtype=float)  # mobile: sommet saisi
            B = np.array(m_b, dtype=float)  # mobile: voisin qui définit l'arête
            U = np.array(t_a, dtype=float)  # cible: point où coller
            V = np.array(t_b, dtype=float)  # cible: deuxième point de l'arête

            # azimuts via helper unifié
            ang_m = self._ang_of_vec(B[0] - A[0], B[1] - A[1])
            ang_t = self._ang_of_vec(V[0] - U[0], V[1] - U[1])
            dtheta = ang_t - ang_m

            R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                        [np.sin(dtheta),  np.cos(dtheta)]], dtype=float)

            new_pts = {}
            for k in ("O", "B", "L"):
                p = np.array(Pm[k], dtype=float)
                p_rot = A + (R @ (p - A))
                new_pts[k] = p_rot

            # 2) translation pour amener A -> U
            delta = U - new_pts[vkey]
            for k in ("O", "B", "L"):
                new_pts[k] = new_pts[k] + delta

            # Applique la géométrie
            for k in ("O", "B", "L"):
                Pm[k][0] = float(new_pts[k][0])
                Pm[k][1] = float(new_pts[k][1])

            # Enregistre/étend les groupes (métadonnées & invariants)
            self._record_group_after_last_snap(idx, vkey, idx_t, vkey_t)
            # fin d'opération : on purge les aides et la sélection
            self._sel = None
            self._reset_assist()
            self._redraw_from(self._last_drawn)
            self.status.config(text="Triangles collés (sommets et arêtes alignés).")
            return

        elif mode == "move_group":
            # Collage du GROUPE quand on l'a déplacé PAR SOMMET (ancre=vertex)
            # et que l'aide de collage était active (pas en déconnexion).
            anchor = self._sel.get("anchor")
            suppress = self._sel.get("suppress_assist")

            # On nettoie l'aide visuelle dans tous les cas
            self._clear_edge_highlights()

            # Si l'aide était active et qu'on a bien un choix d'arête,
            # on applique la même géométrie que pour un triangle seul,
            # mais à TOUS les triangles du groupe.
            choice = getattr(self, "_edge_choice", None)
            if (not suppress
                and (not getattr(self, "_ctrl_down", False))
                and anchor
                and anchor.get("type") == "vertex"
                and choice
                and choice[0] == anchor.get("tid")
                and choice[1] == anchor.get("vkey")):

                # Déballage du choix d'arête: (mob_idx,vkey_m,tgt_idx,vkey_t,(m_a,m_b,t_a,t_b))
                (_, _, idx_t, vkey_t, (m_a, m_b, t_a, t_b)) = choice

                A = np.array(m_a, dtype=float)  # sommet mobile saisi (dans le groupe)
                B = np.array(m_b, dtype=float)  # voisin côté mobile (définit l'arête)
                U = np.array(t_a, dtype=float)  # cible: point d'accroche
                V = np.array(t_b, dtype=float)  # cible: second point arête

                # Angle mobile/cible et rotation à appliquer (helpers unifiés)
                ang_m = self._ang_of_vec(B[0] - A[0], B[1] - A[1])
                ang_t = self._ang_of_vec(V[0] - U[0], V[1] - U[1])
                dtheta = ang_t - ang_m
                R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                              [np.sin(dtheta),  np.cos(dtheta)]], dtype=float)

                # Translation finale pour amener A -> U
                # (après rotation autour de A, A reste à A)
                delta = U - A

                # Appliquer (rotation autour de A) + translation à tous
                gid = self._sel.get("gid")
                g = self.groups.get(gid)
                if g:
                    for node in g["nodes"]:
                        tid = node.get("tid")
                        if 0 <= tid < len(self._last_drawn):
                            P = self._last_drawn[tid]["pts"]
                            for k in ("O", "B", "L"):
                                p = np.array(P[k], dtype=float)
                                p_rot = A + (R @ (p - A))
                                p_fin = p_rot + delta
                                P[k][0] = float(p_fin[0])
                                P[k][1] = float(p_fin[1])

                # ====== FUSION DE GROUPE APRÈS COLLAGE (groupe ↔ groupe uniquement) ======
                tgt_gid = self._last_drawn[idx_t].get("group_id", None)
                mob_gid = gid
                if mob_gid is not None and tgt_gid is not None:
                    if tgt_gid != mob_gid:
                        # Fusionner tgt_gid → mob_gid AVEC LIEN EXPLICITE
                        g_src = self.groups.get(mob_gid)
                        g_tgt = self.groups.get(tgt_gid)
                        anchor = self._sel.get("anchor")  # {"type":"vertex","tid":...,"vkey":...}
                        choice = getattr(self, "_edge_choice", None)  # (... idx_t, vkey_t, ...)
                        if g_src and g_tgt and anchor and choice:
                            # 2.1 Localiser ancre (dans src) et cible (dans tgt)
                            anchor_tid = anchor.get("tid"); anchor_vkey = anchor.get("vkey")
                            (_, _, idx_t, vkey_t, _) = choice
                            nodes_src = g_src["nodes"]; nodes_tgt = g_tgt["nodes"]
                            pos_anchor = next((i for i, nd in enumerate(nodes_src) if nd.get("tid")==anchor_tid), None)
                            pos_target = next((i for i, nd in enumerate(nodes_tgt) if nd.get("tid")==idx_t), None)
                            if pos_anchor is None or pos_target is None:
                                # fallback: conserver l’ancien comportement (append brut)
                                for nd in nodes_tgt:
                                    tid2 = nd.get("tid"); 
                                    if tid2 is None: continue
                                    self._last_drawn[tid2]["group_id"] = mob_gid
                                    nodes_src.append({"tid": tid2,
                                                      "vkey_in": nd.get("vkey_in"),
                                                      "vkey_out": nd.get("vkey_out")})
                            else:
                                # 2.2 Construire une vue pivotée de tgt pour que idx_t soit AU BORD collé
                                def rotate(L, k): 
                                    k = k % len(L); 
                                    return L[k:]+L[:k]
                                # On privilégie "coller APRES l'ancre": ancre --(out)-> [tgt...]
                                insert_after = True
                                tgt_rot = rotate(list(nodes_tgt), pos_target)  # idx_t devient tgt_rot[0]
                                # Poser les marqueurs de lien complémentaires :
                                nd_anchor = nodes_src[pos_anchor]
                                if insert_after:
                                    # ancre sort → cible entre par vkey_in
                                    nd_anchor["vkey_out"] = anchor_vkey
                                    tgt_rot[0] = {"tid": tgt_rot[0]["tid"], "vkey_in": vkey_t, "vkey_out": tgt_rot[0].get("vkey_out")}
                                    # insérer juste après l’ancre
                                    nodes_src[pos_anchor+1:pos_anchor+1] = tgt_rot
                                else:
                                    # ancre entre par vkey_in ← cible sort
                                    nd_anchor["vkey_in"] = anchor_vkey
                                    tgt_rot[-1] = {"tid": tgt_rot[-1]["tid"], "vkey_in": tgt_rot[-1].get("vkey_in"), "vkey_out": vkey_t}
                                    # insérer juste avant l’ancre
                                    nodes_src[pos_anchor:pos_anchor] = tgt_rot
                                # MAJ group_id sur tout le bloc inséré
                                for nd in tgt_rot:
                                    tid2 = nd["tid"]
                                    if 0 <= tid2 < len(self._last_drawn):
                                        self._last_drawn[tid2]["group_id"] = mob_gid
                            # 2.3 Supprimer l'ancien groupe cible et reindexer pos/bbox
                            try: del self.groups[tgt_gid]
                            except Exception: pass
                            for i, nd in enumerate(nodes_src):
                                tid2 = nd["tid"]
                                if 0 <= tid2 < len(self._last_drawn):
                                    self._last_drawn[tid2]["group_id"]  = mob_gid
                                    self._last_drawn[tid2]["group_pos"] = i
                            self._recompute_group_bbox(mob_gid)
                            self.status.config(text=f"Groupes fusionnés avec lien: #{tgt_gid} → #{mob_gid}.")
                    else:
                        # cible déjà dans le même groupe → rien à faire côté structure
                        self.status.config(text=f"Groupe collé (même groupe #{mob_gid}).")
                # ====== /FUSION ======



            # Fin (pas de snap : simple dépôt à la dernière position)
            self._sel = None
            self._reset_assist()
            self._redraw_from(self._last_drawn)
            return

        elif mode == "rotate":
            self._sel = None
            self._reset_assist()
            self.status.config(text="Rotation validée.")
            self._redraw_from(self._last_drawn)
            return

        # Autres modes : on nettoie juste
        self._sel = None
        self._reset_assist()
        self._redraw_from(self._last_drawn)


    def _on_mousewheel(self, event):
        # Normalize wheel delta across platforms
        if hasattr(event, "delta") and event.delta != 0:
            dz = 1.1 if event.delta > 0 else 1/1.1
        else:
            dz = 1.1 if getattr(event, "num", 0) == 4 else 1/1.1

        # World coordinate under cursor BEFORE zoom
        wx = (event.x - self.offset[0]) / self.zoom
        wy = (self.offset[1] - event.y) / self.zoom

        # Apply zoom (clamped)
        self.zoom = max(0.05, min(100.0, self.zoom * dz))

        # Adjust offset so (wx,wy) remains under cursor AFTER zoom
        self.offset = np.array([event.x - wx * self.zoom,
                                event.y + wy * self.zoom], dtype=float)

        self._redraw_from(self._last_drawn)
        # zoom modifie les coords écran -> invalider le pick-cache
        self._invalidate_pick_cache()

    def _on_pan_start(self, event):
        self._pan_anchor = np.array([event.x, event.y], dtype=float)
        self._offset_anchor = self.offset.copy()

    def _on_pan_move(self, event):
        if getattr(self, "_pan_anchor", None) is None:
            return
        d = np.array([event.x, event.y], dtype=float) - self._pan_anchor
        self.offset = self._offset_anchor + d
        self._redraw_from(self._last_drawn)

    def _on_pan_end(self, event):
        self._pan_anchor = None
        # après pan -> invalider cache pick (coords écran changent)
        self._invalidate_pick_cache()

    # ---------- Impression PDF (A4) ----------
    def _export_triangles_pdf(
        self, path, triangles, scale_mm=1.5,
        page_margin_mm=12, cell_pad_mm=6,
        stroke_pt=1.2, font_size_pt=9, label_inset=0.35,
        pack_mode="shelf",           # "shelf" = rangées à hauteur variable
        rotate_to_fit=True           # rotation 90° si ça réduit la largeur en rangée
    ):
        """
        PDF A4 portrait, triangles à la même échelle, placement 'shelf packing'.
        """
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm

        if not triangles:
            raise ValueError("Aucun triangle à imprimer.")

        S = float(scale_mm) * float(mm)      # points par unité
        page_w, page_h = A4
        margin = float(page_margin_mm) * float(mm)
        pad    = float(cell_pad_mm) * float(mm)

        # Prépare bboxes à l’échelle (orientation normale OU rotée)
        items = []  # pour chaque tri : dict avec bbox_n (w,h) et bbox_r (w,h)
        for t in triangles:
            P = t["pts"]
            xs = [float(P["O"][0]), float(P["B"][0]), float(P["L"][0])]
            ys = [float(P["O"][1]), float(P["B"][1]), float(P["L"][1])]
            mnx, mny, mxx, mxy = min(xs), min(ys), max(xs), max(ys)

            tri_w = (mxx - mnx) * S
            tri_h = (mxy - mny) * S
            wN, hN = tri_w + 2*pad, tri_h + 2*pad            # normal
            wR, hR = tri_h + 2*pad, tri_w + 2*pad            # roté 90°

            items.append({
                "data": t,
                "bbox_world": (mnx, mny, mxx, mxy),
                "wN": wN, "hN": hN,
                "wR": wR, "hR": hR,
            })

        # Espace utile page
        content_w = page_w - 2*margin
        content_h = page_h - 2*margin

        # Placement 'shelf' (rangées)
        def new_page():
            return {"rows": [], "height_used": 0.0}

        def close_row(page, row):
            page["rows"].append(row)
            page["height_used"] += row["H"]

        page_list  = []
        page = new_page()
        row  = {"X": 0.0, "Y": 0.0, "H": 0.0, "items": []}

        for it in items:
            cand = []
            if rotate_to_fit:
                cand.append(("R", it["wR"], it["hR"]))
            cand.append(("N", it["wN"], it["hN"]))
            cand.sort(key=lambda x: x[1])  # largeur croissante

            placed = False
            for rot, w, h in cand:
                if row["X"] + w <= content_w or row["X"] == 0.0:
                    Hnew = max(row["H"], h) if row["H"] > 0 else h
                    if (page["height_used"] + Hnew) > content_h and row["X"] == 0.0:
                        if row["H"] > 0:
                            close_row(page, row)
                        page_list.append(page)
                        page = new_page()
                        row  = {"X": 0.0, "Y": 0.0, "H": 0.0, "items": []}

                    if row["X"] + w <= content_w or row["X"] == 0.0:
                        row["items"].append({"x": row["X"], "y": page["height_used"], "w": w, "h": h, "rot": (rot=="R"), "it": it})
                        row["X"] += w
                        row["H"] = max(row["H"], h)
                        placed = True
                        break

            if not placed:
                close_row(page, row)
                row  = {"X": 0.0, "Y": page["height_used"], "H": 0.0, "items": []}
                rot, w, h = cand[0]
                if page["height_used"] + h > content_h:
                    page_list.append(page)
                    page = new_page()
                    row  = {"X": 0.0, "Y": 0.0, "H": 0.0, "items": []}
                row["items"].append({"x": row["X"], "y": page["height_used"], "w": w, "h": h, "rot": (rot=="R"), "it": it})
                row["X"] += w
                row["H"] = max(row["H"], h)

        if row["items"]:
            close_row(page, row)
        page_list.append(page)

        # Dessin
        c = canvas.Canvas(path, pagesize=A4)
        c.setStrokeColorRGB(0, 0, 0)
        c.setFillColorRGB(0, 0, 0)

        def draw_tri(at_x, at_y, box_w, box_h, item, rot90):
            t   = item["data"]
            P   = t["pts"]
            Oname, Bname, Lname = t["labels"]
            mnx, mny, mxx, mxy = item["bbox_world"]

            tri_w = (mxx - mnx) * S
            tri_h = (mxy - mny) * S
            draw_w = tri_h if rot90 else tri_w
            draw_h = tri_w if rot90 else tri_h

            cx = margin + at_x + (box_w - draw_w) / 2.0
            cy = margin + at_y + (box_h - draw_h) / 2.0

            def to_page(p):
                x = (float(p[0]) - mnx) * S
                y = (float(p[1]) - mny) * S
                return (x, y)

            O = to_page(P["O"]); B = to_page(P["B"]); L = to_page(P["L"])

            c.saveState()
            c.translate(cx, cy)
            if rot90:
                c.translate(0, draw_h)
                c.rotate(-90)

            # traits
            c.setLineWidth(stroke_pt)
            c.line(O[0], O[1], B[0], B[1])
            c.line(B[0], B[1], L[0], L[1])
            c.line(L[0], L[1], O[0], O[1])

            # labels
            c.setFont("Helvetica", float(font_size_pt))
            gx = (O[0] + B[0] + L[0]) / 3.0
            gy = (O[1] + B[1] + L[1]) / 3.0
            for (px, py), txt in zip((O, B, L), (Oname, Bname, Lname)):
                lx = (1.0 - label_inset) * px + label_inset * gx
                ly = (1.0 - label_inset) * py + label_inset * gy
                c.drawCentredString(lx, ly, txt)
            c.restoreState()

        for pg in page_list:
            for row in pg["rows"]:
                for cell in row["items"]:
                    draw_tri(cell["x"], cell["y"], cell["w"], cell["h"], cell["it"], cell["rot"])
            c.showPage()
        c.save()

    def print_triangles_dialog(self):
        from tkinter import simpledialog, filedialog
        if getattr(self, "df", None) is None or self.df.empty:
            messagebox.showwarning("Imprimer", "Charge d'abord le fichier Excel.")
            return

        # Paramètres
        start = max(1, int(self.start_index.get()))
        nmax = int(self.df.shape[0] - (start-1))
        n = simpledialog.askinteger("Imprimer", f"Nombre de triangles (max {nmax}) :", initialvalue=min(8, nmax), minvalue=1, maxvalue=nmax)
        if not n:
            return

        scale = simpledialog.askfloat("Imprimer", "Échelle (mm par unité de longueur) :", initialvalue=1.5, minvalue=0.1, maxvalue=100.0)
        if not scale:
            return

        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile="triangles.pdf")
        if not path:
            return

        tris = self._triangles_local(start-1, n)
        self._export_triangles_pdf(path, tris, scale_mm=scale)
        self.status.config(text=f"PDF généré : {path}")

# ---------- Entrée ----------
if __name__ == "__main__":
    app = TriangleViewerManual()
    app.mainloop()
