
import os
import math
import re
import numpy as np
from shapely.geometry import Polygon as _ShPoly, LineString as _ShLine, Point as _ShPoint
from shapely.ops import unary_union as _sh_union
from shapely.ops import unary_union
from shapely.affinity import rotate as _sh_rotate, translate as _sh_translate
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional, List, Dict, Tuple

# ---------- Utils géométrie ----------
def _poly_from_tris(tris):
    """Construit la forme solide (Polygon/MultiPolygon) d'une liste de triangles."""
    polys = []
    for tri in tris:
        # tri peut être {"p": {"O":..,"B":..,"L":..}} ou directement {"O":..,"B":..,"L":..}
        p = tri.get("p", tri)
        poly = _ShPoly([tuple(p["O"]), tuple(p["B"]), tuple(p["L"])])
        if poly.is_valid and poly.area > 0:
            polys.append(poly)
    if not polys:
        return None
    # union + buffer(0) pour "nettoyer" les micro-fentes
    return _sh_union(polys).buffer(0)

def _groups_overlap_solid(mob_pose, tgt_group_tris, mob_group_tris=None, eps=1e-9, debug_label=None):
    """
    Retourne True s'il y a recouvrement d’aire entre :
      - la source (triangle posé ou groupe posé)
      - le groupe cible
    """
    # Si la source est un seul triangle, on fabrique un "groupe" d'un seul tri
    if mob_group_tris is None:
        mob_group_tris = [{"p": mob_pose}]

    mob_poly = _poly_from_tris(mob_group_tris)
    tgt_poly = _poly_from_tris(tgt_group_tris)
    if (mob_poly is None) or (tgt_poly is None):
        return False

    # INTÉRIEURS STRICTS : on ne compte pas les bords collés
    # (un seul shrink ici — évite les faux négatifs si triangles fins)
    mob_in = mob_poly.buffer(-eps).buffer(0)
    tgt_in = tgt_poly.buffer(-eps).buffer(0)
    if mob_in.is_empty or tgt_in.is_empty:
        return False

    inter = mob_in.intersection(tgt_in)
    if debug_label:
        print(f"[DBG] Solid-overlap {debug_label}: area={getattr(inter,'area',0.0):.6g}")
    return (not inter.is_empty) and (getattr(inter, "area", 0.0) > 0.0)

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

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / (n if n else 1.0)

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

# --- Segments utilitaires ---
def _segments_from_pts(P: Dict[str, np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, Tuple[str,str]]]:
    """Retourne les 3 segments d'un triangle (points monde) avec leurs clés (a,b)."""
    return [
        (np.array(P["O"], dtype=float), np.array(P["B"], dtype=float), ("O","B")),
        (np.array(P["B"], dtype=float), np.array(P["L"], dtype=float), ("B","L")),
        (np.array(P["L"], dtype=float), np.array(P["O"], dtype=float), ("L","O")),
    ]

def _seg_intersect_strict(a1, a2, b1, b2, eps: float = 1e-9) -> bool:
    """
    Vrai si [a1,a2] et [b1,b2] s'intersectent de façon "gênante" :
      - croisement propre (en X)  → LineString.crosses
      - OU recouvrement colinéaire de longueur > eps → LineString.overlaps + length
    Toucher par extrémité (simple contact) n’est pas considéré gênant.
    """
    L1 = _ShLine([tuple(map(float, a1)), tuple(map(float, a2))])
    L2 = _ShLine([tuple(map(float, b1)), tuple(map(float, b2))])
    # Croisement propre (intersections des intérieurs)
    if L1.crosses(L2):
        return True
    # Recouvrement colinéaire (les intérieurs se chevauchent sans se contenir)
    if L1.overlaps(L2):
        return L1.intersection(L2).length > eps
    return False

def _point_in_triangle_strict(P, A, B, C, eps=1e-9) -> bool:
    poly = _ShPoly([tuple(map(float, A)),
                    tuple(map(float, B)),
                    tuple(map(float, C))])
    if not poly.is_valid:
        return False
    pt = _ShPoint(tuple(map(float, P)))

    # si le point est sur un bord → pas "strict"
    if poly.touches(pt):
        return False

    # petit "shrink" pour éviter qu’un point quasi-sur-bord soit vu comme inside
    poly_shrunk = poly.buffer(-eps)
    return poly_shrunk.contains(pt)

def _edge_overlaps_group(edge_a: Tuple[np.ndarray,np.ndarray],
                         group_tris: List[Dict],
                         exclude: Optional[Tuple[int,Tuple[str,str]]] = None,
                         eps: float = 1e-9) -> bool:
    """
    Vrai si l'arête 'edge_a' coupe une des arêtes des triangles du groupe.
    - group_tris: liste de dicts {'idx': int, 'P': pts monde}
    - exclude: (idx, ("A","B")) pour ignorer une arête précise du groupe.
    """
    A1, A2 = edge_a
    for tri in group_tris:
        tid = tri["idx"]; P = tri["P"]
        for (B1, B2, keys) in _segments_from_pts(P):
            if exclude and exclude[0] == tid and exclude[1] == keys:
                continue
            if _seg_intersect_strict(A1, A2, B1, B2, eps=eps):
                return True
    return False

# --- Égalité géométrique (avec tolérance) ---
def _same_point(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> bool:
    return np.allclose(p, q, atol=eps, rtol=0.0)

def _point_on_segment(C: np.ndarray, A: np.ndarray, B: np.ndarray, eps: float = 1e-6) -> bool:
    """C est-il sur le segment [A,B] (colinéarité + dans l'intervalle) ?"""
    AB = B - A; AC = C - A
    cross = abs(AB[0]*AC[1] - AB[1]*AC[0])
    if cross > eps: return False
    dot = (AC[0]*AB[0] + AC[1]*AB[1]); ab2 = (AB[0]*AB[0] + AB[1]*AB[1])
    return -eps <= dot <= ab2 + eps

def _same_segment(a1: np.ndarray, a2: np.ndarray,
                  b1: np.ndarray, b2: np.ndarray, eps: float = 1e-6) -> bool:
    """Vrai si (a1,a2) et (b1,b2) représentent la même arête (orientation ignorée)."""
    return (_same_point(a1, b1, eps) and _same_point(a2, b2, eps)) or \
           (_same_point(a1, b2, eps) and _same_point(a2, b1, eps))

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

def _mobile_shape_after_pose(mob_idx: int, vkey_m: str, mob_neighbor_key: str,
                             tgt_idx: int, tgt_vkey: str, tgt_other_key: str,
                             viewer):
    """
    Construit la SHAPE (Polygon/MultiPolygon) du *mobile* après pose :
    - si le mobile n’a pas de group_id → triangle seul posé,
    - sinon → union des triangles du groupe, tous posés avec la même (R,T).
    """
    tri_m = viewer._last_drawn[mob_idx]
    tri_t = viewer._last_drawn[tgt_idx]
    Pm, Pt = tri_m["pts"], tri_t["pts"]

    # 1) paramètres de pose (mêmes pour tous les triangles du groupe mobile)
    R, T, pivot = _pose_params(Pm, vkey_m, mob_neighbor_key, Pm[vkey_m],
                               Pt, tgt_vkey, tgt_other_key, Pt[tgt_vkey])

    def _pose_pts(P):
        out = {}
        for k in ("O","B","L"):
            v = np.array(P[k], float)
            out[k] = (R @ (v - pivot)) + pivot + T
        return out

    gid_m = tri_m.get("group_id")
    if gid_m is None:
        # triangle seul
        Pm_pose = _pose_pts(Pm)
        return _tri_shape(Pm_pose)
    else:
        # groupe mobile → union des triangles posés
        polys = []
        for nd in viewer._group_nodes(gid_m):
            tid = nd.get("tid")
            if tid is None or not (0 <= tid < len(viewer._last_drawn)):
                continue
            P = viewer._last_drawn[tid]["pts"]
            P_pose = _pose_pts(P)
            polys.append(_tri_shape(P_pose))
        if not polys:
            return None
        return unary_union([p.buffer(0) for p in polys])

def _simulate_pose(Pm: Dict[str,np.ndarray], am: str, bm: str, Vm: np.ndarray,
                   Pt: Dict[str,np.ndarray], at: str, bt: str, Vt: np.ndarray) -> Dict[str,np.ndarray]:
    """
    Calcule le triangle mobile posé lorsque (am->bm) est alignée sur (at->bt)
    et le sommet 'Vm' est soudé sur 'Vt'.
    """
    vm_dir = np.array(Pm[bm], dtype=float) - np.array(Pm[am], dtype=float)
    vt_dir = np.array(Pt[bt], dtype=float) - np.array(Pt[at], dtype=float)
    ang_m = math.atan2(vm_dir[1], vm_dir[0])
    ang_t = math.atan2(vt_dir[1], vt_dir[0])
    dtheta = ang_t - ang_m
    c, s = math.cos(dtheta), math.sin(dtheta)
    R = np.array([[c, -s],[s, c]], dtype=float)
    # rotation autour du sommet mobile cliqué, puis translation pour coller sur le sommet cible
    Pm_rot = _apply_R_T_on_P(Pm, R, T=np.zeros(2), pivot=np.array(Vm, dtype=float))
    Vm_rot = np.array(Pm_rot[am], dtype=float)  # 'am' après rotation
    T = np.array(Vt, dtype=float) - Vm_rot
    return _apply_R_T_on_P(Pm_rot, np.eye(2), T, pivot=np.array(Vm_rot, dtype=float))

def _tri_overlaps_group(Ptri: Dict[str,np.ndarray],
                        group_tris: List[Dict[str, np.ndarray]],
                        exclude_edge: Optional[Tuple[int,Tuple[str,str]]] = None,
                        shared_vertex: Optional[np.ndarray] = None,
                        allowed_shared_segment: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        # NEW: arête mobile utilisée pour le collage (à ignorer dans les tests)
                        exclude_m_edge: Optional[Tuple[str,str]] = None,
                        debug_label: Optional[str] = None) -> bool:
    """
    Vrai s'il y a un chevauchement géométrique entre le triangle 'Ptri'
    et au moins un triangle de 'group_tris' :
      - arêtes qui se coupent (croisement ou recouvrement colinéaire),
      - OU inclusion stricte (un triangle entièrement dans l'autre).
    - exclude_edge : (tid, (A,B)) pour ignorer l'arête partagée.
    - shared_vertex : sommet commun à ignorer pour un simple contact.
    """
    # 1) test intersections d'arêtes
    # Log d'en-tête pour cette évaluation
    if debug_label:
        try:
            print(f"[DBG] Test de chevauchement ({debug_label})")
        except Exception:
            pass

    # petit helper pour libeller les arêtes "O-B", "B-L", "L-O"
    def _lab(keys): 
        try:    return f"{keys[0]}-{keys[1]}"
        except: return "?"

    # entête : rappeler l’arête de collage autorisée (si connue)
    if debug_label:
        try:
            if exclude_edge:
                ex_tid, ex_keys = exclude_edge
                print(f"[DBG]   arête de collage cible: t#{ex_tid} {_lab(ex_keys)}")
        except Exception:
            pass

    for (A1, A2, k_m) in _segments_from_pts(Ptri):
        # NEW: ne pas tester l’arête mobile qui sert au collage
        if exclude_m_edge is not None and k_m == exclude_m_edge:
            continue
        for tri in group_tris:
            tid = tri["idx"]; P = tri["P"]
            for (B1, B2, keys) in _segments_from_pts(P):
                if exclude_edge and exclude_edge[0] == tid and exclude_edge[1] == keys:
                    continue
                if shared_vertex is not None:
                    if (np.allclose(A1, shared_vertex) and (np.allclose(B1, shared_vertex) or np.allclose(B2, shared_vertex))) or \
                       (np.allclose(A2, shared_vertex) and (np.allclose(B1, shared_vertex) or np.allclose(B2, shared_vertex))):
                        # simple contact au sommet commun -> on l'ignore
                        continue
                # --- test avec Shapely, en tolérant l'arête de collage ---
                Lm = _ShLine([tuple(map(float, A1)), tuple(map(float, A2))])
                Lg = _ShLine([tuple(map(float, B1)), tuple(map(float, B2))])
                # ignorer un simple contact au sommet commun (robuste)
                if shared_vertex is not None:
                    sv = _ShPoint(tuple(map(float, shared_vertex)))
                    inter = Lm.intersection(Lg)
                    # on tolère si l'intersection est exactement le sommet partagé
                    if inter.geom_type == "Point":
                        # accepte si l'un OU l'autre segment passe par sv
                        if (Lm.distance(sv) < 1e-9 or Lg.distance(sv) < 1e-9):
                            # (Shapely peut ne pas faire equals(sv) à 1e-16 près, on tolère)
                            if inter.distance(sv) < 1e-9:
                                continue
                # croisement propre -> interdit
                if Lm.crosses(Lg):
                    if debug_label:
                        try:
                            print(f"[DBG]  -> CROISEMENT (X)  mobile[{_lab(k_m)}]  ×  groupe[t#{tid} {_lab(keys)}]")
                            # optionnel: suffixe court avec coords
                            print(f"[DBG]     mob {tuple(map(lambda p:(round(p[0],2),round(p[1],2)), Lm.coords))}  /  grp {tuple(map(lambda p:(round(p[0],2),round(p[1],2)), Lg.coords))}")
                        except Exception:
                            pass
                    return True
                # recouvrement colinéaire ?
                if Lm.overlaps(Lg):
                    if allowed_shared_segment is not None:
                        SA = tuple(map(float, allowed_shared_segment[0]))
                        SB = tuple(map(float, allowed_shared_segment[1]))
                        L_ok = _ShLine((SA, SB))
                        inter = Lm.intersection(Lg)
                        # On autorise si le recouvrement est contenu (à eps près) dans l'arête de collage
                        inter_snapped = _sh_snap(inter, L_ok, 1e-9)
                        if inter_snapped.within(L_ok) and inter.length > 0.0:
                            # Snap pour absorber de petites erreurs numériques
                            if debug_label:
                                try:
                                    print(f"[DBG]  -> RECOUVREMENT colinéaire **AUTORISÉ** mobile[{_lab(k_m)}] ~ groupe[t#{tid} {_lab(keys)}] (sur l'arête de collage)")
                                except Exception:
                                    pass
                            continue
                    # sinon, recouvrement ailleurs -> interdit
                    if debug_label:
                        try:
                            ln = float(getattr(Lm.intersection(Lg), "length", 0.0)) if hasattr(Lm.intersection(Lg),'length') else 0.0
                            print(f"[DBG]  -> RECOUVREMENT COLINÉAIRE (≈{ln:.6f}) mobile[{_lab(k_m)}]  //  groupe[t#{tid} {_lab(keys)}]  **INTERDIT**")
                        except Exception:
                            pass
                    return True
    # 2) test inclusion stricte (sans croisement)
    #    (a) un sommet de Ptri à l'intérieur d'un triangle du groupe
    A, B, C = np.array(Ptri["O"],float), np.array(Ptri["B"],float), np.array(Ptri["L"],float)
    for tri in group_tris:
        P = tri["P"]
        tA, tB, tC = np.array(P["O"],float), np.array(P["B"],float), np.array(P["L"],float)
        for V in (A, B, C):
            if _point_in_triangle_strict(V, tA, tB, tC):
                if debug_label:
                    try:
                        print(f"[DBG]  -> INCLUSION stricte: sommet mobile {tuple(map(float,V))} à l'intérieur du triangle cible #{tri['idx']}")
                    except Exception:
                        pass                
                return True
        #    (b) un sommet du groupe à l'intérieur de Ptri
        for V in (tA, tB, tC):
            if _point_in_triangle_strict(V, A, B, C):
                if debug_label:
                    try:
                        print(f"[DBG]  -> INCLUSION stricte: sommet du groupe {tuple(map(float,V))} à l'intérieur du triangle mobile posé")
                    except Exception:
                        pass                
                return True
    if debug_label:
        try:
            print(f"[DBG]  -> pas de chevauchement")
        except Exception:
            pass
    return False


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
        self.zoom = 1.0
        self.offset = np.array([400.0, 350.0], dtype=float)
        self._drag = None             # état de drag & drop depuis la liste
        self._drag_preview_id = None  # id du polygone "fantôme" sur le canvas
        self._sel = None              # sélection sur canvas: {'mode': 'move'|'vertex', 'idx': int}
        self._hit_px = 12             # tolérance de hit (pixels) pour les sommets
        self._marker_px = 6           # rayon des marqueurs (cercles) dessinés aux sommets
        self._ctx_target_idx = None   # index du triangle visé par clic droit (menu contextuel)
        self._placed_ids = set()      # ids déjà posés (retirés de la liste)
        self._ctx_last_rclick = None  # dernière position écran du clic droit (pour pivoter)
        self._nearest_line_id = None  # trait d'aide "sommet le plus proche"
        self._edge_highlight_ids = [] # surlignage des 2 arêtes (mobile/cible)
        self._edge_choice = None      # (i_mob, key_mob, edge_m, i_tgt, key_tgt, edge_t)
        self._edge_highlights = None  # données brutes des aides (candidates + best)

        # mode "déconnexion" activé par CTRL
        self._ctrl_down = False        

        # === Groupes (Étape A) ===
        self.groups: Dict[int, Dict] = {}
        self._next_group_id: int = 1

        # état IHM
        self.triangle_file = tk.StringVar(value="")
        self.start_index = tk.IntVar(value=1)
        self.num_triangles = tk.IntVar(value=8)

        self.excel_path = None
        self.df = None
        self._last_drawn = []   # liste d'items: (labels, P_world)

        self._build_ui()
        # Bind pour annuler avec ESC (drag ou sélection)
        self.bind("<Escape>", self._on_escape_key)

        # auto-load si présent (facultatif)
        default = "../data/triangle.xlsx"
        if os.path.exists(default):
            self.load_excel(default)

    # ---------- UI ----------
    def _build_ui(self):
        top = tk.Frame(self); top.pack(side=tk.TOP, fill=tk.X)
        tk.Button(top, text="Ouvrir Excel...", command=self.open_excel).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Label(top, textvariable=self.triangle_file).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Imprimer…", command=self.print_triangles_dialog).pack(side=tk.RIGHT, padx=5, pady=5)

        main = tk.Frame(self); main.pack(fill=tk.BOTH, expand=True)
        self._build_left_pane(main)
        self._build_canvas(main)

        self.status = tk.Label(self, text="Prêt", bd=1, relief=tk.SUNKEN, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ---------- DEBUG ----------
    def _debug(self, *args):
        try:
            print(*args, flush=True)
        except Exception:
            pass

    def _debug_dump_group(self, gid: int, title: str = ""):
        g = self.groups.get(gid)
        if not g:
            self._debug("[DBG] group", gid, "inexistant.")
            return
        if title:
            self._debug(f"[DBG] {title}")
        self._debug(f"[DBG]  gid={gid}, nodes={len(g['nodes'])}")
        for i, nd in enumerate(g["nodes"]):
            tid = nd.get("tid")
            vin = nd.get("vkey_in"); vout = nd.get("vkey_out")
            lab = None
            try:
                lab = self._last_drawn[tid].get("id")
            except Exception:
                pass
            self._debug(f"       [{i}] tid={tid} (id={lab})  vkey_in={vin}  vkey_out={vout}")
        # mini vérif géométrique du sommet commun pour chaque paire consécutive
        eps = 1e-6
        for i in range(max(0, len(g["nodes"]) - 1)):
            a = g["nodes"][i]; b = g["nodes"][i+1]
            Pa = self._last_drawn[a["tid"]]["pts"]
            Pb = self._last_drawn[b["tid"]]["pts"]
            found = None
            for ka in ("O","B","L"):
                for kb in ("O","B","L"):
                    va = Pa[ka]; vb = Pb[kb]
                    if (abs(float(va[0]-vb[0])) <= eps) and (abs(float(va[1]-vb[1])) <= eps):
                        found = (ka, kb); break
                if found: break
            self._debug(f"       géométrie[{i}->{i+1}] sommet commun = {found}")

    def _build_left_pane(self, parent):
        left = tk.Frame(parent, width=260); left.pack(side=tk.LEFT, fill=tk.Y); left.pack_propagate(False)

        tk.Label(left, text="Triangles (ordre)").pack(anchor="w", padx=6, pady=(6,0))
        # Frame pour Listbox + Scrollbar verticale
        lb_frame = tk.Frame(left)
        lb_frame.pack(fill=tk.X, padx=6)
        self.listbox = tk.Listbox(lb_frame, width=34, height=16, exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_scroll = tk.Scrollbar(lb_frame, orient="vertical", command=self.listbox.yview)
        lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.configure(yscrollcommand=lb_scroll.set)
        # Démarrer le drag dès qu'on clique sur un item
        self.listbox.bind("<ButtonPress-1>", self._on_list_mouse_down)

        opts = tk.LabelFrame(left, text="Affichage"); opts.pack(fill=tk.X, padx=6, pady=6)
        row = tk.Frame(opts); row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(row, text="Début").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=self.start_index, width=6).pack(side=tk.LEFT, padx=(4,10))
        tk.Label(row, text="Nombre N").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=self.num_triangles, width=6).pack(side=tk.LEFT, padx=4)

        btns = tk.Frame(left); btns.pack(fill=tk.X, padx=6, pady=(0,6))
        tk.Button(btns, text="Afficher (brut)", command=self.show_raw_selection).pack(side=tk.LEFT)
        tk.Button(btns, text="Fit à l'écran", command=lambda: self._fit_to_view(self._last_drawn)).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Effacer", command=self.clear_canvas).pack(side=tk.LEFT)

    def _build_canvas(self, parent):
        self.canvas = tk.Canvas(parent, bg="white")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Mouse wheel zoom (Windows/macOS)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        # Mouse wheel zoom (Linux)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)

        # Pan avec clic milieu (garde le clic gauche pour sélectionner/déplacer)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        # Sélection/déplacement au clic gauche
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_left_down)
        self.canvas.bind("<B1-Motion>", self._on_canvas_left_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_left_up)
        # — activer le mode "déconnexion" quand CTRL est enfoncé (et MAJ curseur)
        self.bind_all("<KeyPress-Control_L>", self._on_ctrl_down)
        self.bind_all("<KeyPress-Control_R>", self._on_ctrl_down)
        self.bind_all("<KeyRelease-Control_L>", self._on_ctrl_up)
        self.bind_all("<KeyRelease-Control_R>", self._on_ctrl_up)
        # Suivi souris : drag fantôme OU rotation
        self.canvas.bind("<Motion>", self._on_canvas_motion_update_drag)
        # Clic droit : menu contextuel
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)

        # Menu contextuel
        self._ctx_menu = tk.Menu(self, tearoff=0)
        self._ctx_menu.add_command(label="Supprimer", command=self._ctx_delete_selected)
        self._ctx_menu.add_command(label="Pivoter", command=self._ctx_rotate_selected)
        self._ctx_menu.add_command(label="Inverser", command=self._ctx_flip_selected)
        self._ctx_menu.add_command(label="OL=0°", command=self._ctx_orient_OL_north)

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
        }).dropna(subset=["len_OB","len_OL","len_BL"]).sort_values("id")
        return out.reset_index(drop=True)

    def load_excel(self, path: str):
        df0 = pd.read_excel(path, header=None)
        header_row = self._find_header_row(df0)
        df = pd.read_excel(path, header=header_row)
        self.df = self._build_df(df)
        self.excel_path = path
        self.triangle_file.set(os.path.basename(path))

        self.listbox.delete(0, tk.END)
        for _, r in self.df.iterrows():
            self.listbox.insert(tk.END, f"{int(r['id']):02d}. B:{r['B']}  L:{r['L']}")
        self.status.config(text=f"{len(self.df)} triangles chargés depuis {path}")
        self._last_drawn = []

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
            out.append({
                "labels": ( "Bourges", str(r["B"]), str(r["L"]) ),  # O,B,L
                "pts": P,
                "id": int(r["id"]),
                "mirrored": False,
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
        return {
            "labels": ("Bourges", str(r["B"]), str(r["L"])),
            "pts": P,
            "id": tri_id,
            "mirrored": False,   # <— toujours présent dès la création
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

    # ---------- Dessin / vue ----------
    def show_raw_selection(self):
        try:
            start = max(0, int(self.start_index.get()) - 1)
            n = max(1, int(self.num_triangles.get()))
            tris = self._triangles_local(start, n)
        except Exception as e:
            messagebox.showerror("Afficher", str(e))
            return

        # Mise en page simple pour aperçu
        placed = self._layout_tris_horizontal(tris, gap=0.8, wrap_width=30.0)

        # Dessin
        self.canvas.delete("all")
        self._last_drawn = placed
        for idx, t in enumerate(placed):
            labels = t["labels"]
            P = t["pts"]
            tri_id = t.get("id", int(self.df.iloc[start + idx]["id"]))
            self._draw_triangle_screen(
                P,
                labels=[f"O:{labels[0]}", f"B:{labels[1]}", f"L:{labels[2]}"],
                tri_id=tri_id,
                tri_mirrored=t.get("mirrored", False),
            )

        # Fit à l'écran
        self._fit_to_view(placed)
        self.status.config(text=f"{len(placed)} triangle(s) affiché(s) — aperçu brut (sans assemblage)")

    def clear_canvas(self):
        self.canvas.delete("all")
        self._last_drawn = []
        # aussi remettre à zéro l'aide "sommet le plus proche"
        self._nearest_line_id = None
        self._clear_edge_highlights()
        # Par défaut : aucun choix mémorisé / pas d'aides persistantes
        self._edge_choice = None
        self._edge_highlights = None     
        self.status.config(text="Canvas effacé")

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

        # 2) tracé du triangle
        self.canvas.create_polygon(*coords, outline=outline, fill="", width=width)

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
            self.canvas.create_text(
                sx, sy, text=num_txt,
                anchor="center", font=("Arial", 10, "bold"),
                fill="red", tags="tri_num"
            )
            self.canvas.tag_raise("tri_num")

    # --- helpers: mode déconnexion (CTRL) + curseur ---
    def _on_ctrl_down(self, event=None):
        if not self._ctrl_down:
            self._ctrl_down = True
            try:
                self.canvas.configure(cursor="X_cursor")
            except tk.TclError:
                self.canvas.configure(cursor="crosshair")

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
        # Détermine l'index sous la souris et démarre un drag "virtuel"
        i = self.listbox.nearest(event.y)
        if i < 0:
            return
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(i)
        try:
            tri = self._triangle_from_index(i)
        except Exception as e:
            self.status.config(text=f"Erreur: {e}")
            return
        # on mémorise aussi l'index dans la listbox pour retirer l'entrée au dépôt
        self._drag = {"triangle": tri, "list_index": i}
        # créer un fantôme minimal dès le départ (sera positionné au mouvement)
        if self._drag_preview_id is not None:
            self.canvas.delete(self._drag_preview_id)
            self._drag_preview_id = None
        # Forcer focus sur le canvas pour recevoir les motions
        self.canvas.focus_set()
        self.status.config(text="Glissez le triangle sur le canvas puis relâchez pour le déposer.")

    def _on_canvas_motion_update_drag(self, event):
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
    def _group_outline_segments(self, gid: int, eps: float = 1e-6):
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
        def _push(P):
            for a, b in ((P["O"], P["B"]), (P["B"], P["L"]), (P["L"], P["O"])):
                for s in _split_by_vertices(a, b):
                    m = (s[0] + s[1]) * 0.5
                    if boundary.distance(_ShPoint(m)) <= eps:   # ← le filtre clé
                        outline.append(s)

        for nd in nodes:
            tid = nd.get("tid")
            if 0 <= tid < len(self._last_drawn):
                _push(self._last_drawn[tid]["pts"])

        return outline


    # --- helper : triangle propriétaire d’un sous-segment le long d’une arête d’un groupe ---
    def _owner_triangle_for_segment_at_vertex(self, i_tgt: int, key_tgt: str,
                                              seg_world: Tuple[np.ndarray, np.ndarray],
                                              eps: float = 1e-6) -> Tuple[int, str, Optional[str]]:
        """
        Quand la cible est un GROUPE et qu’on a choisi un SOUS-SEGMENT 'virtuel'
        sur l’enveloppe, il faut lier le mobile à un TRIANGLE précis.
        Retourne un triplet (owner_tid, at_owner, bt_owner) où :
          - owner_tid : index du triangle du groupe qui porte ‘seg_world’ au voisinage de vt,
          - at_owner  : clé O/B/L du **sommet de ce triangle** qui coïncide géométriquement avec vt,
          - bt_owner  : la clé voisine telle que le sous-segment suive [vt → voisin].
        Fallback : (i_tgt, at_key=clé trouvée sur i_tgt, bt_key meilleur voisin par direction).
        """
        A, B = seg_world
        # triangles qui partagent le sommet (dans le même groupe que i_tgt)
        gid = self._last_drawn[i_tgt].get("group_id")
        if gid is None:
            # cible = triangle seul
            P = self._last_drawn[i_tgt]["pts"]
            # at_owner: trouver la clé qui coïncide avec vt
            vt = np.array(self._last_drawn[i_tgt]["pts"][key_tgt], float)
            at_owner = None
            for k in ("O","B","L"):
                if _same_point(np.array(P[k], float), vt, eps):
                    at_owner = k; break
            # bt_owner : voisin de at_owner le plus aligné avec le sous-segment
            bt_owner = None
            if at_owner is not None:
                best = None
                az_seg = math.atan2(*(B - A)[::-1])  # atan2(dy, dx)
                for nb in {"O":("B","L"), "B":("O","L"), "L":("O","B")}[at_owner]:
                    az_nb = math.atan2(*(np.array(P[nb],float) - vt)[::-1])
                    d = abs(((az_nb - az_seg + math.pi) % (2*math.pi)) - math.pi)
                    if (best is None) or (d < best[0]):
                        best = (d, nb)
                bt_owner = best[1] if best else None
            return i_tgt, at_owner or key_tgt, bt_owner
        g = self.groups.get(gid) or {}
        nodes = g.get("nodes", [])
        # vt = point géométrique du sommet cible
        vt = np.array(self._last_drawn[i_tgt]["pts"][key_tgt], float)
        # on cherche un triangle dont UN DE SES SOMMETS coïncide avec vt (peu importe son nom),
        # puis une arête incidente à ce sommet qui supporte (au mieux) le sous-segment [A,B].
        best_owner = None  # (score, tid, at_key, bt_key)
        for nd in nodes:
            tid = nd.get("tid")
            if tid is None or not (0 <= tid < len(self._last_drawn)):
                continue
            P = self._last_drawn[tid]["pts"]
            # trouver la clé du sommet de CE triangle qui coïncide avec vt
            at_key = None
            for k in ("O","B","L"):
                if _same_point(np.array(P[k],float), vt, eps):
                    at_key = k; break
            if at_key is None:
                continue
            # évaluer les deux arêtes incidentes (at_key->voisin)
            for nb in {"O":("B","L"), "B":("O","L"), "L":("O","B")}[at_key]:
                S1 = np.array(P[at_key], float); S2 = np.array(P[nb], float)
                # 1) si (B) est sur [S1,S2] (ou proche) — test principal
                on_seg = _point_on_segment(B, S1, S2, eps) or _point_on_segment(A, S1, S2, eps)
                # 2) score angulaire (plus c’est proche de la direction [A->B], mieux c’est)
                vseg = B - A; vtri = S2 - S1
                az_seg = math.atan2(vseg[1], vseg[0]); az_tri = math.atan2(vtri[1], vtri[0])
                d = abs(((az_tri - az_seg + math.pi) % (2*math.pi)) - math.pi)
                score = (0.0 if on_seg else 1.0) + d  # favorise on_seg, puis petite diff d’angle
                if (best_owner is None) or (score < best_owner[0]):
                    best_owner = (score, tid, at_key, nb)
        if best_owner is not None:
            _, tid, at_key, nb = best_owner
            return tid, at_key, nb
        # fallback : rester sur i_tgt avec appariement directionnel
        P = self._last_drawn[i_tgt]["pts"]
        at_owner = None
        for k in ("O","B","L"):
            if _same_point(np.array(P[k],float), vt, eps):
                at_owner = k; break
        bt_owner = None
        if at_owner is not None:
            az_seg = math.atan2((B-A)[1], (B-A)[0])
            best = None
            for nb in {"O":("B","L"), "B":("O","L"), "L":("O","B")}[at_owner]:
                az_nb = math.atan2((np.array(P[nb],float)-vt)[1], (np.array(P[nb],float)-vt)[0])
                d = abs(((az_nb - az_seg + math.pi) % (2*math.pi)) - math.pi)
                if (best is None) or (d < best[0]):
                    best = (d, nb)
            bt_owner = best[1] if best else None
        return i_tgt, (at_owner or key_tgt), bt_owner

    def _update_edge_highlights(self, mob_idx: int, vkey_m: str, tgt_idx: int, tgt_vkey: str):
        """
        Met à jour l'aide visuelle :
        - calcule toutes les arêtes *extérieures* et/ou *découpées* du groupe cible,
            incidentes au sommet cible (tgt_idx, tgt_vkey),
        - choisit la meilleure par *delta d'azimut* vis-à-vis de l’arête du triangle mobile
            passant par (mob_idx, vkey_m),
        - stocke le choix dans self._edge_choice et les segments gris/orange
            dans self._edge_highlights pour le rendu.
        Règle : seule l’orientation (azimut) compte. La longueur n’entre jamais dans le score.
        """
        import numpy as np
        from math import atan2, pi

        def _to_np(p): 
            return np.array([float(p[0]), float(p[1])], dtype=float)

        def _azim(a, b):
            v = b - a
            return atan2(v[1], v[0])

        def _ang_dist(a1, a2):
            d = (a1 - a2 + pi) % (2 * pi) - pi
            return abs(d)

        def _almost_eq(a, b, eps=1e-6):
            return abs(a[0]-b[0]) <= eps and abs(a[1]-b[1]) <= eps

        # --- Récupère le(s) sommets et arêtes du triangle mobile au niveau de vkey_m
        tri_m = self._last_drawn[mob_idx]
        Pm = tri_m["pts"]  # dict {"O":(x,y), "B":(x,y), "L":(x,y)}
        vm = _to_np(Pm[vkey_m])

        # Les deux arêtes incidentes au sommet mobile
        neighs_m = {"O":("B","L"), "B":("O","L"), "L":("O","B")}[vkey_m]
        m_edges = []
        for nk in neighs_m:
            m_edges.append((vm, _to_np(Pm[nk])))


        # Sommet cible (point attaché)
        tri_t = self._last_drawn[tgt_idx]
        Pt = tri_t["pts"]
        vt = _to_np(Pt[tgt_vkey])

        # --- Collecte des segments candidats autour du sommet cible (UNIQUEMENT l’enveloppe) ---
        # Cas 1 : la cible est un triangle simple → on prend ses 2 arêtes incidentes au sommet.
        # Cas 2 : la cible est un groupe → on prend les sous-segments EXTERIEURS retournés
        #         par group_outline_segments(gid) et on ne garde que ceux incident au sommet cible.
        cand_edges = []
        tgt_gid = tri_t.get("group_id")
        if tgt_gid is None:
            neighs_t = {"O":("B","L"), "B":("O","L"), "L":("O","B")}[tgt_vkey]
            for nk in neighs_t:
                cand_edges.append((_to_np(Pt[tgt_vkey]), _to_np(Pt[nk])))
        else:
            try:
                outline = self._group_outline_segments(tgt_gid, eps=1e-6)
            except Exception:
                outline = []
            # ne retenir que les sous-segments de l’enveloppe incident au point cible
            for (Q0, Q1) in outline:
                A = _to_np(Q0); B = _to_np(Q1)
                if _almost_eq(A, vt) or _almost_eq(B, vt):
                    # on mémorise brut, l’orientation sera forcée plus bas (vt -> autre)
                    cand_edges.append((A, B))
 
        # Si rien d’incident (rare), on tente au moins les 2 arêtes du triangle cible autour de tgt_vkey
        if not cand_edges:
            neighs_t = {"O":("B","L"), "B":("O","L"), "L":("O","B")}[tgt_vkey]
            for nk in neighs_t:
                cand_edges.append((_to_np(Pt[tgt_vkey]), _to_np(Pt[nk])))

        # --- Réduction pour les GROUPES : garder AU PLUS 2 directions autour de vt ---
        if tri_t.get("group_id") is not None and len(cand_edges) > 2:
            # 1) forcer l’orientation vt -> autre et calculer l’azimut + longueur
            oriented = []
            for (A, B) in cand_edges:
                if _almost_eq(A, vt):
                    ta, tb = A, B
                elif _almost_eq(B, vt):
                    ta, tb = B, A
                else:
                    # sécurité: ignorer un candidat qui ne touche pas vt (ne devrait pas arriver)
                    continue
                az = _azim(ta, tb)
                L  = float(np.hypot(*(tb - ta)))
                oriented.append((ta, tb, az, L))

            # 2) regrouper par azimut (tolérance 1e-3 rad) et garder le plus long par groupe
            ang_tol = 1e-3
            buckets = {}  # key=int -> (ta,tb,az,L)
            for (ta, tb, az, L) in oriented:
                key = int(round(az / ang_tol))
                cur = buckets.get(key)
                if (cur is None) or (L > cur[3]):
                    buckets[key] = (ta, tb, az, L)

            # 3) si >2 groupes (rare), garder les deux plus longs
            reps = list(buckets.values())
            reps.sort(key=lambda x: x[3], reverse=True)  # par longueur décroissante
            reps = reps[:2]

            # 4) reconstruire cand_edges (toujours orienté vt -> autre)
            cand_edges = [(ta, tb) for (ta, tb, _, _) in reps]


        # --- Choix par delta d’azimut UNIQUEMENT + filtre anti-chevauchement
        # Prépare les triangles du groupe cible pour le test d'intersections
        tgt_gid = tri_t.get("group_id")
        if tgt_gid is None:
            group_tris = [{"idx": tgt_idx, "P": Pt}]
        else:
            group_tris = []
            for node in self._group_nodes(tgt_gid):
                tid = node["tid"]
                if 0 <= tid < len(self._last_drawn):
                    group_tris.append({"idx": tid, "P": self._last_drawn[tid]["pts"]})

        best = None   # (score, m_edge, t_edge)
        # pour retrouver la clé voisine côté mobile à partir de la position m_b
        def _neighbor_key_for_point(P, vkey, point):
            for nk in {"O":("B","L"), "B":("O","L"), "L":("O","B")}[vkey]:
                if _almost_eq(_to_np(P[nk]), point):
                    return nk
            return None

        for (m_a, m_b) in m_edges:
            az_m = _azim(m_a, m_b)
            mob_neighbor_key = _neighbor_key_for_point(Pm, vkey_m, m_b)
            for (t_a, t_b) in cand_edges:
                # oriente toujours l’arête cible depuis vt vers l’autre extrémité
                if _almost_eq(t_a, vt):
                    t_edge = (t_a, t_b)
                    t_other_point = t_b
                else:
                    t_edge = (t_b, t_a)
                    t_other_point = t_a
                az_t = _azim(t_edge[0], t_edge[1])
                score = _ang_dist(az_m, az_t)     # comparaison modulo 2π (pas d’équivalence à +π)

                # --- Filtre anti-chevauchement (formes solides) ---
                # Sous-segment t_edge → triangle propriétaire + clés (at_owner, bt_owner)
                owner_tid, at_owner, bt_owner = self._owner_triangle_for_segment_at_vertex(
                    tgt_idx, tgt_vkey, t_edge, eps=1e-6
                )

                # shape CIBLE (intérieur strict) : triangle seul ou union du groupe
                if tgt_gid is None:
                    S_t = _tri_shape(Pt)
                else:
                    S_t = _group_shape_from_nodes(self._group_nodes(tgt_gid), self._last_drawn)

                # shape MOBILE après pose, en se basant sur le triangle propriétaire (owner_tid)
                # et la paire (at_owner, bt_owner) cohérente avec t_edge
                S_m = _mobile_shape_after_pose(
                        mob_idx, vkey_m, mob_neighbor_key,
                        owner_tid, at_owner, bt_owner, self)

                # 3) tolérer l’arête de collage en perçant **exactement** le sous-segment choisi (t_edge)
                eps_shape = 1e-6
                L_ok = _ShLine([
                    (float(t_edge[0][0]), float(t_edge[0][1])),
                    (float(t_edge[1][0]), float(t_edge[1][1]))
                ]).buffer(eps_shape)
                S_t_cut = S_t.difference(L_ok)
                S_m_cut = S_m.difference(L_ok)
                # 4) intersection d’intérieurs → on rejette ce candidat
                if S_m_cut.intersects(S_t_cut):
                    continue

                # --- Mémorise le meilleur score restant ---
                if (best is None) or (score < best[0]):
                    best = (score, (m_a, m_b), t_edge)

        # Aide visuelle : segments gris (toutes les candidates) + orange (choix)
        self._edge_highlights = {
            "all": [(tuple(a), tuple(b)) for (a, b) in cand_edges],
            "best": (tuple(best[1][0]), tuple(best[1][1]), tuple(best[2][0]), tuple(best[2][1])) if best else None,
        }
        # Mémorise le choix pour l’étape "mouse up"
        if best:
            # self._edge_choice = (idx_m, vkey_m, idx_t, vkey_t, (m_a, m_b, t_a, t_b))
            (m_a, m_b), (t_a, t_b) = best[1], best[2]
            self._edge_choice = (mob_idx, vkey_m, tgt_idx, tgt_vkey, (tuple(m_a), tuple(m_b), tuple(t_a), tuple(t_b)))
        else:
            self._edge_choice = None

        # Dessine/MAJ les aides sans supprimer les triangles
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
        - candidates en GRIS,
        - meilleure arête en ORANGE côté CIBLE **et** côté MOBILE (manquait auparavant)."""
        self._clear_edge_highlights()
        data = getattr(self, "_edge_highlights", None)
        if not data:
            return
        # Candidates (gris)
        for (a, b) in data.get("all", []):
            _id = self._draw_temp_edge_world(np.array(a, float), np.array(b, float), color="#BBBBBB", width=1)
            if _id:
                self._edge_highlight_ids.append(_id)
        # Meilleure (orange, cible + mobile)
        best = data.get("best")
        if best:
            # best = (m_a, m_b, t_a, t_b)
            m_a, m_b, t_a, t_b = best
            # côté cible (épais)
            _id = self._draw_temp_edge_world(np.array(t_a, float), np.array(t_b, float), color="#FF9900", width=3)
            if _id:
                self._edge_highlight_ids.append(_id)
            # côté mobile (fin) — ajout pour visualiser l'arête du groupe d'origine
            _id2 = self._draw_temp_edge_world(np.array(m_a, float), np.array(m_b, float), color="#FF9900", width=2)
            if _id2:
                self._edge_highlight_ids.append(_id2)

    def _place_dragged_triangle(self):
        if not self._drag or "world_pts" not in self._drag:
            return
        tri = self._drag["triangle"]
        Pw = self._drag["world_pts"]
        # Ajouter au "document courant"
        self._last_drawn.append({
            "labels": tri["labels"],
            "pts": Pw,
            "id": tri.get("id"),
            "mirrored": tri.get("mirrored", False)  # <— on conserve l’état
        })
        # ensure group fields
        try:
            self._ensure_group_fields(self._last_drawn[-1])
        except Exception:
            pass

        self._redraw_from(self._last_drawn)
        self.status.config(text="Triangle déposé.")
        # Retirer l'élément correspondant de la liste (s'il vient de la listbox)
        li = self._drag.get("list_index")
        if isinstance(li, int):
            try:
                self.listbox.delete(li)
            except Exception:
                pass
        # Marquer l'id comme posé (évite réinsertion multiple)
        if tri.get("id") is not None:
            self._placed_ids.add(int(tri["id"]))

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

    def _apply_transform_to_group(self, gid: int, T=None, R=None, pivot=None) -> None:
        import numpy as np
        g = self.groups.get(gid)
        if not g: return
        if T is None and R is None: return
        if R is not None and pivot is None:
            pivot = np.zeros(2, dtype=float)
        for node in g["nodes"]:
            tid = node["tid"]
            P = self._last_drawn[tid]["pts"]
            for k in ("O", "B", "L"):
                v = P[k]
                if R is not None:
                    v = (R @ (v - pivot)) + pivot
                if T is not None:
                    v = v + T
                P[k] = v
        self._recompute_group_bbox(gid)

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
        self._debug(f"[DBG] _find_group_link_for_vertex(idx={idx}, vkey={vkey}) -> gid={gid}")
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
        self._debug(f"[DBG]   node pos={pos} vkey_in={nd.get('vkey_in')} vkey_out={nd.get('vkey_out')}")

        if nd.get("vkey_in") == vkey and pos > 0:
            self._debug(f"[DBG]   MATCH: link_type='in' (vers pos {pos-1})")
            return (gid, pos, "in")   # lien avec le précédent
        if nd.get("vkey_out") == vkey and pos < len(nodes)-1:
            self._debug(f"[DBG]   MATCH: link_type='out' (vers pos {pos+1})")
            return (gid, pos, "out")  # lien avec le suivant
        self._debug("[DBG]   NO LINK on this vertex (métadonnées)")
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
        self._debug_dump_group(gid, "après _apply_group_meta_after_split_")

    def _dissolve_singletons_(self, gid: int):
        """
        Si le groupe gid ne contient qu'1 triangle, on supprime le groupe et
        on remet ce triangle à l'état isolé (group_id=None).
        """
        g = self.groups.get(gid)
        if not g:
            return None
        if len(g["nodes"]) == 1:
            tid = g["nodes"][0]["tid"]
            if 0 <= tid < len(self._last_drawn):
                self._last_drawn[tid]["group_id"]  = None
                self._last_drawn[tid]["group_pos"] = None
            del self.groups[gid]
            return None
        return gid

    def _split_group_at(self, gid: int, split_after_pos: int):
        """
        Scinde le groupe 'gid' ENTRE split_after_pos et split_after_pos+1.
        Retourne (left_gid_or_None, right_gid_or_None).
        - Le groupe de gauche réutilise 'gid' si sa taille >=2, sinon dissout.
        - Le groupe de droite reçoit un nouveau gid si sa taille >=2, sinon dissout.
        - Les triangles isolés (taille==1) deviennent group_id=None.
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

        # Appliquer côté gauche (réutilise gid si >=2 sinon dissout)
        left_gid = None
        if len(left_nodes) >= 2:
            self.groups[gid] = {"id": gid, "nodes": left_nodes, "bbox": None}
            self._apply_group_meta_after_split_(gid)
            left_gid = gid
        elif len(left_nodes) == 1:
            tid = left_nodes[0]["tid"]
            if 0 <= tid < len(self._last_drawn):
                self._last_drawn[tid]["group_id"]  = None
                self._last_drawn[tid]["group_pos"] = None
            # on supprimera 'gid' si la droite crée un nouveau

        # Appliquer côté droit (nouveau gid si >=2)
        right_gid = None
        if len(right_nodes) >= 2:
            new_gid = self._new_group_id()
            self.groups[new_gid] = {"id": new_gid, "nodes": right_nodes, "bbox": None}
            self._apply_group_meta_after_split_(new_gid)
            right_gid = new_gid
        elif len(right_nodes) == 1:
            tid = right_nodes[0]["tid"]
            if 0 <= tid < len(self._last_drawn):
                self._last_drawn[tid]["group_id"]  = None
                self._last_drawn[tid]["group_pos"] = None

        # S'il ne reste rien à gauche (singleton), supprimer l'ancien gid
        if left_gid is None and gid in self.groups:
            del self.groups[gid]
        if left_gid:  self._debug_dump_group(left_gid,  "après split: LEFT")
        if right_gid: self._debug_dump_group(right_gid, "après split: RIGHT")
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
        self._debug_dump_group(gid, "GROUPE CRÉÉ (étape 2)")
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
        self._debug_dump_group(gid, "APRES EXTENSION tri→groupe")
        self.status.config(text=f"Triangle ajouté au groupe #{gid}.")

    def record_group_after_last_snap(self, idx_mob: int, key_mob: str,
                                     idx_tgt: int, key_tgt: str) -> None:
        """
        Appelée après un collage effectif par sommet.
        Cas gérés :
          1) tri isolé ↔ tri isolé  -> création d'un groupe à 2.
          2) tri isolé ↔ groupe      -> extension du groupe par une extrémité.
        (Les fusions groupe↔groupe et les insertions au milieu viendront plus tard.)
        """
        # états actuels
        gm = self._get_group_of_triangle(idx_mob)
        gt = self._get_group_of_triangle(idx_tgt)

        # 1) deux triangles isolés -> créer un nouveau groupe
        if gm is None and gt is None:
            self._link_triangles_into_group(idx_mob, key_mob, idx_tgt, key_tgt)
            return

        # 2) tri isolé (mobile) -> triangle d'un groupe (cible) : extension extrémité
        if gm is None and gt is not None:
            nodes = self._group_nodes(gt)
            # trouver la position du triangle cible dans son groupe
            pos_tgt = None
            for i, nd in enumerate(nodes):
                if nd.get("tid") == idx_tgt:
                    pos_tgt = i
                    break
            if pos_tgt is None:
                return
            # vérifier qu'il s'agit d'une EXTRÉMITÉ
            if pos_tgt == 0 or pos_tgt == len(nodes) - 1:
                self._append_triangle_to_group(gt, pos_tgt, idx_mob, key_mob, key_tgt)
            # sinon : insertion au milieu non gérée pour l'instant
            return

        # 3) groupe (mobile) -> tri isolé (cible) : non géré ici
        # 4) groupe ↔ groupe : non géré ici
        # (on laisse la géométrie telle quelle; pas de modification de groupes)
        return

    def _cancel_drag(self):
        self._drag = None
        if self._drag_preview_id is not None:
            self.canvas.delete(self._drag_preview_id)
            self._drag_preview_id = None

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

    @staticmethod
    def _point_in_triangle_world(P, A, B, C):
        """Test barycentrique en coordonnées monde : P à l'intérieur de triangle ABC ?"""
        v0 = np.array([C[0]-A[0], C[1]-A[1]], dtype=float)
        v1 = np.array([B[0]-A[0], B[1]-A[1]], dtype=float)
        v2 = np.array([P[0]-A[0], P[1]-A[1]], dtype=float)
        # Matrice 2x2 inverse via produits scalaires
        dot00 = v0.dot(v0)
        dot01 = v0.dot(v1)
        dot02 = v0.dot(v2)
        dot11 = v1.dot(v1)
        dot12 = v1.dot(v2)
        denom = (dot00 * dot11 - dot01 * dot01)
        if abs(denom) < 1e-12:
            return False
        inv = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        return (u >= 0.0) and (v >= 0.0) and (u + v <= 1.0)

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

            try:
                self._ctx_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self._ctx_menu.grab_release()

    def _ctx_delete_selected(self):
        """Supprime le triangle ciblé par le menu contextuel."""
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None:
            return
        if 0 <= idx < len(self._last_drawn):
            # Annule une éventuelle sélection en cours liée à ce triangle
            if self._sel and self._sel.get("idx") == idx:
                self._sel = None
            # Supprime et redessine
            removed = self._last_drawn.pop(idx)
            self._redraw_from(self._last_drawn)
            self.status.config(text=f"Triangle supprimé (id={removed.get('id','?')}).")
            # Réinsérer dans la liste déroulante au bon endroit
            self._reinsert_triangle_to_list(removed)

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
        Inverse le triangle par symétrie axiale autour de la droite (O→L) passant par le barycentre.
        Ajoute 'S' après le numéro tant que le triangle est inversé.
        """
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None or not (0 <= idx < len(self._last_drawn)):
            return
        t = self._last_drawn[idx]
        P = t["pts"]
        # axe = direction O->L ; pivot = barycentre
        axis = np.array([P["L"][0] - P["O"][0], P["L"][1] - P["O"][1]], dtype=float)
        nrm = float(np.hypot(axis[0], axis[1]))
        if nrm < 1e-12:
            return
        u = axis / nrm
        pivot = self._tri_centroid(P)
        # matrice de réflexion : R = 2 uu^T - I
        R = np.array([[2*u[0]*u[0] - 1, 2*u[0]*u[1]],
                      [2*u[0]*u[1],     2*u[1]*u[1] - 1]], dtype=float)
        for k in ("O", "B", "L"):
            v = np.array([P[k][0] - pivot[0], P[k][1] - pivot[1]], dtype=float)
            Pv = (R @ v) + pivot
            P[k] = Pv
        # toggle du flag 'mirrored'
        t["mirrored"] = not t.get("mirrored", False)
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Inversion appliquée (id={t.get('id','?')}{'S' if t['mirrored'] else ''}).")

    def _on_canvas_left_down(self, event):
        # Nouveau clic gauche : purge l'éventuelle aide précédente (évite les fantômes)
        self._reset_assist()
        # priorité au drag & drop depuis la liste
        if self._drag:
            return
        mode, idx, extra = self._hit_test(event.x, event.y)
        if mode == "center":
            P = self._last_drawn[idx]["pts"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            # Si le triangle appartient à un groupe -> déplacement de groupe
            gid = self._get_group_of_triangle(idx)
            if gid:
                Gc = self._group_centroid(gid)
                if Gc is None:
                    # fallback: déplacer le triangle seul si souci de centroid
                    Cw = self._tri_centroid(P)
                    orig_pts = {k: np.array(P[k].copy()) for k in ("O","B","L")}
                    self._sel = {
                        "mode": "move",
                        "idx": idx,
                        "grab_offset": np.array([wx, wy]) - Cw,
                        "orig_pts": orig_pts,
                    }
                    self.status.config(text=f"Déplacement du triangle #{self._last_drawn[idx].get('id','?')} (fallback)")
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
                }
                self.status.config(text=f"Déplacement du groupe #{gid} (clic sur tri {self._last_drawn[idx].get('id','?')})")
            else:
                # déplacement triangle isolé (comportement existant)
                Cw = self._tri_centroid(P)
                orig_pts = {k: np.array(P[k].copy()) for k in ("O","B","L")}
                self._sel = {
                    "mode": "move",
                    "idx": idx,
                    "grab_offset": np.array([wx, wy]) - Cw,
                    "orig_pts": orig_pts,
                }
                self.status.config(text=f"Déplacement du triangle #{self._last_drawn[idx].get('id','?')}")
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
            self._debug(f"[DBG] CLICK vertex: tri_idx={idx} (id={tri_id}), vkey={vkey}, gid={gid0}")
            if gid0:
                self._debug_dump_group(gid0, "groupe du triangle cliqué (avant test lien)")

            # si ce sommet est un LIEN -> on DECONNECTE et on démarre un move_group par sommet ---
            gid_link, pos, link_type = self._find_group_link_for_vertex(idx, vkey)
            self._debug(f"[DBG] RESULT lien: gid_link={gid_link}, pos={pos}, link_type={link_type}")
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
                self._debug(f"[DBG] split -> left_gid={left_gid}, right_gid={right_gid}, cut_after={cut_after}, pos={pos}")
                # si pos <= cut_after : le tri cliqué est dans la PARTIE GAUCHE
                if pos <= cut_after and left_gid:
                    mobile_gid = left_gid
                # sinon dans la partie droite
                elif pos > cut_after and right_gid:
                    mobile_gid = right_gid

                # Si la partie qui contient le triangle cliqué est redevenue singleton -> pas de group, on retombe sur move triangle
                if mobile_gid is None:
                    # comportement de fallback: on bouge le TRIANGLE SEUL par le sommet
                    orig_pts = {k: np.array(P[k].copy()) for k in ("O","B","L")}
                    self._sel = {
                        "mode": "vertex",
                        "idx": idx,
                        "vkey": vkey,
                        "grab_offset": np.array([wx, wy]) - np.array(P[vkey], dtype=float),
                        "orig_pts": orig_pts,
                        "suppress_assist": True,
                    }
                    # on purge tout résidu d’aide visuelle
                    self._clear_nearest_line(); self._clear_edge_highlights()
                    self.status.config(text=f"Lien cassé. Déplacement du triangle #{self._last_drawn[idx].get('id','?')} par sommet.")
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
                self._debug(f"[DBG] move_group(ANCHOR vertex) gid={mobile_gid}")
                self.status.config(text=f"Lien cassé. Déplacement du groupe #{mobile_gid} par sommet {vkey}.")
                # purge immédiate de toute aide existante
                self._clear_nearest_line()
                self._clear_edge_highlights()
                self._redraw_from(self._last_drawn)
                return

            # --- NOUVEAU : si le triangle appartient à un groupe et qu'on NE tient PAS CTRL,
            #               on déplace le **groupe entier** ancré sur ce sommet.
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
        Après un collage tri->tri (ou tri->groupe), met à jour la structure des groupes.
        Règles :
        - si aucun des deux triangles n’est groupé : création d’un groupe à 2 nœuds.
        - si l’un est dans un groupe et l’autre non : ajout en tête/queue si le collage touche l’extrémité,
            sinon (cas insertion au milieu) on laisse pour plus tard.
        - si les deux sont dans des groupes : tentative de fusion extrémité<->extrémité (sinon on laisse pour plus tard).
        NB : on reste volontairement conservateur pour ne pas casser ton flux actuel.
        """
        tri_m = self._last_drawn[idx_m]
        tri_t = self._last_drawn[idx_t]
        gid_m = tri_m.get("group_id")
        gid_t = tri_t.get("group_id")

        # --- Cas 1 : aucun groupe -> crée un groupe à 2
        if gid_m is None and gid_t is None:
            gid = self._new_group_id()
            self.groups[gid] = {"id": gid, "nodes": []}
            # m devient tête, t devient queue (ou l’inverse ; important seulement pour vkey_in/out)
            self.groups[gid]["nodes"] = [
                {"tid": idx_m, "vkey_in": None,  "vkey_out": vkey_m},
                {"tid": idx_t, "vkey_in": vkey_t, "vkey_out": None},
            ]
            tri_m["group_id"] = gid; tri_m["group_pos"] = 0
            tri_t["group_id"] = gid; tri_t["group_pos"] = 1
            self.status.config(text=f"Groupe #{gid} créé : t0=tid{idx_m}({vkey_m}) → t1=tid{idx_t}({vkey_t})")
            return

        # --- Cas 2 : mobile dans un groupe, cible non -> ajout en extrémité si possible
        if gid_m is not None and gid_t is None:
            nodes = self.groups[gid_m]["nodes"]
            head = nodes[0]; tail = nodes[-1]
            if tri_m["group_pos"] == 0:
                # ajout avant la tête
                nodes.insert(0, {"tid": idx_t, "vkey_in": None, "vkey_out": vkey_t})
                head["vkey_in"] = vkey_m
                # renumérotation pos
                for i, n in enumerate(nodes):
                    self._last_drawn[n["tid"]]["group_pos"] = i
                tri_t["group_id"] = gid_m; tri_t["group_pos"] = 0
                self.status.config(text=f"Triangle ajouté en tête du groupe #{gid_m}.")
                return
            elif tri_m["group_pos"] == len(nodes)-1:
                # ajout après la queue
                nodes.append({"tid": idx_t, "vkey_in": vkey_t, "vkey_out": None})
                nodes[-2]["vkey_out"] = vkey_m
                for i, n in enumerate(nodes):
                    self._last_drawn[n["tid"]]["group_pos"] = i
                tri_t["group_id"] = gid_m; tri_t["group_pos"] = len(nodes)-1
                self.status.config(text=f"Triangle ajouté en queue du groupe #{gid_m}.")
                return
            # insertion au milieu -> on ne touche pas (hors scope), laisse les triangles collés mais sans groupe
            return

        # --- Cas 3 : cible dans un groupe, mobile non -> symétrique
        if gid_m is None and gid_t is not None:
            nodes = self.groups[gid_t]["nodes"]
            head = nodes[0]; tail = nodes[-1]
            # si on épouse la tête -> ajout avant
            if tri_t["group_pos"] == 0:
                nodes.insert(0, {"tid": idx_m, "vkey_in": None, "vkey_out": vkey_m})
                head["vkey_in"] = vkey_t
                for i, n in enumerate(nodes):
                    self._last_drawn[n["tid"]]["group_pos"] = i
                tri_m["group_id"] = gid_t; tri_m["group_pos"] = 0
                self.status.config(text=f"Triangle ajouté en tête du groupe #{gid_t}.")
                return
            elif tri_t["group_pos"] == len(nodes)-1:
                nodes.append({"tid": idx_m, "vkey_in": vkey_m, "vkey_out": None})
                nodes[-2]["vkey_out"] = vkey_t
                for i, n in enumerate(nodes):
                    self._last_drawn[n["tid"]]["group_pos"] = i
                tri_m["group_id"] = gid_t; tri_m["group_pos"] = len(nodes)-1
                self.status.config(text=f"Triangle ajouté en queue du groupe #{gid_t}.")
                return
            return

        # --- Cas 4 : les deux sont déjà groupés -> on tentera une fusion extrémité<->extrémité plus tard
        # (on ne modifie pas ici si ce n’est pas une jonction d’extrémités)
        return

    def _on_canvas_left_move(self, event):
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
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            # Deux modes d'ancrage : 'centroid' (par défaut) ou 'vertex' (nouveau, après déconnexion)
            anchor = self._sel.get("anchor")
            if anchor and anchor.get("type") == "vertex":
                # retrouver la position ACTUELLE du sommet dans le groupe (il peut avoir changé de gid)
                anchor_tid = anchor.get("tid")
                anchor_vkey = anchor.get("vkey")
                # si le triangle ancre n'est plus dans ce gid (peu probable à ce stade), fallback centroid
                in_group = False
                g = self.groups.get(gid)
                if g:
                    for nd in g["nodes"]:
                        if nd["tid"] == anchor_tid:
                            in_group = True
                            break
                if in_group and (0 <= anchor_tid < len(self._last_drawn)):
                    Panchor = self._last_drawn[anchor_tid]["pts"]
                    cur_anchor = np.array(Panchor[anchor_vkey], dtype=float)
                    target_anchor = np.array([wx, wy]) - self._sel["grab_offset"]
                    d = target_anchor - cur_anchor
                else:
                    # fallback centroid
                    target_c = np.array([wx, wy]) - self._sel["grab_offset"]
                    cur_c = self._group_centroid(gid)
                    if cur_c is None:
                        return
                    d = target_c - cur_c
            else:
                target_c = np.array([wx, wy]) - self._sel["grab_offset"]
                cur_c = self._group_centroid(gid)
                if cur_c is None:
                    return
                d = target_c - cur_c
            g = self.groups.get(gid)
            if not g:
                return
            for node in g["nodes"]:
                tid = node["tid"]
                if 0 <= tid < len(self._last_drawn):
                    P = self._last_drawn[tid]["pts"]
                    for k in ("O","B","L"):
                        P[k] = np.array([P[k][0] + d[0], P[k][1] + d[1]])
            self._recompute_group_bbox(gid)
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

            # Déroulé du collage (rotation+translation) sur le TRIANGLE seul
            (_, _, idx_t, vkey_t, (m_a, m_b, t_a, t_b)) = choice
            tri_m = self._last_drawn[idx]
            Pm = tri_m["pts"]

            A = np.array(m_a, dtype=float)  # mobile: sommet saisi
            B = np.array(m_b, dtype=float)  # mobile: voisin qui définit l'arête
            U = np.array(t_a, dtype=float)  # cible: point où coller
            V = np.array(t_b, dtype=float)  # cible: deuxième point de l'arête

            def _ang(p, q):
                v = q - p
                return atan2(v[1], v[0])

            ang_m = _ang(A, B)
            ang_t = _ang(U, V)
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

                def _ang(p, q):
                    v = q - p
                    return atan2(v[1], v[0])

                # Angle mobile/cible et rotation à appliquer
                ang_m = _ang(A, B)
                ang_t = _ang(U, V)
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

                # ====== NOUVEAU : FUSION DE GROUPE APRÈS COLLAGE ======
                # cible : triangle ou groupe ?
                tgt_gid = self._last_drawn[idx_t].get("group_id", None)
                mob_gid = gid
                if mob_gid is not None:
                    if tgt_gid is None:
                        # 1) CIBLE = TRIANGLE SEUL → l'ajouter au groupe mobile
                        self._last_drawn[idx_t]["group_id"] = mob_gid
                        # vkeys: on met None pour l’instant (orientation fine à gérer ultérieurement)
                        self.groups[mob_gid]["nodes"].append({"tid": idx_t, "vkey_in": None, "vkey_out": None})
                        # bbox
                        self._recompute_group_bbox(mob_gid)
                        self.status.config(text=f"Groupe fusionné : triangle {idx_t} ajouté au groupe #{mob_gid}.")
                    elif tgt_gid != mob_gid:
                        # 2) CIBLE = AUTRE GROUPE → fusionner tgt_gid → mob_gid
                        g_src = self.groups.get(mob_gid)
                        g_tgt = self.groups.get(tgt_gid)
                        if g_src and g_tgt:
                            # rattacher tous les triangles du groupe cible au groupe mobile
                            for nd in g_tgt.get("nodes", []):
                                tid2 = nd.get("tid")
                                if tid2 is None:
                                    continue
                                self._last_drawn[tid2]["group_id"] = mob_gid
                                # on pousse tels quels (vkeys conservées si présentes)
                                g_src["nodes"].append({"tid": tid2,
                                                       "vkey_in": nd.get("vkey_in"),
                                                       "vkey_out": nd.get("vkey_out")})
                            # supprimer l'ancien groupe cible
                            try:
                                del self.groups[tgt_gid]
                            except Exception:
                                pass
                            # bbox finale
                            self._recompute_group_bbox(mob_gid)
                            self.status.config(text=f"Groupes fusionnés : #{mob_gid} ← #{tgt_gid}.")
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

    # ---------- Impression PDF (A4) ----------
    def _bbox_of_triangle(self, P):
        xs = [float(P["O"][0]), float(P["B"][0]), float(P["L"][0])]
        ys = [float(P["O"][1]), float(P["B"][1]), float(P["L"][1])]
        return min(xs), min(ys), max(xs), max(ys)

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
