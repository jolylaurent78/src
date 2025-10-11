
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

        # Épaisseur de trait (px) — utilisée pour le test de chevauchement "shrink seul"
        self.stroke_px = 2

        # mode "déconnexion" activé par CTRL
        self._ctrl_down = False        

        # --- DEBUG: permettre de SKIP le test de chevauchement durant le highlight (F9) ---
        self.debug_skip_overlap_highlight = False

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
        top = tk.Frame(self); top.pack(side=tk.TOP, fill=tk.X)
        tk.Button(top, text="Ouvrir Excel...", command=self.open_excel).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Label(top, textvariable=self.triangle_file).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Imprimer…", command=self.print_triangles_dialog).pack(side=tk.RIGHT, padx=5, pady=5)

        main = tk.Frame(self); main.pack(fill=tk.BOTH, expand=True)
        self._build_left_pane(main)
        self._build_canvas(main)

        self.status = tk.Label(self, text="Prêt", bd=1, relief=tk.SUNKEN, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

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
        self._ctx_menu.add_command(label="Supprimer", command=self._ctx_delete_group)
        self._ctx_menu.add_command(label="Pivoter", command=self._ctx_rotate_selected)
        self._ctx_menu.add_command(label="Inverser", command=self._ctx_flip_selected)
        self._ctx_menu.add_command(label="OL=0°", command=self._ctx_orient_OL_north)

        # DEBUG: F9 pour activer/désactiver le filtrage de chevauchement au highlight
        # (bind_all pour capter même si le focus clavier n'est pas explicitement sur le canvas)
        self.bind_all("<F9>", self._toggle_skip_overlap_highlight)

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

    def _incident_half_edges_at_vertex(self, graph, v, eps=1e-6):
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

    def _normalize_to_outline_granularity(self, outline, edges, eps=1e-6):
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
    def _angle_from(self, p0, p1):
        """Azimut de p0->p1 en radians dans [0, 2π)."""
        import math
        a = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
        if a < 0: 
            a += 2*math.pi
        return a

    def _wrap_delta_angle(self, a, b):
        """Distance angulaire minimale entre a et b, résultat dans [0, π]."""
        import math
        d = abs(a - b) % (2*math.pi)
        return d if d <= math.pi else (2*math.pi - d)

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
        # === Invariance: tout item appartient à un groupe (singleton possible) ===
        self.groups = {}
        self._next_group_id = 1
        for i, t in enumerate(self._last_drawn):
            gid = self._next_group_id; self._next_group_id += 1
            self.groups[gid] = {"nodes": [{"tid": i, "vkey_in": None, "vkey_out": None}]}
            t["group_id"] = gid
            t["group_pos"] = 0
            try:
                self._recompute_group_bbox(gid)
            except Exception:
                pass
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
            return self._group_outline_segments(gid, eps=1e-6)
        except Exception:
            return []


    # --- helper : triangle propriétaire d’un sous-segment le long d’une arête d’un groupe ---
    def _build_outline_adjacency(self, outline):
        """Construire l'adjacence {point->liste de voisins} à partir d'un outline (liste de segments)."""
        def _pt_key(p, eps=1e-9):
            return (round(float(p[0])/eps)*eps, round(float(p[1])/eps)*eps)
        adj = {}
        for a,b in (outline or []):
            a = (float(a[0]), float(a[1])); b = (float(b[0]), float(b[1]))
            ka, kb = _pt_key(a), _pt_key(b)
            adj.setdefault(ka, []).append(b)
            adj.setdefault(kb, []).append(a)
        return adj


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
        def _almost_eq(a,b,eps=1e-6): return abs(a[0]-b[0])<=eps and abs(a[1]-b[1])<=eps

        # Sommets & groupes
        tri_m = self._last_drawn[mob_idx];  Pm = tri_m["pts"]; vm = _to_np(Pm[vkey_m])
        tri_t = self._last_drawn[tgt_idx];  Pt = tri_t["pts"]; vt = _to_np(Pt[tgt_vkey])
        gid_m = tri_m.get("group_id"); gid_t = tri_t.get("group_id")
        if gid_m is None or gid_t is None:
            self._clear_edge_highlights(); self._edge_choice = None; return
        
        # Tolérance (utile à d'autres endroits si besoin)
        tol_world = max(1e-9, float(getattr(self, "stroke_px", 2)) / max(getattr(self, "zoom", 1.0), 1e-9))

        # === FRONTIER GRAPH des deux groupes ===
        mob_outline = self._outline_for_item(mob_idx) or []
        tgt_outline = self._outline_for_item(tgt_idx) or []
        g_m = self._build_boundary_graph(mob_outline)
        g_t = self._build_boundary_graph(tgt_outline)

        # === Demi-arêtes incidentes aux sommets cliqués (strictement sur la frontière) ===
        m_inc_raw = self._incident_half_edges_at_vertex(g_m, vm)
        t_inc_raw = self._incident_half_edges_at_vertex(g_t, vt)
        if not t_inc_raw:
            self._clear_edge_highlights(); self._edge_choice = None; return

        # Normalisation à la granularité de l'outline (lisible à l’écran)
        m_inc = self._normalize_to_outline_granularity(mob_outline, m_inc_raw, eps=1e-6)
        t_inc = self._normalize_to_outline_granularity(tgt_outline, t_inc_raw, eps=1e-6)

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
        m_inc = self._normalize_to_outline_granularity(mob_outline, m_inc_raw, eps=1e-6)
        t_inc = self._normalize_to_outline_granularity(tgt_outline, t_inc_raw, eps=1e-6)

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
        li = self._drag.get("list_index")
        if isinstance(li, int):
            try: self.listbox.delete(li)
            except Exception: pass
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
        self._last_drawn = keep
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
