import os
import math
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from shapely.geometry import Polygon, LineString

# ===================== Utils géométrie =====================

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / (n if n else 1.0)

def _rot_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation 2x2 telle que R @ a_hat = b_hat (stables en 2D)."""
    ax, ay = float(a[0]), float(a[1]); la = math.hypot(ax, ay)
    bx, by = float(b[0]), float(b[1]); lb = math.hypot(bx, by)
    if la == 0 or lb == 0:
        return np.eye(2, dtype=float)
    ax /= la; ay /= la; bx /= lb; by /= lb
    cos_t = max(-1.0, min(1.0, ax*bx + ay*by))
    sin_t = ax*by - ay*bx
    return np.array([[cos_t, -sin_t], [sin_t,  cos_t]], dtype=float)

def _reflect_point_across_line(p: np.ndarray, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    u = _unit(e - s)
    w = p - s
    return s + (2.0 * (w @ u) * u - w)

def _build_local_triangle(OB: float, OL: float, BL: float) -> dict:
    """Construit un triangle local avec O=(0,0), B=(OB,0), L=(x,y) à partir des 3 longueurs."""
    x = (OL*OL - BL*BL + OB*OB) / (2*OB)
    y2 = max(0.0, OL*OL - x*x)
    y = math.sqrt(y2)
    return {
        "O": np.array([0.0,     0.0], dtype=float),
        "B": np.array([float(OB), 0.0], dtype=float),
        "L": np.array([float(x),  float(y)], dtype=float),
    }

def _poly_from_pts(P: dict) -> Polygon:
    return Polygon([(float(P["O"][0]), float(P["O"][1])),
                    (float(P["B"][0]), float(P["B"][1])),
                    (float(P["L"][0]), float(P["L"][1]))])

# ===================== Base plugin =====================

class AlgorithmBase:
    name = "Base"
    def __init__(self):
        self.frame = None

    def build_ui(self, parent) -> tk.Frame:
        self.frame = tk.Frame(parent)
        return self.frame

    # === NOUVEAU : plusieurs solutions ===
    def run_all(self, app, start: int, n: int):
        """
        Retourne une LISTE de solutions.
        Chaque solution = liste de tuples (comme aujourd'hui) :
        ("Bourges", Bname, Lname, P_dict, overlap_bool, index_1based)
        Par défaut on wrappe run(...) dans une unique solution.
        """
        sol = self.run(app, start, n)  # compatibilité avec les algos actuels
        return [sol] if sol else []

    # Ancienne API (1 solution). Les algos existants continuent de l'implémenter.
    def run(self, app, start: int, n: int):
        raise NotImplementedError

class AlgoSommetNonCroise(AlgorithmBase):
    """
    Variante 'Alternance BL Multi' + contraintes globales :
    - Chemin des sommets de connexion (B ou L) sans auto-intersection.
    - (Optionnel) Tournant constant (CW ou CCW) sur ce chemin.
    - Conserve le collage arête-à-arête + anti-chevauchement des triangles.
    """
    name = "Sommet non-croisé (B/L)"

    def __init__(self):
        super().__init__()
        import tkinter as tk
        self.start_anchor = tk.StringVar(value="L puis B")  # comme Alternance BL Multi
        self.force_opposite = tk.BooleanVar(value=True)
        self.mirror_if_overlap = tk.BooleanVar(value=True)
        self.same_type_only   = tk.BooleanVar(value=True)
        self.max_solutions    = tk.IntVar(value=200)

        # nouveau : choix du sommet de connexion et contraintes globales
        self.connect_mode = tk.StringVar(value="Suivre ancrage")  # "Toujours B", "Toujours L"
        self.enforce_simple_path = tk.BooleanVar(value=True)
        self.enforce_constant_turn = tk.BooleanVar(value=True)

    def build_ui(self, parent):
        import tkinter as tk
        from tkinter import ttk
        self.frame = tk.Frame(parent)

        tk.Label(self.frame, text="Premier ancrage").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            self.frame, textvariable=self.start_anchor, width=10, state="readonly",
            values=["L puis B", "B puis L"]
        ).grid(row=0, column=1, sticky="w", padx=4)

        tk.Label(self.frame, text="Sommet de connexion").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            self.frame, textvariable=self.connect_mode, width=14, state="readonly",
            values=["Suivre ancrage", "Toujours B", "Toujours L"]
        ).grid(row=1, column=1, sticky="w", padx=4)

        tk.Checkbutton(self.frame, text="Arêtes de même type uniquement",
                       variable=self.same_type_only).grid(row=2, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(self.frame, text="Toujours côté opposé",
                       variable=self.force_opposite).grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(self.frame, text="Miroir auto si chevauchement",
                       variable=self.mirror_if_overlap).grid(row=4, column=0, columnspan=2, sticky="w")

        tk.Checkbutton(self.frame, text="Chemin sans auto-intersection",
                       variable=self.enforce_simple_path).grid(row=5, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(self.frame, text="Tournant constant (CW/CCW)",
                       variable=self.enforce_constant_turn).grid(row=6, column=0, columnspan=2, sticky="w")

        tk.Label(self.frame, text="Max solutions").grid(row=7, column=0, sticky="w")
        ttk.Spinbox(self.frame, from_=1, to=10000, textvariable=self.max_solutions, width=7).grid(row=7, column=1, sticky="w")
        return self.frame

    # --- Helpers géométriques sur le chemin (liste de points 2D) ---
    @staticmethod
    def _polyline_self_intersects(path):
        """True si la poly-ligne se coupe elle-même (hors segments adjacents)."""
        if len(path) < 4:
            return False
        for i in range(1, len(path)):
            seg_i = LineString([tuple(path[i-1]), tuple(path[i])])
            for j in range(1, i-1):
                seg_j = LineString([tuple(path[j-1]), tuple(path[j])])
                if seg_i.intersects(seg_j):
                    inter = seg_i.intersection(seg_j)
                    if not inter.is_empty:
                        return True
        return False

    @staticmethod
    def _turn_sign(a, b, c, eps=1e-9):
        """Signe du virage en b pour a->b->c : +1, -1, 0."""
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        cx, cy = float(c[0]), float(c[1])
        cross = (bx-ax)*(cy-by) - (by-ay)*(cx-bx)
        if abs(cross) <= eps:
            return 0
        return 1 if cross > 0 else -1

    def run_all(self, app, start, n):
        # sécurités
        if getattr(app, "df", None) is None or app.df.empty:
            raise RuntimeError("Pas de données — ouvre d'abord l’Excel.")
        sub = app.df.iloc[start:start+n]
        if sub.empty:
            return []

        # 1) Triangles locaux (B, L, pts)
        tris_local = []
        for _, r in sub.iterrows():
            pts = _build_local_triangle(float(r["len_OB"]), float(r["len_OL"]), float(r["len_BL"]))
            tris_local.append((r["B"], r["L"], pts))

        # 2) Premier triangle posé
        B0, L0, P0 = tris_local[0]
        A = {"O": P0["O"].copy(), "B": P0["B"].copy(), "L": P0["L"].copy()}
        first = [("Bourges", B0, L0, A, False, start+1)]

        # 3) Choix du "sommet de connexion" (B ou L)
        def conn_point(P, use_L_anchor):
            mode = self.connect_mode.get()
            if mode == "Toujours B":
                return P["B"]
            if mode == "Toujours L":
                return P["L"]
            # "Suivre ancrage" : L si on ancre L–L, sinon B
            return P["L"] if use_L_anchor else P["B"]

        start_with_L = (self.start_anchor.get() == "L puis B")
        cpath = [conn_point(A, use_L_anchor=True if start_with_L else False)]
        turn_dir = 0  # 0=non fixé, +1/-1 = sens constant dès le 1er virage non nul

        solutions = []

        def dfs(chain, i, cpath, turn_dir):
            if len(solutions) >= int(self.max_solutions.get()):
                return
            if i == len(tris_local):
                solutions.append(chain.copy())
                return

            Bname, Lname, pts_local = tris_local[i]
            A_pts = chain[-1][3]

            # alternance L–L / B–B
            use_L_anchor = (i % 2 == 1) if start_with_L else (i % 2 == 0)

            # couples d'arêtes à tester (mêmes-type d'abord)
            if self.same_type_only.get():
                candidates = [("LO","LO"), ("LB","LB")] if use_L_anchor else [("BO","BO"), ("BL","BL")]
            else:
                pairs = [("LO","LO"), ("LB","LB")] if use_L_anchor else [("BO","BO"), ("BL","BL")]
                pairs += [("LB","LO"), ("LO","LB")] if use_L_anchor else [("BL","BO"), ("BO","BL")]
                candidates = pairs

            ranked = []
            for edge_i, edge_next in candidates:
                try:
                    B_world = TriangleAssembler.glue_next_on_edge(
                        A_pts, pts_local, edge_i, edge_next,
                        force_opposite_side=self.force_opposite.get()
                    )
                except Exception:
                    continue

                overlap = app._collides_with_previous(chain, B_world)

                # miroir si ça supprime le chevauchement
                if overlap and self.mirror_if_overlap.get():
                    sA = A_pts[edge_i[0]]; eA = A_pts[edge_i[1]]
                    B_mirror = app._mirror_across_edge(B_world, sA, eA)
                    if not app._collides_with_previous(chain, B_mirror):
                        B_world = B_mirror
                        overlap = False

                # PRUNE chevauchement si pas le dernier
                if overlap and i < (len(tris_local) - 1):
                    continue

                # PRUNE chemin des sommets (B/L) non-croisé
                c_next = conn_point(B_world, use_L_anchor)
                new_path = cpath + [c_next]
                if self.enforce_simple_path.get() and self._polyline_self_intersects(new_path):
                    continue

                # PRUNE tournant constant
                new_turn_dir = turn_dir
                if self.enforce_constant_turn.get() and len(new_path) >= 3:
                    s = self._turn_sign(new_path[-3], new_path[-2], new_path[-1])
                    if s != 0:
                        if new_turn_dir == 0:
                            new_turn_dir = s
                        elif s != new_turn_dir:
                            continue

                ranked.append((overlap, edge_i, edge_next, B_world, c_next, new_turn_dir))

            # essayer d'abord ceux sans chevauchement
            ranked.sort(key=lambda x: x[0])  # False (0) avant True (1)

            for overlap, edge_i, edge_next, B_world, c_next, new_turn_dir in ranked:
                if len(solutions) >= int(self.max_solutions.get()):
                    return
                chain.append(("Bourges", Bname, Lname, B_world, overlap, start+i+1))
                dfs(chain, i+1, cpath + [c_next], new_turn_dir)
                chain.pop()

        dfs(first, 1, cpath, turn_dir)
        return solutions


class AlgoAlternanceLLBB(AlgorithmBase):
    """
    Chaînage en alternant l’ancrage sur l’arête BL :
      - Étapes L–L : mapping  LB(i) ↔ LB(i+1)  (points L confondus, arêtes superposées)
      - Étapes B–B : mapping  BL(i) ↔ BL(i+1)  (points B confondus, arêtes superposées)
    """
    name = "Alternance BL (L–L / B–B)"

    def __init__(self):
        super().__init__()
        import tkinter as tk
        # options
        self.start_anchor = tk.StringVar(value="L puis B")  # ou "B puis L"
        self.mirror_if_overlap = tk.BooleanVar(value=True)
        self.force_opposite    = tk.BooleanVar(value=True)

        self._solutions = []      # liste de solutions (liste de listes)
        self._current_solution_ix = None

    def build_ui(self, parent):
        import tkinter as tk
        from tkinter import ttk
        self.frame = tk.Frame(parent)

        tk.Label(self.frame, text="Premier ancrage").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            self.frame, textvariable=self.start_anchor, width=10, state="readonly",
            values=["L puis B", "B puis L"]
        ).grid(row=0, column=1, sticky="w", padx=4)

        tk.Checkbutton(self.frame, text="Miroir auto si chevauchement",
                       variable=self.mirror_if_overlap).grid(row=1, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(self.frame, text="Toujours côté opposé",
                       variable=self.force_opposite).grid(row=2, column=0, columnspan=2, sticky="w")
        return self.frame

    def run(self, app, start, n):
        if getattr(app, "df", None) is None or app.df.empty:
            raise RuntimeError("Pas de données — ouvre d'abord l’Excel.")
        sub = app.df.iloc[start:start+n]
        if sub.empty:
            return []

        # 1) Triangles locaux
        tris_local = []
        for _, r in sub.iterrows():
            pts = _build_local_triangle(float(r["len_OB"]), float(r["len_OL"]), float(r["len_BL"]))
            tris_local.append((r["B"], r["L"], pts))

        # 2) Poser le premier
        placed = []
        B0, L0, P0 = tris_local[0]
        A = {"O": P0["O"].copy(), "B": P0["B"].copy(), "L": P0["L"].copy()}
        placed.append(("Bourges", B0, L0, A))

        # 3) Collage alterné : L–L puis B–B (ou l’inverse, selon choix)
        start_with_L = (self.start_anchor.get() == "L puis B")
        for j in range(1, len(tris_local)):
            Bname, Lname, pts_local = tris_local[j]
            A_pts = placed[-1][3]

            # j=1 → premier collage
            use_L_anchor = (j % 2 == 1) if start_with_L else (j % 2 == 0)

            if use_L_anchor:
                # L–L : départ L des 2 triangles, on superpose LB(i) ↔ LO(i+1)
                edge_i, edge_next = "LB", "LO"
            else:
                # B–B : départ B des 2 triangles, on superpose BL(i) ↔ BO(i+1)
                edge_i, edge_next = "BL", "BO"

            B_world = TriangleAssembler.glue_next_on_edge(
                A_pts, pts_local, edge_i, edge_next,
                force_opposite_side=self.force_opposite.get()
            )

            if self.mirror_if_overlap.get() and app._collides_with_previous(placed, B_world):
                sA = A_pts[edge_i[0]]; eA = A_pts[edge_i[1]]
                B_mirror = app._mirror_across_edge(B_world, sA, eA)
                if not app._collides_with_previous(placed, B_mirror):
                    B_world = B_mirror
                else:
                    raise RuntimeError(f"Chevauchement persistant au triangle {start+j+1}.")

            placed.append(("Bourges", Bname, Lname, B_world))

        return placed


# ===================== Algo chaîne rigide =====================

class AlgoChaineRigide(AlgorithmBase):
    name = "Chaîne rigide"

    def __init__(self):
        super().__init__()
        self.mapping = tk.StringVar(value="BO-OL")
        self.mirror_if_overlap = tk.BooleanVar(value=True)
        self.force_opposite = tk.BooleanVar(value=True)

    def build_ui(self, parent):
        from tkinter import ttk
        self.frame = tk.Frame(parent)

        tk.Label(self.frame, text="Mapping i→i+1").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            self.frame, textvariable=self.mapping, width=10, state="readonly",
            values=[
                "BO-OL","BO-BL","OB-OL","OB-BL",
                "OL-BO","OL-BL","BL-BO","BL-OL",
                "LO-BO","LO-OB","LB-BO","LB-OB"
            ]
        ).grid(row=0, column=1, sticky="w", padx=4)
        tk.Checkbutton(self.frame, text="Miroir auto si chevauchement",
                       variable=self.mirror_if_overlap).grid(row=1, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(self.frame, text="Toujours côté opposé",
                       variable=self.force_opposite).grid(row=2, column=0, columnspan=2, sticky="w")
        return self.frame

    def run(self, app, start, n):
        if getattr(app, "df", None) is None or app.df.empty:
            raise RuntimeError("Pas de données — ouvre d'abord l'Excel.")
        sub = app.df.iloc[start:start+n]
        if sub.empty:
            return []

        # Triangles locaux
        tris_local = []
        for _, r in sub.iterrows():
            pts = _build_local_triangle(float(r["len_OB"]), float(r["len_OL"]), float(r["len_BL"]))
            tris_local.append((r["B"], r["L"], pts))

        # Pose du premier
        placed = []
        B0, L0, P0 = tris_local[0]
        A = {"O": P0["O"].copy(), "B": P0["B"].copy(), "L": P0["L"].copy()}
        placed.append(("Bourges", B0, L0, A))

        # Collage des suivants + anti-chevauchement
        edge_i, edge_next = self._edge_pair(self.mapping.get())
        for j in range(1, len(tris_local)):
            Bname, Lname, pts_local = tris_local[j]
            A_pts = placed[-1][3]

            B_world = TriangleAssembler.glue_next_on_edge(
                A_pts, pts_local, edge_i, edge_next,
                force_opposite_side=self.force_opposite.get()
            )

            if self.mirror_if_overlap.get() and app._collides_with_previous(placed, B_world):
                sA = A_pts[edge_i[0]]; eA = A_pts[edge_i[1]]
                B_mirror = app._mirror_across_edge(B_world, sA, eA)
                if not app._collides_with_previous(placed, B_mirror):
                    B_world = B_mirror
                else:
                    raise RuntimeError(f"Chevauchement persistant au triangle {start+j+1}.")

            placed.append(("Bourges", Bname, Lname, B_world))

        return placed

    def _edge_pair(self, s: str):
        """
        Convertit 'BO-OL' -> ('BO','OL') avec validation.
        Arêtes autorisées: OB, BO, OL, LO, BL, LB.
        """
        s = (s or "").strip().upper()
        if "-" not in s:
            raise ValueError(f"Mapping invalide: {s!r}")
        left, right = [p.strip() for p in s.split("-", 1)]
        allowed = {"OB", "BO", "OL", "LO", "BL", "LB"}
        if left not in allowed or right not in allowed:
            raise ValueError(f"Arêtes inconnues dans le mapping: {left!r}, {right!r}")
        # Vérifie que chaque côté est bien une arête de 2 lettres parmi O/B/L
        if len(left) != 2 or len(right) != 2:
            raise ValueError(f"Mapping mal formé: {s!r}")
        return left, right


class AlgoAlternanceBLMulti(AlgorithmBase):
    """
    Alternance BL multi-solutions :
    - on alterne ancrage L puis B puis L...
    - à chaque étape, on autorise plusieurs arêtes côté triangle suivant :
        * si L–L : LB ou LO
        * si B–B : BL ou BO
    L’algo explore récursivement toutes les combinaisons et retourne toutes les solutions.
    """
    name = "Alternance BL Multi"

    def __init__(self):
        super().__init__()
        import tkinter as tk
        self.start_anchor = tk.StringVar(value="L puis B")  # ou "B puis L"
        self.force_opposite = tk.BooleanVar(value=True)
        self.mirror_if_overlap = tk.BooleanVar(value=True)
        self.same_type_only   = tk.BooleanVar(value=True) 
        self.allow_only_last_overlap = tk.BooleanVar(value=True)  # prune tôt
        self.max_solutions           = tk.IntVar(value=200)       # coupe la recherche

    def build_ui(self, parent):
        import tkinter as tk
        from tkinter import ttk
        self.frame = tk.Frame(parent)

        tk.Label(self.frame, text="Premier ancrage").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            self.frame, textvariable=self.start_anchor, width=10, state="readonly",
            values=["L puis B", "B puis L"]
        ).grid(row=0, column=1, sticky="w", padx=4)

        tk.Checkbutton(self.frame, text="Toujours côté opposé",
                       variable=self.force_opposite).grid(row=1, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(self.frame, text="Miroir auto si chevauchement",
                       variable=self.mirror_if_overlap).grid(row=2, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(
        self.frame, text="Arêtes de même type uniquement", variable=self.same_type_only).grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(self.frame, text="Autoriser le chevauchement au dernier seulement",
            variable=self.allow_only_last_overlap).grid(row=4, column=0, columnspan=2, sticky="w")
        tk.Label(self.frame, text="Max solutions").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(self.frame, from_=1, to=10000, textvariable=self.max_solutions, width=7).grid(row=5, column=1, sticky="w")
        return self.frame

    def run_all(self, app, start, n):
        if getattr(app, "df", None) is None or app.df.empty:
            raise RuntimeError("Pas de données — ouvre d'abord l’Excel.")
        sub = app.df.iloc[start:start+n]
        if sub.empty:
            return []

        # triangles locaux
        tris_local = []
        for _, r in sub.iterrows():
            pts = _build_local_triangle(float(r["len_OB"]), float(r["len_OL"]), float(r["len_BL"]))
            tris_local.append((r["B"], r["L"], pts))

        # première pose
        B0, L0, P0 = tris_local[0]
        first = [("Bourges", B0, L0,
                  {"O": P0["O"].copy(), "B": P0["B"].copy(), "L": P0["L"].copy()},
                  False, start+1)]

        solutions = []

        start_with_L = (self.start_anchor.get() == "L puis B")

        def dfs(chain, i):
            # coupe immédiatement si on a déjà atteint le quota de solutions
            if len(solutions) >= int(self.max_solutions.get()):
                return

            if i == len(tris_local):
                solutions.append(chain.copy())
                return

            Bname, Lname, pts_local = tris_local[i]
            A_pts = chain[-1][3]

            # ancrage L–L ou B–B (i=1 pour le 2e triangle)
            use_L_anchor = (i % 2 == 1) if start_with_L else (i % 2 == 0)

            # couples d'arêtes à essayer (ordre favorable)
            if self.same_type_only.get():
                candidates = [("LO","LO"), ("LB","LB")] if use_L_anchor else [("BO","BO"), ("BL","BL")]
            else:
                pairs = [("LO","LO"), ("LB","LB")] if use_L_anchor else [("BO","BO"), ("BL","BL")]
                pairs += [("LB","LO"), ("LO","LB")] if use_L_anchor else [("BL","BO"), ("BO","BL")]
                candidates = pairs

            ranked = []
            for edge_i, edge_next in candidates:
                try:
                    B_world = TriangleAssembler.glue_next_on_edge(
                        A_pts, pts_local, edge_i, edge_next,
                        force_opposite_side=self.force_opposite.get()
                    )
                except Exception:
                    continue

                # chevauchement direct ?
                overlap = app._collides_with_previous(chain, B_world)

                # miroir éventuel si ça supprime le chevauchement
                if overlap and self.mirror_if_overlap.get():
                    sA = A_pts[edge_i[0]]; eA = A_pts[edge_i[1]]
                    B_mirror = app._mirror_across_edge(B_world, sA, eA)
                    if not app._collides_with_previous(chain, B_mirror):
                        B_world = B_mirror
                        overlap = False

                # >>> ICI vient le PRUNE FORT (exactement à cet endroit) <<<
                # si rouge et PAS le dernier triangle -> on coupe la branche
                if overlap and self.allow_only_last_overlap.get() and i < (len(tris_local) - 1):
                    continue

                ranked.append((overlap, edge_i, edge_next, B_world))

            # essayer d'abord ceux sans chevauchement
            ranked.sort(key=lambda x: x[0])  # False (0) avant True (1)

            for overlap, edge_i, edge_next, B_world in ranked:
                # sécurité : re-coupe si on a atteint le quota
                if len(solutions) >= int(self.max_solutions.get()):
                    return

                chain.append(("Bourges", Bname, Lname, B_world, overlap, start+i+1))
                dfs(chain, i+1)
                chain.pop()


        dfs(first, 1)
        return solutions

# ===================== Application =====================

class TriangleAssembler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Assembleur de Triangles — Tkinter (robuste)")
        self.geometry("1200x700")

        # état de vue
        self.zoom = 1.0
        self.offset = np.array([400.0, 350.0], dtype=float)

        # état IHM
        self.triangle_file = tk.StringVar(value="")
        self.start_index = tk.IntVar(value=1)
        self.num_triangles = tk.IntVar(value=2)

        self.excel_path = None
        self.df = None
        self._last_placed = []

        self._build_ui()
        # auto-load si présent
        default = "../data/triangle.xlsx"
        if os.path.exists(default):
            self.load_excel(default)

    def _set_solutions(self, solutions):
        """Enregistre les solutions et peuple la combo."""
        self._solutions = solutions or []
        menu = self.solution_combo["menu"]
        menu.delete(0, "end")

        if not self._solutions:
            self.solution_choice.set("")
            self._current_solution_ix = None
            return

        # Noms "Solution 1 (N tris)", "Solution 2 (N tris)", ...
        names = [f"Solution {i+1} ({len(sol)} tris)" for i, sol in enumerate(self._solutions)]
        for i, nm in enumerate(names):
            menu.add_command(label=nm, command=lambda idx=i: self._on_solution_change(idx))
        self._current_solution_ix = 0
        self.solution_choice.set(names[0])

    def _on_solution_change(self, idx):
        self._current_solution_ix = idx
        names = [f"Solution {i+1} ({len(sol)} tris)" for i, sol in enumerate(self._solutions)]
        self.solution_choice.set(names[idx])
        self._render_current_solution()

    def _render_current_solution(self):
        """Efface le canvas et dessine la solution courante."""
        self.clear_canvas()
        if self._current_solution_ix is None:
            return
        sol = self._solutions[self._current_solution_ix]
        if not sol:
            return
        self._last_placed = sol

        # Fit + dessin (rouge = chevauchement)
        self._fit_to_view(sol)
        overlap_indices = []
        for item in sol:
            Oname, Bname, Lname, P = item[0], item[1], item[2], item[3]
            overlap = item[4] if len(item) > 4 else False
            idx1b   = item[5] if len(item) > 5 else None
            color = "red" if overlap else "black"
            self._draw_triangle_screen(P, outline=color,
                                    labels=[f"O:{Oname}", f"B:{Bname}", f"L:{Lname}"])
            if overlap and idx1b is not None:
                overlap_indices.append(idx1b)

        if overlap_indices:
            self.status.config(text="Chevauchements: " + ", ".join(map(str, overlap_indices)))
        else:
            self.status.config(text="Assemblage terminé")


    # ---------- UI ----------
    def _build_ui(self):
        top = tk.Frame(self); top.pack(side=tk.TOP, fill=tk.X)
        tk.Button(top, text="Imprimer…", command=self.print_triangles_dialog).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top, text="Ouvrir Excel...", command=self.open_excel).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Label(top, textvariable=self.triangle_file).pack(side=tk.LEFT, padx=5)

        main = tk.Frame(self); main.pack(fill=tk.BOTH, expand=True)
        self._build_left_pane(main)
        self._build_canvas(main)

        self.status = tk.Label(self, text="Prêt", bd=1, relief=tk.SUNKEN, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self._init_plugins()

    def _build_left_pane(self, parent):
        left = tk.Frame(parent, width=220); left.pack(side=tk.LEFT, fill=tk.Y); left.pack_propagate(False)

        tk.Label(left, text="Triangles (ordre)").pack(anchor="w", padx=6, pady=(6,0))
        self.listbox = tk.Listbox(left, width=28, height=14, exportselection=False)
        self.listbox.pack(fill=tk.X, padx=6)

        algo_box = tk.LabelFrame(left, text="Algorithme"); algo_box.pack(fill=tk.X, padx=6, pady=6)
        tk.Label(algo_box, text="Choix").grid(row=0, column=0, sticky="w")
        self.algo_choice = tk.StringVar()
        self.algo_combo = tk.OptionMenu(algo_box, self.algo_choice, ())
        self.algo_combo.grid(row=0, column=1, sticky="ew"); algo_box.grid_columnconfigure(1, weight=1)

        tk.Label(algo_box, text="Début").grid(row=1, column=0, sticky="w")
        tk.Entry(algo_box, textvariable=self.start_index, width=6).grid(row=1, column=1, sticky="w")
        tk.Label(algo_box, text="Nombre N").grid(row=2, column=0, sticky="w")
        tk.Entry(algo_box, textvariable=self.num_triangles, width=6).grid(row=2, column=1, sticky="w")

        self.algo_params_container = tk.LabelFrame(left, text="Paramètres")
        self.algo_params_container.pack(fill=tk.X, padx=6, pady=(0,6))

        # --- Sélecteur de solution ---
        sol_box = tk.LabelFrame(left, text="Solution")
        sol_box.pack(fill=tk.X, padx=6, pady=(0,6))
        tk.Label(sol_box, text="Choisir").grid(row=0, column=0, sticky="w")
        self.solution_choice = tk.StringVar(value="")
        self.solution_combo = tk.OptionMenu(sol_box, self.solution_choice, ())
        self.solution_combo.grid(row=0, column=1, sticky="ew")
        sol_box.grid_columnconfigure(1, weight=1)

        btns = tk.Frame(left); btns.pack(fill=tk.X, padx=6, pady=(0,6))
        tk.Button(btns, text="Assembler", command=self.run_current_algo).pack(side=tk.LEFT)
        tk.Button(btns, text="Fit à l'écran", command=lambda: self._fit_to_view(self._last_placed)).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Effacer", command=self.clear_canvas).pack(side=tk.LEFT)


    def _build_canvas(self, parent):
        self.canvas = tk.Canvas(parent, bg="white")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Mouse wheel zoom (Windows/macOS)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        # Mouse wheel zoom (Linux)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)

        # Pan avec clic gauche
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)


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
            row_norm = [TriangleAssembler._norm(x) for x in df0.iloc[i].tolist()]
            if any("ouverture" in x for x in row_norm) and \
               any("base"      in x for x in row_norm) and \
               any("lumiere"   in x for x in row_norm):
                return i
        raise KeyError("Impossible de détecter l'entête ('Ouverture', 'Base', 'Lumière').")

    @staticmethod
    def _build_df(df: pd.DataFrame) -> pd.DataFrame:
        cmap = {TriangleAssembler._norm(c): c for c in df.columns}
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
        self._last_placed = []

    # ---------- Dessin / vue ----------
    def _redraw_from(self, placed):
        """Redraws 'placed' with current zoom/offset (no fit, no status reset)."""
        self.canvas.delete("all")
        if not placed:
            return
        for item in placed:
            Oname, Bname, Lname, P = item[0], item[1], item[2], item[3]
            overlap = item[4] if len(item) > 4 else False
            color = "red" if overlap else "black"
            self._draw_triangle_screen(P, outline=color,
                                    labels=[f"O:{Oname}", f"B:{Bname}", f"L:{Lname}"])


    # ---------- Mouse navigation ----------
    def _on_mousewheel(self, event):
        # Normalize wheel delta across platforms
        if hasattr(event, "delta") and event.delta != 0:
            dz = 1.1 if event.delta > 0 else 1/1.1
        else:
            # Linux: Button-4 -> zoom in, Button-5 -> zoom out
            dz = 1.1 if getattr(event, "num", 0) == 4 else 1/1.1

        # World coordinate under cursor BEFORE zoom
        wx = (event.x - self.offset[0]) / self.zoom
        wy = (self.offset[1] - event.y) / self.zoom

        # Apply zoom (clamped)
        self.zoom = max(0.05, min(100.0, self.zoom * dz))

        # Adjust offset so (wx,wy) remains under cursor AFTER zoom
        self.offset = np.array([event.x - wx * self.zoom,
                                event.y + wy * self.zoom], dtype=float)

        self._redraw_from(self._last_placed)

    def _on_pan_start(self, event):
        self._pan_anchor = np.array([event.x, event.y], dtype=float)
        self._offset_anchor = self.offset.copy()

    def _on_pan_move(self, event):
        if getattr(self, "_pan_anchor", None) is None:
            return
        d = np.array([event.x, event.y], dtype=float) - self._pan_anchor
        self.offset = self._offset_anchor + d
        self._redraw_from(self._last_placed)

    def _on_pan_end(self, event):
        self._pan_anchor = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self._last_placed = []
        self.status.config(text="Canvas effacé")

    def _world_to_screen(self, p):
        x = self.offset[0] + float(p[0]) * self.zoom
        y = self.offset[1] - float(p[1]) * self.zoom
        return x, y

    def _fit_to_view(self, placed):
        if not placed:
            return
        xs, ys = [], []
        for item in placed:
            P = item[3]  # 4e champ = dict {'O','B','L'}
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



    def _draw_triangle_screen(self, P, outline="black", width=2, labels=None, inset=0.35):
        """
        P : dict {'O','B','L'} en coordonnées monde (np.array 2D)
        labels : liste de 3 strings pour O,B,L (facultatif)
        inset : 0..1, fraction du chemin du sommet vers le barycentre pour placer le texte
        """
        # 1) coords monde -> écran pour le polygone
        pts_world = [P["O"], P["B"], P["L"]]
        coords = []
        for pt in pts_world:
            sx, sy = self._world_to_screen(pt)
            coords += [sx, sy]

        # 2) tracé du triangle
        self.canvas.create_polygon(*coords, outline=outline, fill="", width=width)

        # 3) barycentre (monde)
        cx = (P["O"][0] + P["B"][0] + P["L"][0]) / 3.0
        cy = (P["O"][1] + P["B"][1] + P["L"][1]) / 3.0

        # 4) placer les labels vers l'intérieur
        if labels:
            for pt, txt in zip(pts_world, labels):
                lx = (1.0 - inset) * pt[0] + inset * cx
                ly = (1.0 - inset) * pt[1] + inset * cy
                sx, sy = self._world_to_screen((lx, ly))
                self.canvas.create_text(sx, sy, text=txt, anchor="center", font=("Arial", 8))



    # ---------- Collage + détection collisions ----------
    @staticmethod
    def glue_next_on_edge(A_pts, B_pts_local, edge_i, edge_next, force_opposite_side=True):
        sA = A_pts[edge_i[0]]; eA = A_pts[edge_i[1]]; vA = eA - sA
        SB = B_pts_local[edge_next[0]]; EB = B_pts_local[edge_next[1]]; vB = EB - SB
        R = _rot_from_to(vB, vA)
        t = sA - (R @ SB)
        T = lambda p: (R @ p) + t
        B_world = {k: T(v) for k, v in B_pts_local.items()}
        if force_opposite_side:
            thirdA = ({'O','B','L'} - set(edge_i)).pop()
            thirdB = ({'O','B','L'} - set(edge_next)).pop()
            def side(P,S,E):
                v = E - S; w = P - S
                return np.sign(v[0]*w[1] - v[1]*w[0])
            if side(A_pts[thirdA], sA, eA) == side(B_world[thirdB], sA, eA):
                for k in B_world:
                    B_world[k] = _reflect_point_across_line(B_world[k], sA, eA)
        return B_world

    def _collides_with_previous(self, placed, cand_pts, eps=1e-9) -> bool:
        """Retourne True s'il y a chevauchement strict avec l'un des triangles déjà placés."""
        P = _poly_from_pts(cand_pts)
        for item in placed:
            prev_pts = item[3]  # 4e champ = dict {'O','B','L'}
            inter = P.intersection(_poly_from_pts(prev_pts))
            if not inter.is_empty and getattr(inter, "area", 0.0) > eps:
                return True
        return False


    def _mirror_across_edge(self, P, sA, eA):
        return {
            "O": _reflect_point_across_line(P["O"], sA, eA),
            "B": _reflect_point_across_line(P["B"], sA, eA),
            "L": _reflect_point_across_line(P["L"], sA, eA),
        }

    # ---------- Plugins ----------
    def _init_plugins(self):
        self.algorithms = [
            AlgoChaineRigide(),
            AlgoAlternanceLLBB(), 
            AlgoAlternanceBLMulti(),
            AlgoSommetNonCroise()
        ]
        menu = self.algo_combo["menu"]; menu.delete(0, "end")
        names = [a.name for a in self.algorithms]
        for nm in names:
            menu.add_command(label=nm, command=lambda v=nm: self._select_algo(v))
        if names:
            self._select_algo(names[0])
    
    def _select_algo(self, name):
        self.algo_choice.set(name)
        for w in self.algo_params_container.winfo_children():
            w.destroy()

        self.current_algo = next(a for a in self.algorithms if a.name == name)

        # Titre unique de la frame de paramètres
        self.algo_params_container.configure(text=f"Paramètres — {self.current_algo.name}")

        frm = self.current_algo.build_ui(self.algo_params_container)  # => retourne un Frame simple
        frm.pack(fill=tk.X)


    def run_current_algo(self):
        start = max(0, int(self.start_index.get()) - 1)
        n = max(1, int(self.num_triangles.get()))

        solutions = self.current_algo.run_all(self, start, n)

        def is_valid(sol):
            overlaps = [i for i,it in enumerate(sol) if len(it) > 4 and it[4]]
            if not overlaps:
                return True  # aucune collision
            if len(overlaps) == 1 and overlaps[0] == len(sol)-1:
                return True  # une seule collision, sur le dernier triangle
            return False

        # filtrer
        solutions = [s for s in solutions if is_valid(s)]

        # trier : solutions sans rouge d'abord
        def score(sol):
            return sum(1 for it in sol if len(it) > 4 and it[4])
        solutions = sorted(solutions, key=score)

        self._set_solutions(solutions)
        self._render_current_solution()



    # ---------- Impression PDF (A4) ----------

    def _triangles_local(self, start, n):
        """Construit les triangles (coord. locales) à partir du DF pour [start:start+n]."""
        if getattr(self, "df", None) is None or self.df.empty:
            raise RuntimeError("Pas de données — ouvre d'abord l'Excel.")
        sub = self.df.iloc[start:start+n]
        out = []
        for _, r in sub.iterrows():
            P = _build_local_triangle(float(r["len_OB"]), float(r["len_OL"]), float(r["len_BL"]))
            out.append({
                "labels": ( "Bourges", str(r["B"]), str(r["L"]) ),  # O,B,L
                "pts": P
            })
        return out

    def _bbox_of_triangle(self, P):
        xs = [float(P["O"][0]), float(P["B"][0]), float(P["L"][0])]
        ys = [float(P["O"][1]), float(P["B"][1]), float(P["L"][1])]
        return min(xs), min(ys), max(xs), max(ys)

    def _export_triangles_pdf(
        self, path, triangles, scale_mm=0.125,
        page_margin_mm=12, cell_pad_mm=6,
        stroke_pt=1.2, font_size_pt=9, label_inset=0.35,
        pack_mode="shelf",           # "shelf" = rangées à hauteur variable (recommandé)
        rotate_to_fit=True           # rotation 90° si ça réduit la largeur en rangée
    ):
        """
        PDF A4 portrait, triangles à la même échelle, placement optimisé par 'shelf packing'.
        - scale_mm: mm par unité -> même échelle pour tous (c’est la seule contrainte forte).
        - pack_mode: "shelf" (rangées à hauteur variable). Facile, très efficace.
        - rotate_to_fit: autorise la rotation de 90° si ça fait gagner de la place.
        """
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        import math

        if not triangles:
            raise ValueError("Aucun triangle à imprimer.")

        S = float(scale_mm) * float(mm)      # points par unité
        page_w, page_h = A4
        margin = float(page_margin_mm) * float(mm)
        pad    = float(cell_pad_mm) * float(mm)

        # --- Prépare bboxes à l’échelle (chaque tri peut être dessiné en orientation normale OU rotée)
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

        # --- Espace utile page
        content_w = page_w - 2*margin
        content_h = page_h - 2*margin

        # --- Placement 'shelf' (rangées)
        placements = []  # par page: liste de dict {x,y,w,h,rot,t}
        page_list  = []
        def new_page():
            return {"rows": [], "height_used": 0.0}

        def close_row(page, row):
            page["rows"].append(row)
            page["height_used"] += row["H"]

        page = new_page()
        row  = {"X": 0.0, "Y": 0.0, "H": 0.0, "items": []}
        page_no = 1

        for it in items:
            # choisir orientation qui tient le mieux dans la rangée
            # (on privilégie la largeur plus petite)
            cand = []
            if rotate_to_fit:
                cand.append(("R", it["wR"], it["hR"]))
            cand.append(("N", it["wN"], it["hN"]))
            cand.sort(key=lambda x: x[1])  # largeur croissante

            placed = False
            for rot, w, h in cand:
                if row["X"] + w <= content_w or row["X"] == 0.0:
                    # si ça tient → place
                    # si rangée vide, sa hauteur = h
                    Hnew = max(row["H"], h) if row["H"] > 0 else h
                    # si la nouvelle hauteur dépasse la page, on ferme la page d'abord
                    if (page["height_used"] + Hnew) > content_h and row["X"] == 0.0:
                        # rien placé dans cette rangée et ça ne tiendra jamais → nouvelle page
                        if row["H"] > 0:
                            close_row(page, row)
                        page_list.append(page)
                        page = new_page()
                        row  = {"X": 0.0, "Y": 0.0, "H": 0.0, "items": []}
                        # recalcul Hnew ici inutile: row vide

                    # re-teste: si ça ne tient pas en largeur, on essaie la prochaine orientation
                    if row["X"] + w <= content_w or row["X"] == 0.0:
                        row["items"].append({"x": row["X"], "y": page["height_used"], "w": w, "h": h, "rot": (rot=="R"), "it": it})
                        row["X"] += w
                        row["H"] = max(row["H"], h)
                        placed = True
                        break

            if not placed:
                # nouvelle rangée
                close_row(page, row)
                row  = {"X": 0.0, "Y": page["height_used"], "H": 0.0, "items": []}
                # place dans la nouvelle rangée (reprend la meilleure orientation)
                rot, w, h = cand[0]
                # si la nouvelle rangée dépasse la page → nouvelle page
                if page["height_used"] + h > content_h:
                    # ferme page, enregistre, repart à zéro
                    page_list.append(page)
                    page = new_page()
                    row  = {"X": 0.0, "Y": 0.0, "H": 0.0, "items": []}
                row["items"].append({"x": row["X"], "y": page["height_used"], "w": w, "h": h, "rot": (rot=="R"), "it": it})
                row["X"] += w
                row["H"] = max(row["H"], h)

        # terminer dernière rangée/page
        if row["items"]:
            close_row(page, row)
        page_list.append(page)

        # --- Dessin
        c = canvas.Canvas(path, pagesize=A4)
        c.setStrokeColorRGB(0, 0, 0)
        c.setFillColorRGB(0, 0, 0)

        def draw_tri(at_x, at_y, box_w, box_h, item, rot90):
            """Dessine un triangle centré dans la boîte (at_x,at_y,box_w,box_h) en option roté 90°."""
            t   = item["data"]
            P   = t["pts"]
            Oname, Bname, Lname = t["labels"]
            mnx, mny, mxx, mxy = item["bbox_world"]

            tri_w = (mxx - mnx) * S
            tri_h = (mxy - mny) * S
            # si rotation, on inverse w/h pour centrer
            draw_w = tri_h if rot90 else tri_w
            draw_h = tri_w if rot90 else tri_h

            cx = margin + at_x + (box_w - draw_w) / 2.0 + pad
            cy = margin + at_y + (box_h - draw_h) / 2.0 + pad

            def to_page(p):
                x = (float(p[0]) - mnx) * S
                y = (float(p[1]) - mny) * S
                return (x, y)

            O = to_page(P["O"]); B = to_page(P["B"]); L = to_page(P["L"])

            c.saveState()
            # placer l'origine
            c.translate(cx, cy)
            if rot90:
                # rotation autour du coin bas-gauche de la zone de dessin
                c.translate(0, draw_h)
                c.rotate(-90)

            # traits
            c.setLineWidth(stroke_pt)
            c.line(O[0], O[1], B[0], B[1])
            c.line(B[0], B[1], L[0], L[1])
            c.line(L[0], L[1], O[0], O[1])

            # labels (police FIXE, même taille partout)
            c.setFont("Helvetica", float(font_size_pt))
            gx = (O[0] + B[0] + L[0]) / 3.0
            gy = (O[1] + B[1] + L[1]) / 3.0
            for (px, py), txt in zip((O, B, L), (Oname, Bname, Lname)):
                lx = (1.0 - label_inset) * px + label_inset * gx
                ly = (1.0 - label_inset) * py + label_inset * gy
                c.drawCentredString(lx, ly, txt)
            c.restoreState()

        # boucle pages
        for pg in page_list:
            for row in pg["rows"]:
                for cell in row["items"]:
                    draw_tri(cell["x"], cell["y"], cell["w"], cell["h"], cell["it"], cell["rot"])
            c.showPage()
        c.save()


    def print_triangles_dialog(self):
        """Petit dialogue minimal pour exporter un PDF des triangles (échelle & sélection)."""
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

# ===================== Entrée =====================

if __name__ == "__main__":
    app = TriangleAssembler()
    app.mainloop()
