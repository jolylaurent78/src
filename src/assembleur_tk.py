
import os
import math
import re
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------- Utils géométrie ----------

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
        # Par défaut : aucun choix mémorisé
        self._edge_choice = None        
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

        # numéro du triangle au centre (toujours au-dessus)
        if tri_id is not None:
            sx, sy = self._world_to_screen((cx, cy))
            num_txt = f"{tri_id}{'S' if tri_mirrored else ''}"
            self.canvas.create_text(
                sx, sy, text=num_txt,
                anchor="center", font=("Arial", 10, "bold"),
                fill="red", tags="tri_num"
            )
            # amener les numéros en avant-plan
            self.canvas.tag_raise("tri_num")

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

    # ---------- Lien + surlignage faces candidates ----------
    def _find_nearest_vertex(self, v_world, exclude_idx=None):
        """Retourne (idx_triangle, key('O'|'B'|'L'), pos_world) du sommet d'un AUTRE triangle le plus proche."""
        best = None
        best_d2 = None
        for j, t in enumerate(self._last_drawn):
            if j == exclude_idx:
                continue
            P = t["pts"]
            for k in ("O","B","L"):
                w = np.array(P[k], dtype=float)
                d2 = float((w[0]-v_world[0])**2 + (w[1]-v_world[1])**2)
                if (best_d2 is None) or (d2 < best_d2):
                    best_d2 = d2
                    best = (j, k, w)
        return best

    def _update_nearest_line(self, v_world, exclude_idx=None):
        """Dessine (ou MAJ) un trait fin entre v_world et le sommet le plus proche d'un AUTRE triangle."""
        found = self._find_nearest_vertex(v_world, exclude_idx=exclude_idx)
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

    def _update_edge_highlights(self, i_mob, key_mob, i_tgt, key_tgt):
        """
        Surligne les DEUX arêtes candidates à coller.
        Règle retenue :
          1) azimuts sur [0, 2π) (pas de +π).
          2) on choisit le couple au Δ d’azimut minimal (distance circulaire).
          3) filtre anti-chevauchement très simple basé sur le SIGNE :
             si au mobile, l'autre arête (depuis le même sommet) est à +δ (sens anti-horaire)
             par rapport à l'arête choisie, alors au triangle cible l'autre arête doit être à −δ
             (sens horaire) — i.e. signes opposés.
        """
        self._clear_edge_highlights()

        def incident_edges(vkey):
            if vkey == "O":   return [("O","B"), ("O","L")]
            if vkey == "B":   return [("B","O"), ("B","L")]
            else:             return [("L","O"), ("L","B")]

        def dir_angle_2pi(P, a, b):
            """Angle orienté de l’arête (a->b) sur [0, 2π)."""
            vx = float(P[b][0] - P[a][0]); vy = float(P[b][1] - P[a][1])
            ang = math.atan2(vy, vx)
            return ang + (2*math.pi if ang < 0 else 0.0)

        def ang_diff_2pi(a, b):
            """Distance angulaire circulaire min(|a-b|, 2π-|a-b|)."""
            d = abs(a - b)
            return (2*math.pi - d) if d > math.pi else d

        def wrap_pi(a):
            """Normalise un angle en (-π, π]."""
            while a <= -math.pi: a += 2*math.pi
            while a >   math.pi: a -= 2*math.pi
            return a

        def incident_edges(vkey):
            if vkey == "O":   return [("O","B"), ("O","L")]
            if vkey == "B":   return [("B","O"), ("B","L")]
            else:             return [("L","O"), ("L","B")]

        def other_endpoint_key(vkey, chosen):
            """Retourne la clé du 2e voisin depuis vkey, autre que chosen[1]."""
            if vkey == "O":
                return "L" if chosen == ("O","B") else "B"
            if vkey == "B":
                return "L" if chosen == ("B","O") else "O"
            # vkey == "L"
            return "B" if chosen == ("L","O") else "O"

        Pm = self._last_drawn[i_mob]["pts"]
        Pt = self._last_drawn[i_tgt]["pts"]
        Vm = np.array(Pm[key_mob], dtype=float)
        Vt = np.array(Pt[key_tgt], dtype=float)

        cand_m = incident_edges(key_mob)
        cand_t = incident_edges(key_tgt)

        # Recherche du couple (am->bm) / (at->bt) au Δ d’azimut minimal (mod 2π),
        # avec filtre signe opposé sur l'angle vers "l'autre" arête autour du sommet.
        best_m, best_t, best_delta = None, None, 1e9
        for am, bm in cand_m:
            am_ang = dir_angle_2pi(Pm, am, bm)
            # angle signé vers l'autre arête au MOBILE
            mob_other_key = other_endpoint_key(key_mob, (am, bm))
            am_other_ang  = dir_angle_2pi(Pm, key_mob, mob_other_key)
            mob_signed    = wrap_pi(am_other_ang - am_ang)  # >0: l'autre est à gauche (CCW)
            for at, bt in cand_t:
                at_ang = dir_angle_2pi(Pt, at, bt)
                # angle signé vers l'autre arête au CIBLE
                tgt_other_key = other_endpoint_key(key_tgt, (at, bt))
                at_other_ang  = dir_angle_2pi(Pt, key_tgt, tgt_other_key)
                tgt_signed    = wrap_pi(at_other_ang - at_ang)

                # filtre "signe opposé" : on veut mob_signed * tgt_signed < 0
                # (si l'un est +δ, l'autre doit être −δ).
                if mob_signed == 0 or tgt_signed == 0:
                    # cas très rare (arêtes colinéaires) — on n'exclut pas, mais on peut
                    # ignorer ce filtre; sinon, imposer strictement <0 :
                    pass
                else:
                    if mob_signed * tgt_signed >= 0:
                        continue

                d = ang_diff_2pi(am_ang, at_ang)
                if d < best_delta:
                    best_delta = d
                    best_m = (am, bm)
                    best_t = (at, bt)

        # Dessin (couleur orange, un peu plus épais)
        def draw_edge(P, a, b, color="#FF7F0E"):
            x1,y1 = self._world_to_screen(P[a]); x2,y2 = self._world_to_screen(P[b])
            return self.canvas.create_line(x1,y1,x2,y2, fill=color, width=3)

        if best_m and best_t:
            self._edge_highlight_ids.append(draw_edge(Pm, *best_m, color="#FF7F0E"))
            self._edge_highlight_ids.append(draw_edge(Pt, *best_t, color="#FF7F0E"))
            for _id in self._edge_highlight_ids:
                try: self.canvas.tag_raise(_id)
                except Exception: pass
            # Mémoriser le couple retenu (servira au "collage" à la fin du drag)
            self._edge_choice = (i_mob, key_mob, best_m, i_tgt, key_tgt, best_t)

    def _clear_edge_highlights(self):
        if self._edge_highlight_ids:
            for _id in self._edge_highlight_ids:
                try: self.canvas.delete(_id)
                except Exception: pass
        self._edge_highlight_ids = []
        self._edge_choice = None

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
            # rollback si snapshot disponible
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
        # conversion écran -> monde pour le test d'intérieur
        wx = (x - self.offset[0]) / self.zoom
        wy = (self.offset[1] - y) / self.zoom
        Pw = np.array([wx, wy], dtype=float)        

        # prioriser visuellement le dessus
        for idx in reversed(range(len(self._last_drawn))):
            t = self._last_drawn[idx]
            P = t["pts"]

            # sommets
            for k in ("O", "B", "L"):
                sx, sy = self._world_to_screen(P[k])
                if (sx - x) ** 2 + (sy - y) ** 2 <= tol2:
                    return ("vertex", idx, k)
            # intérieur du triangle (hors disques sommets)
            if self._point_in_triangle_world(Pw, P["O"], P["B"], P["L"]):
                # Exclure une zone autour des sommets (rayon des marqueurs)
                excl_r = (self._marker_px + 2.0) ** 2  # léger coussin
                for k in ("O", "B", "L"):
                    vx, vy = self._world_to_screen(P[k])
                    if (vx - x) ** 2 + (vy - y) ** 2 <= excl_r:
                        break
                else:
                    return ("center", idx, None)
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
        P = self._last_drawn[idx]["pts"]
        pivot = self._tri_centroid(P)
        # snapshot pour rollback
        orig_pts = {k: np.array(P[k].copy()) for k in ("O","B","L")}
        # angle de départ = angle (pivot -> curseur au clic droit)
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
        # priorité au drag & drop depuis la liste
        if self._drag:
            return
        mode, idx, extra = self._hit_test(event.x, event.y)
        if mode == "center":
            P = self._last_drawn[idx]["pts"]
            Cw = self._tri_centroid(P)
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            # snapshot pour rollback
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
                (j, tgt_key, w) = tgt
                self._update_nearest_line(v_world, exclude_idx=idx)
                self._update_edge_highlights(idx, vkey, j, tgt_key)
            self.status.config(text=f"Déplacement par sommet {vkey}.")
        else:
            # clic ailleurs : pan au clic gauche
            self._on_pan_start(event)

    def _on_canvas_left_move(self, event):
        if self._drag:
            return  # le drag liste gère déjà le mouvement
        if not self._sel:
            self._on_pan_move(event)
            return
        if self._sel["mode"] == "move":
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
            # Redessine d'abord les triangles (efface tout), puis recrée le trait d'aide
            self._redraw_from(self._last_drawn)
            v_world = np.array(P[vkey], dtype=float)
            tgt = self._find_nearest_vertex(v_world, exclude_idx=idx)
            if tgt is not None:
                (j, tgt_key, w) = tgt
                self._update_nearest_line(v_world, exclude_idx=idx)
                self._update_edge_highlights(idx, vkey, j, tgt_key)
        elif self._sel["mode"] == "rotate":
            # désormais la rotation est gérée dans <Motion> (pas besoin de maintenir le clic)
            pass

    def _on_canvas_left_up(self, event):
        if self._drag:
            # Si aucun mouvement n'a été reçu, on calcule la position de dépôt maintenant
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
            # Déposer puis nettoyer l'état de drag
            self._place_dragged_triangle()
            self._cancel_drag()
            return
        if self._sel:
            if self._sel.get("mode") == "rotate":
                # validation de la rotation au clic gauche
                self._sel = None
                self.status.config(text="Rotation validée.")
            elif self._sel.get("mode") == "vertex":
                # --- COLLAGE EFFECTIF SI ON A UN COUPLE D'ARÊTES CHOISI ---
                try:
                    choice = self._edge_choice
                    idx   = self._sel.get("idx")
                    vkey  = self._sel.get("vkey")
                    if choice and idx is not None and choice[0] == idx and choice[1] == vkey:
                        i_mob, key_mob, edge_m, i_tgt, key_tgt, edge_t = choice
                        Pm = self._last_drawn[i_mob]["pts"]
                        Pt = self._last_drawn[i_tgt]["pts"]
                        # 1) Translation : faire coïncider les sommets reliés (key_mob -> key_tgt)
                        vm = np.array(Pm[key_mob], dtype=float)
                        vt = np.array(Pt[key_tgt], dtype=float)
                        T  = vt - vm
                        for k in ("O","B","L"):
                            Pm[k] = np.array(Pm[k], dtype=float) + T
                        # 2) Rotation autour du sommet commun pour coller les arêtes (opposées)
                        #    mobile : key_mob -> other_m, cible : key_tgt -> other_t
                        other_m = edge_m[1]
                        other_t = edge_t[1]
                        vm = np.array(Pm[key_mob], dtype=float)      # (mis à jour après translation)
                        bm = np.array(Pm[other_m], dtype=float)
                        vt = np.array(Pt[key_tgt], dtype=float)
                        bt = np.array(Pt[other_t], dtype=float)
                        v_m = bm - vm
                        v_t = bt - vt
                        ang_m = math.atan2(float(v_m[1]), float(v_m[0]))
                        ang_t = math.atan2(float(v_t[1]), float(v_t[0]))
                        # on veut que v_m s'aligne sur v_t en SENS OPPOSÉ (partage d'arête)
                        dtheta = (ang_t - ang_m)

                        # (optionnel) normaliser dans (-π, π] pour éviter une grande rotation
                        while dtheta <= -math.pi: dtheta += 2*math.pi
                        while dtheta >   math.pi: dtheta -= 2*math.pi
                        c, s = math.cos(dtheta), math.sin(dtheta)
                        R = np.array([[c, -s],[s, c]], dtype=float)
                        for k in ("O","B","L"):
                            p = np.array(Pm[k], dtype=float)
                            Pm[k] = (R @ (p - vm)) + vm
                        # Redessin + message
                        self._redraw_from(self._last_drawn)
                        self.status.config(text="Triangles collés (sommets et arêtes alignés).")
                except Exception:
                    # en cas de souci, on termine simplement la sélection
                    pass
                self._sel = None
            else:
                self._sel = None
                self.status.config(text="Sélection terminée.")
            # effacer le trait de proximité s'il existe
            self._clear_nearest_line()
            self._clear_edge_highlights()
        else:
            self._on_pan_end(event)

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
