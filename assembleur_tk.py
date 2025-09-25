
import os
import math
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

        # état IHM
        self.triangle_file = tk.StringVar(value="")
        self.start_index = tk.IntVar(value=1)
        self.num_triangles = tk.IntVar(value=8)

        self.excel_path = None
        self.df = None
        self._last_drawn = []   # liste d'items: (labels, P_world)

        self._build_ui()

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

        # Pan avec clic milieu (préserve le clic gauche pour le drop)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        # Compat : si pas en drag, on autorise le pan au clic gauche
        self.canvas.bind("<ButtonPress-1>", self._on_left_press_canvas)
        self.canvas.bind("<B1-Motion>", self._on_left_motion_canvas)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release_canvas)
        # Suivi du fantôme pendant un drag
        self.canvas.bind("<Motion>", self._on_canvas_motion_update_drag)


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
                "pts": P
            })
        return out

    # ---------- Mise en page simple (aperçu brut) ----------
    def _triangle_from_index(self, idx):
        """Construit un triangle 'local' (coord locales) depuis l'index visuel de la listbox."""
        if getattr(self, "df", None) is None or self.df.empty:
            raise RuntimeError("Pas de données — ouvre d'abord l’Excel.")
        r = self.df.iloc[idx]
        P = _build_local_triangle(float(r["len_OB"]), float(r["len_OL"]), float(r["len_BL"]))
        return {"labels": ("Bourges", str(r["B"]), str(r["L"])), "pts": P}


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
            placed.append({"labels": t["labels"], "pts": Pw})

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
        for t in placed:
            labels = t["labels"]
            P = t["pts"]
            self._draw_triangle_screen(P, labels=[f"O:{labels[0]}", f"B:{labels[1]}", f"L:{labels[2]}"])

        # Fit à l'écran
        self._fit_to_view(placed)
        self.status.config(text=f"{len(placed)} triangle(s) affiché(s) — aperçu brut (sans assemblage)")

    def clear_canvas(self):
        self.canvas.delete("all")
        self._last_drawn = []
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
        for t in placed:
            labels = t["labels"]
            P = t["pts"]
            self._draw_triangle_screen(P, labels=[f"O:{labels[0]}", f"B:{labels[1]}", f"L:{labels[2]}"])

    def _draw_triangle_screen(self, P, outline="black", width=2, labels=None, inset=0.35):
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
        # 2) tracé
        self.canvas.create_polygon(*coords, outline=outline, fill="", width=width)
        # 3) barycentre (monde)
        cx = (P["O"][0] + P["B"][0] + P["L"][0]) / 3.0
        cy = (P["O"][1] + P["B"][1] + P["L"][1]) / 3.0
        # 4) labels
        if labels:
            for pt, txt in zip(pts_world, labels):
                lx = (1.0 - inset) * pt[0] + inset * cx
                ly = (1.0 - inset) * pt[1] + inset * cy
                sx, sy = self._world_to_screen((lx, ly))
                self.canvas.create_text(sx, sy, text=txt, anchor="center", font=("Arial", 8))

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
        self._drag = {"triangle": tri}
        # créer un fantôme minimal dès le départ (sera positionné au mouvement)
        if self._drag_preview_id is not None:
            self.canvas.delete(self._drag_preview_id)
            self._drag_preview_id = None
        # Forcer focus sur le canvas pour recevoir les motions
        self.canvas.focus_set()
        self.status.config(text="Glissez le triangle sur le canvas puis relâchez pour le déposer.")

    def _on_canvas_motion_update_drag(self, event):
        if not self._drag:
            return
        # Position monde sous curseur
        wx = (event.x - self.offset[0]) / self.zoom
        wy = (self.offset[1] - event.y) / self.zoom
        tri = self._drag["triangle"]
        P = tri["pts"]
        # Translate: placer O sur le curseur
        dx = wx - float(P["O"][0])
        dy = wy - float(P["O"][1])
        O = np.array([P["O"][0] + dx, P["O"][1] + dy])
        B = np.array([P["B"][0] + dx, P["B"][1] + dy])
        L = np.array([P["L"][0] + dx, P["L"][1] + dy])
        # Enregistrer la version monde courante (pré-drop)
        self._drag["world_pts"] = {"O": O, "B": B, "L": L}
        # Dessin du fantôme
        coords = []
        for pt in (O, B, L):
            sx, sy = self._world_to_screen(pt)
            coords += [sx, sy]
        if self._drag_preview_id is None:
            self._drag_preview_id = self.canvas.create_polygon(*coords, outline="gray50", dash=(4,2), fill="", width=2)
        else:
            self.canvas.coords(self._drag_preview_id, *coords)

    def _place_dragged_triangle(self):
        if not self._drag or "world_pts" not in self._drag:
            return
        tri = self._drag["triangle"]
        Pw = self._drag["world_pts"]
        # Ajouter au "document courant"
        self._last_drawn.append({"labels": tri["labels"], "pts": Pw})
        self._redraw_from(self._last_drawn)
        self.status.config(text="Triangle déposé.")

    def _cancel_drag(self):
        self._drag = None
        if self._drag_preview_id is not None:
            self.canvas.delete(self._drag_preview_id)
            self._drag_preview_id = None

    # Compat pan au clic gauche si pas en drag
    def _on_left_press_canvas(self, event):
        if self._drag:
            # on ne démarre pas de pan si on est en drag
            return
        self._on_pan_start(event)

    def _on_left_motion_canvas(self, event):
        if self._drag:
            # en drag : le mouvement est géré par _on_canvas_motion_update_drag
            return
        self._on_pan_move(event)

    def _on_left_release_canvas(self, event):
        if self._drag:
            # Fin de drag: déposer si curseur sur le canvas
            self._place_dragged_triangle()
            self._cancel_drag()
            return
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
