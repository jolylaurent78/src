"""
TriangleViewerBackgroundMapMixin

Ce module est généré pour découper assembleur_tk.py.
"""

from __future__ import annotations
import os
import re
import json
import io
import datetime as _dt
import numpy as np
from tkinter import messagebox, filedialog


# --- Dépendances optionnelles (Pillow / SVG rendering / calibration) ---
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

try:
    from svglib.svglib import svg2rlg
except Exception:
    svg2rlg = None

try:
    from reportlab.graphics import renderPDF
except Exception:
    renderPDF = None

try:
    from reportlab.pdfgen import canvas as _rl_canvas
except Exception:
    _rl_canvas = None

try:
    import pypdfium2 as pdfium
except Exception:
    pdfium = None

try:
    from pyproj import Transformer
except Exception:
    Transformer = None


class TriangleViewerBackgroundMapMixin:
    """Mixin: méthodes extraites de assembleur_tk.py."""
    pass

    def _bg_calibrate_start(self):
        """Démarre un calibrage du fond en cliquant 3 points (villes) définis dans un fichier JSON.

        Fichier attendu (dans ../data/maps) :
            <nom_de_la_carte>.calib_points.json
        Exemple :
            {
              "points": [
                {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
                {"name": "Lyon",  "lat": 45.7640, "lon": 4.8357},
                {"name": "Marseille", "lat": 43.2965, "lon": 5.3698}
              ]
            }

        Résultat sauvegardé (dans ../data/maps) :
            <nom_de_la_carte>.json
        """
        if not self._bg or not self._bg.get("path"):
            messagebox.showwarning("Calibration", "Aucun fond SVG chargé. Charge d'abord une carte.")
            return

        svg_path = str(self._bg.get("path"))
        base = os.path.splitext(os.path.basename(svg_path))[0]
        cfg_path = os.path.join(self.maps_dir, f"{base}.calib_points.json")

        if not os.path.isfile(cfg_path):
            messagebox.showerror(
                "Calibration",
                "Fichier de points de calibration introuvable.\n\n"
                f"Attendu :\n  {cfg_path}\n\n"
                "Crée ce fichier avec 3 villes (name/lat/lon) puis relance."
            )
            return

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            messagebox.showerror("Calibration", f"Impossible de lire {cfg_path}\n\n{e}")
            return

        pts = cfg.get("points") if isinstance(cfg, dict) else None
        if not isinstance(pts, list) or len(pts) != 3:
            messagebox.showerror(
                "Calibration",
                "Le fichier de calibration doit contenir exactement 3 points :\n"
                '  {"points":[{"name":..,"lat":..,"lon":..}, ... x3]}'
            )
            return

        for i, p in enumerate(pts):
            if not isinstance(p, dict) or ("lat" not in p) or ("lon" not in p):
                messagebox.showerror(
                    "Calibration",
                    f"Point #{i+1} invalide : chaque point doit contenir au moins 'lat' et 'lon'."
                )
                return

        self._bg_calib_points_cfg = {"path": cfg_path, "points": pts, "svg": svg_path}
        self._bg_calib_clicked_world = []
        self._bg_calib_step = 0
        self._bg_calib_active = True

        name0 = str(pts[0].get("name") or "Point 1")
        self.status.config(
            text=(
                f"Calibration carte : CTRL+clic sur {name0} (1/3) | "
                "clic/drag = pan | molette = zoom | ESC = annuler"
            )
        )
        self.canvas.configure(cursor="crosshair")

    def _bg_calibrate_cancel(self):
        if not getattr(self, "_bg_calib_active", False):
            return
        self._bg_calib_active = False
        self._bg_calib_points_cfg = None
        self._bg_calib_clicked_world = []
        self._bg_calib_step = 0
        self.canvas.configure(cursor="")
        self.status.config(text="Calibration annulée.")

    def _bg_calibrate_handle_click(self, event):
        """Enregistre un clic de calibration (en coordonnées monde)."""
        if not getattr(self, "_bg_calib_active", False):
            return "continue"

        # Pendant la calibration, on veut pouvoir pan/zoom pour viser précisément.
        # On valide donc un point UNIQUEMENT sur CTRL+clic.
        # Sans CTRL, le clic gauche redevient un "pan start".
        if not getattr(self, "_ctrl_down", False):
            self._on_pan_start(event)
            return "break"

        cfg = self._bg_calib_points_cfg or {}
        pts = cfg.get("points") or []
        if len(pts) != 3:
            self._bg_calibrate_cancel()
            return "break"

        w = self._screen_to_world(event.x, event.y)

        self._bg_calib_clicked_world.append((float(w[0]), float(w[1])))
        self._bg_calib_step += 1

        if self._bg_calib_step < 3:
            nxt = pts[self._bg_calib_step]
            name = str(nxt.get("name") or f"Point {self._bg_calib_step+1}")
            self.status.config(text=f"Calibration carte : clique sur {name} ({self._bg_calib_step+1}/3) — ESC pour annuler.")
            # petit feedback visuel
            self._redraw_from(self._last_drawn)
            return "break"

        # 3 clics : calcul + sauvegarde
        try:
            self._bg_calibrate_finish()
        except Exception as e:
            messagebox.showerror("Calibration", f"Échec du calibrage :\n\n{e}")
            self._bg_calibrate_cancel()
            return "break"

        self._bg_calib_active = False
        self.canvas.configure(cursor="")

        self._redraw_from(self._last_drawn)
        return "break"

    def _bg_calibrate_finish(self):
        cfg = self._bg_calib_points_cfg or {}
        pts = cfg.get("points") or []
        svg_path = cfg.get("svg") or (self._bg.get("path") if self._bg else None)
        if len(pts) != 3 or len(self._bg_calib_clicked_world) != 3 or not svg_path:
            raise RuntimeError("Données de calibration incomplètes.")

        # GPS -> Lambert93 (m) -> km
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
        lambert_km = []
        for p in pts:
            lon = float(p["lon"])
            lat = float(p["lat"])
            x_m, y_m = transformer.transform(lon, lat)
            lambert_km.append((x_m / 1000.0, y_m / 1000.0))

        world = self._bg_calib_clicked_world

        # Résolution affine : (xw,yw) -> (xl,yl)
        # xl = a*xw + b*yw + c
        # yl = d*xw + e*yw + f
        A = []
        B = []
        for (xw, yw), (xl, yl) in zip(world, lambert_km):
            A.append([xw, yw, 1, 0, 0, 0])
            A.append([0, 0, 0, xw, yw, 1])
            B.append(xl)
            B.append(yl)

        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        params = np.linalg.solve(A, B)  # 6 paramètres exacts (3 points)
        a, b, c, d, e, f = [float(v) for v in params.tolist()]

        det = a * e - b * d
        if abs(det) < 1e-12:
            raise RuntimeError("Points de calibration dégénérés (transformée non inversible).")

        # inverse (Lambert km -> monde)
        inv_a = e / det
        inv_b = -b / det
        inv_d = -d / det
        inv_e = a / det
        inv_c = -(inv_a * c + inv_b * f)
        inv_f = -(inv_d * c + inv_e * f)

        # Sauvegarde
        base = os.path.splitext(os.path.basename(str(svg_path)))[0]
        out_path = os.path.join(self.maps_dir, f"{base}.json")

        payload = {
            "type": "bg_calibration_3points",
            "date": _dt.datetime.now().isoformat(timespec="seconds"),
            "svgPath": str(svg_path),
            "bgWorldRectAtCalibration": (
                {
                    "x0": float(self._bg.get("x0")),
                    "y0": float(self._bg.get("y0")),
                    "w":  float(self._bg.get("w")),
                    "h":  float(self._bg.get("h")),
                }
                if (self._bg is not None and all(k in self._bg for k in ("x0", "y0", "w", "h")))
                else None
            ),
            "points": [
                {
                    "name": str(p.get("name") or f"Point {i+1}"),
                    "lat": float(p["lat"]),
                    "lon": float(p["lon"]),
                    "lambert93Km": [lambert_km[i][0], lambert_km[i][1]],
                    "worldClicked": [world[i][0], world[i][1]],
                }
                for i, p in enumerate(pts)
            ],
            "affineWorldToLambertKm": [a, b, c, d, e, f],
            "affineLambertKmToWorld": [inv_a, inv_b, inv_c, inv_d, inv_e, inv_f],
        }

        os.makedirs(self.maps_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(payload, f_out, ensure_ascii=False, indent=2)

        self.status.config(text=f"Calibration enregistrée : {out_path}")

    # =========================
    # Fond SVG en coordonnées monde
    # =========================

    def _bg_clear(self, persist: bool = True):
        self._bg = None
        self._bg_base_pil = None
        self._bg_photo = None
        self._bg_resizing = None
        self._bg_calib_data = None
        self._bg_scale_base_w = None

        if persist:
            self._persistBackgroundConfig()
        self._redraw_from(self._last_drawn)

    def _bg_try_load_calibration(self, svg_path: str):
        """Charge data/maps/<carte>.json si présent (calibration 3 points).
        Objectif: disposer de affineLambertKmToWorld pour convertir (Lambert93 km) -> coordonnées monde.
        """
        try:
            base = os.path.splitext(os.path.basename(str(svg_path)))[0]
            calib_path = os.path.join(self.maps_dir, f"{base}.json")
            if not os.path.isfile(calib_path):
                self._bg_calib_data = None
                self._bg_scale_base_w = None
                return

            with open(calib_path, "r", encoding="utf-8") as f_in:
                data = json.load(f_in)

            aff = data.get("affineLambertKmToWorld")
            if not (isinstance(aff, list) and len(aff) == 6):
                self._bg_calib_data = None
                return

            # Normaliser en float
            data["affineLambertKmToWorld"] = [float(v) for v in aff]

            # Optionnel mais recommandé: inverse monde -> Lambert93 (km)
            aff_inv = data.get("affineWorldToLambertKm")
            if isinstance(aff_inv, list):
                if len(aff_inv) != 6:
                    raise ValueError("affineWorldToLambertKm doit contenir 6 valeurs")
                data["affineWorldToLambertKm"] = [float(v) for v in aff_inv]

            self._bg_calib_data = data

            # Référence d'échelle pour l'affichage :
            # - si le JSON contient la géométrie monde du fond au moment de la calibration,
            #   on s'y réfère (=> le "x1" est stable et cohérent entre sessions)
            # - sinon, fallback : largeur actuelle au moment du chargement.

            rect = data.get("bgWorldRectAtCalibration") if isinstance(data, dict) else None
            if isinstance(rect, dict) and ("w" in rect):
                self._bg_scale_base_w = float(rect.get("w"))
            elif self._bg is not None:
                self._bg_scale_base_w = float(self._bg.get("w"))
            else:
                self._bg_scale_base_w = None

        except Exception:
            # best-effort
            self._bg_calib_data = None
            self._bg_scale_base_w = None

    def _bg_load_svg_dialog(self):
        path = filedialog.askopenfilename(
            title="Choisir une carte (SVG/PNG/JPG)",
            initialdir=getattr(self, "maps_dir", None) or None,
            filetypes=[
                ("Cartes", "*.svg *.png *.jpg *.jpeg"),
                ("SVG", "*.svg"),
                ("PNG", "*.png"),
                ("JPG", "*.jpg *.jpeg"),
                ("Tous fichiers", "*.*"),
            ],
        )
        if not path:
            return
        self._bg_set_map(path)

    def _bg_set_map(self, path: str, rect_override: dict | None = None, persist: bool = True):
        """Charge une carte (fond) depuis un fichier .svg ou une image raster (.png/.jpg/.jpeg).

        - SVG : rasterisation (svglib/reportlab/pypdfium2) comme avant
        - PNG/JPG/JPEG : chargement direct via Pillow

        Note : on conserve les clés de config 'bgSvgPath/bgWorldRect' pour compatibilité.
        """
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".svg":
            return self._bg_set_svg(path, rect_override=rect_override, persist=persist)
        if ext in (".png", ".jpg", ".jpeg"):
            return self._bg_set_png(path, rect_override=rect_override, persist=persist)

        messagebox.showerror(
            "Format non supporté",
            f"Carte non supportée: {path}\nFormats acceptés: .svg, .png, .jpg, .jpeg"
        )

    def _bg_set_png(self, png_path: str, rect_override: dict | None = None, persist: bool = True):
        if (Image is None or ImageTk is None):
            messagebox.showerror(
                "Dépendances manquantes",
                "Pour afficher un fond PNG, installe :\n"
                "  - pillow\n\n"
                "pip install pillow"
            )
            return

        try:
            pil0 = Image.open(png_path).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Erreur image", f"Impossible de charger l'image :\n{e}")
            return

        # aspect depuis l'image
        w0, h0 = pil0.size
        aspect = float(w0) / float(h0) if h0 else 1.0
        aspect = max(1e-6, aspect)

        # base normalisée : max 4096 sur le plus grand côté (comme SVG)
        max_dim = 4096
        w0, h0 = pil0.size
        if w0 <= 0 or h0 <= 0:
            raise ValueError("image vide")
        if max(w0, h0) > max_dim:
            if w0 >= h0:
                W = int(max_dim)
                H = max(1, int(round(W / aspect)))
            else:
                H = int(max_dim)
                W = max(1, int(round(H * aspect)))
            pil0 = pil0.resize((W, H), Image.LANCZOS)

        # Si on a une géométrie sauvegardée (monde), on la réapplique telle quelle
        if isinstance(rect_override, dict) and all(k in rect_override for k in ("x0", "y0", "w", "h")):
            x0 = float(rect_override.get("x0", 0.0))
            y0 = float(rect_override.get("y0", 0.0))
            w = float(rect_override.get("w", 1.0))
            h = float(rect_override.get("h", 1.0))
            if w > 1e-9 and h > 1e-9:
                self._bg = {"path": png_path, "x0": x0, "y0": y0, "w": w, "h": h, "aspect": aspect}
                self._bg_base_pil = pil0
                self._bg_try_load_calibration(png_path)
                if self._bg_calib_data is not None:
                    print(f"[BG] Calibration chargée : {os.path.splitext(os.path.basename(str(png_path)))[0]}.json")

                if persist:
                    self._persistBackgroundConfig()

                self._redraw_from(self._last_drawn)

                if persist:
                    print(f"[Fond image] Fichier chargé: {os.path.basename(str(png_path))} ({png_path})")
                return

        # Position/taille initiale : calée sur bbox des triangles si dispo, sinon vue écran
        if self._last_drawn:
            xs, ys = [], []
            for t in self._last_drawn:
                P = t["pts"]
                for k in ("O", "B", "L"):
                    xs.append(float(P[k][0]))
                    ys.append(float(P[k][1]))
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
        else:
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

        # Ajuster pour conserver le ratio
        if (bw / bh) > aspect:
            w = bw
            h = w / aspect
        else:
            h = bh
            w = h * aspect

        self._bg = {"path": png_path, "x0": cx - w/2, "y0": cy - h/2, "w": w, "h": h, "aspect": aspect}
        self._bg_base_pil = pil0

        self._bg_try_load_calibration(png_path)
        if self._bg_calib_data is not None:
            print(f"[BG] Calibration chargée : {os.path.splitext(os.path.basename(str(png_path)))[0]}.json")

        if persist:
            self._persistBackgroundConfig()
            print(f"[Fond PNG] Fichier chargé: {os.path.basename(str(png_path))} ({png_path})")

        self._redraw_from(self._last_drawn)

    def _bg_set_svg(self, svg_path: str, rect_override: dict | None = None, persist: bool = True):
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
        # Si on a une géométrie sauvegardée (monde), on la réapplique telle quelle
        if isinstance(rect_override, dict) and all(k in rect_override for k in ("x0", "y0", "w", "h")):
            x0 = float(rect_override.get("x0", 0.0))
            y0 = float(rect_override.get("y0", 0.0))
            w = float(rect_override.get("w", 1.0))
            h = float(rect_override.get("h", 1.0))
            if w > 1e-9 and h > 1e-9:
                self._bg = {"path": svg_path, "x0": x0, "y0": y0, "w": w, "h": h, "aspect": aspect}
                # Raster base "normalisée" (taille fixe, ratio respecté)
                self._bg_base_pil = self._bg_render_base(svg_path, aspect, max_dim=4096)
                if self._bg_base_pil is not None:
                    print(f"[Fond SVG] Fichier chargé: {os.path.basename(str(svg_path))} ({svg_path})")
                # calibration associée (si fichier data/<carte>.json existe)
                self._bg_try_load_calibration(svg_path)
                if self._bg_calib_data is not None:
                    print(f"[BG] Calibration chargée : {os.path.basename(str(svg_path))}.json")

                if persist:
                    self._persistBackgroundConfig()
                self._redraw_from(self._last_drawn)
                return

        # Position/taille initiale : calée sur bbox des triangles si dispo, sinon vue écran
        if self._last_drawn:
            xs, ys = [], []
            for t in self._last_drawn:
                P = t["pts"]
                for k in ("O", "B", "L"):
                    xs.append(float(P[k][0]))
                    ys.append(float(P[k][1]))
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

        # calibration associée (si fichier data/<carte>.json existe)
        if self._bg_calib_data is not None:
            print(f"[BG] Calibration chargée : {os.path.basename(str(svg_path))}.json")
        self._bg_try_load_calibration(svg_path)

        if persist:
            self._persistBackgroundConfig()
            if self._bg_base_pil is not None:
                print(f"[Fond SVG] Fichier chargé: {os.path.basename(str(svg_path))} ({svg_path})")

        self._redraw_from(self._last_drawn)

    def _bg_parse_aspect(self, svg_path: str) -> float:
        s = open(svg_path, "r", encoding="utf-8", errors="ignore").read(8192)

        # viewBox="minx miny width height"
        m = re.search(r'viewBox\s*=\s*["\']\s*([-\d\.eE]+)\s+([-\d\.eE]+)\s+([-\d\.eE]+)\s+([-\d\.eE]+)\s*["\']', s)
        if m:
            vw = float(m.group(3))
            vh = float(m.group(4))
            if abs(vh) > 1e-12:
                return max(1e-6, vw / vh)

        # width="xxx" height="yyy" (units possible)
        mw = re.search(r'width\s*=\s*["\']\s*([-\d\.eE]+)', s)
        mh = re.search(r'height\s*=\s*["\']\s*([-\d\.eE]+)', s)
        if mw and mh:
            w = float(mw.group(1))
            h = float(mh.group(1))
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
            if hasattr(drawing, "getBounds"):
                b = drawing.getBounds()
                if b and len(b) == 4:
                    xmin, ymin, xmax, ymax = map(float, b)
                    bw = max(0.0, xmax - xmin)
                    bh = max(0.0, ymax - ymin)

            # Dimensions de référence : d'abord le bounding-box, sinon width/height svglib
            dw = float(bw) if bw > 1e-9 else float(getattr(drawing, "width", 0) or 0)
            dh = float(bh) if bh > 1e-9 else float(getattr(drawing, "height", 0) or 0)
            if dw <= 0 or dh <= 0:
                # fallback: utiliser le ratio calculé, avec une base arbitraire
                dw = float(W)
                dh = float(H)

            # Si le contenu est décalé (xmin/ymin), on le ramène à l'origine avant le scale.
            if bw > 1e-9 and bh > 1e-9:
                if hasattr(drawing, "translate"):
                    drawing.translate(-xmin, -ymin)

            # svglib exprime en "points" (1/72 inch). En rasterisant à 72dpi,
            # 1 point ~= 1 pixel. On scale le dessin puis on force le canvas aux dims voulues.
            sx = float(W) / dw
            sy = float(H) / dh

            # On applique les deux échelles pour respecter EXACTEMENT W/H (le ratio vient déjà de 'aspect')
            drawing.scale(sx, sy)
            drawing.width = float(W)
            drawing.height = float(H)

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
            page = doc[0]
            bitmap = page.render(scale=1)  # ~72 dpi : 1 point ≈ 1 pixel
            pil = bitmap.to_pil().convert("RGBA")
            doc.close()

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
            self.update_idletasks()
            cw = int(self.canvas.winfo_width() or 0)
            ch = int(self.canvas.winfo_height() or 0)

        if cw <= 2 or ch <= 2:
            return

        bx0 = float(self._bg["x0"])
        by0 = float(self._bg["y0"])
        bw = float(self._bg["w"])
        bh = float(self._bg["h"])
        bx1 = bx0 + bw
        by1 = by0 + bh

        # Vue monde actuelle (canvas entier)
        xA, yTop = self._screen_to_world(0, 0)
        xB, yBot = self._screen_to_world(cw, ch)
        vx0, vx1 = (min(xA, xB), max(xA, xB))
        vy0, vy1 = (min(yBot, yTop), max(yBot, yTop))

        # Intersection vue <-> fond
        ix0 = max(vx0, bx0)
        ix1 = min(vx1, bx1)
        iy0 = max(vy0, by0)
        iy1 = min(vy1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            return

        baseW, baseH = self._bg_base_pil.size

        # Crop dans l'image base
        left = int((ix0 - bx0) / bw * baseW)
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

        # IMPORTANT: fond blanc pour éviter un rendu gris quand la carte est semi-transparente
        # (Tk peut composer les pixels transparents sur un fond non-blanc selon la plateforme).
        out = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
        px = int(round(sx0))
        py = int(round(syTop))

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
        # Appliquer l'opacité utilisateur (0..100) sur l'alpha du fond
        op = int(float(self.map_opacity.get()))

        op = max(0, min(100, op))
        if op <= 0:
            return
        if op < 100:
            if crop2.mode != "RGBA":
                crop2 = crop2.convert("RGBA")
            r, g, b, a = crop2.split()
            a = a.point(lambda p: int(p * op / 100))
            crop2.putalpha(a)
        out.paste(crop2, (paste_x0, paste_y0), crop2)

        self._bg_photo = ImageTk.PhotoImage(out)
        self.canvas.create_image(0, 0, anchor="nw", image=self._bg_photo, tags=("bg_world",))
        self.canvas.tag_lower("bg_world")

    def _bg_corners_world(self):
        if not self._bg:
            return None
        x0 = float(self._bg["x0"])
        y0 = float(self._bg["y0"])
        w = float(self._bg["w"])
        h = float(self._bg["h"])
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
        tl = c["tl"]
        br = c["br"]

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

    def _bg_start_move(self, sx: int, sy: int):
        """Démarre un déplacement du fond (mode resize actif mais pas sur une poignée)."""
        if not self._bg:
            return
        mx, my = self._screen_to_world(sx, sy)
        self._bg_moving = {
            "start_mouse": (float(mx), float(my)),
            "start_xy": (float(self._bg.get("x0", 0.0)), float(self._bg.get("y0", 0.0))),
        }

    def _bg_update_move(self, sx: int, sy: int):
        if not getattr(self, "_bg_moving", None) or not self._bg:
            return
        mx, my = self._screen_to_world(sx, sy)
        smx, smy = self._bg_moving["start_mouse"]
        x0, y0 = self._bg_moving["start_xy"]
        dx = float(mx - smx)
        dy = float(my - smy)
        self._bg["x0"] = float(x0 + dx)
        self._bg["y0"] = float(y0 + dy)

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
        self._bg["w"] = float(w)
        self._bg["h"] = float(h)

        # Afficher l'échelle relative (x1, x1/3.15, x2.49...) pendant le resize
        self._bg_update_scale_status()

    def _bg_compute_scale_factor(self) -> float | None:
        """Retourne le scale *carte vs triangles*.

        On veut comparer :
          - l'échelle "monde" des triangles (1 unité monde == 1 km)
          - l'échelle "monde" implicite de la carte via la calibration 3 points.

        La calibration fournit affineLambertKmToWorld (km -> monde) :
            xw = a*xkm + b*ykm + c
            yw = d*xkm + e*ykm + f

        Les colonnes (a,d) et (b,e) donnent la taille en unités monde pour 1 km (Est / Nord).
        On moyenne les deux normes pour obtenir un facteur global.

        Si le fond a été redimensionné depuis la calibration, on applique le ratio (w_cur / w_ref).
        """
        if not self._bg:
            return None

        data = getattr(self, "_bg_calib_data", None)
        if not (isinstance(data, dict) and "affineLambertKmToWorld" in data):
            return getattr(self, "_bg_scale_factor_override", None)

        aff = data.get("affineLambertKmToWorld")
        if not (isinstance(aff, list) and len(aff) == 6):
            return getattr(self, "_bg_scale_factor_override", None)

        a, b, c, d, e, f = [float(v) for v in aff]

        # Taille en unités monde pour 1 km (axes Est / Nord)
        import math
        norm_e = math.hypot(a, d)
        norm_n = math.hypot(b, e)
        base_norm = 0.5 * (norm_e + norm_n)

        if base_norm <= 1e-12:
            return None

        # Ratio de redimensionnement du fond (si on a une référence)
        ratio = 1.0
        if self._bg_scale_base_w is not None:
            base_w = float(self._bg_scale_base_w)
            cur_w = float(self._bg.get("w"))
            if base_w > 1e-12 and cur_w > 1e-12:
                ratio = cur_w / base_w

        s = base_norm * ratio
        self._bg_scale_factor_override = s
        return s

    def _bg_format_scale(self, s: float | None) -> str:
        if s is None:
            return "x?"
        if abs(s - 1.0) < 1e-3:
            return "x1"
        if s >= 1.0:
            return f"x{s:.2f}"
        # plus petit que la référence -> x1/k
        k = 1.0 / max(1e-12, s)
        return f"x1/{k:.2f}"

    def _bg_update_scale_status(self):
        # On n'affiche l'échelle que si le mode redimensionnement est actif.
        if not self.bg_resize_mode.get() or not self._bg:
            return
        s = self._bg_compute_scale_factor()
        self.status.config(text=f"Échelle carte : {self._bg_format_scale(s)}")
