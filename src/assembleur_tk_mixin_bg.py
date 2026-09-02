"""Renderer raster et interactions de la carte de scénario Tk."""

from __future__ import annotations

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - dépendance applicative optionnelle
    Image = None
    ImageTk = None


class TriangleViewerBackgroundMapMixin:
    """Projection graphique ``_bg`` de la carte de scénario résolue.

    Le modèle métier, le choix de carte et sa calibration appartiennent au
    catalogue et à ``ScenarioMapState``. Ce mixin ne contient que le rendu et
    l'interaction canvas de la projection dérivée.
    """

    def _bg_draw_world_layer(self):
        """Dessine la carte raster dans la vue monde courante."""
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

        x_a, y_top = self._screen_to_world(0, 0)
        x_b, y_bottom = self._screen_to_world(cw, ch)
        vx0, vx1 = min(x_a, x_b), max(x_a, x_b)
        vy0, vy1 = min(y_bottom, y_top), max(y_bottom, y_top)
        ix0 = max(vx0, bx0)
        ix1 = min(vx1, bx1)
        iy0 = max(vy0, by0)
        iy1 = min(vy1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            return

        base_width, base_height = self._bg_base_pil.size
        left = int((ix0 - bx0) / bw * base_width)
        right = int((ix1 - bx0) / bw * base_width)
        upper = int((by1 - iy1) / bh * base_height)
        lower = int((by1 - iy0) / bh * base_height)
        left = max(0, min(base_width - 1, left))
        right = max(left + 1, min(base_width, right))
        upper = max(0, min(base_height - 1, upper))
        lower = max(upper + 1, min(base_height, lower))
        crop = self._bg_base_pil.crop((left, upper, right, lower))

        sx0, sy_top = self._world_to_screen((ix0, iy1))
        sx1, sy_bottom = self._world_to_screen((ix1, iy0))
        width_px = int(round(sx1 - sx0))
        height_px = int(round(sy_bottom - sy_top))
        if width_px <= 1 or height_px <= 1:
            return
        crop = crop.resize((width_px, height_px), Image.LANCZOS)

        out = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
        px = int(round(sx0))
        py = int(round(sy_top))
        paste_x0 = max(0, px)
        paste_y0 = max(0, py)
        paste_x1 = min(cw, px + width_px)
        paste_y1 = min(ch, py + height_px)
        if paste_x1 <= paste_x0 or paste_y1 <= paste_y0:
            return

        src_x0 = paste_x0 - px
        src_y0 = paste_y0 - py
        src_x1 = src_x0 + (paste_x1 - paste_x0)
        src_y1 = src_y0 + (paste_y1 - paste_y0)
        crop = crop.crop((src_x0, src_y0, src_x1, src_y1))

        opacity = max(0, min(100, int(float(self.map_opacity.get()))))
        if opacity <= 0:
            return
        if opacity < 100:
            if crop.mode != "RGBA":
                crop = crop.convert("RGBA")
            _red, _green, _blue, alpha = crop.split()
            crop.putalpha(alpha.point(lambda value: int(value * opacity / 100)))
        out.paste(crop, (paste_x0, paste_y0), crop)

        self._bg_photo = ImageTk.PhotoImage(out)
        self.canvas.create_image(0, 0, anchor="nw", image=self._bg_photo, tags=("bg_world",))
        self.canvas.tag_lower("bg_world")

    def _bg_corners_world(self):
        if not self._bg:
            return None
        x0 = float(self._bg["x0"])
        y0 = float(self._bg["y0"])
        width = float(self._bg["w"])
        height = float(self._bg["h"])
        return {
            "bl": (x0, y0), "br": (x0 + width, y0),
            "tl": (x0, y0 + height), "tr": (x0 + width, y0 + height),
        }

    def _bg_corners_screen(self):
        corners = self._bg_corners_world()
        if not corners:
            return None
        return {key: self._world_to_screen(value) for key, value in corners.items()}

    def _bg_draw_resize_handles(self):
        if not self._bg or not self.bg_resize_mode.get():
            return
        corners = self._bg_corners_screen()
        if not corners:
            return
        top_left = corners["tl"]
        bottom_right = corners["br"]
        self.canvas.create_rectangle(
            top_left[0], top_left[1], bottom_right[0], bottom_right[1],
            outline="gray30", dash=(3, 2), width=1, tags=("bg_ui",),
        )
        radius = 6
        for key in ("tl", "tr", "bl", "br"):
            x, y = corners[key]
            self.canvas.create_rectangle(
                x - radius, y - radius, x + radius, y + radius,
                outline="gray10", fill="white", width=1, tags=("bg_ui",),
            )

    def _bg_hit_test_handle(self, sx: float, sy: float):
        corners = self._bg_corners_screen()
        if not corners:
            return None
        radius = 8
        for key in ("tl", "tr", "bl", "br"):
            x, y = corners[key]
            if (sx - x) ** 2 + (sy - y) ** 2 <= radius ** 2:
                return key
        return None

    def _bg_start_resize(self, handle: str, sx: int, sy: int):
        opposite = {"tl": "br", "br": "tl", "tr": "bl", "bl": "tr"}[handle]
        corners = self._bg_corners_world()
        fixed_x, fixed_y = corners[opposite]
        mouse_x, mouse_y = self._screen_to_world(sx, sy)
        self._bg_resizing = {
            "handle": handle,
            "fixed": (fixed_x, fixed_y),
            "start_mouse": (mouse_x, mouse_y),
            "start_rect": (
                float(self._bg["x0"]), float(self._bg["y0"]),
                float(self._bg["w"]), float(self._bg["h"]),
            ),
        }

    def _bg_start_move(self, sx: int, sy: int):
        if not self._bg:
            return
        mouse_x, mouse_y = self._screen_to_world(sx, sy)
        self._bg_moving = {
            "start_mouse": (float(mouse_x), float(mouse_y)),
            "start_xy": (float(self._bg["x0"]), float(self._bg["y0"])),
        }

    def _bg_update_move(self, sx: int, sy: int):
        if not self._bg_moving or not self._bg:
            return
        mouse_x, mouse_y = self._screen_to_world(sx, sy)
        start_x, start_y = self._bg_moving["start_mouse"]
        x0, y0 = self._bg_moving["start_xy"]
        self._bg["x0"] = float(x0 + mouse_x - start_x)
        self._bg["y0"] = float(y0 + mouse_y - start_y)

    def _bg_update_resize(self, sx: int, sy: int):
        if not self._bg_resizing or not self._bg:
            return
        aspect = float(self._bg["aspect"])
        fixed_x, fixed_y = self._bg_resizing["fixed"]
        mouse_x, mouse_y = self._screen_to_world(sx, sy)
        delta_x = mouse_x - fixed_x
        delta_y = mouse_y - fixed_y
        raw_width = abs(delta_x)
        raw_height = abs(delta_y)
        if raw_width < 1e-6 or raw_height < 1e-6:
            return
        if raw_width / raw_height > aspect:
            width = raw_width
            height = width / aspect
        else:
            height = raw_height
            width = height * aspect
        width = max(1e-3, width)
        height = max(1e-3, height)
        self._bg["x0"] = float(fixed_x if delta_x >= 0 else fixed_x - width)
        self._bg["y0"] = float(fixed_y if delta_y >= 0 else fixed_y - height)
        self._bg["w"] = float(width)
        self._bg["h"] = float(height)
        self._bg_update_scale_status()

    def _bg_format_scale(self, scale: float | None) -> str:
        if scale is None:
            return "x?"
        if abs(scale - 1.0) < 1e-3:
            return "x1"
        if scale >= 1.0:
            return f"x{scale:.2f}"
        return f"x1/{1.0 / max(1e-12, scale):.2f}"

    def _bg_update_scale_status(self):
        if not self.bg_resize_mode.get() or not self._bg:
            return
        scale = self._bg_compute_scale_factor()
        self.status.config(text=f"Échelle carte : {self._bg_format_scale(scale)}")
