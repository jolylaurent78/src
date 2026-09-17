"""Projection graphique et interactions runtime de la carte de scénario."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - optional application dependency
    Image = None
    ImageTk = None


@dataclass(frozen=True)
class BackgroundMapWorldRect:
    x0: float
    y0: float
    w: float
    h: float


def format_scale(scale: float | None) -> str:
    if scale is None:
        return "x?"
    if abs(scale - 1.0) < 1e-3:
        return "x1"
    if scale >= 1.0:
        return f"x{scale:.2f}"
    return f"x1/{1.0 / max(1e-12, scale):.2f}"


class BackgroundMapLayer:
    """Contient le rendu raster et la géométrie temporaire de la carte.

    Le composant ne connaît ni le viewer ni l'état métier du scénario.
    """

    def __init__(
        self,
        world_to_screen: Callable[[tuple[float, float]], tuple[float, float]],
        screen_to_world: Callable[[float, float], tuple[float, float]],
        on_geometry_changed: Callable[[], None],
    ) -> None:
        self._world_to_screen = world_to_screen
        self._screen_to_world = screen_to_world
        self._on_geometry_changed = on_geometry_changed
        self._canvas = None
        self._base_image = None
        self._world_rect: BackgroundMapWorldRect | None = None
        self._asset_path: str | None = None
        self._photo = None
        self._resizing = None
        self._moving = None

    @property
    def base_image(self):
        return self._base_image

    @property
    def world_rect(self) -> BackgroundMapWorldRect | None:
        return self._world_rect

    @property
    def is_resizing(self) -> bool:
        return self._resizing is not None

    @property
    def is_moving(self) -> bool:
        return self._moving is not None

    @property
    def has_map(self) -> bool:
        return self._world_rect is not None and self._base_image is not None

    def attach_canvas(self, canvas) -> None:
        self._canvas = canvas

    def set_map(self, image, world_rect: BackgroundMapWorldRect, asset_path: str | None = None) -> None:
        if world_rect.w <= 0 or world_rect.h <= 0:
            raise ValueError("La carte doit avoir un rectangle monde strictement positif.")
        self._base_image = image
        self._world_rect = world_rect
        self._asset_path = asset_path
        self._photo = None
        self.cancel_interaction()

    def clear(self) -> None:
        self._base_image = None
        self._world_rect = None
        self._asset_path = None
        self._photo = None
        self.cancel_interaction()

    def draw(self, opacity: int) -> None:
        if not self.has_map or self._canvas is None or Image is None or ImageTk is None:
            return
        canvas = self._canvas
        cw, ch = int(canvas.winfo_width() or 0), int(canvas.winfo_height() or 0)
        if cw <= 2 or ch <= 2:
            canvas.update_idletasks()
            cw, ch = int(canvas.winfo_width() or 0), int(canvas.winfo_height() or 0)
        if cw <= 2 or ch <= 2:
            return
        rect = self._world_rect
        x_a, y_top = self._screen_to_world(0, 0)
        x_b, y_bottom = self._screen_to_world(cw, ch)
        vx0, vx1, vy0, vy1 = min(x_a, x_b), max(x_a, x_b), min(y_bottom, y_top), max(y_bottom, y_top)
        ix0, ix1 = max(vx0, rect.x0), min(vx1, rect.x0 + rect.w)
        iy0, iy1 = max(vy0, rect.y0), min(vy1, rect.y0 + rect.h)
        if ix0 >= ix1 or iy0 >= iy1:
            return
        base_width, base_height = self._base_image.size
        left = max(0, min(base_width - 1, int((ix0 - rect.x0) / rect.w * base_width)))
        right = max(left + 1, min(base_width, int((ix1 - rect.x0) / rect.w * base_width)))
        upper = max(0, min(base_height - 1, int((rect.y0 + rect.h - iy1) / rect.h * base_height)))
        lower = max(upper + 1, min(base_height, int((rect.y0 + rect.h - iy0) / rect.h * base_height)))
        crop = self._base_image.crop((left, upper, right, lower))
        sx0, sy_top = self._world_to_screen((ix0, iy1))
        sx1, sy_bottom = self._world_to_screen((ix1, iy0))
        width_px, height_px = int(round(sx1 - sx0)), int(round(sy_bottom - sy_top))
        if width_px <= 1 or height_px <= 1:
            return
        crop = crop.resize((width_px, height_px), Image.LANCZOS)
        out = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
        px, py = int(round(sx0)), int(round(sy_top))
        paste_x0, paste_y0, paste_x1, paste_y1 = max(0, px), max(0, py), min(cw, px + width_px), min(ch, py + height_px)
        if paste_x1 <= paste_x0 or paste_y1 <= paste_y0:
            return
        src_x0, src_y0 = paste_x0 - px, paste_y0 - py
        crop = crop.crop((src_x0, src_y0, src_x0 + paste_x1 - paste_x0, src_y0 + paste_y1 - paste_y0))
        opacity = max(0, min(100, int(float(opacity))))
        if opacity <= 0:
            return
        if opacity < 100:
            if crop.mode != "RGBA":
                crop = crop.convert("RGBA")
            _red, _green, _blue, alpha = crop.split()
            crop.putalpha(alpha.point(lambda value: int(value * opacity / 100)))
        out.paste(crop, (paste_x0, paste_y0), crop)
        self._photo = ImageTk.PhotoImage(out)
        canvas.create_image(0, 0, anchor="nw", image=self._photo, tags=("bg_world",))
        canvas.tag_lower("bg_world")

    def _corners_world(self):
        if self._world_rect is None:
            return None
        rect = self._world_rect
        return {"bl": (rect.x0, rect.y0), "br": (rect.x0 + rect.w, rect.y0), "tl": (rect.x0, rect.y0 + rect.h), "tr": (rect.x0 + rect.w, rect.y0 + rect.h)}

    def _corners_screen(self):
        corners = self._corners_world()
        return None if corners is None else {key: self._world_to_screen(value) for key, value in corners.items()}

    def draw_resize_handles(self, resize_enabled: bool) -> None:
        if not resize_enabled or self._canvas is None:
            return
        corners = self._corners_screen()
        if corners is None:
            return
        canvas = self._canvas
        top_left, bottom_right = corners["tl"], corners["br"]
        canvas.create_rectangle(*top_left, *bottom_right, outline="gray30", dash=(3, 2), width=1, tags=("bg_ui",))
        for key in ("tl", "tr", "bl", "br"):
            x, y = corners[key]
            canvas.create_rectangle(x - 6, y - 6, x + 6, y + 6, outline="gray10", fill="white", width=1, tags=("bg_ui",))

    def hit_test_handle(self, sx: float, sy: float) -> str | None:
        corners = self._corners_screen()
        if corners is None:
            return None
        for key in ("tl", "tr", "bl", "br"):
            x, y = corners[key]
            if (sx - x) ** 2 + (sy - y) ** 2 <= 8 ** 2:
                return key
        return None

    def start_resize(self, handle: str, sx: int, sy: int) -> None:
        if self._world_rect is None:
            return
        opposite = {"tl": "br", "br": "tl", "tr": "bl", "bl": "tr"}[handle]
        corners = self._corners_world()
        self._resizing = {"fixed": corners[opposite]}

    def start_move(self, sx: int, sy: int) -> None:
        if self._world_rect is None:
            return
        mouse_x, mouse_y = self._screen_to_world(sx, sy)
        self._moving = {"start_mouse": (float(mouse_x), float(mouse_y)), "start_xy": (self._world_rect.x0, self._world_rect.y0)}

    def update_move(self, sx: int, sy: int) -> bool:
        if self._moving is None or self._world_rect is None:
            return False
        mouse_x, mouse_y = self._screen_to_world(sx, sy)
        start_x, start_y = self._moving["start_mouse"]
        x0, y0 = self._moving["start_xy"]
        self._world_rect = BackgroundMapWorldRect(float(x0 + mouse_x - start_x), float(y0 + mouse_y - start_y), self._world_rect.w, self._world_rect.h)
        self._on_geometry_changed()
        return True

    def update_resize(self, sx: int, sy: int) -> bool:
        if self._resizing is None or self._world_rect is None:
            return False
        fixed_x, fixed_y = self._resizing["fixed"]
        mouse_x, mouse_y = self._screen_to_world(sx, sy)
        raw_width, raw_height = abs(mouse_x - fixed_x), abs(mouse_y - fixed_y)
        if raw_width < 1e-6 or raw_height < 1e-6:
            return False
        aspect = self._world_rect.w / self._world_rect.h
        width, height = (raw_width, raw_width / aspect) if raw_width / raw_height > aspect else (raw_height * aspect, raw_height)
        delta_x, delta_y = mouse_x - fixed_x, mouse_y - fixed_y
        self._world_rect = BackgroundMapWorldRect(float(fixed_x if delta_x >= 0 else fixed_x - width), float(fixed_y if delta_y >= 0 else fixed_y - height), float(max(1e-3, width)), float(max(1e-3, height)))
        self._on_geometry_changed()
        return True

    def finish_move(self) -> None:
        self._moving = None

    def finish_resize(self) -> None:
        self._resizing = None

    def cancel_interaction(self) -> None:
        self._moving = None
        self._resizing = None
