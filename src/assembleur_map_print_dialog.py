"""Tk dialog for the independent Assembleur PDF print preview."""

from __future__ import annotations

import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from src.assembleur_map_print import (
    MARGIN_PRESETS_MM,
    AssembleurPrintSettings,
    AssembleurPrintSnapshot,
    AssembleurPrintViewport,
    calculate_print_layout,
    render_print_page_raster,
)
from src.assembleur_map_print_pdf import export_print_pdf


class AssembleurMapPrintDialog(tk.Toplevel):
    """A self-contained print editor: it never accesses the main canvas state."""

    _PREVIEW_DPI = 96

    def __init__(self, master: tk.Misc, snapshot: AssembleurPrintSnapshot, viewport: AssembleurPrintViewport, settings: AssembleurPrintSettings) -> None:
        super().__init__(master)
        self.title("Imprimer / Exporter la carte")
        self.geometry("1100x750")
        self.minsize(850, 550)
        self.transient(master)
        self.snapshot = snapshot
        self.viewport = viewport
        self._map_rect: tuple[float, float, float, float] | None = None
        self._page_rect: tuple[float, float, float, float] | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._resize_job: str | None = None
        self._drag_anchor: tuple[float, float] | None = None

        self.paper_var = tk.StringVar(value=settings.paper_format)
        self.orientation_var = tk.StringVar(value=settings.orientation)
        self.title_var = tk.StringVar(value=settings.title)
        self.margin_var = tk.StringVar(value=_margin_preset_for(settings.margin_mm))
        self.map_var = tk.BooleanVar(value=settings.has_layer("map"))
        self.assembly_var = tk.BooleanVar(value=settings.has_layer("assembly"))
        self.beacons_var = tk.BooleanVar(value=settings.has_layer("beacons"))
        self.opacity_var = tk.IntVar(value=settings.map_opacity)
        self.opacity_text = tk.StringVar()

        self._build_ui()
        self._bind_events()
        self._update_opacity_state()
        self.after_idle(self.redraw_preview)

    def _build_ui(self) -> None:
        panes = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        controls = ttk.Frame(panes, padding=10)
        preview = ttk.Frame(panes)
        panes.add(controls, weight=0)
        panes.add(preview, weight=1)

        ttk.Label(controls, text="Format").pack(anchor="w")
        ttk.Combobox(controls, textvariable=self.paper_var, values=("A4",), state="readonly", width=18).pack(fill="x", pady=(2, 12))
        ttk.Label(controls, text="Orientation").pack(anchor="w")
        ttk.Radiobutton(controls, text="Portrait", variable=self.orientation_var, value="portrait", command=self._on_layout_change).pack(anchor="w")
        ttk.Radiobutton(controls, text="Paysage", variable=self.orientation_var, value="landscape", command=self._on_layout_change).pack(anchor="w", pady=(0, 12))
        ttk.Label(controls, text="Titre").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.title_var, width=30).pack(fill="x", pady=(2, 12))

        ttk.Label(controls, text="Éléments à imprimer").pack(anchor="w")
        layers = ttk.Frame(controls)
        layers.pack(fill="x", padx=(12, 0), pady=(2, 12))
        ttk.Checkbutton(layers, text="Carte", variable=self.map_var, command=self._on_map_toggle).pack(anchor="w")
        ttk.Checkbutton(layers, text="Assemblage", variable=self.assembly_var, command=self.redraw_preview).pack(anchor="w")
        ttk.Checkbutton(layers, text="Balises", variable=self.beacons_var, command=self.redraw_preview).pack(anchor="w")

        ttk.Label(controls, text="Marges").pack(anchor="w")
        self.margin_combo = ttk.Combobox(controls, textvariable=self.margin_var, values=tuple(MARGIN_PRESETS_MM), state="readonly", width=18)
        self.margin_combo.pack(fill="x", pady=(2, 12))
        ttk.Label(controls, text="Opacité de la carte").pack(anchor="w")
        opacity_row = ttk.Frame(controls)
        opacity_row.pack(fill="x")
        self.opacity_scale = ttk.Scale(opacity_row, from_=0, to=100, variable=self.opacity_var, command=lambda _value: self._on_opacity_change())
        self.opacity_scale.pack(side="left", fill="x", expand=True)
        ttk.Label(opacity_row, textvariable=self.opacity_text, width=5).pack(side="left", padx=(6, 0))
        ttk.Frame(controls).pack(fill="both", expand=True)
        buttons = ttk.Frame(controls)
        buttons.pack(fill="x", pady=(12, 0))
        ttk.Button(buttons, text="Exporter en PDF…", command=self._export).pack(side="left")
        ttk.Button(buttons, text="Annuler", command=self.destroy).pack(side="right")

        self.preview_canvas = tk.Canvas(preview, background="#b8b8b8", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True)

    def _bind_events(self) -> None:
        self.title_var.trace_add("write", lambda *_args: self.redraw_preview())
        self.paper_var.trace_add("write", lambda *_args: self._on_layout_change())
        self.margin_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_layout_change())
        self.preview_canvas.bind("<Configure>", self._on_canvas_resize)
        self.preview_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.preview_canvas.bind("<Button-4>", lambda event: self._zoom_event(event, 1.15))
        self.preview_canvas.bind("<Button-5>", lambda event: self._zoom_event(event, 1 / 1.15))
        self.preview_canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.preview_canvas.bind("<B1-Motion>", self._on_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", lambda _event: setattr(self, "_drag_anchor", None))

    def _settings(self) -> AssembleurPrintSettings:
        layers = tuple(layer for layer, value in (("map", self.map_var.get()), ("assembly", self.assembly_var.get()), ("beacons", self.beacons_var.get())) if value)
        return AssembleurPrintSettings(
            paper_format=self.paper_var.get(), orientation=self.orientation_var.get(),
            title=self.title_var.get(), margin_mm=MARGIN_PRESETS_MM[self.margin_var.get()],
            selected_layers=layers, map_opacity=int(self.opacity_var.get()),
        )

    def _on_layout_change(self) -> None:
        settings = self._settings()
        layout = calculate_print_layout(settings)
        self.viewport = self.viewport.expanded_to_aspect(layout.map_rect_mm[2] / layout.map_rect_mm[3])
        self.redraw_preview()

    def _on_map_toggle(self) -> None:
        self._update_opacity_state()
        self.redraw_preview()

    def _update_opacity_state(self) -> None:
        self.opacity_scale.state(("!disabled",) if self.map_var.get() else ("disabled",))
        self.opacity_text.set(f"{int(self.opacity_var.get())} %")

    def _on_opacity_change(self) -> None:
        self.opacity_var.set(max(0, min(100, int(float(self.opacity_var.get())))))
        self._update_opacity_state()
        self.redraw_preview()

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self._redraw_after_resize)

    def _redraw_after_resize(self) -> None:
        self._resize_job = None
        self.redraw_preview()

    def redraw_preview(self) -> None:
        if not self.winfo_exists():
            return
        canvas_w, canvas_h = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
        if canvas_w < 20 or canvas_h < 20:
            return
        settings = self._settings()
        page_w, page_h = settings.page_size_mm
        display_scale = min((canvas_w - 40) / page_w, (canvas_h - 40) / page_h)
        width, height = max(1, round(page_w * display_scale)), max(1, round(page_h * display_scale))
        page = render_print_page_raster(self.snapshot, settings, self.viewport, self._PREVIEW_DPI)
        page = page.resize((width, height))
        left, top = (canvas_w - width) / 2, (canvas_h - height) / 2
        layout = calculate_print_layout(settings)
        mx, my, mw, mh = layout.map_rect_mm
        self._page_rect = (left, top, width, height)
        self._map_rect = (left + mx / page_w * width, top + my / page_h * height, mw / page_w * width, mh / page_h * height)
        self._preview_photo = ImageTk.PhotoImage(page)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(left, top, image=self._preview_photo, anchor="nw")
        self.preview_canvas.create_rectangle(left, top, left + width, top + height, outline="#707070")

    def _point_in_map(self, x: float, y: float) -> bool:
        if self._map_rect is None:
            return False
        left, top, width, height = self._map_rect
        return left <= x <= left + width and top <= y <= top + height

    def _zoom_event(self, event: tk.Event, factor: float) -> str | None:
        if not self._point_in_map(event.x, event.y) or self._map_rect is None:
            return None
        left, top, width, height = self._map_rect
        self.viewport = self.viewport.zoom_at((event.x - left) / width, (event.y - top) / height, factor)
        self.redraw_preview()
        return "break"

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        return self._zoom_event(event, 1.15 if event.delta > 0 else 1 / 1.15)

    def _on_drag_start(self, event: tk.Event) -> None:
        self._drag_anchor = (event.x, event.y) if self._point_in_map(event.x, event.y) else None

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_anchor is None or self._map_rect is None:
            return
        previous_x, previous_y = self._drag_anchor
        _left, _top, width, height = self._map_rect
        self.viewport = self.viewport.panned((event.x - previous_x) / width, (event.y - previous_y) / height)
        self._drag_anchor = (event.x, event.y)
        self.redraw_preview()

    def _export(self) -> None:
        title = re.sub(r"[^A-Za-z0-9._ -]+", "_", self._settings().title).strip() or "assemblage"
        path = filedialog.asksaveasfilename(title="Exporter en PDF", defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile=f"assemblage_{title[:40]}.pdf")
        if not path:
            return
        try:
            export_print_pdf(path, self.snapshot, self._settings(), self.viewport)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Export PDF", f"Impossible de générer le PDF :\n{exc}", parent=self)
            return
        messagebox.showinfo("Export PDF", f"Export terminé avec succès.\n\nFichier :\n{path}", parent=self)


def _margin_preset_for(value: float) -> str:
    return min(MARGIN_PRESETS_MM, key=lambda name: abs(MARGIN_PRESETS_MM[name] - value))
