"""Dialogue modal d'édition de la sélection d'un chemin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import tkinter as tk
from tkinter import messagebox


def snapshot_index_from_view(view_index: int, count: int, inverted: bool) -> int:
    if not inverted:
        return int(view_index)
    if int(view_index) == 0:
        return 0
    return int(count - view_index)


def selection_mask_from_view(values: Sequence[bool], orientation_user: str, boundary_orientation: str) -> tuple[bool, ...]:
    result = [False] * len(values)
    inverted = orientation_user != boundary_orientation
    for view_index, selected in enumerate(values):
        result[snapshot_index_from_view(view_index, len(values), inverted)] = bool(selected)
    return tuple(result)


@dataclass(frozen=True)
class CheminEditResult:
    orientation_user: str
    selection_mask: tuple[bool, ...]
    selected_measures: tuple[str, ...]


class CheminEditDialog(tk.Toplevel):
    """Gère l'état temporaire UI, sans accès au viewer ni au Core."""

    def __init__(self, parent, *, snapshot_nodes: Sequence[str], selection_mask: Sequence[bool], boundary_orientation: str, current_orientation: str, measures_specs: Sequence[dict], selected_measures: Sequence[str], node_label_provider: Callable[[str], str | None]):
        super().__init__(parent)
        self.result: CheminEditResult | None = None
        self._nodes, self._mask = tuple(map(str, snapshot_nodes)), list(map(bool, selection_mask))
        self._boundary_orientation, self._node_label_provider = boundary_orientation, node_label_provider
        self.title("Éditer le chemin")
        self.transient(parent); self.grab_set(); self.resizable(True, True); self.minsize(320, 420)
        root = tk.Frame(self, padx=10, pady=10); root.pack(fill=tk.BOTH, expand=True)
        self._orientation = tk.StringVar(value=current_orientation)
        orientation_row = tk.Frame(root); orientation_row.pack(anchor="w", fill=tk.X, pady=(0, 8))
        tk.Label(orientation_row, text="Sens").pack(side=tk.LEFT, padx=(0, 10))
        tk.Radiobutton(orientation_row, text="Sens horaire", value="cw", variable=self._orientation).pack(side=tk.LEFT, padx=(0, 10))
        tk.Radiobutton(orientation_row, text="Sens inverse", value="ccw", variable=self._orientation).pack(side=tk.LEFT)
        tk.Label(root, text="Mesures (A, O, B):").pack(anchor="w")
        measures_row = tk.Frame(root); measures_row.pack(fill=tk.X, pady=(2, 8))
        self._measure_vars = {}
        for spec in measures_specs:
            key = str(spec.get("key")); var = tk.BooleanVar(value=key in selected_measures); self._measure_vars[key] = var
            tk.Checkbutton(measures_row, text=str(spec.get("label")), variable=var).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(root, text="Liste des nœuds").pack(anchor="w", pady=(0, 2))
        frame = tk.Frame(root, bd=1, relief=tk.GROOVE); frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(frame, highlightthickness=0, bd=0); scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview); canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._nodes_frame = tk.Frame(canvas); window_id = canvas.create_window((0, 0), window=self._nodes_frame, anchor="nw")
        self._nodes_frame.bind("<Configure>", lambda _evt: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda _evt: canvas.itemconfigure(window_id, width=canvas.winfo_width()))
        def on_mouse_wheel(event):
            if event.delta:
                canvas.yview_scroll(int(-event.delta / 120), "units")
        def bind_wheel(_event=None):
            self.bind_all("<MouseWheel>", on_mouse_wheel)
            self.bind_all("<Button-4>", lambda _event: canvas.yview_scroll(-1, "units"))
            self.bind_all("<Button-5>", lambda _event: canvas.yview_scroll(1, "units"))
        def unbind_wheel(_event=None):
            self.unbind_all("<MouseWheel>")
            self.unbind_all("<Button-4>")
            self.unbind_all("<Button-5>")
        canvas.bind("<Enter>", bind_wheel)
        canvas.bind("<Leave>", unbind_wheel)
        self._view_vars: list[tk.BooleanVar] = []; self._view_inverted = current_orientation != boundary_orientation
        self._rebuild_view(); self._orientation.trace_add("write", lambda *_: self._rebuild_view())
        actions = tk.Frame(root); actions.pack(fill=tk.X, pady=(8, 0))
        tk.Button(actions, text="Annuler", command=self._on_cancel).pack(side=tk.RIGHT)
        tk.Button(actions, text="OK", command=self._on_ok).pack(side=tk.RIGHT, padx=(0, 6))
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def show(self) -> CheminEditResult | None:
        self.wait_visibility(); self.focus_set(); self.wait_window(); return self.result

    def _rebuild_view(self) -> None:
        for i, variable in enumerate(self._view_vars): self._mask[snapshot_index_from_view(i, len(self._view_vars), self._view_inverted)] = bool(variable.get())
        for widget in self._nodes_frame.winfo_children(): widget.destroy()
        self._view_vars.clear(); self._view_inverted = self._orientation.get().strip().lower() != self._boundary_orientation
        for view_index in range(len(self._nodes)):
            snapshot_index = snapshot_index_from_view(view_index, len(self._nodes), self._view_inverted)
            label = self._node_label_provider(self._nodes[snapshot_index]); text = str(label).strip() if label is not None else ""
            variable = tk.BooleanVar(value=self._mask[snapshot_index]); self._view_vars.append(variable)
            tk.Checkbutton(self._nodes_frame, text=text or "(sans label)", variable=variable, anchor="w", justify="left").pack(anchor="w")

    def _on_ok(self) -> None:
        orientation = self._orientation.get().strip().lower()
        if orientation not in ("cw", "ccw"): raise RuntimeError(f"Édition du chemin impossible : orientationUser invalide ({orientation}).")
        mask = selection_mask_from_view([var.get() for var in self._view_vars], orientation, self._boundary_orientation)
        if sum(mask) < 3:
            messagebox.showerror("Éditer le chemin", "Au moins 3 nœuds doivent rester sélectionnés.", parent=self); return
        measures = tuple(key for key, variable in self._measure_vars.items() if variable.get()) or ("angle",)
        self.result = CheminEditResult(orientation, mask, measures); self.destroy()

    def _on_cancel(self) -> None:
        self.result = None; self.destroy()
