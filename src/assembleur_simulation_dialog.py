"""Dialogue modal de paramétrage de la simulation d'assemblage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
import tkinter as tk
from tkinter import messagebox

from src.assembleur_sim import InitialTriangleOrientation

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class AutoOrientationReference:
    """Référence Core proposée par le viewer au dialogue de simulation."""
    beacon_id: str
    element_id: str
    tri_rank: int
    theta_rad: float


class DialogSimulationAssembler(tk.Toplevel):
    """Boîte de dialogue 'Simulation > Assembler…'"""

    def __init__(
        self,
        parent,
        algo_items: List[Tuple[str, str]],
        n_max: int,
        default_algo_id: str,
        default_n: int,
        default_order: str = "forward",
        beacon_items: List[Tuple[str, str]] | None = None,
        orientation_reference_by_beacon: Dict[str, "AutoOrientationReference | None"] | None = None,
        default_beacon_id: str = "",
        default_first_edge: str = "OL",
    ):
        super().__init__(parent)
        self._beacon_items = list(beacon_items or ())
        if not self._beacon_items:
            raise ValueError("Simulation: une balise d'ancrage est obligatoire")
        self._beacon_id_by_display = {
            display_label: beacon_id
            for beacon_id, display_label in self._beacon_items
        }
        if len(self._beacon_id_by_display) != len(self._beacon_items):
            raise ValueError("Simulation: libellés de balises dupliqués")
        self._orientation_reference_by_beacon = dict(orientation_reference_by_beacon or {})
        self.title("Assembler (simulation)")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result = None  # (algo_id, n, order, beacon_id, InitialTriangleOrientation)

        # Imports locaux (évite d'imposer ttk partout)
        from tkinter import ttk

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Algorithme :").grid(row=0, column=0, sticky="w")
        self.algo_var = tk.StringVar(value=default_algo_id or (algo_items[0][0] if algo_items else ""))
        self.algo_combo = ttk.Combobox(
            frm,
            textvariable=self.algo_var,
            state="readonly",
            values=[f"{aid} - {label}" for aid, label in algo_items],
            width=48
        )
        self._algo_items = list(algo_items)
        sel_index = 0
        for i, (aid, _lbl) in enumerate(self._algo_items):
            if aid == self.algo_var.get():
                sel_index = i
                break
        if algo_items:
            self.algo_combo.current(sel_index)
        self.algo_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        ttk.Label(frm, text="Nombre de triangles (n premiers) :").grid(row=2, column=0, sticky="w")
        vcmd = (self.register(self._validate_even), "%P")
        self.var_nb_triangles = tk.IntVar(value=int(default_n))
        self.spin_nb_triangles = ttk.Spinbox(
            frm,
            from_=2,                 # minimum pair
            to=max(2, int(n_max)),
            increment=2,             # flèches +2 / -2
            textvariable=self.var_nb_triangles,
            width=6,
            validate="key",          # empêche les impairs au clavier
            validatecommand=vcmd
        )
        self.spin_nb_triangles.grid(row=2, column=1, sticky="e")

        # --- Ordre d'assemblage ---
        d_order = str(default_order or "forward").strip().lower()
        if d_order not in ("forward", "reverse"):
            d_order = "forward"
        self.order_var = tk.StringVar(value=d_order)  # "forward" | "reverse"
        ttk.Label(frm, text="Ordre d’assemblage :").grid(row=3, column=0, sticky="w", pady=(8, 0))
        order_frm = ttk.Frame(frm)
        order_frm.grid(row=3, column=1, sticky="e", pady=(8, 0))
        ttk.Radiobutton(order_frm, text="Normal", value="forward", variable=self.order_var).grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(order_frm, text="Inverse", value="reverse", variable=self.order_var).grid(row=0, column=1)

        # --- Balise d'ancrage ---
        beacon_labels = [display_label for _beacon_id, display_label in self._beacon_items]
        beacon_index = next(
            (
                index for index, (beacon_id, _label) in enumerate(self._beacon_items)
                if beacon_id == default_beacon_id
            ),
            0,
        )
        self.beacon_var = tk.StringVar(value=beacon_labels[beacon_index])
        ttk.Label(frm, text="Balise d’ancrage :").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        self.beacon_combo = ttk.Combobox(
            frm,
            textvariable=self.beacon_var,
            state="readonly",
            values=beacon_labels,
            width=32,
        )
        self.beacon_combo.current(beacon_index)
        self.beacon_combo.grid(row=4, column=1, sticky="e", pady=(8, 0))

        # --- Orientation initiale ---
        d_edge = str(default_first_edge or "OL").upper().strip()
        if d_edge not in ("OL", "BL"):
            d_edge = "OL"
        self.first_edge_var = tk.StringVar(value=d_edge)  # "OL" | "BL" | référence
        ttk.Label(frm, text="Orientation initiale :").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.first_edge_combo = ttk.Combobox(
            frm,
            textvariable=self.first_edge_var,
            state="readonly",
            values=[],
            width=12
        )
        self.first_edge_combo.grid(row=5, column=1, sticky="e", pady=(8, 0))
        self.beacon_combo.bind("<<ComboboxSelected>>", self._on_beacon_changed)
        self._rebuild_initial_orientation_choices(prefer_reference=True)

        btns = ttk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Annuler", command=self._on_cancel).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="OK", command=self._on_ok).grid(row=0, column=1)

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.spin_nb_triangles.focus_set()
        self.spin_nb_triangles.selection_range(0, tk.END)

    def _validate_even(self, value):
        if value == "":
            return True
        try:
            return int(value) % 2 == 0
        except ValueError:
            return False

    def _on_cancel(self):
        self.result = None
        self.destroy()

    def _selected_beacon_id(self) -> str:
        beacon_id = self._beacon_id_by_display.get(self.beacon_var.get())
        if beacon_id is None:
            raise RuntimeError("Simulation: balise sélectionnée introuvable")
        return beacon_id

    def _rebuild_initial_orientation_choices(self, prefer_reference: bool) -> None:
        previous = self.first_edge_var.get()
        reference = self._orientation_reference_by_beacon.get(self._selected_beacon_id())
        values = ["BL = 0°", "OL = 0°"]
        reference_label = None
        if reference is not None:
            reference_label = f"Comme T{reference.tri_rank}"
            values.insert(0, reference_label)
        self.first_edge_combo.configure(values=values)
        if previous.startswith("Comme T"):
            selected = reference_label or "OL = 0°"
        elif prefer_reference and reference_label is not None:
            selected = reference_label
        elif "BL" in previous:
            selected = "BL = 0°"
        else:
            selected = "OL = 0°"
        self.first_edge_var.set(selected)

    def _on_beacon_changed(self, _event=None) -> None:
        self._rebuild_initial_orientation_choices(prefer_reference=False)

    def _on_ok(self):
        raw = str(self.algo_combo.get() or "")
        algo_id = raw.split(" - ", 1)[0].strip() if " - " in raw else raw.strip()
        n = int(self.var_nb_triangles.get())

        if n <= 0:
            messagebox.showerror("Assembler", "Nombre de triangles invalide.")
            return

        if n % 2 == 1:
            n2 = n - 1
            if n2 < 2:
                messagebox.showerror("Assembler", "n doit être pair (minimum 2).")
                return
            # n doit être pair : on ajuste silencieusement (pas de popup)
            LOGGER.info("[SIM] n impair -> utilisation de n=%s", n2)
            n = n2
            self.var_nb_triangles.set(n)

        order = str(self.order_var.get() or "forward")
        beacon_id = self._beacon_id_by_display.get(self.beacon_var.get())
        if beacon_id is None:
            messagebox.showerror("Assembler", "Sélectionne une balise d'ancrage.")
            return
        reference = self._orientation_reference_by_beacon.get(beacon_id)
        first_raw = str(self.first_edge_var.get() or "OL")
        if first_raw.startswith("Comme T"):
            if reference is None:
                raise RuntimeError("Simulation: référence d'orientation absente")
            initial_orientation = InitialTriangleOrientation.reference(
                reference.element_id, reference.tri_rank, reference.theta_rad
            )
        else:
            initial_orientation = InitialTriangleOrientation.edge_north(
                "BL" if "BL" in first_raw else "OL"
            )

        self.result = (algo_id, int(n), order, beacon_id, initial_orientation)

        self.destroy()
