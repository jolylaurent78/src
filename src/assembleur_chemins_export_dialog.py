"""Tk configurator for one-at-a-time Chemins Excel exports."""

from __future__ import annotations

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.assembleur_chemins_export import (
    BeaconsExportOptions,
    CheminsExportError,
    PointsExportOptions,
    TripletsExportOptions,
    export_chemins_xlsx,
)


class CheminsExportDialog(tk.Toplevel):
    """Local, non-persistent options; never changes the Chemins toolbar state."""

    def __init__(self, master, *, world, chemins, scenario_name: str, exports_dir: str,
                 catalogue, beacon_options: list[tuple[str, str]], selected_beacon_id: str | None,
                 map_transform=None, map_name: str | None = None, beacon_world_resolver=None,
                 on_success=None) -> None:
        super().__init__(master)
        self.title("Exporter le chemin vers Excel")
        self.transient(master)
        self.grab_set()
        self.resizable(False, False)
        self.world, self.chemins, self.catalogue = world, chemins, catalogue
        self.scenario_name, self.exports_dir = scenario_name, exports_dir
        self.map_transform, self.map_name = map_transform, map_name
        self.beacon_world_resolver, self.on_success = beacon_world_resolver, on_success
        self._beacon_by_label = {label: beacon_id for beacon_id, label in beacon_options}
        self._label_by_beacon = {beacon_id: label for beacon_id, label in beacon_options}
        initial_label = self._label_by_beacon.get(selected_beacon_id) or (beacon_options[0][1] if beacon_options else "")

        self.triplet_labels = tk.BooleanVar(value=False)
        self.triplet_angles = tk.BooleanVar(value=False)
        self.triplet_distances = tk.BooleanVar(value=False)
        self.triplet_reference = tk.BooleanVar(value=False)
        self.triplet_beacon = tk.StringVar(value=initial_label)
        self.points_order = tk.BooleanVar(value=False)
        self.points_node_id = tk.BooleanVar(value=False)
        self.points_triangles = tk.BooleanVar(value=False)
        self.points_cities = tk.BooleanVar(value=False)
        self.points_types = tk.BooleanVar(value=False)
        self.points_pixel = tk.BooleanVar(value=False)
        self.points_lambert = tk.BooleanVar(value=False)
        self.points_reference = tk.BooleanVar(value=False)
        self.points_beacon = tk.StringVar(value=initial_label)
        self.beacons_id = tk.BooleanVar(value=False)
        self.beacons_name = tk.BooleanVar(value=False)
        self.beacons_city_id = tk.BooleanVar(value=False)
        self.beacons_group = tk.BooleanVar(value=False)
        self.beacons_order = tk.BooleanVar(value=False)
        self.beacons_anchor = tk.BooleanVar(value=False)
        self.beacons_note = tk.BooleanVar(value=False)
        self.beacons_pixel = tk.BooleanVar(value=False)
        self.beacons_lambert = tk.BooleanVar(value=False)
        self.beacons_reference = tk.BooleanVar(value=False)
        self.beacons_origin = tk.StringVar(value=initial_label)
        self._build_ui(beacon_options)
        self._refresh_controls()

    def _build_ui(self, beacon_options: list[tuple[str, str]]) -> None:
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        self.notebook = ttk.Notebook(body)
        self.notebook.pack(fill="both", expand=True)
        triplets = ttk.Frame(self.notebook, padding=12)
        points = ttk.Frame(self.notebook, padding=12)
        beacons = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(triplets, text="Triplets")
        self.notebook.add(points, text="Points")
        self.notebook.add(beacons, text="Balises")
        self.notebook.bind("<<NotebookTabChanged>>", lambda _event: self._refresh_controls())

        self._check(triplets, "Labels du triplet", self.triplet_labels)
        self._check(triplets, "Angles", self.triplet_angles)
        angle_ref = ttk.Frame(triplets)
        angle_ref.pack(fill="x", padx=(20, 0), pady=(0, 6))
        ttk.Label(angle_ref, text="Balise de référence :").pack(anchor="w")
        self.triplet_beacon_combo = ttk.Combobox(angle_ref, textvariable=self.triplet_beacon, values=[label for _id, label in beacon_options], state="disabled", width=34)
        self.triplet_beacon_combo.pack(fill="x", pady=(2, 0))
        self._check(triplets, "Distances", self.triplet_distances)
        self._check(triplets, "Inclure les informations de référence", self.triplet_reference)

        self._check(points, "Ordre", self.points_order)
        self._check(points, "Node ID", self.points_node_id)
        self._check(points, "ID Triangle(s)", self.points_triangles)
        self._check(points, "Ville(s) associée(s)", self.points_cities)
        self._check(points, "Type(s) de nœud", self.points_types)
        self.points_pixel_check = self._check(points, "Pixel", self.points_pixel)
        pixel_ref = ttk.Frame(points)
        pixel_ref.pack(fill="x", padx=(20, 0), pady=(0, 6))
        ttk.Label(pixel_ref, text="Origine (0,0) :").pack(anchor="w")
        self.points_beacon_combo = ttk.Combobox(pixel_ref, textvariable=self.points_beacon, values=[label for _id, label in beacon_options], state="disabled", width=34)
        self.points_beacon_combo.pack(fill="x", pady=(2, 0))
        self.points_lambert_check = self._check(points, "Lambert-93", self.points_lambert)
        self._check(points, "Inclure les informations de référence", self.points_reference)

        self._check(beacons, "ID Balise", self.beacons_id)
        self._check(beacons, "Nom", self.beacons_name)
        self._check(beacons, "ID Ville", self.beacons_city_id)
        self._check(beacons, "Groupe", self.beacons_group)
        self._check(beacons, "Ordre", self.beacons_order)
        self._check(beacons, "Ancrage", self.beacons_anchor)
        self._check(beacons, "Note", self.beacons_note)
        self.beacons_pixel_check = self._check(beacons, "Pixel", self.beacons_pixel)
        beacon_pixel_ref = ttk.Frame(beacons)
        beacon_pixel_ref.pack(fill="x", padx=(20, 0), pady=(0, 6))
        ttk.Label(beacon_pixel_ref, text="Origine (0,0) :").pack(anchor="w")
        self.beacons_origin_combo = ttk.Combobox(
            beacon_pixel_ref, textvariable=self.beacons_origin,
            values=[label for _id, label in beacon_options], state="disabled", width=34,
        )
        self.beacons_origin_combo.pack(fill="x", pady=(2, 0))
        self.beacons_lambert_check = self._check(beacons, "Lambert-93", self.beacons_lambert)
        self._check(beacons, "Informations de référence", self.beacons_reference)
        if self.map_transform is None:
            self.points_pixel_check.state(("disabled",))
            self.points_lambert_check.state(("disabled",))
            self.beacons_pixel_check.state(("disabled",))
            self.beacons_lambert_check.state(("disabled",))

        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(12, 0))
        self.export_button = ttk.Button(buttons, text="Exporter…", command=self._export)
        self.export_button.pack(side="left")
        ttk.Button(buttons, text="Annuler", command=self.destroy).pack(side="right")

    def _check(self, parent, label: str, variable: tk.BooleanVar):
        check = ttk.Checkbutton(parent, text=label, variable=variable, command=self._refresh_controls)
        check.pack(anchor="w", pady=2)
        if label == "Pixel":
            self.pixel_check = check
        elif label == "Lambert-93":
            self.lambert_check = check
        return check

    def _refresh_controls(self) -> None:
        angle_ready = self.triplet_angles.get() and bool(self._beacon_by_label)
        pixel_ready = self.points_pixel.get() and bool(self._beacon_by_label)
        beacons_pixel_ready = self.beacons_pixel.get() and bool(self._beacon_by_label)
        self.triplet_beacon_combo.configure(state="readonly" if angle_ready else "disabled")
        self.points_beacon_combo.configure(state="readonly" if pixel_ready else "disabled")
        self.beacons_origin_combo.configure(state="readonly" if beacons_pixel_ready else "disabled")
        self.export_button.state(("!disabled",) if self._active_has_data() else ("disabled",))

    def _active_has_data(self) -> bool:
        if self.notebook.index(self.notebook.select()) == 0:
            return any((self.triplet_labels.get(), self.triplet_angles.get(), self.triplet_distances.get()))
        if self.notebook.index(self.notebook.select()) == 1:
            return any((self.points_order.get(), self.points_node_id.get(), self.points_triangles.get(), self.points_cities.get(), self.points_types.get(), self.points_pixel.get(), self.points_lambert.get()))
        return any((self.beacons_id.get(), self.beacons_name.get(), self.beacons_city_id.get(),
                    self.beacons_group.get(), self.beacons_order.get(), self.beacons_anchor.get(),
                    self.beacons_note.get(), self.beacons_pixel.get(), self.beacons_lambert.get()))

    def _triplets_options(self) -> TripletsExportOptions:
        return TripletsExportOptions(
            include_labels=self.triplet_labels.get(), include_angles=self.triplet_angles.get(),
            include_distances=self.triplet_distances.get(), reference_beacon_id=self._beacon_by_label.get(self.triplet_beacon.get()),
            include_reference_info=self.triplet_reference.get(),
        )

    def _points_options(self) -> PointsExportOptions:
        return PointsExportOptions(
            include_order=self.points_order.get(), include_node_id=self.points_node_id.get(),
            include_triangles=self.points_triangles.get(), include_cities=self.points_cities.get(),
            include_node_types=self.points_types.get(), include_pixel=self.points_pixel.get(),
            pixel_origin_beacon_id=self._beacon_by_label.get(self.points_beacon.get()),
            include_lambert=self.points_lambert.get(), include_reference_info=self.points_reference.get(),
        )

    def _beacons_options(self) -> BeaconsExportOptions:
        return BeaconsExportOptions(
            include_beacon_id=self.beacons_id.get(), include_name=self.beacons_name.get(),
            include_city_id=self.beacons_city_id.get(), include_group=self.beacons_group.get(),
            include_order=self.beacons_order.get(), include_anchor=self.beacons_anchor.get(),
            include_note=self.beacons_note.get(), include_pixel=self.beacons_pixel.get(),
            pixel_origin_beacon_id=self._beacon_by_label.get(self.beacons_origin.get()),
            include_lambert=self.beacons_lambert.get(),
            include_reference_info=self.beacons_reference.get(),
        )

    def _export(self) -> None:
        if not self._active_has_data():
            return
        active_tab = self.notebook.index(self.notebook.select())
        kind = ("triplets", "points", "beacons")[active_tab]
        try:
            if kind == "triplets":
                options = self._triplets_options()
            elif kind == "points":
                options = self._points_options()
            else:
                options = self._beacons_options()
        except CheminsExportError as exc:
            messagebox.showerror("Exporter en Excel", str(exc), parent=self)
            return
        safe_name = re.sub(r'[<>:"/\\|?*]+', "_", self.scenario_name).strip() or "Scenario"
        filename_kind = {"triplets": "Triplets", "points": "Points", "beacons": "Balises"}[kind]
        path = filedialog.asksaveasfilename(title="Exporter en Excel", defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")], initialdir=self.exports_dir, initialfile=f"Chemin {safe_name} - {filename_kind}.xlsx", parent=self)
        if not path:
            return
        path = os.path.normpath(path)
        if os.path.exists(path) and not messagebox.askyesno("Écraser ?", f"Le fichier existe déjà : {os.path.basename(path)}\n\nÉcraser ce fichier ?", parent=self):
            return
        try:
            result = export_chemins_xlsx(path, export_kind=kind, world=self.world, chemins=self.chemins, scenario_name=self.scenario_name, options=options, catalogue=self.catalogue, map_transform=self.map_transform, beacon_world_resolver=self.beacon_world_resolver, map_name=self.map_name)
        except (CheminsExportError, OSError) as exc:
            messagebox.showerror("Exporter en Excel", str(exc), parent=self)
            return
        if self.on_success is not None:
            self.on_success(result)
        self.destroy()
