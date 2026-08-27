"""Vue transactionnelle des ScenarioCity d'un ScenarioReference draft."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from src.assembleur_geo_map_view import CalibratedGeoMap, GeoMapMarker, GeoMapView
from src.assembleur_geometry_reference import ScenarioReference


class ScenarioCitiesView(ttk.Frame):
    """Liste, nom et carte des SCITY locales d'un référentiel draft."""

    def __init__(
        self,
        parent,
        *,
        scenario_reference: ScenarioReference,
        maps_dir: str,
        on_reference_changed=None,
    ):
        super().__init__(parent)
        self._reference = scenario_reference
        self._scenario_city_ids: list[str] = []
        self._selected_city_ref_id: str | None = None
        self._fit_applied = False
        self._on_reference_changed = on_reference_changed
        self._name_var = tk.StringVar()
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        panes = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        panes.grid(row=0, column=0, sticky="nsew")
        list_pane = ttk.Frame(panes, padding=(0, 0, 8, 0))
        map_pane = ttk.Frame(panes, padding=(8, 0, 0, 0))
        panes.add(list_pane, weight=1)
        panes.add(map_pane, weight=2)

        list_frame = ttk.Frame(list_pane)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(list_frame, exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_selected)

        map_pane.rowconfigure(1, weight=1)
        map_pane.columnconfigure(0, weight=1)
        properties = ttk.Frame(map_pane, padding=(0, 0, 0, 8))
        properties.grid(row=0, column=0, sticky="ew")
        ttk.Label(properties, text="Nom :").grid(row=0, column=0, sticky="w")
        self._name_entry = ttk.Entry(
            properties, textvariable=self._name_var, state=tk.DISABLED
        )
        self._name_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        properties.columnconfigure(1, weight=1)
        self._name_entry.bind("<Return>", self._on_name_committed)
        self._name_entry.bind("<FocusOut>", self._on_name_committed)

        self.map_view = GeoMapView(
            map_pane,
            on_marker_selected=self._on_map_marker_selected,
            initial_fit_zoom=2.25,
            minimum_fit_zoom=2.25,
            maximum_zoom=1.0,
        )
        self.map_view.grid(row=1, column=0, sticky="nsew")
        try:
            self.map_view.set_map(CalibratedGeoMap.load_map("france_michelin", maps_dir))
        except (FileNotFoundError, OSError, ValueError):
            pass
        self.refresh(scenario_reference)

    def _sync_name_field(self) -> None:
        if not hasattr(self, "_name_var"):
            return
        city_id = self._selected_city_ref_id
        city = self._reference.cities.get(city_id) if city_id is not None else None
        self._name_var.set("" if city is None else city.name)
        self._name_entry.configure(state=tk.NORMAL if city is not None else tk.DISABLED)

    def _on_name_committed(self, _event=None) -> None:
        if not hasattr(self, "_name_var"):
            return True
        city_id = self._selected_city_ref_id
        if city_id is None:
            return True
        city = self._reference.cities.get(city_id)
        if city is None or self._name_var.get() == city.name:
            return True
        try:
            self._reference.rename_city(city_id, self._name_var.get())
        except ValueError as exc:
            messagebox.showerror("Nom de la ville", str(exc), parent=self.winfo_toplevel())
            self._name_var.set(city.name)
            return False
        self.refresh()
        if self._on_reference_changed is not None:
            self._on_reference_changed()
        return True

    def refresh(self, scenario_reference: ScenarioReference | None = None) -> None:
        """Affiche toutes les SCITY publiées, y compris les orphelines."""
        if scenario_reference is not None and scenario_reference is not self._reference:
            self._reference = scenario_reference
            self._fit_applied = False
        cities = sorted(
            self._reference.cities.values(),
            key=lambda city: (city.name.casefold(), city.city_ref_id),
        )
        self._scenario_city_ids = [city.city_ref_id for city in cities]
        if self._selected_city_ref_id not in self._scenario_city_ids:
            self._selected_city_ref_id = None

        self.listbox.delete(0, tk.END)
        for city in cities:
            self.listbox.insert(tk.END, city.name)
        if self._selected_city_ref_id is not None:
            index = self._scenario_city_ids.index(self._selected_city_ref_id)
            self.listbox.selection_set(index)
            self.listbox.activate(index)
            self.listbox.see(index)
        self._sync_name_field()

        self.map_view.set_markers(
            GeoMapMarker(city.city_ref_id, city.latitude, city.longitude, city.name)
            for city in cities
        )
        self.map_view.set_selected_marker(self._selected_city_ref_id)
        if cities and not self._fit_applied:
            self.map_view.fit_to_bounds((city.latitude, city.longitude) for city in cities)
            self._fit_applied = True

    def _on_list_selected(self, _event=None) -> None:
        self._on_name_committed()
        selection = self.listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        if not 0 <= index < len(self._scenario_city_ids):
            return
        self._selected_city_ref_id = self._scenario_city_ids[index]
        self.map_view.set_selected_marker(self._selected_city_ref_id, recenter=True)
        self._sync_name_field()

    def _on_map_marker_selected(self, city_ref_id) -> None:
        self._on_name_committed()
        if city_ref_id not in self._scenario_city_ids:
            self._selected_city_ref_id = None
            self.listbox.selection_clear(0, tk.END)
            self._sync_name_field()
            return
        index = self._scenario_city_ids.index(city_ref_id)
        self._selected_city_ref_id = city_ref_id
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.activate(index)
        self.listbox.see(index)
        self._sync_name_field()
