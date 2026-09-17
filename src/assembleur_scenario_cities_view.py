"""Vue transactionnelle des ScenarioCity d'un ScenarioReference draft."""

from __future__ import annotations

from collections.abc import Callable, Collection
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_map_assets import CatalogueMapAssetResolver, load_calibrated_catalogue_map
from src.assembleur_geo_map_view import GeoMapMarker, GeoMapView
from src.assembleur_geometry_reference import ScenarioReference
from src.assembleur_paths import ApplicationPaths
from src.assembleur_tooltip import attach_tooltip


class ScenarioCitiesView(ttk.Frame):
    """Liste et carte des SCITY locales d'un referentiel draft."""

    def __init__(self, parent, *, scenario_reference: ScenarioReference, catalogue: Catalogue,
                 active_triangle_ref_ids: Callable[[], Collection[str]], on_reference_changed=None):
        super().__init__(parent)
        self._reference = scenario_reference
        self._active_triangle_ref_ids = active_triangle_ref_ids
        self._scenario_city_ids: list[str] = []
        self._selected_city_ref_id: str | None = None
        self._on_reference_changed = on_reference_changed
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
        images_dir = ApplicationPaths.from_runtime().images_dir
        self._icon_rename = tk.PhotoImage(file=str(images_dir / "rename.png"))
        self._icon_delete = tk.PhotoImage(file=str(images_dir / "delete.png"))
        toolbar = ttk.Frame(list_frame)
        toolbar.pack(fill=tk.X, pady=(0, 4))
        self._rename_button = tk.Button(toolbar, image=self._icon_rename,
                                        command=self._rename_selected_city, state=tk.DISABLED,
                                        relief=tk.FLAT, bd=1)
        self._rename_button.pack(side=tk.LEFT)
        self._delete_button = tk.Button(toolbar, image=self._icon_delete,
                                        command=self._delete_selected_city, state=tk.DISABLED,
                                        relief=tk.FLAT, bd=1)
        self._delete_button.pack(side=tk.LEFT, padx=(4, 0))
        attach_tooltip(self._rename_button, "Renommer le point")
        attach_tooltip(self._delete_button, "Supprimer le point temporaire")
        self.listbox = tk.Listbox(list_frame, exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_selected)

        map_pane.rowconfigure(0, weight=1)
        map_pane.columnconfigure(0, weight=1)
        self.map_view = GeoMapView(map_pane, on_marker_selected=self._on_map_marker_selected,
                                   initial_fit_zoom=2.25, minimum_fit_zoom=2.25, maximum_zoom=1.0)
        self.map_view.grid(row=0, column=0, sticky="nsew")
        self._load_catalogue_reference_map(catalogue)
        self.refresh(scenario_reference)

    def _load_catalogue_reference_map(self, catalogue: Catalogue) -> None:
        """Charge uniquement la carte geographique de reference du Catalogue."""
        reference_map_id = catalogue.catalogue_reference_map_id
        if reference_map_id is None:
            return
        calibrated_map = load_calibrated_catalogue_map(
            catalogue.get_map(reference_map_id),
            CatalogueMapAssetResolver(ApplicationPaths.from_runtime()),
        )
        self.map_view.set_map(calibrated_map)

    def _selected_city_is_used(self) -> bool:
        city_id = self._selected_city_ref_id
        if city_id is None:
            return False
        active_triangle_ids = set(self._active_triangle_ref_ids())
        return any(triangle.triangle_ref_id in active_triangle_ids
                   for triangle in self._reference.get_triangles_referencing_city(city_id))

    def _refresh_action_states(self) -> None:
        has_selection = self._selected_city_ref_id is not None
        self._rename_button.configure(state=tk.NORMAL if has_selection else tk.DISABLED)
        self._delete_button.configure(state=tk.NORMAL if has_selection and not self._selected_city_is_used()
                                      else tk.DISABLED)

    def refresh_usage_state(self) -> None:
        """Recalcule les actions depuis le draft d'hypothese courant."""
        self._refresh_action_states()

    def _rename_selected_city(self) -> None:
        city_id = self._selected_city_ref_id
        if city_id is None:
            return
        city = self._reference.cities[city_id]
        name = simpledialog.askstring("Renommer le point", "Nouveau nom :",
                                      initialvalue=city.name, parent=self.winfo_toplevel())
        if name is None:
            return
        try:
            self._reference.rename_city(city_id, name)
        except ValueError as exc:
            messagebox.showerror("Nom de la ville", str(exc), parent=self.winfo_toplevel())
            return
        self.refresh()
        if self._on_reference_changed is not None:
            self._on_reference_changed()

    def _delete_selected_city(self) -> None:
        city_id = self._selected_city_ref_id
        if city_id is None or self._selected_city_is_used():
            return
        for triangle in self._reference.get_triangles_referencing_city(city_id):
            self._reference.remove_triangle(triangle.triangle_ref_id)
        self._reference.remove_city(city_id)
        self._selected_city_ref_id = None
        self.refresh()
        if self._on_reference_changed is not None:
            self._on_reference_changed()

    def refresh(self, scenario_reference: ScenarioReference | None = None) -> None:
        """Affiche toutes les SCITY publiees, y compris les orphelines."""
        if scenario_reference is not None and scenario_reference is not self._reference:
            self._reference = scenario_reference
        cities = sorted(self._reference.cities.values(), key=lambda city: (city.name.casefold(), city.city_ref_id))
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
        self.map_view.set_markers(GeoMapMarker(city.city_ref_id, city.latitude, city.longitude, city.name)
                                  for city in cities)
        self.map_view.set_selected_marker(self._selected_city_ref_id)
        self._refresh_action_states()

    def _on_list_selected(self, _event=None) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self._scenario_city_ids):
            self._select_city(self._scenario_city_ids[index], recenter=True)

    def _on_map_marker_selected(self, city_ref_id) -> None:
        self._select_city(city_ref_id if city_ref_id in self._scenario_city_ids else None)

    def _select_city(self, city_ref_id: str | None, *, recenter: bool = False) -> None:
        self._selected_city_ref_id = city_ref_id
        self.listbox.selection_clear(0, tk.END)
        if city_ref_id is None:
            self.map_view.set_selected_marker(None)
            self._refresh_action_states()
            return
        index = self._scenario_city_ids.index(city_ref_id)
        self.listbox.selection_set(index)
        self.listbox.activate(index)
        self.listbox.see(index)
        self.map_view.set_selected_marker(city_ref_id, recenter=recenter)
        self._refresh_action_states()
