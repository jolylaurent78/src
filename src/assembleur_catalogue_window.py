"""Fenêtre UX autonome du prototype de gestion du catalogue."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.assembleur_dms_editor import DmsCoordinateEditor
from src.assembleur_geo_map_view import GeoMapMarker, GeoMapView


@dataclass
class CatalogueCity:
    """Ville temporaire : état d'interface seulement, sans persistance ni identifiant."""

    name: str
    latitude: float
    longitude: float
    archived: bool = False
    usage: int = 0


class CatalogueWindow(tk.Toplevel):
    """Prototype non modal du catalogue et de son premier onglet Villes."""

    _CSV_HEADER = ("Nom", "Latitude", "Longitude")

    def __init__(self, parent, *, maps_dir: str | Path | None = None):
        super().__init__(parent)
        self.title("Gestion du catalogue")
        self.geometry("1000x700")
        self.minsize(760, 500)
        self._maps_dir = maps_dir
        self.cities: list[CatalogueCity] = []
        self._selected_city: CatalogueCity | None = None
        self._updating_detail = False
        self._validated_cities: list[CatalogueCity] = []
        self._is_dirty = False

        self._search_var = tk.StringVar()
        self._show_archived_var = tk.BooleanVar(value=False)
        self._name_var = tk.StringVar()
        self._archived_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._refresh_city_list()
        self._load_map()
        self.protocol("WM_DELETE_WINDOW", self.request_close)

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        notebook = ttk.Notebook(root)
        notebook.grid(row=0, column=0, sticky="nsew")
        self._cities_tab = ttk.Frame(notebook, padding=8)
        notebook.add(self._cities_tab, text="Villes (0)")
        notebook.add(ttk.Frame(notebook), text="Triangles (0)")
        notebook.add(ttk.Frame(notebook), text="Templates (0)")
        self._catalogue_notebook = notebook
        self._build_cities_tab()

        bottom = ttk.Frame(root)
        bottom.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(bottom, text="Importer...", command=self._import_csv).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Exporter...", state=tk.DISABLED).pack(side=tk.LEFT, padx=(6, 0))
        edit_actions = ttk.Frame(bottom)
        edit_actions.pack(side=tk.RIGHT)
        self._apply_button = ttk.Button(edit_actions, text="Appliquer", command=self._apply_changes, state=tk.DISABLED)
        self._apply_button.pack(side=tk.LEFT)
        self._cancel_button = ttk.Button(edit_actions, text="Annuler", command=self._cancel_changes, state=tk.DISABLED)
        self._cancel_button.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(edit_actions, text="Fermer", command=self.request_close).pack(side=tk.LEFT, padx=(6, 0))

    def _build_cities_tab(self):
        self._cities_tab.rowconfigure(0, weight=1)
        self._cities_tab.columnconfigure(0, weight=1)
        panes = ttk.PanedWindow(self._cities_tab, orient=tk.HORIZONTAL)
        panes.grid(row=0, column=0, sticky="nsew")
        master = ttk.Frame(panes, padding=(0, 0, 8, 0))
        detail = ttk.Frame(panes, padding=(8, 0, 0, 0))
        panes.add(master, weight=1)
        panes.add(detail, weight=2)

        ttk.Label(master, text="Rechercher").pack(anchor="w")
        ttk.Entry(master, textvariable=self._search_var).pack(fill=tk.X, pady=(2, 8))
        self._search_var.trace_add("write", lambda *_: self._refresh_city_list())
        ttk.Checkbutton(master, text="Afficher les villes archivées", variable=self._show_archived_var,
                        command=self._refresh_city_list).pack(anchor="w", pady=(0, 8))
        actions = ttk.Frame(master)
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text="Ajouter", command=self._add_city).pack(side=tk.LEFT)
        ttk.Button(actions, text="Archiver", command=self._archive_selected_city).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Supprimer", command=self._delete_selected_city).pack(side=tk.LEFT)
        list_frame = ttk.Frame(master)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self._city_listbox = tk.Listbox(list_frame, exportselection=False)
        self._city_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._city_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._city_listbox.configure(yscrollcommand=scrollbar.set)
        self._city_listbox.bind("<<ListboxSelect>>", self._on_city_selected)

        detail.rowconfigure(4, weight=1)
        detail.columnconfigure(1, weight=1)
        ttk.Label(detail, text="Nom").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(detail, textvariable=self._name_var).grid(row=0, column=1, sticky="ew", pady=(0, 6))
        ttk.Label(detail, text="Coordonnées").grid(row=1, column=0, sticky="w", pady=(0, 6))
        coordinates = ttk.Frame(detail)
        coordinates.grid(row=1, column=1, sticky="w", pady=(0, 6))
        self._latitude_editor = DmsCoordinateEditor(
            coordinates, coordinate_type="latitude", on_change=self._save_detail_changes
        )
        self._latitude_editor.grid(row=0, column=0, sticky="w")
        self._longitude_editor = DmsCoordinateEditor(
            coordinates, coordinate_type="longitude", on_change=self._save_detail_changes
        )
        self._longitude_editor.grid(row=0, column=1, sticky="w", padx=(12, 0))
        icon_path = Path(__file__).resolve().parent.parent / "images" / "clipboard.png"
        self._clipboard_icon = tk.PhotoImage(file=icon_path)
        ttk.Button(coordinates, image=self._clipboard_icon, command=self._paste_coordinates).grid(
            row=0, column=2, sticky="w", padx=(6, 0)
        )
        ttk.Checkbutton(detail, text="Ville archivée", variable=self._archived_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 6)
        )
        usage = ttk.Frame(detail)
        usage.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(usage, text="Référencée par :").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(usage, text="Triangles : 0").pack(side=tk.LEFT)
        ttk.Button(usage, text="...", width=3, state=tk.DISABLED).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(usage, text="Balises : 0").pack(side=tk.LEFT)
        ttk.Button(usage, text="...", width=3, state=tk.DISABLED).pack(side=tk.LEFT, padx=(4, 0))
        self._map_view = GeoMapView(
            detail,
            on_marker_selected=self._on_map_marker_selected,
            initial_fit_zoom=2.25,
            minimum_fit_zoom=2.25,
        )
        self._map_view.grid(row=4, column=0, columnspan=2, sticky="nsew")

        self._name_var.trace_add("write", self._save_detail_changes)
        self._archived_var.trace_add("write", self._save_detail_changes)

    def _load_map(self):
        try:
            self._map_view.load_map("france_michelin", self._maps_dir)
        except (FileNotFoundError, OSError, ValueError) as exc:
            messagebox.showwarning("Carte du catalogue", str(exc), parent=self)

    @staticmethod
    def _clone_cities(cities: list[CatalogueCity]) -> list[CatalogueCity]:
        return [
            CatalogueCity(city.name, city.latitude, city.longitude, city.archived, city.usage)
            for city in cities
        ]

    def _set_dirty(self, is_dirty: bool) -> None:
        self._is_dirty = is_dirty
        state = tk.NORMAL if is_dirty else tk.DISABLED
        self._apply_button.configure(state=state)
        self._cancel_button.configure(state=state)

    def _mark_dirty(self) -> None:
        self._set_dirty(True)

    def _apply_changes(self) -> None:
        """Valide uniquement l'état courant de cette session, sans persistance."""
        self._validated_cities = self._clone_cities(self.cities)
        self._set_dirty(False)

    def _cancel_changes(self) -> None:
        """Restaure le dernier instantané validé localement."""
        self.cities = self._clone_cities(self._validated_cities)
        self._selected_city = None
        self._refresh_city_list()
        self._load_selected_city()
        self._set_dirty(False)

    def request_close(self) -> bool:
        """Ferme après confirmation lorsqu'un état local non validé existe."""
        if self._is_dirty and not messagebox.askyesno(
            "Gestion du catalogue",
            "Des modifications ne sont pas appliquées. Fermer sans les appliquer ?",
            parent=self,
        ):
            return False
        self.destroy()
        return True

    def _import_csv(self):
        path = filedialog.askopenfilename(parent=self, title="Importer des villes",
                                          filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")])
        if not path:
            return
        try:
            self.cities.extend(self._read_cities_csv(path))
        except (OSError, UnicodeError, ValueError, csv.Error) as exc:
            messagebox.showerror("Importer des villes", str(exc), parent=self)
            return
        self._refresh_city_list()
        self._mark_dirty()

    @classmethod
    def _read_cities_csv(cls, path: str) -> list[CatalogueCity]:
        with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
            rows = [row for row in csv.reader(csv_file, delimiter=";") if any(str(value).strip() for value in row)]
        if not rows:
            raise ValueError("Le fichier CSV est vide.")
        if tuple(str(value).strip() for value in rows[0]) != cls._CSV_HEADER:
            raise ValueError("L'en-tête CSV doit être : Nom;Latitude;Longitude")
        cities = []
        for line_number, row in enumerate(rows[1:], start=2):
            if len(row) != 3:
                raise ValueError(f"Ligne {line_number} : trois colonnes sont attendues.")
            name, raw_latitude, raw_longitude = (str(value).strip() for value in row)
            if not name:
                raise ValueError(f"Ligne {line_number} : nom de ville vide.")
            cities.append(CatalogueCity(
                name,
                DmsCoordinateEditor.parse_coordinate(raw_latitude, "latitude"),
                DmsCoordinateEditor.parse_coordinate(raw_longitude, "longitude"),
            ))
        return cities

    def _visible_cities(self) -> list[CatalogueCity]:
        search = self._search_var.get().strip().casefold()
        return [city for city in self.cities if (self._show_archived_var.get() or not city.archived)
                and (not search or search in city.name.casefold())]

    def _refresh_city_list(self):
        selected, visible = self._selected_city, self._visible_cities()
        self._city_listbox.delete(0, tk.END)
        for city in visible:
            self._city_listbox.insert(tk.END, city.name)
        if selected in visible:
            index = visible.index(selected)
            self._city_listbox.selection_set(index)
            self._city_listbox.activate(index)
        elif selected is not None:
            self._selected_city = None
            self._load_selected_city()
        self._catalogue_notebook.tab(self._cities_tab, text=f"Villes ({len(self.cities)})")
        self._map_view.set_markers(
            GeoMapMarker(id(city), city.latitude, city.longitude, city.name) for city in self.cities
        )
        self._map_view.set_selected_marker(id(self._selected_city) if self._selected_city else None)

    def _on_city_selected(self, _event=None):
        selection, visible = self._city_listbox.curselection(), self._visible_cities()
        self._selected_city = visible[selection[0]] if selection else None
        self._load_selected_city()
        self._map_view.set_selected_marker(id(self._selected_city) if self._selected_city else None, recenter=True)

    def _on_map_marker_selected(self, marker_id):
        self._selected_city = next((city for city in self.cities if id(city) == marker_id), None)
        self._refresh_city_list()
        self._load_selected_city()

    def _load_selected_city(self):
        self._updating_detail = True
        city = self._selected_city
        self._name_var.set(city.name if city else "")
        self._latitude_editor.set_decimal(city.latitude if city else 0.0)
        self._longitude_editor.set_decimal(city.longitude if city else 0.0)
        self._archived_var.set(city.archived if city else False)
        self._updating_detail = False

    def _save_detail_changes(self, *_args):
        if self._updating_detail or self._selected_city is None:
            return
        city = self._selected_city
        city.name = self._name_var.get().strip()
        city.latitude = self._latitude_editor.get_decimal()
        city.longitude = self._longitude_editor.get_decimal()
        city.archived = bool(self._archived_var.get())
        self._refresh_city_list()
        self._mark_dirty()

    def _paste_coordinates(self):
        try:
            latitude, longitude = DmsCoordinateEditor.parse_coordinate_pair(self.clipboard_get())
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Coller coordonnées", str(exc), parent=self)
            return
        self._latitude_editor.set_decimal(latitude)
        self._longitude_editor.set_decimal(longitude)
        self._save_detail_changes()

    def _add_city(self):
        self._selected_city = CatalogueCity("Nouvelle ville", 0.0, 0.0)
        self.cities.append(self._selected_city)
        self._refresh_city_list()
        self._load_selected_city()
        self._mark_dirty()

    def _archive_selected_city(self):
        if self._selected_city is not None:
            self._selected_city.archived = True
            self._refresh_city_list()
            self._load_selected_city()
            self._mark_dirty()

    def _delete_selected_city(self):
        if self._selected_city is not None:
            self.cities.remove(self._selected_city)
            self._selected_city = None
            self._refresh_city_list()
            self._load_selected_city()
            self._mark_dirty()
