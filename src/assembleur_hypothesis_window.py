"""Editeur transactionnel d'une :class:`ScenarioHypothesis` manuelle."""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

from src.assembleur_catalogue import Catalogue, CatalogueTriangle
from src.assembleur_catalogue_window import TemplatePairRow, TemplateRankSlot
from src.assembleur_scenario import (
    HypothesisImpact,
    ScenarioHypothesis,
    ScenarioHypothesisChangePlan,
    analyze_hypothesis_change,
    create_hypothesis_from_template,
)
from src.assembleur_geometry_reference import GeometryReferenceResolver
from src.assembleur_geometry_reference import ScenarioReference
from src.assembleur_scenario_cities_view import ScenarioCitiesView


_NOTE_ORDER = {"do": 0, "si": 1, "la": 2, "sol": 3, "fa": 4, "mi": 5, "re": 6, "zone": 7}
_BASE_COLUMN_WIDTH = 120
_DRAG_THRESHOLD = 6


@dataclass(frozen=True)
class ScenarioHypothesisDialogResult:
    hypothesis: ScenarioHypothesis
    reference: ScenarioReference


class ScenarioHypothesisDialog(tk.Toplevel):
    """Edite un draft indépendant, sans jamais toucher à la topologie."""

    def __init__(
        self,
        parent,
        *,
        catalogue: Catalogue,
        hypothesis: ScenarioHypothesis,
        resolver: GeometryReferenceResolver,
        scenario_reference: ScenarioReference,
    ):
        super().__init__(parent)
        self.title("Modifier l'hypothèse du scénario")
        self.transient(parent)
        self.geometry("1450x780")
        self.minsize(1100, 620)
        self.catalogue = catalogue
        self.resolver = resolver
        self._original_reference = scenario_reference
        self._reference_draft = scenario_reference.clone()
        self._original = hypothesis
        self._draft = hypothesis.clone()
        self.result: ScenarioHypothesisDialogResult | None = None
        self.change_plan: ScenarioHypothesisChangePlan | None = None
        self._template_ids: list[str] = []
        self._triangle_by_tree_iid: dict[str, str] = {}
        self._selected_slot: TemplateRankSlot | None = None
        self._drag_triangle_id: str | None = None
        self._drag_started = False
        self._drag_start_root: tuple[int, int] | None = None
        self._drag_source_slot: TemplateRankSlot | None = None
        self._drag_target_slot: TemplateRankSlot | None = None
        self._drag_ghost: tk.Toplevel | None = None
        self._template_var = tk.StringVar()
        self._note_var = tk.StringVar(value="Tous")
        self._base_var = tk.StringVar()
        self._light_var = tk.StringVar()
        self._build_ui()
        self._refresh_template_selector()
        self._refresh_triangle_tree()
        self._refresh_ranks()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.grab_set()

    def show(self) -> ScenarioHypothesisDialogResult | None:
        self.wait_window()
        return self.result

    def _build_ui(self) -> None:
        dialog_root = ttk.Frame(self, padding=10)
        dialog_root.pack(fill=tk.BOTH, expand=True)
        dialog_root.rowconfigure(0, weight=1)
        dialog_root.columnconfigure(0, weight=1)

        self._notebook = ttk.Notebook(dialog_root)
        self._notebook.grid(row=0, column=0, sticky="nsew")
        self._cities_tab = ttk.Frame(self._notebook)
        self._triangles_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._cities_tab, text="Villes")
        self._notebook.add(self._triangles_tab, text="Liste des triangles")
        self.cities_view = ScenarioCitiesView(
            self._cities_tab,
            scenario_reference=self._reference_draft,
            catalogue=self.catalogue,
            on_reference_changed=self._refresh_ranks,
        )
        self.cities_view.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        triangles_root = self._triangles_tab
        triangles_root.rowconfigure(2, weight=1)
        triangles_root.columnconfigure(0, weight=1)

        template_row = ttk.Frame(triangles_root)
        template_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        template_row.columnconfigure(1, weight=1)
        ttk.Label(template_row, text="Repartir du template :").grid(row=0, column=0, sticky="w")
        self._template_combo = ttk.Combobox(template_row, textvariable=self._template_var, state="readonly")
        self._template_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self._template_combo.bind("<<ComboboxSelected>>", self._on_template_selected)

        filters = ttk.Frame(triangles_root)
        filters.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(filters, text="Note :").grid(row=0, column=0)
        self._note_combo = ttk.Combobox(
            filters, textvariable=self._note_var, state="readonly", width=7,
        )
        self._note_combo.grid(row=0, column=1, padx=(0, 8))
        ttk.Label(filters, text="Base :").grid(row=0, column=2)
        ttk.Entry(filters, textvariable=self._base_var, width=16).grid(
            row=0, column=3, padx=(0, 8),
        )
        ttk.Label(filters, text="Lumière :").grid(row=0, column=4)
        ttk.Entry(filters, textvariable=self._light_var, width=16).grid(row=0, column=5)
        self._note_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_triangle_tree(),
        )
        self._base_var.trace_add("write", lambda *_: self._refresh_triangle_tree())
        self._light_var.trace_add("write", lambda *_: self._refresh_triangle_tree())

        content = ttk.Frame(triangles_root)
        content.grid(row=2, column=0, sticky="nsew")
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, minsize=400, weight=0)
        content.columnconfigure(2, minsize=650, weight=1)

        source = ttk.Frame(content, padding=(0, 0, 8, 0))
        source.grid(row=0, column=0, sticky="nsew")
        ttk.Separator(content, orient=tk.VERTICAL).grid(
            row=0, column=1, sticky="ns", padx=(0, 8),
        )
        target = ttk.Frame(content, padding=(0, 0, 0, 0))
        target.grid(row=0, column=2, sticky="nsew")
        self._build_source(source)
        self._build_ranks(target)

        buttons = ttk.Frame(dialog_root)
        buttons.grid(row=1, column=0, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Appliquer", command=self._apply).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Annuler", command=self.destroy).pack(side=tk.LEFT, padx=(6, 0))

    def _build_source(self, parent) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self._tree = ttk.Treeview(parent, show="tree", selectmode="browse")
        self._tree.heading("#0", text="Catalogue")
        self._tree.column("#0", stretch=True, minwidth=190)
        self._tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self._tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._tree.bind("<ButtonPress-1>", self._on_tree_press, add="+")
        self._tree.bind("<B1-Motion>", self._on_tree_drag, add="+")
        self._tree.bind("<ButtonRelease-1>", self._on_tree_release, add="+")

    def _build_ranks(self, parent) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self._canvas = tk.Canvas(parent, highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self._canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._canvas.configure(yscrollcommand=scrollbar.set)
        self._grid = ttk.Frame(self._canvas, padding=(0, 0, 4, 0))
        self._grid_window = self._canvas.create_window((0, 0), window=self._grid, anchor="nw")
        self._grid.bind("<Configure>", lambda _event: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", self._resize_columns)
        for column, text in enumerate(("Base", "Rang impair", "Rang pair")):
            ttk.Label(self._grid, text=text, anchor="center", padding=(6, 5), relief=tk.RIDGE, font=(None, 9, "bold")).grid(row=0, column=column, sticky="nsew")
        self._pair_rows: list[TemplatePairRow] = []
        for pair_number in range(1, 17):
            row = TemplatePairRow(self._grid, pair_number, self._clear_pair_forbidden)
            row.grid(row=pair_number, column=0, columnspan=3, sticky="ew", pady=(3, 0))
            row.base_slot.clear_button.grid_remove()
            self._pair_rows.append(row)
            for slot in (row.odd_rank_slot, row.even_rank_slot):
                for widget in (slot, slot._badge, slot._entry):
                    widget.bind("<ButtonPress-1>", lambda _event, value=slot: self._select_slot(value), add="+")
        self.after_idle(lambda: self._resize_columns(None))

    def _resize_columns(self, event) -> None:
        width = event.width if event is not None else self._canvas.winfo_width()
        self._canvas.itemconfigure(self._grid_window, width=width)
        base_width = min(_BASE_COLUMN_WIDTH, max(90, width - 220))
        rank_width = max(110, (max(width, base_width + 220) - base_width) // 2)
        for column, minimum in enumerate((base_width, rank_width, rank_width)):
            self._grid.columnconfigure(column, minsize=minimum, weight=0)
        for row in self._pair_rows:
            row.set_column_widths(base_width, rank_width)

    def _clear_pair_forbidden(self, _row: TemplatePairRow) -> None:
        raise RuntimeError("Une ScenarioHypothesis ne peut pas contenir de rang vide.")

    def _refresh_template_selector(self) -> None:
        templates = [
            template for template in self.catalogue.iter_templates()
            if not template.archived and self.catalogue.get_template_validation_status(template.template_id).state == "Valide"
        ]
        self._template_ids = [template.template_id for template in templates]
        names = [template.name for template in templates]
        source_id = self._draft.source_template_id
        if source_id in self._template_ids:
            self._template_var.set(names[self._template_ids.index(source_id)])
        elif source_id is not None:
            self._template_var.set("(Template source indisponible)")
        else:
            self._template_var.set("")
        self._template_combo.configure(values=tuple(names))

    def _on_template_selected(self, _event=None) -> None:
        index = self._template_combo.current()
        if index < 0:
            return
        self._replace_draft_from_template(self._template_ids[index])
        self._refresh_ranks()

    def _replace_draft_from_template(self, template_id: str) -> None:
        """Repart transactionnellement d'une copie indépendante du template choisi."""
        template = self.catalogue.get_template(template_id)
        self._draft = create_hypothesis_from_template(self.catalogue, template)

    def _note_key(self, note: str) -> tuple[int, str]:
        normalized = note.strip().casefold()
        return _NOTE_ORDER.get(normalized, len(_NOTE_ORDER)), normalized

    def _visible_triangles(self) -> list[CatalogueTriangle]:
        note, base, light = self._note_var.get(), self._base_var.get().strip().casefold(), self._light_var.get().strip().casefold()
        return sorted(
            (
                triangle for triangle in self.catalogue.iter_triangles()
                if not triangle.archived
                and (note == "Tous" or triangle.note == note)
                and (not base or base in self.catalogue.get_city(triangle.base_city_id).name.casefold())
                and (not light or light in self.catalogue.get_city(triangle.light_city_id).name.casefold())
            ),
            key=lambda triangle: (self._note_key(triangle.note), self.catalogue.get_city(triangle.base_city_id).name.casefold(), self.catalogue.get_city(triangle.light_city_id).name.casefold()),
        )

    def _refresh_triangle_tree(self) -> None:
        active_notes = sorted({triangle.note for triangle in self.catalogue.iter_triangles() if not triangle.archived}, key=self._note_key)
        self._note_combo.configure(values=("Tous", *active_notes))
        if self._note_var.get() not in {"Tous", *active_notes}:
            self._note_var.set("Tous")
        self._tree.delete(*self._tree.get_children())
        self._triangle_by_tree_iid = {}
        grouped: dict[str, list[CatalogueTriangle]] = {}
        for triangle in self._visible_triangles():
            grouped.setdefault(self.catalogue.get_city(triangle.base_city_id).name, []).append(triangle)
        for base_index, base in enumerate(sorted(grouped, key=str.casefold)):
            group_id = f"base-{base_index}"
            self._tree.insert("", tk.END, iid=group_id, text=base, open=True)
            for item_index, triangle in enumerate(grouped[base]):
                leaf_id = f"{group_id}-triangle-{item_index}"
                self._tree.insert(group_id, tk.END, iid=leaf_id, text=f"{triangle.note} : {self.catalogue.get_city(triangle.light_city_id).name}")
                self._triangle_by_tree_iid[leaf_id] = triangle.triangle_id

    def _refresh_ranks(self) -> None:
        resolver = self._draft_resolver()
        ranks = self._draft.triangle_ids_by_rank
        for pair, row in enumerate(self._pair_rows):
            odd, even = (
                resolver.resolve_triangle(ranks[pair * 2]),
                resolver.resolve_triangle(ranks[pair * 2 + 1]),
            )
            odd_light = resolver.resolve_city(odd.light_city_ref_id).name
            even_light = resolver.resolve_city(even.light_city_ref_id).name
            base = (
                resolver.resolve_city(odd.base_city_ref_id).name
                if odd.base_city_ref_id == even.base_city_ref_id
                else "Bases différentes"
            )
            row.set_triangles(odd.ref_id, odd_light, even.ref_id, even_light, base)
        if self._selected_slot is not None:
            self._selected_slot.set_selected(True)

    def _draft_resolver(self) -> GeometryReferenceResolver:
        reference = getattr(self, "_reference_draft", None)
        if reference is None:
            return self.resolver
        return GeometryReferenceResolver(self.catalogue, reference)

    def _select_slot(self, slot: TemplateRankSlot) -> None:
        if self._selected_slot is not None and self._selected_slot is not slot:
            self._selected_slot.set_selected(False)
        self._selected_slot = slot
        slot.set_selected(True)

    def _slot_at(self, widget) -> TemplateRankSlot | None:
        while widget is not None:
            if isinstance(widget, TemplateRankSlot):
                return widget
            widget = widget.master
        return None

    def _on_tree_press(self, event) -> None:
        iid = self._tree.identify_row(event.y)
        self._drag_triangle_id = self._triangle_by_tree_iid.get(iid)
        self._drag_start_root = (event.x_root, event.y_root) if self._drag_triangle_id is not None else None
        self._drag_started = False

    def _on_tree_drag(self, event) -> None:
        triangle_id, start = self._drag_triangle_id, self._drag_start_root
        if triangle_id is None or start is None:
            return
        if not self._drag_started:
            if math.hypot(event.x_root - start[0], event.y_root - start[1]) < _DRAG_THRESHOLD:
                return
            self._drag_started = True
            source_index = self._find_rank(triangle_id)
            if source_index is not None:
                self._drag_source_slot = self._slot_for_rank(source_index)
                self._drag_source_slot.set_drop_state("source")
            self._show_ghost(triangle_id, event.x_root, event.y_root)
            self._set_cursor("hand2")
        self._move_ghost(event.x_root, event.y_root)
        slot = self._slot_at(self.winfo_containing(event.x_root, event.y_root))
        if slot is self._drag_target_slot:
            return
        if self._drag_target_slot is not None:
            self._restore_slot(self._drag_target_slot)
        self._drag_target_slot = slot
        if slot is not None:
            action, valid, _message, _preview = self._plan_drop(triangle_id, slot)
            slot.set_drop_state({"valid": "valid", "replace": "replace", "swap": "swap", "noop": "normal", "invalid": "invalid"}[action])

    def _on_tree_release(self, _event) -> None:
        if self._drag_started and self._drag_triangle_id is not None and self._drag_target_slot is not None:
            action, valid, message, preview = self._plan_drop(self._drag_triangle_id, self._drag_target_slot)
            if valid and action != "noop":
                self._draft.triangle_ids_by_rank[:] = preview
                self._select_slot(self._drag_target_slot)
                self._refresh_ranks()
            elif message:
                messagebox.showwarning("Hypothèse", message, parent=self)
        self._reset_drag()

    def _find_rank(self, triangle_id: str) -> int | None:
        return next((index for index, value in enumerate(self._draft.triangle_ids_by_rank) if value == triangle_id), None)

    def _slot_for_rank(self, rank_index: int) -> TemplateRankSlot:
        row = self._pair_rows[rank_index // 2]
        return row.odd_rank_slot if rank_index % 2 == 0 else row.even_rank_slot

    def _plan_drop(self, triangle_id: str, slot: TemplateRankSlot) -> tuple[str, bool, str | None, list[str]]:
        target_index = slot.rank_number - 1
        ranks = self._draft.triangle_ids_by_rank
        source_index = self._find_rank(triangle_id)
        if source_index == target_index:
            return "noop", True, None, list(ranks)
        preview = list(ranks)
        target_triangle_id = preview[target_index]
        if source_index is None:
            preview[target_index] = triangle_id
            action = "replace"
        else:
            preview[source_index] = target_triangle_id
            preview[target_index] = triangle_id
            action = "swap"
        try:
            ScenarioHypothesis(preview, self._draft.source_template_id).validate(
                self._draft_resolver()
            )
        except ValueError as exc:
            return "invalid", False, str(exc), list(ranks)
        return action, True, None, preview

    def _restore_slot(self, slot: TemplateRankSlot) -> None:
        slot.set_drop_state("source" if slot is self._drag_source_slot else "normal")

    def _show_ghost(self, triangle_id: str, x_root: int, y_root: int) -> None:
        self._destroy_ghost()
        ghost = tk.Toplevel(self)
        ghost.overrideredirect(True)
        ghost.configure(background="#ffffff", borderwidth=1, relief=tk.SOLID)
        triangle = self.catalogue.get_triangle(triangle_id)
        ttk.Label(ghost, text=self.catalogue.get_city(triangle.light_city_id).name, padding=(10, 6), font=(None, 9, "bold")).pack()
        self._drag_ghost = ghost
        self._move_ghost(x_root, y_root)

    def _move_ghost(self, x_root: int, y_root: int) -> None:
        if self._drag_ghost is not None and self._drag_ghost.winfo_exists():
            self._drag_ghost.geometry(f"+{x_root + 16}+{y_root + 16}")

    def _destroy_ghost(self) -> None:
        if self._drag_ghost is not None and self._drag_ghost.winfo_exists():
            self._drag_ghost.destroy()
        self._drag_ghost = None

    def _set_cursor(self, cursor: str) -> None:
        self._tree.configure(cursor=cursor)
        for row in self._pair_rows:
            for slot in (row.odd_rank_slot, row.even_rank_slot):
                for widget in (slot, slot._badge, slot._entry):
                    widget.configure(cursor=cursor)

    def _reset_drag(self) -> None:
        if self._drag_target_slot is not None:
            self._drag_target_slot.set_drop_state("normal")
        if self._drag_source_slot is not None and self._drag_source_slot is not self._drag_target_slot:
            self._drag_source_slot.set_drop_state("normal")
        self._drag_triangle_id = None
        self._drag_started = False
        self._drag_start_root = None
        self._drag_source_slot = None
        self._drag_target_slot = None
        self._destroy_ghost()
        self._set_cursor("")

    def _apply(self) -> None:
        if hasattr(self, "cities_view"):
            if not self.cities_view._on_name_committed():
                return
        try:
            resolver = self._draft_resolver()
            self._draft.validate(resolver)
        except ValueError as exc:
            messagebox.showerror("Hypothèse", str(exc), parent=self)
            return
        self.change_plan = analyze_hypothesis_change(
            resolver, self._original, self._draft
        )
        self.result = ScenarioHypothesisDialogResult(
            hypothesis=self._draft.clone(),
            reference=self._reference_draft.clone(),
        )
        self.destroy()
