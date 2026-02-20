"""
TriangleViewerDictionaryMixin

Ce module est généré pour découper assembleur_tk.py.
"""

from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk, messagebox
from tksheet import Sheet
from typing import Optional, Tuple

from src.DictionnaireEnigmes import DictionnaireEnigmes, DicoScope, _normalizeWordLocal

DICO_TAG_EXCLURE = "exclure"
CFG_KEY_DICO_EXCLURE_MOTS_CODES = "dicoExclureMotsCodes"


class TriangleViewerDictionaryMixin:
    """Mixin: méthodes extraites de assembleur_tk.py."""
    pass

    def _initDicoExcludeMotsCodesFromConfig(self) -> None:
        exclude = self.getAppConfigValue(CFG_KEY_DICO_EXCLURE_MOTS_CODES, False)
        if not isinstance(exclude, bool):
            raise ValueError(
                f"Invalid config type for {CFG_KEY_DICO_EXCLURE_MOTS_CODES}: {type(exclude).__name__}"
            )
        self._dicoExcludeMotsCodesValue = exclude
        self._dicoExcludeMotsCodesVar = tk.BooleanVar(value=exclude)

    def _getDicoTagExclure(self) -> str | None:
        self._dicoExcludeMotsCodesValue = bool(self._dicoExcludeMotsCodesVar.get())
        return DICO_TAG_EXCLURE if self._dicoExcludeMotsCodesValue else None

    def _onToggleDicoExcludeMotsCodes(self) -> None:
        value = bool(self._dicoExcludeMotsCodesVar.get())
        self._dicoExcludeMotsCodesValue = value
        self.setAppConfigValue(CFG_KEY_DICO_EXCLURE_MOTS_CODES, value)
        self.saveAppConfig()

        tagExclure = self._getDicoTagExclure()
        self._init_dictionary(tagExclure=tagExclure)
        self._build_dico_grid()

    def _init_dictionary(self, *, tagExclure: str | None) -> None:
        """Construit le dictionnaire en lisant ../data/livre.txt."""
        if DictionnaireEnigmes is None:
            raise ImportError("Module DictionnaireEnigmes introuvable")
        livre_path = os.path.join(self.data_dir, "livre.txt")
        if not os.path.isfile(livre_path):
            raise FileNotFoundError(livre_path)
        self.dico = DictionnaireEnigmes(livre_path, tagExclure=tagExclure)
        nb_lignes = len(self.dico)
        self.status.config(text=f"Dico chargé: {nb_lignes} lignes depuis {livre_path}")

    # ---------- Dictionnaire : affichage dans le panneau bas ----------
    def _build_dico_grid(self):
        """
        Construit/affiche la grille tksheet du dictionnaire dans self.dicoPanel.
        N’opère que si self.dico est chargé et tksheet disponible.
        """

        # Le panneau bas doit exister (créé dans _build_canvas)
        if not hasattr(self, "dicoPanel"):
            return
        # Nettoyer le contenu existant (placeholder, ancienne grille…)
        for child in list(self.dicoPanel.children.values()):
            child.destroy()

        # Vérifs
        if self.dico is None:
            tk.Label(self.dicoPanel, text="Dictionnaire non chargé",
                     bg="#f3f3f3", anchor="w").pack(fill="x", padx=8, pady=6)
            return

        # ===== Paramètres Dico (N) + référentiel logique (volatile) =====
        # On affiche toujours 2N colonnes correspondant à j ∈ [-N .. N-1].
        # Le "0 logique" doit correspondre au premier mot du livre (j=0),
        # donc à la colonne physique c = N (pas à la colonne 0).
        self._dico_nb_mots_max = self.dico.nbMotMax()

        # Origine logique = cellule physique (row0,col0) qui correspond à (0,0) logique.
        # Par défaut / reset : (0, N) physique => 0 logique sur le 1er mot du livre.
        if not hasattr(self, "_dico_origin_cell") or self._dico_origin_cell is None:
            self._dico_origin_cell = (0, self._dico_nb_mots_max)

        # Mode de référence (volatile, RAM) :
        #   None      -> mode ABS
        #   "origin"  -> relatif normal (addition)
        #   "target"  -> relatif inversé (soustraction)
        # Compat : si une origine non-default existe sans mode, on considère "origin".
        if not hasattr(self, "_dico_ref_mode"):
            default_origin = (0, self._dico_nb_mots_max)
            self._dico_ref_mode = "origin" if tuple(self._dico_origin_cell) != tuple(default_origin) else None

        # --- Layout du panneau bas : [sidebar recherche] | [grille] ---
        container = tk.Frame(self.dicoPanel, bg="#f3f3f3")
        container.pack(fill="both", expand=True)
        # colonne gauche (recherche + occurrences)
        left = tk.Frame(container, width=180, bg="#f3f3f3", bd=1, relief=tk.GROOVE)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        # colonne droite (grille)
        right = tk.Frame(container, bg="#f3f3f3")
        right.pack(side="left", fill="both", expand=True)

        # ===== Barre "recherche" =====
        tk.Checkbutton(
            left,
            text="Exclure mots codés",
            variable=self._dicoExcludeMotsCodesVar,
            command=self._onToggleDicoExcludeMotsCodes,
            bg="#f3f3f3",
        ).pack(anchor="w", padx=8, pady=(8, 2))
        search_row = tk.Frame(left, bg="#f3f3f3")
        search_row.pack(fill="x", padx=8, pady=(2, 6))

        self._dico_search_var = tk.StringVar(value="")
        self._dico_search_entry = tk.Entry(search_row, textvariable=self._dico_search_var)
        self._dico_search_entry.pack(side="left", fill="x", expand=True)

        self._dico_icon_check_green = self._load_icon("check16_green.png")
        self._dico_icon_check_red = self._load_icon("check16_red.png")
        if self._dico_icon_check_green is None:
            raise FileNotFoundError("Icone introuvable: check16_green.png")
        if self._dico_icon_check_red is None:
            raise FileNotFoundError("Icone introuvable: check16_red.png")

        self._dico_search_status_icon = tk.Label(search_row, image=self._dico_icon_check_red, bg="#f3f3f3")
        self._dico_search_status_icon.image = self._dico_icon_check_red
        self._dico_search_status_icon.pack(side="left", padx=(4, 2))

        tree_frame = tk.Frame(left, bg="#f3f3f3")
        tree_frame.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        self._dico_search_tree = ttk.Treeview(
            tree_frame,
            columns=("Col", "Ligne"),
            show="headings",
            selectmode="browse",
        )
        self._dico_search_tree.heading("Col", text="Col")
        self._dico_search_tree.heading("Ligne", text="Ligne")
        self._dico_search_tree.column("Col", width=70, anchor="center")
        self._dico_search_tree.column("Ligne", width=70, anchor="center")
        self._dico_search_tree.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(tree_frame, orient="vertical", command=self._dico_search_tree.yview)
        sb.pack(side="right", fill="y")
        self._dico_search_tree.configure(yscrollcommand=sb.set)

        self._dico_search_items = []
        self._dico_search_debounce_after_id = None

        self._dico_search_entry.bind("<Return>", self._dico_search_refresh)
        self._dico_search_entry.bind("<KeyRelease>", self._dico_search_schedule_debounce)
        self._dico_search_entry.bind("<Tab>", self._dico_search_autocomplete_tab)
        self._dico_search_tree.bind("<<TreeviewSelect>>", self._dico_search_goto_selected)
        self._dico_search_tree.bind("<Double-Button-1>", self._dico_search_goto_selected)

        # cellule visée par le clic-droit (row,col) en coordonnées TkSheet
        self._dico_ctx_cell = None

        # ===== Grille (tksheet) =====
        # Construire la matrice
        # Deux modes d’affichage :
        # - Mode ABS (origine par défaut (0, nbm)) : colonnes sans 0 (1=1er mot), lignes 1..10
        # - Mode DELTA (origine cliquée) : colonnes/énigmes en delta avec 0 (lignes en modulo 10)
        default_origin = (0, self._dico_nb_mots_max)
        r0, c0 = (self._dico_origin_cell or default_origin)
        refMode = self._dico_ref_mode
        isDelta = (tuple((r0, c0)) != tuple(default_origin)) and (refMode in ("origin", "target"))

        # ---- Colonnes Extended (sans 0) : [-N..-1] U [1..N] ----
        colsExt = self.dico.getExtendedColumns()

        # Entêtes d'affichage
        if isDelta:
            # DELTA : 0 sur la colonne d’origine
            # ---- Colonnes Extended (sans 0) : [-N-j0..N-j0] ----
            j0 = int(c0) - self._dico_nb_mots_max
            colRefExt = int(j0) if int(j0) < 0 else (int(j0) + 1)
            headers = self.dico.getRelativeColumns(colRefExt, refMode=refMode)
        else:
            # ABS : 1 = 1er mot (j=0), pas de colonne 0
            # ---- Colonnes Extended (sans 0) : [-N..-1] U [1..N] ----
            headers = colsExt

        # On remplit la TkSheet de [-N..-1] U [1..N] indépendamment de isDelta
        data = []
        for rowExt in range(1, self.dico.getNbEnigmes() + 1):
            row = [self.dico.getMotExtended(rowExt, colExt) for colExt in colsExt]
            data.append(row)

        # utilisé par le decryptor / compas (NE PAS TOUCHER)
        row_titles_raw = self.dico.getTitres()   # ["530", "780", ...]
        self._dico_row_index = list(row_titles_raw)

        # affichage UI : indexes
        # - DELTA : lignes en modulo 10 (pas d'énigmes négatives), origine = 0
        # - ABS   : 1..10
        if isDelta:
            labels = self.dico.getRowLabelsRel(r0, refMode=refMode)
        else:
            labels = self.dico.getRowLabelsAbs()
        rowIndexDisplay = [f'{lab} - "{t}"' for lab, t in zip(labels, row_titles_raw)]

        self._dico_all_words = set()
        for rowExt, colExt in self.dico.iterCoords(DicoScope.EXTENDED):
            self._dico_all_words.add(self.dico.getMotExtended(rowExt, colExt))

        # Créer la grille
        self.dicoSheet = Sheet(
            right,
            data=data,
            headers=headers,
            row_index=rowIndexDisplay,
            show_row_index=True,
            height=max(120, getattr(self, "dico_panel_height", 220) - 10),
            empty_vertical=0,
        )
        # NOTE: on désactive le menu contextuel interne TkSheet ("copy", etc.)
        # car il entre en conflit avec notre menu "Définir comme origine (0,0)".
        self.dicoSheet.enable_bindings((
            "single_select", "row_select", "column_select",
            "arrowkeys", "copy", "rc_select", "double_click"
        ))
        self.dicoSheet.align_columns(columns="all", align="center")
        self.dicoSheet.set_options(cell_align="center")
        self.dicoSheet.pack(expand=True, fill="both")

        # --- MAJ horloge sur sélection de cellule (événement unique & propre) ---
        def _on_dico_cell_select(event=None):
            # La sélection directe dans la grille doit toujours synchroniser le compas.
            # Source unique de vérité : cellule actuellement sélectionnée
            sel = self.dicoSheet.get_selected_cells()
            if sel:
                r, c = next(iter(sel))
                self._update_clock_from_cell(int(r), int(c))

        # Un seul binding tksheet, propre : "cell_select"
        def _on_dico_double_click(event=None):
            # On ne traite QUE si double-clic sur une cellule.
            # (évite les pièges : header / row_index / vide)
            cur = self.dicoSheet.get_currently_selected()
            rr, cc = int(cur.row), int(cur.column)

            # Toggle : même cellule => reset défaut (0 logique = 1er mot du livre)
            self._dico_origin_cell = (rr, cc)
            self._dico_ref_mode = "origin"
            # Rebuild complet (robuste vis-à-vis des API TkSheet)
            self._build_dico_grid()

        self.dicoSheet.extra_bindings([
            ("cell_select", _on_dico_cell_select),
        ])

        # IMPORTANT (TkSheet) :
        # - Le widget qui reçoit réellement les clics est souvent MainTable (sheet.MT).
        # - On bind donc en priorité sur MT, sinon fallback sur Sheet.
        def _bind_to_sheet_widget(widget, sequence, callback):
            if widget is None:
                return
            try:
                widget.bind(sequence, callback, add="+")
            except TypeError:
                widget.bind(sequence, callback)

        target_widget = getattr(self.dicoSheet, "MT", None) or self.dicoSheet
        _bind_to_sheet_widget(target_widget, "<Double-Button-1>", _on_dico_double_click)

        # ===== Menu contextuel clic-droit : "Set as (0,0)" =====
        # (On n’utilise pas le menu popup interne de TkSheet, on ajoute le nôtre en Tk.Menu)
        if not hasattr(self, "_ctx_menu_dico"):
            self._ctx_menu_dico = tk.Menu(self, tearoff=0)
            self._ctx_menu_dico.add_command(
                label="Définir comme origine (0,0)",
                command=lambda: self._dico_set_origin_from_context_cell()
            )
            self._ctx_menu_dico.add_command(
                label="Définir comme cible (0,0)",
                command=lambda: self._dico_set_target_from_context_cell()
            )
            self._ctx_menu_dico.add_command(
                label="Réinitialiser origine (0,0)",
                command=lambda: self._dico_reset_origin()
            )
            self._ctx_menu_dico.add_separator()
            self._ctx_menu_dico.add_command(
                label="Occurrence précédente",
                command=lambda: self._dico_move_context_occurrence(-1),
                state=tk.DISABLED,
            )
            self._ctx_dico_idx_prev = self._ctx_menu_dico.index("end")
            self._ctx_menu_dico.add_command(
                label="Occurrence suivante",
                command=lambda: self._dico_move_context_occurrence(+1),
                state=tk.DISABLED,
            )
            self._ctx_dico_idx_next = self._ctx_menu_dico.index("end")

        def _dico_cell_from_event(event):
            """Retourne (row, col) pour la cellule sous la souris (clic droit).
            IMPORTANT (TkSheet) :
            - identify_row/identify_column attendent un *event* Tk, pas un int.
            - get_row_at_y/get_column_at_x attendent des coordonnées (x/y).
            """
            mt = getattr(self.dicoSheet, "MT", None)
            x, y = int(getattr(event, "x", 0)), int(getattr(event, "y", 0))

            def _call_rowcol(obj, meth_row: str, meth_col: str):
                if not (obj is not None and hasattr(obj, meth_row) and hasattr(obj, meth_col)):
                    return None

                if meth_row.startswith("identify_"):
                    r = getattr(obj, meth_row)(event)
                    c = getattr(obj, meth_col)(event)
                else:
                    r = getattr(obj, meth_row)(y)
                    c = getattr(obj, meth_col)(x)
                if r is None or c is None:
                    return None
                return int(r), int(c)

            # 1) Priorité à MainTable (MT) si présent
            for meth_row, meth_col in (
                ("identify_row", "identify_column"),
                ("get_row_at_y", "get_column_at_x"),
            ):
                rc = _call_rowcol(mt, meth_row, meth_col)
                if rc is not None:
                    return rc
                rc = _call_rowcol(self.dicoSheet, meth_row, meth_col)
                if rc is not None:
                    return rc
            return None

        def _on_dico_right_click(event=None):
            # Sélectionner d’abord la cellule sous la souris puis ouvrir le menu
            rc = _dico_cell_from_event(event) if event is not None else None
            if rc:
                rr, cc = rc
                self._dico_ctx_cell = (int(rr), int(cc))
                self.dicoSheet.select_cell(rr, cc, redraw=False)

            # Activer/désactiver "précédent/suivant" selon la cellule visée
            self._dico_update_occurrence_ctx_menu_state()

            self._ctx_menu_dico.tk_popup(event.x_root, event.y_root)
            self._ctx_menu_dico.grab_release()

            # IMPORTANT: empêcher TkSheet de traiter aussi le clic-droit (sinon popup interne)
            return "break"

        _bind_to_sheet_widget(target_widget, "<Button-3>", _on_dico_right_click)

        # --- Centrer l’affichage par défaut sur la colonne 0 ---
        # Centre sur l'origine LOGIQUE (0,0) => cellule physique (r0,c0)
        self.dicoSheet.select_cell(int(r0), int(c0), redraw=False)
        self.dicoSheet.see(row=int(r0), column=int(c0))  # amène la colonne 0 logique dans la vue (le plus centré possible)

        # La sélection du dico doit rester possible pour synchroniser le compas,
        # même si aucun arc n'est disponible (le filtrage, lui, restera grisé).
        self._dico_set_selection_enabled(True)

        # Appliquer style origine + filtre (si actif)
        self._dico_apply_origin_style()
        if getattr(self, "_dico_filter_active", False):
            self._dico_apply_filter_styles()

        self._dico_search_refresh()
        self.status.config(text="Dico affiché dans le panneau bas")

    def _dico_search_schedule_debounce(self, event=None) -> None:
        if self._dico_search_debounce_after_id is not None:
            self.after_cancel(self._dico_search_debounce_after_id)
        self._dico_search_debounce_after_id = self.after(250, self._dico_search_refresh)

    def _dico_search_refresh(self, event=None) -> None:
        self._dico_search_debounce_after_id = None

        raw = self._dico_search_var.get()
        mot = _normalizeWordLocal(raw)

        for iid in self._dico_search_tree.get_children():
            self._dico_search_tree.delete(iid)
        self._dico_search_items = []

        if mot == "":
            self._dico_search_status_icon.configure(image=self._dico_icon_check_red)
            self._dico_search_status_icon.image = self._dico_icon_check_red
            return

        nbm = int(self._dico_nb_mots_max)
        default_origin = (0, nbm)
        r0, c0 = self._dico_origin_cell or default_origin
        refMode = self._dico_ref_mode
        isDelta = (tuple((int(r0), int(c0))) != tuple(default_origin)) and (refMode in ("origin", "target"))

        n = int(self.dico.getNbLignes())
        rowOAbs = (int(r0) % n) + 1
        j0 = int(c0) - nbm
        colOAbs = int(j0) if int(j0) < 0 else (int(j0) + 1)
        originAbs = (rowOAbs, colOAbs)

        colsExt = self.dico.getExtendedColumns()
        occurrences = []

        for rowExt, colExt in self.dico.iterCoords(DicoScope.EXTENDED):
            w = self.dico.getMotExtended(rowExt, colExt)
            if w != mot:
                continue

            if isDelta:
                dr, dc = self.dico.computeDeltaAbs(originAbs, (rowExt, colExt))
                if refMode == "target":
                    dr = (-int(dr)) % n
                    dc = -int(dc)
                rowDisp = int(dr)
                colDisp = int(dc)
            else:
                rowDisp = int(rowExt)
                colDisp = int(colExt)

            rowTk = int(rowExt) - 1
            colTk = colsExt.index(int(colExt))
            occurrences.append((colDisp, rowDisp, rowTk, colTk))

        occurrences.sort(key=lambda it: (it[0], it[1]))

        for i, (colDisp, rowDisp, rowTk, colTk) in enumerate(occurrences):
            self._dico_search_tree.insert("", "end", iid=str(i), values=(int(colDisp), int(rowDisp)))
            self._dico_search_items.append((rowTk, colTk))

        if len(occurrences) > 0:
            self._dico_search_status_icon.configure(image=self._dico_icon_check_green)
            self._dico_search_status_icon.image = self._dico_icon_check_green
        else:
            self._dico_search_status_icon.configure(image=self._dico_icon_check_red)
            self._dico_search_status_icon.image = self._dico_icon_check_red

    def _dico_search_autocomplete_tab(self, event=None):
        prefix = _normalizeWordLocal(self._dico_search_var.get())
        candidates = sorted([w for w in self._dico_all_words if w.startswith(prefix)])
        if len(candidates) == 1:
            self._dico_search_var.set(candidates[0])
            self._dico_search_entry.icursor(tk.END)
            self._dico_search_entry.selection_clear()
            self._dico_search_refresh()
            return "break"
        return None

    def _dico_search_goto_selected(self, event=None) -> None:
        sel = self._dico_search_tree.selection()
        if not sel:
            return
        i = int(sel[0])
        rowTk, colTk = self._dico_search_items[i]
        self.dicoSheet.select_cell(rowTk, colTk, redraw=False)
        self.dicoSheet.see(row=rowTk, column=colTk)
        self._update_clock_from_cell(rowTk, colTk)

    # ---------- DICO : origine logique via menu contextuel ----------

    def _dico_set_origin_from_context_cell(self):
        # Action "Set as (0,0)" : utiliser la cellule du clic-droit (robuste, sans dépendre de la sélection)
        rr, cc = self._dico_ctx_cell
        self._dico_origin_cell = (rr, cc)
        self._dico_ref_mode = "origin"  # exclusif
        self._build_dico_grid()

    def _dico_set_target_from_context_cell(self):
        # Action "Set as target (0,0)" : cellule du clic-droit
        rr, cc = self._dico_ctx_cell
        self._dico_origin_cell = (rr, cc)
        self._dico_ref_mode = "target"  # exclusif
        self._build_dico_grid()

    # ---------- DICO : navigation par occurrences (même mot) ----------

    def _dico_find_occurrence_in_row(self, row: int, col: int, direction: int):
        """Retourne la colonne de la prochaine occurrence du même mot sur la même ligne.
        direction = +1 (droite) ou -1 (gauche). Recherche dans la grille affichée uniquement.
        """
        n_cols = 2 * self._dico_nb_mots_max

        if row < 0 or row >= self.dico.getNbEnigmes():
            return None
        if col < 0 or col >= n_cols:
            return None

        w0 = str(self.dicoSheet.get_cell_data(row, col)).strip()
        if not w0:
            return None

        c = col + direction
        while 0 <= c < n_cols:
            w = str(self.dicoSheet.get_cell_data(row, c)).strip()
            if w and w == w0:
                return c
            c += direction

        return None

    def _dico_update_occurrence_ctx_menu_state(self):
        """Active/désactive 'Occurrence précédente/suivante' selon le mot cliqué."""
        if not hasattr(self, "_ctx_menu_dico"):
            return
        if not hasattr(self, "_ctx_dico_idx_prev") or not hasattr(self, "_ctx_dico_idx_next"):
            return

        prev_state = tk.DISABLED
        next_state = tk.DISABLED

        rr, cc = self._dico_ctx_cell
        if self._dico_find_occurrence_in_row(rr, cc, -1) is not None:
            prev_state = tk.NORMAL
        if self._dico_find_occurrence_in_row(rr, cc, +1) is not None:
            next_state = tk.NORMAL

        self._ctx_menu_dico.entryconfig(self._ctx_dico_idx_prev, state=prev_state)
        self._ctx_menu_dico.entryconfig(self._ctx_dico_idx_next, state=next_state)

    def _dico_move_context_occurrence(self, direction: int):
        """Déplace la sélection vers l'occurrence précédente/suivante du même mot (même ligne)."""
        rr, cc = self._dico_ctx_cell
        new_c = self._dico_find_occurrence_in_row(rr, cc, direction)
        if new_c is None:
            self._dico_update_occurrence_ctx_menu_state()
            return

        # Mettre à jour le "contexte" pour permettre des clics successifs
        self._dico_ctx_cell = (rr, new_c)

        self.dicoSheet.select_cell(rr, new_c, redraw=False)
        self.dicoSheet.see(row=rr, column=new_c)
        self._update_clock_from_cell(rr, new_c)

        self._dico_update_occurrence_ctx_menu_state()

    def _dico_reset_origin(self):
        # Origine par défaut = 1er mot du livre :
        #  - ligne physique 0 (énigme 0)
        #  - colonne physique nbm (car headers = [-nbm..0..+nbm])
        self._dico_origin_cell = (0, self._dico_nb_mots_max)
        self._dico_ref_mode = None   # retour ABS (donc style bleu)
        self._build_dico_grid()

    # ---------- DICO → Horloge ----------
    def _tkToExtAbs(self, row: int, col: int, *, nbm: int) -> tuple[int, int]:
        """
        TkSheet (row,col) -> (rowAbs 1..10, colExt sans 0) en mode ABS.
        """
        rowAbs = (row % 10) + 1
        j = col - nbm          # [-nbm..nbm-1]
        colExt = j if j < 0 else j + 1  # pas de 0
        return rowAbs, colExt

    def _tkToRel(self, row: int, col: int, *, r0: int, c0: int, refMode: str) -> tuple[int, int]:
        """
        TkSheet (row,col) -> (dRow 0..9, dCol signé) en mode DELTA.
        """
        dr = (row - r0) % 10
        dc = col - c0
        if refMode == "target":
            dr = (-dr) % 10
            dc = -dc
        return dr, dc

    def _update_clock_from_cell(self, row: int, col: int):
        """Met à jour l'horloge à partir d'une cellule de la grille Dico.
        row et col sont des colonne Tk. Ex en Mode Abs Col Tk 100 = 1
                La conversion (row,col)->(hour,minute,label) est déléguée au decryptor actif.
        """
        if not getattr(self, "dicoSheet", None):
            return

        word = str(self.dicoSheet.get_cell_data(int(row), int(col))).strip()

        # Externalisation : conversion via decryptor
        # --- Passage en référentiel LOGIQUE pour le compas/decryptor ---
        default_origin = (0, self._dico_nb_mots_max)
        r0, c0 = (self._dico_origin_cell or default_origin)
        refMode = self._dico_ref_mode  # "origin" / "target"
        isDelta = (tuple((int(r0), int(c0))) != tuple(default_origin)) and (refMode in ("origin", "target"))

        if isDelta:
            # DELTA : 0 autorisé. Lignes en modulo 10 (pas d’énigmes négatives)
            rowVal, colVal = self._tkToRel(row, col, r0=r0, c0=c0, refMode=refMode)
            mode = "delta"
        else:
            # ABS : pas de 0. Lignes 1..10, Colonnes … -2, -1, 1, 2, …
            rowVal, colVal = self._tkToExtAbs(row, col, nbm=self._dico_nb_mots_max)
            mode = "abs"

        st = self.decryptor.clockStateFromDicoCell(
            row=rowVal,
            col=colVal,
            word=word,
            mode=mode,
        )

        self._clock_state.update({"hour": float(st.hour), "minute": st.minute, "label": st.label})
        self._redraw_overlay_only()

    # ---------- DICO : filtrage visuel par angle ----------

    def _dico_apply_filter_styles(self):
        """Applique le style (gras / gris) à l'ensemble du dictionnaire."""
        if not getattr(self, "dicoSheet", None):
            return
        if not self._dico_filter_active:
            return
        if self._dico_filter_ref_angle_deg is None:
            return

        if not hasattr(self.dicoSheet, "highlight_cells"):
            raise AttributeError("tksheet.Sheet.highlight_cells non disponible")

        default_origin = (0, self._dico_nb_mots_max)
        r0, c0 = (self._dico_origin_cell or default_origin)
        refMode = self._dico_ref_mode
        isDelta = (tuple((int(r0), int(c0))) != tuple(default_origin)) and (refMode in ("origin", "target"))
        for r in range(self.dico.getNbEnigmes()):
            for c in range(2 * self._dico_nb_mots_max):
                word = str(self.dicoSheet.get_cell_data(r, c)).strip()
                if not word:
                    continue
                # On détecte dans quelle partie du dictionnaire on se trouve pour la couleur de fond
                rowAbs, colAbs = self._tkToExtAbs(r, c, nbm=self._dico_nb_mots_max)
                scopeMirroring = self.dico.isInScope(DicoScope.MIRRORING, rowAbs, colAbs)

                # --- Passage en référentiel LOGIQUE pour l'angle ---
                if isDelta:
                    rowVal, colVal = self._tkToRel(r, c, r0=r0, c0=c0, refMode=refMode)
                    mode = "delta"
                else:
                    rowVal, colVal = self._tkToExtAbs(r, c, nbm=self._dico_nb_mots_max)
                    mode = "abs"

                # IMPORTANT:
                #   Le filtre doit être cohérent avec ce que le compas affichera quand on clique une cellule.
                #   Or le compas affiche Δ = angle entre aiguilles (heure/minute) calculé à partir de
                #   clockStateFromDicoCell().
                #   On n'utilise donc PAS deltaAngleFromDicoCell (qui peut avoir une convention différente)
                #   mais exactement la même définition que l'overlay du compas.
                st = self.decryptor.clockStateFromDicoCell(
                    row=rowVal,
                    col=colVal,
                    word=word,
                    mode=mode,
                )
                ref = float(self._dico_filter_ref_angle_deg) % 180.0
                ok = abs(st.deltaDeg180 - ref) <= self._dico_filter_tolerance_deg

                if ok and not scopeMirroring:
                    # Match: texte noir + fond légèrement marqué
                    self.dicoSheet.highlight_cells(r, c, fg="#0B2A5B", bg="#CCD8EA")
                elif not ok and not scopeMirroring:
                    # Match: texte noir + fond légèrement marqué
                    self.dicoSheet.highlight_cells(r, c, fg="#8FAAD6", bg="#EDF0F4")
                elif ok and scopeMirroring:
                    # Match: texte noir + fond légèrement marqué
                    self.dicoSheet.highlight_cells(r, c, fg="#000000", bg="#E8E8E8")
                else:
                    # Non match: gris clair (pas de fond)
                    self.dicoSheet.highlight_cells(r, c, fg="#B0B0B0")

        # Priorité visuelle à l'origine
        self._dico_apply_origin_style()

        if hasattr(self.dicoSheet, "refresh"):
            self.dicoSheet.refresh()

    def _dico_clear_filter_styles(self):
        """Réinitialise les styles appliqués par _dico_apply_filter_styles."""
        if not getattr(self, "dicoSheet", None):
            return
        # Version récente : méthode dédiée
        if hasattr(self.dicoSheet, "dehighlight_all"):
            self.dicoSheet.dehighlight_all()
        else:
            # Fallback minimal si jamais (mais si tu mets à jour, tu ne passes jamais ici)
            if hasattr(self.dicoSheet, "dehighlight_cells"):
                self.dicoSheet.dehighlight_cells()
        # Ré-appliquer l'origine après nettoyage
        self._dico_apply_origin_style()
        if hasattr(self.dicoSheet, "refresh"):
            self.dicoSheet.refresh()

    def _dico_apply_origin_style(self):
        """Applique le style visuel de la cellule origine (0,0) logique."""
        if not getattr(self, "dicoSheet", None):
            return
        if not hasattr(self.dicoSheet, "highlight_cells"):
            return
        r0, c0 = (self._dico_origin_cell or (0, 0))
        refMode = getattr(self, "_dico_ref_mode", None)
        bg = "#BFDFFF" if refMode != "target" else "#FFCCCC"
        self.dicoSheet.highlight_cells(int(r0), int(c0), fg="#9A9A9A", bg=bg)

    # ---------- DICO : lecture de la sélection courante ----------

    def _get_selected_dico_word(self) -> Optional[Tuple[str, int, int]]:
        """
        Retourne (word, row, col) depuis la sélection de tksheet, ou None si rien.
        """
        if not getattr(self, "dicoSheet", None):
            return None
        sel = self.dicoSheet.get_selected_cells()
        r = c = None
        if sel:
            r, c = next(iter(sel))
        word = str(self.dicoSheet.get_cell_data(r, c)).strip()

        if not word:
            return None
        return (word, r, c)

    # ---------- Contexte : actions mot <-> triangle ----------

    def _ctx_add_or_replace_word(self):
        """Ajoute/remplace le mot sélectionné du dico sur le triangle ciblé."""
        if self._ctx_target_idx is None or not (0 <= self._ctx_target_idx < len(self._last_drawn)):
            return
        tri = self._last_drawn[self._ctx_target_idx]
        tri_id = int(tri.get("id"))
        sel = self._get_selected_dico_word()
        if not sel:
            messagebox.showinfo("Association mot", "Aucun mot sélectionné dans le dictionnaire.")
            return
        word, row, col = sel
        self._tri_words[tri_id] = {"word": word, "row": row, "col": col}
        self._redraw_from(self._last_drawn)

    def _ctx_clear_word(self):
        """Efface l'association de mot du triangle ciblé, si présente."""
        if self._ctx_target_idx is None or not (0 <= self._ctx_target_idx < len(self._last_drawn)):
            return
        tri = self._last_drawn[self._ctx_target_idx]
        tri_id = int(tri.get("id"))
        if tri_id in self._tri_words:
            del self._tri_words[tri_id]
            self._redraw_from(self._last_drawn)

    def _rebuild_ctx_word_entries(self):
        """Reconstruit la partie 'mot' du menu contextuel en fonction du triangle visé + sélection dico."""
        # Nettoyer les deux entrées dynamiques existantes
        # On supprime depuis la fin pour conserver l'index d'ancrage
        end = self._ctx_menu.index("end")
        while end > self._ctx_idx_words_start:
            self._ctx_menu.delete(end)
            end = self._ctx_menu.index("end")

        # Recréer deux entrées selon contexte
        label_add = "Ajouter…"
        cmd_add = None
        sel = self._get_selected_dico_word()
        if sel:
            label_add = f"Ajouter « {sel[0]} »"
            cmd_add = self._ctx_add_or_replace_word
        # Triangle ciblé ?
        has_target = (self._ctx_target_idx is not None) and (0 <= self._ctx_target_idx < len(self._last_drawn))
        exists = False
        label_del = "Effacer…"
        if has_target:
            tri = self._last_drawn[self._ctx_target_idx]
            tri_id = int(tri.get("id"))
            if tri_id in self._tri_words:
                exists = True
                cur_word = self._tri_words[tri_id]["word"]
                # si on a aussi une sélection, on préfère le verbe "Remplacer"
                if sel:
                    label_add = f"Remplacer par « {sel[0]} »"
                label_del = f"Effacer « {cur_word} »"
        # Ajouter les entrées (activées/désactivées selon contexte)
        self._ctx_menu.add_command(label=label_add, command=cmd_add, state=("normal" if cmd_add and has_target else "disabled"))
        self._ctx_menu.add_command(label=label_del, command=(self._ctx_clear_word if exists else None),
                                   state=("normal" if exists else "disabled"))

    # ---------- DEBUG: toggle du filtre d'intersection au highlight ----------

    def _simulation_cancel_dictionary_filter(self):
        """Annule le filtrage visuel du dictionnaire (styles)."""
        was_active = bool(self._dico_filter_active) or (self._dico_filter_ref_angle_deg is not None)
        self._dico_filter_active = False
        self._dico_filter_ref_angle_deg = None
        self._dico_clear_filter_styles()
        self._update_compass_ctx_menu_and_dico_state()
        if was_active:
            self.status.config(text="Dico: filtrage annule")

    # =========================
    #  SIMULATION (AUTO ASSEMBLAGE)
    # =========================

    def _dico_clear_selection(self, refresh: bool = True):
        """Supprime toute selection visible dans la TkSheet du dictionnaire."""
        if not getattr(self, 'dicoSheet', None):
            return

        if hasattr(self.dicoSheet, 'deselect'):
            self.dicoSheet.deselect('all')
        else:
            if refresh and hasattr(self.dicoSheet, 'refresh'):
                self.dicoSheet.refresh()
            return

        for meth in ('deselect_all', 'delete_selection', 'dehighlight_all'):
            if hasattr(self.dicoSheet, meth):
                getattr(self.dicoSheet, meth)()
                break

        if refresh and hasattr(self.dicoSheet, 'refresh'):
            self.dicoSheet.refresh()

    def _dico_set_selection_enabled(self, enabled: bool):
        """(De)sactive la selection utilisateur sur la TkSheet du dictionnaire.

        Objectif : quand le menu contextuel 'Filtrer le dictionnaire' est grise
        (pas d'arc mesure/affiche), on desactive aussi la selection dans la grille.

        Implementation :
        - si tksheet expose disable_bindings/enable_bindings, on s'appuie dessus;
        - sinon, on garde un garde-fou cote callback 'cell_select'."""
        if not getattr(self, 'dicoSheet', None):
            # On memorise quand meme l'etat pour le callback, meme si le widget n'existe pas.
            self._dico_selection_enabled = bool(enabled)
            return

        self._dico_selection_enabled = bool(enabled)

        # 1) Desactiver/activer les bindings principaux de selection
        # NB: on ne touche pas a 'copy' et au popup menu pour rester neutre.
        bindings = ('single_select', 'row_select', 'column_select', 'rc_select', 'arrowkeys')
        if hasattr(self.dicoSheet, 'disable_bindings') and hasattr(self.dicoSheet, 'enable_bindings'):
            if self._dico_selection_enabled:
                self.dicoSheet.enable_bindings(bindings)
            else:
                self.dicoSheet.disable_bindings(bindings)

        # 2) Si on desactive, on efface aussi la selection courante (visuellement)
        if not self._dico_selection_enabled:
            self._dico_clear_selection(refresh=False)

        if hasattr(self.dicoSheet, 'refresh'):
            self.dicoSheet.refresh()
