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

from src.DictionnaireEnigmes import DictionnaireEnigmes


class TriangleViewerDictionaryMixin:
    """Mixin: méthodes extraites de assembleur_tk.py."""
    pass

    def _init_dictionary(self):
        """Construit le dictionnaire en lisant ../data/livre.txt (si présent)."""
        try:
            if DictionnaireEnigmes is None:
                raise ImportError("Module DictionnaireEnigmes introuvable")
            livre_path = os.path.join(self.data_dir, "livre.txt")
            if not os.path.isfile(livre_path):
                self.status.config(text="Dico: fichier 'livre.txt' non trouvé dans ../data")
                return
            self.dico = DictionnaireEnigmes(livre_path)  # charge tout le livre
            nb_lignes = len(self.dico)
            self.status.config(text=f"Dico chargé: {nb_lignes} lignes depuis {livre_path}")
        except Exception as e:
            self.status.config(text=f"Dico: échec de chargement — {e}")

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

        # --- Layout du panneau bas : [sidebar catégories] | [grille] ---
        container = tk.Frame(self.dicoPanel, bg="#f3f3f3")
        container.pack(fill="both", expand=True)
        # colonne gauche (catégories + liste)
        left = tk.Frame(container, width=180, bg="#f3f3f3", bd=1, relief=tk.GROOVE)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        # colonne droite (grille)
        right = tk.Frame(container, bg="#f3f3f3")
        right.pack(side="left", fill="both", expand=True)

        # ===== Barre "catégories" =====
        tk.Label(left, text="Catégorie :", anchor="w", bg="#f3f3f3").pack(anchor="w", padx=8, pady=(8, 2))
        cats = list(self.dico.getCategories())

        # Préserver la catégorie sélectionnée lors d'un rebuild
        cat_default = getattr(self, "_dico_cat_selected", None)
        if cat_default not in (cats or []):
            cat_default = (cats[0] if cats else "")
        self._dico_cat_var = tk.StringVar(value=cat_default)
        self._dico_cat_combo = ttk.Combobox(left, state="readonly", values=cats, textvariable=self._dico_cat_var)
        self._dico_cat_combo.pack(fill="x", padx=8, pady=(0, 6))

        # Liste des mots de la catégorie sélectionnée
        lb_frame = tk.Frame(left, bg="#f3f3f3")
        lb_frame.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        self._dico_cat_list = tk.Listbox(lb_frame, exportselection=False)
        self._dico_cat_list.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(lb_frame, orient="vertical", command=self._dico_cat_list.yview)
        sb.pack(side="right", fill="y")
        self._dico_cat_list.configure(yscrollcommand=sb.set)

        # Remplissage initial + binding de la combo
        self._dico_cat_items = []

        # cellule visée par le clic-droit (row,col) en coordonnées TkSheet
        self._dico_ctx_cell = None

        def _refresh_cat_list(*_):
            cat = self._dico_cat_var.get()
            self._dico_cat_selected = cat
            self._dico_cat_list.delete(0, tk.END)
            if not cat:
                return

            # getListeCategorie -> [(mot, enigme, indexMot), ...]
            items = self.dico.getListeCategorie(cat)

            # mémoriser pour la synchro avec la grille
            self._dico_cat_items = list(items)
            # Afficher: "mot — (eDisp, mDisp)" (index d’affichage)
            # - Mode ABS (origine par défaut) : colonnes sans 0 (1=1er mot), lignes 1..10
            # - Mode DELTA (origine cliquée) : colonnes/énigmes en delta avec 0 (pas d’énigmes négatives -> modulo 10)
            default_origin = (0, self._dico_nb_mots_max)
            r0, c0 = (self._dico_origin_cell or default_origin)
            refMode = getattr(self, "_dico_ref_mode", None)
            isDelta = (tuple((r0, c0)) != tuple(default_origin)) and (refMode in ("origin", "target"))
            isTarget = (refMode == "target")

            origin_indexMot = int(c0) - self._dico_nb_mots_max
            for mot, e, m in items:
                # Lignes : pas d’énigmes négatives
                eLogRaw = int(e) - int(r0)
                eLog = (-int(eLogRaw) % 10) if isTarget else (int(eLogRaw) % 10)
                eDisp = int(eLog) if isDelta else (int(eLog) + 1)

                # Colonnes : indexMot (absolu) ou delta
                mLog = int(m) - int(origin_indexMot)
                if isDelta:
                    mDisp = int(mLog)  # delta, 0 autorisé
                else:
                    # ABS : pas de colonne 0 ; 1 = 1er mot
                    mDisp = int(mLog) if int(mLog) < 0 else (int(mLog) + 1)

                self._dico_cat_list.insert(tk.END, f"{mot} — ({eDisp}, {mDisp})")

        self._dico_cat_combo.bind("<<ComboboxSelected>>", _refresh_cat_list)
        _refresh_cat_list()

        # Synchronisation: clic/double-clic dans la liste -> centrer/sélectionner le mot dans la grille
        def _goto_selected_word(event=None):
            if not getattr(self, "dicoSheet", None):
                return
            sel = self._dico_cat_list.curselection()
            if not sel:
                return
            i = int(sel[0])
            mot, enigme, indexMot = self._dico_cat_items[i]

            # convertir indexMot [-N..N) -> colonne [0..2N)
            col = int(indexMot) + self._dico_nb_mots_max
            row = int(enigme)
            # sélectionner et faire voir la cellule ; see() centre autant que possible
            self.dicoSheet.select_cell(row, col, redraw=False)
            self.dicoSheet.see(row=row, column=col)
            # MAJ horloge (sélection indirecte via la liste)
            self._update_clock_from_cell(row, col)

        # Clic et double-clic compatibles
        self._dico_cat_list.bind("<<ListboxSelect>>", _goto_selected_word)
        self._dico_cat_list.bind("<Double-Button-1>", _goto_selected_word)

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

        self.status.config(text="Dico affiché dans le panneau bas")

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
                if ok:
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
