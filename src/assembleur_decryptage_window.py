"""Fenêtre autonome du déchiffreur de chemins."""

from __future__ import annotations

import threading
from typing import Callable, Mapping
import tkinter as tk
from tkinter import ttk, messagebox

from src.assembleur_decryptor import ClockDicoDecryptor, DecryptorConfig, DECRYPTORS
from src.assembleur_decryptor_engine import DecryptorEngine
from src.DictionnaireEnigmes import DicoScope, ListePatterns, Pattern
from src.assembleur_engine_runtime import EngineControl, EventQueue, RunControlConfig


def createDecryptor(decryptor_id: str):
    """Instancie le déchiffreur demandé, avec le fallback historique."""
    decryptor_class = DECRYPTORS.get(str(decryptor_id))
    return ClockDicoDecryptor() if decryptor_class is None else decryptor_class()


class DecryptageEngineWindow(tk.Toplevel):
    """Possède l'état UI et d'exécution de la fenêtre de décryptage."""

    def __init__(self, parent, *, get_config: Callable[[str, object], object], set_config: Callable[[str, object], None], scenario_provider: Callable[[], object], dico_provider: Callable[[], object], decryptor_provider: Callable[[], object], icon_loader: Callable[[str], object], icons: Mapping[str, object], on_close: Callable[["DecryptageEngineWindow"], None]):
        super().__init__(parent)
        self._get_config = get_config
        self._set_config = set_config
        self._scenario_provider = scenario_provider
        self._dico_provider = dico_provider
        self._decryptor_provider = decryptor_provider
        self._icon_loader = icon_loader
        self._icons = icons
        self._on_close_callback = on_close

        win = self
        win.title("Décrypteur de chemins")
        win.transient(parent)
        win.minsize(820, 520)

        def _on_close():
            win.destroy()
            self._on_close_callback(self)

        win.protocol("WM_DELETE_WINDOW", _on_close)

        root = ttk.Frame(win, padding=10)
        root.grid(row=0, column=0, sticky="nsew")
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=0)

        # --- Zone supérieure : 3 blocs ---
        top_zone = ttk.Frame(root)
        top_zone.grid(row=0, column=0, sticky="nsew")

        top_zone.grid_columnconfigure(0, weight=0)
        top_zone.grid_columnconfigure(1, weight=0)
        top_zone.grid_columnconfigure(2, weight=1)
        top_zone.grid_rowconfigure(0, weight=0)

        decrypt_frame = ttk.LabelFrame(top_zone, text="Décrypteur")
        decrypt_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 8), pady=(0, 8))

        mid_col = ttk.Frame(top_zone)
        mid_col.grid(row=0, column=1, sticky="nsw", padx=(0, 8), pady=(0, 8))

        patterns_frame = ttk.LabelFrame(top_zone, text="Patterns de mots à trouver")
        patterns_frame.grid(row=0, column=2, sticky="nsew", pady=(0, 8))

        # --- Bloc Décrypteur ---
        decrypt_items = [f"{d.id} — {d.label}" for d in DECRYPTORS.values()]
        id_to_item = {d.id: f"{d.id} — {d.label}" for d in DECRYPTORS.values()}
        item_to_id = {v: k for k, v in id_to_item.items()}

        default_decrypt_id = "clock_dico_v1"
        if default_decrypt_id not in id_to_item:
            if getattr(self._decryptor_provider(), "id", None) in id_to_item:
                default_decrypt_id = str(self._decryptor_provider().id)
            else:
                default_decrypt_id = next(iter(id_to_item.keys()), "")
        default_decrypt = id_to_item.get(default_decrypt_id, decrypt_items[0] if decrypt_items else "")

        win.var_decryptor = tk.StringVar(value=default_decrypt)
        win.var_angle180 = tk.BooleanVar(value=True)
        win.var_az_a = tk.BooleanVar(value=False)
        win.var_az_b = tk.BooleanVar(value=False)
        win.var_tolerance = tk.IntVar(value=4)

        decrypt_combo = ttk.Combobox(
            decrypt_frame,
            values=decrypt_items,
            state="readonly",
            textvariable=win.var_decryptor,
            width=32,
        )
        decrypt_combo.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        ttk.Checkbutton(
            decrypt_frame,
            text="Angle 180°",
            variable=win.var_angle180,
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            decrypt_frame,
            text="Azimut A",
            variable=win.var_az_a,
        ).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(
            decrypt_frame,
            text="Azimut B",
            variable=win.var_az_b,
        ).grid(row=3, column=0, sticky="w")

        tol_row = ttk.Frame(decrypt_frame)
        tol_row.grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Label(tol_row, text="Tolérance :").pack(side=tk.LEFT)
        ttk.Spinbox(
            tol_row,
            from_=0,
            to=6,
            increment=1,
            textvariable=win.var_tolerance,
            width=4,
        ).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Label(tol_row, text="°").pack(side=tk.LEFT)

        # --- Colonne 1 : Mode / Scope / Solution ---
        mid_col.grid_rowconfigure(0, weight=0)
        mid_col.grid_rowconfigure(1, weight=0)
        mid_col.grid_rowconfigure(2, weight=0)
        mid_col.grid_columnconfigure(0, weight=0)

        mode_frame = ttk.LabelFrame(mid_col, text="Mode de recherche")
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        win.var_mode = tk.StringVar(value="Absolu")
        ttk.Radiobutton(
            mode_frame,
            text="Absolu",
            value="Absolu",
            variable=win.var_mode,
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_frame,
            text="Relatif",
            value="Relatif",
            variable=win.var_mode,
        ).pack(anchor="w")

        scope_frame = ttk.LabelFrame(mid_col, text="Scope Dictionnaire")
        scope_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        win.var_scope = tk.StringVar(value="Strict")
        ttk.Combobox(
            scope_frame,
            values=["Strict", "Mirroring", "Extended"],
            state="readonly",
            textvariable=win.var_scope,
            width=16,
        ).pack(fill="x")

        solution_frame = ttk.LabelFrame(mid_col, text="Solution")
        solution_frame.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        win.var_max_solutions = tk.IntVar(value=50)
        sol_row = ttk.Frame(solution_frame)
        sol_row.pack(fill="x")
        ttk.Label(sol_row, text="Stopper si plus de :").pack(side=tk.LEFT)
        ttk.Spinbox(
            sol_row,
            from_=1,
            to=9999,
            increment=1,
            textvariable=win.var_max_solutions,
            width=6,
        ).pack(side=tk.LEFT, padx=(6, 0))

        win._patterns = []

        def _extract_decryptor_id(item_text: str) -> str:
            if item_text in item_to_id:
                return item_to_id[item_text]
            s = str(item_text or "")
            if " — " in s:
                return s.split(" — ", 1)[0].strip()
            return s.strip()

        def loadDecryptGuiConfig() -> None:
            cfg_id = self._get_config("decryptGuiDecryptorId", default_decrypt_id)
            cfg_id = str(cfg_id or default_decrypt_id).strip()
            win.var_decryptor.set(id_to_item.get(cfg_id, default_decrypt))

            cfg_mode = str(self._get_config("decryptGuiMode", "ABS") or "ABS").strip().upper()
            win.var_mode.set("Relatif" if cfg_mode.startswith("REL") else "Absolu")

            scope = str(self._get_config("decryptGuiScopeDico", "Strict") or "Strict").strip()
            if scope not in ("Strict", "Mirroring", "Extended"):
                scope = "Strict"
            win.var_scope.set(scope)

            win.var_angle180.set(bool(self._get_config("decryptGuiUseAngle180", True)))
            win.var_az_a.set(bool(self._get_config("decryptGuiUseAzimutA", False)))
            win.var_az_b.set(bool(self._get_config("decryptGuiUseAzimutB", False)))

            try:
                win.var_tolerance.set(int(self._get_config("decryptGuiToleranceDeg", 4)))
            except (TypeError, ValueError):
                win.var_tolerance.set(4)

            try:
                win.var_max_solutions.set(int(self._get_config("decryptGuiStopIfMoreThan", 50)))
            except (TypeError, ValueError):
                win.var_max_solutions.set(50)

            raw_patterns = self._get_config("decryptPatterns", [])
            patterns = []
            if isinstance(raw_patterns, list):
                for item in raw_patterns:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", "")).strip()
                    if not text:
                        continue
                    active = bool(item.get("active", True))
                    patterns.append({"text": text, "active": active})
            win._patterns = patterns

        def persistDecryptPatterns() -> None:
            serialized = [
                {"text": pat.get("text", ""), "active": bool(pat.get("active", False))}
                for pat in (win._patterns or [])
            ]
            self._set_config("decryptPatterns", serialized)

        def persistDecryptGuiConfig() -> None:
            decrypt_id = _extract_decryptor_id(win.var_decryptor.get())
            mode_ui = str(win.var_mode.get() or "Absolu").strip().lower()
            mode_cfg = "REL" if mode_ui.startswith("rel") else "ABS"
            scope = str(win.var_scope.get() or "Strict").strip()
            if scope not in ("Strict", "Mirroring", "Extended"):
                scope = "Strict"

            self._set_config("decryptGuiDecryptorId", decrypt_id)
            self._set_config("decryptGuiMode", mode_cfg)
            self._set_config("decryptGuiScopeDico", scope)
            self._set_config("decryptGuiUseAngle180", bool(win.var_angle180.get()))
            self._set_config("decryptGuiUseAzimutA", bool(win.var_az_a.get()))
            self._set_config("decryptGuiUseAzimutB", bool(win.var_az_b.get()))
            self._set_config("decryptGuiToleranceDeg", int(win.var_tolerance.get()))
            self._set_config("decryptGuiStopIfMoreThan", int(win.var_max_solutions.get()))

        loadDecryptGuiConfig()

        # --- Bloc Patterns ---
        patterns_toolbar = ttk.Frame(patterns_frame)
        patterns_toolbar.pack(anchor="w", pady=(0, 4))

        def _make_pattern_btn(icon, text, cmd):
            if icon is not None:
                return tk.Button(patterns_toolbar, image=icon, command=cmd, relief=tk.FLAT)
            return tk.Button(patterns_toolbar, text=text, command=cmd, width=2, relief=tk.FLAT)

        patterns_list_frame = tk.Frame(patterns_frame, relief=tk.SUNKEN, borderwidth=1)
        patterns_list_frame.pack(fill=tk.BOTH, expand=True)

        win._pattern_vars = []  # list[tk.BooleanVar]
        win._pattern_row_frames = []  # list[ttk.Frame]
        win._pattern_selected_index = None  # int | None

        # Canvas + frame intérieur = vraie liste de Checkbutton scrollable
        patterns_canvas = tk.Canvas(
            patterns_list_frame,
            bg="white",
            height=105,                 # moitié de 240
            highlightthickness=1,
        )
        patterns_scroll = ttk.Scrollbar(patterns_list_frame, orient="vertical", command=patterns_canvas.yview)
        patterns_canvas.configure(yscrollcommand=patterns_scroll.set)

        patterns_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        patterns_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        patterns_inner = ttk.Frame(patterns_canvas)
        patterns_inner_id = patterns_canvas.create_window((0, 0), window=patterns_inner, anchor="nw")

        def _on_patterns_inner_configure(_evt=None):
            patterns_canvas.configure(scrollregion=patterns_canvas.bbox("all"))

        def _on_patterns_canvas_configure(evt):
            # force largeur du frame intérieur = largeur visible du canvas
            patterns_canvas.itemconfigure(patterns_inner_id, width=evt.width)

        patterns_inner.bind("<Configure>", _on_patterns_inner_configure)
        patterns_canvas.bind("<Configure>", _on_patterns_canvas_configure)

        def _set_selected_pattern_index(idx: int | None) -> None:
            win._pattern_selected_index = idx

            # surlignage simple (tk.Frame): on joue sur bg
            for i, row in enumerate(win._pattern_row_frames):
                is_sel = (idx is not None and i == idx)
                bg = "#eaeaea" if is_sel else "white"   # ajuste si tu veux

                row.configure(bg=bg)
                # applique aussi aux enfants (checkbox + label)
                for child in row.winfo_children():
                    child.configure(bg=bg)

            _update_pattern_buttons()

        def _update_pattern_buttons(_evt=None):
            has_sel = win._pattern_selected_index is not None
            btn_pat_edit.configure(state=(tk.NORMAL if has_sel else tk.DISABLED))
            btn_pat_delete.configure(state=(tk.NORMAL if has_sel else tk.DISABLED))

        def _get_selected_pattern_index() -> int | None:
            return win._pattern_selected_index

        def _on_toggle_pattern(idx: int) -> None:
            if not (0 <= idx < len(win._patterns)):
                return
            val = bool(win._pattern_vars[idx].get())
            win._patterns[idx]["active"] = val
            persistDecryptPatterns()

        def _on_row_click(idx: int) -> None:
            _set_selected_pattern_index(idx)

        def _on_row_double_click(idx: int) -> None:
            _set_selected_pattern_index(idx)
            _open_pattern_editor(idx)

        def _refresh_patterns_list():
            # clear rows
            for row in win._pattern_row_frames:
                row.destroy()
            win._pattern_vars = []
            win._pattern_row_frames = []

            for idx, item in enumerate(win._patterns):
                row = tk.Frame(patterns_inner, bg="white")
                row.pack(fill="x", pady=1)
                win._pattern_row_frames.append(row)

                var = tk.BooleanVar(value=bool(item.get("active")))
                win._pattern_vars.append(var)

                chk = tk.Checkbutton(
                    row,
                    variable=var,
                    command=lambda i=idx: (_set_selected_pattern_index(i), _on_toggle_pattern(i)),
                    bg="white",
                )
                chk.pack(side=tk.LEFT, padx=(2, 6))

                lbl = tk.Label(row, text=str(item.get("text", "") or ""), anchor="w", bg="white")
                lbl.pack(side=tk.LEFT, fill="x", expand=True)

                # clic / double-clic sur la ligne (label ou frame)
                row.bind("<Button-1>", lambda e, i=idx: _on_row_click(i))
                lbl.bind("<Button-1>", lambda e, i=idx: _on_row_click(i))
                row.bind("<Double-1>", lambda e, i=idx: _on_row_double_click(i))
                lbl.bind("<Double-1>", lambda e, i=idx: _on_row_double_click(i))
                chk.bind("<Button-1>", lambda e, i=idx: _on_row_click(i))
                chk.bind("<Double-1>", lambda e, i=idx: _on_row_double_click(i))

            patterns_inner.update_idletasks()
            patterns_canvas.configure(scrollregion=patterns_canvas.bbox("all"))

            # si sélection invalide, reset
            if win._pattern_selected_index is not None:
                if not (0 <= win._pattern_selected_index < len(win._patterns)):
                    win._pattern_selected_index = None
            _set_selected_pattern_index(win._pattern_selected_index)

        def _insert_token_at_cursor(entry: tk.Entry, token: str) -> None:
            pos = entry.index(tk.INSERT)
            text = entry.get()
            before = text[:pos]
            after = text[pos:]
            ins = token
            if before and not before.endswith(" "):
                ins = " " + ins
            if after and not after.startswith(" "):
                ins = ins + " "
            entry.insert(pos, ins)
            entry.focus_set()

        def _open_pattern_editor(edit_index: int | None = None) -> None:
            dico = self._dico_provider()
            if dico is None:
                messagebox.showerror("Pattern", "Dictionnaire indisponible.", parent=win)
                return

            is_edit = edit_index is not None
            dlg = tk.Toplevel(win)
            dlg.title("Éditer un pattern" if is_edit else "Ajouter un pattern")
            dlg.transient(win)
            dlg.resizable(False, False)

            # position près du clic
            x = int(win.winfo_pointerx()) + 10
            y = int(win.winfo_pointery()) + 10

            # taille FIXE (et on la ré-appliquera après layout)
            W, H = 500, 120
            dlg.geometry(f"{W}x{H}+{x}+{y}")
            dlg.minsize(W, H)
            dlg.maxsize(W, H)
            dlg.grab_set()

            # IMPORTANT : permettre au contenu de remplir la toplevel
            dlg.grid_rowconfigure(0, weight=1)
            dlg.grid_columnconfigure(0, weight=1)

            dlg.icon_chevrons_down = self._icon_loader("chevrons-down16.png")
            dlg.icon_check_gray = self._icon_loader("check16_gray.png")
            dlg.icon_check_green = self._icon_loader("check16_green.png")
            dlg.icon_check_red = self._icon_loader("check16_red.png")

            frm = ttk.Frame(dlg, padding=10)
            frm.grid(row=0, column=0, sticky="nsew")
            frm.grid_columnconfigure(1, weight=1)

            categories = list(dico.getCategories() or [])
            values = categories + ["Joker"]
            cat_var = tk.StringVar(value=(values[0] if values else "Joker"))

            ttk.Label(frm, text="Catégorie :").grid(row=0, column=0, sticky="w", padx=(0, 6))
            cat_combo = ttk.Combobox(
                frm,
                values=values,
                state="readonly",
                textvariable=cat_var,
                width=18,
            )
            cat_combo.grid(row=0, column=1, sticky="ew")

            def _on_insert_cat():
                cat = str(cat_var.get() or "").strip()
                token = "[*]" if cat == "Joker" else f"[{cat}]"
                _insert_token_at_cursor(pattern_entry, token)
                _schedule_autocheck()

            btn_insert = tk.Button(
                frm,
                image=dlg.icon_chevrons_down,
                text="v" if dlg.icon_chevrons_down is None else "",
                command=_on_insert_cat,
                relief=tk.FLAT,
                padx=0,
                pady=0,
            )
            btn_insert.grid(row=0, column=2, sticky="w", padx=(4, 0), pady=0)

            ttk.Label(frm, text="Pattern :").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
            pattern_var = tk.StringVar()
            pattern_entry = ttk.Entry(frm, textvariable=pattern_var)
            pattern_entry.grid(row=1, column=1, sticky="ew", pady=(8, 0))

            if is_edit:
                pattern_var.set(str(win._patterns[edit_index].get("text", "") or ""))

            def _set_check_state(state: str) -> None:
                icon = None
                label = ""
                if state == "gray":
                    icon = dlg.icon_check_gray
                    label = "OK"
                elif state == "green":
                    icon = dlg.icon_check_green
                    label = "OK"
                elif state == "red":
                    icon = dlg.icon_check_red
                    label = "ERR"
                if icon is not None:
                    btn_check.configure(image=icon, text="")
                else:
                    btn_check.configure(text=label)

            def _check_pattern(show_error: bool) -> tuple[bool, str]:
                syntax_raw = str(pattern_var.get() or "").strip()
                if not syntax_raw:
                    _set_check_state("gray")
                    if show_error:
                        messagebox.showerror("Pattern", "Pattern vide", parent=dlg)
                    return False, ""
                p = Pattern(dico)
                ok, msg = p.setSyntax(syntax_raw, allow_short=False)
                if not ok:
                    _set_check_state("red")
                    if show_error:
                        messagebox.showerror("Pattern", msg, parent=dlg)
                    return False, ""
                syntax_norm = p.getSyntax()
                _set_check_state("green")
                return True, syntax_norm

            _autocheck_after_id = None

            def _run_autocheck():
                nonlocal _autocheck_after_id
                _autocheck_after_id = None
                _check_pattern(show_error=False)

            def _schedule_autocheck(*_args):
                nonlocal _autocheck_after_id
                if _autocheck_after_id is not None:
                    try:
                        dlg.after_cancel(_autocheck_after_id)
                    except tk.TclError:
                        pass
                _autocheck_after_id = dlg.after(250, _run_autocheck)

            btn_check = tk.Button(
                frm,
                image=dlg.icon_check_gray,
                text="OK" if dlg.icon_check_gray is None else "",
                command=lambda: _check_pattern(show_error=False),
                relief=tk.FLAT,
                padx=0,
                pady=0,
            )
            btn_check.grid(row=1, column=2, sticky="w", padx=(4, 0), pady=(8, 0))
            _set_check_state("gray")
            pattern_var.trace_add("write", _schedule_autocheck)
            _schedule_autocheck()

            actions = ttk.Frame(frm)
            actions.grid(row=2, column=0, columnspan=3, sticky="e", pady=(12, 0))

            def _on_close_pattern():
                dlg.destroy()

            def _on_validate_pattern():
                ok, syntax_norm = _check_pattern(show_error=True)
                if not ok:
                    return
                if is_edit:
                    if 0 <= edit_index < len(win._patterns):
                        win._patterns[edit_index]["text"] = syntax_norm
                else:
                    win._patterns.append({"text": syntax_norm, "active": True})
                persistDecryptPatterns()
                _refresh_patterns_list()
                if win._patterns:
                    sel_idx = edit_index if is_edit else (len(win._patterns) - 1)
                    if sel_idx is not None and 0 <= sel_idx < len(win._patterns):
                        _set_selected_pattern_index(sel_idx)
                _update_pattern_buttons()
                dlg.destroy()

            ttk.Button(actions, text="Fermer", command=_on_close_pattern).pack(side=tk.RIGHT)
            ttk.Button(actions, text="Valider", command=_on_validate_pattern).pack(side=tk.RIGHT, padx=(0, 6))

            dlg.wait_visibility()
            pattern_entry.focus_set()

        def _delete_selected_pattern():
            idx = _get_selected_pattern_index()
            if idx is None:
                return
            if idx < 0 or idx >= len(win._patterns):
                return
            del win._patterns[idx]
            # selection: reste sur l'index courant (ou précédent si fin)
            if win._patterns:
                win._pattern_selected_index = min(idx, len(win._patterns) - 1)
            else:
                win._pattern_selected_index = None
            persistDecryptPatterns()
            _refresh_patterns_list()

        btn_pat_add = _make_pattern_btn(self._icons["new"], "+", lambda: _open_pattern_editor(None))
        btn_pat_add.pack(side=tk.LEFT, padx=1)
        btn_pat_edit = _make_pattern_btn(self._icons["props"], "E", lambda: _open_pattern_editor(_get_selected_pattern_index()))
        btn_pat_edit.pack(side=tk.LEFT, padx=1)
        btn_pat_delete = _make_pattern_btn(self._icons["delete"], "X", _delete_selected_pattern)
        btn_pat_delete.pack(side=tk.LEFT, padx=1)

        _refresh_patterns_list()
        _update_pattern_buttons()

        # --- Zone : Progression + Status + Start/Stop (une seule ligne) ---
        progress_frame = ttk.Frame(root)
        progress_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        # colonnes : [Progression label][bar][Status label][status][Start/Stop]
        progress_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(progress_frame, text="Progression:").grid(row=0, column=0, sticky="w", padx=(0, 8))

        win._progressVar = tk.DoubleVar(value=0.0)
        win._progressBar = ttk.Progressbar(progress_frame, variable=win._progressVar, maximum=1.0)
        win._progressBar.grid(row=0, column=1, sticky="ew")

        ttk.Label(progress_frame, text="Status:").grid(row=0, column=2, sticky="e", padx=(16, 6))

        win._engineStatusVar = tk.StringVar(value="IDLE")
        ttk.Label(progress_frame, textvariable=win._engineStatusVar, width=10).grid(row=0, column=3, sticky="w")

        win._btnStartStop = ttk.Button(progress_frame, text="Start")  # wiring ensuite
        win._btnStartStop.grid(row=0, column=4, sticky="e", padx=(16, 0))

        # --- Zone inférieure : Solutions ---
        solutions_frame = ttk.LabelFrame(root, text="Solutions")
        solutions_frame.grid(row=2, column=0, sticky="nsew")

        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        solutions_list_frame = ttk.Frame(solutions_frame)
        solutions_list_frame.pack(fill=tk.BOTH, expand=True)

        solutions_list = tk.Listbox(solutions_list_frame, height=10, selectmode="browse")
        solutions_scroll = ttk.Scrollbar(
            solutions_list_frame, orient="vertical", command=solutions_list.yview
        )
        solutions_list.configure(yscrollcommand=solutions_scroll.set)
        solutions_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        solutions_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        win._solutions = []

        def _format_solution_item(sol):
            # Affichage: MOT(r,c) MOT(r,c) ... - 2.27°
            parts = []
            for (word, (r, c)) in zip(sol.words, sol.coordsAbs):
                parts.append(f"{word}({r},{c})")

            phrase = " ".join(parts)
            score_txt = f"{sol.scoreMax:.2f}°"

            return f"{phrase} - {score_txt}"

        def _refresh_solutions_list():
            solutions_list.delete(0, tk.END)
            for sol in win._solutions:
                solutions_list.insert(tk.END, _format_solution_item(sol))

        _refresh_solutions_list()

        # --- Actions ---
        actions = ttk.Frame(root)
        actions.grid(row=3, column=0, sticky="e", pady=(8, 0))

        def _on_start_decryptage():
            persistDecryptGuiConfig()
            persistDecryptPatterns()

            scen = self._scenario_provider()
            world = scen.topoWorld
            tc = world.topologyChemins
            dico = self._dico_provider()

            patterns_actifs = []
            for p in list(win._patterns or []):
                if not isinstance(p, dict):
                    continue
                if not bool(p.get("active", False)):
                    continue
                text = str(p.get("text", "")).strip()
                if not text:
                    continue
                patterns_actifs.append(text)

            if not patterns_actifs:
                messagebox.showerror("Décryptage", "Aucun pattern actif.", parent=win)
                return

            use_angle180 = bool(win.var_angle180.get())
            use_az_a = bool(win.var_az_a.get())
            use_az_b = bool(win.var_az_b.get())
            if not (use_angle180 or use_az_a or use_az_b):
                messagebox.showerror("Décryptage", "Aucune mesure active.", parent=win)
                return

            scope_ui = str(win.var_scope.get() or "Strict").strip().lower()
            if scope_ui.startswith("mirror"):
                scope = DicoScope.MIRRORING
            elif scope_ui.startswith("ext"):
                scope = DicoScope.EXTENDED
            else:
                scope = DicoScope.STRICT

            mode_ui = str(win.var_mode.get() or "Absolu").strip().lower()
            mode_abs = not mode_ui.startswith("rel")

            # Si le décrypteur courant est celui utilisé, on garde l'instance avec son paamétage
            decryptor_id = _extract_decryptor_id(win.var_decryptor.get())
            if self._decryptor_provider().id != decryptor_id :
                decryptor = createDecryptor(decryptor_id)
            else:
                decryptor = self._decryptor_provider()

            tol = float(win.var_tolerance.get())
            liste_patterns = ListePatterns(dico, patterns_actifs)

            decryptor_cfg = DecryptorConfig(
                decryptor=decryptor,
                useAzA=use_az_a,
                useAzB=use_az_b,
                useAngle180=use_angle180,
                toleranceDeg=tol,
            )

            # Le controleur e la file d'attente pour communiquer avec l'engine
            engineControl = EngineControl()
            eventQueue = EventQueue()

            runControlConfig = RunControlConfig(
                maxSolutions=int(win.var_max_solutions.get()),
                minBatchCells=500,
                maxBatchCells=200_000,
                targetBatchSec=0.05,
                progressMinIntervalSec=0.2,
            )

            engine = DecryptorEngine(tc, dico, runControlConfig, engineControl, eventQueue)

            # stocker sur la fenêtre (prochaine étape : thread + poll queue)
            win._engine = engine
            win._engineControl = engineControl
            win._eventQueue = eventQueue

            win._solutions = []
            _refresh_solutions_list()

            if mode_abs:
                engine.runAbs(scope, liste_patterns, decryptor_cfg, patternMode="last")
            else:
                engine.runRel(scope, liste_patterns, decryptor_cfg, patternMode="last")

        # état simple du bouton Start/Stop
        win._runActive = False

        def _set_run_state(active: bool):
            win._runActive = bool(active)
            win._btnStartStop.config(text=("Stop" if win._runActive else "Start"))
            win._engineStatusVar.set("running" if win._runActive else "IDLE")

        def _start_worker():
            _on_start_decryptage()
            # quand le worker se termine normalement
            win.after(0, lambda: _set_run_state(False))

        def _on_startstop_clicked():
            if not win._runActive:
                _set_run_state(True)

                t = threading.Thread(target=_start_worker, daemon=True)
                win._workerThread = t
                t.start()

                _poll_event_queue()   # démarre le polling UI
            else:
                # Stop demandé par l'utilisateur
                win._engineControl.requestStop()
                win._engineStatusVar.set("stopping")

        def _poll_event_queue():
            q = getattr(win, "_eventQueue", None)
            if q is not None:
                while True:
                    evt = q.getNowait()
                    if evt is None:
                        break

                    etype = evt.type
                    payload = evt.payload

                    if etype == "STATUS":
                        win._engineStatusVar.set(str(payload))
                    elif etype == "PROGRESS":
                        win._progressVar.set(float(payload))
                    elif etype == "STOPPED":
                        win._engineStatusVar.set("stopped")
                        _set_run_state(False)
                    elif etype == "DONE":
                        win._engineStatusVar.set("done")
                        _set_run_state(False)
                    elif etype == "SOLUTION":
                        sol = payload
                        if sol is not None:
                            win._solutions.append(sol)
                            solutions_list.insert(tk.END, _format_solution_item(sol))
                            # optionnel : auto-scroll en bas
                            solutions_list.see(tk.END)

            # replanifie le poll
            win.after(100, _poll_event_queue)

        win._btnStartStop.config(command=_on_startstop_clicked)
        ttk.Button(actions, text="Fermer", command=_on_close).pack(side=tk.RIGHT)
