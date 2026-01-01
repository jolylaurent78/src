
import os
import json
import datetime as _dt
import math
import re
import copy
import io
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Type
from pyproj import Transformer

from tkinter import filedialog, messagebox, simpledialog
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from tksheet import Sheet


from PIL import Image, ImageTk

# --- SVG rasterization (svglib/reportlab) ---
# IMPORTANT:
# - On utilise svglib + reportlab (renderPDF) + pdfium pour rasteriser le SVG.
# - On évite volontairement renderPM ici : sur certains environnements (notamment Windows/Python 3.13),
#   reportlab peut tenter d'utiliser un backend Cairo absent et planter dès l'import.
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas as _rl_canvas
import pypdfium2 as pdfium

from shapely.geometry import Polygon as _ShPoly, LineString as _ShLine, Point as _ShPoint
from shapely.ops import unary_union as _sh_union
from shapely.ops import unary_union
from shapely.affinity import rotate, translate 

import xml.etree.ElementTree as ET


# === Modules externalisés (découpage maintenable) ===
from src.assembleur_core import (
    _overlap_shrink,
    _tri_shape,
    _group_shape_from_nodes,
    _build_local_triangle,
    _pose_params,
    _apply_R_T_on_P,
    ScenarioAssemblage,
)

from src.assembleur_sim import (
    AlgorithmeAssemblage,
    AlgoQuadrisParPaires,
    MoteurSimulationAssemblage,
    ALGOS,
)

from src.assembleur_sim import (
    DecryptorBase,
    ClockDicoDecryptor,
    DECRYPTORS,
)

import src.assembleur_io as _assembleur_io

# --- Dictionnaire (chargement livre.txt) ---
from src.DictionnaireEnigmes import DictionnaireEnigmes

EPS_WORLD = 1e-6

def createDecryptor(decryptorId: str) -> DecryptorBase:
    """Instancie un decryptor à partir de son id (registre DECRYPTORS)."""
    cls = DECRYPTORS.get(str(decryptorId))
    if cls is None:
        # Fallback explicite
        return ClockDicoDecryptor()
    return cls()


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
        default_first_edge: str = "OL",
    ):
        super().__init__(parent)
        self.title("Assembler (simulation)")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result = None  # (algo_id, n, order, first_edge)

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
        self.n_var = tk.IntVar(value=int(default_n))
        self.n_spin = ttk.Spinbox(frm, from_=2, to=max(2, int(n_max)), textvariable=self.n_var, width=8)
        self.n_spin.grid(row=2, column=1, sticky="e")

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

        # --- Placement du 1er triangle ---
        d_edge = str(default_first_edge or "OL").upper().strip()
        if d_edge not in ("OL", "BL"):
            d_edge = "OL"
        self.first_edge_var = tk.StringVar(value=d_edge)  # "OL" | "BL"
        ttk.Label(frm, text="Placement 1er triangle :").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.first_edge_combo = ttk.Combobox(
            frm,
            textvariable=self.first_edge_var,
            state="readonly",
            values=["OL = 0°", "BL = 0°"],
            width=12
        )
        self.first_edge_combo.current(1 if d_edge == "BL" else 0)
        self.first_edge_combo.grid(row=4, column=1, sticky="e", pady=(8, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Annuler", command=self._on_cancel).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="OK", command=self._on_ok).grid(row=0, column=1)

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.n_spin.focus_set()
        self.n_spin.selection_range(0, tk.END)


    def _on_cancel(self):
        self.result = None
        self.destroy()

    def _on_ok(self):
        raw = str(self.algo_combo.get() or "")
        algo_id = raw.split(" - ", 1)[0].strip() if " - " in raw else raw.strip()

        try:
            n = int(self.n_var.get())
        except Exception:
            n = 0

        if n <= 0:
            messagebox.showerror("Assembler", "Nombre de triangles invalide.")
            return

        if n % 2 == 1:
            n2 = n - 1
            if n2 < 2:
                messagebox.showerror("Assembler", "n doit être pair (minimum 2).")
                return
            # n doit être pair : on ajuste silencieusement (pas de popup)
            print(f"[SIM] n impair -> utilisation de n={n2}")
            n = n2
            try:
                self.n_var.set(n)
            except Exception:
                pass

        order = str(getattr(self, "order_var", tk.StringVar(value="forward")).get() or "forward")

        first_raw = str(getattr(self, "first_edge_var", tk.StringVar(value="OL")).get() or "OL")
        first_edge = "BL" if "BL" in first_raw else "OL"

        self.result = (algo_id, int(n), order, first_edge)

        self.destroy()

# ---------- Application (MANUEL — sans algorithmes) ----------
class TriangleViewerManual(tk.Tk):
    """
    Version épurée pour travail manuel :
      - Chargement Excel
      - Liste des triangles
      - Affichage "brut" (sans assemblage) avec mise en page simple en ligne(s)
      - Fit à l'écran, pan/zoom
      - Impression PDF des triangles bruts (même échelle)
    """
    def __init__(self):
        super().__init__()
        self.title("Assembleur de Triangles — Mode Manuel")
        self.geometry("1200x700")

        # état de vue
        # Référence STABLE pour l'anti-chevauchement en simulation.
        # IMPORTANT : ne doit pas bouger quand on fait un "fit à l'écran" (qui modifie self.zoom
        self.simulationOverlapZoomRef = 1.0
        self.zoom = 1.0
        self.offset = np.array([400.0, 350.0], dtype=float)
        self._drag = None             # état de drag & drop depuis la liste
        self._drag_preview_id = None  # id du polygone "fantôme" sur le canvas
        self._sel = None              # sélection sur canvas: {'mode': 'move'|'vertex', 'idx': int}
        self._hit_px = 12             # tolérance de hit (pixels) pour les sommets
        self._marker_px = 6           # rayon des marqueurs (cercles) dessinés aux sommets
        self._ctx_target_idx = None   # index du triangle visé par clic droit (menu contextuel)
        self._placed_ids = set()      # ids déjà posés dans le scénario actif
        self._ctx_last_rclick = None  # dernière position écran du clic droit (pour pivoter)
        self._nearest_line_id = None  # trait d'aide "sommet le plus proche"
        self._edge_highlight_ids = [] # surlignage des 2 arêtes (mobile/cible)
        self._edge_choice = None      # (i_mob, key_mob, edge_m, i_tgt, key_tgt, edge_t)
        self._edge_highlights = None  # données brutes des aides (candidates + best)
        self._tooltip = None          # tk.Toplevel
        self._tooltip_label = None    # tk.Label

        # --- cache pick (écran) régénéré après load / zoom / pan ---
        self._pick_cache_valid = False
        # chaque item de self._last_drawn recevra:
        #   t["_pick_poly"] : liste de points écran [(x,y),...]
        #   t["_pick_pts"]  : dict {'O':(x,y), 'B':(x,y), 'L':(x,y)}

        # Association triangle -> mot du dictionnaire: { tri_id: {"word": str, "row": int, "col": int} }
        self._tri_words: Dict[int, Dict[str, int | str]] = {}

        # distance écran supplémentaire pour l'ancrage du tooltip (px)
        self._tooltip_cushion_px = 14

        # Épaisseur de trait (px) — utilisée pour le test de chevauchement "shrink seul"
        self.stroke_px = 2

        # Hauteur fixe du panneau "Dico" (en pixels) sous le canvas
        self.dico_panel_height = 290
        # État d'affichage du panneau dictionnaire (toggle via menu Visualisation)
        self.show_dico_panel = tk.BooleanVar(value=True)
        # État d'affichage du compas horaire (overlay horloge)
        self.show_clock_overlay = tk.BooleanVar(value=True)
        # Mode "contours uniquement" : n'afficher que le contour de chaque groupe (pas les arêtes internes)
        self.show_only_group_contours = tk.BooleanVar(value=False)

        # Gestion des layers (visibilité)
        self.show_map_layer = tk.BooleanVar(value=True)
        self.show_triangles_layer = tk.BooleanVar(value=True)
        # Opacité du layer "carte" (0..100). 100 = opaque, 0 = invisible.
        self.map_opacity = tk.IntVar(value=70)
        self._map_opacity_redraw_job = None


        # Recentrage automatique (Fit à l'écran) lors de la sélection d'un scénario
        self.auto_fit_scenario_select = tk.BooleanVar(value=False)

        # --- Fond SVG (coordonnées monde) ---
        self.bg_resize_mode = tk.BooleanVar(value=False)
        self._bg = None  # dict: {path,x0,y0,w,h,aspect}
        self._bg_base_pil = None  # PIL.Image RGBA (base, normalisée)
        self._bg_photo = None     # ImageTk.PhotoImage (vue écran)
        self._bg_resizing = None  # dict état drag poignée
        self._bg_moving = None    # dict état drag déplacement (mode resize)

        # --- Calibration fond (3 points) ---
        self._bg_calib_active = False
        self._bg_calib_points_cfg = None  # dict chargé depuis data/maps/<carte>.calib_points.json
        self._bg_calib_clicked_world = []  # [(x,y), ...] en coordonnées monde
        self._bg_calib_step = 0

        # calibration chargée (si data/<carte>.json existe)
        self._bg_calib_data = None  # dict du fichier JSON de calibration (affineLambertKmToWorld)

        # Référence d'échelle "x1" : largeur monde du fond au moment où la calibration est chargée.
        # Sert uniquement à afficher un facteur (x1, x1/3.15, x2.49) pendant le redimensionnement.
        self._bg_scale_base_w = None

        # --- Calibration chargée (Lambert93 km -> monde) ---
        self._bg_affine_lambert_to_world = None  # [a, b, c, d, e, f]

        # mode "déconnexion" activé par CTRL
        self._ctrl_down = False        

        # --- DEBUG: permettre de SKIP le test de chevauchement durant le highlight (F9) ---
        self.debug_skip_overlap_highlight = False
        # --- DEBUG: traces console pour l'assist de collage (F10 pour basculer) ---
        self._debug_snap_assist = False

        # === Groupes (Étape A) ===
        self.groups: Dict[int, Dict] = {}
        self._next_group_id: int = 1

        # === Scénarios d'assemblage ===
        # Liste de scénarios (1 scénario manuel actif + futurs scénarios auto).
        self.scenarios: List[ScenarioAssemblage] = []
        self.active_scenario_index: int = 0

        # Scénario de référence (pour comparaison des auto)
        self.ref_scenario_token: Optional[int] = None  # id(scen)
        self._comparison_diff_indices: set = set()

        # état IHM
        self.triangle_file = tk.StringVar(value="")
        self.start_index = tk.IntVar(value=1)
        self.num_triangles = tk.IntVar(value=8)

        self.excel_path = None
        # Répertoires par défaut
        self.data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
        # Sous-répertoire dédié aux cartes (fond + fichiers de calibration)
        self.maps_dir = os.path.join(self.data_dir, "maps")
        self.scenario_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "scenario"))
        # Répertoire des icônes
        self.images_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "images"))
        os.makedirs(self.scenario_dir, exist_ok=True)
        os.makedirs(self.maps_dir, exist_ok=True)
        # === Config (persistance des paramètres) ===
        self.config_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "config"))
        self.config_path = os.path.join(self.config_dir, "assembleur_config.json")
        self.appConfig: Dict = {}
        self.loadAppConfig()

        # === UI : persistance des toggles de visualisation (dico + compas) ===
        # Doit être fait AVANT _build_ui() pour que le checkbutton + le pack initial
        # reflètent correctement l'état sauvegardé.
        self.show_dico_panel.set(bool(self.getAppConfigValue("uiShowDicoPanel", True)))
        self.show_clock_overlay.set(bool(self.getAppConfigValue("uiShowClockOverlay", True)))
        self.show_only_group_contours.set(bool(self.getAppConfigValue("uiShowOnlyGroupContours", False)))
        self.auto_fit_scenario_select.set(bool(self.getAppConfigValue("uiAutoFitScenario", False)))
        self.map_opacity.set(int(self.getAppConfigValue("uiMapOpacity", 70)))

        # === Simulation : derniers paramètres utilisés (dialog 'Simulation > Assembler…') ===
        self._simulation_last_algo_id = str(self.getAppConfigValue("simLastAlgoId", "") or "").strip()
        self._simulation_last_n = int(self.getAppConfigValue("simLastN", 8) or 8)
        self._simulation_last_order = str(self.getAppConfigValue("simLastOrder", "forward") or "forward").strip().lower()
        if self._simulation_last_order not in ("forward", "reverse"):
            self._simulation_last_order = "forward"
        self._simulation_last_first_edge = str(self.getAppConfigValue("simLastFirstEdge", "OL") or "OL").strip().upper()
        if self._simulation_last_first_edge not in ("OL", "BL"):
            self._simulation_last_first_edge = "OL"

        # Fond SVG : au démarrage le canvas n'a pas toujours une taille valide.
        # On déclenche donc un redessin différé (au 1er <Configure>) après rechargement.
        self._bg_defer_redraw = False
        self._bg_startup_scheduled = False

        self.df = None
        self._last_drawn = []   # liste d'items: (labels, P_world)


        # Crée le scénario manuel "par défaut" qui pointe sur l'état runtime.
        # last_drawn et groups sont partagés par référence : toute modification
        # manuelle met automatiquement à jour ce scénario.
        manual = ScenarioAssemblage(
            name="Scénario manuel",
            source_type="manual",
            algo_id=None,
            tri_ids=[],
        )
        manual.last_drawn = self._last_drawn
        manual.groups = self.groups
        self.scenarios.append(manual)

        # --- Horloge (overlay fixe) : état par défaut ---
        # hour peut être un float (si l'aiguille des heures avance avec les minutes)
        self._clock_state = {"hour": 5.0, "minute": 9, "label": "Trouver — (5h, 9')"}

        # --- Décryptage : stratégie active (extensible) ---
        # Par défaut: mapping "horloge <-> dico" v1
        self.decryptor: DecryptorBase = ClockDicoDecryptor()

        # Position & état de drag de l'horloge (coords CANVAS)
        self._clock_cx = None
        self._clock_cy = None
        self._clock_R  = 69    # rayon px (mis à jour dans le draw)
        # Rayon "souhaité" du compas (modifiable via boutons < > dans les layers)
        self._clock_radius = 69
        # Azimut de référence (0° = Nord, sens horaire). Sert de base pour l'axe 0 du compas.
        self._clock_ref_azimuth_deg = float(self.getAppConfigValue("uiClockRefAzimuth", 0.0) or 0.0)

        # Mode interactif : définition de l'azimut de référence
        self._clock_setref_active = False
        self._clock_setref_line_id = None
        self._clock_setref_text_id = None

        # Mode interactif : mesure d'un azimut (relatif à l'azimut de référence)
        self._clock_measure_active = False
        self._clock_measure_line_id = None
        self._clock_measure_text_id = None
        self._clock_measure_last = None  # tuple(sx, sy, az_abs, az_rel)

        # Mode interactif : mesure d'un arc d'angle (entre 2 points sur le compas)
        self._clock_arc_active = False
        self._clock_arc_step = 0        # 0: attente P1, 1: attente P2
        self._clock_arc_p1 = None       # (sx, sy, az_abs)
        self._clock_arc_p2 = None       # (sx, sy, az_abs)
        self._clock_arc_line1_id = None
        self._clock_arc_line2_id = None
        self._clock_arc_arc_id = None
        self._clock_arc_text_id = None

        # Dernière mesure validée (persistée tant que le compas reste sur le même ancrage)
        # {"az1": float, "az2": float, "angle": float}
        self._clock_arc_last = None
        self._clock_arc_last_angle_deg = None
 
        # --- Dictionnaire : filtrage visuel par angle ---
        self._dico_filter_active: bool = False
        self._dico_filter_ref_angle_deg: Optional[float] = None
        self._dico_filter_tolerance_deg: float = 4.0

        self._clock_dragging = False
        self._clock_drag_dx = 0
        self._clock_drag_dy = 0
        # "Snap" compas sur sommet le plus proche pendant le drag (CTRL au relâché = pas de snap)
        self._clock_snap_target = None

        self._build_ui()
        # Centraliser / garantir les bindings (création initiale)
        self._bind_canvas_handlers()      

        # Bind pour annuler avec ESC (drag ou sélection)
        self.bind("<Escape>", self._on_escape_key)

        # === Dictionnaire : chargement automatique de ../data/livre.txt ===
        self.dico = None
        self._init_dictionary()
        self._build_dico_grid()

    # ---------- Dictionnaire : init ----------
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
            try:
                self.status.config(text=f"Dico: échec de chargement — {e}")
            except Exception:
                pass

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
            try:
                child.destroy()
            except Exception:
                pass
        # Vérifs
        if self.dico is None:
            tk.Label(self.dicoPanel, text="Dictionnaire non chargé",
                     bg="#f3f3f3", anchor="w").pack(fill="x", padx=8, pady=6)
            return

        # ===== Paramètres Dico (N) + référentiel logique (volatile) =====
        # On affiche toujours 2N colonnes correspondant à j ∈ [-N .. N-1].
        # Le "0 logique" doit correspondre au premier mot du livre (j=0),
        # donc à la colonne physique c = N (pas à la colonne 0).
        nb_mots_max = int(self.dico.nbMotMax())
        self._dico_nb_mots_max = int(nb_mots_max)
 
        # Origine logique = cellule physique (row0,col0) qui correspond à (0,0) logique.
        # Par défaut / reset : (0, N) physique => 0 logique sur le 1er mot du livre.
        if not hasattr(self, "_dico_origin_cell") or self._dico_origin_cell is None:
            self._dico_origin_cell = (0, int(nb_mots_max))

        # --- Layout du panneau bas : [sidebar catégories] | [grille] ---
        from tkinter import ttk
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
        tk.Label(left, text="Catégorie :", anchor="w", bg="#f3f3f3").pack(anchor="w", padx=8, pady=(8,2))
        cats = []
        try:
            cats = list(self.dico.getCategories())
        except Exception:
            cats = []
        # Préserver la catégorie sélectionnée lors d'un rebuild
        cat_default = getattr(self, "_dico_cat_selected", None)
        if cat_default not in (cats or []):
            cat_default = (cats[0] if cats else "")
        self._dico_cat_var = tk.StringVar(value=cat_default)
        self._dico_cat_combo = ttk.Combobox(left, state="readonly", values=cats, textvariable=self._dico_cat_var)
        self._dico_cat_combo.pack(fill="x", padx=8, pady=(0,6))

        # Liste des mots de la catégorie sélectionnée
        lb_frame = tk.Frame(left, bg="#f3f3f3"); lb_frame.pack(fill="both", expand=True, padx=6, pady=(0,8))
        self._dico_cat_list = tk.Listbox(lb_frame, exportselection=False)
        self._dico_cat_list.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(lb_frame, orient="vertical", command=self._dico_cat_list.yview)
        sb.pack(side="right", fill="y")
        self._dico_cat_list.configure(yscrollcommand=sb.set)

        # Remplissage initial + binding de la combo
        self._dico_cat_items = []
        def _refresh_cat_list(*_):
            cat = self._dico_cat_var.get()
            self._dico_cat_selected = cat
            self._dico_cat_list.delete(0, tk.END)
            if not cat:
                return
            try:
                # getListeCategorie -> [(mot, enigme, indexMot), ...]
                items = self.dico.getListeCategorie(cat)
            except Exception:
                items = []
            # mémoriser pour la synchro avec la grille
            self._dico_cat_items = list(items)
            # Afficher: "mot — (eDisp, mDisp)" (index d’affichage)
            # - Mode ABS (origine par défaut) : colonnes sans 0 (1=1er mot), lignes 1..10
            # - Mode DELTA (origine cliquée) : colonnes/énigmes en delta avec 0 (pas d’énigmes négatives -> modulo 10)
            nb_mots_max = self._dico_nb_mots_max if hasattr(self, "_dico_nb_mots_max") else self.dico.nbMotMax()
            default_origin = (0, int(nb_mots_max))
            r0, c0 = (self._dico_origin_cell or default_origin)
            isDelta = tuple((r0, c0)) != tuple(default_origin)

            origin_indexMot = int(c0) - int(nb_mots_max)
            for mot, e, m in items:
                # Lignes : pas d’énigmes négatives
                eLogRaw = int(e) - int(r0)
                eLog = int(eLogRaw) % 10
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
            try:
                sel = self._dico_cat_list.curselection()
                if not sel:
                    return
                i = int(sel[0])
                mot, enigme, indexMot = self._dico_cat_items[i]
            except Exception:
                return
            # convertir indexMot [-N..N) -> colonne [0..2N)
            try:
                nb_mots_max = self._dico_nb_mots_max if hasattr(self, "_dico_nb_mots_max") else self.dico.nbMotMax()
            except Exception:
                nb_mots_max = 50
            col = int(indexMot) + int(nb_mots_max)
            row = int(enigme)
            # sélectionner et faire voir la cellule ; see() centre autant que possible
            try:
                self.dicoSheet.select_cell(row, col, redraw=False)
            except Exception:
                pass
            try:
                self.dicoSheet.see(row=row, column=col)
            except Exception:
                pass
            # MAJ horloge (sélection indirecte via la liste)
            try:
                self._update_clock_from_cell(row, col)
            except Exception:
                pass

        # Clic et double-clic compatibles
        self._dico_cat_list.bind("<<ListboxSelect>>", _goto_selected_word)
        self._dico_cat_list.bind("<Double-Button-1>", _goto_selected_word)

        # ===== Grille (tksheet) =====
        # Construire la matrice
        # Deux modes d’affichage :
        # - Mode ABS (origine par défaut (0, nbm)) : colonnes sans 0 (1=1er mot), lignes 1..10
        # - Mode DELTA (origine cliquée) : colonnes/énigmes en delta avec 0 (lignes en modulo 10)
        default_origin = (0, int(nb_mots_max))
        r0, c0 = (self._dico_origin_cell or default_origin)
        isDelta = tuple((r0, c0)) != tuple(default_origin)

        # Entêtes d'affichage
        if isDelta:
            # DELTA : 0 sur la colonne d’origine
            headers = [int(c) - int(c0) for c in range(0, 2 * int(nb_mots_max))]
        else:
            # ABS : 1 = 1er mot (j=0), pas de colonne 0
            # Physique c ∈ [0..2N-1] ↔ j = c - N ∈ [-N..N-1]
            headers = []
            for c in range(0, 2 * int(nb_mots_max)):
                j = int(c) - int(nb_mots_max)
                headers.append(int(j) if int(j) < 0 else (int(j) + 1))

        data = []
        for i in range(len(self.dico)):
            try:
                row = [self.dico[i][j] for j in range(-nb_mots_max, nb_mots_max)]
            except Exception:
                # en cas de dépassement, remplir de chaînes vides
                row = ["" for _ in range(2 * nb_mots_max)]
            data.append(row)


        row_titles_raw = self.dico.getTitres()   # ["530", "780", ...]


        # utilisé par le decryptor / compas (NE PAS TOUCHER)
        self._dico_row_index = list(row_titles_raw)

        # affichage UI : indexes
        # - DELTA : lignes en modulo 10 (pas d'énigmes négatives), origine = 0
        # - ABS   : 1..10
        if isDelta:
            row_index_display = [f'{((i - int(r0)) % 10)} - "{t}"' for i, t in enumerate(row_titles_raw)]
        else:
            row_index_display = [f'{(((i - int(r0)) % 10) + 1)} - "{t}"' for i, t in enumerate(row_titles_raw)]

        # Créer la grille
        self.dicoSheet = Sheet(
            right,
            data=data,
            headers=headers,
            row_index=row_index_display,
            show_row_index=True,
            height=max(120, getattr(self, "dico_panel_height", 220) - 10),
            empty_vertical=0,
        )
        # NOTE: on désactive le menu contextuel interne TkSheet ("copy", etc.)
        # car il entre en conflit avec notre menu "Définir comme origine (0,0)".
        self.dicoSheet.enable_bindings((
            "single_select","row_select","column_select",
            "arrowkeys","copy","rc_select","double_click"
        ))
        self.dicoSheet.align_columns(columns="all", align="center")
        self.dicoSheet.set_options(cell_align="center")
        self.dicoSheet.pack(expand=True, fill="both")

        # --- MAJ horloge sur sélection de cellule (événement unique & propre) ---
        def _on_dico_cell_select(event=None):
            # La sélection directe dans la grille doit toujours synchroniser le compas.
            # Source unique de vérité : cellule actuellement sélectionnée
            sel = self.dicoSheet.get_selected_cells()
            r = c = None
            if sel:
                # tksheet peut renvoyer un set({(r,c), ...}) ou une liste([(r,c), ...])
                if isinstance(sel, set):
                    r, c = next(iter(sel))
                elif isinstance(sel, (list, tuple)):
                    r, c = sel[0]
            if r is None or c is None:
                # Secours léger : cellule "courante" si aucune sélection renvoyée
                cur = self.dicoSheet.get_currently_selected()
                if isinstance(cur, tuple) and len(cur) == 2 and cur[0] == "cell":
                    r, c = cur[1]
            if r is None or c is None:
                return
            self._update_clock_from_cell(int(r), int(c))

        # Un seul binding tksheet, propre : "cell_select"
        def _on_dico_double_click(event=None):
            # On ne traite QUE si double-clic sur une cellule.
            # (évite les pièges : header / row_index / vide)
            cur = self.dicoSheet.get_currently_selected()
            if not (isinstance(cur, tuple) and len(cur) == 2 and cur[0] == "cell"):
                return
            try:
                rr, cc = cur[1]
                rr = int(rr); cc = int(cc)
            except Exception:
                return
            # Toggle : même cellule => reset défaut (0 logique = 1er mot du livre)
            nbm = int(getattr(self, "_dico_nb_mots_max", 50))
            default_origin = (0, nbm)
            if tuple(self._dico_origin_cell or default_origin) == (rr, cc):
                self._dico_origin_cell = default_origin
            else:
                self._dico_origin_cell = (rr, cc)
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
                label="Réinitialiser origine (0,0)",
                command=lambda: self._dico_reset_origin()
            )
        # cellule visée par le clic-droit (row,col) en coordonnées TkSheet
        self._dico_ctx_cell = None

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
                try:
                    if meth_row.startswith("identify_"):
                        r = getattr(obj, meth_row)(event)
                        c = getattr(obj, meth_col)(event)
                    else:
                        r = getattr(obj, meth_row)(y)
                        c = getattr(obj, meth_col)(x)
                    if r is None or c is None:
                        return None
                    return int(r), int(c)
                except Exception:
                    return None

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
                try:
                    self.dicoSheet.select_cell(rr, cc, redraw=False)
                except Exception:
                    # si select_cell diffère selon version, on laisse la sélection courante
                    pass
            try:
                self._ctx_menu_dico.tk_popup(event.x_root, event.y_root)
                self._ctx_menu_dico.grab_release()
            except Exception:
                pass
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
        if not getattr(self, "_dico_ctx_cell", None):
            return
        rr, cc = self._dico_ctx_cell
        rr = int(rr); cc = int(cc)
        # Toggle (même cellule => reset défaut: 0 logique = 1er mot du livre)
        nbm = int(getattr(self, "_dico_nb_mots_max", 50))
        default_origin = (0, nbm)
        if tuple(self._dico_origin_cell or default_origin) == (rr, cc):
            self._dico_origin_cell = default_origin
        else:
            self._dico_origin_cell = (rr, cc)
        self._build_dico_grid()

    def _dico_reset_origin(self):
        # Origine par défaut = 1er mot du livre :
        #  - ligne physique 0 (énigme 0)
        #  - colonne physique nbm (car headers = [-nbm..0..+nbm])
        nbm = int(getattr(self, "_dico_nb_mots_max", 50))
        self._dico_origin_cell = (0, nbm)
        self._build_dico_grid()

    # ---------- DICO → Horloge ----------
    def _update_clock_from_cell(self, row: int, col: int):
        """Met à jour l'horloge à partir d'une cellule de la grille Dico.
        La conversion (row,col)->(hour,minute,label) est déléguée au decryptor actif.
        """
        if not getattr(self, "dicoSheet", None):
            return

 
        nbm = int(getattr(self, "_dico_nb_mots_max", 50))
        row_titles = getattr(self, "_dico_row_index", [])
        try:
            word = str(self.dicoSheet.get_cell_data(int(row), int(col))).strip()
        except Exception:
            word = ""
 
        # Externalisation : conversion via decryptor
        try:
            # --- Passage en référentiel LOGIQUE pour le compas/decryptor ---
            nbm = int(getattr(self, "_dico_nb_mots_max", 50))

            default_origin = (0, int(nbm))
            r0, c0 = (self._dico_origin_cell or default_origin)
            isDelta = tuple((int(r0), int(c0))) != tuple(default_origin)
            mode = "delta" if isDelta else "abs"

            if isDelta:
                # DELTA : 0 autorisé. Lignes en modulo 10 (pas d’énigmes négatives)
                rowVal = (int(row) - int(r0)) % 10
                colVal = int(col) - int(c0)  # delta signé (0 autorisé)
            else:
                # ABS : pas de 0. Lignes 1..10, Colonnes … -2, -1, 1, 2, …
                rowVal = ((int(row) - int(r0)) % 10) + 1
                j = int(col) - int(nbm)  # j ∈ [-nbm..nbm-1]
                colVal = int(j) if int(j) < 0 else (int(j) + 1)

            st = self.decryptor.clockStateFromDicoCell(
                row=int(rowVal),
                col=int(colVal),
                nbMotsMax=int(nbm),
                rowTitles=list(row_titles) if row_titles else None,
                word=word,
                mode=str(mode),
            )
        except Exception:
            # En cas d'erreur du decryptor, ne pas casser l'UI
            return

        self._clock_state.update({"hour": float(st.hour), "minute": int(st.minute), "label": str(st.label)})
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

        nbm = int(getattr(self, "_dico_nb_mots_max", 50))
        row_titles = getattr(self, "_dico_row_index", [])
        n_rows = int(len(self.dico)) if self.dico is not None else 0
        n_cols = int(2 * nbm)

        ref = float(self._dico_filter_ref_angle_deg) % 180.0
        tol = float(getattr(self, "_dico_filter_tolerance_deg", 4.0))

        default_origin = (0, int(nbm))
        r0, c0 = (self._dico_origin_cell or default_origin)
        isDelta = tuple((int(r0), int(c0))) != tuple(default_origin)
        mode = "delta" if isDelta else "abs"
        for r in range(n_rows):
            for c in range(n_cols):
                word = str(self.dicoSheet.get_cell_data(r, c)).strip()
                if not word:
                    continue
                # --- Passage en référentiel LOGIQUE pour l'angle ---
                if isDelta:
                    rowVal = (int(r) - int(r0)) % 10
                    colVal = int(c) - int(c0)
                else:
                    rowVal = ((int(r) - int(r0)) % 10) + 1
                    j = int(c) - int(nbm)
                    colVal = int(j) if int(j) < 0 else (int(j) + 1)

                # IMPORTANT:
                #   Le filtre doit être cohérent avec ce que le compas affichera quand on clique une cellule.
                #   Or le compas affiche Δ = angle entre aiguilles (heure/minute) calculé à partir de
                #   clockStateFromDicoCell().
                #   On n'utilise donc PAS deltaAngleFromDicoCell (qui peut avoir une convention différente)
                #   mais exactement la même définition que l'overlay du compas.
                st = self.decryptor.clockStateFromDicoCell(
                    row=int(rowVal),
                    col=int(colVal),
                    nbMotsMax=int(nbm),
                    rowTitles=list(row_titles) if row_titles else None,
                    word=word,
                    mode=str(mode),
                )
                hFloat = float(getattr(st, "hour", 0.0)) % 12.0
                m = int(getattr(st, "minute", 0)) % 60
                ang_hour = (hFloat * 30.0) % 360.0
                ang_min = (m * 6.0) % 360.0
                ang = float(self._clock_arc_compute_angle_deg(float(ang_hour), float(ang_min)))
                ok = abs(ang - ref) <= tol
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
        self.dicoSheet.highlight_cells(int(r0), int(c0), fg="#9A9A9A", bg="#BFDFFF")

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
            if isinstance(sel, set):
                r, c = next(iter(sel))
            elif isinstance(sel, (list, tuple)):
                r, c = sel[0]
        if r is None or c is None:
            cur = self.dicoSheet.get_currently_selected()
            if isinstance(cur, tuple) and len(cur) == 2 and cur[0] == "cell":
                r, c = cur[1]
        if r is None or c is None:
            return None
        try:
            word = str(self.dicoSheet.get_cell_data(int(r), int(c))).strip()
        except Exception:
            word = ""
        if not word:
            return None
        return (word, int(r), int(c))

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
        try:
            # On supprime depuis la fin pour conserver l'index d'ancrage
            end = self._ctx_menu.index("end")
            while end > self._ctx_idx_words_start:
                self._ctx_menu.delete(end)
                end = self._ctx_menu.index("end")
        except Exception:
            pass
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
    def _toggle_skip_overlap_highlight(self, event=None):
        self.debug_skip_overlap_highlight = not self.debug_skip_overlap_highlight
        state = "IGNORE" if self.debug_skip_overlap_highlight else "APPLIQUE"
        self.status.config(text=f"[DEBUG] Chevauchement (highlight): {state} — F9 pour basculer")

    def _toggle_debug_snap_assist(self, event=None):
        self._debug_snap_assist = not bool(getattr(self, "_debug_snap_assist", False))
        state = "ON" if self._debug_snap_assist else "OFF"
        self.status.config(text=f"[DEBUG] Snap assist: {state} — F10 pour basculer")

    # ---------- UI ----------
    def _build_ui(self):
        # --- Barre de menus ---
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # --- Menu Scénario (save/load XML) ---
        self.menu_scenario = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Scénario", menu=self.menu_scenario)
        self.menu_scenario.add_command(label="Nouveau", command=self._new_empty_scenario)
        self.menu_scenario.add_command(label="Charger…", command=self._scenario_load_dialog)
        self.menu_scenario.add_command(label="Enregistrer…", command=self._scenario_save_dialog)
        self.menu_scenario.add_separator()
        # placeholder pour la liste des .xml du dossier 'scenario'
        self._menu_scenario_files_anchor = self.menu_scenario.index("end")
        self.menu_scenario.add_command(label="(scan des scénarios…)", state="disabled")
        self._rebuild_scenario_file_list_menu()

        # --- Menu Triangle (save/load XML) ---
        self.menu_triangle = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Triangle", menu=self.menu_triangle)

        # --- Menu Simulation (assemblage automatique) ---
        self.menu_simulation = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulation", menu=self.menu_simulation)
        self.menu_simulation.add_command(label="Assembler…", command=self._simulation_assemble_dialog)
        self.menu_simulation.add_command(label="Supprimer les scénarios automatiques", command=self._simulation_clear_auto_scenarios)
        self.menu_simulation.add_separator()

        # --- Menu Visualisation ---
        self.menu_visual = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualisation", menu=self.menu_visual)
        self.menu_visual.add_command(
            label="Fit à l'écran",
            command=lambda: self._fit_to_view(self._last_drawn)
        )
        self.menu_visual.add_checkbutton(
            label="Recentrer automatiquement sur le scénario",
            variable=self.auto_fit_scenario_select,
            command=self._toggle_auto_fit_scenario_select
        )
        # Effacer l'affichage depuis le menu
        self.menu_visual.add_command(
            label="Effacer l'affichage…",
            command=self.clear_canvas
        )

        # Toggle d'affichage du dictionnaire (panneau bas + combo/liste)
        self.menu_visual.add_checkbutton(
            label="Afficher le dictionnaire",
            variable=self.show_dico_panel,
            command=self._toggle_dico_panel
        )

        # --- Menu Carte ---
        self.menu_carte = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Carte", menu=self.menu_carte)
        self.menu_carte.add_command(label="Charger une carte…", command=self._bg_load_svg_dialog)
        self.menu_carte.add_command(label="Calibrer la carte…", command=self._bg_calibrate_start)

        self.menu_carte.add_checkbutton(
            label="Redimensioner et déplacer la carte",
            variable=self.bg_resize_mode,
            command=self._toggle_bg_resize_mode
        )
        self.menu_carte.add_command(label="Supprimer fond", command=self._bg_clear)

        # Items fixes
        self.menu_triangle.add_command(label="Ouvrir un fichier Triangle…", command=self.open_excel)
        self.menu_triangle.add_separator()
        # Espace réservé: liste de fichiers du répertoire data (reconstruite dynamiquement)
        # On pose un placeholder que l'on remplace juste après.
        self._menu_triangle_files_start_index = self.menu_triangle.index("end") + 1 if self.menu_triangle.index("end") is not None else 2
        self.menu_triangle.add_command(label="(scan en cours)")
        self.menu_triangle.add_separator()
        self.menu_triangle.add_command(label="Imprimer", command=self.print_triangles_dialog)
        # Construit la liste de fichiers disponibles
        self._rebuild_triangle_file_list_menu()

        # Zone principale : panneau gauche redimensionnable (liste) | panneau droit (canvas+dico)
        main = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6)
        main.pack(fill=tk.BOTH, expand=True)
        self.mainPaned = main

        # Panneau gauche (ajouté automatiquement dans le PanedWindow)
        self._build_left_pane(main)

        # Panneau droit
        right = tk.Frame(main)
        main.add(right, minsize=400)
        self._build_canvas(right)

        self.status = tk.Label(self, text="Prêt", bd=1, relief=tk.SUNKEN, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Auto-load: recharger le dernier fichier triangles ouvert (config)
        # sinon fallback sur data/triangle.xlsx si présent.
        self.autoLoadTrianglesFileAtStartup()
        self.autoLoadBackgroundAtStartup()

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
    def _simulation_get_tri_ids_first_n(self, n: int) -> List[int]:
        """Retourne les IDs logiques des n premiers triangles (ordre de la listbox)."""
        ids: List[int] = []
        if not hasattr(self, "listbox"):
            return ids
        max_n = int(self.listbox.size())
        n2 = min(int(n), max_n)
        for i in range(n2):
            s = str(self.listbox.get(i))
            m = re.match(r"\s*(\d+)\.", s)
            if m:
                ids.append(int(m.group(1)))
        return ids

    def _simulation_get_tri_ids_by_order(self, n: int, order: str = "normal") -> List[int]:
        """Retourne les IDs logiques des triangles selon l'ordre choisi:
        - normal : n premiers (début de listbox)
        - inverse : n derniers en partant du dernier (ex: 32,31,30,...)
        """
        ids: List[int] = []
        if not hasattr(self, "listbox"):
            return ids

        max_n = int(self.listbox.size())
        n2 = min(int(n), max_n)

        if str(order).lower() in ("inverse", "reverse"):
            # prendre les n derniers, en commençant par le dernier (max_n-1)
            for k in range(n2):
                i = (max_n - 1) - k
                s = str(self.listbox.get(i))
                m = re.match(r"\s*(\d+)\.", s)
                if m:
                    ids.append(int(m.group(1)))
            return ids

        # normal
        return self._simulation_get_tri_ids_first_n(n)

    def _simulation_clear_auto_scenarios(self):
        """Supprime tous les scénarios 'auto' (conserve les manuels)."""
        if not getattr(self, "scenarios", None):
            return
        kept = [s for s in self.scenarios if getattr(s, "source_type", "manual") == "manual"]
        removed = len(self.scenarios) - len(kept)
        self.scenarios = kept
        if not self.scenarios:
            self.scenarios = [ScenarioAssemblage(name="Scénario manuel", source_type="manual")]
        self.active_scenario_index = min(self.active_scenario_index, len(self.scenarios) - 1)
        self._set_active_scenario(self.active_scenario_index)
        self._refresh_scenario_listbox()
        try:
            self.status.config(text=f"Scénarios auto supprimés : {removed}")
        except Exception:
            pass

    def _simulation_assemble_dialog(self):
        """Ouvre la boîte de dialogue 'Assembler…' et lance l'algo choisi."""
        if getattr(self, "df", None) is None:
            messagebox.showwarning("Assembler", "Charge d'abord un fichier Triangle.")
            return
        if not hasattr(self, "listbox"):
            messagebox.showwarning("Assembler", "Listbox des triangles introuvable.")
            return

        n_max = int(self.listbox.size())
        if n_max < 2:
            messagebox.showwarning("Assembler", "Il faut au moins 2 triangles dans la liste.")
            return

        algo_items = [(aid, cls.label) for aid, cls in ALGOS.items()]
        default_algo_id = getattr(self, "_simulation_last_algo_id", "") or (algo_items[0][0] if algo_items else "")
        default_n = getattr(self, "_simulation_last_n", n_max)
        default_order = getattr(self, "_simulation_last_order", "forward")
        default_first_edge = getattr(self, "_simulation_last_first_edge", "OL")
        default_n = min(int(default_n), n_max)
        if default_n < 2:
            default_n = 2

        dlg = DialogSimulationAssembler(
            self,
            algo_items,
            n_max=n_max,
            default_algo_id=default_algo_id,
            default_n=default_n,
            default_order=default_order,
            default_first_edge=default_first_edge,
        )
        self.wait_window(dlg)
        if not getattr(dlg, "result", None):
            return

        algo_id, n, order, first_edge = dlg.result
        self._simulation_last_order = order
        self._simulation_last_algo_id = algo_id
        self._simulation_last_n = int(n)

        self._simulation_last_first_edge = str(first_edge or "OL").upper().strip()
        if self._simulation_last_first_edge not in ("OL", "BL"):
            self._simulation_last_first_edge = "OL"

        # Persister (app config)
        self.setAppConfigValue("simLastAlgoId", str(self._simulation_last_algo_id or ""))
        self.setAppConfigValue("simLastN", int(self._simulation_last_n or 0))
        self.setAppConfigValue("simLastOrder", str(self._simulation_last_order or "forward"))
        self.setAppConfigValue("simLastFirstEdge", str(self._simulation_last_first_edge or "OL"))
        self.saveAppConfig()

        # Par design : on détruit systématiquement les scénarios auto existants
        self._simulation_clear_auto_scenarios()

        # --- Construire la liste des triangles selon l'ordre choisi ---
        tri_ids = self._simulation_get_tri_ids_by_order(n, order)

        # Sécurité
        if len(tri_ids) < 2:
            messagebox.showwarning("Assembler", "Impossible de construire la liste des IDs de triangles.")
            return

        # Forcer n pair
        if len(tri_ids) % 2 == 1:
            tri_ids = tri_ids[:-1]

        try:
            engine = MoteurSimulationAssemblage(self)
            engine.firstTriangleEdge = str(first_edge or "OL").upper()
            algo_cls = ALGOS.get(algo_id)
            if algo_cls is None:
                raise ValueError(f"Algo inconnu: {algo_id}")
            algo = algo_cls(engine)
            scenarios = algo.run(tri_ids)
        except NotImplementedError as e:
            messagebox.showinfo("Assembler", str(e))
            return
        except Exception as e:
            messagebox.showerror("Assembler", str(e))
            return

        if not scenarios:
            print("[SIM] Aucun scénario généré.")
            dbg = getattr(engine, "debug_last", None)
            if isinstance(dbg, dict):
                print(f"[SIM] debug.tri_ids = {dbg.get('tri_ids')}")
                if dbg.get("pair") is not None:
                    print(f"[SIM] debug.pair = {dbg.get('pair')}")
                if dbg.get("step"):
                    print(f"[SIM] debug.step = {dbg.get('step')}")
                if dbg.get("reason"):
                    print(f"[SIM] debug.reason = {dbg.get('reason')}")
                if dbg.get("detail"):
                    print(f"[SIM] debug.detail = {dbg.get('detail')}")
            return

        # --- DEBUG : cohérence des GROUPES retournés par l'algo ---
        # Objectif: comprendre pourquoi, pour un même jeu de triangles, l'assemblage auto
        # peut laisser plusieurs groupes (ou un triangle "non assemblé") alors que le manuel aboutit.
        # On ne modifie PAS la logique métier ici : on trace uniquement.
        try:
            tri_ids_set = {int(x) for x in (tri_ids or [])}
        except Exception:
            tri_ids_set = set()

        for _k, _sc in enumerate(scenarios):
            try:
                ld = getattr(_sc, "last_drawn", None) or []
                gr = getattr(_sc, "groups", None) or {}
                drawn_ids = {int(t.get("id")) for t in ld if t.get("id") is not None}

                grouped_ids = set()
                for _gid, _g in (gr or {}).items():
                    for _nd in (_g or {}).get("nodes", []) or []:
                        _tid = _nd.get("tid")
                        if _tid is None or not (0 <= int(_tid) < len(ld)):
                            continue
                        _id = ld[int(_tid)].get("id")
                        if _id is not None:
                            grouped_ids.add(int(_id))

                missing_in_groups = sorted(drawn_ids - grouped_ids)
                missing_in_drawn = sorted(tri_ids_set - drawn_ids) if tri_ids_set else []

                if missing_in_groups or missing_in_drawn or len(gr or {}) != 1:
                    # Trace uniquement les scénarios "suspects" pour éviter le spam.
                    print(
                        f"[SIM][CHK] scen#{_k+1} groups={len(gr or {})} "
                        f"drawn={len(drawn_ids)} grouped={len(grouped_ids)} "
                        f"missing_in_groups={missing_in_groups} missing_in_drawn={missing_in_drawn}"
                    )
                    if len(gr or {}) > 1:
                        for __gid, __g in (gr or {}).items():
                            __ids = []
                            for __nd in (__g or {}).get("nodes", []) or []:
                                __tid = __nd.get("tid")
                                if __tid is None or not (0 <= int(__tid) < len(ld)):
                                    continue
                                __id = ld[int(__tid)].get("id")
                                if __id is not None:
                                    __ids.append(int(__id))
                            print(f"[SIM][CHK]   group {__gid}: tri_ids={__ids}")
            except Exception:
                # Debug: ne jamais casser l'IHM sur une trace.
                pass

        base_idx = len(self.scenarios)
        count_auto = sum(1 for s in self.scenarios if s.source_type == "auto")
        for k, scen in enumerate(scenarios):
            if not isinstance(scen, ScenarioAssemblage):
                continue
            scen.source_type = "auto"
            scen.algo_id = scen.algo_id or algo_id
            scen.tri_ids = scen.tri_ids or list(tri_ids)
            if not scen.name:
                scen.name = f"Auto #{count_auto + k + 1}"
            self.scenarios.append(scen)

        self._refresh_scenario_listbox()
        try:
            self._set_active_scenario(base_idx)
        except Exception:
            pass
        try:
            self.status.config(text=f"Simulation: {len(scenarios)} scénario(s) généré(s) (algo={algo_id}, n={n})")
        except Exception:
            pass

    def _rebuild_triangle_file_list_menu(self):
        """
        Reconstruit la portion du menu 'Triangle' listant les fichiers disponibles
        dans self.data_dir. Charge direct au clic.
        """
        m = self.menu_triangle
        # Supprimer les anciens items listés entre la 1re séparatrice et la 2e.
        # Organisation actuelle: [Ouvrir][sep][FICHIERS...][sep][Imprimer]
        # On repère la 2e séparatrice en partant de la fin.
        last_index = m.index("end")
        if last_index is None:
            return
        # Trouver l'index de la 2ème séparatrice (celle avant "Imprimer")
        # Simplification: on sait que l’avant-dernier item est une séparatrice.
        second_sep_index = last_index - 1
        # On efface tous les items entre la 1re séparatrice (index 1) exclue et second_sep_index exclu
        # Indices actuels: 0:"Ouvrir", 1:"sep", [2..n-2]=fichiers, n-1:"sep", n:"Imprimer"
        # On retire de n-2 jusqu’à 2 pour éviter le décalage pendant la suppression
        for i in range(second_sep_index - 1, 1, -1):
            try:
                m.delete(i)
            except Exception:
                pass
        # Repeupler
        try:
            files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".xlsx")]
        except Exception:
            files = []
        if not files:
            m.insert_command(2, label="(aucun fichier trouvé dans data)", state="disabled")
        else:
            # Trier par nom lisible
            files.sort(key=lambda s: s.lower())
            for idx, fname in enumerate(files):
                full = os.path.join(self.data_dir, fname)
                # Insérer à partir de l’index 2 (après la 1re séparatrice)
                m.insert_command(2 + idx, label=fname, command=lambda p=full: self.load_excel(p))

    # ---------- Icônes ----------
    def _load_icon(self, filename: str):
        """
        Charge une icône depuis le dossier images.
        Retourne un tk.PhotoImage ou None si échec (fichier manquant, etc.).
        """
        try:
            base = getattr(self, "images_dir", None)
            if not base:
                return None
            path = os.path.join(base, filename)
            if not os.path.isfile(path):
                return None
            return tk.PhotoImage(file=path)
        except Exception:
            return None

    def _build_left_pane(self, parent):
        style = ttk.Style()
        style.configure(
            "Bold.TLabelframe.Label",
            font=(None, 9, "bold")
        )

        left = tk.Frame(parent, width=260)
        # Si le parent est un PanedWindow horizontal, on ajoute le panneau gauche dedans
        # => l'utilisateur peut ensuite redimensionner la largeur via le séparateur.
        if isinstance(parent, tk.PanedWindow):
            parent.add(left, minsize=220)
        else:
            left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        # PanedWindow vertical : triangles | layers | scénarios
        # IMPORTANT: sash visible (séparateur) pour rendre le redimensionnement horizontal clair.
        pw = tk.PanedWindow(left, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=6)
        pw.pack(fill=tk.BOTH, expand=True)

        # --- Panneau haut : liste des triangles ---
        # Pas de LabelFrame ici : on retire le cadre (il ne sert plus à rien avec le header pliable).
        tri_frame = tk.Frame(pw, bd=0, highlightthickness=0)

        # Hauteurs mini (expanded vs collapsed)
        tri_minsize_expanded = 150  # hauteur mini raisonnable pour les triangles

        # État plié/déplié (on garde la variable si elle existe déjà)
        if not hasattr(self, "_ui_triangles_collapsed"):
            self._ui_triangles_collapsed = tk.BooleanVar(value=False)

        header = tk.Frame(tri_frame)
        header.pack(fill=tk.X, pady=(0, 2))

        def _calcTrianglesExpandedHeightPx():
            """
            Calcule une hauteur 'confort' pour afficher ~10 lignes dans la listbox.
            Retourne None si la listbox n'existe pas encore.
            """
            if not hasattr(self, "listbox") or self.listbox is None:
                return None
            try:
                # hauteur d'une ligne selon la police réelle de la listbox
                f = tkfont.Font(font=self.listbox.cget("font"))
                line_h = int(f.metrics("linespace"))
                # 10 lignes + padding interne + petits extras (bords / marges)
                rows = 10
                lb_h = rows * line_h
                # un peu de marge pour éviter d'être "pile"
                lb_h += 10
                # header + content paddings (approximations stables)
                hdr_h = int(header.winfo_reqheight() or 26)
                return int(hdr_h + lb_h + 18)
            except Exception:
                return None

        def _toggleTrianglesPanel():
            collapsed = bool(self._ui_triangles_collapsed.get())
            self._ui_triangles_collapsed.set(not collapsed)
            if self._ui_triangles_collapsed.get():
                # plier : cacher le contenu
                self._ui_triangles_content.pack_forget()
                self._ui_triangles_toggle_btn.config(text="▸")
            else:
                # déplier : ré-afficher le contenu
                self._ui_triangles_content.pack(fill=tk.BOTH, expand=True)
                self._ui_triangles_toggle_btn.config(text="▾")

            # Rafraîchir la géométrie (important avec le PanedWindow)
            tri_frame.update_idletasks()

            # Réduire/étendre réellement la pane pour éviter l'espace vide.
            hdr_h = int(header.winfo_reqheight() or 0)
            tri_minsize_collapsed = max(28, hdr_h + 10)
            if self._ui_triangles_collapsed.get():
                pw.paneconfigure(tri_frame, minsize=tri_minsize_collapsed, height=tri_minsize_collapsed)
            else:
                target_h = _calcTrianglesExpandedHeightPx()
                if target_h is None:
                    pw.paneconfigure(tri_frame, minsize=tri_minsize_expanded)
                else:
                    target_h = max(int(tri_minsize_expanded), int(target_h))
                    pw.paneconfigure(tri_frame, minsize=tri_minsize_expanded, height=target_h)

        # Bouton toggle + titre cliquable
        self._ui_triangles_toggle_btn = tk.Button(
            header,
            text=("▸" if self._ui_triangles_collapsed.get() else "▾"),
            width=2,
            command=_toggleTrianglesPanel
        )
        self._ui_triangles_toggle_btn.pack(side=tk.LEFT, padx=(0, 4))

        title_lbl = tk.Label(header, text="Triangles (ordre)", font=(None, 9, "bold"))
        title_lbl.pack(side=tk.LEFT, anchor="w")
        # Cliquer sur le titre plie/déplie aussi (plus “VS Code”)
        title_lbl.bind("<Button-1>", lambda _e: _toggleTrianglesPanel())
        header.bind("<Button-1>", lambda _e: _toggleTrianglesPanel())

        # Contenu pliable
        self._ui_triangles_content = tk.Frame(tri_frame)
        # (pack conditionnel selon l'état initial)
        if not self._ui_triangles_collapsed.get():
            self._ui_triangles_content.pack(fill=tk.BOTH, expand=True)

        lb_frame = tk.Frame(self._ui_triangles_content, bd=0, highlightthickness=0)
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        # Listbox sans « cadre » (cohérent avec le panneau sans LabelFrame)
        self.listbox = tk.Listbox(
            lb_frame,
            width=34,
            exportselection=False,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_scroll = tk.Scrollbar(lb_frame, orient="vertical", command=self.listbox.yview)
        lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.configure(yscrollcommand=lb_scroll.set)
        # Gestion de la sélection / triangles déjà utilisés
        self._last_triangle_selection = None
        self._in_triangle_select_guard = False
        self.listbox.bind("<<ListboxSelect>>", self._on_triangle_list_select)
        # Démarrer le drag dès qu'on clique sur un item de triangle
        self.listbox.bind("<ButtonPress-1>", self._on_list_mouse_down)

        pw.add(tri_frame, minsize=tri_minsize_expanded)  # hauteur mini raisonnable pour les triangles

        # Si on démarre "déplié", on force une hauteur par défaut (~10 lignes visibles)
        if not bool(self._ui_triangles_collapsed.get()):
            tri_frame.update_idletasks()
            h0 = _calcTrianglesExpandedHeightPx()
            if h0 is not None:
                h0 = max(int(tri_minsize_expanded), int(h0))
                pw.paneconfigure(tri_frame, height=h0)


        # --- Panneau intermédiaire : layers ---
        # Même approche que "Triangles" : header pliable (sans encadrement) + resize réel de la pane.
        layer_minsize_expanded = 80

        if not hasattr(self, "_ui_layers_collapsed"):
            self._ui_layers_collapsed = tk.BooleanVar(value=False)

        layer_frame = tk.Frame(pw, bd=0, highlightthickness=0)

        layer_header = tk.Frame(layer_frame)
        layer_header.pack(fill=tk.X, pady=(0, 2))

        # Séparateur visuel sous le header (utile quand les cadres sont supprimés)
        ttk.Separator(layer_frame, orient="horizontal").pack(fill=tk.X, pady=(0, 4))

        self._ui_layers_content = tk.Frame(layer_frame)
        if not self._ui_layers_collapsed.get():
            self._ui_layers_content.pack(fill=tk.BOTH, expand=True)

        def _calcLayersExpandedHeightPx():
            """Hauteur 'exacte' pour afficher tous les widgets du panneau Layers."""
            layer_frame.update_idletasks()
            hdr_h = int(layer_header.winfo_reqheight() or 26)
            # + séparateur (≈4) + padding/marges (≈18)
            content_h = int(self._ui_layers_content.winfo_reqheight() or 0)
            return int(hdr_h + content_h + 22)


        def _toggleLayersPanel():
            collapsed = bool(self._ui_layers_collapsed.get())
            self._ui_layers_collapsed.set(not collapsed)
            if self._ui_layers_collapsed.get():
                self._ui_layers_content.pack_forget()
                self._ui_layers_toggle_btn.config(text="▸")
            else:
                self._ui_layers_content.pack(fill=tk.BOTH, expand=True)
                self._ui_layers_toggle_btn.config(text="▾")

            layer_frame.update_idletasks()
            hdr_h = int(layer_header.winfo_reqheight() or 0)
            layer_minsize_collapsed = max(28, hdr_h + 10)
            if self._ui_layers_collapsed.get():
                pw.paneconfigure(layer_frame, minsize=layer_minsize_collapsed, height=layer_minsize_collapsed)
            else:
                target_h = _calcLayersExpandedHeightPx()
                if target_h is None:
                    pw.paneconfigure(layer_frame, minsize=layer_minsize_expanded)
                else:
                    target_h = max(int(layer_minsize_expanded), int(target_h))
                    pw.paneconfigure(layer_frame, minsize=layer_minsize_expanded, height=target_h)

        self._ui_layers_toggle_btn = tk.Button(
            layer_header,
            text=("▸" if self._ui_layers_collapsed.get() else "▾"),
            width=2,
            command=_toggleLayersPanel
        )
        self._ui_layers_toggle_btn.pack(side=tk.LEFT, padx=(0, 4))

        layer_title = tk.Label(layer_header, text="Layers", font=(None, 9, "bold"))
        layer_title.pack(side=tk.LEFT, anchor="w")
        layer_title.bind("<Button-1>", lambda _e: _toggleLayersPanel())
        layer_header.bind("<Button-1>", lambda _e: _toggleLayersPanel())

        # Checkboxes de visibilité
        cb_wrap = tk.Frame(self._ui_layers_content, bd=0, highlightthickness=0)
        cb_wrap.pack(anchor="w", fill="x", padx=6, pady=(0, 6))

        # Colonne "contrôles" à droite : même largeur pour Carte / Triangle / Compas
        rightColWidth = 140

        # Ligne "Carte" : checkbox + slider sur la même ligne, slider aligné à droite
        row_map = tk.Frame(cb_wrap)
        # même espacement vertical que les autres checkboxes
        row_map.pack(anchor="w", fill="x", pady=(2, 0))
        row_map.grid_columnconfigure(1, weight=1)

        tk.Checkbutton(
            row_map, text="Carte",
            variable=self.show_map_layer,
            command=self._toggle_layers,
        ).grid(row=0, column=0, sticky="w")

        # spacer pour pousser la colonne de droite au bord droit
        tk.Frame(row_map).grid(row=0, column=1, sticky="ew")

        row_map_right = tk.Frame(row_map, width=rightColWidth)
        row_map_right.grid(row=0, column=2, sticky="e")
        row_map_right.grid_propagate(False)

        self.mapOpacityScale = tk.Scale(
            row_map_right, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.map_opacity,
            showvalue=False,
            length=120,
            command=self._on_map_opacity_change,
        )
        self.mapOpacityScale.pack(side=tk.RIGHT, padx=(0, 4), anchor="e")

        # Ligne "Triangle" : checkbox + 2 radios (icônes) pour le mode d'affichage
        #  - value=0 : triangles + arêtes internes
        #  - value=1 : contour uniquement (sans arêtes internes)
        row_tri = tk.Frame(cb_wrap)
        row_tri.pack(anchor="w", fill="x", pady=(2, 0))
        row_tri.grid_columnconfigure(1, weight=1)

        tk.Checkbutton(
            row_tri,
            text="Triangle",
            variable=self.show_triangles_layer,
            command=self._toggle_layers,
        ).grid(row=0, column=0, sticky="w")

        # spacer pour pousser la colonne de droite au bord droit
        tk.Frame(row_tri).grid(row=0, column=1, sticky="ew")

        row_tri_right = tk.Frame(row_tri, width=rightColWidth)
        row_tri_right.grid(row=0, column=2, sticky="e")
        row_tri_right.grid_propagate(False)

        # charger les icônes depuis le répertoire (pas de génération online)
        if not hasattr(self, "iconTriModeEdges"):
            # noms de fichiers à créer/poser dans images_dir (on les fera ensemble ensuite)
            self.iconTriModeEdges = self._load_icon("tri_mode_edges.png")
            self.iconTriModeContour = self._load_icon("tri_mode_contour.png")

        if not hasattr(self, "_ui_triangleContourMode"):
            self._ui_triangleContourMode = tk.IntVar(
                value=(1 if bool(self.show_only_group_contours.get()) else 0)
            )
        else:
            # resynchroniser au cas où la valeur a changé depuis une autre action
            self._ui_triangleContourMode.set(1 if bool(self.show_only_group_contours.get()) else 0)


        def _onTriangleContourModeChange():
            only = bool(self._ui_triangleContourMode.get() == 1)
            if bool(self.show_only_group_contours.get()) != only:
                self.show_only_group_contours.set(only)
                self._toggle_only_group_contours()
            else:
                # forcer un redraw (utile si on a juste re-cliqué)
                self._redraw()

        # Radios à droite (icône-only). Fallback texte si l'icône n'est pas dispo.
        rb_kwargs = dict(
            variable=self._ui_triangleContourMode,
            indicatoron=0,
            padx=0,
            pady=0,
            command=_onTriangleContourModeChange,
        )
        if getattr(self, "iconTriModeContour", None) is not None:
            tk.Radiobutton(
                row_tri_right,
                image=self.iconTriModeContour,
                value=1,
                **rb_kwargs,
            ).pack(side=tk.RIGHT, padx=(2, 0))
        else:
            tk.Radiobutton(
                row_tri_right,
                text="Contour",
                value=1,
                **rb_kwargs,
            ).pack(side=tk.RIGHT, padx=(2, 0))

        if getattr(self, "iconTriModeEdges", None) is not None:
            tk.Radiobutton(
                row_tri_right,
                image=self.iconTriModeEdges,
                value=0,
                **rb_kwargs,
            ).pack(side=tk.RIGHT)
        else:
            tk.Radiobutton(
                row_tri_right,
                text="Arêtes",
                value=0,
                **rb_kwargs,
            ).pack(side=tk.RIGHT)

        # Ligne "Compas" : checkbox + boutons de taille (< >)
        row_clock = tk.Frame(cb_wrap)
        row_clock.pack(anchor="w", fill="x")
        row_clock.grid_columnconfigure(1, weight=1)

        tk.Checkbutton(
            row_clock, text="Compas",
            variable=self.show_clock_overlay,
            command=self._toggle_clock_overlay
        ).grid(row=0, column=0, sticky="w")

        # spacer pour pousser la colonne de droite au bord droit
        tk.Frame(row_clock).grid(row=0, column=1, sticky="ew")

        row_clock_right = tk.Frame(row_clock, width=rightColWidth)
        row_clock_right.grid(row=0, column=2, sticky="e")
        row_clock_right.grid_propagate(False)

        # Boutons taille compas : < réduit, > agrandit (pas de 5)
        tk.Button(
            row_clock_right, text=">", width=2,
            command=lambda: self._clock_change_radius(+5)
        ).pack(side=tk.RIGHT, padx=(0, 2))
        tk.Button(
            row_clock_right, text="<", width=2,
            command=lambda: self._clock_change_radius(-5)
        ).pack(side=tk.RIGHT, padx=(4, 0))

        pw.add(layer_frame, minsize=layer_minsize_expanded)

        # Si on démarre "déplié", on force une hauteur qui montre tous les widgets du panneau.
        if not bool(self._ui_layers_collapsed.get()):
            layer_frame.update_idletasks()
            h0 = _calcLayersExpandedHeightPx()
            if h0 is not None:
                h0 = max(int(layer_minsize_expanded), int(h0))
                pw.paneconfigure(layer_frame, height=h0)

        # --- Panneau : Décryptage (paramètres rapides) ---
        # Même principe que Triangles/Layers : header pliable + resize réel de la pane
        decrypt_minsize_expanded = 90

        if not hasattr(self, "_ui_decrypt_collapsed"):
            self._ui_decrypt_collapsed = tk.BooleanVar(value=False)

        decrypt_frame = tk.Frame(pw, bd=0, highlightthickness=0)

        decrypt_header = tk.Frame(decrypt_frame)
        decrypt_header.pack(fill=tk.X, pady=(0, 2))

        # Séparateur visuel sous le header
        ttk.Separator(decrypt_frame, orient="horizontal").pack(fill=tk.X, pady=(0, 4))

        self._ui_decrypt_content = tk.Frame(decrypt_frame)
        if not self._ui_decrypt_collapsed.get():
            self._ui_decrypt_content.pack(fill=tk.BOTH, expand=True)

        def _calcDecryptExpandedHeightPx():
            """Hauteur 'exacte' pour afficher tous les widgets du panneau Décryptage."""
            decrypt_frame.update_idletasks()
            hdr_h = int(decrypt_header.winfo_reqheight() or 26)
            content_h = int(self._ui_decrypt_content.winfo_reqheight() or 0)
            return int(hdr_h + content_h + 22)

        def _applyDecryptorFromUI():
            """Applique les paramètres UI au decryptor + redraw overlay."""
            idx = int(self._ui_decrypt_combo.current())
            decryptorId = list(DECRYPTORS.keys())[idx]

            # Algo déjà existant → on garde l’instance si possible
            if getattr(self.decryptor, "id", None) != decryptorId:
                self.decryptor = createDecryptor(decryptorId)
            self.decryptor.hourMovesWithMinutes = bool(self._ui_decrypt_hourMoveVar.get())

            # Bases minutes/heures (60/100 et 12/10)
            mb = int(self._ui_decrypt_minutesBaseVar.get())
            if hasattr(self.decryptor, "setMinutesBase"):
                self.decryptor.setMinutesBase(mb)
            else:
                self.decryptor.minutesBase = mb

            hb = int(self._ui_decrypt_hoursBaseVar.get())
            if hasattr(self.decryptor, "setHoursBase"):
                self.decryptor.setHoursBase(hb)
            else:
                self.decryptor.hoursBase = hb

            # On ne redessine que l'overlay (horloge)
            self._redraw_overlay_only()

        def _toggleDecryptPanel():
            collapsed = bool(self._ui_decrypt_collapsed.get())
            self._ui_decrypt_collapsed.set(not collapsed)
            if self._ui_decrypt_collapsed.get():
                self._ui_decrypt_content.pack_forget()
                self._ui_decrypt_toggle_btn.config(text="▸")
            else:
                self._ui_decrypt_content.pack(fill=tk.BOTH, expand=True)
                self._ui_decrypt_toggle_btn.config(text="▾")

            decrypt_frame.update_idletasks()
            hdr_h = int(decrypt_header.winfo_reqheight() or 0)
            decrypt_minsize_collapsed = max(28, hdr_h + 10)
            if self._ui_decrypt_collapsed.get():
                pw.paneconfigure(decrypt_frame, minsize=decrypt_minsize_collapsed, height=decrypt_minsize_collapsed)
            else:
                target_h = _calcDecryptExpandedHeightPx()
                target_h = max(int(decrypt_minsize_expanded), int(target_h))
                pw.paneconfigure(decrypt_frame, minsize=decrypt_minsize_expanded, height=target_h)

        self._ui_decrypt_toggle_btn = tk.Button(
            decrypt_header,
            text=("▸" if self._ui_decrypt_collapsed.get() else "▾"),
            width=2,
            command=_toggleDecryptPanel
        )
        self._ui_decrypt_toggle_btn.pack(side=tk.LEFT, padx=(0, 4))

        decrypt_title = tk.Label(decrypt_header, text="Décryptage", font=(None, 9, "bold"))
        decrypt_title.pack(side=tk.LEFT, anchor="w")
        decrypt_title.bind("<Button-1>", lambda _e: _toggleDecryptPanel())
        decrypt_header.bind("<Button-1>", lambda _e: _toggleDecryptPanel())

        # --- Widgets de décryptage ---
        decrypt_wrap = tk.Frame(self._ui_decrypt_content, bd=0, highlightthickness=0)
        decrypt_wrap.pack(fill="x", padx=6, pady=(0, 6))

        ttk.Label(decrypt_wrap, text="Algorithme").pack(anchor="w")

        values = [f"{d.id} — {d.label}" for d in DECRYPTORS.values()]
        self._ui_decrypt_combo = ttk.Combobox(decrypt_wrap, values=values, state="readonly")
        self._ui_decrypt_combo.pack(fill="x", pady=(0, 6))

        # Sélection initiale (courant)
        cur_id = getattr(self.decryptor, "id", None)
        for i, d in enumerate(DECRYPTORS.values()):
            if d.id == cur_id:
                self._ui_decrypt_combo.current(i)
                break
        else:
            self._ui_decrypt_combo.current(0)

        self._ui_decrypt_hourMoveVar = tk.BooleanVar(value=getattr(self.decryptor, "hourMovesWithMinutes", True))
        ttk.Checkbutton(
            decrypt_wrap,
            text="L’aiguille Heure avance avec les minutes",
            variable=self._ui_decrypt_hourMoveVar,
            command=_applyDecryptorFromUI
        ).pack(anchor="w", pady=(2, 0))

        # --- Bases du cadran (minutes/heures) ---
        # Valeurs initiales depuis le decryptor (fallback 60/12)
        cur_min_base = 60
        cur_hour_base = 12
        cur_min_base = int(getattr(self.decryptor, "getMinutesBase", lambda: getattr(self.decryptor, "minutesBase", 60))())
        cur_hour_base = int(getattr(self.decryptor, "getHoursBase", lambda: getattr(self.decryptor, "hoursBase", 12))())

        self._ui_decrypt_minutesBaseVar = tk.IntVar(value=cur_min_base if cur_min_base in (60, 100) else 60)
        self._ui_decrypt_hoursBaseVar = tk.IntVar(value=cur_hour_base if cur_hour_base in (12, 10) else 12)

        bases_box = tk.Frame(decrypt_wrap, bd=0, highlightthickness=0)
        bases_box.pack(anchor="w", fill="x", pady=(6, 0))

        # Minutes
        row_m = tk.Frame(bases_box, bd=0, highlightthickness=0)
        row_m.pack(anchor="w", fill="x")
        ttk.Label(row_m, text="Minutes").pack(side=tk.LEFT)
        ttk.Radiobutton(
            row_m, text="60", value=60,
            variable=self._ui_decrypt_minutesBaseVar,
            command=_applyDecryptorFromUI
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(
            row_m, text="100", value=100,
            variable=self._ui_decrypt_minutesBaseVar,
            command=_applyDecryptorFromUI
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Heures
        row_h = tk.Frame(bases_box, bd=0, highlightthickness=0)
        row_h.pack(anchor="w", fill="x", pady=(2, 0))
        ttk.Label(row_h, text="Heures").pack(side=tk.LEFT)
        ttk.Radiobutton(
            row_h, text="12", value=12,
            variable=self._ui_decrypt_hoursBaseVar,
            command=_applyDecryptorFromUI
        ).pack(side=tk.LEFT, padx=(18, 0))
        ttk.Radiobutton(
            row_h, text="10", value=10,
            variable=self._ui_decrypt_hoursBaseVar,
            command=_applyDecryptorFromUI
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Appliquer quand l'algo change
        try:
            self._ui_decrypt_combo.bind("<<ComboboxSelected>>", lambda _e: _applyDecryptorFromUI())
        except Exception:
            pass

        pw.add(decrypt_frame, minsize=decrypt_minsize_expanded)

        # Si on démarre "déplié", on force une hauteur qui montre tous les widgets du panneau.
        if not bool(self._ui_decrypt_collapsed.get()):
            decrypt_frame.update_idletasks()
            h0 = _calcDecryptExpandedHeightPx()
            if h0 is not None:
                h0 = max(int(decrypt_minsize_expanded), int(h0))
                pw.paneconfigure(decrypt_frame, height=h0)


        # --- Panneau bas : scénarios + barre d'icônes ---
        # Même approche que Triangles / Layers : panneau pliable (sans encadrement) + resize réel de la pane.
        scen_minsize_expanded = 120

        if not hasattr(self, "_ui_scenarios_collapsed"):
            self._ui_scenarios_collapsed = tk.BooleanVar(value=False)

        scen_frame = tk.Frame(pw, bd=0, highlightthickness=0)

        scen_header = tk.Frame(scen_frame)
        scen_header.pack(fill=tk.X, pady=(0, 2))

        # Séparateur visuel sous le header
        ttk.Separator(scen_frame, orient="horizontal").pack(fill=tk.X, pady=(0, 4))

        self._ui_scenarios_content = tk.Frame(scen_frame)
        if not self._ui_scenarios_collapsed.get():
            self._ui_scenarios_content.pack(fill=tk.BOTH, expand=True)

        def _calcScenariosExpandedHeightPx(fill_bottom=False):
            """Hauteur pour afficher le contenu (et optionnellement remplir jusqu'en bas)."""
            scen_frame.update_idletasks()
            hdr_h = int(scen_header.winfo_reqheight() or 26)
            content_h = int(self._ui_scenarios_content.winfo_reqheight() or 0)
            base_h = int(hdr_h + content_h + 22)  # + séparateur/paddings
            if fill_bottom:
                try:
                    avail = int(pw.winfo_height() or 0)
                except Exception:
                    avail = 0
                if avail > 0:
                    base_h = max(base_h, avail)
            return int(base_h)

        def _toggleScenariosPanel():
            collapsed = bool(self._ui_scenarios_collapsed.get())
            self._ui_scenarios_collapsed.set(not collapsed)
            if self._ui_scenarios_collapsed.get():
                self._ui_scenarios_content.pack_forget()
                self._ui_scenarios_toggle_btn.config(text="▸")
            else:
                self._ui_scenarios_content.pack(fill=tk.BOTH, expand=True)
                self._ui_scenarios_toggle_btn.config(text="▾")

            scen_frame.update_idletasks()
            hdr_h = int(scen_header.winfo_reqheight() or 0)
            scen_minsize_collapsed = max(28, hdr_h + 10)

            if self._ui_scenarios_collapsed.get():
                pw.paneconfigure(scen_frame, minsize=scen_minsize_collapsed, height=scen_minsize_collapsed)
            else:
                target_h = _calcScenariosExpandedHeightPx(fill_bottom=True)
                target_h = max(int(scen_minsize_expanded), int(target_h))
                pw.paneconfigure(scen_frame, minsize=scen_minsize_expanded, height=target_h)

        # Bouton toggle + titre cliquable
        self._ui_scenarios_toggle_btn = tk.Button(
            scen_header,
            text=("▸" if self._ui_scenarios_collapsed.get() else "▾"),
            width=2,
            command=_toggleScenariosPanel
        )
        self._ui_scenarios_toggle_btn.pack(side=tk.LEFT, padx=(0, 4))

        scen_title = tk.Label(scen_header, text="Scénarios", font=(None, 9, "bold"))
        scen_title.pack(side=tk.LEFT, anchor="w")
        scen_title.bind("<Button-1>", lambda _e: _toggleScenariosPanel())
        scen_header.bind("<Button-1>", lambda _e: _toggleScenariosPanel())

        # Barre d'icônes (Nouveau, Charger, Propriétés, Sauver, Dupliquer, Supprimer)
        toolbar = tk.Frame(self._ui_scenarios_content, bd=0, highlightthickness=0)
        toolbar.pack(anchor="w", padx=6, pady=(0, 2), fill="x")

        # Chargement des icônes (tu peux adapter les noms de fichiers PNG)
        self.icon_scen_new   = self._load_icon("scenario_new.png")
        self.icon_scen_open  = self._load_icon("scenario_open.png")
        self.icon_scen_props = self._load_icon("scenario_props.png")
        self.icon_scen_save  = self._load_icon("scenario_save.png")
        self.icon_scen_dup   = self._load_icon("scenario_duplicate.png")
        self.icon_scen_del   = self._load_icon("scenario_delete.png")

        # Icônes de type (affichées dans la Treeview)
        self.icon_scen_manual = self._load_icon("scenario_manual.png")
        self.icon_scen_auto   = self._load_icon("scenario_auto.png")

        def _make_btn(parent, icon, text, cmd, tooltip_text: str = ""):
            if icon is not None:
                b = tk.Button(parent, image=icon, command=cmd, relief=tk.FLAT)
            else:
                # fallback texte si l'icône n'est pas trouvée
                b = tk.Button(parent, text=text, command=cmd, width=2, relief=tk.FLAT)
            # tooltips UI
            self._ui_attach_tooltip(b, tooltip_text)
            return b

        _make_btn(toolbar, self.icon_scen_new,   "N", self._new_empty_scenario,
                  "Nouveau").pack(side=tk.LEFT, padx=1)
        _make_btn(toolbar, self.icon_scen_open,  "O", self._scenario_load_dialog,
                  "Charger...").pack(side=tk.LEFT, padx=1)
        _make_btn(toolbar, self.icon_scen_props, "P", self._scenario_edit_properties,
                  "Propriétés...").pack(side=tk.LEFT, padx=1)
        _make_btn(toolbar, self.icon_scen_save,  "S", self._scenario_save_dialog,
                  "Enregistrer...").pack(side=tk.LEFT, padx=1)
        _make_btn(toolbar, self.icon_scen_dup,   "D", self._scenario_duplicate,
                  "Dupliquer").pack(side=tk.LEFT, padx=1)
        _make_btn(toolbar, self.icon_scen_del,   "X", self._scenario_delete,
                  "Supprimer...").pack(side=tk.LEFT, padx=1)
 
        # --- Filtre des scénarios automatiques (par triangle "bascule") ---
        # Valeurs: "Tous" ou "(26)" etc. (uniquement les IDs présents dans les libellés "+(id)")
        self.scenario_filter_var = tk.StringVar(value="Tous")
        self.scenario_filter_combo = ttk.Combobox(
            toolbar,
            textvariable=self.scenario_filter_var,
            state="readonly",
            width=8,
            values=["Tous"],
        )
        self.scenario_filter_combo.pack(side=tk.RIGHT, padx=(6, 0))
        self.scenario_filter_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._refresh_scenario_listbox(),
        )

        scen_lb_frame = tk.Frame(self._ui_scenarios_content, bd=0, highlightthickness=0)
        scen_lb_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Liste des scénarios (Treeview) : groupes Manuels / Automatiques (colonnes plus tard)
        columns = ()

        # IMPORTANT:
        # - ttk ajoute une indentation (~20px) pour les items enfants => gros espace avant l'icône.
        # - en plus, les colonnes (algo/status/n) réduisent #0 => libellés tronqués.
        style = ttk.Style()
        style.configure("Scenario.Treeview", indent=0)  # <- colle les icônes à gauche

        self.scenario_tree = ttk.Treeview(
            scen_lb_frame,
            columns=columns,
            show="tree",
            selectmode="browse",
            height=6,
            style="Scenario.Treeview",
        )
        self.scenario_tree.column("#0", width=280, stretch=True)
        # Scrollbar verticale (toujours visible)
        # IMPORTANT: garder une référence (sinon GC => scrollbar détruite et elle "disparaît")
        self.scenario_scroll = ttk.Scrollbar(
            scen_lb_frame, orient="vertical", command=self.scenario_tree.yview
        )
        self.scenario_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.scenario_tree.configure(yscrollcommand=self.scenario_scroll.set)

        # Pack après la scrollbar pour réserver l'espace à droite
        self.scenario_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scenario_tree.bind("<<TreeviewSelect>>", self._on_scenario_select)
        self.scenario_tree.bind("<Double-1>", self._on_scenario_double_click)

        # Police "référence" (gras)
        base_font = tkfont.nametofont("TkDefaultFont")
        self._font_scenario_ref = base_font.copy()
        self._font_scenario_ref.configure(weight="bold")
        self.scenario_tree.tag_configure("ref", font=self._font_scenario_ref)

        pw.add(scen_frame, minsize=scen_minsize_expanded)  # hauteur mini pour la liste des scénarios

        # Les panneaux du haut ne doivent pas "aspirer" la hauteur quand la fenêtre grandit.
        # On laisse Scénarios prendre la place restante (remplissage jusqu'en bas).
        pw.paneconfigure(tri_frame, stretch="never")
        pw.paneconfigure(layer_frame, stretch="never")
        pw.paneconfigure(scen_frame, stretch="always")

        # Si on démarre "déplié", on force une hauteur qui remplit jusqu'en bas.
        if not bool(self._ui_scenarios_collapsed.get()):
            scen_frame.update_idletasks()
            h0 = _calcScenariosExpandedHeightPx(fill_bottom=True)
            if h0 is not None:
                h0 = max(int(scen_minsize_expanded), int(h0))
                pw.paneconfigure(scen_frame, height=h0)

        # Remplir la liste des scénarios existants (pour l'instant : le manuel)
        self._refresh_scenario_listbox()

    def _refresh_scenario_listbox(self):
        """Met à jour la liste visible des scénarios (Treeview) dans le panneau de gauche."""
        if not hasattr(self, "scenario_tree"):
            return
        tree = self.scenario_tree

 
        def _parseScenarioDisplayId(name: str):
            """Extrait l'ID affiché (#n) depuis un libellé de scénario."""
            s = str(name or "").strip()
            if s.startswith("★"):
                s = s.lstrip("★ ").strip()
            m = re.match(r"^#(\d+)", s)
            return int(m.group(1)) if m else None

 
        def _parseScenarioRefId(name: str):
            """Extrait la référence (#q) depuis un libellé du type '#n=#q+(tri)'."""
            s = str(name or "")
            m = re.search(r"=#(\d+)\+\(", s)
            return int(m.group(1)) if m else None

 
        def _parseScenarioBranchTriId(name: str):
            """Extrait le triangle '(id)' depuis un libellé du type '#n=#q+(id)'."""
            s = str(name or "")
            m = re.search(r"\+\((\d+)\)", s)
            return int(m.group(1)) if m else None

        # --- Filtre sélectionné ("Tous" ou "(26)") ---
        selected_tri = None
        if hasattr(self, "scenario_filter_var"):
            v = str(self.scenario_filter_var.get() or "").strip()
            if v and v != "Tous":
                m = re.search(r"(\d+)", v)
                if m:
                    selected_tri = int(m.group(1))

 
         # --- Recalcul des valeurs possibles de la combo (sur l'ensemble des scénarios auto) ---
        tri_values = set()
        for _sc in (self.scenarios or []):
            if getattr(_sc, "source_type", None) != "auto":
                continue
            tid = _parseScenarioBranchTriId(getattr(_sc, "name", ""))
            if tid is not None:
                tri_values.add(int(tid))
        tri_values_sorted = sorted(tri_values)
        combo_values = ["Tous"] + [f"({t})" for t in tri_values_sorted]

        # Mettre à jour la combo sans casser la sélection courante
        if hasattr(self, "scenario_filter_combo"):
            cur = str(self.scenario_filter_var.get() or "Tous")
            self.scenario_filter_combo["values"] = combo_values
            if cur not in combo_values:
                self.scenario_filter_var.set("Tous")
                selected_tri = None
 
        # --- Construire l'ensemble des scénarios auto à afficher (filtre) ---
        visible_auto_display_ids = None  # None => pas de filtre
        if selected_tri is not None:
            # 1) scénarios '#n=#q+(selected_tri)'
            matched_display_ids = set()
            matched_ref_display_ids = set()
            for _sc in (self.scenarios or []):
                if getattr(_sc, "source_type", None) != "auto":
                    continue
                name = getattr(_sc, "name", "")
                tid = _parseScenarioBranchTriId(name)
                if tid is None or int(tid) != int(selected_tri):
                    continue
                did = _parseScenarioDisplayId(name)
                if did is not None:
                    matched_display_ids.add(int(did))
                rid = _parseScenarioRefId(name)
                if rid is not None:
                    matched_ref_display_ids.add(int(rid))
 
            # 2) + les scénarios '#q' associés (références)
            visible_auto_display_ids = matched_display_ids.union(matched_ref_display_ids)

        # Nettoyage
        for ch in tree.get_children(""):
            tree.delete(ch)

        # Groupes
        manual_count = 0
        auto_count = 0
        grp_manual = tree.insert("", tk.END, iid="grp_manual", text="Manuels", open=True)
        grp_auto   = tree.insert("", tk.END, iid="grp_auto",   text="Automatiques", open=True)

        for i, scen in enumerate(self.scenarios):
            iid = f"scen_{i}"
            parent = grp_manual if scen.source_type == "manual" else grp_auto
            if scen.source_type == "manual":
                manual_count += 1
            else:
                # Filtrage des scénarios auto
                if visible_auto_display_ids is not None:
                    did = _parseScenarioDisplayId(getattr(scen, "name", ""))
                    if did is None or int(did) not in visible_auto_display_ids:
                        continue
                auto_count += 1

            tags = []
            is_ref = (
                scen.source_type == "auto" and
                self.ref_scenario_token is not None and
                id(scen) == self.ref_scenario_token
            )
            if is_ref:
                tags.append("ref")

            # Texte + icône
            text = scen.name
            if is_ref:
                text = "★ " + text
            img = self.icon_scen_manual if scen.source_type == "manual" else self.icon_scen_auto


            kwargs = {"text": text, "tags": tags}
            if img is not None:
                kwargs["image"] = img
            tree.insert(parent, tk.END, iid=iid, **kwargs)

        # Mettre à jour les titres des groupes avec compteur
        tree.item(grp_manual, text=f"Manuels ({manual_count})")
        tree.item(grp_auto,   text=f"Automatiques ({auto_count})")


        # Sélectionner le scénario actif si possible
        if 0 <= self.active_scenario_index < len(self.scenarios):
            active_iid = f"scen_{self.active_scenario_index}"
            # Défensif : lors d'un chargement/import, l'Excel peut être rechargé et déclencher
            # des refresh intermédiaires ; dans ce cas l'item n'est pas toujours encore présent.
            if hasattr(tree, "exists") and tree.exists(active_iid):
                tree.selection_set(active_iid)
                tree.see(active_iid)
            else:
                # fallback : sélectionner le 1er scénario (s'il existe)
                for parent in ("grp_manual", "grp_auto", ""):
                    try:
                        kids = tree.get_children(parent)
                    except Exception:
                        kids = []
                    if kids:
                        tree.selection_set(kids[0])
                        tree.see(kids[0])
                        break



    def _update_triangle_listbox_colors(self):
        """
        Met à jour la couleur des entrées de la listbox des triangles
        en fonction de leur utilisation dans le scénario actif.
        Triangles utilisés → grisés, triangles disponibles → noir.
        """
        if not hasattr(self, "listbox"):
            return
        lb = self.listbox
        try:
            size = lb.size()
        except Exception:
            return

        for idx in range(size):
            try:
                txt = lb.get(idx)
            except Exception:
                continue

            tri_id = None
            try:
                m = re.match(r"\s*(\d+)\.", str(txt))
                if m:
                    tri_id = int(m.group(1))
            except Exception:
                tri_id = None

            if tri_id is not None and tri_id in self._placed_ids:
                # Triangle déjà posé dans le scénario → grisé
                try:
                    lb.itemconfig(idx, fg="gray50")
                except Exception:
                    pass
            else:
                # Triangle disponible
                try:
                    lb.itemconfig(idx, fg="black")
                except Exception:
                    pass

    def _on_scenario_select(self, event=None):
        """Callback quand l'utilisateur sélectionne un scénario (target) dans la Treeview."""
        if not hasattr(self, "scenario_tree"):
            return
        tree = self.scenario_tree
        sel = tree.selection()
        if not sel:
            return
        iid = str(sel[0])
        if not iid.startswith("scen_"):
            # clic sur un groupe (Manuels/Automatiques)
            return
        try:
            idx = int(iid.split("_", 1)[1])
        except Exception:
            return
        self._set_active_scenario(idx)

    def _on_scenario_double_click(self, event=None):
        """Double-clic sur un scénario auto : le définit comme scénario de référence (gras/★)."""
        if not hasattr(self, "scenario_tree"):
            return

        iid = self.scenario_tree.identify_row(event.y)

        if not iid or not str(iid).startswith("scen_"):
            return
        idx = int(str(iid).split("_", 1)[1])
        if idx < 0 or idx >= len(self.scenarios):
            return
        scen = self.scenarios[idx]
        if getattr(scen, "source_type", None) != "auto":
            # On ne marque en référence que les scénarios automatiques
            return

        # Toggle: si on double-clique la référence actuelle => on désactive la comparaison
        if self.ref_scenario_token is not None and self.ref_scenario_token == id(scen):
            self.ref_scenario_token = None
            self._comparison_diff_indices = set()
            self._refresh_scenario_listbox()
            self.status.config(text="Mode comparaison désactivé (référence retirée).")
            self._redraw_from(self._last_drawn)
            return

        # Sinon: définir comme référence
        self.ref_scenario_token = id(scen)
        self._comparison_diff_indices = set()
        self._refresh_scenario_listbox()
        self.status.config(text=f"Référence auto : {scen.name}")
        self._redraw_from(self._last_drawn)
 
    def _get_reference_scenario(self) -> Optional[ScenarioAssemblage]:
        token = self.ref_scenario_token
        if token is None:
            return None
        for scen in self.scenarios:
            if id(scen) == token:
                return scen
        return None

    def _scenario_connections_signature(self, scen):
        """
        Signature de connexions basée sur les groups:
        triId -> { edgeName -> (neighborTriId, neighborEdgeName) }

        - triId = triangle["id"] (stable)
        - edgeName = "OB" / "BL" / "LO"
        - neighborEdgeName = arête correspondante côté voisin

        On déduit l'arête depuis vkey (sommet) via l'arête opposée:
        IMPORTANT:
        Dans groups[n]["nodes"], vkey_in/out doit représenter le **sommet opposé**
        à l'arête partagée (et pas le sommet de contact).
 
        On déduit alors l'arête partagée depuis vkey via l'arête opposée:        O -> BL
        B -> LO
        L -> OB
        """
        sig = {}
        if scen is None:
            return sig

        last_drawn = getattr(scen, "last_drawn", None)
        groups = getattr(scen, "groups", None)

        if not last_drawn or not groups:
            return sig

        # vkey = sommet ("O","B","L") -> arête opposée
        opp_edge = {"O": "BL", "B": "LO", "L": "OB"}

        def tri_id_from_tid(tid):
            """tid = index dans last_drawn ; retourne triangle["id"]"""
            try:
                tid = int(tid)
                if 0 <= tid < len(last_drawn):
                    return int(last_drawn[tid].get("id"))
            except Exception:
                pass
            return None

        def add_link(tri_a, edge_a, tri_b, edge_b):
            if tri_a is None or tri_b is None or edge_a is None or edge_b is None:
                return
            sig.setdefault(tri_a, {})[edge_a] = (tri_b, edge_b)

        # Parcours de tous les groupes (un groupe = une chaîne de nodes)
        for g in groups.values():
            nodes = (g or {}).get("nodes") or []
            if len(nodes) < 2:
                continue

            # Connexions = paires consécutives dans la chaîne
            for i in range(len(nodes) - 1):
                left = nodes[i]
                right = nodes[i + 1]
                if not left or not right:
                    continue

                tri_left = tri_id_from_tid(left.get("tid"))
                tri_right = tri_id_from_tid(right.get("tid"))

                # 1) priorité à l'arête explicitement stockée (LB/LO/BL/OB...)
                edge_left = left.get("edge_out")
                edge_right = right.get("edge_in")
                # 2) fallback ancien comportement (via vkey opposé)
                if not edge_left or not edge_right:
                    v_out = left.get("vkey_out")
                    v_in = right.get("vkey_in")
                    edge_left = opp_edge.get(v_out) if v_out else None
                    edge_right = opp_edge.get(v_in) if v_in else None

                add_link(tri_left, edge_left, tri_right, edge_right)
                add_link(tri_right, edge_right, tri_left, edge_left)

        return sig


    def _update_current_scenario_differences(self):
        """Met à jour self._comparison_diff_indices en comparant le scénario actif à la référence."""
        self._comparison_diff_indices = set()

        ref_scen = self._get_reference_scenario()
        if ref_scen is None:
            return

        if not (0 <= self.active_scenario_index < len(self.scenarios)):
            return

        cur_scen = self.scenarios[self.active_scenario_index]
        if cur_scen is ref_scen:
            return

        ref_sig = self._scenario_connections_signature(ref_scen)
        cur_sig = self._scenario_connections_signature(cur_scen)

        # map triId -> idx dans last_drawn (pour pouvoir marquer les voisins)
        tri_id_to_idx = {}
        for idx, tri in enumerate(cur_scen.last_drawn):
            try:
                tri_id_to_idx[int(tri.get("id"))] = idx
            except Exception:
                pass

        def normalize_edge_map(m):
            """
            Normalise une map d'arêtes pour rendre la comparaison stable.
            Supporte:
            - edge -> neighborId
            - edge -> (neighborId, neighborEdge)
            """
            if not isinstance(m, dict):
                return {}
            out = {}
            for k, v in m.items():
                kk = str(k)
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    try:
                        out[kk] = (int(v[0]), str(v[1]))
                    except Exception:
                        out[kk] = (v[0], str(v[1]))
                else:
                    try:
                        out[kk] = int(v)
                    except Exception:
                        out[kk] = v
            return out

        # 1) Détecter les triangles du courant qui diffèrent
        for idx, tri in enumerate(cur_scen.last_drawn):
            try:
                tri_id = int(tri.get("id"))
            except Exception:
                self._comparison_diff_indices.add(idx)
                continue

            ref_edges = normalize_edge_map(ref_sig.get(tri_id, {}))
            cur_edges = normalize_edge_map(cur_sig.get(tri_id, {}))

            if ref_edges != cur_edges:
                # Ne marquer que les arêtes qui changent réellement
                changed_edges = set(ref_edges.keys()) | set(cur_edges.keys())
                changed_edges = {e for e in changed_edges if ref_edges.get(e) != cur_edges.get(e)}

                # Marquer ce triangle
                self._comparison_diff_indices.add(idx)

                # Marquer uniquement les voisins impliqués dans les arêtes modifiées
                for e in changed_edges:
                    for m in (cur_edges, ref_edges):
                        link = m.get(e)
                        if link is None:
                            continue
                        neigh_id = link[0] if isinstance(link, tuple) and len(link) == 2 else link
                        neigh_idx = tri_id_to_idx.get(neigh_id)
                        if neigh_idx is not None:
                            self._comparison_diff_indices.add(neigh_idx)

    def _set_active_scenario(self, index: int):
        """
        Bascule vers le scénario d'index donné.
        Pour l'instant, on n'a qu'un scénario manuel qui partage les mêmes
        structures _last_drawn / groups, mais cette méthode sera utilisée
        plus tard pour les scénarios automatiques (copies séparées).
        """
        if index < 0 or index >= len(self.scenarios):
            return
        if index == self.active_scenario_index:
            return

        scen = self.scenarios[index]
        self.active_scenario_index = index

        # Rattacher les structures géométriques du scénario courant
        self._last_drawn = scen.last_drawn
        self.groups = scen.groups

        # --- AUTO: réconcilier les métadonnées de groupe (group_id/group_pos) avec scen.groups ---
        # Certains chemins (redraw / sélection / tools) se basent encore sur les champs portés par les triangles.
        # Si un scénario auto fournit un dictionnaire groups cohérent mais que les triangles n'ont pas été annotés,
        # on peut se retrouver avec un triangle "orphelin" ou des groupes recomposés de travers.
        try:
            if getattr(scen, "source_type", "manual") == "auto" and isinstance(scen.groups, dict) and scen.last_drawn:
                # 1) reset défensif
                for _t in scen.last_drawn:
                    _t["group_id"] = None
                    _t["group_pos"] = None

                # 2) appliquer groups -> triangles
                for _gid, _g in scen.groups.items():
                    _nodes = (_g or {}).get("nodes", []) or []
                    for _pos, _nd in enumerate(_nodes):
                        _tid = _nd.get("tid")
                        if _tid is None:
                            continue
                        try:
                            _tid_i = int(_tid)
                        except Exception:
                            continue
                        if 0 <= _tid_i < len(scen.last_drawn):
                            scen.last_drawn[_tid_i]["group_id"] = _gid
                            scen.last_drawn[_tid_i]["group_pos"] = int(_pos)

                # 3) fallback : si un triangle n'est dans aucun node, on le rattache au 1er groupe
                _first_gid = next(iter(scen.groups.keys()), None)
                if _first_gid is not None:
                    for _t in scen.last_drawn:
                        if _t.get("group_id") is None:
                            _t["group_id"] = _first_gid
                            _t["group_pos"] = 0

                # 4) bbox
                for _gid in list(scen.groups.keys()):
                    try:
                        self._recompute_group_bbox(_gid)
                    except Exception:
                        pass
        except Exception:
            pass

        # Recalibrer _next_group_id si besoin
        try:
            self._next_group_id = (max(self.groups.keys()) + 1) if self.groups else 1
        except Exception:
            self._next_group_id = 1

        # Recalcule la liste des triangles déjà utilisés pour ce scénario
        try:
            self._placed_ids = {
                int(t["id"]) for t in self._last_drawn
                if t.get("id") is not None
            }
        except Exception:
            self._placed_ids = set()
        self._update_triangle_listbox_colors()

        # Invalider le cache de pick et redessiner
        self._invalidate_pick_cache()

        # Fit à l'écran optionnel lors de la sélection d'un scénario
        if self._last_drawn and bool(self.auto_fit_scenario_select.get()):
            # _fit_to_view redessine déjà via _redraw_from()
            self._fit_to_view(self._last_drawn)
        else:
            self._redraw_from(self._last_drawn)

        self._redraw_overlay_only()

        # Mettre à jour la sélection visuelle dans la liste (au cas d'appel programmatique)
        if hasattr(self, "scenario_tree"):
            active_iid = f"scen_{self.active_scenario_index}"
            tree = self.scenario_tree
            # Défensif : _set_active_scenario() peut être appelé avant que la Treeview
            # n'ait été rafraîchie (ex: import d'un scénario => append dans self.scenarios
            # puis activation avant _refresh_scenario_listbox()). Dans ce cas, l'item
            # n'existe pas encore et Tk lève "Item scen_X not found".
            if hasattr(tree, "exists") and tree.exists(active_iid):
                tree.selection_set(active_iid)
                tree.see(active_iid)


        self.status.config(text=f"Scénario actif : {scen.name}")


    def _new_empty_scenario(self):
        """
        Crée un nouveau scénario *vide* (sans triangles assemblés),
        l'ajoute à la liste et le rend actif.
        Les triangles sources (dans la listbox) restent évidemment disponibles.
        """
        # Nom par défaut : "Scénario N" (N = nombre total de scénarios après ajout)
        new_index = len(self.scenarios)  # l'index qu'il prendra une fois append
        name = f"Scénario {new_index + 1}"

        scen = ScenarioAssemblage(
            name=name,
            source_type="manual",
            algo_id=None,
            tri_ids=[],
        )
        # Scénario vide : nouvelles structures indépendantes
        scen.last_drawn = []
        scen.groups = {}

        self.scenarios.append(scen)
        # Bascule sur ce nouveau scénario
        self._set_active_scenario(new_index)
        # Rafraîchir la liste visible
        self._refresh_scenario_listbox()

        self.status.config(text=f"Nouveau scénario créé : {scen.name}")


    def _scenario_edit_properties(self):
        """
        Modifie les propriétés du scénario actif (pour l'instant : uniquement le nom).
        """
        if not self.scenarios:
            return
        idx = self.active_scenario_index
        if idx < 0 or idx >= len(self.scenarios):
            return
        scen = self.scenarios[idx]
        new_name = simpledialog.askstring(
            "Propriétés du scénario",
            "Nom du scénario :",
            initialvalue=scen.name,
            parent=self,
        )
        if new_name is None:
            return  # annulé
        new_name = new_name.strip()
        if not new_name:
            return
        scen.name = new_name
        self._refresh_scenario_listbox()
        try:
            self.status.config(text=f"Nom du scénario mis à jour : {scen.name}")
        except Exception:
            pass

    def _scenario_duplicate(self):
        """
        Duplique le scénario actif dans un nouveau scénario indépendant.
        Les triangles et groupes sont copiés (deepcopy).
        """
        if not self.scenarios:
            return
        idx = self.active_scenario_index
        if idx < 0 or idx >= len(self.scenarios):
            return
        src = self.scenarios[idx]

        base_name = src.name or "Scénario"
        new_name = f"{base_name} (copie)"
        n = 2
        # garantir un nom unique
        while any(s.name == new_name for s in self.scenarios):
            new_name = f"{base_name} (copie {n})"
            n += 1

        new_index = len(self.scenarios)
        dup = ScenarioAssemblage(
            name=new_name,
            source_type=src.source_type,
            algo_id=src.algo_id,
            tri_ids=list(src.tri_ids),
        )
        # copies indépendantes
        dup.last_drawn = copy.deepcopy(src.last_drawn)
        dup.groups = copy.deepcopy(src.groups)
        dup.status = src.status

        self.scenarios.append(dup)
        self._refresh_scenario_listbox()
        self._set_active_scenario(new_index)
        try:
            self.status.config(text=f"Scénario dupliqué : {dup.name}")
        except Exception:
            pass

    def _scenario_delete(self):
        """
        Supprime le scénario actif.
        On interdit la suppression du scénario manuel de base pour garder un point d'appui.
        """
        if len(self.scenarios) <= 1:
            messagebox.showinfo("Supprimer le scénario",
                                "Impossible de supprimer le dernier scénario.")
            return
        idx = self.active_scenario_index
        if idx < 0 or idx >= len(self.scenarios):
            return

        scen = self.scenarios[idx]

        # Si on supprime le scénario de référence, on efface la référence
        try:
            if self.ref_scenario_token is not None and id(scen) == self.ref_scenario_token:
                self.ref_scenario_token = None
        except Exception:
            pass

        if idx == 0 and scen.source_type == "manual":
            messagebox.showinfo("Supprimer le scénario",
                                "Le scénario manuel ne peut pas être supprimé.")
            return

        if not messagebox.askyesno(
            "Supprimer le scénario",
            f"Supprimer le scénario « {scen.name} » ?",
            parent=self,
        ):
            return

        # Retirer le scénario
        self.scenarios.pop(idx)
        # Choisir le nouveau scénario actif (celui d'avant si possible)
        if idx >= len(self.scenarios):
            idx = len(self.scenarios) - 1

        self._set_active_scenario(idx)
        self._refresh_scenario_listbox()
        try:
            self.status.config(text=f"Scénario supprimé : {scen.name}")
        except Exception:
            pass

    def _build_canvas(self, parent):
        # Conteneur de droite : canvas (haut, expansible) + panel dico (bas, hauteur fixe)
        self.rightPane = tk.Frame(parent)
        self.rightPane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas d’affichage des triangles
        self.canvas = tk.Canvas(self.rightPane, bg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Redessiner l'overlay si la taille du canvas change
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Panel dico (placeholder à hauteur fixe, prêt pour intégrer la grille)
        self.dicoPanel = tk.Frame(self.rightPane, height=self.dico_panel_height,
                                  bd=1, relief=tk.SUNKEN, bg="#f3f3f3")
        # empêcher le panel de rétrécir sur le contenu
        self.dicoPanel.pack_propagate(False)
        # Pack initial seulement si le toggle est actif
        if self.show_dico_panel.get():
            self.dicoPanel.pack(side=tk.BOTTOM, fill=tk.X)
            # Placeholder visuel (sera remplacé par _build_dico_grid)
            tk.Label(self.dicoPanel, text="Dictionnaire — (grille à intégrer)",
                     bg="#f3f3f3", anchor="w").pack(fill=tk.X, padx=8, pady=4)

        # Menu contextuel COMPAS (clic droit sur le compas)
        self._ctx_menu_compass = tk.Menu(self, tearoff=0)
        self._ctx_menu_compass.add_command(label="Définir l'azimut de ref…", command=self._ctx_define_clock_ref_azimuth)
        self._ctx_menu_compass.add_command(label="Mesurer un azimut…", command=self._ctx_measure_clock_azimuth)
        self._ctx_menu_compass.add_command(label="Mesurer un arc d'angle…", command=self._ctx_measure_clock_arc_angle)
        self._ctx_menu_compass.add_separator()
        self._ctx_menu_compass.add_command(label="Filtrer le dictionnaire…", command=self._ctx_filter_dictionary_by_clock_arc, state=tk.DISABLED)
        self._ctx_compass_idx_filter_dico = self._ctx_menu_compass.index("end")
        self._ctx_menu_compass.add_command(label="Annuler le filtrage", command=self._simulation_cancel_dictionary_filter, state=tk.DISABLED)
        self._ctx_compass_idx_cancel_dico_filter = self._ctx_menu_compass.index("end")
        self._update_compass_ctx_menu_and_dico_state()

        # Menu contextuel
        self._ctx_menu = tk.Menu(self, tearoff=0)
        self._ctx_menu.add_command(label="Supprimer", command=self._ctx_delete_group)
        self._ctx_menu.add_command(label="Pivoter", command=self._ctx_rotate_selected)
        self._ctx_menu.add_command(label="Inverser", command=self._ctx_flip_selected)
        self._ctx_menu.add_command(label="Filtrer les scénarios…", command=self._ctx_filter_scenarios)
        self._ctx_menu.add_command(label="OL=0°", command=self._ctx_orient_OL_north)
        self._ctx_menu.add_command(label="BL=0°", command=self._ctx_orient_BL_north)

        # Mémoriser l'index des entrées "OL=0°" / "BL=0°" pour pouvoir les (dés)activer au vol
        self._ctx_idx_ol0 = self._ctx_menu.index("end") - 1
        self._ctx_idx_bl0 = self._ctx_menu.index("end")

        # Séparateur et zone dynamique pour les actions "mot"
        self._ctx_menu.add_separator()
        self._ctx_idx_words_start = self._ctx_menu.index("end")  # point d'ancrage
        # entrées placeholders (seront remplacées dynamiquement)
        self._ctx_menu.add_command(label="Ajouter …", state="disabled")
        self._ctx_menu.add_command(label="Effacer …", state="disabled")

        # DEBUG: F9 pour activer/désactiver le filtrage de chevauchement au highlight
        # (bind_all pour capter même si le focus clavier n'est pas explicitement sur le canvas)
        self.bind_all("<F9>", self._toggle_skip_overlap_highlight)
        self.bind_all("<F10>", self._toggle_debug_snap_assist)
        # Premier rendu de l'horloge (overlay)
        self._draw_clock_overlay()

    def _on_canvas_configure(self, event=None):
        """Handler unique pour <Configure> (resize du canvas).

        - Redessine l'overlay (horloge, etc.)
        - Invalide le pick-cache
        - Si un fond SVG a été rechargé au démarrage alors que le canvas était trop petit,
          on force UNE fois un redraw complet dès que la taille devient valide.
        """
        # Note: Tk peut spammer <Configure> lors d'un resize -> on debounce.
        try:
            cw_now = int(self.canvas.winfo_width() or 0)
            ch_now = int(self.canvas.winfo_height() or 0)
        except Exception:
            cw_now, ch_now = 0, 0

        # Si la taille a réellement changé, on programme un redraw complet.
        # (sinon, le fond de carte reste « figé » jusqu'au prochain pan/zoom)
        if cw_now > 2 and ch_now > 2:
            last_sz = getattr(self, "_last_canvas_size", None)
            if last_sz != (cw_now, ch_now):
                self._last_canvas_size = (cw_now, ch_now)
                try:
                    if getattr(self, "_resize_redraw_after_id", None) is not None:
                        self.after_cancel(self._resize_redraw_after_id)
                except Exception:
                    pass
                try:
                    self._resize_redraw_after_id = self.after(40, self._do_resize_redraw)
                except Exception:
                    self._resize_redraw_after_id = None

        if getattr(self, "_bg_defer_redraw", False) and getattr(self, "_bg", None):
            cw = int(self.canvas.winfo_width() or 0)
            ch = int(self.canvas.winfo_height() or 0)
            if cw > 2 and ch > 2:
                # Un seul redraw complet (sinon c'est lourd)
                self._bg_defer_redraw = False
                self._redraw_from(self._last_drawn)


        # Overlay + pick-cache (après le redraw complet, car _redraw_from() fait delete('all'))
        self._redraw_overlay_only()
        self._invalidate_pick_cache()


    def _do_resize_redraw(self):
        """Redraw complet après un resize (debounced).

        Important : on redessine tout (fond + triangles + compas), sinon le fond
        peut rester partiellement « non rafraîchi » après agrandissement.
        """
        self._resize_redraw_after_id = None

        try:
            self._redraw_from(self._last_drawn)
        except Exception:
            # Ne pas casser l'IHM sur un resize
            return

        # Overlay + pick-cache (après delete('all') dans _redraw_from)
        self._redraw_overlay_only()
        self._invalidate_pick_cache()

    def _toggle_dico_panel(self):
        """
        Affiche / cache le panneau dictionnaire (combo + liste + grille)
        en fonction de self.show_dico_panel (toggle du menu Visualisation).
        """
        if not hasattr(self, "dicoPanel"):
            return

        show = bool(self.show_dico_panel.get())
        if show:
            # Si le panneau n'est pas déjà packé, on le repack en bas
            if not self.dicoPanel.winfo_ismapped():
                self.dicoPanel.pack(side=tk.BOTTOM, fill=tk.X)
                # Si la grille n'a jamais été construite, on la (re)construit
                if not getattr(self, "dicoSheet", None):
                    self._build_dico_grid()

        else:
            # Cacher le panneau (sans le détruire, pour pouvoir le réafficher)
            self.dicoPanel.pack_forget()

        # Persistance : mémoriser l'état (affiché/caché)
        self.setAppConfigValue("uiShowDicoPanel", bool(show))

    def _toggle_only_group_contours(self):
        """Toggle: afficher uniquement les contours des groupes."""
        self.setAppConfigValue("uiShowOnlyGroupContours", bool(self.show_only_group_contours.get()))
        # Un mode purement visuel -> redraw complet
        self._redraw_from(self._last_drawn)

    def _clock_change_radius(self, delta: int):
        """Modifie le rayon du compas (min=50) et redessine l'overlay."""
        r = int(getattr(self, "_clock_radius", 69))
        r = max(50, r + int(delta))
        self._clock_radius = int(r)
        # Redessiner uniquement l'overlay (ne touche pas aux triangles)
        self._redraw_overlay_only()

    def _toggle_auto_fit_scenario_select(self):
        """Active/désactive le Fit automatique lors de la sélection d'un scénario."""
        self.setAppConfigValue("uiAutoFitScenario", bool(self.auto_fit_scenario_select.get()))

    def _toggle_bg_resize_mode(self):
        """Active/désactive le mode d'édition du fond (redimensionnement + déplacement).
        Important : on force la persistance de la géométrie du fond (x0/y0/w/h)
        quand on quitte le mode, comme pour la largeur/hauteur.
        """
        # Si on sort du mode : purge tout état de drag du fond + sauver la config.
        if not bool(self.bg_resize_mode.get()):
            self._bg_resizing = None
            self._bg_moving = None
            self.canvas.configure(cursor="")
            self._persistBackgroundConfig()
            self._bg_update_scale_status()

        self._redraw_from(self._last_drawn)


    def _toggle_layers(self):
        """Redessine le canvas suite à un changement de visibilité d'un layer."""
        self._redraw_from(self._last_drawn)


    def _on_map_opacity_change(self, value=None):
        """Callback du slider d'opacité de la carte (debounced)."""
        v = int(float(self.map_opacity.get()))
        v = max(0, min(100, v))
        self.map_opacity.set(v)
        self.setAppConfigValue("uiMapOpacity", int(v))
        if self._map_opacity_redraw_job is not None:
            self.after_cancel(self._map_opacity_redraw_job)

        def _do():
            self._map_opacity_redraw_job = None
            self._redraw_from(self._last_drawn)

        self._map_opacity_redraw_job = self.after(60, _do)

    def _toggle_clock_overlay(self):
        """Affiche ou cache le compas horaire (overlay horloge)."""
        if not getattr(self, "canvas", None):
            return
        # Effacer systématiquement l'overlay courant
        self.canvas.delete("clock_overlay")

        # Si l'option est active, on redessine l'horloge (sans toucher aux triangles)
        if self.show_clock_overlay.get():
            self._draw_clock_overlay()

        # Persistance : mémoriser l'état (affiché/caché)
        self.setAppConfigValue("uiShowClockOverlay", bool(self.show_clock_overlay.get()))

    # -- pick cache helpers ---------------------------------------------------
    def _invalidate_pick_cache(self):
        """À appeler dès que zoom/offset ou _last_drawn peuvent changer."""
        self._pick_cache_valid = False

    def _ensure_pick_cache(self):
        """Reconstruit le pick-cache si nécessaire (appel paresseux côté input)."""
        if not self._pick_cache_valid:
            self._rebuild_pick_cache()

    def _rebuild_pick_cache(self):
        """Reconstruit les polygones écran utilisés pour le hit-test."""
        if not self._last_drawn:
            self._pick_cache_valid = True
            return
        Z = float(getattr(self, "zoom", 1.0))
        Ox, Oy = (float(self.offset[0]), float(self.offset[1]))
        def W2S(p):
            # monde -> écran (canvas)
            return (Ox + p[0] * Z, Oy - p[1] * Z)
        for t in self._last_drawn:
            P = t.get("pts", {})
            O = P.get("O"); B = P.get("B"); L = P.get("L")
            if O is None or B is None or L is None:
                t["_pick_poly"] = None
                t["_pick_pts"] = {}
                continue
            Os = W2S(O); Bs = W2S(B); Ls = W2S(L)
            t["_pick_pts"]  = {"O": Os, "B": Bs, "L": Ls}
            t["_pick_poly"] = [Os, Bs, Ls]
        self._pick_cache_valid = True


    # centralisation des bindings canvas/clavier --
    def _bind_canvas_handlers(self):
        """(Ré)applique tous les bindings nécessaires au canvas et au clavier.
        À appeler après création du canvas ET après un chargement de scénario."""
        if not getattr(self, "canvas", None):
            return
        # Purge défensive pour éviter les doublons (Tk ignore les doublons, mais on nettoie)
        self.canvas.unbind("<MouseWheel>")
        self.canvas.unbind("<Button-4>")
        self.canvas.unbind("<Button-5>")
        self.canvas.unbind("<ButtonPress-2>")
        self.canvas.unbind("<B2-Motion>")
        self.canvas.unbind("<ButtonRelease-2>")
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.unbind("<Motion>")
        self.canvas.unbind("<Button-3>")

        # Zoom (wheel)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)   # Linux
        self.canvas.bind("<Button-5>", self._on_mousewheel)   # Linux
        # Pan (middle)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        # Drag/Move (left)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_left_down)
        self.canvas.bind("<B1-Motion>", self._on_canvas_left_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_left_up)
        # Suivi hover / drag preview
        self.canvas.bind("<Motion>", self._on_canvas_motion_update_drag)
        # Menu contextuel
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)

        # Clavier (CTRL pour déconnexion ; F9 pour debug)
        self.bind_all("<KeyPress-Control_L>", self._on_ctrl_down)
        self.bind_all("<KeyPress-Control_R>", self._on_ctrl_down)
        self.bind_all("<KeyRelease-Control_L>", self._on_ctrl_up)
        self.bind_all("<KeyRelease-Control_R>", self._on_ctrl_up)
        self.bind_all("<F9>", self._toggle_skip_overlap_highlight)

    # --- conversions utilitaires (utilisées par le pick) ---
    def _screen_to_world(self, x, y):
        Z = float(getattr(self, "zoom", 1.0))
        Ox, Oy = (float(self.offset[0]), float(self.offset[1]))
        return ((x - Ox) / Z, (Oy - y) / Z)


    # =========================
    #  SCENARIO: SAVE / LOAD
    # =========================
    def _scenario_save_dialog(self):
        """Boîte de dialogue pour enregistrer un scénario en XML."""
        # nom par défaut daté dans le dossier 'scenario'
        ts = _dt.datetime.now().strftime("scenario-%Y%m%d-%H%M.xml")
        initial = os.path.join(self.scenario_dir, ts)
        path = filedialog.asksaveasfilename(
            title="Enregistrer le scénario",
            defaultextension=".xml",
            initialfile=os.path.basename(initial),
            initialdir=self.scenario_dir,
            filetypes=[("Scenario XML", "*.xml"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return
        try:
            self.save_scenario_xml(path)
            self.status.config(text=f"Scénario enregistré dans {path}")
        except Exception as e:
            messagebox.showerror("Enregistrer le scénario", str(e))


    def _load_scenario_into_new_scenario(self, path: str):
        """
        Charge un fichier scénario XML dans un *nouveau* scénario.
        - Le scénario courant n'est pas modifié.
        - Le nouveau scénario porte le nom du fichier XML (sans extension).
        """
        # Nom lisible dérivé du fichier
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        if not name:
            name = base

        prev_index = self.active_scenario_index
        new_index = len(self.scenarios)

        scen = ScenarioAssemblage(
            name=name,
            source_type="manual",
            algo_id=None,
            tri_ids=[],
        )
        # Structures indépendantes pour ce scénario
        scen.last_drawn = []
        scen.groups = {}

        self.scenarios.append(scen)

        try:
            # On bascule sur ce nouveau scénario AVANT de charger,
            # ainsi load_scenario_xml() écrit bien dans ses structures.
            self._set_active_scenario(new_index)
            self.load_scenario_xml(path)
        except Exception:
            # En cas d'erreur, on nettoie le scénario incomplet
            try:
                self.scenarios.pop(new_index)
            except Exception:
                pass
            # On revient au scénario précédent (s'il existe encore)
            try:
                if self.scenarios:
                    if 0 <= prev_index < len(self.scenarios):
                        self._set_active_scenario(prev_index)
                    else:
                        self._set_active_scenario(len(self.scenarios) - 1)
            except Exception:
                pass
            self._refresh_scenario_listbox()
            raise

        # Succès : rafraîchir la liste et le statut
        self._refresh_scenario_listbox()
        try:
            self.status.config(text=f"Scénario importé : {scen.name}")
        except Exception:
            pass

    def _scenario_load_dialog(self):
        """Boîte de dialogue pour charger un scénario XML."""
        path = filedialog.askopenfilename(
            title="Charger un scénario",
            initialdir=self.scenario_dir,
            filetypes=[("Scenario XML", "*.xml"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return
        try:
            self._load_scenario_into_new_scenario(path)
        except Exception as e:
            messagebox.showerror("Charger le scénario", str(e))
    
    def _rebuild_scenario_file_list_menu(self):
        """Recrée la liste des scénarios (XML) disponibles dans le menu Scénario."""
        m = self.menu_scenario
        if m is None:
            return
        # supprimer tout ce qui suit l’ancre
        end = m.index("end")
        while end is not None and end > self._menu_scenario_files_anchor:
            m.delete(end)
            end = m.index("end")
        # re-remplir
        try:
            files = [f for f in os.listdir(self.scenario_dir) if f.lower().endswith(".xml")]
        except Exception:
            files = []
        if not files:
            m.add_command(label="(aucun scénario dans 'scenario')", state="disabled")
            return
        files.sort(key=str.lower)
        for fname in files:
            full = os.path.join(self.scenario_dir, fname)
            # Chaque fichier XML crée désormais un *nouveau* scénario
            m.add_command(label=fname, command=lambda p=full: self._load_scenario_into_new_scenario(p))

    def _pt_to_xml(self, p):
        return f"{float(p[0]):.9g},{float(p[1]):.9g}"

    def _xml_to_pt(self, s):
        x, y = str(s).split(",")
        return np.array([float(x), float(y)], dtype=float)

    def save_scenario_xml(self, path: str):
        return _assembleur_io.saveScenarioXml(self, path)

    def load_scenario_xml(self, path: str):
        return _assembleur_io.loadScenarioXml(self, path)

    def _is_in_clock(self, x, y, pad=10):
        """Vrai si (x,y) (coords canvas) est à l'intérieur du disque de l'horloge (+marge)."""
        if self._clock_cx is None or self._clock_cy is None:
            return False
        dx = x - self._clock_cx
        dy = y - self._clock_cy
        return (dx*dx + dy*dy) <= (self._clock_R + pad) ** 2

    # ---------- Tooltip helpers ----------
    def _ui_attach_tooltip(self, widget, text: str):
        """Attache un tooltip (petit Toplevel) à un widget Tk (ex: bouton icône)."""
        if widget is None:
            return
        tip = str(text or "").strip()
        if not tip:
            return
        # stocker le texte sur le widget (pratique pour MAJ future)
        widget._ui_tooltip_text = tip
        widget.bind("<Enter>", lambda e, w=widget: self._ui_show_tooltip(w), add="+")
        widget.bind("<Leave>", lambda e: self._ui_hide_tooltip(), add="+")
        widget.bind("<ButtonPress>", lambda e: self._ui_hide_tooltip(), add="+")
 
    def _ui_show_tooltip(self, widget):
        """Affiche le tooltip UI près du widget."""
        if widget is None:
            return
        text = str(getattr(widget, "_ui_tooltip_text", "") or "").strip()
        if not text:
            return
 
        # créer si nécessaire
        if not hasattr(self, "_ui_tooltip"):
            self._ui_tooltip = None
            self._ui_tooltip_label = None
 
        if self._ui_tooltip is None or not self._ui_tooltip.winfo_exists():
            self._ui_tooltip = tk.Toplevel(self)
            self._ui_tooltip.wm_overrideredirect(True)
            self._ui_tooltip.attributes("-topmost", True)
            self._ui_tooltip_label = tk.Label(
                self._ui_tooltip,
                text=text,
                bg="#ffffe0",
                relief="solid",
                borderwidth=1,
                font=("Arial", 9),
                justify="left",
                anchor="w",
            )
            self._ui_tooltip_label.pack(ipadx=4, ipady=2)
        else:
            self._ui_tooltip_label.config(text=text, justify="left", anchor="w")

        # placer près du widget (léger offset)
        self._ui_tooltip.update_idletasks()
        tw = max(1, int(self._ui_tooltip.winfo_width()))
        th = max(1, int(self._ui_tooltip.winfo_height()))
        x = int(widget.winfo_rootx() + 10)
        y = int(widget.winfo_rooty() + widget.winfo_height() + 8)

        # clamp dans l'écran courant (simple)
        sw = int(self.winfo_screenwidth())
        sh = int(self.winfo_screenheight())
        x = max(0, min(x, sw - tw))
        y = max(0, min(y, sh - th))
 
        self._ui_tooltip.wm_geometry(f"+{x}+{y}")
 
    def _ui_hide_tooltip(self):
        if hasattr(self, "_ui_tooltip") and self._ui_tooltip is not None and self._ui_tooltip.winfo_exists():
            self._ui_tooltip.destroy()
        if hasattr(self, "_ui_tooltip"):
            self._ui_tooltip = None
            self._ui_tooltip_label = None
 
    def _show_tooltip_at_center(self, text: str, cx_canvas: float, cy_canvas: float):
        """Affiche/MAJ le tooltip en le **centrant** sur (cx_canvas, cy_canvas) (coords CANVAS)."""
        if not text:
            self._hide_tooltip(); return
        if self._tooltip is None or not self._tooltip.winfo_exists():
            self._tooltip = tk.Toplevel(self)
            self._tooltip.wm_overrideredirect(True)
            try:
                self._tooltip.attributes("-topmost", True)
            except Exception:
                pass
            self._tooltip_label = tk.Label(self._tooltip, text=text, bg="#ffffe0",
                                           relief="solid", borderwidth=1, font=("Arial", 9),
                                           justify="left", anchor="w")
            self._tooltip_label.pack(ipadx=4, ipady=2)
        else:
            self._tooltip_label.config(text=text, justify="left", anchor="w")
        # Mesurer la taille réelle du tooltip
        self._tooltip.update_idletasks()
        tw = max(1, int(self._tooltip.winfo_width()))
        th = max(1, int(self._tooltip.winfo_height()))
        # Convertir coords CANVAS -> écran et centrer
        base_x = self.canvas.winfo_rootx()
        base_y = self.canvas.winfo_rooty()
        x = int(base_x + cx_canvas - tw/2)
        y = int(base_y + cy_canvas - th/2)
        c_w = int(self.canvas.winfo_width())
        c_h = int(self.canvas.winfo_height())
        min_x = base_x
        max_x = base_x + c_w - tw
        min_y = base_y
        max_y = base_y + c_h - th
        if max_x < min_x:
            max_x = min_x
        if max_y < min_y:
            max_y = min_y
        x = max(min_x, min(x, max_x))
        y = max(min_y, min(y, max_y))
        self._tooltip.wm_geometry(f"+{x}+{y}")

    def _hide_tooltip(self):
        if self._tooltip is not None and self._tooltip.winfo_exists():
            try: self._tooltip.destroy()
            except Exception: pass
        self._tooltip = None
        self._tooltip_label = None

    # ---------- Config (JSON) ----------
    def loadAppConfig(self):
        return _assembleur_io.loadAppConfig(self)

    def saveAppConfig(self):
        return _assembleur_io.saveAppConfig(self)

    def getAppConfigValue(self, key, default=None):
        return _assembleur_io.getAppConfigValue(self, key, default)

    def setAppConfigValue(self, key, value):
        return _assembleur_io.setAppConfigValue(self, key, value)

    def _persistBackgroundConfig(self):
        """Sauvegarde en config l'état du fond SVG (path + rect monde)."""
        try:
            if not hasattr(self, "appConfig") or self.appConfig is None:
                self.appConfig = {}
            if not getattr(self, "_bg", None):
                changed = False
                for k in ("bgSvgPath", "bgWorldRect"):
                    if k in self.appConfig:
                        del self.appConfig[k]
                        changed = True
                if changed:
                    self.saveAppConfig()
                return

            self.appConfig["bgSvgPath"] = str(self._bg.get("path") or "")
            self.appConfig["bgWorldRect"] = {
                "x0": float(self._bg.get("x0", 0.0)),
                "y0": float(self._bg.get("y0", 0.0)),
                "w": float(self._bg.get("w", 1.0)),
                "h": float(self._bg.get("h", 1.0)),
                "aspect": float(self._bg.get("aspect", 1.0)),
            }
            self.saveAppConfig()
        except Exception:
            pass

    def autoLoadBackgroundAtStartup(self):
        """Recharge le fond SVG et sa géométrie depuis la config.

        Important : au moment où _build_ui() s'exécute, le canvas peut encore avoir une taille
        (winfo_width/height) quasi nulle. Dans ce cas, _bg_draw_world_layer() ne peut pas
        rasteriser/afficher correctement et le fond semble "non rechargé".

        Solution : charger le SVG *après* la 1ère passe de layout (after_idle) et forcer
        un redraw complet au 1er <Configure> valide.
        """
        try:
            if getattr(self, "_bg_startup_scheduled", False):
                return
            self._bg_startup_scheduled = True
            self.after_idle(self._autoLoadBackgroundAfterLayout)
        except Exception:
            # best-effort
            self._bg_startup_scheduled = False

    def _autoLoadBackgroundAfterLayout(self):
        """(interne) Effectue le vrai rechargement du fond après layout."""
        try:
            self._bg_startup_scheduled = False
            svg_path = self.getAppConfigValue("bgSvgPath", "") or ""
            rect = self.getAppConfigValue("bgWorldRect", None)
            if not svg_path:
                return

            # Normaliser le chemin (Windows tolère / et \ mais os.path.isfile peut être sensible
            # selon certains contextes d'exécution).
            svg_path = os.path.normpath(str(svg_path))

            if not os.path.isfile(svg_path):
                return

            # S'assurer que la géométrie Tk est calculée (best-effort)
            try:
                self.update_idletasks()
            except Exception:
                pass

            if isinstance(rect, dict) and all(k in rect for k in ("x0", "y0", "w", "h")):
                self._bg_set_map(svg_path, rect_override=rect, persist=False)
            else:
                self._bg_set_map(svg_path, rect_override=None, persist=False)

            # Si le canvas était encore trop petit au moment du redraw, on redessinera une fois
            # dès qu'un <Configure> avec une taille valide arrive.
            self._bg_defer_redraw = True
        except Exception:
            # best-effort
            self._bg_startup_scheduled = False

    def autoLoadTrianglesFileAtStartup(self):
        """Recharge au démarrage le dernier fichier triangles ouvert."""
        # 1) Dernier fichier explicitement chargé
        last = self.getAppConfigValue("lastTriangleExcel", "")
        if last and os.path.isfile(str(last)):
            try:
                self.load_excel(str(last))
                return
            except Exception:
                # On tente un fallback si le fichier est devenu invalide
                pass
        # 2) Fallback : data/triangle.xlsx
        try:
            default = os.path.join(getattr(self, "data_dir", ""), "triangle.xlsx")
            if default and os.path.isfile(default):
                self.load_excel(default)
        except Exception:
            pass

    # ---------- Chargement Excel ----------
    def open_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if path:
            self.load_excel(path)

    @staticmethod
    def _norm(s: str) -> str:
        import unicodedata, re
        s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()
        return re.sub(r"[^a-z0-9]+", "", s)

    @staticmethod
    def _find_header_row(df0: pd.DataFrame) -> int:
        for i in range(min(12, len(df0))):
            row_norm = [TriangleViewerManual._norm(x) for x in df0.iloc[i].tolist()]
            if any("ouverture" in x for x in row_norm) and \
               any("base"      in x for x in row_norm) and \
               any("lumiere"   in x for x in row_norm):
                return i
        raise KeyError("Impossible de détecter l'entête ('Ouverture', 'Base', 'Lumière').")

    @staticmethod
    def _build_df(df: pd.DataFrame) -> pd.DataFrame:
        cmap = {TriangleViewerManual._norm(c): c for c in df.columns}
        col_id = cmap.get("rang") or cmap.get("id")
        col_B  = cmap.get("base")
        col_L  = cmap.get("lumiere")
        col_OB = cmap.get("ouverturebase")
        col_OL = cmap.get("ouverturelumiere")
        col_BL = cmap.get("lumierebase")
        col_OR = cmap.get("orientation")  # colonne Orientation (CCW/CW/COL)  
        missing = [n for n,c in {
            "Base":col_B, "Lumière":col_L, "Ouverture-Base":col_OB,
            "Ouverture-Lumière":col_OL, "Lumière-Base":col_BL
        }.items() if c is None]
        if missing:
            raise KeyError("Colonnes manquantes: " + ", ".join(missing))
        out = pd.DataFrame({
            "id": df[col_id] if col_id else range(1, len(df)+1),
            "B":  df[col_B],
            "L":  df[col_L],
            "len_OB": pd.to_numeric(df[col_OB], errors="coerce"),
            "len_OL": pd.to_numeric(df[col_OL], errors="coerce"),
            "len_BL": pd.to_numeric(df[col_BL], errors="coerce"),
            # Orientation normalisée (par défaut CCW si absent)
            "orient": (
                df[col_OR].astype(str).str.upper().str.strip()
                if col_OR else pd.Series(["CCW"] * len(df))
            ),
        }).dropna(subset=["len_OB","len_OL","len_BL"]).sort_values("id")
        return out.reset_index(drop=True)

    def load_excel(self, path: str):
        df0 = pd.read_excel(path, header=None)
        header_row = self._find_header_row(df0)
        df = pd.read_excel(path, header=header_row)
        self.df = self._build_df(df)
        path_norm = os.path.normpath(os.path.abspath(path))
        self.excel_path = path_norm
        self.setAppConfigValue("lastTriangleExcel", path_norm)
        self.triangle_file.set(os.path.basename(path_norm))
        print(f"[Triangles] Fichier chargé: {os.path.basename(path_norm)} ({path_norm})")
        # MAJ du menu fichiers si un nouveau fichier arrive dans data
        self._rebuild_triangle_file_list_menu()

        self.listbox.delete(0, tk.END)
        for _, r in self.df.iterrows():
            self.listbox.insert(tk.END, f"{int(r['id']):02d}. B:{r['B']}  L:{r['L']}")
        self.status.config(text=f"{len(self.df)} triangles chargés depuis {path_norm}")
        # Réinitialiser les triangles posés du scénario actif
        # IMPORTANT : on vide la liste en place pour préserver le lien
        # avec scen.last_drawn du scénario manuel.
        self._last_drawn.clear()
        # Aucun triangle n'est encore utilisé dans le scénario actif        self._placed_ids = set()
        self._update_triangle_listbox_colors()

    # ---------- Triangles (données locales) ----------
    def _triangles_local(self, start, n):
        """
        Construit les triangles (coord. locales) à partir du DF pour [start:start+n].
        Retour: liste de dict { 'labels':(O,B,L), 'pts':{'O','B','L'} }
        """
        if getattr(self, "df", None) is None or self.df.empty:
            raise RuntimeError("Pas de données — ouvre d'abord l’Excel.")
        sub = self.df.iloc[start:start+n]
        out = []
        for _, r in sub.iterrows():
            P = _build_local_triangle(float(r["len_OB"]), float(r["len_OL"]), float(r["len_BL"]))
            # Si orientation = CW, on applique une symétrie verticale (y -> -y)
            try:
                ori = str(r.get("orient", "CCW")).upper()
            except Exception:
                ori = "CCW"
            if ori == "CW":
                P = {"O": np.array([P["O"][0], -P["O"][1]]),
                     "B": np.array([P["B"][0], -P["B"][1]]),
                     "L": np.array([P["L"][0], -P["L"][1]])}
            out.append({
                "labels": ( "Bourges", str(r["B"]), str(r["L"]) ),  # O,B,L
                "pts": P,
                "id": int(r["id"]),
                "mirrored": (ori == "CW"),
            })
        return out

    # ---------- Mise en page simple (aperçu brut) ----------
    def _triangle_from_index(self, idx):
        """Construit un triangle 'local' depuis l’élément sélectionné de la listbox.
        IMPORTANT: on parse l'ID affiché (NN.) au lieu d'utiliser df.iloc[idx],
        car la listbox peut avoir des éléments retirés → indices décalés.
        """
        if getattr(self, "df", None) is None or self.df.empty:
            raise RuntimeError("Pas de données — ouvre d'abord l’Excel.")
        # Récupérer le texte de la listbox et extraire l'id (NN. ...)
        lb_txt = ""
        try:
            if 0 <= idx < self.listbox.size():
                lb_txt = self.listbox.get(idx)
        except Exception:
            pass
        tri_id = None
        m = re.match(r"\s*(\d+)\.", str(lb_txt))
        if m:
            tri_id = int(m.group(1))
            row = self.df[self.df["id"] == tri_id]
            if not row.empty:
                r = row.iloc[0]
            else:
                # secours si pas trouvé (ne devrait pas arriver)
                r = self.df.iloc[min(max(idx, 0), len(self.df)-1)]
                tri_id = int(r["id"])
        else:
            # si le libellé n'est pas conforme, on retombe sur l'index
            r = self.df.iloc[min(max(idx, 0), len(self.df)-1)]
            tri_id = int(r["id"])

        P = _build_local_triangle(float(r["len_OB"]), float(r["len_OL"]), float(r["len_BL"]))
        # Normalisation de l’orientation
        try:
            ori = str(r.get("orient", "CCW")).upper()
        except Exception:
            ori = "CCW"
        if ori == "CW":
            P = {"O": np.array([P["O"][0], -P["O"][1]]),
                 "B": np.array([P["B"][0], -P["B"][1]]),
                 "L": np.array([P["L"][0], -P["L"][1]])}
        return {
            "labels": ("Bourges", str(r["B"]), str(r["L"])),
            "pts": P,
            "id": tri_id,
            "mirrored": (ori == "CW"),   # reflet initial si orientation horaire
        }

    def _layout_tris_horizontal(self, tris, gap=0.5, wrap_width=None):
        """
        Place les triangles les uns à côté des autres (dans le repère "monde"),
        avec un espacement 'gap' (unités mêmes que les longueurs).
        Si wrap_width est fourni (en unités monde), on passe à la ligne quand la largeur est dépassée.
        Retour: liste de dict { 'labels', 'pts' (coordonnées MONDE après translation) }
        """
        placed = []
        x_cursor = 0.0
        y_cursor = 0.0
        line_height = 0.0

        if wrap_width is None:
            # largeur de ligne par défaut : environ 20 unités de longueur
            wrap_width = 20.0

        for t in tris:
            P = t["pts"]
            xs = [float(P["O"][0]), float(P["B"][0]), float(P["L"][0])]
            ys = [float(P["O"][1]), float(P["B"][1]), float(P["L"][1])]
            mnx, mny, mxx, mxy = min(xs), min(ys), max(xs), max(ys)
            w = (mxx - mnx)
            h = (mxy - mny)

            # saut de ligne si nécessaire
            if x_cursor > 0.0 and (x_cursor + w) > wrap_width:
                x_cursor = 0.0
                y_cursor -= (line_height + gap)  # vers le bas (coord monde Y+ en haut -> inversé en écran)
                line_height = 0.0

            # translation pour placer ce triangle
            dx = x_cursor - mnx
            dy = y_cursor - mny
            Pw = {
                "O": np.array([P["O"][0] + dx, P["O"][1] + dy]),
                "B": np.array([P["B"][0] + dx, P["B"][1] + dy]),
                "L": np.array([P["L"][0] + dx, P["L"][1] + dy]),
            }
            placed.append({"labels": t["labels"], "pts": Pw, "id": t.get("id"), "mirrored": t.get("mirrored", False)})

            x_cursor += w + gap
            line_height = max(line_height, h)

        return placed

    # ====== FRONTIER GRAPH HELPERS (factorisation) ===============================================
    def _edge_dir(self, a, b):
        return (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))

    def _ang_wrap(self, x):
        import math
        # wrap sur [-pi, pi]
        x = (x + math.pi) % (2*math.pi) - math.pi
        return x

    def _ang_of_vec(self, vx, vy):
        import math
        return math.atan2(vy, vx)

    def _ang_diff(self, a, b):
        # plus petit écart absolu d’angle
        return abs(self._ang_wrap(a - b))

    def _pt_key_eps(self, p, eps=EPS_WORLD):
        # IMPORTANT:
        # Toute la logique "boundary graph" doit utiliser la même granularité (EPS_WORLD),
        # sinon on casse les lookup adjacence (graph["adj"]) et on se retrouve avec 0 demi-arêtes.
        eps = float(eps) if eps is not None else float(EPS_WORLD)
        if eps <= 0.0:
            eps = float(EPS_WORLD)
        return (
            round(float(p[0]) / eps) * eps,
            round(float(p[1]) / eps) * eps,
        )

    def _build_boundary_graph(self, outline):
        """
        Construire un graphe de frontière léger (half-edges) à partir de 'outline'
        outline: liste de segments [(a,b), ...]
        Retour: dict {
            "adj": {key(point)-> [point_voisin,...]},
            "pts": {key(point)-> point_float_tuple}
        }
        """
        adj = {}
        pts = {}
        for a,b in (outline or []):
            a = (float(a[0]), float(a[1])); b = (float(b[0]), float(b[1]))
            ka, kb = self._pt_key_eps(a), self._pt_key_eps(b)
            pts.setdefault(ka, a); pts.setdefault(kb, b)
            adj.setdefault(ka, []).append(b)
            adj.setdefault(kb, []).append(a)
        return {"adj": adj, "pts": pts}

    def _incident_half_edges_at_vertex(self, graph, v, eps=EPS_WORLD):
        """
        Renvoie les deux demi-arêtes (si existantes) qui partent de 'v' le long de la frontière.
        Retour: liste de 0..2 éléments, chaque élément est ((ax,ay),(bx,by))
        """
        key = self._pt_key_eps(v)
        neighs = graph["adj"].get(key, [])
        out = []
        for w in neighs:
            out.append(( (float(v[0]),float(v[1])), (float(w[0]),float(w[1])) ))
        # garder au maximum 2 (enveloppe standard) ; si plus, on trie par angle autour de v et on prend 2 extrêmes
        if len(out) > 2:
            import math
            def ang(e):
                (a,b)=e; vx,vy=self._edge_dir(a,b); return math.atan2(vy,vx)
            out = sorted(out, key=ang)
            # choisir deux qui maximisent l'écart angulaire (les bords de l'enveloppe)
            # heuristique simple: prendre le premier et celui qui maximise |Δ|
            a0 = out[0]; ang0 = ang(a0)
            a1 = max(out[1:], key=lambda e: self._ang_diff(ang(e), ang0))
            out = [a0,a1]
        return out

    def _incident_half_edges_at_point(self, graph, v, outline, eps=EPS_WORLD):
        """Retourne 2 demi-arêtes incidentes au point `v` sur la frontière.

        Cas gérés:
     - `v` est un sommet du graphe de frontière -> _incident_half_edges_at_vertex
        - `v` n'est pas un sommet mais tombe sur un segment de l'outline :
          on retourne les 2 demi-segments (v->A) et (v->B) du segment (A,B).
        """
        v = (float(v[0]), float(v[1]))
        out = self._incident_half_edges_at_vertex(graph, v, eps=eps)
        if len(out) >= 2:
            return out

        # --- Fallback 1 : v est "près" d'un sommet de l'outline, mais ne match pas la clé d'adjacence ---
        # Cas typique : le snap s'accroche à un noeud du triangle (coordonnées exactes),
        # mais l'outline provient d'une union Shapely et ses sommets sont légèrement décalés.
        # On cherche alors les arêtes de l'outline qui ont une extrémité à <= eps de v,
        # et on retourne les demi-arêtes sortantes depuis ce sommet.
        if outline:
            px, py = float(v[0]), float(v[1])
            cand = []
            e2 = float(eps) * float(eps)
            for (a, b) in outline:
                ax, ay = float(a[0]), float(a[1])
                bx, by = float(b[0]), float(b[1])
                da2 = (px - ax) * (px - ax) + (py - ay) * (py - ay)
                db2 = (px - bx) * (px - bx) + (py - by) * (py - by)
                if da2 <= e2:
                    cand.append(((ax, ay), (bx, by)))
                if db2 <= e2:
                    cand.append(((bx, by), (ax, ay)))

            # Dédupliquer
            uniq = []
            seen = set()
            for (a, b) in cand:
                k = (round(a[0], 9), round(a[1], 9), round(b[0], 9), round(b[1], 9))
                if k in seen:
                    continue
                seen.add(k)
                uniq.append((a, b))

            if len(uniq) >= 2:
                out2 = [((u[0][0], u[0][1]), (u[1][0], u[1][1])) for u in uniq]
                if len(out2) > 2:
                    import math
                    def ang(e):
                        (a, b) = e
                        vx, vy = self._edge_dir(a, b)
                        return math.atan2(vy, vx)
                    out2 = sorted(out2, key=ang)
                    a0 = out2[0]
                    ang0 = ang(a0)
                    a1 = max(out2[1:], key=lambda e: self._ang_diff(ang(e), ang0))
                    out2 = [a0, a1]
                return out2[:2]

        def _pt_seg_dist2(px, py, ax, ay, bx, by):
            # distance^2 point-segment + projection t
            vx, vy = bx - ax, by - ay
            wx, wy = px - ax, py - ay
            vv = vx * vx + vy * vy
            if vv <= 1e-18:
               # segment dégénéré
                dx, dy = px - ax, py - ay
                return (dx * dx + dy * dy, 0.0)
            t = (wx * vx + wy * vy) / vv
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            qx, qy = ax + t * vx, ay + t * vy
            dx, dy = px - qx, py - qy
            return (dx * dx + dy * dy, float(t))

        px, py = float(v[0]), float(v[1])
        best = None  # (dist2, (a,b))
        for (a, b) in outline:
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            d2, t = _pt_seg_dist2(px, py, ax, ay, bx, by)
            # On veut vraiment "sur" le segment (tolérance). On accepte aussi les extrémités
            # car certains noeuds "collés" ne matchent pas forcément la clé d'adjacence.
            if d2 <= float(eps) * float(eps):
                if best is None or d2 < best[0]:
                    best = (d2, ((ax, ay), (bx, by)))

        if best is None:
            return out

        (a, b) = best[1]
        return [((px, py), (float(a[0]), float(a[1]))), ((px, py), (float(b[0]), float(b[1])))]

    def _normalize_to_outline_granularity(self, outline, edges, eps=EPS_WORLD):
        """
        Décompose chaque segment incident en chaîne de micro-segments collés à l'outline (granularité identique),
        en s'appuyant sur l'adjacence du graphe de frontière. Retourne une liste de segments.
        """
        g = self._build_boundary_graph(outline)
        def almost(p,q):
            return abs(p[0]-q[0])<=eps and abs(p[1]-q[1])<=eps
        def dir_forward(u,v,w):
            uvx,uvy = v[0]-u[0], v[1]-u[1]
            uwx,uwy = w[0]-u[0], w[1]-u[1]
            cross = abs(uvx*uwy - uvy*uwx)
            dot   = (uvx*uwx + uvy*uwy)
            return cross <= 1e-9 and dot > 0.0
        out=[]
        for (a,b) in (edges or []):
            a=(float(a[0]),float(a[1])); b=(float(b[0]),float(b[1]))
            cur=a; guard=0; chain=[]
            while not almost(cur,b) and guard<2048:
                neigh = g["adj"].get(self._pt_key_eps(cur), [])
                nxt=None
                for w in neigh:
                    if dir_forward(cur,b,w):
                        nxt=w; break
                if nxt is None: break
                chain.append((cur,nxt))
                cur=nxt; guard+=1
            if chain and almost(cur,b):
                out.extend(chain)
            else:
                # fallback: garder le segment brut si pas décomposable finement
                out.append((a,b))
        return out



    # ------------------------------
    # Géométrie / utils divers
    # ------------------------------
    def _ang_of_vec(self, vx, vy):
        import math
        return math.atan2(vy, vx)
 
    def _ang_diff(self, a, b):
        # plus petit écart absolu d’angle
        return abs(self._ang_wrap(a - b))
         

    def _refresh_listbox_from_df(self):
        """Recharge la listbox des triangles depuis self.df (sans filtrage)."""
        try:
            self.listbox.delete(0, tk.END)
            if getattr(self, "df", None) is not None and not self.df.empty:
                for _, r in self.df.iterrows():
                    self.listbox.insert(tk.END, f"{int(r['id']):02d}. B:{r['B']}  L:{r['L']}")
        except Exception:
            pass

    def clear_canvas(self):
        """Efface l'affichage après confirmation, et remet à jour la liste des triangles."""
        if not messagebox.askyesno(
            "Effacer l'affichage",
            "Voulez-vous effacer l'affichage et réinitialiser la liste des triangles ?"
        ):
            return
        try:
            self.canvas.delete("all")
        except Exception:
            pass
        # Réinitialiser l'état d'affichage du scénario actif
        # (vidage en place pour garder le lien scen.last_drawn)
        self._last_drawn.clear()
        self._nearest_line_id = None
        self._clear_edge_highlights()
        self._edge_choice = None
        self._edge_highlights = None
        # Reconstruire la listbox à partir de la DF courante
        if hasattr(self, "triangle_df") and self.triangle_df is not None:
            self._refresh_listbox_from_df()
        # Après effacement, plus aucun triangle n'est considéré comme "déjà utilisé"
        # → on peut à nouveau les sélectionner pour un drag & drop.
        self._placed_ids.clear()
        self._update_triangle_listbox_colors()

        self.status.config(text="Affichage effacé")
        self._hide_tooltip()

        # L’horloge reste visible (overlay)
        self._draw_clock_overlay()

    # ---------- Overlay Horloge (indépendant du zoom/pan) ----------
    def _redraw_overlay_only(self):
        """Efface/redessine uniquement l'overlay (horloge)."""
        if not getattr(self, "canvas", None):
            return
        try:
            self.canvas.delete("clock_overlay")
        except Exception:
            pass
        self._draw_clock_overlay()

    def _draw_clock_overlay(self):
        """
        Dessine une horloge en haut-gauche du canvas, à taille FIXE (px),
        indépendante du zoom/pan (coordonnées canvas).
        Utilise self._clock_state = {'hour':h, 'minute':m, 'label':str}.
        """
        if not getattr(self, "canvas", None):
            return
        # Nettoyer l'ancien overlay
        self.canvas.delete("clock_overlay")

        # Si le compas est masqué via le menu, ne rien dessiner
        if hasattr(self, "show_clock_overlay") and not self.show_clock_overlay.get():
            return
        # Paramètres d'aspect
        margin = 12              # marge par rapport aux bords du canvas (px)
        # rayon (px) — modifiable via UI (min=50)
        R = max(50, int(getattr(self, "_clock_radius", 69)))
        # Si un ancrage monde existe (et qu'on ne drag pas), le centre du compas suit pan/zoom
        if getattr(self, "_clock_anchor_world", None) is not None and not getattr(self, "_clock_dragging", False):
            sx, sy = self._world_to_screen(self._clock_anchor_world)
            self._clock_cx, self._clock_cy = float(sx), float(sy)
        # Si on a déjà une position écran mais pas d'ancrage monde, on l'initialise
        if getattr(self, "_clock_anchor_world", None) is None and self._clock_cx is not None and self._clock_cy is not None and not getattr(self, "_clock_dragging", False):
            wx, wy = self._screen_to_world(self._clock_cx, self._clock_cy)
            self._clock_anchor_world = np.array([wx, wy], dtype=float)

        # Si première fois, placer en haut-gauche ; sinon garder la position utilisateur
        if self._clock_cx is None or self._clock_cy is None:
            cx = margin + R
            cy = margin + R
            self._clock_cx, self._clock_cy = cx, cy
        else:
            cx, cy = float(self._clock_cx), float(self._clock_cy)
        self._clock_R = R

        # Axe 0 de référence (azimut)
        ref_az = float(getattr(self, "_clock_ref_azimuth_deg", 0.0)) % 360.0

        # Couleurs
        col_circle = "#b0b0b0"   # gris cercle
        col_ticks  = "#707070"
        col_hour   = "#0b3d91"   # bleue (petite aiguille)
        col_min    = "#000000"   # noire  (grande aiguille)

        # Données
        # NOTE: l'algo de décryptage peut fournir une heure "float" (si l'aiguille avance avec les minutes)
        #       ou une heure "int" (si elle reste sur l'heure pile). On respecte donc cette valeur telle quelle.
        # Bases dynamiques (60/100 et 12/10) pilotées par le decryptor
        hBase = int(getattr(self.decryptor, "getHoursBase", lambda: getattr(self.decryptor, "hoursBase", 12))())
        mBase = int(getattr(self.decryptor, "getMinutesBase", lambda: getattr(self.decryptor, "minutesBase", 60))())
        hBase = max(1, int(hBase))
        mBase = max(1, int(mBase))

        hFloat = float(self._clock_state.get("hour", 5.0)) % float(hBase)
        m = int(self._clock_state.get("minute", 9)) % int(mBase)
        label = str(self._clock_state.get("label", ""))

        # Cercle
        self.canvas.create_oval(cx-R, cy-R, cx+R, cy+R,
                                outline=col_circle, width=2, tags="clock_overlay")

        # Dessiner l'axe de référence (0°) selon l'azimut de référence
        th = math.radians(ref_az)
        x_ref = cx + (R * 0.92) * math.sin(th)
        y_ref = cy - (R * 0.92) * math.cos(th)
        self.canvas.create_line(cx, cy, x_ref, y_ref, width=2, fill="#404040", tags=("clock_overlay",))

        # Graduations minutes : un trait toutes les minutes
        deg_per_min = 360.0 / float(mBase)
        for k in range(int(mBase)):
            ang = math.radians(ref_az + k * deg_per_min)  # 360/mBase + ref az
            if k % 5 == 0:
                inner = R - 10
                w = 1
            else:
                inner = R - 6
                w = 1
            outer = R
            x1 = cx + inner * math.sin(ang)
            y1 = cy - inner * math.cos(ang)
            x2 = cx + outer * math.sin(ang)
            y2 = cy - outer * math.cos(ang)
            self.canvas.create_line(x1, y1, x2, y2, width=w, fill=col_ticks, tags="clock_overlay")

        # Graduations heures (hBase traits)
        deg_per_hour = 360.0 / float(hBase)
        for hmark in range(int(hBase)):
            ang = math.radians(ref_az + hmark * deg_per_hour)  # 360/hBase + ref az
            # longueur du trait
            # Repères plus longs : quarts si possible
            if (hBase % 4 == 0 and hmark % int(hBase // 4) == 0) or (hBase == 12 and hmark % 3 == 0):
                inner = R - 14
                w = 2
            else:
                inner = R - 8
                w = 1
            outer = R
            x1 = cx + inner * math.sin(ang)
            y1 = cy - inner * math.cos(ang)
            x2 = cx + outer * math.sin(ang)
            y2 = cy - outer * math.cos(ang)
            self.canvas.create_line(x1, y1, x2, y2, width=w, fill=col_ticks, tags="clock_overlay")

        # Repères (alignés sur l'axe de référence)
        font_marks = ("Arial", 11, "bold")

        def _pos(angle_deg, rad):
            a = math.radians(angle_deg)
            return (cx + rad * math.sin(a), cy - rad * math.cos(a))

        r_text = R - 18
        x12, y12 = _pos(ref_az + 0.0,   r_text)
        x3,  y3  = _pos(ref_az + 90.0,  r_text)
        x6,  y6  = _pos(ref_az + 180.0, r_text)
        x9,  y9  = _pos(ref_az + 270.0, r_text)

        def _fmt_mark(v):
            try:
                fv = float(v)
                if abs(fv - round(fv)) < 1e-9:
                    return str(int(round(fv)))
                # 1 décimale max
                return f"{fv:.1f}".rstrip("0").rstrip(".")
            except Exception:
                return str(v)

        # Haut = hBase (12 ou 10)
        self.canvas.create_text(x12, y12, text=_fmt_mark(hBase), font=font_marks,
                                fill=col_ticks, tags="clock_overlay")
        # Droite / bas / gauche : quarts (peuvent être décimaux si hBase=10)
        self.canvas.create_text(x3,  y3,  text=_fmt_mark(hBase / 4.0),  font=font_marks,
                                fill=col_ticks, tags="clock_overlay")
        self.canvas.create_text(x6,  y6,  text=_fmt_mark(hBase / 2.0),  font=font_marks,
                                fill=col_ticks, tags="clock_overlay")
        self.canvas.create_text(x9,  y9,  text=_fmt_mark(3.0 * hBase / 4.0),  font=font_marks,
                                fill=col_ticks, tags="clock_overlay")

        # Aiguilles
        # Convention : angle 0° = 12h, sens horaire ; conversion vers coords canvas:
        #   x = cx + R * sin(theta), y = cy - R * cos(theta)
        def _end_point(angle_deg, length):
            import math
            a = math.radians(angle_deg)
            return (cx + length * math.sin(a), cy - length * math.cos(a))
        # Angles via decryptor (cohérence complète avec les bases minutes/heures)
        ang_hour_0, ang_min_0 = self.decryptor.anglesFromClock(hour=float(hFloat), minute=int(m))
        ang_min = ref_az + float(ang_min_0)
        # IMPORTANT: l'avance avec les minutes (ou non) est déjà encodée dans hFloat par le decryptor.
        ang_hour = ref_az + float(ang_hour_0)

        # Écart entre aiguilles (0..180) — même définition que l'angle d'arc (plus petit angle)
        # On repasse en [0..360) avant calcul.
        delta_needles_deg = None
        delta_needles_deg = self._clock_arc_compute_angle_deg(float(ang_hour) % 360.0, float(ang_min) % 360.0)

        # Longueurs des aiguilles
        L_min  = R * 0.86
        L_hour = R * 0.58
        x2m, y2m = _end_point(ang_min,  L_min)
        x2h, y2h = _end_point(ang_hour, L_hour)
        # traits
        self.canvas.create_line(cx, cy, x2h, y2h, width=3, fill=col_hour, tags="clock_overlay")
        self.canvas.create_line(cx, cy, x2m, y2m, width=2, fill=col_min,  tags="clock_overlay")
        # axe central
        self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill=col_min, outline=col_min, tags="clock_overlay")
        # Libellé sous l'horloge
        if label:
            label_disp = str(label)
            if delta_needles_deg is not None:
                label_disp = f"{label_disp} — Δ={float(delta_needles_deg):0.0f}°"
            # Si le filtrage dico est actif, afficher aussi l'azimut théorique du 12h (référence)
            # pour aligner les aiguilles sur les 2 droites mesurées (az1/az2).
            if getattr(self, "_dico_filter_active", False):
                last = getattr(self, "_clock_arc_last", None)
                if isinstance(last, dict) and ("az1" in last) and ("az2" in last):
                    ref_theo = self._clock_compute_theoretical_ref_azimuth_deg(
                        az1=float(last["az1"]),
                        az2=float(last["az2"]),
                        ang_hour_0=float(ang_hour_0),
                        ang_min_0=float(ang_min_0),
                    )
                    label_disp = f"{label_disp} — Ref={ref_theo:0.1f}°"
            self.canvas.create_text(cx, cy + R + 20, text=label_disp,
                                    font=("Arial", 11, "bold"), fill="#000000",
                                    anchor="n", tags="clock_overlay")
        # Dernière mesure d'arc (persistée) : affichage tant que le compas reste au même ancrage
        self._clock_arc_draw_last(cx, cy, R)


    # ---------- Horloge : snap sur sommet le plus proche ----------
    def _clock_clear_snap_target(self):
        """Efface le marqueur visuel du sommet 'target' pendant le drag du compas."""
        self._clock_snap_target = None
        if getattr(self, "canvas", None):
            self.canvas.delete("clock_snap_target")


    def _clock_update_snap_target(self, sx: float, sy: float):
        """Calcule le sommet de triangle le plus proche du curseur (en écran/canvas) et l'affiche en rouge."""
        prev = getattr(self, "_clock_snap_target", None)
        # Pas de triangles -> rien à viser
        if not getattr(self, "_last_drawn", None):
            self._clock_clear_snap_target()
            # Si on quitte un ancrage (plus de snap), on perd aussi l'arc persisté + dico
            if prev is not None and self._clock_arc_is_available():
                self._clock_arc_clear_last()
            return
        try:
            w = self._screen_to_world(sx, sy)
            v_world = np.array([float(w[0]), float(w[1])], dtype=float)
        except Exception:
            self._clock_clear_snap_target()
            if prev is not None and self._clock_arc_is_available():
                self._clock_arc_clear_last()
            return

        found = self._find_nearest_vertex(v_world, exclude_idx=None, exclude_gid=None)
        if found is None:
            self._clock_clear_snap_target()
            if prev is not None and self._clock_arc_is_available():
                self._clock_arc_clear_last()
            return

        idx, vkey, wbest = found
        # Si on change de noeud (idx/vkey), l'arc mesuré n'est plus valide => on reset arc + dico
        if prev is not None:
            prev_key = (int(prev.get("idx")), str(prev.get("vkey")))
            new_key = (int(idx), str(vkey))
            if prev_key is not None and new_key != prev_key and self._clock_arc_is_available():
                self._clock_arc_clear_last()
        self._clock_snap_target = {"idx": int(idx), "vkey": str(vkey), "world": np.array(wbest, dtype=float)}

        # Auto-mesure : si on vient de s'accrocher à un nouveau noeud,
        # on calcule automatiquement l'angle entre les 2 segments EXTÉRIEURS
        # incidents à ce noeud (sur le contour du groupe).
        try:
            prev_key = (int(prev.get("idx")), str(prev.get("vkey"))) if isinstance(prev, dict) else None
        except Exception:
            prev_key = None
        try:
            new_key = (int(idx), str(vkey))
        except Exception:
            new_key = None

        if new_key is not None and new_key != prev_key:
            self._clock_arc_auto_measure_from_snap()

        # Marqueur visuel : un anneau rouge autour du sommet
        if getattr(self, "canvas", None):
            self.canvas.delete("clock_snap_target")
            px, py = self._world_to_screen(wbest)
            r = 10  # rayon px fixe
            self.canvas.create_oval(px - r, py - r, px + r, py + r,
                                    outline="#FF0000", width=3,
                                    fill="", tags="clock_snap_target")
            self.canvas.tag_raise("clock_snap_target")


    def _clock_arc_auto_get_two_neighbors(self, idx: int, v_world, outline_eps=None):
        """Helper commun (compas) : retourne 2 voisins (w1, w2) sur le contour (outline)
        incident au point v_world.

        Objectif : éviter la duplication entre _clock_arc_auto_measure_from_snap() et
        _clock_arc_auto_from_snap_target().

        - outline_eps=None : utiliser l'outline par défaut (ne pas changer le comportement)
        - outline_eps=float : forcer un eps spécifique lors du calcul de l'outline

        Retourne (w1, w2) en coordonnées monde (np.array), ou None.
        """
        try:
            idx = int(idx)
            v_world = np.array(v_world, dtype=float)
        except Exception:
            return None
        if not (0 <= idx < len(self._last_drawn)):
            return None

        # Outline = contour du groupe auquel appartient idx
        if outline_eps is None:
            outline = self._outline_for_item(idx) or []
        else:
            outline = self._outline_for_item(idx, eps=float(outline_eps)) or []
        if not outline:
            return None

        # 2 segments extérieurs incidents à v_world
        g = self._build_boundary_graph(outline)
        tol_world = max(
            float(EPS_WORLD),
            float(getattr(self, "stroke_px", 2.0)) / max(float(getattr(self, "zoom", 1.0)), 1e-9)
        )
        inc = self._incident_half_edges_at_point(g, v_world, outline, eps=tol_world) or []
        if len(inc) < 2:
            return None

        try:
            w1 = np.array(inc[0][1], dtype=float)
            w2 = np.array(inc[1][1], dtype=float)
        except Exception:
            return None

        return (w1, w2)


    def _clock_arc_auto_measure_from_snap(self):
        """Si le compas est accroché à un noeud, calcule automatiquement l'angle (arc) entre
        les 2 arêtes **EXTÉRIEURES** incidentes à ce noeud (c'est-à-dire sur l'outline du groupe).
        La mesure est persistée dans self._clock_arc_last (comme la mesure manuelle)."""
        # NOTE: ici on calcule justement la mesure persistée.
        # Ne PAS dépendre de _clock_arc_is_available(), sinon la fonction ne se déclenche jamais.
        if not getattr(self, "canvas", None):
            return
        if not getattr(self, "show_clock_overlay", None) or not self.show_clock_overlay.get():
            return
        tgt = getattr(self, "_clock_snap_target", None)
        if not (isinstance(tgt, dict) and tgt.get("world") is not None):
            # Plus d'ancrage : ne pas garder une mesure invalide
            self._clock_arc_clear_last()
            return

        try:
            idx = int(tgt.get("idx"))
            v_world = np.array(tgt.get("world"), dtype=float)
        except Exception:
            return

        # IMPORTANT (compas) : on veut des sommets "tels quels" (pas l'eps de rendu),
        # sinon le buffer décale légèrement les sommets et v_world ne tombe plus sur un sommet du graphe.
        neigh = self._clock_arc_auto_get_two_neighbors(idx, v_world, outline_eps=EPS_WORLD)
        if not neigh:
            self._clock_arc_clear_last()
            return

        w1, w2 = neigh

        # Point d'ancrage = centre du compas = noeud snap
        cx, cy = self._world_to_screen(v_world)
        # Forcer le centre sur le noeud ancré (overlay cohérent)
        self._clock_cx = float(cx)
        self._clock_cy = float(cy)

        # Azimuts vers les 2 voisins (en écran)
        sx1, sy1 = self._world_to_screen(w1)
        sx2, sy2 = self._world_to_screen(w2)
        az1 = float(self._clock_compute_azimuth_deg(int(sx1), int(sy1)))
        az2 = float(self._clock_compute_azimuth_deg(int(sx2), int(sy2)))
        angle_deg = float(self._clock_arc_compute_angle_deg(az1, az2))

        # Persister : même format que la mesure manuelle
        self._clock_arc_last = {"az1": az1, "az2": az2, "angle": angle_deg}
        self._clock_arc_last_angle_deg = float(angle_deg)

        # Rafraîchir l'overlay (et donc le filtre dico s'il est activé)
        self._draw_clock_overlay()

        # Rafraîchir l'overlay + états UI (menu compas / dico)
        self._update_compass_ctx_menu_and_dico_state()
        self._redraw_overlay_only()
        self.status.config(text=f"Arc auto (EXT) : {angle_deg:0.0f}°")

    # ==========================
    # Calibration fond (3 points)
    # ==========================

    def _bg_calibrate_start(self):
        """Démarre un calibrage du fond en cliquant 3 points (villes) définis dans un fichier JSON.

        Fichier attendu (dans ../data/maps) :
            <nom_de_la_carte>.calib_points.json
        Exemple :
            {
              "points": [
                {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
                {"name": "Lyon",  "lat": 45.7640, "lon": 4.8357},
                {"name": "Marseille", "lat": 43.2965, "lon": 5.3698}
              ]
            }

        Résultat sauvegardé (dans ../data/maps) :
            <nom_de_la_carte>.json
        """
        if not self._bg or not self._bg.get("path"):
            messagebox.showwarning("Calibration", "Aucun fond SVG chargé. Charge d'abord une carte.")
            return

        svg_path = str(self._bg.get("path"))
        base = os.path.splitext(os.path.basename(svg_path))[0]
        cfg_path = os.path.join(self.maps_dir, f"{base}.calib_points.json")

        if not os.path.isfile(cfg_path):
            messagebox.showerror(
                "Calibration",
                "Fichier de points de calibration introuvable.\n\n"
                f"Attendu :\n  {cfg_path}\n\n"
                "Crée ce fichier avec 3 villes (name/lat/lon) puis relance."
            )
            return

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            messagebox.showerror("Calibration", f"Impossible de lire {cfg_path}\n\n{e}")
            return

        pts = cfg.get("points") if isinstance(cfg, dict) else None
        if not isinstance(pts, list) or len(pts) != 3:
            messagebox.showerror(
                "Calibration",
                "Le fichier de calibration doit contenir exactement 3 points :\n"
                '  {"points":[{"name":..,"lat":..,"lon":..}, ... x3]}'
            )
            return

        for i, p in enumerate(pts):
            if not isinstance(p, dict) or ("lat" not in p) or ("lon" not in p):
                messagebox.showerror(
                    "Calibration",
                    f"Point #{i+1} invalide : chaque point doit contenir au moins 'lat' et 'lon'."
                )
                return

        self._bg_calib_points_cfg = {"path": cfg_path, "points": pts, "svg": svg_path}
        self._bg_calib_clicked_world = []
        self._bg_calib_step = 0
        self._bg_calib_active = True

        name0 = str(pts[0].get("name") or "Point 1")
        self.status.config(
            text=(
                f"Calibration carte : CTRL+clic sur {name0} (1/3) | "
                "clic/drag = pan | molette = zoom | ESC = annuler"
            )
        )
        self.canvas.configure(cursor="crosshair")

    def _bg_calibrate_cancel(self):
        if not getattr(self, "_bg_calib_active", False):
            return
        self._bg_calib_active = False
        self._bg_calib_points_cfg = None
        self._bg_calib_clicked_world = []
        self._bg_calib_step = 0
        try:
            self.canvas.configure(cursor="")
        except Exception:
            pass
        self.status.config(text="Calibration annulée.")

    def _bg_calibrate_handle_click(self, event):
        """Enregistre un clic de calibration (en coordonnées monde)."""
        if not getattr(self, "_bg_calib_active", False):
            return "continue"

        # Pendant la calibration, on veut pouvoir pan/zoom pour viser précisément.
        # On valide donc un point UNIQUEMENT sur CTRL+clic.
        # Sans CTRL, le clic gauche redevient un "pan start".
        if not getattr(self, "_ctrl_down", False):
            self._on_pan_start(event)
            return "break"

        cfg = self._bg_calib_points_cfg or {}
        pts = cfg.get("points") or []
        if len(pts) != 3:
            self._bg_calibrate_cancel()
            return "break"

        try:
            w = self._screen_to_world(event.x, event.y)
        except Exception:
            return "break"

        self._bg_calib_clicked_world.append((float(w[0]), float(w[1])))
        self._bg_calib_step += 1

        if self._bg_calib_step < 3:
            nxt = pts[self._bg_calib_step]
            name = str(nxt.get("name") or f"Point {self._bg_calib_step+1}")
            self.status.config(text=f"Calibration carte : clique sur {name} ({self._bg_calib_step+1}/3) — ESC pour annuler.")
            # petit feedback visuel
            self._redraw_from(self._last_drawn)
            return "break"

        # 3 clics : calcul + sauvegarde
        try:
            self._bg_calibrate_finish()
        except Exception as e:
            messagebox.showerror("Calibration", f"Échec du calibrage :\n\n{e}")
            self._bg_calibrate_cancel()
            return "break"

        self._bg_calib_active = False
        try:
            self.canvas.configure(cursor="")
        except Exception:
            pass
        self._redraw_from(self._last_drawn)
        return "break"

    def _bg_calibrate_finish(self):
        cfg = self._bg_calib_points_cfg or {}
        pts = cfg.get("points") or []
        svg_path = cfg.get("svg") or (self._bg.get("path") if self._bg else None)
        if len(pts) != 3 or len(self._bg_calib_clicked_world) != 3 or not svg_path:
            raise RuntimeError("Données de calibration incomplètes.")

        # GPS -> Lambert93 (m) -> km
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
        lambert_km = []
        for p in pts:
            lon = float(p["lon"])
            lat = float(p["lat"])
            x_m, y_m = transformer.transform(lon, lat)
            lambert_km.append((x_m / 1000.0, y_m / 1000.0))

        world = self._bg_calib_clicked_world

        # Résolution affine : (xw,yw) -> (xl,yl)
        # xl = a*xw + b*yw + c
        # yl = d*xw + e*yw + f
        A = []
        B = []
        for (xw, yw), (xl, yl) in zip(world, lambert_km):
            A.append([xw, yw, 1, 0, 0, 0])
            A.append([0, 0, 0, xw, yw, 1])
            B.append(xl)
            B.append(yl)

        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        params = np.linalg.solve(A, B)  # 6 paramètres exacts (3 points)
        a, b, c, d, e, f = [float(v) for v in params.tolist()]

        det = a * e - b * d
        if abs(det) < 1e-12:
            raise RuntimeError("Points de calibration dégénérés (transformée non inversible).")

        # inverse (Lambert km -> monde)
        inv_a = e / det
        inv_b = -b / det
        inv_d = -d / det
        inv_e = a / det
        inv_c = -(inv_a * c + inv_b * f)
        inv_f = -(inv_d * c + inv_e * f)

        # Sauvegarde
        base = os.path.splitext(os.path.basename(str(svg_path)))[0]
        out_path = os.path.join(self.maps_dir, f"{base}.json")

        payload = {
            "type": "bg_calibration_3points",
            "date": _dt.datetime.now().isoformat(timespec="seconds"),
            "svgPath": str(svg_path),
            "bgWorldRectAtCalibration": (
                {
                    "x0": float(self._bg.get("x0")),
                    "y0": float(self._bg.get("y0")),
                    "w":  float(self._bg.get("w")),
                    "h":  float(self._bg.get("h")),
                }
                if (self._bg is not None and all(k in self._bg for k in ("x0","y0","w","h")))
                else None
            ),
            "points": [
                {
                    "name": str(p.get("name") or f"Point {i+1}"),
                    "lat": float(p["lat"]),
                    "lon": float(p["lon"]),
                    "lambert93Km": [lambert_km[i][0], lambert_km[i][1]],
                    "worldClicked": [world[i][0], world[i][1]],
                }
                for i, p in enumerate(pts)
            ],
            "affineWorldToLambertKm": [a, b, c, d, e, f],
            "affineLambertKmToWorld": [inv_a, inv_b, inv_c, inv_d, inv_e, inv_f],
        }

        os.makedirs(self.maps_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(payload, f_out, ensure_ascii=False, indent=2)

        self.status.config(text=f"Calibration enregistrée : {out_path}")

    # =========================
    # Fond SVG en coordonnées monde
    # =========================

    def _bg_clear(self):
        self._bg = None
        self._bg_base_pil = None
        self._bg_photo = None
        self._bg_resizing = None
        self._bg_calib_data = None
        self._bg_scale_base_w = None

        self._persistBackgroundConfig()
        self._redraw_from(self._last_drawn)

    def _bg_try_load_calibration(self, svg_path: str):
        """Charge data/maps/<carte>.json si présent (calibration 3 points).

        Objectif: disposer de affineLambertKmToWorld pour convertir (Lambert93 km) -> coordonnées monde.
        """
        try:
            base = os.path.splitext(os.path.basename(str(svg_path)))[0]
            calib_path = os.path.join(self.maps_dir, f"{base}.json")
            if not os.path.isfile(calib_path):
                self._bg_calib_data = None
                self._bg_scale_base_w = None
                return

            with open(calib_path, "r", encoding="utf-8") as f_in:
                data = json.load(f_in)

            aff = data.get("affineLambertKmToWorld")
            if not (isinstance(aff, list) and len(aff) == 6):
                self._bg_calib_data = None
                return

            # Normaliser en float
            data["affineLambertKmToWorld"] = [float(v) for v in aff]
            self._bg_calib_data = data

            # Référence d'échelle pour l'affichage :
            # - si le JSON contient la géométrie monde du fond au moment de la calibration,
            #   on s'y réfère (=> le "x1" est stable et cohérent entre sessions)
            # - sinon, fallback : largeur actuelle au moment du chargement.
            try:
                rect = data.get("bgWorldRectAtCalibration") if isinstance(data, dict) else None
                if isinstance(rect, dict) and ("w" in rect):
                    self._bg_scale_base_w = float(rect.get("w"))
                elif self._bg is not None:
                    self._bg_scale_base_w = float(self._bg.get("w"))
                else:
                    self._bg_scale_base_w = None
            except Exception:
                self._bg_scale_base_w = None
        except Exception:
            # best-effort
            self._bg_calib_data = None
            self._bg_scale_base_w = None

    def _bg_load_svg_dialog(self):
        path = filedialog.askopenfilename(
            title="Choisir une carte (SVG/PNG/JPG)",
            initialdir=getattr(self, "maps_dir", None) or None,
            filetypes=[
                ("Cartes", "*.svg *.png *.jpg *.jpeg"),
                ("SVG", "*.svg"),
                ("PNG", "*.png"),
                ("JPG", "*.jpg *.jpeg"),
                ("Tous fichiers", "*.*"),
            ],
        )
        if not path:
            return
        self._bg_set_map(path)

    def _bg_set_map(self, path: str, rect_override: dict | None = None, persist: bool = True):
        """Charge une carte (fond) depuis un fichier .svg ou une image raster (.png/.jpg/.jpeg).

        - SVG : rasterisation (svglib/reportlab/pypdfium2) comme avant
        - PNG/JPG/JPEG : chargement direct via Pillow

        Note : on conserve les clés de config 'bgSvgPath/bgWorldRect' pour compatibilité.
        """
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".svg":
            return self._bg_set_svg(path, rect_override=rect_override, persist=persist)
        if ext in (".png", ".jpg", ".jpeg"):
            return self._bg_set_png(path, rect_override=rect_override, persist=persist)

        messagebox.showerror(
            "Format non supporté",
            f"Carte non supportée: {path}\nFormats acceptés: .svg, .png, .jpg, .jpeg"
        )

    def _bg_set_png(self, png_path: str, rect_override: dict | None = None, persist: bool = True):
        if (Image is None or ImageTk is None):
            messagebox.showerror(
                "Dépendances manquantes",
                "Pour afficher un fond PNG, installe :\n"
                "  - pillow\n\n"
                "pip install pillow"
            )
            return

        try:
            pil0 = Image.open(png_path).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Erreur image", f"Impossible de charger l'image :\n{e}")
            return

        # aspect depuis l'image
        try:
            w0, h0 = pil0.size
            aspect = float(w0) / float(h0) if h0 else 1.0
            aspect = max(1e-6, aspect)
        except Exception:
            aspect = 1.0

        # base normalisée : max 4096 sur le plus grand côté (comme SVG)
        try:
            max_dim = 4096
            w0, h0 = pil0.size
            if w0 <= 0 or h0 <= 0:
                raise ValueError("image vide")
            if max(w0, h0) > max_dim:
                if w0 >= h0:
                    W = int(max_dim)
                    H = max(1, int(round(W / aspect)))
                else:
                    H = int(max_dim)
                    W = max(1, int(round(H * aspect)))
                pil0 = pil0.resize((W, H), Image.LANCZOS)
        except Exception:
            pass

        # Si on a une géométrie sauvegardée (monde), on la réapplique telle quelle
        if isinstance(rect_override, dict) and all(k in rect_override for k in ("x0", "y0", "w", "h")):
            try:
                x0 = float(rect_override.get("x0", 0.0))
                y0 = float(rect_override.get("y0", 0.0))
                w  = float(rect_override.get("w", 1.0))
                h  = float(rect_override.get("h", 1.0))
                if w > 1e-9 and h > 1e-9:
                    self._bg = {"path": png_path, "x0": x0, "y0": y0, "w": w, "h": h, "aspect": aspect}
                    self._bg_base_pil = pil0
                    self._bg_try_load_calibration(png_path)
                    if self._bg_calib_data is not None:
                        print(f"[BG] Calibration chargée : {os.path.splitext(os.path.basename(str(png_path)))[0]}.json")

                    if persist:
                        self._persistBackgroundConfig()

                    self._redraw_from(self._last_drawn)

                    if persist:
                        print(f"[Fond image] Fichier chargé: {os.path.basename(str(png_path))} ({png_path})")
                    return
            except Exception:
                pass

        # Position/taille initiale : calée sur bbox des triangles si dispo, sinon vue écran
        if self._last_drawn:
            xs, ys = [], []
            for t in self._last_drawn:
                P = t["pts"]
                for k in ("O", "B", "L"):
                    xs.append(float(P[k][0])); ys.append(float(P[k][1]))
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
        else:
            cw = max(2, int(self.canvas.winfo_width() or 2))
            ch = max(2, int(self.canvas.winfo_height() or 2))
            x0, yTop = self._screen_to_world(0, 0)
            x1, yBot = self._screen_to_world(cw, ch)
            xmin, xmax = (min(x0, x1), max(x0, x1))
            ymin, ymax = (min(yBot, yTop), max(yBot, yTop))

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        bw = max(1e-6, (xmax - xmin) * 1.10)
        bh = max(1e-6, (ymax - ymin) * 1.10)

        # Ajuster pour conserver le ratio
        if (bw / bh) > aspect:
            w = bw
            h = w / aspect
        else:
            h = bh
            w = h * aspect

        self._bg = {"path": png_path, "x0": cx - w/2, "y0": cy - h/2, "w": w, "h": h, "aspect": aspect}
        self._bg_base_pil = pil0

        self._bg_try_load_calibration(png_path)
        if self._bg_calib_data is not None:
            print(f"[BG] Calibration chargée : {os.path.splitext(os.path.basename(str(png_path)))[0]}.json")

        if persist:
            self._persistBackgroundConfig()
            print(f"[Fond PNG] Fichier chargé: {os.path.basename(str(png_path))} ({png_path})")

        self._redraw_from(self._last_drawn)

    def _bg_set_svg(self, svg_path: str, rect_override: dict | None = None, persist: bool = True):
        if (Image is None or ImageTk is None or svg2rlg is None
                or renderPDF is None or _rl_canvas is None or pdfium is None):
            messagebox.showerror(
                "Dépendances manquantes",
                "Pour afficher un fond SVG (svglib/reportlab, sans Cairo), installe :\n"
                "  - pillow\n  - svglib\n  - reportlab\n  - pypdfium2\n\n"
                "pip install pillow svglib reportlab pypdfium2"
            )
            return

        aspect = self._bg_parse_aspect(svg_path)
        # Si on a une géométrie sauvegardée (monde), on la réapplique telle quelle
        if isinstance(rect_override, dict) and all(k in rect_override for k in ("x0", "y0", "w", "h")):
            try:
                x0 = float(rect_override.get("x0", 0.0))
                y0 = float(rect_override.get("y0", 0.0))
                w  = float(rect_override.get("w", 1.0))
                h  = float(rect_override.get("h", 1.0))
                if w > 1e-9 and h > 1e-9:
                    self._bg = {"path": svg_path, "x0": x0, "y0": y0, "w": w, "h": h, "aspect": aspect}
                    # Raster base "normalisée" (taille fixe, ratio respecté)
                    self._bg_base_pil = self._bg_render_base(svg_path, aspect, max_dim=4096)
                    if self._bg_base_pil is not None:
                        print(f"[Fond SVG] Fichier chargé: {os.path.basename(str(svg_path))} ({svg_path})")
                    # calibration associée (si fichier data/<carte>.json existe)
                    self._bg_try_load_calibration(svg_path)
                    if self._bg_calib_data is not None:
                        print(f"[BG] Calibration chargée : {os.path.basename(str(svg_path))}.json")

                    if persist:
                        self._persistBackgroundConfig()
                    self._redraw_from(self._last_drawn)
                    return
            except Exception:
                pass

        # Position/taille initiale : calée sur bbox des triangles si dispo, sinon vue écran
        if self._last_drawn:
            xs, ys = [], []
            for t in self._last_drawn:
                P = t["pts"]
                for k in ("O", "B", "L"):
                    xs.append(float(P[k][0])); ys.append(float(P[k][1]))
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
        else:
            # bbox monde visible
            cw = max(2, int(self.canvas.winfo_width() or 2))
            ch = max(2, int(self.canvas.winfo_height() or 2))
            x0, yTop = self._screen_to_world(0, 0)
            x1, yBot = self._screen_to_world(cw, ch)
            xmin, xmax = (min(x0, x1), max(x0, x1))
            ymin, ymax = (min(yBot, yTop), max(yBot, yTop))

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        bw = max(1e-6, (xmax - xmin) * 1.10)
        bh = max(1e-6, (ymax - ymin) * 1.10)

        # Ajuster pour conserver le ratio du SVG
        if (bw / bh) > aspect:
            w = bw
            h = w / aspect
        else:
            h = bh
            w = h * aspect

        self._bg = {"path": svg_path, "x0": cx - w/2, "y0": cy - h/2, "w": w, "h": h, "aspect": aspect}

        # Raster base "normalisée" (taille fixe, ratio respecté)
        self._bg_base_pil = self._bg_render_base(svg_path, aspect, max_dim=4096)

        # calibration associée (si fichier data/<carte>.json existe)
        if self._bg_calib_data is not None:
            print(f"[BG] Calibration chargée : {os.path.basename(str(svg_path))}.json")
        self._bg_try_load_calibration(svg_path)

        if persist:
            self._persistBackgroundConfig()
            if self._bg_base_pil is not None:
                print(f"[Fond SVG] Fichier chargé: {os.path.basename(str(svg_path))} ({svg_path})")

        self._redraw_from(self._last_drawn)

    def _bg_parse_aspect(self, svg_path: str) -> float:
        try:
            s = open(svg_path, "r", encoding="utf-8", errors="ignore").read(8192)
        except Exception:
            return 1.0

        # viewBox="minx miny width height"
        m = re.search(r'viewBox\s*=\s*["\']\s*([-\d\.eE]+)\s+([-\d\.eE]+)\s+([-\d\.eE]+)\s+([-\d\.eE]+)\s*["\']', s)
        if m:
            vw = float(m.group(3)); vh = float(m.group(4))
            if abs(vh) > 1e-12:
                return max(1e-6, vw / vh)

        # width="xxx" height="yyy" (units possible)
        mw = re.search(r'width\s*=\s*["\']\s*([-\d\.eE]+)', s)
        mh = re.search(r'height\s*=\s*["\']\s*([-\d\.eE]+)', s)
        if mw and mh:
            w = float(mw.group(1)); h = float(mh.group(1))
            if abs(h) > 1e-12:
                return max(1e-6, w / h)

        return 1.0

    def _bg_render_base(self, svg_path: str, aspect: float, max_dim: int = 4096):
        # base normalisée : max_dim sur le plus grand côté
        if aspect >= 1.0:
            W = int(max_dim)
            H = max(1, int(round(W / aspect)))
        else:
            H = int(max_dim)
            W = max(1, int(round(H * aspect)))

        try:
            drawing = svg2rlg(svg_path) if svg2rlg is not None else None
            if drawing is None:
                raise RuntimeError("svg2rlg() a retourné None")

            # --- IMPORTANT : éviter le SVG tronqué ---
            # svglib peut produire un Drawing dont le contenu a un bounding-box décalé (xmin/ymin != 0),
            # ou des width/height non représentatifs. On normalise sur getBounds() quand c'est possible.
            xmin = ymin = 0.0
            bw = bh = 0.0
            try:
                if hasattr(drawing, "getBounds"):
                    b = drawing.getBounds()
                    if b and len(b) == 4:
                        xmin, ymin, xmax, ymax = map(float, b)
                        bw = max(0.0, xmax - xmin)
                        bh = max(0.0, ymax - ymin)
            except Exception:
                bw = bh = 0.0

            # Dimensions de référence : d'abord le bounding-box, sinon width/height svglib
            dw = float(bw) if bw > 1e-9 else float(getattr(drawing, "width", 0) or 0)
            dh = float(bh) if bh > 1e-9 else float(getattr(drawing, "height", 0) or 0)
            if dw <= 0 or dh <= 0:
                # fallback: utiliser le ratio calculé, avec une base arbitraire
                dw = float(W)
                dh = float(H)

            # Si le contenu est décalé (xmin/ymin), on le ramène à l'origine avant le scale.
            if bw > 1e-9 and bh > 1e-9:
                try:
                    if hasattr(drawing, "translate"):
                        drawing.translate(-xmin, -ymin)
                except Exception:
                    pass

            # svglib exprime en "points" (1/72 inch). En rasterisant à 72dpi,
            # 1 point ~= 1 pixel. On scale le dessin puis on force le canvas aux dims voulues.
            sx = float(W) / dw
            sy = float(H) / dh
            # On applique les deux échelles pour respecter EXACTEMENT W/H (le ratio vient déjà de 'aspect')
            try:
                drawing.scale(sx, sy)
            except Exception:
                # certains objets retournés par svg2rlg peuvent ne pas exposer scale()
                pass
            try:
                drawing.width = float(W)
                drawing.height = float(H)
            except Exception:
                pass

            # Rasterisation sans Cairo : SVG -> PDF (reportlab) -> bitmap (pypdfium2)
            if renderPDF is None or _rl_canvas is None or pdfium is None:
                raise RuntimeError("Rasterisation SVG indisponible : pip install svglib reportlab pypdfium2")

            buf = io.BytesIO()
            c = _rl_canvas.Canvas(buf, pagesize=(W, H))
            renderPDF.draw(drawing, c, 0, 0)
            c.showPage()
            c.save()
            pdf_bytes = buf.getvalue()

            doc = pdfium.PdfDocument(pdf_bytes)
            try:
                page = doc[0]
                bitmap = page.render(scale=1)  # ~72 dpi : 1 point ≈ 1 pixel
                pil = bitmap.to_pil().convert("RGBA")
            finally:
                try:
                    doc.close()
                except Exception:
                    pass

            if pil.size != (W, H):
                pil = pil.resize((W, H), Image.LANCZOS)
            return pil
        except Exception as e:
            messagebox.showerror("Erreur SVG", f"Impossible de rasteriser le SVG (svglib/reportlab):\n{e}")
            return None

    def _bg_draw_world_layer(self):
        """Dessine le fond en 'monde' : on recadre la base en fonction pan/zoom et on l'affiche en plein canvas."""
        if not self._bg or self._bg_base_pil is None or Image is None or ImageTk is None:
            return

        cw = int(self.canvas.winfo_width() or 0)
        ch = int(self.canvas.winfo_height() or 0)
        if cw <= 2 or ch <= 2:
            try:
                self.update_idletasks()
                cw = int(self.canvas.winfo_width() or 0)
                ch = int(self.canvas.winfo_height() or 0)
            except Exception:
                return
        if cw <= 2 or ch <= 2:
            return

        bx0 = float(self._bg["x0"]); by0 = float(self._bg["y0"])
        bw = float(self._bg["w"]);  bh = float(self._bg["h"])
        bx1 = bx0 + bw; by1 = by0 + bh

        # Vue monde actuelle (canvas entier)
        xA, yTop = self._screen_to_world(0, 0)
        xB, yBot = self._screen_to_world(cw, ch)
        vx0, vx1 = (min(xA, xB), max(xA, xB))
        vy0, vy1 = (min(yBot, yTop), max(yBot, yTop))

        # Intersection vue <-> fond
        ix0 = max(vx0, bx0); ix1 = min(vx1, bx1)
        iy0 = max(vy0, by0); iy1 = min(vy1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            return

        baseW, baseH = self._bg_base_pil.size

        # Crop dans l'image base
        left  = int((ix0 - bx0) / bw * baseW)
        right = int((ix1 - bx0) / bw * baseW)
        # y base : 0 en haut, donc on inverse
        upper = int((by1 - iy1) / bh * baseH)  # iy1 = top
        lower = int((by1 - iy0) / bh * baseH)  # iy0 = bottom

        # clamp
        left = max(0, min(baseW-1, left))
        right = max(left+1, min(baseW, right))
        upper = max(0, min(baseH-1, upper))
        lower = max(upper+1, min(baseH, lower))

        crop = self._bg_base_pil.crop((left, upper, right, lower))

        # Où coller sur l'écran
        sx0, syTop = self._world_to_screen((ix0, iy1))
        sx1, syBot = self._world_to_screen((ix1, iy0))
        wpx = int(round(sx1 - sx0))
        hpx = int(round(syBot - syTop))
        if wpx <= 1 or hpx <= 1:
            return

        crop = crop.resize((wpx, hpx), Image.LANCZOS)

        # IMPORTANT: fond blanc pour éviter un rendu gris quand la carte est semi-transparente
        # (Tk peut composer les pixels transparents sur un fond non-blanc selon la plateforme).
        out = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
        px = int(round(sx0)); py = int(round(syTop))

        # clip paste
        paste_x0 = max(0, px)
        paste_y0 = max(0, py)
        paste_x1 = min(cw, px + wpx)
        paste_y1 = min(ch, py + hpx)
        if paste_x1 <= paste_x0 or paste_y1 <= paste_y0:
            return

        src_x0 = paste_x0 - px
        src_y0 = paste_y0 - py
        src_x1 = src_x0 + (paste_x1 - paste_x0)
        src_y1 = src_y0 + (paste_y1 - paste_y0)

        crop2 = crop.crop((src_x0, src_y0, src_x1, src_y1))
        # Appliquer l'opacité utilisateur (0..100) sur l'alpha du fond
        op = int(float(self.map_opacity.get()))

        op = max(0, min(100, op))
        if op <= 0:
            return
        if op < 100:
            if crop2.mode != "RGBA":
                crop2 = crop2.convert("RGBA")
            r, g, b, a = crop2.split()
            a = a.point(lambda p: int(p * op / 100))
            crop2.putalpha(a)
        out.paste(crop2, (paste_x0, paste_y0), crop2)

        self._bg_photo = ImageTk.PhotoImage(out)
        self.canvas.create_image(0, 0, anchor="nw", image=self._bg_photo, tags=("bg_world",))
        self.canvas.tag_lower("bg_world")

    def _bg_corners_world(self):
        if not self._bg:
            return None
        x0 = float(self._bg["x0"]); y0 = float(self._bg["y0"])
        w = float(self._bg["w"]);  h = float(self._bg["h"])
        return {
            "bl": (x0,     y0),
            "br": (x0 + w, y0),
            "tl": (x0,     y0 + h),
            "tr": (x0 + w, y0 + h),
        }

    def _bg_corners_screen(self):
        c = self._bg_corners_world()
        if not c:
            return None
        return {k: self._world_to_screen(v) for k, v in c.items()}

    def _bg_draw_resize_handles(self):
        if not self._bg or not self.bg_resize_mode.get():
            return
        c = self._bg_corners_screen()
        if not c:
            return
        tl = c["tl"]; br = c["br"]

        self.canvas.create_rectangle(tl[0], tl[1], br[0], br[1], outline="gray30", dash=(3, 2), width=1, tags=("bg_ui",))

        r = 6
        for k in ("tl", "tr", "bl", "br"):
            x, y = c[k]
            self.canvas.create_rectangle(x-r, y-r, x+r, y+r, outline="gray10", fill="white", width=1, tags=("bg_ui",))

    def _bg_hit_test_handle(self, sx: float, sy: float):
        c = self._bg_corners_screen()
        if not c:
            return None
        r = 8
        for k in ("tl", "tr", "bl", "br"):
            x, y = c[k]
            if (sx - x)*(sx - x) + (sy - y)*(sy - y) <= r*r:
                return k
        return None

    def _bg_start_resize(self, handle: str, sx: int, sy: int):
        # handle: tl,tr,bl,br ; corner opposée fixe
        opp = {"tl": "br", "br": "tl", "tr": "bl", "bl": "tr"}[handle]
        corners = self._bg_corners_world()
        fx, fy = corners[opp]
        mx, my = self._screen_to_world(sx, sy)
        self._bg_resizing = {
            "handle": handle,
            "fixed": (fx, fy),
            "start_mouse": (mx, my),
            "start_rect": (float(self._bg["x0"]), float(self._bg["y0"]), float(self._bg["w"]), float(self._bg["h"])),
        }

    def _bg_start_move(self, sx: int, sy: int):
        """Démarre un déplacement du fond (mode resize actif mais pas sur une poignée)."""
        if not self._bg:
            return
        mx, my = self._screen_to_world(sx, sy)
        self._bg_moving = {
            "start_mouse": (float(mx), float(my)),
            "start_xy": (float(self._bg.get("x0", 0.0)), float(self._bg.get("y0", 0.0))),
        }

    def _bg_update_move(self, sx: int, sy: int):
        if not getattr(self, "_bg_moving", None) or not self._bg:
            return
        mx, my = self._screen_to_world(sx, sy)
        smx, smy = self._bg_moving["start_mouse"]
        x0, y0 = self._bg_moving["start_xy"]
        dx = float(mx - smx)
        dy = float(my - smy)
        self._bg["x0"] = float(x0 + dx)
        self._bg["y0"] = float(y0 + dy)

    def _bg_update_resize(self, sx: int, sy: int):
        if not self._bg_resizing or not self._bg:
            return
        aspect = float(self._bg["aspect"])
        fx, fy = self._bg_resizing["fixed"]
        mx, my = self._screen_to_world(sx, sy)

        dx = mx - fx
        dy = my - fy
        w0 = abs(dx)
        h0 = abs(dy)
        if w0 < 1e-6 or h0 < 1e-6:
            return

        # Conserver ratio : choisir la dominante (horizontal vs vertical)
        if (w0 / h0) > aspect:
            w = w0
            h = w / aspect
        else:
            h = h0
            w = h * aspect

        w = max(1e-3, w)
        h = max(1e-3, h)

        # Recomposer x0/y0 selon le quadrant (fixed est la corner opposée)
        # On place le rectangle de sorte que fixed reste fixe
        x0 = fx if dx >= 0 else (fx - w)
        y0 = fy if dy >= 0 else (fy - h)

        self._bg["x0"] = float(x0)
        self._bg["y0"] = float(y0)
        self._bg["w"]  = float(w)
        self._bg["h"]  = float(h)

        # Afficher l'échelle relative (x1, x1/3.15, x2.49...) pendant le resize
        self._bg_update_scale_status()

    def _bg_compute_scale_factor(self) -> float | None:
        """Retourne le scale *carte vs triangles*.

        On veut comparer :
          - l'échelle "monde" des triangles (1 unité monde == 1 km)
          - l'échelle "monde" implicite de la carte via la calibration 3 points.

        La calibration fournit affineLambertKmToWorld (km -> monde) :
            xw = a*xkm + b*ykm + c
            yw = d*xkm + e*ykm + f

        Les colonnes (a,d) et (b,e) donnent la taille en unités monde pour 1 km (Est / Nord).
        On moyenne les deux normes pour obtenir un facteur global.

        Si le fond a été redimensionné depuis la calibration, on applique le ratio (w_cur / w_ref).
        """
        if not self._bg:
            return None

        data = getattr(self, "_bg_calib_data", None)
        if not (isinstance(data, dict) and "affineLambertKmToWorld" in data):
            return None

        aff = data.get("affineLambertKmToWorld")
        if not (isinstance(aff, list) and len(aff) == 6):
            return None

        try:
            a, b, c, d, e, f = [float(v) for v in aff]
        except Exception:
            return None

        # Taille en unités monde pour 1 km (axes Est / Nord)
        import math
        norm_e = math.hypot(a, d)
        norm_n = math.hypot(b, e)
        base_norm = 0.5 * (norm_e + norm_n)

        if base_norm <= 1e-12:
            return None

        # Ratio de redimensionnement du fond (si on a une référence)
        ratio = 1.0
        if self._bg_scale_base_w is not None:
            try:
                base_w = float(self._bg_scale_base_w)
                cur_w = float(self._bg.get("w"))
                if base_w > 1e-12 and cur_w > 1e-12:
                    ratio = cur_w / base_w
            except Exception:
                ratio = 1.0

        return base_norm * ratio

    def _bg_format_scale(self, s: float | None) -> str:
        if s is None:
            return "x?"
        if abs(s - 1.0) < 1e-3:
            return "x1"
        if s >= 1.0:
            return f"x{s:.2f}"
        # plus petit que la référence -> x1/k
        k = 1.0 / max(1e-12, s)
        return f"x1/{k:.2f}"

    def _bg_update_scale_status(self):
        # On n'affiche l'échelle que si le mode redimensionnement est actif.
        if not self.bg_resize_mode.get() or not self._bg:
            return
        s = self._bg_compute_scale_factor()
        self.status.config(text=f"Échelle carte : {self._bg_format_scale(s)}")

    def _world_to_screen(self, p):
        x = self.offset[0] + float(p[0]) * self.zoom
        y = self.offset[1] - float(p[1]) * self.zoom
        return x, y

    def worldFromLambertKm(self, x_km: float, y_km: float):
        """Convertit un point (Lambert93, en km) -> coordonnées monde (celles de la scène).

        Nécessite que la calibration 3 points associée au fond soit chargée (data/<carte>.json).
        """
        data = getattr(self, "_bg_calib_data", None)
        if not data or not isinstance(data, dict) or "affineLambertKmToWorld" not in data:
            raise RuntimeError("Calibration fond absente : lance d'abord la calibration (3 points) ou charge le fichier .json associé à la carte.")

        aff = data.get("affineLambertKmToWorld", [])
        if not (isinstance(aff, list) and len(aff) == 6):
            raise RuntimeError("Calibration fond invalide : affineLambertKmToWorld manquant/incomplet (6 paramètres attendus).")
        a, b, c, d, e, f = [float(v) for v in aff]
        xw = a * float(x_km) + b * float(y_km) + c
        yw = d * float(x_km) + e * float(y_km) + f
        return (xw, yw)

    def _fit_to_view(self, placed):
        if not placed:
            return
        xs, ys = [], []
        for t in placed:
            P = t["pts"]
            for k in ("O", "B", "L"):
                xs.append(float(P[k][0]))
                ys.append(float(P[k][1]))
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        w, h = maxx - minx, maxy - miny
        if w <= 0 or h <= 0:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        margin = 40
        zx = (cw - 2 * margin) / w
        zy = (ch - 2 * margin) / h
        self.zoom = max(0.1, min(zx, zy))
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        self.offset = np.array(
            [cw / 2.0 - cx * self.zoom, ch / 2.0 + cy * self.zoom],
            dtype=float
        )
        # redraw for fit
        self._redraw_from(self._last_drawn)

    def _redraw_from(self, placed):
        try:
            self._update_current_scenario_differences()
        except Exception:
            self._comparison_diff_indices = set()
        self.canvas.delete("all")
        # Fond carte (si layer visible)
        if getattr(self, "show_map_layer", None) is None or self.show_map_layer.get():
            self._bg_draw_world_layer()     

        # l'ID de la ligne n'est plus valide après delete("all")
        self._nearest_line_id = None
        # on efface les IDs de surlignage déjà dessinés (mais on conserve le choix et les données)
        self._clear_edge_highlights()

        # Mode "contours uniquement" : même visualisation (noeuds, tags, numéros, mot),
        # mais SANS les arêtes internes. En plus, on trace l'enveloppe extérieure des groupes.
        showContoursMode = bool(
            getattr(self, "show_only_group_contours", None) is not None
            and self.show_only_group_contours.get()
        )

        onlyContours = False
        v = getattr(self, "only_group_contours", None)
        if v is not None and hasattr(v, "get"):
            onlyContours = bool(v.get())
        else:
            onlyContours = bool(getattr(self, "_only_group_contours", False))

        # Si on est en mode "contour only", on force la suppression des arêtes internes.
        if showContoursMode:
            onlyContours = True

        # 1) Triangles (toujours dessinés si layer actif) : on coupe juste les arêtes internes.
        if getattr(self, "show_triangles_layer", None) is None or self.show_triangles_layer.get():
            for i, t in enumerate(placed):
                labels = t["labels"]
                P = t["pts"]
                tri_id = t.get("id")
                fill = "#ffd6d6" if i in getattr(self, "_comparison_diff_indices", set()) else None
                self._draw_triangle_screen(
                    P,
                    labels=[f"O:{labels[0]}", f"B:{labels[1]}", f"L:{labels[2]}"],
                    tri_id=tri_id,
                    tri_mirrored=t.get("mirrored", False),
                    fill=fill,
                    diff_outline=bool(fill),
                    drawEdges=(not onlyContours),
                )

        # 2) Contour des groupes par-dessus (lisible), si demandé.
        if showContoursMode:
            self._draw_group_outlines()

        if not showContoursMode:
            # — Recrée l'aide visuelle UNIQUEMENT si une sélection 'vertex' est encore active —
            if self._sel and self._sel.get("mode") == "vertex":
                # redessine candidates + best si on a des données
                if getattr(self, "_edge_highlights", None):
                    self._redraw_edge_highlights()
                # remet la ligne grise
                idx = self._sel["idx"]; vkey = self._sel["vkey"]
                P = self._last_drawn[idx]["pts"]
                v_world = np.array(P[vkey], dtype=float)
                self._update_nearest_line(v_world, exclude_idx=idx)
            else:
                # pas de sélection active -> pas d'aides persistantes
                self._edge_highlights = None
                self._edge_choice = None

        # Poignées de redimensionnement fond (overlay UI)
        self._bg_draw_resize_handles()
        # Redessiner l'horloge (overlay indépendant)
        self._draw_clock_overlay()
        # Après tout redraw, le cache de pick n'est plus valide
        self._invalidate_pick_cache()

    def _draw_group_outlines(self):
        """Dessine uniquement le contour de chaque groupe (enveloppe extérieure)."""
        self.canvas.delete("group_outline")

        # Si aucun groupe n'est défini, on n'a rien de spécial à tracer.
        if not getattr(self, "groups", None):
            return

        for gid, g in self.groups.items():
            nodes = g.get("nodes") or []
            if not nodes:
                continue
            outline = self._group_outline_segments(gid)
            for p1, p2 in outline:
                x1, y1 = self._world_to_screen(p1)
                x2, y2 = self._world_to_screen(p2)
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill="#000000",
                    width=3,
                    tags=("group_outline",),
                )

    def _draw_triangle_screen(self, P, outline="black", width=2, labels=None, inset=0.35, tri_id=None, tri_mirrored=False, fill=None, diff_outline=False, drawEdges=True):
        """
        P : dict {'O','B','L'} en coordonnées monde (np.array 2D)
        labels : liste de 3 strings pour O,B,L (facultatif)
        inset : 0..1, fraction du chemin du sommet vers le barycentre pour placer le texte
        """
        # 1) coords monde -> écran
        pts_world = [P["O"], P["B"], P["L"]]
        coords = []
        for pt in pts_world:
            sx, sy = self._world_to_screen(pt)
            coords += [sx, sy]

        # 1b) remplissage optionnel (comparaison avec le scénario de référence)
        if fill:
            self.canvas.create_polygon(coords, fill=fill, outline="")
        if diff_outline:
            self.canvas.create_polygon(coords, outline="#ff0000", width=4, fill="")

        # 2) tracé du triangle (arêtes colorées selon les sommets)
        #    - Lumière (L) -> Ouverture (O) : noir
        #    - Lumière (L) -> Base (B)      : bleu foncé
        #    - Base (B) -> Ouverture (O)    : gris
        Ox, Oy, Bx, By, Lx, Ly = coords
        if drawEdges:
            self.canvas.create_line(Ox, Oy, Lx, Ly, fill="#000000", width=width)
            self.canvas.create_line(Bx, By, Lx, Ly, fill="#00008B", width=width)
            self.canvas.create_line(Bx, By, Ox, Oy, fill="#808080", width=width)

        # 2b) marqueurs colorés par type de bord
        # O = Ouverture (noir), B = Base (bleu), L = Lumière (jaune)
        marker_px = 6  # rayon en pixels (indépendant du zoom)
        def _dot(x, y, fill, outline="black"):
            r = marker_px
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=1)

        # Ouverture / O (noir)
        Ox, Oy = self._world_to_screen(P["O"])
        _dot(Ox, Oy, fill="#000000", outline="#000000")
        # Base / B (bleu)
        Bx, By = self._world_to_screen(P["B"])
        _dot(Bx, By, fill="#0000FF", outline="#000000")
        # Lumière / L (jaune)
        Lx, Ly = self._world_to_screen(P["L"])
        _dot(Lx, Ly, fill="#FFD700", outline="#000000")

        # 3) barycentre (monde)
        cx = (P["O"][0] + P["B"][0] + P["L"][0]) / 3.0
        cy = (P["O"][1] + P["B"][1] + P["L"][1]) / 3.0

        # 4) labels (sans "O:" et sans préfixes "B:" / "L:")
        if labels:
            for pt, txt in zip(pts_world, labels):
                # Supprimer les préfixes "O:", "B:", "L:" si présents
                if ":" in txt:
                    prefix, value = txt.split(":", 1)
                else:
                    prefix, value = "", txt
                prefix = prefix.strip().lower()
                value  = value.strip()

                # Ne rien afficher pour l'ouverture (on a le point noir)
                if prefix in ("o", "ouverture", "ouv"):
                    continue

                # Pour Base/Lumière, n'afficher que la valeur (sans codes/lettres)
                display = value
                if not display:
                    continue

                lx = (1.0 - inset) * pt[0] + inset * cx
                ly = (1.0 - inset) * pt[1] + inset * cy
                sx, sy = self._world_to_screen((lx, ly))
                self.canvas.create_text(sx, sy, text=display, anchor="center", font=("Arial", 8), tags="tri_label")

        # 5) numéro du triangle (toujours affiché après les labels, en avant-plan)
        if tri_id is not None:
            sx, sy = self._world_to_screen((cx, cy))
            num_txt = f"{tri_id}{'S' if tri_mirrored else ''}"
            # Bloc ID + mot recentré verticalement autour de sy, avec un écart réduit
            gap = 8  # écart vertical (px) entre ID et mot
            id_y   = sy - (gap // 2)
            word_y = sy + (gap - gap // 2)

            self.canvas.create_text(
                sx, id_y, text=num_txt,
                anchor="center", font=("Arial", 10, "bold"),
                fill="red", tags="tri_num"
            )
            self.canvas.tag_raise("tri_num")

            # 5b) si un mot est associé à ce triangle, l’afficher sous l’ID (italique)
            winfo = self._tri_words.get(int(tri_id))
            if winfo and isinstance(winfo.get("word"), str) and winfo["word"].strip():
                self.canvas.create_text(
                    sx, word_y, text=winfo["word"],
                    anchor="n", font=("Arial", 9, "italic"),
                    fill="#222", tags="tri_word"
                )
    def _dbgSnap(self, msg: str):
        """Trace console pour diagnostiquer l'assist de collage (activable via F10)."""
        if getattr(self, "_debug_snap_assist", False):
            print(msg)

    # --- helpers: mode déconnexion (CTRL) + curseur ---
    def _on_ctrl_down(self, event=None):
        # Pendant une mesure d'azimut du compas, CTRL sert uniquement à désactiver le snap.
        # On évite donc d'activer le mode "déconnexion" des triangles (curseur + aides).
        if getattr(self, "_clock_measure_active", False):
            self._ctrl_down = True
            return

        if not self._ctrl_down:
            self._ctrl_down = True
            # Masquer tout tooltip en mode déconnexion
            self._hide_tooltip()
            try:
                self.canvas.configure(cursor="X_cursor")
            except tk.TclError:
                self.canvas.configure(cursor="crosshair")


            # --- NOUVEAU (étape 2) ---
            # Si CTRL est pressé *après* avoir sélectionné un sommet (assist ON),
           # on applique immédiatement la ROTATION d'alignement (comme au release),
            # mais SANS translation et SANS collage (CTRL empêchera le snap au relâchement).
            try:
                if self._sel and (not self._sel.get("suppress_assist")):
                    mode = self._sel.get("mode")
                    choice = getattr(self, "_edge_choice", None)
                    if choice:
                        import numpy as np

                        # -------- Cas 1 : triangle seul (mode vertex) --------
                        if mode == "vertex":
                            idx = self._sel.get("idx")
                            vkey = self._sel.get("vkey")
                            if choice[0] == idx and choice[1] == vkey:
                                (_, _, idx_t, vkey_t, (m_a, m_b, t_a, t_b)) = choice
                                tri_m = self._last_drawn[idx]
                                Pm = tri_m["pts"]

                                A = np.array(m_a, dtype=float)  # mobile: sommet saisi
                                B = np.array(m_b, dtype=float)  # mobile: voisin arête
                                U = np.array(t_a, dtype=float)  # cible: sommet (pas utilisé ici)
                                V = np.array(t_b, dtype=float)  # cible: voisin arête

                                ang_m = self._ang_of_vec(B[0] - A[0], B[1] - A[1])
                                ang_t = self._ang_of_vec(V[0] - U[0], V[1] - U[1])
                                dtheta = ang_t - ang_m

                                if abs(dtheta) > 1e-12:
                                    R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                                                  [np.sin(dtheta),  np.cos(dtheta)]], dtype=float)
                                    for k in ("O", "B", "L"):
                                        p = np.array(Pm[k], dtype=float)
                                        p_rot = A + (R @ (p - A))
                                        Pm[k][0] = float(p_rot[0])
                                        Pm[k][1] = float(p_rot[1])

                                    # Redessiner + remettre les aides (rouge/gris)
                                    self._redraw_from(self._last_drawn)
                                    v_world = np.array(self._last_drawn[idx]["pts"][vkey], dtype=float)
                                    self._update_nearest_line(v_world, exclude_idx=idx)
                                    self._update_edge_highlights(idx, vkey, idx_t, vkey_t)

                        # -------- Cas 2 : déplacement de groupe ancré sur sommet --------
                        elif mode == "move_group":
                            anchor = self._sel.get("anchor")
                            gid = self._sel.get("gid")
                            if anchor and anchor.get("type") == "vertex":
                                anchor_tid = anchor.get("tid")
                                anchor_vkey = anchor.get("vkey")
                                if choice[0] == anchor_tid and choice[1] == anchor_vkey:
                                    (_, _, idx_t, vkey_t, (m_a, m_b, t_a, t_b)) = choice

                                    A = np.array(m_a, dtype=float)
                                    B = np.array(m_b, dtype=float)
                                    U = np.array(t_a, dtype=float)
                                    V = np.array(t_b, dtype=float)

                                    ang_m = self._ang_of_vec(B[0] - A[0], B[1] - A[1])
                                    ang_t = self._ang_of_vec(V[0] - U[0], V[1] - U[1])
                                    dtheta = ang_t - ang_m

                                    if abs(dtheta) > 1e-12:
                                        R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                                                      [np.sin(dtheta),  np.cos(dtheta)]], dtype=float)

                                        g = self.groups.get(gid)
                                        if g:
                                            for node in g["nodes"]:
                                                tid = node.get("tid")
                                                if 0 <= tid < len(self._last_drawn):
                                                    P = self._last_drawn[tid]["pts"]
                                                    for k in ("O", "B", "L"):
                                                        p = np.array(P[k], dtype=float)
                                                        p_rot = A + (R @ (p - A))
                                                        P[k][0] = float(p_rot[0])
                                                        P[k][1] = float(p_rot[1])

                                        # Redessiner + remettre les aides (en excluant le groupe mobile)
                                        try:
                                            self._recompute_group_bbox(gid)
                                        except Exception:
                                            pass
                                        self._redraw_from(self._last_drawn)
                                        v_world = np.array(self._last_drawn[anchor_tid]["pts"][anchor_vkey], dtype=float)
                                        self._update_nearest_line(v_world, exclude_idx=anchor_tid, exclude_gid=gid)
                                        self._update_edge_highlights(anchor_tid, anchor_vkey, idx_t, vkey_t)
            except Exception:
                pass

    def _on_ctrl_up(self, event=None):
        if self._ctrl_down:
            self._ctrl_down = False
            try:
                self.canvas.configure(cursor="")
            except tk.TclError:
                pass


    # ---------- Mouse navigation ----------
    # Drag depuis la liste
    def _on_triangle_list_select(self, event=None):
        """Empêche la sélection d'un triangle déjà utilisé dans le scénario courant."""
        # éviter la récursion quand on modifie la sélection nous-mêmes
        if getattr(self, "_in_triangle_select_guard", False):
            return
        if not hasattr(self, "listbox"):
            return

        sel = self.listbox.curselection()
        if not sel:
            self._last_triangle_selection = None
            return

        idx = int(sel[0])
        try:
            tri = self._triangle_from_index(idx)
            tid = int(tri.get("id"))
        except Exception:
            tid = None

        placed = getattr(self, "_placed_ids", set()) or set()
        if tid is not None and tid in placed:
            # Triangle déjà posé : on annule la sélection visuelle
            self._in_triangle_select_guard = True
            try:
                self.listbox.selection_clear(0, tk.END)
                if (self._last_triangle_selection is not None and
                        0 <= self._last_triangle_selection < self.listbox.size()):
                    self.listbox.selection_set(self._last_triangle_selection)
            finally:
                self._in_triangle_select_guard = False
            if hasattr(self, "status"):
                self.status.config(text=f"Triangle {tid} déjà utilisé dans ce scénario.")
            return

        # Sélection valide
        self._last_triangle_selection = idx

    def _on_ctrl_up(self, event=None):
        if self._ctrl_down:
            self._ctrl_down = False
            try:
                self.canvas.configure(cursor="")
            except tk.TclError:
                pass

    # ---------- Mouse navigation ----------
    # Drag depuis la liste
    def _on_list_mouse_down(self, event):
        """
        Démarre un drag & drop depuis la listbox,
        sauf si le triangle est déjà utilisé dans le scénario courant.
        """
        # Pas de DF → rien à faire
        if getattr(self, "df", None) is None or self.df.empty:
            return

        # Index de la ligne cliquée dans la listbox
        i = self.listbox.nearest(event.y)
        if i < 0:
            return

        # Sélection visuelle
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(i)

        # Récupération de la définition du triangle
        try:
            tri = self._triangle_from_index(i)
        except Exception as e:
            try:
                self.status.config(text=f"Erreur: {e}")
            except Exception:
                pass
            return

        tri_id = tri.get("id")
        # Si le triangle est déjà posé dans ce scénario, on bloque le drag
        if tri_id is not None and int(tri_id) in getattr(self, "_placed_ids", set()):
            try:
                self.status.config(text=f"Triangle {tri_id} déjà utilisé dans ce scénario.")
            except Exception:
                pass
            self._drag = None
            return

        # Préparation de la structure de drag pour le moteur commun
        start_screen = np.array([event.x_root, event.y_root], dtype=float)
        # NOTE : last_canvas n'est plus utilisé dans la nouvelle logique de drag & drop
        # (tout est géré en coords monde/écran via _world_to_screen / _screen_to_world).
        self._drag = {
            "from": "list",
            "triangle": tri,
            "list_index": i,
            "start_screen": start_screen,
            "mirrored": tri.get("mirrored", False),
        }

        # Réinitialiser un éventuel fantôme existant
        if self._drag_preview_id is not None:
            self.canvas.delete(self._drag_preview_id)
            self._drag_preview_id = None

        # Forcer le focus sur le canvas pour recevoir les mouvements
        self.canvas.focus_set()

        # Curseur "main" pendant le drag
        try:
            self.canvas.configure(cursor="hand2")
        except tk.TclError:
            pass

        try:
            self.status.config(text="Glissez le triangle sur le canvas puis relâchez pour le déposer.")
        except Exception:
            pass

    # ---------- Neighbours for tooltip ----------
    def _point_on_segment(self, P, A, B, eps):
        """Vrai si P appartient au segment [A,B] (colinéarité + projection dans [0,1])."""
        Ax,Ay = float(A[0]), float(A[1]); Bx,By = float(B[0]), float(B[1])
        Px,Py = float(P[0]), float(P[1])
        ABx, ABy = Bx-Ax, By-Ay
        APx, APy = Px-Ax, Py-Ay
        cross = abs(ABx*APy - ABy*APx)
        if cross > eps: 
            return False
        dot  = ABx*APx + ABy*APy
        ab2  = ABx*ABx + ABy*ABy
        if ab2 <= eps: 
            return False
        t = dot/ab2
        return -eps <= t <= 1.0+eps

    def _display_name(self, key, labels_tuple):
        """Retourne le libellé affiché (sans préfixe) pour O/B/L."""
        try:
            if   key == "O": raw = labels_tuple[0]
            elif key == "B": raw = labels_tuple[1]
            else:            raw = labels_tuple[2]
        except Exception:
            raw = ""
        s = str(raw or "").strip()
        return s
 
    # --- Azimut Nord->horaire (repère monde: Y vers le haut).
    #     Horloge : 360° = 12h et 360° = 60' (minutes d'horloge)
    def _azimutDegEtMinute(self, pSrc, pDst):
        """
        Renvoie (degInt, minFloat, hourFloat) pour le vecteur pSrc->pDst :
          - degInt    : angle depuis le Nord (sens horaire) en degrés entiers 0..359
          - minFloat  : minutes d’horloge au dixième (0.0 .. ≤60.0)   [360° = 60' → 1' = 6°]
          - hourFloat : heures d’horloge au dixième (0.0 .. ≤12.0)    [360° = 12h → 1h = 30°]
        """
        import math
        x1, y1 = float(pSrc[0]), float(pSrc[1])
        x2, y2 = float(pDst[0]), float(pDst[1])
        dx, dy = (x2 - x1), (y2 - y1)
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return None
        # Mesure depuis l'axe +Y (Nord) en tournant horaire
        angRad = math.atan2(dx, dy)
        angDeg = (math.degrees(angRad) + 360.0) % 360.0
        degInt   = int(round(angDeg)) % 360
        # Heures: 360° = 12h → 1h = 30° ; Minutes: 360° = 60' → 1' = 6°
        hourFloat = round(angDeg / 30.0, 1)        # h = deg / 30
        minFloat  = round(angDeg / 6.0, 1)         # ' = deg / 6  (0..60)
        return (degInt, minFloat, hourFloat)

    def _connected_vertices_from(self, v_world, gid, eps):
        """
        Depuis un sommet (ou un point tombant sur une arête), trouver toutes les
        extrémités connectées à ce point dans le GROUPE `gid`.
         Déduplique par direction et garde, sur une même direction, le point le plus éloigné.
         Retourne une liste d'objets { "name": str, "pt": (x, y) }.
        """
        nodes = self._group_nodes(gid) if gid else []
        out_candidates = []   # [(name, tid, dx, dy, dist, neigh_xy)]
        for nd in nodes:
            tid = nd.get("tid")
            if tid is None or not (0 <= tid < len(self._last_drawn)):
                continue
            tri = self._last_drawn[tid]
            Pt  = tri["pts"]
            lab = tri.get("labels", ("Bourges","",""))
            # 3 arêtes du triangle
            edges = (("O","B"), ("B","L"), ("L","O"))
            for a,b in edges:
                A, B = Pt[a], Pt[b]
                # 1) si v_world == A ➜ connecté à B ; si v_world == B ➜ connecté à A
                if (abs(v_world[0]-A[0]) <= eps and abs(v_world[1]-A[1]) <= eps):
                    name = self._display_name(b, lab)
                    dx, dy = float(B[0]-v_world[0]), float(B[1]-v_world[1])
                    dist = (dx*dx+dy*dy)**0.5
                    if name:
                        out_candidates.append((name, int(tid), dx, dy, dist, (float(B[0]), float(B[1]))))
                    continue
                if (abs(v_world[0]-B[0]) <= eps and abs(v_world[1]-B[1]) <= eps):
                    name = self._display_name(a, lab)
                    dx, dy = float(A[0]-v_world[0]), float(A[1]-v_world[1])
                    dist = (dx*dx+dy*dy)**0.5
                    if name:
                        out_candidates.append((name, int(tid), dx, dy, dist, (float(A[0]), float(A[1]))))
                    continue
                # 2) sinon : le point est-il sur le SEGMENT [A,B] ? ➜ connecté aux deux extrémités
                if self._point_on_segment(v_world, A, B, eps):
                    name_a = self._display_name(a, lab)
                    name_b = self._display_name(b, lab)
                    dx, dy = float(A[0]-v_world[0]), float(A[1]-v_world[1])
                    dist = (dx*dx+dy*dy)**0.5
                    if name_a:
                        out_candidates.append((name_a, int(tid), dx, dy, dist, (float(A[0]), float(A[1]))))
                    dx, dy = float(B[0]-v_world[0]), float(B[1]-v_world[1])
                    dist = (dx*dx+dy*dy)**0.5
                    if name_b:
                        out_candidates.append((name_b, int(tid), dx, dy, dist, (float(B[0]), float(B[1]))))
        if not out_candidates:
            return []
        # Déduplication UNIQUE par (nom + tid + position voisine quantifiée)
        by_key = {}  # key = (name, tid, qx, qy) -> (dist, pt)
        def q(p):
            return (round(p[0]/eps)*eps, round(p[1]/eps)*eps)
        for (name, tid, dx, dy, dist, neigh_xy) in out_candidates:
            qx, qy = q(neigh_xy)
            key = (name, int(tid), qx, qy)
            if (key not in by_key) or (dist > by_key[key][0]):
                by_key[key] = (dist, neigh_xy)
        # Restituer [{name, pt, tid}] triés par distance décroissante (optionnel)
        out = []
        for (name, tid, qx, qy), (dist, pt) in sorted(by_key.items(), key=lambda kv: -kv[1][0]):
            out.append({"name": name, "pt": (float(pt[0]), float(pt[1])), "tid": int(tid)})

        return out

    def _on_canvas_motion_update_drag(self, event):
        # Mode compas : mesure d'un azimut (relatif à la référence)
        if getattr(self, "_clock_measure_active", False):
            self._clock_measure_update_preview(int(event.x), int(event.y))
            return "break"

        # Mode compas : mesure d'arc d'angle
        if getattr(self, "_clock_arc_active", False):
            self._clock_arc_update_preview(int(event.x), int(event.y))
            return "break"

        # Mode compas : définition de l'azimut de référence
        if getattr(self, "_clock_setref_active", False):
            self._clock_setref_update_preview(int(event.x), int(event.y))
            return "break"

        # Toujours garantir un pick-cache à jour avant tout hit/tooltip
        self._ensure_pick_cache()

        # 1) Drag & drop depuis la liste → fantôme
        if self._drag:
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            tri = self._drag["triangle"]
            P = tri["pts"]
            dx = wx - float(P["O"][0])
            dy = wy - float(P["O"][1])
            O = np.array([P["O"][0] + dx, P["O"][1] + dy])
            B = np.array([P["B"][0] + dx, P["B"][1] + dy])
            L = np.array([P["L"][0] + dx, P["L"][1] + dy])
            self._drag["world_pts"] = {"O": O, "B": B, "L": L}
            coords = []
            for pt in (O, B, L):
                sx, sy = self._world_to_screen(pt)
                coords += [sx, sy]
            if self._drag_preview_id is None:
                self._drag_preview_id = self.canvas.create_polygon(*coords, outline="gray50", dash=(4,2), fill="", width=2)
            else:
                self.canvas.coords(self._drag_preview_id, *coords)
            return

        # 2) Mode rotation : suivre la souris sans bouton appuyé
        if self._sel and self._sel.get("mode") == "rotate":
            idx = self._sel["idx"]
            sel = self._sel
            pivot = sel["pivot"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            cur_angle = math.atan2(wy - pivot[1], wx - pivot[0])
            dtheta = cur_angle - sel["start_angle"]
            c, s = math.cos(dtheta), math.sin(dtheta)
            R = np.array([[c, -s], [s, c]], dtype=float)
            P = self._last_drawn[idx]["pts"]
            for k in ("O", "B", "L"):
                v = sel["orig_pts"][k] - pivot
                P[k] = (R @ v) + pivot
            self._redraw_from(self._last_drawn)
            self._sel["last_angle"] = cur_angle
            return
        # 2b) Mode rotation de GROUPE : suivre la souris (sans bouton appuyé)
        if self._sel and self._sel.get("mode") == "rotate_group":
            sel = self._sel
            gid = sel["gid"]
            pivot = sel["pivot"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            cur_angle = math.atan2(wy - pivot[1], wx - pivot[0])
            dtheta = cur_angle - sel["start_angle"]
            c, s = math.cos(dtheta), math.sin(dtheta)
            R = np.array([[c, -s], [s, c]], dtype=float)
            g = self.groups.get(gid)
            if not g:
                return
            # repartir du snapshot pour éviter l'accumulation d'erreurs
            for node in g["nodes"]:
                tid = node["tid"]
                if 0 <= tid < len(self._last_drawn) and tid in sel["orig_group_pts"]:
                    Pt = self._last_drawn[tid]["pts"]
                    Orig = sel["orig_group_pts"][tid]
                    for k in ("O","B","L"):
                        v = np.array(Orig[k], dtype=float) - pivot
                        Pt[k] = (R @ v) + pivot
            self._recompute_group_bbox(gid)
            self._redraw_from(self._last_drawn)
            self._sel["last_angle"] = cur_angle
            return

        # 3) Pas de drag/rotation : gestion du TOOLTIP (survol de sommet)
        try:
            mode, idx, extra = self._hit_test(event.x, event.y)
        except Exception:
            mode, idx, extra = (None, None, None)
        # En mode déconnexion (CTRL), ne pas afficher de tooltip
        if getattr(self, "_ctrl_down", False):
            self._hide_tooltip()
            return

        if mode == "vertex" and idx is not None:
            vkey = extra if isinstance(extra, str) else None
            # position monde du sommet visé
            try:
                P0 = self._last_drawn[idx]["pts"]
                v_world = np.array(P0[vkey], dtype=float) if vkey in ("O","B","L") else None
            except Exception:
                v_world = None
            lines = []
            if v_world is not None:
                # tolérance monde proportionnelle à l'affichage (hit_px/zoom)
                tol_world = max(1e-9, float(getattr(self, "_hit_px", 12)) / max(self.zoom, 1e-9))
                def same_pos(a, b):
                    return (abs(float(a[0]) - float(b[0])) <= tol_world) and \
                           (abs(float(a[1]) - float(b[1])) <= tol_world)
                # parcourir le GROUPE uniquement
                gid = self._get_group_of_triangle(idx)
                nodes = self._group_nodes(gid) if gid else []
                seen = set()  # éviter doublons texte
                for nd in nodes:
                    tid = nd.get("tid")
                    if tid is None or not (0 <= tid < len(self._last_drawn)):
                        continue
                    tri = self._last_drawn[tid]
                    Pt = tri["pts"]
                    labels = tri.get("labels", ("Bourges","",""))
                    # Préfixes comme dans la liste : O:..., B:..., L:...
                    for key, lbl in (("O", labels[0] if len(labels) > 0 else ""),
                                     ("B", labels[1] if len(labels) > 1 else ""),
                                     ("L", labels[2] if len(labels) > 2 else "")):

                        if same_pos(Pt[key], v_world):
                            triNum = int(tri.get("id", tid+1))
                            txt = f"{key}:{str(lbl).strip()}({triNum}S)"
                            if txt and txt not in seen:
                                lines.append(txt); seen.add(txt)

                # ---- LIGNE(S) DU DESSOUS : arêtes connectées ----
                try:
                    connected = self._connected_vertices_from(v_world, gid, tol_world)
                except Exception:
                    connected = []
                if connected:
                    # séparateur visuel entre “sommet(s)” et “connectés”
                    lines.append("")  # ligne vide
                    # chaque voisin: "-> Nom — ddd° / mm.m' / h.h"
                    for item in connected:
                        try:
                            name = item.get("name", "")
                            pt   = item.get("pt", None)
                            tid2 = item.get("tid", None)
                        except Exception:
                            name, pt = (str(item), None)
                        if not name:
                            continue
                        # Ajouter l'id du triangle (ex: 2S) pour lever l'ambiguïté
                        nameDisp = name
                        if tid2 is not None and 0 <= int(tid2) < len(self._last_drawn):
                            tri2 = self._last_drawn[int(tid2)]
                            triNum2 = int(tri2.get("id", int(tid2)+1))
                            nameDisp = f"{name} ({triNum2}S)"

                        if pt is None:
                            lines.append(f"-> {nameDisp}")
                            continue
                        az = self._azimutDegEtMinute(v_world, pt)
                        if az is None:
                            lines.append(f"-> {nameDisp} — azimut indéterminé")
                        else:
                            degInt, minFloat, hourFloat = az
                            # Alignements lisibles: degrés sur 3, minutes & heures avec 1 décimale
                            lines.append(f"-> {nameDisp} — {degInt:03d}° / {minFloat:04.1f}' / {hourFloat:0.1f}h")
            tooltip_txt = "\n".join(lines)
            if tooltip_txt:
                # === Placement robuste par CENTRE du tooltip ===
                # 1) Coordonnées CANVAS du sommet et du barycentre
                sx_v, sy_v = self._world_to_screen(v_world)
                try:
                    Cx = (P0["O"][0] + P0["B"][0] + P0["L"][0]) / 3.0
                    Cy = (P0["O"][1] + P0["B"][1] + P0["L"][1]) / 3.0
                except Exception:
                    Cx, Cy = float(v_world[0]), float(v_world[1])
                sx_c, sy_c = self._world_to_screen((Cx, Cy))
                # 2) Direction (centroïde -> sommet) en PIXELS CANVAS
                vx = float(sx_v) - float(sx_c)
                vy = float(sy_v) - float(sy_c)
                n  = (vx*vx + vy*vy) ** 0.5
                if n <= 1e-6:
                    # secours : pousser vers le haut écran
                    ux, uy = 0.0, -1.0
                else:
                    ux, uy = (vx / n), (vy / n)

                # 3) Mesurer le tooltip pour connaître sa diagonale
                #    (on l'affiche/MAJ en mémoire, sans se soucier de la position pour l'instant)
                if self._tooltip is None or not self._tooltip.winfo_exists():
                    # créer/mesurer via helper centre, posé provisoirement au sommet
                    self._show_tooltip_at_center(tooltip_txt, sx_v, sy_v)
                else:
                    self._tooltip_label.config(text=tooltip_txt)
                    self._tooltip.update_idletasks()
                tw = max(1, int(self._tooltip.winfo_width()))
                th = max(1, int(self._tooltip.winfo_height()))

                # 4) Distance à partir du SOMMET jusqu'au CENTRE du tooltip
                #    Utiliser le support d'un rectangle axis-aligné : r(θ)=|rx·cosθ|+|ry·sinθ|
                #    pour projeter la demi-taille du tooltip le long de (centroïde→sommet).
                rx = 0.5 * tw
                ry = 0.5 * th
                ang = math.atan2(uy, ux)
                r_eff = abs(rx * math.cos(ang)) + abs(ry * math.sin(ang))
                cushion_px  = float(getattr(self, "_tooltip_cushion_px", 14))
                dist = r_eff + cushion_px
                cx_tip = sx_v + ux * dist
                cy_tip = sy_v + uy * dist
    
                # 5) Placement centré (convertit canvas->écran en interne)
                self._show_tooltip_at_center(tooltip_txt, cx_tip, cy_tip)
            else:
                self._hide_tooltip()
        else:
            # pas de sommet → masquer tooltip
            self._hide_tooltip()

        # sécurité : si le cache n’est pas prêt (post-load), le régénérer pour les tooltips
        if not self._pick_cache_valid:
            self._rebuild_pick_cache()

    # ---------- Lien + surlignage faces candidates ----------
    def _find_nearest_vertex(self, v_world, exclude_idx=None, exclude_gid=None):
        """Retourne (idx_triangle, key('O'|'B'|'L'), pos_world) du sommet d'un AUTRE triangle le plus proche.
        On peut exclure un triangle précis (exclude_idx) et/ou tout un groupe (exclude_gid)."""
        best = None
        best_d2 = None
        for j, t in enumerate(self._last_drawn):
            if j == exclude_idx:
                continue
            if exclude_gid is not None and t.get("group_id", None) == exclude_gid:
                continue            
            P = t["pts"]
            for k in ("O","B","L"):
                w = np.array(P[k], dtype=float)
                d2 = float((w[0]-v_world[0])**2 + (w[1]-v_world[1])**2)
                if (best_d2 is None) or (d2 < best_d2):
                    best_d2 = d2
                    best = (j, k, w)
        return best

    def _update_nearest_line(self, v_world, exclude_idx=None, exclude_gid=None):
        """Dessine (ou MAJ) un trait fin entre v_world et le sommet le plus proche d'un AUTRE triangle."""
        found = self._find_nearest_vertex(v_world, exclude_idx=exclude_idx, exclude_gid=exclude_gid)

        if found is None:
            self._clear_nearest_line()
            return
        _, _, best = found
        # tracer/mettre à jour
        x1, y1 = self._world_to_screen(v_world)
        x2, y2 = self._world_to_screen(best)
        if self._nearest_line_id is None:
            self._nearest_line_id = self.canvas.create_line(x1, y1, x2, y2, fill="#888888", width=1)
        else:
            self.canvas.coords(self._nearest_line_id, x1, y1, x2, y2)

    def _clear_nearest_line(self):
        if self._nearest_line_id is not None:
            try:
                self.canvas.delete(self._nearest_line_id)
            except Exception:
                pass
            self._nearest_line_id = None

    def _reset_assist(self):
        """Nettoie TOUTES les aides visuelles et l'état associé."""
        self._clear_nearest_line()
        self._clear_edge_highlights()
        self._edge_highlights = None
        self._edge_choice = None

    def _draw_temp_edge_world(self, p1, p2, color="#ff7f00", width=3):
        """
        Trace une ligne temporaire en coordonnées MONDE entre p1 et p2.
        p1/p2 : np.array([x, y]) ou tuple (x, y)
        """
        x1, y1 = self._world_to_screen(p1)
        x2, y2 = self._world_to_screen(p2)
        try:
            _id = self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
        except Exception:
            _id = None
        return _id

    # ---------- utilitaires contour de groupe ----------
    def _group_outline_segments(self, gid: int, eps: float = EPS_WORLD):
        """
        Retourne une liste de sous-segments (P1,P2) qui appartiennent au
        **contour extérieur** de l’union des triangles du groupe `gid`.
        (On retire donc toutes les arêtes internes.)
        """
        g = self.groups.get(gid) or {}
        nodes = g.get("nodes", [])
        if not nodes:
            return []

        # 1) collecter triangles (monde) + tous les sommets
        tris = []
        vertices = []
        for nd in nodes:
            tid = nd.get("tid")
            if tid is None or not (0 <= tid < len(self._last_drawn)):
                continue
            P = self._last_drawn[tid]["pts"]
            tris.append(_ShPoly([tuple(P["O"]), tuple(P["B"]), tuple(P["L"])]))
            vertices.extend([np.array(P["O"], float), np.array(P["B"], float), np.array(P["L"], float)])

        if not tris:
            return []

        # 2) union → polygone du groupe
        # NOTE: on part de l'union géométrique, puis on extrait directement les
        # exterieures (pas les trous) pour obtenir **uniquement** le contour du groupe.
        u = _sh_union([p for p in tris if p.is_valid and p.area > 0])
        # IMPORTANT: l'auto-assemblage passe par des rotations/translations flottantes.
        # Les sommets/côtés qui devraient coïncider peuvent être séparés d'un micro-écart.
        # Sans "snap", l'union devient MultiPolygon et les arêtes communes réapparaissent
        # en contour. On recolle donc les micro-écarts via buffer(+eps)/buffer(-eps).
        if eps and eps > 0:
            poly = u.buffer(eps).buffer(-eps).buffer(0)
        else:
            poly = u.buffer(0)
        if getattr(poly, "is_empty", True):
            return []

        # helper
        def _split_by_vertices(A, B):
            """Découpe [A,B] par les sommets colinéaires situés dessus."""
            A = np.array(A, float); B = np.array(B, float)
            AB = B - A
            ts = [0.0, 1.0]
            for V in vertices:
                AV = V - A
                cross = abs(AB[0]*AV[1] - AB[1]*AV[0])
                if cross > eps:
                    continue
                dot = AV[0]*AB[0] + AV[1]*AB[1]
                ab2 = AB[0]*AB[0] + AB[1]*AB[1]
                if -eps <= dot <= ab2 + eps:
                    t = 0.0 if ab2 == 0 else dot / ab2
                    if 1e-9 < t < 1 - 1e-9:
                        ts.append(float(t))
            ts = sorted(set(ts))
            out = []
            for i in range(len(ts) - 1):
                q0 = A + (B - A) * ts[i]
                q1 = A + (B - A) * ts[i+1]
                if np.linalg.norm(q1 - q0) > eps:
                    out.append((q0, q1))
            return out

        # 3) Extraire le contour EXTERIEUR depuis l'union, puis le "re-découper" sur
        #    les sommets d'origine pour conserver la granularité (utile pour snap/compas).
        outline = []

        # Récupérer uniquement les exteriors (pas les trous)
        lines = []
        try:
            gtype = getattr(poly, "geom_type", "")
            if gtype == "Polygon":
                lines = [poly.exterior]
            elif gtype == "MultiPolygon":
                lines = [p.exterior for p in getattr(poly, "geoms", [])]
            else:
                # fallback: boundary brute (peut inclure des trous)
                b = getattr(poly, "boundary", None)
                if b is not None:
                    if getattr(b, "geom_type", "") == "MultiLineString":
                        lines = list(getattr(b, "geoms", []))
                    else:
                        lines = [b]
        except Exception:
            lines = []

        for ln in (lines or []):
            try:
                coords = list(getattr(ln, "coords", []))
            except Exception:
                coords = []
            if len(coords) < 2:
                continue
            for i in range(len(coords) - 1):
                A = coords[i]
                B = coords[i + 1]
                for s in _split_by_vertices(A, B):
                    outline.append(s)

        return outline

    # --- util: segments d'enveloppe (groupe ou triangle seul) ---
    def _outline_for_item(self, idx: int, eps: float | None = None):
        """
        Retourne les segments (p1,p2) de l'enveloppe **du groupe** auquel appartient idx.
        Hypothèse d'architecture : tout item a un group_id (singleton possible).
        """
        gid = self._last_drawn[idx].get("group_id")
        assert gid is not None, "Invariant brisé: item sans group_id"
        try:
            # Par défaut (eps=None), on calcule un eps "rendu" pour recoller les micro-écarts
            # numériques (auto-assemblage) et éviter que des arêtes internes ressortent comme contours.
            # IMPORTANT : certains appels (ex: snap/assist) ont besoin des sommets exacts ; dans ce cas,
            # passer explicitement eps=EPS_WORLD (ou autre) pour éviter tout regroupement de sommets.
            if eps is None:
                tol_world = max(
                    1e-9,
                    float(getattr(self, "stroke_px", 2)) / max(getattr(self, "zoom", 1.0), 1e-9)
                )
                eps = max(EPS_WORLD, 0.5 * tol_world)
            return self._group_outline_segments(gid, eps=float(eps))
        except Exception:
            return []


    def _update_edge_highlights(self, mob_idx: int, vkey_m: str, tgt_idx: int, tgt_vkey: str):
        """Version factorisée 'graphe de frontière' (symétrique) :
        1) On récupère l'outline des deux groupes
        2) On construit un graphe de frontière (half-edges) côté mobile & cible
        3) Au sommet cliqué de chaque côté, on prend **les 2 demi-arêtes incidentes**
        4) On sélectionne la paire (mobile×cible) qui **minimise globalement** Δ-angle
        5) Affichage : toujours bleues (incidentes), rouge pour la paire choisie
        """

        import numpy as np
        from math import atan2, pi

        def _to_np(p): return np.array([float(p[0]), float(p[1])], dtype=float)
        def _azim(a, b):
            # tuple-safe: accepte tuples ou np.array
            from math import atan2
            return atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))
        def _ang_dist(a, b):
             from math import pi
             d = abs(a - b) % (2*pi)
             return d if d <= pi else (2*pi - d)
        def _almost_eq(a,b,eps=EPS_WORLD): return abs(a[0]-b[0])<=eps and abs(a[1]-b[1])<=eps

        # Sommets & groupes
        tri_m = self._last_drawn[mob_idx];  Pm = tri_m["pts"]; vm = _to_np(Pm[vkey_m])
        tri_t = self._last_drawn[tgt_idx];  Pt = tri_t["pts"]; vt = _to_np(Pt[tgt_vkey])
        gid_m = tri_m.get("group_id"); gid_t = tri_t.get("group_id")
        if gid_m is None or gid_t is None:
            self._clear_edge_highlights(); self._edge_choice = None; return


        # DEBUG
        self._dbgSnap(
            f"[snap] update_edge_highlights mob={mob_idx}:{vkey_m} (gid={gid_m}) -> tgt={tgt_idx}:{tgt_vkey} (gid={gid_t})"
        )

        # Tolérance (utile à d'autres endroits si besoin)
        tol_world = max(1e-9, float(getattr(self, "stroke_px", 2)) / max(getattr(self, "zoom", 1.0), 1e-9))

        # === Collecte via graphe de frontière puis argmin global (Δ-angle) ===
        # 1) outlines des deux groupes
        # IMPORTANT (snap) : on veut les sommets "tels quels" pour que vm/vt appartiennent au graphe.
        # L'eps "rendu" (contour-only) peut fusionner / décaler légèrement les sommets et casser
        # l'incidence au sommet cliqué => aucune paire d'arêtes candidate.
        mob_outline = self._outline_for_item(mob_idx, eps=EPS_WORLD) or []
        tgt_outline = self._outline_for_item(tgt_idx, eps=EPS_WORLD) or []

        # Fallback : si l'outline "groupe" est vide (ou indisponible),
        # utiliser l'outline du triangle lui-même pour ne pas casser l'assist.
        if not mob_outline:
            mob_outline = [(tuple(Pm["O"]), tuple(Pm["B"])), (tuple(Pm["B"]), tuple(Pm["L"])), (tuple(Pm["L"]), tuple(Pm["O"]))]
        if not tgt_outline:
            tgt_outline = [(tuple(Pt["O"]), tuple(Pt["B"])), (tuple(Pt["B"]), tuple(Pt["L"])), (tuple(Pt["L"]), tuple(Pt["O"]))]

        self._dbgSnap(f"[snap] outlines: mob={len(mob_outline)} tgt={len(tgt_outline)}")

        # 2) half-edges incidentes au sommet cliqué, côté mobile et côté cible
        g_m = self._build_boundary_graph(mob_outline)
        g_t = self._build_boundary_graph(tgt_outline)
        m_inc_raw = self._incident_half_edges_at_vertex(g_m, vm)
        t_inc_raw = self._incident_half_edges_at_vertex(g_t, vt)
        # 3) normalisation (granularité identique à l’outline) pour l’affichage
        m_inc = self._normalize_to_outline_granularity(mob_outline, m_inc_raw, eps=EPS_WORLD)
        t_inc = self._normalize_to_outline_granularity(tgt_outline, t_inc_raw, eps=EPS_WORLD)

        # 4) sélection globale : minimiser l’écart d’azimut + anti-chevauchement
        best = None  # (score, m_edge, t_edge)
        # Unions géométriques brutes des groupes (avant pose)
        S_t = _group_shape_from_nodes(self._group_nodes(gid_t), self._last_drawn)
        S_m_base = _group_shape_from_nodes(self._group_nodes(gid_m), self._last_drawn)
        # Helper de pose rigide (vm->vt, alignement (vm→mB) // (vt→tB))
        from math import atan2, pi
        def _place_union_on_pair(S_base, vm_pt, mB_pt, vt_pt, tB_pt):
            try:
                from shapely.affinity import translate, rotate
            except Exception:
                return None  # si Shapely indispo → on ne filtre pas
            vmx, vmy = float(vm_pt[0]), float(vm_pt[1])
            mBx, mBy = float(mB_pt[0]), float(mB_pt[1])
            vtx, vty = float(vt_pt[0]), float(vt_pt[1])
            tBx, tBy = float(tB_pt[0]), float(tB_pt[1])
            ang_m = atan2(mBy - vmy, mBx - vmx)
            ang_t = atan2(tBy - vty, tBx - vtx)
            dtheta = (ang_t - ang_m + pi) % (2*pi) - pi  # wrap [-pi,pi]
            # T =  translate(vt) ∘ rotate(dtheta around (0,0)) ∘ translate(-vm)
            S = translate(S_base, xoff=-vmx, yoff=-vmy)
            S = rotate(S, dtheta * 180.0 / pi, origin=(0.0, 0.0), use_radians=False)
            S = translate(S, xoff=vtx, yoff=vty)
            return S
        m_oriented = [ (a,b) if _almost_eq(a, vm) else (b,a) for (a,b) in (m_inc_raw or []) ]
        t_oriented = [ (a,b) if _almost_eq(a, vt) else (b,a) for (a,b) in (t_inc_raw or []) ]
        for me in m_oriented:
            azm = _azim(*me)
            for te in t_oriented:
                azt = _azim(*te)
                score = _ang_dist(azm, azt)
                # --- Anti-chevauchement (shrink-only) remis en service ---
                if not getattr(self, "debug_skip_overlap_highlight", False):
                    S_m_pose = _place_union_on_pair(S_m_base, me[0], me[1], te[0], te[1])
                    if S_m_pose is not None:
                        if _overlap_shrink(S_m_pose, S_t,
                                           getattr(self, "stroke_px", 2),
                                           max(getattr(self, "zoom", 1.0), 1e-9)):
                            continue  # paire rejetée pour chevauchement
                # argmin global (parmi les paires admissibles)
                if (best is None) or (score < best[0]):
                    best = (score, me, te)
        # 5) sorties visuelles
        mo = [(tuple(a), tuple(b)) for (a, b) in (mob_outline or [])]
        self._edge_highlights = {
            "all":        [(tuple(a), tuple(b)) for (a, b) in (t_inc or [])],
            "mob_inc":    [(tuple(a), tuple(b)) for (a, b) in (m_inc or [])],
            "tgt_inc":    [(tuple(a), tuple(b)) for (a, b) in (t_inc or [])],
            "best":       (tuple(best[1][0]), tuple(best[1][1]),
                           tuple(best[2][0]), tuple(best[2][1])) if best else None,
            "mob_outline": mo,
            "tgt_outline": [(tuple(a), tuple(b)) for (a, b) in (tgt_outline or [])],
        }

        self._edge_choice = None
        if best:
            (mA, mB), (tA, tB) = best[1], best[2]
            self._edge_choice = (mob_idx, vkey_m, tgt_idx, tgt_vkey, (tuple(mA), tuple(mB), tuple(tA), tuple(tB)))

        # Ajout des contours pour le debug (bleu)
        try:  mob_outline = [(tuple(a), tuple(b)) for (a,b) in self._outline_for_item(mob_idx)]
        except Exception: mob_outline = []
        try:  tgt_outline = [(tuple(a), tuple(b)) for (a,b) in self._outline_for_item(tgt_idx)]
        except Exception: tgt_outline = []
        self._redraw_edge_highlights()


    def _clear_edge_highlights(self):
        """Efface du canvas les lignes d'aide déjà dessinées.
        N'efface PAS self._edge_choice ni self._edge_highlights (elles peuvent être réutilisées)."""
        if self._edge_highlight_ids:
            for _id in self._edge_highlight_ids:
                try:
                    self.canvas.delete(_id)
                except Exception:
                    pass
        self._edge_highlight_ids = []

    def _redraw_edge_highlights(self):
        """Redessine les aides à partir de self._edge_highlights :
        - tout le périmètre (segments EXTÉRIEURS) des 2 groupes en BLEU (fin),
        - uniquement les segments INCIDENTS (candidats possibles) en BLEU **épais**,
        - meilleure paire (mobile & cible) en **ROUGE** par-dessus."""
        self._clear_edge_highlights()
        data = getattr(self, "_edge_highlights", None)
        if not data:
            return
        # 0) Contours (bleu, fin)
        blue = "#0B3D91"
        for key in ("mob_outline", "tgt_outline"):
            for (a, b) in data.get(key, []):
                _id = self._draw_temp_edge_world(np.array(a, float), np.array(b, float), color=blue, width=2)
                if _id:
                    self._edge_highlight_ids.append(_id)
        # 1) Segments INCIDENTS (candidats possibles connectés au sommet) — plus épais
        for key in ("mob_inc", "tgt_inc"):
            for (a, b) in data.get(key, []):
                _id = self._draw_temp_edge_world(np.array(a, float), np.array(b, float), color=blue, width=4)
                if _id:
                    self._edge_highlight_ids.append(_id)
        # 2) (optionnel) autres candidates calculées — en gris fin si présentes
        for (a, b) in data.get("all", []):
            _id = self._draw_temp_edge_world(np.array(a, float), np.array(b, float), color="#BBBBBB", width=1)
            if _id:
                self._edge_highlight_ids.append(_id)
        # 3) Meilleure (ROUGE, cible + mobile) — toujours par-dessus
        best = data.get("best")
        if best:
            # best = (m_a, m_b, t_a, t_b)
            m_a, m_b, t_a, t_b = best
            # Côté cible (épais)
            _id = self._draw_temp_edge_world(np.array(t_a, float), np.array(t_b, float), color="#FF0000", width=4)
            if _id:
                self._edge_highlight_ids.append(_id)
            # Côté mobile (épais également)
            _id = self._draw_temp_edge_world(np.array(m_a, float), np.array(m_b, float), color="#FF0000", width=4)
            if _id:
                self._edge_highlight_ids.append(_id)

            # côté mobile (fin) — l'arête entrée en contact
            _id2 = self._draw_temp_edge_world(np.array(m_a, float), np.array(m_b, float), color="#FF0000", width=3)
            if _id2:
                self._edge_highlight_ids.append(_id2)

    def _place_dragged_triangle(self):
        """Dépose un triangle ET crée immédiatement un groupe singleton cohérent."""
        if not self._drag or "world_pts" not in self._drag:
            return
        tri = self._drag["triangle"]
        Pw = self._drag["world_pts"]
        # 1) Ajout de l'item dans le document
        self._last_drawn.append({
            "labels": tri["labels"],
            "pts": Pw,
            "id": tri.get("id"),
            "mirrored": tri.get("mirrored", False),
        })
        new_tid = len(self._last_drawn) - 1
        # 2) Création d'un groupe singleton
        self._ensure_group_fields(self._last_drawn[new_tid])
        gid = self._new_group_id()
        self.groups[gid] = {
            "id": gid,
            "nodes": [ {"tid": new_tid, "vkey_in": None, "vkey_out": None} ],
            "bbox": None,
        }
        self._last_drawn[new_tid]["group_id"]  = gid
        self._last_drawn[new_tid]["group_pos"] = 0
        self._recompute_group_bbox(gid)
        # 3) UI
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Triangle déposé → Groupe #{gid} créé.")
        li = self._drag.get("list_index")  # conservé pour compat, même si on ne supprime plus
        # On ne supprime plus l'entrée de la listbox : on marque le triangle comme utilisé
        if tri.get("id") is not None:
            self._placed_ids.add(int(tri["id"]))
            self._update_triangle_listbox_colors()

        # Après le dépôt, on désélectionne le triangle dans la listbox
        # pour ne pas laisser un item actif alors qu'il est déjà utilisé
        self._in_triangle_select_guard = True
        if hasattr(self, "listbox"):
            self.listbox.selection_clear(0, tk.END)
        self._last_triangle_selection = None
        self._in_triangle_select_guard = False

    # =============   Groupes : helpers   ====================
    def _new_group_id(self) -> int:
        gid = self._next_group_id
        self._next_group_id += 1
        return gid

    def _ensure_group_fields(self, tri_dict: Dict) -> None:
        """Idempotent: garantit la présence des champs de groupe sur un triangle."""
        if "group_id" not in tri_dict:
            tri_dict["group_id"] = None
        if "group_pos" not in tri_dict:
            tri_dict["group_pos"] = None

    def _get_group_of_triangle(self, idx: int) -> Optional[int]:
        try:
            return self._last_drawn[idx].get("group_id", None)
        except Exception:
            return None

    def _group_nodes(self, gid: int) -> List[Dict]:
        g = self.groups.get(gid)
        return [] if not g else g["nodes"]

    def _recompute_group_bbox(self, gid: int):
        g = self.groups.get(gid)
        if not g or not g["nodes"]:
            return None
        xs, ys = [], []
        for node in g["nodes"]:
            tid = node["tid"]
            P = self._last_drawn[tid]["pts"]
            for k in ("O", "B", "L"):
                xs.append(float(P[k][0])); ys.append(float(P[k][1]))
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        g["bbox"] = (xmin, ymin, xmax, ymax)
        return g["bbox"]

    def _group_centroid(self, gid: int) -> Optional[np.ndarray]:
        """Barycentre de tous les sommets (O,B,L) du groupe gid."""
        g = self.groups.get(gid)
        if not g or not g.get("nodes"):
            return None
        sx = sy = 0.0
        n = 0
        for node in g["nodes"]:
            tid = node["tid"]
            if not (0 <= tid < len(self._last_drawn)):
                continue
            P = self._last_drawn[tid]["pts"]
            for k in ("O","B","L"):
                sx += float(P[k][0]); sy += float(P[k][1]); n += 1
        if n == 0:
            return None
        return np.array([sx/n, sy/n], dtype=float)

    # utilitaires de déconnexion/scission ---
    def _find_group_link_for_vertex(self, idx: int, vkey: str):
        """
        Si (idx,vkey) est un sommet de LIAISON dans son groupe, retourne
        (gid, pos, link_type) où link_type ∈ {"in","out"}.
        Sinon retourne (None, None, None).
        """
        gid = self._get_group_of_triangle(idx)
        if not gid:
            return (None, None, None)
        nodes = self._group_nodes(gid)
        pos = None
        for i, nd in enumerate(nodes):
            if nd.get("tid") == idx:
                pos = i
                break
        if pos is None:
            return (None, None, None)

        nd = nodes[pos]

        if nd.get("vkey_in") == vkey and pos > 0:
            return (gid, pos, "in")   # lien avec le précédent
        if nd.get("vkey_out") == vkey and pos < len(nodes)-1:
            return (gid, pos, "out")  # lien avec le suivant
        return (None, None, None)

    def _apply_group_meta_after_split_(self, gid: int):
        """Recalcule bbox et group_pos de tous les noeuds du groupe."""
        g = self.groups.get(gid)
        if not g:
            return
        for i, nd in enumerate(g["nodes"]):
            tid = nd["tid"]
            if 0 <= tid < len(self._last_drawn):
                self._last_drawn[tid]["group_id"]  = gid
                self._last_drawn[tid]["group_pos"] = i
        self._recompute_group_bbox(gid)


    def _split_group_at(self, gid: int, split_after_pos: int):
        """
        Scinde le groupe 'gid' ENTRE split_after_pos et split_after_pos+1.
        Retourne (left_gid, right_gid).
        Simplification : on conserve l'invariant « tout triangle appartient à un groupe ».
        - Le groupe de gauche réutilise 'gid' (même singleton).
        - Le groupe de droite reçoit toujours un nouveau gid (même singleton).
        """
        g = self.groups.get(gid)
        if not g:
            return (None, None)
        nodes = g["nodes"]
        if not (0 <= split_after_pos < len(nodes)-1):
            return (gid, None)  # rien à couper

        left_nodes  = [dict(n) for n in nodes[:split_after_pos+1]]
        right_nodes = [dict(n) for n in nodes[split_after_pos+1:]]

        # Corriger les vkey aux frontières : tête/queue
        if left_nodes:
            left_nodes[0]["vkey_in"] = None
            left_nodes[-1]["vkey_out"] = None
        if right_nodes:
            right_nodes[0]["vkey_in"] = None
            right_nodes[-1]["vkey_out"] = None

        # Appliquer côté gauche : réutilise gid (même singleton)
        left_gid = None
        if left_nodes:
            self.groups[gid] = {"id": gid, "nodes": left_nodes, "bbox": None}
            self._apply_group_meta_after_split_(gid)
            left_gid = gid

        # Appliquer côté droit : nouveau gid (même singleton)
        right_gid = None
        if right_nodes:
            new_gid = self._new_group_id()
            self.groups[new_gid] = {"id": new_gid, "nodes": right_nodes, "bbox": None}
            self._apply_group_meta_after_split_(new_gid)
            right_gid = new_gid

        return (left_gid, right_gid)

    def _link_triangles_into_group(self, idx_mob: int, vkey_mob: str, idx_tgt: int, vkey_tgt: str) -> None:
        try:
            tri_m = self._last_drawn[idx_mob]
            tri_t = self._last_drawn[idx_tgt]
        except Exception:
            return
        self._ensure_group_fields(tri_m); self._ensure_group_fields(tri_t)
        if tri_m["group_id"] is not None or tri_t["group_id"] is not None:
            return
        gid = self._new_group_id()
        nodes = [
            {"tid": idx_mob, "vkey_in": None,     "vkey_out": vkey_mob},
            {"tid": idx_tgt, "vkey_in": vkey_tgt, "vkey_out": None},
        ]
        self.groups[gid] = {"id": gid, "nodes": nodes, "bbox": None}
        tri_m["group_id"] = gid; tri_m["group_pos"] = 0
        tri_t["group_id"] = gid; tri_t["group_pos"] = 1
        self._recompute_group_bbox(gid)
        self.status.config(text=f"Groupe #{gid} créé : "
                                f"t0=tid{nodes[0]['tid']}({nodes[0]['vkey_out']}) "
                                f"→ t1=tid{nodes[1]['tid']}({nodes[1]['vkey_in']})")

    # --- Ajout d'un triangle isolé sur une extrémité de groupe ---
    def _append_triangle_to_group(self, gid: int, pos_tgt: int,
                                  idx_new: int, key_new: str, key_tgt: str) -> None:
        """
        Ajoute le triangle 'idx_new' (isolé) au groupe 'gid'.
        - Si pos_tgt == 0 -> PREPEND (le nouveau devient tête, relié à l'ancienne tête)
        - Si pos_tgt == len(nodes)-1 -> APPEND (le nouveau devient queue, relié à l'ancienne queue)
        key_new  : sommet utilisé côté 'nouveau'
        key_tgt  : sommet utilisé côté 'cible' (triangle d'extrémité)
        """
        g = self.groups.get(gid)
        if not g: return
        nodes = g["nodes"]
        n = len(nodes)
        if n == 0: return

        # sécurité : le mobile doit être isolé
        self._ensure_group_fields(self._last_drawn[idx_new])
        if self._last_drawn[idx_new]["group_id"] is not None:
            return  # on ne gère pas l'insertion groupe↔groupe ici

        if pos_tgt == 0:
            # PREPEND : new -> head
            head = nodes[0]
            head["vkey_in"] = key_tgt
            new_node = {"tid": idx_new, "vkey_in": None, "vkey_out": key_new}
            nodes.insert(0, new_node)
        elif pos_tgt == n-1:
            # APPEND : tail -> new
            tail = nodes[-1]
            tail["vkey_out"] = key_tgt
            new_node = {"tid": idx_new, "vkey_in": key_new, "vkey_out": None}
            nodes.append(new_node)
        else:
            # pas une extrémité -> on ne gère pas encore l'insertion au milieu
            return

        # MAJ meta group_id/group_pos et bbox
        for i, nd in enumerate(nodes):
            tid = nd["tid"]
            if 0 <= tid < len(self._last_drawn):
                self._ensure_group_fields(self._last_drawn[tid])
                self._last_drawn[tid]["group_id"]  = gid
                self._last_drawn[tid]["group_pos"] = i
        self._recompute_group_bbox(gid)
        self.status.config(text=f"Triangle ajouté au groupe #{gid}.")

    def _cancel_drag(self):
        # Mémoriser la source du drag avant de le remettre à zéro
        drag_info = self._drag
        self._drag = None

        if self._drag_preview_id is not None:
            self.canvas.delete(self._drag_preview_id)
            self._drag_preview_id = None

        # Toujours remettre le curseur normal quand on annule un drag (ESC ou autre)
        self.canvas.configure(cursor="")

        # Si le drag venait de la liste, on annule aussi la sélection du triangle
        if drag_info and drag_info.get("from") == "list" and hasattr(self, "listbox"):
            self._in_triangle_select_guard = True
            self.listbox.selection_clear(0, tk.END)
            self._in_triangle_select_guard = False
            self._last_triangle_selection = None

    def _on_escape_key(self, event):
        """Annuler un drag&drop (liste) ou un déplacement/selection de triangle (avec rollback)."""
        # Annule les modes compas (arc / mesure azimut / définition azimut ref)
        if self._clock_arc_active:
            self._clock_arc_cancel()
            return

        # Annule le mode de mesure d'azimut du compas
        if self._clock_measure_active:
            self._clock_measure_cancel()
            return

        # Annule le mode de définition d'azimut du compas
        if self._clock_setref_active:
            self._clock_setref_cancel()
            return

        # Annule une calibration en cours
        if self._bg_calib_active:
            self._bg_calibrate_cancel()
            return
 
        if self._drag:
            self._cancel_drag()
            self.status.config(text="Drag annulé (ESC).")
            return

        if self._sel:
            # rollback rotation de GROUPE
            if self._sel.get("mode") == "rotate_group":
                gid = self._sel.get("gid")
                orig = self._sel.get("orig_group_pts")
                if gid is not None and isinstance(orig, dict):
                    for tid, pts in orig.items():
                        if 0 <= tid < len(self._last_drawn):
                            self._last_drawn[tid]["pts"] = {k: np.array(pts[k].copy()) for k in ("O","B","L")}
                    self._recompute_group_bbox(gid)
                    self._redraw_from(self._last_drawn)
                self._sel = None
                self.status.config(text="Rotation de groupe annulée (ESC).")
                self._clear_nearest_line()
                self._clear_edge_highlights()
                return
            # rollback déplacement de GROUPE
            if self._sel.get("mode") == "move_group":
                gid = self._sel.get("gid")
                orig = self._sel.get("orig_group_pts")
                if gid is not None and isinstance(orig, dict):
                    for tid, pts in orig.items():
                        if 0 <= tid < len(self._last_drawn):
                            self._last_drawn[tid]["pts"] = {k: np.array(pts[k].copy()) for k in ("O","B","L")}
                    self._recompute_group_bbox(gid)
                    self._redraw_from(self._last_drawn)
                self._sel = None
                self.status.config(text="Déplacement de groupe annulé (ESC).")
                self._clear_nearest_line()
                self._clear_edge_highlights()
                return
            # rollback déplacement/édition d'un triangle seul
            idx = self._sel.get("idx")
            orig = self._sel.get("orig_pts")
            if idx is not None and orig is not None and 0 <= idx < len(self._last_drawn):
                self._last_drawn[idx]["pts"] = {k: np.array(orig[k].copy()) for k in ("O","B","L")}
                self._redraw_from(self._last_drawn)
            self._sel = None
            self.status.config(text="Action annulée (rollback).")
            self._clear_nearest_line()
            self._clear_edge_highlights()            

#
# ---------- Sélection / déplacement sur canvas ----------
    def _tri_centroid(self, P):
        return np.array([
            (P["O"][0] + P["B"][0] + P["L"][0]) / 3.0,
            (P["O"][1] + P["B"][1] + P["L"][1]) / 3.0
        ], dtype=float)

    def _hit_test(self, x, y):
        """Retourne ('center'|'vertex'|None, idx, extra) selon la zone cliquée.
        - 'vertex' si clic dans un disque autour d'un sommet
        - 'center' si clic à l'intérieur du triangle (hors disques sommets)
        """
        if not self._last_drawn:
            return (None, None, None)
        tol2 = float(self._hit_px) ** 2
        center_tol2 = float(getattr(self, "_center_hit_px", self._hit_px)) ** 2
        # Parcourt les triangles dans l'ordre inverse de dessin (avant-plan d'abord)
        for i in reversed(range(len(self._last_drawn))):
            t = self._last_drawn[i]
            P = t["pts"]
            # 1) tester d'abord les SOMMETS (priorité au mode "vertex")
            for key in ("O", "B", "L"):
                v = P[key]
                vs = np.array(self._world_to_screen(v))
                dv2 = (x - vs[0])**2 + (y - vs[1])**2
                if dv2 <= tol2:
                    return ("vertex", i, key)
            # 2) puis le CENTRE (tolérance plus petite)
            Cw = (P["O"] + P["B"] + P["L"]) / 3.0
            Cs = np.array(self._world_to_screen(Cw))
            ds2 = (x - Cs[0])**2 + (y - Cs[1])**2
            if ds2 <= center_tol2:
                return ("center", i, None)
        return (None, None, None)

    # ---------- Gestion liste déroulante : retrait / réinsertion ----------
    def _reinsert_triangle_to_list(self, tri):
        """Réinsère un triangle supprimé du canvas dans la listbox, trié par id."""
        tri_id = tri.get("id")
        if tri_id is None:
            return
        # Construire le libellé comme au chargement
        labels = tri.get("labels", ("", "", ""))
        b_val = labels[1] if len(labels) > 1 else ""
        l_val = labels[2] if len(labels) > 2 else ""
        entry = f"{int(tri_id):02d}. B:{b_val}  L:{l_val}"

        # Éviter les doublons si déjà présent
        for idx in range(self.listbox.size()):
            try:
                existing = self.listbox.get(idx)
                # on parse l'id au début (2 chiffres)
                ex_id = int(str(existing)[:2])
                if ex_id == int(tri_id):
                    break
            except Exception:
                continue
        else:
            # Trouver la position triée par id
            insert_at = self.listbox.size()
            for idx in range(self.listbox.size()):
                try:
                    ex = self.listbox.get(idx)
                    ex_id = int(str(ex)[:2])
                    if int(tri_id) < ex_id:
                        insert_at = idx
                        break
                except Exception:
                    continue
            self.listbox.insert(insert_at, entry)
        # Dé-marquer l'id comme posé
        if tri_id in self._placed_ids:
            self._placed_ids.discard(int(tri_id))
        # Mettre à jour la couleur des entrées (ce triangle redevient disponible)
        self._update_triangle_listbox_colors()

    # ---------- Clic droit / menu contextuel ----------
    def _on_canvas_right_click(self, event):
        """Affiche le menu contextuel si un triangle est cliqué (intérieur ou sommet)."""
        # Pas de menu si on est en train de drag depuis la liste
        if self._drag:
            return
        # Si clic droit sur le compas : menu dédié (même si aucun triangle)
        if self._is_point_in_clock(event.x, event.y):
            self._ctx_target_idx = None
            self._ctx_last_rclick = (event.x, event.y)
            self._update_compass_ctx_menu_and_dico_state()
            self._ctx_menu_compass.tk_popup(event.x_root, event.y_root)
            self._ctx_menu_compass.grab_release()
            return

        mode, idx, extra = self._hit_test(event.x, event.y)
        if idx is None:
            return
        # On ne propose Supprimer que si on est sur un triangle
        if mode in ("center", "vertex"):
            self._ctx_target_idx = idx
            self._ctx_last_rclick = (event.x, event.y)

            # "OL=0°" et "BL=0°" sont valables aussi sur un groupe :
            # on oriente tout le groupe en prenant le triangle cliqué comme référence.
            if hasattr(self, "_ctx_idx_ol0") and self._ctx_idx_ol0 is not None:
                self._ctx_menu.entryconfig(self._ctx_idx_ol0, state=tk.NORMAL)
            if hasattr(self, "_ctx_idx_bl0") and self._ctx_idx_bl0 is not None:
                self._ctx_menu.entryconfig(self._ctx_idx_bl0, state=tk.NORMAL)

            # (ré)construire la section "mot" du menu selon le triangle visé + selection dico
            self._rebuild_ctx_word_entries()
            self._ctx_menu.tk_popup(event.x_root, event.y_root)
            self._ctx_menu.grab_release()

    def _is_point_in_clock(self, sx: float, sy: float) -> bool:
        """True si (sx,sy) est dans le disque du compas (coords canvas)."""
        if not getattr(self, "show_clock_overlay", None) or not self.show_clock_overlay.get():
            return False
        cx = float(getattr(self, "_clock_cx", None))
        cy = float(getattr(self, "_clock_cy", None))
        R  = float(getattr(self, "_clock_R", getattr(self, "_clock_radius", 69)))
        if cx is None or cy is None:
            return False
        dx = float(sx) - cx
        dy = float(sy) - cy
        return (dx*dx + dy*dy) <= (R + 6.0) * (R + 6.0)

    # ---- Compas : définition interactive de l'azimut de référence -----------------

    def _ctx_define_clock_ref_azimuth(self):
        """Entrée de menu : active le mode 'définir azimut de référence' du compas."""
        # Repartir d'un état propre
        self._clock_setref_cancel(silent=True)

        if not getattr(self, "canvas", None):
            return
        if not getattr(self, "show_clock_overlay", None) or not self.show_clock_overlay.get():
            self.status.config(text="Compas masqué : affiche-le pour définir l'azimut de référence.")
            return

        # Initialiser le mode
        self._clock_setref_active = True
        self.canvas.focus_set()
        sx, sy = self._clock_get_initial_cursor_xy()
        self._clock_setref_update_preview(sx, sy)

        self.status.config(text="Définir azimut de référence : clic gauche pour valider, ESC pour annuler.")


    # ---- Compas : mesure interactive d'un azimut (relatif à l'azimut de référence) ---------
    def _ctx_measure_clock_azimuth(self):
        """Entrée de menu : active le mode 'mesurer un azimut' (relatif à l'azimut de référence)."""
        # Repartir d'un état propre
        self._clock_measure_cancel(silent=True)
        self._clock_setref_cancel(silent=True)

        if not getattr(self, "canvas", None):
            return
        if not getattr(self, "show_clock_overlay", None) or not self.show_clock_overlay.get():
            self.status.config(text="Compas masqué : affiche-le pour mesurer un azimut.")
            return

        self._clock_measure_active = True
        self._clock_measure_last = None
        self.canvas.focus_set()
        sx, sy = self._clock_get_initial_cursor_xy()
        self._clock_measure_update_preview(sx, sy)

        self.status.config(text="Mesurer un azimut : clic gauche pour valider, ESC pour annuler. (Snap noeuds, CTRL = désactiver snap)")


    # ---- Compas : mesure interactive d'un arc d'angle (entre 2 points) -------------------
    def _ctx_measure_clock_arc_angle(self):
        """Entrée de menu : active le mode 'mesurer un arc d'angle' (entre 2 points)."""
        # Repartir d'un état propre
        self._clock_arc_cancel(silent=True)
        self._clock_measure_cancel(silent=True)
        self._clock_setref_cancel(silent=True)

        if not hasattr(self, "canvas") or self.canvas is None:
            return
        if not getattr(self, "show_clock_overlay", None) or not self.show_clock_overlay.get():
            self.status.config(text="Compas masqué : affiche-le pour mesurer un arc d'angle.")
            return

        self._clock_arc_active = True
        self._clock_arc_step = 0
        self._clock_arc_p1 = None
        self._clock_arc_p2 = None
        self._clock_clear_snap_target()
        self.canvas.focus_set()

        self.status.config(text="Mesurer un arc d'angle : clic gauche P1 puis P2, ESC pour annuler. (Snap noeuds, CTRL = désactiver snap)")

    def _ctx_filter_dictionary_by_clock_arc(self):
        """Filtre visuellement le dictionnaire selon l'angle de référence mesuré."""
        if not getattr(self, "dicoSheet", None):
            messagebox.showinfo("Filtrer le dictionnaire", "Le dictionnaire n'est pas affiché.")
            return
        ref = getattr(self, "_clock_arc_last_angle_deg", None)
        if ref is None:
            messagebox.showinfo("Filtrer le dictionnaire", "Aucun arc n'a été mesuré.\n\nMesure d'abord un arc d'angle sur le compas.")
            return
        self._dico_filter_active = True
        self._dico_filter_ref_angle_deg = float(ref)
        self._dico_apply_filter_styles()
        self._update_compass_ctx_menu_and_dico_state()
        self.status.config(text=f"Dico filtré (angle ref={float(ref):0.0f}°, tol=±{float(self._dico_filter_tolerance_deg):0.0f}°)")

    def _clock_arc_is_available(self) -> bool:
        """True si une mesure d'arc persistee est disponible (et affichee tant que le compas reste sur le meme noeud)."""
        return (getattr(self, '_clock_arc_last_angle_deg', None) is not None) and isinstance(getattr(self, '_clock_arc_last', None), dict)

    def _update_compass_ctx_menu_and_dico_state(self):
        """Synchronise le menu compas et l'état de filtrage du dico selon la dispo de l'arc.

        IMPORTANT:
        - Le dico doit rester sélectionnable même s'il n'y a pas d'arc mesuré.
        - Seule l'action "Filtrer le dictionnaire…" dépend de l'existence d'un arc.
        """
        arc_ok = bool(self._clock_arc_is_available())

        menu = getattr(self, '_ctx_menu_compass', None)
        idx = getattr(self, '_ctx_compass_idx_filter_dico', None)
        if menu is not None and idx is not None:
            menu.entryconfig(idx, state=(tk.NORMAL if arc_ok else tk.DISABLED))

        # Activer/désactiver "Annuler le filtrage" selon l'état courant
        idx_cancel = getattr(self, '_ctx_compass_idx_cancel_dico_filter', None)
        if menu is not None and idx_cancel is not None:
            menu.entryconfig(idx_cancel, state=(tk.NORMAL if bool(getattr(self, "_dico_filter_active", False)) else tk.DISABLED))

        # Le dico reste sélectionnable dans tous les cas.
        self._dico_set_selection_enabled(True)

        # Si on perd l'arc alors qu'un filtrage était actif, on annule le filtrage.
        # IMPORTANT: ne pas appeler _simulation_cancel_dictionary_filter() si aucun filtrage n'est actif,
        # sinon recursion infinie (cancel -> update -> cancel -> ...).
        if (not arc_ok) and (bool(getattr(self, "_dico_filter_active", False)) or (getattr(self, "_dico_filter_ref_angle_deg", None) is not None)):
            if hasattr(self, "_simulation_cancel_dictionary_filter"):
                self._simulation_cancel_dictionary_filter()

    def _clock_update_compass_ctx_menu_states(self):
        """Alias historique -> _update_compass_ctx_menu_and_dico_state."""
        self._update_compass_ctx_menu_and_dico_state()

    def _dico_clear_selection(self, refresh: bool = True):
        """Supprime toute selection visible dans la TkSheet du dictionnaire."""
        if not getattr(self, 'dicoSheet', None):
            return

        if hasattr(self.dicoSheet, 'deselect'):
            try:
                self.dicoSheet.deselect('all')
            except Exception:
                pass
            else:
                if refresh and hasattr(self.dicoSheet, 'refresh'):
                    self.dicoSheet.refresh()
                return

        for meth in ('deselect_all', 'delete_selection', 'dehighlight_all'):
            if hasattr(self.dicoSheet, meth):
                try:
                    getattr(self.dicoSheet, meth)()
                except Exception:
                    continue
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

    def _clock_arc_handle_click(self, sx: int, sy: int):
        """Gestion des clics gauche dans le mode 'arc d'angle'."""
        if not getattr(self, "_clock_arc_active", False):
            return

        # Snap sur noeud (CTRL = pas de snap)
        sx2, sy2 = self._clock_apply_optional_snap(int(sx), int(sy), enable_snap=True)

        az_abs = float(self._clock_compute_azimuth_deg(sx2, sy2))

        if int(getattr(self, "_clock_arc_step", 0)) == 0:
            self._clock_arc_p1 = (int(sx2), int(sy2), float(az_abs))
            self._clock_arc_step = 1
            self.status.config(text="Mesurer un arc d'angle : sélectionne le point P2 (clic gauche), ESC pour annuler. (Snap noeuds, CTRL = désactiver snap)")
            # Premier rendu immédiat (au même point)
            self._clock_arc_update_preview(int(sx2), int(sy2))
            return

        # step==1 : valider P2 et conclure
        self._clock_arc_p2 = (int(sx2), int(sy2), float(az_abs))
        az1 = float(self._clock_arc_p1[2])
        az2 = float(self._clock_arc_p2[2])
        angle_deg = float(self._clock_arc_compute_angle_deg(az1, az2))

        # Persister la mesure : on l'affichera lors des redraw de l'overlay.
        self._clock_arc_last = {"az1": az1, "az2": az2, "angle": angle_deg}
        self._clock_arc_last_angle_deg = float(angle_deg)
        self._update_compass_ctx_menu_and_dico_state()

        # Sortir du mode interactif en nettoyant uniquement le preview
        self._clock_arc_cancel(silent=True)
        self._redraw_overlay_only()
        self.status.config(text=f"Arc mesuré : {angle_deg:0.0f}°")

    def _clock_arc_update_preview(self, sx: int, sy: int):
        """Met à jour le preview (2 rayons + arc + label) pendant la sélection de P2."""
        if not getattr(self, "_clock_arc_active", False):
            return
        if int(getattr(self, "_clock_arc_step", 0)) != 1:
            return
        if self._clock_arc_p1 is None:
            return
        if not hasattr(self, "canvas") or self.canvas is None:
            return

        cx = float(getattr(self, "_clock_cx", 0.0))
        cy = float(getattr(self, "_clock_cy", 0.0))
        R  = float(getattr(self, "_clock_R", getattr(self, "_clock_radius", 69)))

        # Snap P2 si activé
        sx2, sy2 = self._clock_apply_optional_snap(int(sx), int(sy), enable_snap=True)
        az2 = float(self._clock_compute_azimuth_deg(sx2, sy2))
        az1 = float(self._clock_arc_p1[2])

        # 2 rayons (centre->P1 et centre->P2)
        x1, y1 = int(self._clock_arc_p1[0]), int(self._clock_arc_p1[1])
        if self._clock_arc_line1_id is None:
            self._clock_arc_line1_id = self.canvas.create_line(
                cx, cy, x1, y1, width=2, dash=(4, 3),
                fill="#202020", tags=("clock_overlay", "clock_arc_preview")
            )
        else:
            self.canvas.coords(self._clock_arc_line1_id, cx, cy, x1, y1)

        if self._clock_arc_line2_id is None:
            self._clock_arc_line2_id = self.canvas.create_line(
                cx, cy, sx2, sy2, width=2, dash=(4, 3),
                fill="#202020", tags=("clock_overlay", "clock_arc_preview")
            )
        else:
            self.canvas.coords(self._clock_arc_line2_id, cx, cy, sx2, sy2)

        # Arc (plus petit arc entre az1 et az2)
        start_deg_tk, extent_deg_tk, angle_deg, mid_az = self._clock_arc_compute_tk_arc(az1, az2)

        bbox = (cx - R, cy - R, cx + R, cy + R)
        if self._clock_arc_arc_id is None:
            self._clock_arc_arc_id = self.canvas.create_arc(
                *bbox,
                start=float(start_deg_tk),
                extent=float(extent_deg_tk),
                style="arc",
                width=2,
                outline="#202020",
                tags=("clock_overlay", "clock_arc_preview")
            )
        else:
            self.canvas.coords(self._clock_arc_arc_id, *bbox)
            self.canvas.itemconfig(self._clock_arc_arc_id, start=float(start_deg_tk), extent=float(extent_deg_tk))

        # Texte angle, placé près du milieu de l'arc (légèrement à l'extérieur)
        tx, ty = self._clock_point_on_circle(mid_az, R * 1.08)
        label = f"{angle_deg:0.0f}°"
        if self._clock_arc_text_id is None:
            self._clock_arc_text_id = self.canvas.create_text(
                tx, ty, text=label, anchor="center",
                fill="#202020", font=("Arial", 12, "bold"),
                tags=("clock_overlay", "clock_arc_preview")
            )
        else:
            self.canvas.itemconfig(self._clock_arc_text_id, text=label)
            self.canvas.coords(self._clock_arc_text_id, tx, ty)

    def _clock_arc_cancel(self, silent: bool=False):
        if not getattr(self, "_clock_arc_active", False):
            return
        self._clock_arc_active = False
        self._clock_arc_step = 0
        self._clock_arc_p1 = None
        self._clock_arc_p2 = None

        # Nettoyage items (pas de try/except silencieux : on veut voir les erreurs)
        for attr in ("_clock_arc_line1_id", "_clock_arc_line2_id", "_clock_arc_arc_id", "_clock_arc_text_id"):
            item_id = getattr(self, attr, None)
            if item_id is not None and getattr(self, "canvas", None):
                self.canvas.delete(item_id)
            setattr(self, attr, None)

        self._clock_clear_snap_target()
        if not silent:
            self.status.config(text="Mesure d'arc annulée.")

    def _clock_arc_clear_last(self):
        """Efface la dernière mesure persistée (appelé notamment quand le compas bouge)."""
        # Si le dico était filtré sur la base de cet arc, on doit annuler le filtrage
        # dès que l'arc n'est plus disponible (changement de noeud compas).
        if bool(getattr(self, "_dico_filter_active", False)) or (getattr(self, "_dico_filter_ref_angle_deg", None) is not None):
            # Réutilise l'action standard de l'app (menu simulation "Annuler filtrage")
            if hasattr(self, "_simulation_cancel_dictionary_filter"):
                self._simulation_cancel_dictionary_filter()
            else:
                # fallback minimal (au cas où)
                self._dico_filter_active = False
                self._dico_filter_ref_angle_deg = None
                self._dico_clear_filter_styles()

        self._clock_arc_last = None
        self._clock_arc_last_angle_deg = None
        self._update_compass_ctx_menu_and_dico_state()

    def _clock_arc_auto_from_snap_target(self, snap_tgt: dict) -> bool:
        """Calcule automatiquement un arc (EXT) quand le compas s'accroche à un noeud.

        Objectif : reproduire "Mesure un arc d'angle" sans interaction utilisateur,
       en utilisant les 2 directions de frontière *extérieures* au point d'accroche.

        Retourne True si une mesure a été calculée/persistée.

        Remarques:
        - Le point d'accroche peut être un sommet *ou* un point sur une arête (noeud connecté à une arête).
        - En cas d'ambiguïté (>2 arêtes incidentes), on applique la même heuristique que
          _incident_half_edges_at_vertex (2 extrêmes angulaires).
        """
        if not isinstance(snap_tgt, dict) or snap_tgt.get("world") is None:
            return False

        idx = int(snap_tgt.get("idx"))
        v_world = snap_tgt.get("world")
        if not (0 <= idx < len(self._last_drawn)):
            return False

        neigh = self._clock_arc_auto_get_two_neighbors(idx, v_world, outline_eps=None)
        if not neigh:
            return False

        w1, w2 = neigh

        # Récupérer 2 directions (azimuts) depuis le point d'ancrage
        az1 = float(self._azimuth_world_deg(v_world, w1))
        az2 = float(self._azimuth_world_deg(v_world, w2))
        angle_deg = float(self._clock_arc_compute_angle_deg(az1, az2))

        self._clock_arc_last = {"az1": az1, "az2": az2, "angle": angle_deg}
        self._clock_arc_last_angle_deg = float(angle_deg)
        self._update_compass_ctx_menu_and_dico_state()
        self.status.config(text=f"Arc auto (EXT) : {angle_deg:0.0f}°")
        return True


    def _azimuth_world_deg(self, a, b) -> float:
        """Azimut absolu en degrés (0°=Nord, 90°=Est) entre 2 points monde."""
        import math
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dx, dy = (bx - ax), (by - ay)
        # dx vers Est, dy vers Nord (monde Lambert-like). Convertir en azimut.
        ang = math.degrees(math.atan2(dx, dy)) % 360.0
        return float(ang)


    def _clock_arc_draw_last(self, cx: float, cy: float, R: float):
        """Dessine la dernière mesure persistée (si présente) dans l'overlay."""
        last = getattr(self, "_clock_arc_last", None)
        if not isinstance(last, dict):
            return
        if not getattr(self, "canvas", None):
            return
        try:
            az1 = float(last.get("az1"))
            az2 = float(last.get("az2"))
        except Exception:
            return

        start_deg_tk, extent_deg_tk, angle_deg, mid_az = self._clock_arc_compute_tk_arc(az1, az2)

        # 2 rayons (centre->az1 et centre->az2)
        x1, y1 = self._clock_point_on_circle(az1, R)
        x2, y2 = self._clock_point_on_circle(az2, R)
        self.canvas.create_line(
            cx, cy, x1, y1, width=2, dash=(4, 3),
            fill="#202020", tags=("clock_overlay", "clock_arc_persist")
        )
        self.canvas.create_line(
            cx, cy, x2, y2, width=2, dash=(4, 3),
            fill="#202020", tags=("clock_overlay", "clock_arc_persist")
        )

        # Arc
        bbox = (cx - R, cy - R, cx + R, cy + R)
        self.canvas.create_arc(
            *bbox,
            start=float(start_deg_tk),
            extent=float(extent_deg_tk),
            style="arc",
            width=2,
            outline="#202020",
            tags=("clock_overlay", "clock_arc_persist"),
        )

        # Texte angle
        tx, ty = self._clock_point_on_circle(mid_az, R * 1.08)
        self.canvas.create_text(
            tx, ty,
            text=f"{float(angle_deg):0.0f}°",
            anchor="center",
            fill="#202020",
            font=("Arial", 12, "bold"),
            tags=("clock_overlay", "clock_arc_persist"),
        )

    def _clock_arc_compute_angle_deg(self, az1: float, az2: float) -> float:
        """Retourne le plus petit angle (0..180) entre deux azimuts."""
        d = (float(az2) - float(az1)) % 360.0
        if d > 180.0:
            d = 360.0 - d
        return float(d)

    def _clock_arc_compute_tk_arc(self, az1: float, az2: float):
        """Prépare les paramètres Tk (start/extent) pour dessiner le plus petit arc entre az1 et az2.

        Retourne (start_deg_tk, extent_deg_tk, angle_deg, mid_az).
        - start/extent sont dans le repère Tk (0° à 3h, CCW+)
        - mid_az est l'azimut (0°=Nord, horaire) au milieu de l'arc choisi, utile pour placer le label.
        """
        a1 = float(az1) % 360.0
        a2 = float(az2) % 360.0

        # delta horaire (cw) de a1 vers a2
        d_cw = (a2 - a1) % 360.0
        d_ccw = (a1 - a2) % 360.0  # delta si on va anti-horaire

        # Choisir le plus petit arc
        if d_cw <= d_ccw:
            # arc horaire de taille d_cw : Tk extent doit être négatif
            angle = d_cw
            start_az = a1
            mid_az = (a1 + angle * 0.5) % 360.0
            start_tk = (90.0 - start_az) % 360.0
            extent_tk = -angle
        else:
            # arc anti-horaire de taille d_ccw : Tk extent positif
            angle = d_ccw
            start_az = a1
            mid_az = (a1 - angle * 0.5) % 360.0
            start_tk = (90.0 - start_az) % 360.0
            extent_tk = angle

        return (float(start_tk), float(extent_tk), float(angle), float(mid_az))

    def _clock_point_on_circle(self, az_deg: float, radius: float):
        """Point écran (sx,sy) à un azimut donné autour du centre du compas."""
        cx = float(getattr(self, "_clock_cx", 0.0))
        cy = float(getattr(self, "_clock_cy", 0.0))
        a = math.radians(float(az_deg) % 360.0)
        sx = cx + float(radius) * math.sin(a)
        sy = cy - float(radius) * math.cos(a)
        return (sx, sy)

    def _clock_apply_optional_snap(self, sx: int, sy: int, *, enable_snap: bool) -> Tuple[int, int]:
        """Applique le snap sur noeud si activé (et si CTRL n'est pas pressé)."""
        sx2, sy2 = int(sx), int(sy)
        if not enable_snap:
            return sx2, sy2
        if getattr(self, "_ctrl_down", False):
            self._clock_clear_snap_target()
            return sx2, sy2

        self._clock_update_snap_target(float(sx2), float(sy2))
        tgt = getattr(self, "_clock_snap_target", None)
        if isinstance(tgt, dict) and tgt.get("world") is not None:
            sxx, syy = self._world_to_screen(tgt["world"])
            return (int(sxx), int(syy))
        return sx2, sy2

    def _clock_measure_update_preview(self, sx: int, sy: int):
        if not self._clock_measure_active:
            return
        self._clock_measure_line_id, self._clock_measure_text_id, out = self._clock_update_azimuth_preview(
            int(sx), int(sy),
            line_id=self._clock_measure_line_id,
            text_id=self._clock_measure_text_id,
            preview_tag="clock_measure_preview",
            relative_to_ref=True,
            enable_snap=True,
        )
        if out:
            sx2, sy2, az_abs, az_rel = out
            self._clock_measure_last = (int(sx2), int(sy2), float(az_abs), float(az_rel))

    def _clock_measure_confirm(self):
        if not self._clock_measure_active:
            return
        last = self._clock_measure_last
        if not last:
            self._clock_measure_cancel(silent=True)
            return
        (_, _, az_abs, az_rel) = last
        self._clock_measure_cancel(silent=True)
        self.status.config(text=f"Azimut mesuré : {az_rel:0.0f}° (ref={float(getattr(self, '_clock_ref_azimuth_deg', 0.0))%360.0:0.0f}°, abs={az_abs:0.0f}°)")


    def _clock_measure_cancel(self, silent: bool=False):
        if not getattr(self, "_clock_measure_active", False):
            return
        self._clock_measure_active = False

        if getattr(self, "_clock_measure_line_id", None) is not None:
            try:
                self.canvas.delete(self._clock_measure_line_id)
            except Exception:
                pass
        if getattr(self, "_clock_measure_text_id", None) is not None:
            try:
                self.canvas.delete(self._clock_measure_text_id)
            except Exception:
                pass

        self._clock_measure_line_id = None
        self._clock_measure_text_id = None
        self._clock_measure_last = None
        self._clock_clear_snap_target()

        if not silent:
            self.status.config(text="Mesure d'azimut annulée.")

    def _clock_compute_azimuth_deg(self, sx: float, sy: float) -> float:
        """Azimut (degrés) depuis le centre du compas vers (sx,sy). 0°=Nord, sens horaire."""
        cx = float(getattr(self, "_clock_cx", 0.0))
        cy = float(getattr(self, "_clock_cy", 0.0))
        vx = float(sx) - cx
        vy = float(sy) - cy
        # Tk: y vers le bas -> Nord = -y
        ang = math.degrees(math.atan2(vx, -vy))  # 0=N, 90=E, 180=S, 270=O
        ang = ang % 360.0
        return ang


    def _clock_angle_diff_deg(self, a: float, b: float) -> float:
        """Différence angulaire minimale |a-b| sur un cercle (en degrés), résultat dans [0..180]."""
        da = (float(a) - float(b)) % 360.0
        if da > 180.0:
            da = 360.0 - da
        return abs(da)

    def _clock_compute_theoretical_ref_azimuth_deg(
        self,
        *,
        az1: float,
        az2: float,
        ang_hour_0: float,
        ang_min_0: float,
    ) -> float:
        """Calcule l'azimut absolu (0=N) du '12h' qui ferait coïncider les aiguilles
        (heure/minute) avec les 2 droites mesurées (az1/az2).

        On teste les 2 correspondances possibles (heure->az1, minute->az2) et (heure->az2, minute->az1)
        et on retient celle qui minimise l'erreur résiduelle.
        """
        a1 = float(az1) % 360.0
        a2 = float(az2) % 360.0
        h0 = float(ang_hour_0) % 360.0
        m0 = float(ang_min_0) % 360.0

        # Hypothèse 1 : heure->a1, minute->a2
        ref1 = (a1 - h0) % 360.0
        pred_m1 = (ref1 + m0) % 360.0
        err1 = self._clock_angle_diff_deg(pred_m1, a2)

        # Hypothèse 2 : heure->a2, minute->a1
        ref2 = (a2 - h0) % 360.0
        pred_m2 = (ref2 + m0) % 360.0
        err2 = self._clock_angle_diff_deg(pred_m2, a1)

        return float(ref1 if err1 <= err2 else ref2)

    # ---------- Compas : helpers communs (éviter duplication setref/measure) ----------
    def _clock_get_initial_cursor_xy(self) -> Tuple[int, int]:
        """Point de départ pour les modes compas: clic droit si dispo, sinon position souris."""
        if getattr(self, "_ctx_last_rclick", None):
            sx, sy = self._ctx_last_rclick
            return int(sx), int(sy)
        # coords canvas sous la souris
        sx = int(self.canvas.winfo_pointerx() - self.canvas.winfo_rootx())
        sy = int(self.canvas.winfo_pointery() - self.canvas.winfo_rooty())
        return sx, sy

    def _clock_update_azimuth_preview(
        self,
        sx: int,
        sy: int,
        *,
        line_id: Optional[int],
        text_id: Optional[int],
        preview_tag: str,
        relative_to_ref: bool,
        enable_snap: bool,
    ) -> Tuple[Optional[int], Optional[int], Optional[Tuple[int, int, float, float]]]:
        """Rendu commun (ligne centre->point + texte azimut clampé).

        Retourne (new_line_id, new_text_id, (sx2, sy2, az_abs, az_val)).

        Notes:
        - Cette fonction NE vérifie PAS si un mode est actif : le caller doit le faire.
        - `preview_tag` est un tag spécifique (en plus de 'clock_overlay') pour identifier le preview.
        """
        # Canvas / centre compas doivent exister
        if self.canvas is None:
            return (line_id, text_id, None)
        if self._clock_cx is None or self._clock_cy is None:
            return (line_id, text_id, None)

        cx = float(self._clock_cx)
        cy = float(self._clock_cy)

        sx2, sy2 = int(sx), int(sy)

        # Snap optionnel (CTRL => pas de snap)
        if enable_snap:
            if self._ctrl_down:
                self._clock_clear_snap_target()
            else:
                self._clock_update_snap_target(float(sx), float(sy))
                tgt = self._clock_snap_target
                if isinstance(tgt, dict) and tgt.get("world") is not None:
                    sxx, syy = self._world_to_screen(tgt["world"])
                    sx2, sy2 = int(sxx), int(syy)

        # Ligne centre -> point

        if line_id is None:
            line_id = self.canvas.create_line(
                cx, cy, sx2, sy2, width=2, dash=(4, 3),
                fill="#202020", tags=("clock_overlay", preview_tag)
            )
        else:
            self.canvas.coords(line_id, cx, cy, sx2, sy2)

        # Calcul azimut
        az_abs = float(self._clock_compute_azimuth_deg(sx2, sy2))
        if relative_to_ref:
            ref_az = float(self._clock_ref_azimuth_deg) % 360.0
            az_val = (az_abs - ref_az) % 360.0
        else:
            az_val = az_abs
        label = f"{az_val:0.0f}°"

        # Clamp du texte pour rester visible
        cw = int(self.canvas.winfo_width() or 0)
        ch = int(self.canvas.winfo_height() or 0)
        tx = int(sx2 + 14)
        ty = int(sy2 + 10)
        pad = 6
        est_w = 60
        est_h = 20
        if cw > 0:
            if tx + est_w > cw - pad:
                tx = int(sx2 - est_w - 14)
            tx = max(pad, min(tx, cw - est_w - pad))
        if ch > 0:
            if ty + est_h > ch - pad:
                ty = int(sy2 - est_h - 10)
            ty = max(pad, min(ty, ch - est_h - pad))

        if text_id is None:
            text_id = self.canvas.create_text(
                tx, ty, text=label, anchor="nw",
                fill="#202020", font=("Arial", 12, "bold"),
                tags=("clock_overlay", preview_tag)
            )
        else:
            self.canvas.itemconfig(text_id, text=label)
            self.canvas.coords(text_id, tx, ty)

        return (line_id, text_id, (sx2, sy2, az_abs, float(az_val)))

    def _clock_setref_update_preview(self, sx: int, sy: int):
        if not self._clock_setref_active:
            return
        self._clock_setref_line_id, self._clock_setref_text_id, out = self._clock_update_azimuth_preview(
            int(sx), int(sy),
            line_id=self._clock_setref_line_id,
            text_id=self._clock_setref_text_id,
            preview_tag="clock_ref_preview",
            relative_to_ref=False,
            enable_snap=False,
        )
        if out:
            sx2, sy2, az_abs, az_val = out
            self._clock_setref_last = (int(sx2), int(sy2), float(az_abs), float(az_val))

    def _clock_setref_confirm(self, sx: int, sy: int):
        if not self._clock_setref_active:
            return
        az = self._clock_compute_azimuth_deg(sx, sy)
        self._clock_ref_azimuth_deg = float(az)
        self.setAppConfigValue("uiClockRefAzimuth", float(self._clock_ref_azimuth_deg))

        self._clock_setref_cancel(silent=True)
        self.status.config(text=f"Azimut de référence défini : {az:0.0f}°")

        self._redraw_overlay_only()

    def _clock_setref_cancel(self, silent: bool=False):
        if not self._clock_setref_active:
            return
        self._clock_setref_active = False

        if self._clock_setref_line_id is not None:
            self.canvas.delete(self._clock_setref_line_id)

        if self._clock_setref_text_id is not None:
            self.canvas.delete(self._clock_setref_text_id)

        self._clock_setref_line_id = None
        self._clock_setref_text_id = None
        # On laisse l'overlay se redessiner via les flux habituels
        if not silent:
            self.status.config(text="Définition d'azimut annulée.")


    def _ctx_delete_group(self):
        """Supprime **tout le groupe** du triangle ciblé, réinsère les triangles dans la liste,
        remappe les tids restants et recalcule les bbox de chaque groupe."""
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None or not (0 <= idx < len(self._last_drawn)):
            return
        gid = self._get_group_of_triangle(idx)
        if not gid:
            return
        g = self.groups.get(gid)
        if not g or not g.get("nodes"):
            return

        # --- Confirmation si le groupe comporte au moins 2 triangles ---
        try:
            n_nodes = len(g.get("nodes", []))
        except Exception:
            n_nodes = 0
        if n_nodes >= 2:
            confirm = messagebox.askyesno(
                "Supprimer le groupe",
                "Voulez-vous supprimer le groupe ?"
            )
            if not confirm:
                return

        # 1) Réinsertion listbox (avant de modifier _last_drawn)
        removed_tids = sorted({nd["tid"] for nd in g["nodes"] if "tid" in nd}, reverse=True)
        removed_set  = set(removed_tids)
        for tid in removed_tids:
            if 0 <= tid < len(self._last_drawn):
                self._reinsert_triangle_to_list(self._last_drawn[tid])

        # 2) Reconstruire _last_drawn et table de remap old->new
        keep, old2new = [], {}
        for old_i, tri in enumerate(self._last_drawn):
            if old_i in removed_set:
                continue
            new_i = len(keep)
            old2new[old_i] = new_i
            keep.append(tri)
        # IMPORTANT : on remplace le contenu de la liste en place
        # pour ne pas casser la référence scen.last_drawn.
        self._last_drawn[:] = keep

        # 3) Supprimer le groupe et remapper les autres
        if gid in self.groups:
            del self.groups[gid]
        for g2 in list(self.groups.values()):
            new_nodes = []
            for nd in g2.get("nodes", []):
                otid = nd.get("tid")
                if otid in removed_set:
                    continue
                nd2 = dict(nd)
                nd2["tid"] = old2new.get(otid, otid)
                new_nodes.append(nd2)
            g2["nodes"] = new_nodes
            # Reposer group_id/group_pos cohérents
            for i, nd in enumerate(g2["nodes"]):
                tid = nd["tid"]
                if 0 <= tid < len(self._last_drawn):
                    self._last_drawn[tid]["group_id"]  = g2["id"]
                    self._last_drawn[tid]["group_pos"] = i
            self._recompute_group_bbox(g2["id"])

        # 4) Fin : purge sélection/assist et redraw
        self._sel = None
        self._reset_assist()
        self._redraw_from(self._last_drawn)
        try:
            self.status.config(text=f"Groupe supprimé (gid={gid}, {len(removed_tids)} triangle(s)).")
        except Exception:
            pass

    def _ctx_rotate_selected(self):
        """Passe en mode rotation autour du barycentre pour le triangle ciblé."""
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None or not (0 <= idx < len(self._last_drawn)):
            return
        # Si le triangle fait partie d'un groupe : PIVOTER LE GROUPE
        gid = self._get_group_of_triangle(idx)
        if gid:
            pivot = self._group_centroid(gid)
            if pivot is None:
                return
            # snapshot complet du groupe pour rollback
            orig_group_pts = {}
            for node in self._group_nodes(gid):
                tid = node["tid"]
                if 0 <= tid < len(self._last_drawn):
                    Pt = self._last_drawn[tid]["pts"]
                    orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O","B","L")}
            # angle de départ = angle (pivot -> curseur au clic droit)
            if self._ctx_last_rclick:
                sx, sy = self._ctx_last_rclick
            else:
                sx, sy = self._world_to_screen(pivot)
            wx = (sx - self.offset[0]) / self.zoom
            wy = (self.offset[1] - sy) / self.zoom
            start_angle = math.atan2(wy - pivot[1], wx - pivot[0])
            self._sel = {
                "mode": "rotate_group",
                "gid": gid,
                "orig_group_pts": orig_group_pts,
                "pivot": np.array(pivot, dtype=float),
                "start_angle": start_angle,
            }
            self.status.config(text=f"Mode pivoter GROUPE #{gid} : bouge la souris pour tourner, clic gauche pour valider, ESC pour annuler.")
            return
        # Sinon : rotation TRIANGLE seul (comportement existant)
        P = self._last_drawn[idx]["pts"]
        pivot = self._tri_centroid(P)
        orig_pts = {k: np.array(P[k].copy()) for k in ("O","B","L")}
        if self._ctx_last_rclick:
            sx, sy = self._ctx_last_rclick
        else:
            sx, sy = self._world_to_screen(pivot)
        wx = (sx - self.offset[0]) / self.zoom
        wy = (self.offset[1] - sy) / self.zoom
        start_angle = math.atan2(wy - pivot[1], wx - pivot[0])
        self._sel = {
            "mode": "rotate",
            "idx": idx,
            "orig_pts": orig_pts,
            "pivot": np.array(pivot, dtype=float),
            "start_angle": start_angle,
        }
        self.status.config(text="Mode pivoter : bouge la souris pour tourner, clic gauche pour valider, ESC pour annuler.")
 
    def _ctx_orient_segment_north(self, from_key: str, to_key: str, status_label: str):
        """
        Oriente automatiquement le TRIANGLE ou le GROUPE pour que l'azimut du segment
        (from_key -> to_key) du triangle cliqué soit 0° = vers le Nord (axe +Y en coords monde).

        - Triangle seul : rotation autour du barycentre du triangle cliqué.
        - Groupe : rotation RIGIDE de tout le groupe autour du barycentre du triangle cliqué.
        """
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None or not (0 <= idx < len(self._last_drawn)):
            return

        # Déterminer le groupe (si présent)
        gid = self._get_group_of_triangle(idx)

        # Triangle de référence
        P_ref = self._last_drawn[idx]["pts"]
        v = np.array(
            [P_ref[to_key][0] - P_ref[from_key][0], P_ref[to_key][1] - P_ref[from_key][1]],
            dtype=float
        )

        if float(np.hypot(v[0], v[1])) < 1e-12:
            return  # triangle dégénéré, rien à faire

        # Orientation cible : Nord = +Y
        cur = math.atan2(v[1], v[0])
        target = math.pi / 2.0
        dtheta = target - cur
        c, s = math.cos(dtheta), math.sin(dtheta)
        R = np.array([[c, -s], [s, c]], dtype=float)

        # Pivot = barycentre du triangle cliqué (cohérent avec "Pivoter")
        pivot = self._tri_centroid(P_ref)
        pivot = np.array(pivot, dtype=float)

        def rot_point(pt):
            pt = np.array(pt, dtype=float)
            return (R @ (pt - pivot)) + pivot

        # Appliquer à tout le groupe si groupé, sinon au seul triangle
        if gid and len(self._group_nodes(gid) or []) > 1:
            g = self.groups.get(gid)
            if not g:
                return
            for node in g["nodes"]:
                tid = node.get("tid")
                if tid is None or not (0 <= tid < len(self._last_drawn)):
                    continue
                Pt = self._last_drawn[tid]["pts"]
                for k in ("O", "B", "L"):
                    Pt[k] = rot_point(Pt[k])
            self._recompute_group_bbox(gid)
            self._redraw_from(self._last_drawn)
            self.status.config(text=f"Orientation appliquée : GROUPE — {status_label} au Nord (0°).")
        else:
            P = P_ref
            for k in ("O", "B", "L"):
                P[k] = rot_point(P[k])
            self._redraw_from(self._last_drawn)
            self.status.config(text=f"Orientation appliquée : {status_label} au Nord (0°).")

    def _ctx_orient_OL_north(self):
        return self._ctx_orient_segment_north("O", "L", "O→L")

    def _ctx_orient_BL_north(self):
        return self._ctx_orient_segment_north("B", "L", "B→L")

    def _ctx_filter_scenarios(self):
        """
        Filtre les scénarios automatiques en conservant uniquement ceux
        dont la chaîne (ordre + arêtes utilisées entre triangles consécutifs)
        correspond au préfixe validé dans le scénario actif.

        Le scénario actif est la référence absolue et ne peut jamais être supprimé.
        """
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None or not (0 <= idx < len(self._last_drawn)):
            return

        try:
            clicked_tid = int(self._last_drawn[idx].get("id"))
        except Exception:
            return

        ok = messagebox.askyesno(
            "Filtrer les scénarios",
            "Cette action va supprimer définitivement les scénarios automatiques incompatibles.\n\nContinuer ?"
        )
        if not ok:
            return

        self._filter_auto_scenarios_by_prefix_edges(clicked_tid)

    def _scenario_prefix_edge_steps(self, scen, upto_index: int):
        """
        Retourne la liste des (edge_out, edge_in) pour les liaisons entre
        triangles consécutifs, de 0->1->...->upto_index.
        - upto_index est INCLUS côté triangle, donc il y a upto_index étapes.
        """
        if scen is None:
            return None
        groups = getattr(scen, "groups", None)
        if not groups:
            return None

        # prendre la chaîne la plus longue (design actuel = un seul groupe)
        best_nodes = None
        for g in (groups or {}).values():
            nodes = (g or {}).get("nodes") or []
            if best_nodes is None or len(nodes) > len(best_nodes):
                best_nodes = nodes
        nodes = best_nodes or []
        if len(nodes) < upto_index + 1:
            return None

        # fallback si edge_* absent : déduction via vkey_* (sommet opposé à l'arête)
        opp_edge = {"O": "BL", "B": "LO", "L": "OB"}

        steps = []
        for i in range(upto_index):
            a = nodes[i] or {}
            b = nodes[i + 1] or {}

            eout = a.get("edge_out")
            if not eout:
                vout = a.get("vkey_out")
                eout = opp_edge.get(vout) if vout else None

            ein = b.get("edge_in")
            if not ein:
                vin = b.get("vkey_in")
                ein = opp_edge.get(vin) if vin else None

            steps.append((eout, ein))
        return steps

    def _filter_auto_scenarios_by_prefix_edges(self, clicked_tid: int):
        """
        Filtre les scénarios automatiques en conservant uniquement ceux
        qui commencent par le même préfixe de triangles ET les mêmes arêtes
        (edge_out / edge_in) entre triangles consécutifs, jusqu'au triangle cliqué.

        Le scénario actif est la référence absolue et ne peut jamais être supprimé.
        """
        if not self.scenarios or not (0 <= self.active_scenario_index < len(self.scenarios)):
            return

        active = self.scenarios[self.active_scenario_index]
        if getattr(active, "source_type", "manual") != "auto":
            return  # filtrage auto uniquement

        tri_ids = list(getattr(active, "tri_ids", None) or [])
        if clicked_tid not in tri_ids:
            # Impossible par design → bug
            raise RuntimeError(f"[FILTER] Triangle {clicked_tid} absent du scénario actif")

        idx = tri_ids.index(clicked_tid)
        prefix = tri_ids[: idx + 1]

        ref_steps = self._scenario_prefix_edge_steps(active, idx)
        if ref_steps is None:
            raise RuntimeError("[FILTER] Données d'arêtes manquantes dans le scénario actif (groups/nodes)")

        kept = []
        removed = 0

        for scen in self.scenarios:
            # Le scénario actif est TOUJOURS conservé
            if scen is active:
                kept.append(scen)
                continue

            if getattr(scen, "source_type", "manual") != "auto":
                kept.append(scen)
                continue

            # 1) même préfixe de triangles (ordre)
            if list(getattr(scen, "tri_ids", None) or [])[: len(prefix)] != prefix:
                removed += 1
                continue

            # 2) mêmes arêtes utilisées pour chaîner le préfixe
            steps = self._scenario_prefix_edge_steps(scen, idx)
            if steps != ref_steps:
                removed += 1
                continue

            kept.append(scen)

        if active not in kept:
            raise RuntimeError("[FILTER] Le scénario actif a été supprimé (BUG)")

        self.scenarios = kept
        self.active_scenario_index = self.scenarios.index(active)

        self._refresh_scenario_listbox()
        self._set_active_scenario(self.active_scenario_index)

    def _ctx_flip_selected(self):
        """
        Inverse **tout le GROUPE** par symétrie axiale rigide.
        Axe = direction (O→L) du triangle ciblé ; la droite passe par le **barycentre du groupe**.
        Chaque triangle du groupe garde sa forme; on bascule le flag 'mirrored' de tous.
        """
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None or not (0 <= idx < len(self._last_drawn)):
            return
        # --- Étendue : GROUPE du triangle ciblé ---
        gid = self._get_group_of_triangle(idx)
        if not gid:
            return
        nodes = self._group_nodes(gid)  # liste des triangles du groupe
        if not nodes:
            return

        # Axe = O→L du triangle ciblé (en monde)
        t0 = self._last_drawn[idx]
        P0 = t0["pts"]
        axis = np.array([P0["L"][0] - P0["O"][0], P0["L"][1] - P0["O"][1]], dtype=float)
        nrm = float(np.hypot(axis[0], axis[1]))
        if nrm < 1e-12:
            return  # triangle dégénéré
        u = axis / nrm

        # Pivot = barycentre du GROUPE (rigide pour tous les triangles)
        pivot = self._group_centroid(gid)
        if pivot is None:
            # secours : barycentre du triangle ciblé
            pivot = self._tri_centroid(P0)

        # Matrice de réflexion autour de la droite passant par 'pivot' et dirigée par u
        R = np.array([[2*u[0]*u[0] - 1, 2*u[0]*u[1]],
                      [2*u[0]*u[1],     2*u[1]*u[1] - 1]], dtype=float)

        # Appliquer la réflexion à TOUS les triangles du groupe
        for nd in nodes:
            tid = nd.get("tid")
            if tid is None or not (0 <= tid < len(self._last_drawn)):
                continue
            Pt = self._last_drawn[tid]["pts"]
            for k in ("O", "B", "L"):
                v = np.array([Pt[k][0] - pivot[0], Pt[k][1] - pivot[1]], dtype=float)
                Pt[k] = (R @ v) + pivot
            # Toggle du flag 'mirrored' pour l'affichage "S"
            self._last_drawn[tid]["mirrored"] = not self._last_drawn[tid].get("mirrored", False)

        # BBox groupe puis redraw
        try:
            self._recompute_group_bbox(gid)
        except Exception:
            pass
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Inversion appliquée au groupe #{gid}.")

    def _move_group_world(self, gid, dx_w, dy_w):
        """Translate rigide de tout le groupe en coordonnées monde."""
        g = self.groups.get(gid)
        if not g:
            return
        d = np.array([dx_w, dy_w], dtype=float)
        for node in g["nodes"]:
            tid = node["tid"]
            if 0 <= tid < len(self._last_drawn):
                P = self._last_drawn[tid]["pts"]
                for k in ("O","B","L"):
                    P[k] = np.array([P[k][0] + d[0], P[k][1] + d[1]], dtype=float)
        self._recompute_group_bbox(gid)

    def _on_canvas_left_down(self, event):
        # Mode compas : arc d'angle (clic pour P1/P2)
        if self._clock_arc_active:
            self._clock_arc_handle_click(int(event.x), int(event.y))
            return "break"

        # Mode compas : clic pour valider une mesure d'azimut
        if self._clock_measure_active:
            self._clock_measure_confirm()
            return "break"

        # Mode compas : clic pour valider l'azimut de référence
        if self._clock_setref_active:
            self._clock_setref_confirm(int(event.x), int(event.y))
            return "break"

        # garantir un cache pick à jour avant tout hit-test
        self._ensure_pick_cache()
        # mémoriser l'ancre monde de la souris pour des déplacements "delta"
        try:
            self._mouse_world_prev = self._screen_to_world(event.x, event.y)
        except Exception:
            self._mouse_world_prev = None

        # Calibration fond (3 points) : intercepte le clic gauche
        if getattr(self, "_bg_calib_active", False):
            return self._bg_calibrate_handle_click(event)

        # Horloge : démarrer drag si clic dans le disque (marge 10px)
        if self._is_in_clock(event.x, event.y):
            # ne pas intercepter si un drag de triangle est en cours
            if not getattr(self, "_drag", None):
                self._clock_dragging = True
                self._clock_drag_dx = event.x - (self._clock_cx or event.x)
                self._clock_drag_dy = event.y - (self._clock_cy or event.y)
                # Mode "snap compas" : dès le mouse-down, viser le sommet le plus proche
                self._clock_update_snap_target(event.x, event.y)
                self.canvas.configure(cursor="fleur")
                return "break"  # on court-circuite la logique des triangles

        # Fond SVG : si mode resize et clic sur poignée -> on capture et on court-circuite le reste
        if self.bg_resize_mode.get() and self._bg:
            h = self._bg_hit_test_handle(event.x, event.y)
            if h:
                self._bg_start_resize(h, event.x, event.y)
                try: self.canvas.configure(cursor="sizing")
                except Exception: pass
                return "break"
            # sinon, en mode redimensionnement : clic maintenu = déplacement du fond
            self._bg_start_move(event.x, event.y)
            try: self.canvas.configure(cursor="fleur")
            except Exception: pass
            return "break"

        # Nouveau clic gauche : purge l'éventuelle aide précédente (évite les fantômes)
        # et masque le tooltip s'il est visible.
        self._hide_tooltip()
        self._reset_assist()
        # priorité au drag & drop depuis la liste
        if self._drag:
            return
        mode, idx, extra = self._hit_test(event.x, event.y)
        if mode == "center":
            P = self._last_drawn[idx]["pts"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            # Groupe obligatoire : on démarre toujours un move_group
            gid = self._get_group_of_triangle(idx)
            # En théorie, gid existe toujours (même un triangle seul est un groupe).
            # Si jamais non (état inattendu), on refuse de créer un mode 'move' triangle.
            if not gid:
                return
            Gc = self._group_centroid(gid)
            if Gc is None:
                return
            # snapshot complet du groupe pour rollback
            orig_group_pts = {}
            for node in self._group_nodes(gid):
                tid = node["tid"]
                if 0 <= tid < len(self._last_drawn):
                    Pt = self._last_drawn[tid]["pts"]
                    orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O","B","L")}
            self._sel = {
                "mode": "move_group",
                "gid": gid,
                "grab_offset": np.array([wx, wy]) - Gc,
                "orig_group_pts": orig_group_pts,
                "mouse_world_prev": np.array([wx, wy], dtype=float),
            }
            self.status.config(text=f"Déplacement du groupe #{gid} (clic sur tri {self._last_drawn[idx].get('id','?')})")
        elif mode == "vertex":
            # déplacement par sommet (translation comme 'center', mais calée sur le sommet choisi)
            vkey = extra or "O"
            P = self._last_drawn[idx]["pts"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            # NOTE: si on est en déconnexion, on désactivera toute aide au collage

            # ----- DEBUG étape 3 : clic sur sommet -----
            gid0 = self._get_group_of_triangle(idx)
            tri_id = self._last_drawn[idx].get("id","?")

            # si ce sommet est un LIEN -> on DECONNECTE et on démarre un move_group par sommet ---
            gid_link, pos, link_type = self._find_group_link_for_vertex(idx, vkey)
            if gid_link and self._ctrl_down:
                # Déterminer où couper :
                # - si click sur vkey_in (lien vers le précédent), on coupe AVANT 'pos' -> le morceau MOBILE = suffixe [pos..]
                # - si click sur vkey_out (lien vers le suivant), on coupe APRÈS  'pos' -> le morceau MOBILE = préfixe [..pos]
                nodes = self._group_nodes(gid_link)
                if link_type == "in":
                    cut_after = pos - 1
                    if cut_after < 0:  # sécurité (ne devrait pas arriver car vkey_in implies pos>0)
                        cut_after = 0
                else:  # "out"
                    cut_after = pos

                left_gid, right_gid = self._split_group_at(gid_link, cut_after)

                # Quel groupe contient le triangle cliqué après split ?
                mobile_gid = None
                # si pos <= cut_after : le tri cliqué est dans la PARTIE GAUCHE
                if pos <= cut_after and left_gid:
                    mobile_gid = left_gid
                # sinon dans la partie droite
                elif pos > cut_after and right_gid:
                    mobile_gid = right_gid

                # Invariant "100% groupes" : même singleton => on force le move_group (assist OFF)
                if mobile_gid is None:
                    mobile_gid = gid_link
                # --- Démarre un déplacement de GROUPE par le SOMMET cliqué ---
                orig_group_pts = {}
                for node in self._group_nodes(mobile_gid):
                    tid = node["tid"]
                    if 0 <= tid < len(self._last_drawn):
                        Pt = self._last_drawn[tid]["pts"]
                        orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O","B","L")}
                anchor_world = np.array(P[vkey], dtype=float)
                self._sel = {
                    "mode": "move_group",
                    "gid": mobile_gid,
                    "orig_group_pts": orig_group_pts,
                    "anchor": {"type": "vertex", "tid": idx, "vkey": vkey},
                    "grab_offset": np.array([wx, wy]) - anchor_world,
                    "suppress_assist": True,
                }
                self.status.config(text=f"Lien cassé. Déplacement du groupe #{mobile_gid} par sommet {vkey}.")
                self._clear_nearest_line()
                self._clear_edge_highlights()
                self._redraw_from(self._last_drawn)
                return

                # --- Démarre un déplacement de GROUPE par le SOMMET cliqué ---
                # Snapshot du groupe mobile pour rollback
                orig_group_pts = {}
                for node in self._group_nodes(mobile_gid):
                    tid = node["tid"]
                    if 0 <= tid < len(self._last_drawn):
                        Pt = self._last_drawn[tid]["pts"]
                        orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O","B","L")}

                # Ancre = sommet (du triangle cliqué) — on translate le groupe pour suivre ce sommet
                anchor_world = np.array(P[vkey], dtype=float)
                self._sel = {
                    "mode": "move_group",
                    "gid": mobile_gid,
                    "orig_group_pts": orig_group_pts,
                    "anchor": {"type": "vertex", "tid": idx, "vkey": vkey},
                    "grab_offset": np.array([wx, wy]) - anchor_world,
                    "suppress_assist": True,
                }
                self.status.config(text=f"Lien cassé. Déplacement du groupe #{mobile_gid} par sommet {vkey}.")
                # purge immédiate de toute aide existante
                self._clear_nearest_line()
                self._clear_edge_highlights()
                self._redraw_from(self._last_drawn)
                return

            # --- NOUVEAU (CTRL sans lien) : si CTRL est enfoncé mais que le sommet cliqué n'est PAS un lien,
            #                                on déplace le GROUPE entier ancré sur ce sommet, SANS assist.
            if self._ctrl_down:
                gid0 = self._get_group_of_triangle(idx)
                if gid0:
                    # snapshot du groupe (rollback sûr pendant le drag)
                    orig_group_pts = {}
                    for nd in self._group_nodes(gid0):
                        tid = nd["tid"]
                        if 0 <= tid < len(self._last_drawn):
                            Pt = self._last_drawn[tid]["pts"]
                            orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O", "B", "L")}

                    anchor_world = np.array(P[vkey], dtype=float)
                    self._sel = {
                        "mode": "move_group",
                        "gid": gid0,
                        "orig_group_pts": orig_group_pts,
                        "anchor": {"type": "vertex", "tid": idx, "vkey": vkey},
                        "grab_offset": np.array([wx, wy]) - anchor_world,
                        "suppress_assist": True,  # pas d'aides visuelles en CTRL
                    }
                    # nettoyer toute aide existante
                    self._reset_assist()
                    try:
                        self.status.config(text=f"CTRL sans lien : déplacement du groupe #{gid0} par sommet {vkey}.")
                    except Exception:
                        pass
                    self._redraw_from(self._last_drawn)
                    return

            # --- NOUVEAU : si le triangle appartient à un groupe et qu'on NE tient PAS CTRL,            #               on déplace le **groupe entier** ancré sur ce sommet.
            gid0 = self._get_group_of_triangle(idx)
            if gid0 and not self._ctrl_down:
                # snapshot du groupe (rollback sûr pendant le drag)
                orig_group_pts = {}
                for nd in self._group_nodes(gid0):
                    tid = nd["tid"]
                    if 0 <= tid < len(self._last_drawn):
                        Pt = self._last_drawn[tid]["pts"]
                        orig_group_pts[tid] = {k: np.array(Pt[k].copy()) for k in ("O","B","L")}

                anchor_world = np.array(P[vkey], dtype=float)
                self._sel = {
                    "mode": "move_group",
                    "gid": gid0,
                    "orig_group_pts": orig_group_pts,
                    "anchor": {"type": "vertex", "tid": idx, "vkey": vkey},
                    "grab_offset": np.array([wx, wy]) - anchor_world,
                    # on veut l'aide de collage active
                    "suppress_assist": False,
                }
                self.status.config(text=f"Déplacement du groupe #{gid0} par sommet {vkey}.")
                # Aide immédiate : viser un sommet d'un AUTRE triangle (exclure le groupe lui-même)
                v_world = np.array(P[vkey], dtype=float)
                tgt = self._find_nearest_vertex(v_world, exclude_idx=idx, exclude_gid=gid0)
                if tgt is not None:
                    j, tgt_key, _ = tgt
                    self._update_nearest_line(v_world, exclude_idx=idx, exclude_gid=gid0)
                    self._update_edge_highlights(idx, vkey, j, tgt_key)
                else:
                    self._clear_nearest_line()
                    self._clear_edge_highlights()
                return

            orig_pts = {k: np.array(P[k].copy()) for k in ("O","B","L")}
            self._sel = {
                "mode": "vertex",
                "idx": idx,
                "vkey": vkey,
                "grab_offset": np.array([wx, wy]) - np.array(P[vkey], dtype=float),
                "orig_pts": orig_pts,
            }
            # Affiche immédiatement la liaison + surlignage des arêtes candidates
            v_world = np.array(P[vkey], dtype=float)
            tgt = self._find_nearest_vertex(v_world, exclude_idx=idx)
            if tgt is not None:
                j, tgt_key, _ = tgt
                self._update_nearest_line(v_world, exclude_idx=idx)
                self._update_edge_highlights(idx, vkey, j, tgt_key)
            else:
                self._clear_nearest_line()
                self._clear_edge_highlights()
            self.status.config(text=f"Déplacement par sommet {vkey}.")
            return
        else:
            # clic ailleurs : pan au clic gauche
            self._on_pan_start(event)

    def _record_group_after_last_snap(self, idx_m: int, vkey_m: str, idx_t: int, vkey_t: str):
        """
        Version simplifiée : on ne gère que des **GROUPES** (les triangles sont
        toujours en groupe singleton). On fusionne **gid_m** et **gid_t** UNIQUEMENT
        en extrémité↔extrémité :
          - tail(m) → head(t) : tail_m.vkey_out = vkey_m ; head_t.vkey_in = vkey_t
          - head(m) → tail(t) : head_m.vkey_in = vkey_m ; tail_t.vkey_out = vkey_t
        Invariants :
          head.vkey_in is None ; tail.vkey_out is None ; liens internes complets.
        """
        tri_m = self._last_drawn[idx_m]
        tri_t = self._last_drawn[idx_t]
        gid_m = tri_m.get("group_id")
        gid_t = tri_t.get("group_id")

        # Rien à faire si mêmes groupes (collage interne non géré ici)
        if not gid_m or not gid_t or gid_m == gid_t:
            return

        nodes_m = self.groups[gid_m]["nodes"]
        nodes_t = self.groups[gid_t]["nodes"]
        head_m, tail_m = nodes_m[0], nodes_m[-1]
        head_t, tail_t = nodes_t[0], nodes_t[-1]

        # Cas A : queue(m) -> tête(t)
        if tri_m["group_pos"] == len(nodes_m) - 1 and tri_t["group_pos"] == 0:
            tail_m["vkey_out"] = vkey_m
            head_t["vkey_in"]  = vkey_t
            # concat t à la suite de m
            nodes_m.extend(nodes_t)
            for i, nd in enumerate(nodes_m):
                tid = nd["tid"]
                self._last_drawn[tid]["group_id"]  = gid_m
                self._last_drawn[tid]["group_pos"] = i
            # supprimer l'ancien groupe t
            del self.groups[gid_t]
            self._recompute_group_bbox(gid_m)
            return

        # Cas B : tête(m) -> queue(t)
        if tri_m["group_pos"] == 0 and tri_t["group_pos"] == len(nodes_t) - 1:
            head_m["vkey_in"]   = vkey_m
            tail_t["vkey_out"]  = vkey_t
            # concat m après t
            nodes_t.extend(nodes_m)
            for i, nd in enumerate(nodes_t):
                tid = nd["tid"]
                self._last_drawn[tid]["group_id"]  = gid_t
                self._last_drawn[tid]["group_pos"] = i
            del self.groups[gid_m]
            self._recompute_group_bbox(gid_t)
            return

        # Autres combinaisons (tête↔tête, queue↔queue, insertion interne) non gérées
        # dans cette version simplifiée. On ne modifie rien.
        return

    def _on_canvas_left_move(self, event):
        # Horloge : drag en cours -> on déplace le centre et on redessine l’overlay
        if getattr(self, "_clock_dragging", False):
            self._clock_cx = event.x - self._clock_drag_dx
            self._clock_cy = event.y - self._clock_drag_dy
            # Cible snap (sommet le plus proche du CENTRE du compas)
            self._clock_update_snap_target(self._clock_cx, self._clock_cy)
            self._redraw_overlay_only()
            return "break"

        # Mode déplacement fond d'écran (mode resize actif, clic maintenu hors poignée)
        if getattr(self, "_bg_moving", None):
            self._bg_update_move(event.x, event.y)
            self._redraw_from(self._last_drawn)
            return "break"
        
        # Mode resize fond d'écran
        if self._bg_resizing:
            self._bg_update_resize(event.x, event.y)
            self._redraw_from(self._last_drawn)
            return "break"

        if self._drag:
            return  # le drag liste gère déjà le mouvement
        if not self._sel:
            self._on_pan_move(event)
            return

        # --- Déplacement de GROUPE ---
        if self._sel["mode"] == "move_group":
            # Pendant une déconnexion, ne jamais montrer l'aide de collage
            if self._sel.get("suppress_assist"):
                self._clear_nearest_line()
                self._clear_edge_highlights()
            gid = self._sel["gid"]
            # déplacement CALÉ SUR DELTA MONDE: w -> w_prev
            wx, wy = self._screen_to_world(event.x, event.y)
            prev = self._sel.get("mouse_world_prev")
            if prev is None:
                self._sel["mouse_world_prev"] = np.array([wx, wy], dtype=float)
                return
            dx, dy = float(wx - prev[0]), float(wy - prev[1])
            if dx != 0.0 or dy != 0.0:
                self._move_group_world(gid, dx, dy)
                # avancer l'ancre
                self._sel["mouse_world_prev"] = np.array([wx, wy], dtype=float)
            self._redraw_from(self._last_drawn)
            # --- NOUVEAU : pendant un move_group ancré sur SOMMET, afficher l'aide de collage ---
            if not self._sel.get("suppress_assist"):
                anchor = self._sel.get("anchor")
                if anchor and anchor.get("type") == "vertex":
                    anchor_tid = anchor.get("tid")
                    anchor_vkey = anchor.get("vkey")
                    if 0 <= anchor_tid < len(self._last_drawn):
                        Panchor = self._last_drawn[anchor_tid]["pts"]
                        v_world = np.array(Panchor[anchor_vkey], dtype=float)
                        # chercher une cible HORS du groupe mobile
                        tgt = self._find_nearest_vertex(v_world, exclude_idx=anchor_tid, exclude_gid=gid)
                        if tgt is not None:
                            j, tgt_key, _ = tgt
                            self._update_nearest_line(v_world, exclude_idx=anchor_tid, exclude_gid=gid)
                            self._update_edge_highlights(anchor_tid, anchor_vkey, j, tgt_key)
                        else:
                            self._clear_nearest_line(); self._clear_edge_highlights()            
            return
        elif self._sel["mode"] == "move":
            idx = self._sel["idx"]
            P = self._last_drawn[idx]["pts"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            target_c = np.array([wx, wy]) - self._sel["grab_offset"]
            cur_c = self._tri_centroid(P)
            d = target_c - cur_c
            for k in ("O", "B", "L"):
                P[k] = np.array([P[k][0] + d[0], P[k][1] + d[1]])
            self._redraw_from(self._last_drawn)
        elif self._sel["mode"] == "vertex":
            # translation calée sur un sommet précis
            idx = self._sel["idx"]
            vkey = self._sel["vkey"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            target_v = np.array([wx, wy]) - self._sel["grab_offset"]
            P = self._last_drawn[idx]["pts"]
            cur_v = np.array(P[vkey], dtype=float)
            d = target_v - cur_v
            for k in ("O", "B", "L"):
                P[k] = np.array([P[k][0] + d[0], P[k][1] + d[1]])
            # Redessine d'abord les triangles (efface tout)
            self._redraw_from(self._last_drawn)
            # En déconnexion, NE RIEN AFFICHER (pas de ligne grise, pas d'arêtes orange)
            if not self._sel.get("suppress_assist"):
                v_world = np.array(P[vkey], dtype=float)
                tgt = self._find_nearest_vertex(v_world, exclude_idx=idx)
                if tgt is not None:
                    (j, tgt_key, w) = tgt
                    self._update_nearest_line(v_world, exclude_idx=idx)
                    self._update_edge_highlights(idx, vkey, j, tgt_key)
            else:
                self._clear_nearest_line()
                self._clear_edge_highlights()
        elif self._sel["mode"] == "rotate":
            # désormais la rotation est gérée dans <Motion> (pas besoin de maintenir le clic)
            pass

    def _on_canvas_left_up(self, event):
        # Horloge : fin de drag
        if getattr(self, "_clock_dragging", False):
            # On capture la cible de snap *avant* de la nettoyer pour pouvoir déclencher
            # une éventuelle mesure d'arc automatique.
            snap_tgt = getattr(self, "_clock_snap_target", None)

            # Si CTRL au relâché : on sort du mode sans "snap" (le compas reste où il est)
            if getattr(self, "_ctrl_down", False):
                try:
                    wx, wy = self._screen_to_world(self._clock_cx, self._clock_cy)
                    self._clock_anchor_world = np.array([wx, wy], dtype=float)
                except Exception:
                    self._clock_anchor_world = None
            else:
                # cible snap calculée pendant le drag (capturée au début via snap_tgt)
                tgt = snap_tgt
                if isinstance(tgt, dict) and tgt.get("world") is not None:
                    self._clock_anchor_world = np.array(tgt["world"], dtype=float)
                    sx, sy = self._world_to_screen(tgt["world"])
                    self._clock_cx, self._clock_cy = float(sx), float(sy)
                else:
                    # pas de target : ancrer la position courante en monde
                    try:
                        wx, wy = self._screen_to_world(self._clock_cx, self._clock_cy)
                        self._clock_anchor_world = np.array([wx, wy], dtype=float)
                    except Exception:
                        self._clock_anchor_world = None
            self._clock_dragging = False
            self.canvas.configure(cursor="")
            self._clock_clear_snap_target()
            # Spéc : si on déplace le compas et qu'il s'accroche à un noeud,
            # on tente une mesure d'arc automatique (EXT). Sinon on reset.
            measured = False
            if (not getattr(self, "_ctrl_down", False)) and isinstance(snap_tgt, dict):
                measured = bool(self._clock_arc_auto_from_snap_target(snap_tgt))
            if not measured:
                self._clock_arc_clear_last()
            self._redraw_overlay_only()
            return "break"

        if self._bg_resizing:
            self._bg_resizing = None
            try: self.canvas.configure(cursor="")
            except Exception: pass
            self._persistBackgroundConfig()
            self._bg_update_scale_status()
            self._redraw_from(self._last_drawn)
            return "break"

        if getattr(self, "_bg_moving", None):
            self._bg_moving = None
            try: self.canvas.configure(cursor="")
            except Exception: pass
            self._persistBackgroundConfig()
            self._redraw_from(self._last_drawn)
            return "break"

        # Pan au clic gauche : fin du pan même si aucun triangle n'est sélectionné
        if getattr(self, "_pan_anchor", None) is not None and not getattr(self, "_sel", None) and not getattr(self, "_drag", None):
            self._on_pan_end(event)
            return

        """Fin du drag au clic gauche : dépôt de triangle (drag liste) OU fin d'une édition."""

        # 0) Dépôt d'un triangle glissé depuis la liste
        if self._drag:
            # Si le fantôme existe, on le supprime
            if self._drag_preview_id is not None:
                try:
                    self.canvas.delete(self._drag_preview_id)
                except Exception:
                    pass
                self._drag_preview_id = None

            # Sécurité : si world_pts n'a pas été posé (pas de <Motion>), le calculer ici
            if "world_pts" not in self._drag:
                tri = self._drag["triangle"]
                P = tri["pts"]
                wx = (event.x - self.offset[0]) / self.zoom
                wy = (self.offset[1] - event.y) / self.zoom
                dx = wx - float(P["O"][0])
                dy = wy - float(P["O"][1])
                O = np.array([P["O"][0] + dx, P["O"][1] + dy])
                B = np.array([P["B"][0] + dx, P["B"][1] + dy])
                L = np.array([P["L"][0] + dx, P["L"][1] + dy])
                self._drag["world_pts"] = {"O": O, "B": B, "L": L}

            # Dépose réellement le triangle dans le document
            self._place_dragged_triangle()
            # Reset état de drag
            self._drag = None
            # Remettre le curseur normal en fin de drag
            self.canvas.configure(cursor="")
            return

        # 1) Le reste est ton comportement existant (fin de drag/rotation/snap)
        if not self._sel:
            return

        mode = self._sel.get("mode")
        if mode == "vertex":
            idx = self._sel["idx"]
            vkey = self._sel["vkey"]

            # On a nécessairement choisi la meilleure arête dans _update_edge_highlights
            choice = getattr(self, "_edge_choice", None)

            # Nettoie correctement l'aide visuelle (liste d'ids + choix)
            self._clear_edge_highlights()

            if not choice or choice[0] != idx or choice[1] != vkey:
                # rien à coller -> simple fin de drag
                self._sel = None
                self._redraw_from(self._last_drawn)
                return

            # si CTRL est enfoncé AU RELÂCHEMENT, on ne colle pas.
            # (On garde l'aide visuelle pendant le drag si CTRL a été pressé après le clic.)

            # DEBUG : état du snap au relâchement
            ctrl_state = bool(getattr(event, "state", 0) & 0x0004)
            self._dbgSnap(
                f"[snap] release(vertex) ctrl_down={getattr(self,'_ctrl_down',False)} ctrl_state={ctrl_state} choice={'OK' if choice else 'None'}"
            )

            if getattr(self, "_ctrl_down", False):
                self._sel = None
                self._reset_assist()
                self._redraw_from(self._last_drawn)
                self.status.config(text="Dépôt sans collage (CTRL).")
                return


            # Déroulé du collage (rotation+translation) sur le TRIANGLE seul
            (_, _, idx_t, vkey_t, (m_a, m_b, t_a, t_b)) = choice
            tri_m = self._last_drawn[idx]
            Pm = tri_m["pts"]

            A = np.array(m_a, dtype=float)  # mobile: sommet saisi
            B = np.array(m_b, dtype=float)  # mobile: voisin qui définit l'arête
            U = np.array(t_a, dtype=float)  # cible: point où coller
            V = np.array(t_b, dtype=float)  # cible: deuxième point de l'arête

            # azimuts via helper unifié
            ang_m = self._ang_of_vec(B[0] - A[0], B[1] - A[1])
            ang_t = self._ang_of_vec(V[0] - U[0], V[1] - U[1])
            dtheta = ang_t - ang_m

            R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                        [np.sin(dtheta),  np.cos(dtheta)]], dtype=float)

            new_pts = {}
            for k in ("O", "B", "L"):
                p = np.array(Pm[k], dtype=float)
                p_rot = A + (R @ (p - A))
                new_pts[k] = p_rot

            # 2) translation pour amener A -> U
            delta = U - new_pts[vkey]
            for k in ("O", "B", "L"):
                new_pts[k] = new_pts[k] + delta

            # Applique la géométrie
            for k in ("O", "B", "L"):
                Pm[k][0] = float(new_pts[k][0])
                Pm[k][1] = float(new_pts[k][1])

            # Enregistre/étend les groupes (métadonnées & invariants)
            self._record_group_after_last_snap(idx, vkey, idx_t, vkey_t)
            # fin d'opération : on purge les aides et la sélection
            self._sel = None
            self._reset_assist()
            self._redraw_from(self._last_drawn)
            self.status.config(text="Triangles collés (sommets et arêtes alignés).")
            return

        elif mode == "move_group":
            # Collage du GROUPE quand on l'a déplacé PAR SOMMET (ancre=vertex)
            # et que l'aide de collage était active (pas en déconnexion).
            anchor = self._sel.get("anchor")
            suppress = self._sel.get("suppress_assist")

            # On nettoie l'aide visuelle dans tous les cas
            self._clear_edge_highlights()

            # Si l'aide était active et qu'on a bien un choix d'arête,
            # on applique la même géométrie que pour un triangle seul,
            # mais à TOUS les triangles du groupe.
            choice = getattr(self, "_edge_choice", None)
            self._dbgSnap(
                f"[snap] release(move_group) suppress_assist={self._sel.get('suppress_assist')} choice={'OK' if choice else 'None'}"
            )
            if (not suppress
                and (not getattr(self, "_ctrl_down", False))
                and anchor
                and anchor.get("type") == "vertex"
                and choice
                and choice[0] == anchor.get("tid")
                and choice[1] == anchor.get("vkey")):

                # Déballage du choix d'arête: (mob_idx,vkey_m,tgt_idx,vkey_t,(m_a,m_b,t_a,t_b))
                (_, _, idx_t, vkey_t, (m_a, m_b, t_a, t_b)) = choice

                A = np.array(m_a, dtype=float)  # sommet mobile saisi (dans le groupe)
                B = np.array(m_b, dtype=float)  # voisin côté mobile (définit l'arête)
                U = np.array(t_a, dtype=float)  # cible: point d'accroche
                V = np.array(t_b, dtype=float)  # cible: second point arête

                # Angle mobile/cible et rotation à appliquer (helpers unifiés)
                ang_m = self._ang_of_vec(B[0] - A[0], B[1] - A[1])
                ang_t = self._ang_of_vec(V[0] - U[0], V[1] - U[1])
                dtheta = ang_t - ang_m
                R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                              [np.sin(dtheta),  np.cos(dtheta)]], dtype=float)

                # Translation finale pour amener A -> U
                # (après rotation autour de A, A reste à A)
                delta = U - A

                # Appliquer (rotation autour de A) + translation à tous
                gid = self._sel.get("gid")
                g = self.groups.get(gid)
                if g:
                    for node in g["nodes"]:
                        tid = node.get("tid")
                        if 0 <= tid < len(self._last_drawn):
                            P = self._last_drawn[tid]["pts"]
                            for k in ("O", "B", "L"):
                                p = np.array(P[k], dtype=float)
                                p_rot = A + (R @ (p - A))
                                p_fin = p_rot + delta
                                P[k][0] = float(p_fin[0])
                                P[k][1] = float(p_fin[1])

                # ====== FUSION DE GROUPE APRÈS COLLAGE (groupe ↔ groupe uniquement) ======
                tgt_gid = self._last_drawn[idx_t].get("group_id", None)
                mob_gid = gid
                if mob_gid is not None and tgt_gid is not None:
                    if tgt_gid != mob_gid:
                        # Fusionner tgt_gid → mob_gid AVEC LIEN EXPLICITE
                        g_src = self.groups.get(mob_gid)
                        g_tgt = self.groups.get(tgt_gid)
                        anchor = self._sel.get("anchor")  # {"type":"vertex","tid":...,"vkey":...}
                        choice = getattr(self, "_edge_choice", None)  # (... idx_t, vkey_t, ...)
                        if g_src and g_tgt and anchor and choice:
                            # 2.1 Localiser ancre (dans src) et cible (dans tgt)
                            anchor_tid = anchor.get("tid"); anchor_vkey = anchor.get("vkey")
                            (_, _, idx_t, vkey_t, (m_a, m_b, t_a, t_b)) = choice
                            nodes_src = g_src["nodes"]; nodes_tgt = g_tgt["nodes"]
                            pos_anchor = next((i for i, nd in enumerate(nodes_src) if nd.get("tid")==anchor_tid), None)
                            pos_target = next((i for i, nd in enumerate(nodes_tgt) if nd.get("tid")==idx_t), None)
                            if pos_anchor is None or pos_target is None:
                                # fallback: conserver l’ancien comportement (append brut)
                                for nd in nodes_tgt:
                                    tid2 = nd.get("tid"); 
                                    if tid2 is None: continue
                                    self._last_drawn[tid2]["group_id"] = mob_gid
                                    nodes_src.append({"tid": tid2,
                                                      "vkey_in": nd.get("vkey_in"),
                                                      "vkey_out": nd.get("vkey_out")})
                            else:
                                # --- util: retrouver la vkey ("O","B","L") d'un point exact d'un triangle ---
                                def _vkey_from_point(P, pt, eps=EPS_WORLD):
                                    try:
                                        x, y = float(pt[0]), float(pt[1])
                                    except Exception:
                                        return None
                                    for kk in ("O","B","L"):
                                        try:
                                            if abs(float(P[kk][0]) - x) <= eps and abs(float(P[kk][1]) - y) <= eps:
                                                return kk
                                        except Exception:
                                            pass
                                    return None

                                # --- util: arête depuis 2 sommets (ex: ("L","B") -> "BL") ---
                                def _edge_from_vkeys(a, b):
                                    if not a or not b: 
                                        return None
                                    if a == b:
                                        return None
                                    s = {a, b}
                                    if s == {"O","B"}: return "OB"
                                    if s == {"B","L"}: return "BL"
                                    if s == {"L","O"}: return "LO"
                                    return None

                                # --- util: sommet opposé à l'arête (a,b) ---
                                def _opp_vertex(a, b):
                                    if a is None or b is None:
                                        return None
                                    for kk in ("O","B","L"):
                                        if kk != a and kk != b:
                                            return kk
                                    return None

                                # 2.2 Construire une vue pivotée de tgt pour que idx_t soit AU BORD collé
                                def rotate(L, k): 
                                    k = k % len(L); 
                                    return L[k:]+L[:k]
                                # On privilégie "coller APRES l'ancre": ancre --(out)-> [tgt...]
                                insert_after = True
                                tgt_rot = rotate(list(nodes_tgt), pos_target)  # idx_t devient tgt_rot[0]
                                # Poser les marqueurs de lien complémentaires :
                                nd_anchor = nodes_src[pos_anchor]
                                if insert_after:
                                    # ancre sort → cible entre : on stocke le SOMMET OPPOSÉ à l'arête partagée
                                    Pm = self._last_drawn[anchor_tid]["pts"]
                                    Pt = self._last_drawn[idx_t]["pts"]
                                    mob_neigh = _vkey_from_point(Pm, m_b)
                                    tgt_neigh = _vkey_from_point(Pt, t_b)
                                    nd_anchor["vkey_out"] = _opp_vertex(anchor_vkey, mob_neigh)
                                    nd_anchor["edge_out"] = _edge_from_vkeys(anchor_vkey, mob_neigh)
                                    tgt_rot[0] = {"tid": tgt_rot[0]["tid"],
                                                  "vkey_in": _opp_vertex(vkey_t, tgt_neigh),
                                                  "vkey_out": tgt_rot[0].get("vkey_out"),
                                                  "edge_in": _edge_from_vkeys(vkey_t, tgt_neigh),
                                                  "edge_out": tgt_rot[0].get("edge_out")}
                                    # insérer juste après l’ancre
                                    nodes_src[pos_anchor+1:pos_anchor+1] = tgt_rot
                                else:
                                    # ancre entre ← cible sort : idem, stocker sommet opposé à l'arête partagée
                                    Pm = self._last_drawn[anchor_tid]["pts"]
                                    Pt = self._last_drawn[idx_t]["pts"]
                                    mob_neigh = _vkey_from_point(Pm, m_b)
                                    tgt_neigh = _vkey_from_point(Pt, t_b)
                                    nd_anchor["vkey_in"] = _opp_vertex(anchor_vkey, mob_neigh)
                                    nd_anchor["edge_in"] = _edge_from_vkeys(anchor_vkey, mob_neigh)
                                    tgt_rot[-1] = {"tid": tgt_rot[-1]["tid"],
                                                   "vkey_in": tgt_rot[-1].get("vkey_in"),
                                                   "vkey_out": _opp_vertex(vkey_t, tgt_neigh),
                                                   "edge_in": tgt_rot[-1].get("edge_in"),
                                                   "edge_out": _edge_from_vkeys(vkey_t, tgt_neigh)}
                                    # insérer juste avant l’ancre
                                    nodes_src[pos_anchor:pos_anchor] = tgt_rot
                                # MAJ group_id sur tout le bloc inséré
                                for nd in tgt_rot:
                                    tid2 = nd["tid"]
                                    if 0 <= tid2 < len(self._last_drawn):
                                        self._last_drawn[tid2]["group_id"] = mob_gid
                            # 2.3 Supprimer l'ancien groupe cible et reindexer pos/bbox
                            try: del self.groups[tgt_gid]
                            except Exception: pass
                            for i, nd in enumerate(nodes_src):
                                tid2 = nd["tid"]
                                if 0 <= tid2 < len(self._last_drawn):
                                    self._last_drawn[tid2]["group_id"]  = mob_gid
                                    self._last_drawn[tid2]["group_pos"] = i
                            self._recompute_group_bbox(mob_gid)
                            self.status.config(text=f"Groupes fusionnés avec lien: #{tgt_gid} → #{mob_gid}.")
                    else:
                        # cible déjà dans le même groupe → rien à faire côté structure
                        self.status.config(text=f"Groupe collé (même groupe #{mob_gid}).")
                # ====== /FUSION ======



            # Fin (pas de snap : simple dépôt à la dernière position)
            self._sel = None
            self._reset_assist()
            self._redraw_from(self._last_drawn)
            return

        elif mode == "rotate":
            self._sel = None
            self._reset_assist()
            self.status.config(text="Rotation validée.")
            self._redraw_from(self._last_drawn)
            return

        # Autres modes : on nettoie juste
        self._sel = None
        self._reset_assist()
        self._redraw_from(self._last_drawn)


    def _on_mousewheel(self, event):
        # Normalize wheel delta across platforms
        if hasattr(event, "delta") and event.delta != 0:
            dz = 1.1 if event.delta > 0 else 1/1.1
        else:
            dz = 1.1 if getattr(event, "num", 0) == 4 else 1/1.1

        # World coordinate under cursor BEFORE zoom
        wx = (event.x - self.offset[0]) / self.zoom
        wy = (self.offset[1] - event.y) / self.zoom

        # Apply zoom (clamped)
        self.zoom = max(0.05, min(100.0, self.zoom * dz))

        # Adjust offset so (wx,wy) remains under cursor AFTER zoom
        self.offset = np.array([event.x - wx * self.zoom,
                                event.y + wy * self.zoom], dtype=float)

        self._redraw_from(self._last_drawn)
        # zoom modifie les coords écran -> invalider le pick-cache
        self._invalidate_pick_cache()

    def _on_pan_start(self, event):
        self._pan_anchor = np.array([event.x, event.y], dtype=float)
        self._offset_anchor = self.offset.copy()

    def _on_pan_move(self, event):
        if getattr(self, "_pan_anchor", None) is None:
            return
        d = np.array([event.x, event.y], dtype=float) - self._pan_anchor
        self.offset = self._offset_anchor + d
        self._redraw_from(self._last_drawn)

    def _on_pan_end(self, event):
        self._pan_anchor = None
        # après pan -> invalider cache pick (coords écran changent)
        self._invalidate_pick_cache()

    # ---------- Impression PDF (A4) ----------
    def _export_triangles_pdf(
        self, path, triangles, scale_mm=1.5,
        page_margin_mm=12, cell_pad_mm=6,
        stroke_pt=1.2, font_size_pt=9, label_inset=0.35,
        pack_mode="shelf",           # "shelf" = rangées à hauteur variable
        rotate_to_fit=True           # rotation 90° si ça réduit la largeur en rangée
    ):
        """
        PDF A4 portrait, triangles à la même échelle, placement 'shelf packing'.
        """
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm

        if not triangles:
            raise ValueError("Aucun triangle à imprimer.")

        S = float(scale_mm) * float(mm)      # points par unité
        page_w, page_h = A4
        margin = float(page_margin_mm) * float(mm)
        pad    = float(cell_pad_mm) * float(mm)

        # Prépare bboxes à l’échelle (orientation normale OU rotée)
        items = []  # pour chaque tri : dict avec bbox_n (w,h) et bbox_r (w,h)
        for t in triangles:
            P = t["pts"]
            xs = [float(P["O"][0]), float(P["B"][0]), float(P["L"][0])]
            ys = [float(P["O"][1]), float(P["B"][1]), float(P["L"][1])]
            mnx, mny, mxx, mxy = min(xs), min(ys), max(xs), max(ys)

            tri_w = (mxx - mnx) * S
            tri_h = (mxy - mny) * S
            wN, hN = tri_w + 2*pad, tri_h + 2*pad            # normal
            wR, hR = tri_h + 2*pad, tri_w + 2*pad            # roté 90°

            items.append({
                "data": t,
                "bbox_world": (mnx, mny, mxx, mxy),
                "wN": wN, "hN": hN,
                "wR": wR, "hR": hR,
            })

        # Espace utile page
        content_w = page_w - 2*margin
        content_h = page_h - 2*margin

        # Placement 'shelf' (rangées)
        def new_page():
            return {"rows": [], "height_used": 0.0}

        def close_row(page, row):
            page["rows"].append(row)
            page["height_used"] += row["H"]

        page_list  = []
        page = new_page()
        row  = {"X": 0.0, "Y": 0.0, "H": 0.0, "items": []}

        for it in items:
            cand = []
            if rotate_to_fit:
                cand.append(("R", it["wR"], it["hR"]))
            cand.append(("N", it["wN"], it["hN"]))
            cand.sort(key=lambda x: x[1])  # largeur croissante

            placed = False
            for rot, w, h in cand:
                if row["X"] + w <= content_w or row["X"] == 0.0:
                    Hnew = max(row["H"], h) if row["H"] > 0 else h
                    if (page["height_used"] + Hnew) > content_h and row["X"] == 0.0:
                        if row["H"] > 0:
                            close_row(page, row)
                        page_list.append(page)
                        page = new_page()
                        row  = {"X": 0.0, "Y": 0.0, "H": 0.0, "items": []}

                    if row["X"] + w <= content_w or row["X"] == 0.0:
                        row["items"].append({"x": row["X"], "y": page["height_used"], "w": w, "h": h, "rot": (rot=="R"), "it": it})
                        row["X"] += w
                        row["H"] = max(row["H"], h)
                        placed = True
                        break

            if not placed:
                close_row(page, row)
                row  = {"X": 0.0, "Y": page["height_used"], "H": 0.0, "items": []}
                rot, w, h = cand[0]
                if page["height_used"] + h > content_h:
                    page_list.append(page)
                    page = new_page()
                    row  = {"X": 0.0, "Y": 0.0, "H": 0.0, "items": []}
                row["items"].append({"x": row["X"], "y": page["height_used"], "w": w, "h": h, "rot": (rot=="R"), "it": it})
                row["X"] += w
                row["H"] = max(row["H"], h)

        if row["items"]:
            close_row(page, row)
        page_list.append(page)

        # Dessin
        c = canvas.Canvas(path, pagesize=A4)
        c.setStrokeColorRGB(0, 0, 0)
        c.setFillColorRGB(0, 0, 0)

        def draw_tri(at_x, at_y, box_w, box_h, item, rot90):
            t   = item["data"]
            P   = t["pts"]
            Oname, Bname, Lname = t["labels"]
            mnx, mny, mxx, mxy = item["bbox_world"]

            tri_w = (mxx - mnx) * S
            tri_h = (mxy - mny) * S
            draw_w = tri_h if rot90 else tri_w
            draw_h = tri_w if rot90 else tri_h

            cx = margin + at_x + (box_w - draw_w) / 2.0
            cy = margin + at_y + (box_h - draw_h) / 2.0

            def to_page(p):
                x = (float(p[0]) - mnx) * S
                y = (float(p[1]) - mny) * S
                return (x, y)

            O = to_page(P["O"]); B = to_page(P["B"]); L = to_page(P["L"])

            c.saveState()
            c.translate(cx, cy)
            if rot90:
                c.translate(0, draw_h)
                c.rotate(-90)

            # traits
            c.setLineWidth(stroke_pt)
            c.line(O[0], O[1], B[0], B[1])
            c.line(B[0], B[1], L[0], L[1])
            c.line(L[0], L[1], O[0], O[1])

            # labels
            c.setFont("Helvetica", float(font_size_pt))
            gx = (O[0] + B[0] + L[0]) / 3.0
            gy = (O[1] + B[1] + L[1]) / 3.0
            for (px, py), txt in zip((O, B, L), (Oname, Bname, Lname)):
                lx = (1.0 - label_inset) * px + label_inset * gx
                ly = (1.0 - label_inset) * py + label_inset * gy
                c.drawCentredString(lx, ly, txt)
            c.restoreState()

        for pg in page_list:
            for row in pg["rows"]:
                for cell in row["items"]:
                    draw_tri(cell["x"], cell["y"], cell["w"], cell["h"], cell["it"], cell["rot"])
            c.showPage()
        c.save()

    def print_triangles_dialog(self):
        from tkinter import simpledialog, filedialog
        if getattr(self, "df", None) is None or self.df.empty:
            messagebox.showwarning("Imprimer", "Charge d'abord le fichier Excel.")
            return

        # Paramètres
        start = max(1, int(self.start_index.get()))
        nmax = int(self.df.shape[0] - (start-1))
        n = simpledialog.askinteger("Imprimer", f"Nombre de triangles (max {nmax}) :", initialvalue=min(8, nmax), minvalue=1, maxvalue=nmax)
        if not n:
            return

        scale = simpledialog.askfloat("Imprimer", "Échelle (mm par unité de longueur) :", initialvalue=1.5, minvalue=0.1, maxvalue=100.0)
        if not scale:
            return

        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile="triangles.pdf")
        if not path:
            return

        tris = self._triangles_local(start-1, n)
        self._export_triangles_pdf(path, tris, scale_mm=scale)
        self.status.config(text=f"PDF généré : {path}")

# ---------- Entrée ----------
if __name__ == "__main__":
    app = TriangleViewerManual()
    app.mainloop()
