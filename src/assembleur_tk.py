
import os
import datetime as _dt
import math
from math import atan2, pi
import numpy as np
import re
import copy
import pandas as pd
from typing import Optional, List, Dict, Tuple

from tkinter import filedialog, messagebox, simpledialog
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

from PIL import Image

from shapely.geometry import Polygon as _ShPoly
from shapely.ops import unary_union as _sh_union

# === Modules externalisés (découpage maintenable) ===
from src.assembleur_core import (
    _group_shape_from_nodes,
    _build_local_triangle,
    ScenarioAssemblage,
    TopologyWorld, TopologyElement, TopologyNodeType,
)

from src.assembleur_sim import (
    MoteurSimulationAssemblage,
    ALGOS,
)

from src.assembleur_decryptor import (
    DecryptorBase,
    ClockDicoDecryptor,
    DECRYPTORS,
)

import src.assembleur_io as _assembleur_io

# --- Tk split: mixins (découpage assembleur_tk.py) ---
from src.assembleur_tk_mixin_dictionary import TriangleViewerDictionaryMixin
from src.assembleur_tk_mixin_frontier import TriangleViewerFrontierGraphMixin
from src.assembleur_tk_mixin_bg import TriangleViewerBackgroundMapMixin
from src.assembleur_tk_mixin_clockarc import TriangleViewerClockArcMixin
from src.assembleur_edgechoice import buildEdgeChoiceEptsFromBest


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
        last_run_available_by_order: Optional[Dict[str, bool]] = None,
    ):
        super().__init__(parent)
        self._last_run_available_by_order = dict(last_run_available_by_order or {})
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

        def _updateFirstEdgeComboValues(*_args):
            # Base options
            vals = ["OL = 0°", "BL = 0°"]
            ordv = str(self.order_var.get() or "").strip().lower()
            if self._last_run_available_by_order.get(ordv, False):
                vals.append("Même dernier run")
            cur = str(self.first_edge_combo.get() or "").strip()
            self.first_edge_combo.configure(values=vals)
            # Si la sélection courante n'existe plus (ex: switch d'ordre), fallback sur OL
            if cur and cur not in vals:
                self.first_edge_combo.current(0)

        self._updateFirstEdgeComboValues = _updateFirstEdgeComboValues
        self.order_var.trace_add("write", self._updateFirstEdgeComboValues)
        self._updateFirstEdgeComboValues()

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2, sticky="e", pady=(10, 0))
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
            print(f"[SIM] n impair -> utilisation de n={n2}")
            n = n2
            self.var_nb_triangles.set(n)

        order = str(self.order_var.get() or "forward")
        first_raw = str(self.first_edge_var.get() or "OL")
        if "Même" in first_raw or "même" in first_raw:
            first_edge = "__LASTRUN__"
        else:
            first_edge = "BL" if "BL" in first_raw else "OL"

        self.result = (algo_id, int(n), order, first_edge)

        self.destroy()


# ---------- Application (MANUEL — sans algorithmes) ----------
class TriangleViewerManual(
    TriangleViewerDictionaryMixin,
    TriangleViewerFrontierGraphMixin,
    TriangleViewerBackgroundMapMixin,
    TriangleViewerClockArcMixin,
    tk.Tk,
):
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

        # -----------------------------
        # 1 TopologyWorld par scénario
        # - IDs scénarios lisibles : SM1/SM2... (manuels), SA1/SA2... (autos)
        # - Tk reste centré sur ses objets graphiques ; on annote simplement avec les IDs topo.
        # -----------------------------
        self._topo_next_manual_id = 1
        self._topo_next_auto_id = 1

        # état de vue
        # Référence STABLE pour l'anti-chevauchement en simulation.
        # IMPORTANT : ne doit pas bouger quand on fait un "fit à l'écran" (qui modifie self.zoom
        self.simulationOverlapZoomRef = 1.0
        self.zoom = 1.0
        self.offset = np.array([400.0, 350.0], dtype=float)
        self._drag = None              # état de drag & drop depuis la liste
        self._drag_preview_id = None   # id du polygone "fantôme" sur le canvas
        self._sel = None               # sélection sur canvas: {'mode': 'move'|'vertex', 'idx': int}
        self._hit_px = 12              # tolérance de hit (pixels) pour les sommets
        self._center_hit_px = 12       # même défaut que hit_px historique
        self._marker_px = 6            # rayon des marqueurs (cercles) dessinés aux sommets
        self._pan_anchor = None
        self._offset_anchor = None
        self._last_canvas_size = (0, 0)
        self._resize_redraw_after_id = None

        self._ctx_target_idx = None    # index du triangle visé par clic droit (menu contextuel)
        self._ctx_last_rclick = None   # dernière position écran du clic droit (pour pivoter)
        self._ctx_nearest_vertex_key = None  # 'O'|'B'|'L' sommet le plus proche du clic droit        
        self.ctxGroupId = None         # contexte chemin: groupId Core canonique (clic droit)
        self.ctxStartNodeId = None     # contexte chemin: startNodeId DSU canonique (clic droit)
        self._placed_ids = set()       # ids déjà posés dans le scénario actif
        self._nearest_line_id = None   # trait d'aide "sommet le plus proche"
        self._edge_highlight_ids = []  # surlignage des 2 arêtes (mobile/cible)
        self._edge_choice = None       # (i_mob, key_mob, edge_m, i_tgt, key_tgt, edge_t)
        self._edge_highlights = None   # données brutes des aides (candidates + best)
        self._tooltip = None           # tk.Toplevel
        self._tooltip_label = None     # tk.Label

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

        # Le dico à créer
        self.dicoSheet = None
        # Hauteur fixe du panneau "Dico" (en pixels) sous le canvas
        self.dico_panel_height = 290
        # État d'affichage du panneau dictionnaire (toggle via menu Visualisation)
        self.show_dico_panel = tk.BooleanVar(value=True)
        # État d'affichage du compas horaire (overlay horloge)
        self.show_clock_overlay = tk.BooleanVar(value=True)
        self._clock_anchor_world = None
        # Mode "contours uniquement" : n'afficher que le contour de chaque groupe (pas les arêtes internes)
        self.show_only_group_contours = tk.BooleanVar(value=False)

        # Gestion des layers (visibilité)
        self.show_map_layer = tk.BooleanVar(value=True)
        self.show_triangles_layer = tk.BooleanVar(value=True)
        # Opacité du layer "carte" (0..100). 100 = opaque, 0 = invisible.
        self.map_opacity = tk.IntVar(value=70)
        self._map_opacity_redraw_job = None
        # Gestion du contour
        self.show_only_group_contours = tk.BooleanVar(value=False)
        self.only_group_contours = None
        self._only_group_contours = False

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
 
        # Valeur mémorisée (persistable) de l'échelle carte, pour l'affichage quand la calibration n'est pas disponible.
        self._bg_scale_factor_override: float | None = None

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
 
        # Carte partagée pour les scénarios automatiques (snapshot au lancement de la simu)
        self.auto_map_state: dict | None = None
        self.auto_view_state: dict | None = None

        # état géométrique global pour scénarios automatiques (transform commun)
        # P_world = (ox,oy) + R(thetaDeg) * P_local
        self.auto_geom_state: dict | None = None  # {'ox':float,'oy':float,'thetaDeg':float}

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
        # Exports (artefacts diffables / validation)
        self.exports_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "exports"))
        self.topo_xml_dir = os.path.join(self.exports_dir, "TopoXML")
        # Répertoire des icônes
        self.images_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "images"))
        os.makedirs(self.scenario_dir, exist_ok=True)
        os.makedirs(self.maps_dir, exist_ok=True)
        os.makedirs(self.topo_xml_dir, exist_ok=True)
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
        manual.view_state = self._capture_view_state()
        manual.map_state = self._capture_map_state()
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


    # ======================================================================
    #  Topologie (bridge minimal Tk -> Core)
    # ======================================================================
    def getTidForTopoElementId(self, topoElementId: str) -> int | None:
        e = str(topoElementId)
        for tid, tri in enumerate(self._last_drawn or []):
            if str(tri.get("topoElementId", "")) == e:
                return int(tid)
        return None

    def _sync_group_elements_pose_to_core(self, ui_gid: int, scen: ScenarioAssemblage = None) -> None:
        """Synchronise les poses (R,T,mirrored) de TOUS les éléments du groupe UI vers le Core.

        Important:
        - Tk manipule la géométrie au niveau groupe (déplacement/rotation).
        - Le Core ne porte AUCUNE pose de groupe : les groupes sont topologiques.
        - La vérité géométrique persistable est donc : pose par élément.
        """
        if scen is None:
            scen = self._get_active_scenario()
        world = scen.topoWorld 

        grp = self.groups.get(ui_gid)
        if not grp:
            return

        nodes = grp.get("nodes", [])
        if not nodes:
            return

        # Modèle "pose" :
        # - mirrored est réservé AU FLIP utilisateur (fonction _ctx_flip_selected).
        # - La pose initiale d'un triangle ne doit PAS activer mirrored.
        # - L'orientation CW/CCW provenant de l'Excel doit être résolue en amont
        #   (normalisation des coordonnées locales), pas ici.
        M = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)

        for node in nodes:
            tid = int(node.get("tid", -1))
            if not (0 <= tid < len(self._last_drawn)):
                continue

            tri = self._last_drawn[tid]
            element_id = tri.get("topoElementId", None)
            if not element_id:
                continue

            Pw = tri.get("pts") or tri.get("world_pts")
            if Pw is None:
                continue

            # World: O,B,L (nécessaire pour capturer le handedness après un flip legacy)
            Ow = np.array(Pw["O"], dtype=float)
            Bw = np.array(Pw["B"], dtype=float)
            Lw = np.array(Pw["L"], dtype=float)

            # Local: O,B depuis le Core (coordonnées locales figées)
            el = world.elements.get(str(element_id))
            if el is None:
                continue

            pO = np.array(el.vertex_local_xy.get(0, (0.0, 0.0)), dtype=float)
            pB = np.array(el.vertex_local_xy.get(1, (0.0, 0.0)), dtype=float)
            pL = np.array(el.vertex_local_xy.get(2, (0.0, 0.0)), dtype=float)

            # mirrored = état "flip" uniquement (par défaut False à la pose initiale)
            mirrored = bool(tri.get("mirrored", False))

            # local effectif : si flip, on applique une réflexion locale M (le Core appliquera aussi mirrored)
            # NB: On fit R,T sur X' = M@X afin d'obtenir Y ≈ R @ (M @ X) + T.
            pO2 = (M @ pO) if mirrored else pO
            pB2 = (M @ pB) if mirrored else pB
            pL2 = (M @ pL) if mirrored else pL

            # Fit orthonormal 2D sur 3 points (Kabsch) — R DOIT être une rotation (det=+1)
            X = np.stack([pO2, pB2, pL2], axis=0)  # (3,2)
            Y = np.stack([Ow,  Bw,  Lw], axis=0)  # (3,2)
            Xc = X - X.mean(axis=0)
            Yc = Y - Y.mean(axis=0)
            H = Xc.T @ Yc
            U, _S, Vt = np.linalg.svd(H)
            R = (Vt.T @ U.T)
            # Forcer R ∈ SO(2) (det=+1). La réflexion est portée uniquement par `mirrored`.
            if np.linalg.det(R) < 0.0:
                Vt[1, :] *= -1.0
                R = (Vt.T @ U.T)
            T = Y.mean(axis=0) - (R @ X.mean(axis=0))

            world.setElementPose(str(element_id), R=R, T=T, mirrored=mirrored)


    def _autoSyncAllTopoPoses(self) -> None:
        """Auto: repère global unique => synchroniser *tous* les topoWorld auto depuis la géométrie monde."""
        for scen in (self.scenarios or []):
            if getattr(scen, "source_type", "manual") != "auto":
                continue
            # On parcours chaque groupe du scénario automatique (en principe un seul) et on synchronise le core
            for gid in scen.groups:
                self._sync_group_elements_pose_to_core(gid, scen)

    def _get_active_scenario(self) -> ScenarioAssemblage | None:
        if not self.scenarios:
            return None
        idx = int(self.active_scenario_index or 0)
        if idx < 0 or idx >= len(self.scenarios):
            return None
        return self.scenarios[idx]

    def _on_export_topodump_key(self, event=None):
        """Export TopoDump_<scenarioId>.xml du scénario actif (F11/F12)."""
        scen = self._get_active_scenario()
        world = scen.topoWorld
        out_name = "TopoDump.xml"
        out_path = os.path.join(self.topo_xml_dir, out_name)
        world.export_topo_dump_xml(out_path, orientation="cw")
        self.status.config(text=f"TopoDump exporté : {out_name}")

    # ---------- Dictionnaire : init ----------
    def _toggle_skip_overlap_highlight(self, event=None):
        self.debug_skip_overlap_highlight = not self.debug_skip_overlap_highlight
        state = "IGNORE" if self.debug_skip_overlap_highlight else "APPLIQUE"
        self.status.config(text=f"[DEBUG] Chevauchement (highlight): {state} — F9 pour basculer")

    def _toggle_debug_snap_assist(self, event=None):
        self._debug_snap_assist = not bool(self._debug_snap_assist)
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
        menubar.add_cascade(label="Affichage", menu=self.menu_visual)
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

        self.menu_visual.add_separator()
        self.menu_visual.add_command(
            label="Exporter en pdf…",
            command=self._export_view_pdf_dialog
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
        if not self.scenarios:
            return
        kept = [s for s in self.scenarios if getattr(s, "source_type", "manual") == "manual"]
        removed = len(self.scenarios) - len(kept)
        self.scenarios = kept
        if not self.scenarios:
            self.scenarios = [ScenarioAssemblage(name="Scénario manuel", source_type="manual")]
        self.active_scenario_index = min(self.active_scenario_index, len(self.scenarios) - 1)
        self._set_active_scenario(self.active_scenario_index)
        self._refresh_scenario_listbox()
        self.status.config(text=f"Scénarios auto supprimés : {removed}")

    # === Simulation : persistance "Même dernier run" (position + rotation auto) ===
    def _simulationGetMapKey(self) -> str:
        """Clé de carte (basename) pour la persistance simulation."""
        p = str(self.getAppConfigValue("bgSvgPath", "") or "").strip()
        if not p and isinstance(self._bg, dict):
            p = str(self._bg.get("path", "") or "").strip()
        return os.path.basename(p) if p else ""

    def _simulationGetLastAutoPlacement(self, map_key: str, order: str) -> Optional[Dict]:
        data = self.getAppConfigValue("simAutoPlacementByMap", {}) or {}
        mk = str(map_key or "").strip()
        ordk = str(order or "").strip().lower()
        if not mk or ordk not in ("forward", "reverse"):
            return None
        per_map = data.get(mk) or {}
        st = per_map.get(ordk)
        if isinstance(st, dict) and all(k in st for k in ("ox", "oy", "thetaDeg")):
            return st
        return None

    def _simulationSetLastAutoPlacement(self, map_key: str, order: str, state: Dict, save: bool = True):
        mk = str(map_key or "").strip()
        ordk = str(order or "").strip().lower()
        if not mk or ordk not in ("forward", "reverse"):
            return
        if not isinstance(state, dict):
            return
        st = {
            "ox": float(state.get("ox", 0.0)),
            "oy": float(state.get("oy", 0.0)),
            "thetaDeg": float(state.get("thetaDeg", 0.0)),
        }
        data = self.getAppConfigValue("simAutoPlacementByMap", {}) or {}
        if not isinstance(data, dict):
            data = {}
        per_map = data.get(mk)
        if not isinstance(per_map, dict):
            per_map = {}
        per_map[ordk] = st
        data[mk] = per_map
        self.setAppConfigValue("simAutoPlacementByMap", data)
        if save:
            self.saveAppConfig()

    def _simulationPersistCurrentAutoPlacement(self, order: Optional[str] = None, save: bool = True):
        """Persiste le repère global auto courant (ox/oy/thetaDeg) pour (carte, ordre)."""
        if self.auto_geom_state is None:
            return
        ordk = str(order or self._simulation_last_order or "forward").strip().lower()
        mk = self._simulationGetMapKey()
        self._simulationSetLastAutoPlacement(mk, ordk, self.auto_geom_state, save=save)

    def _simulation_assemble_dialog(self):
        """Ouvre la boîte de dialogue 'Assembler…' et lance l'algo choisi."""
        if self.df is None:
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
        default_algo_id = self._simulation_last_algo_id or (algo_items[0][0] if algo_items else "")
        default_n = self._simulation_last_n if self._simulation_last_n is not None else n_max
        default_order = self._simulation_last_order if self._simulation_last_order is not None else "forward"
        default_first_edge = self._simulation_last_first_edge if self._simulation_last_first_edge is not None else "OL"
        default_n = min(int(default_n), n_max)
        if default_n < 2:
            default_n = 2

        map_key = self._simulationGetMapKey()
        last_run_available_by_order = {
            "forward": self._simulationGetLastAutoPlacement(map_key, "forward") is not None,
            "reverse": self._simulationGetLastAutoPlacement(map_key, "reverse") is not None,
        }

        dlg = DialogSimulationAssembler(
            self,
            algo_items,
            n_max=n_max,
            default_algo_id=default_algo_id,
            default_n=default_n,
            default_order=default_order,
            default_first_edge=default_first_edge,
            last_run_available_by_order=last_run_available_by_order,
        )
        self.wait_window(dlg)
        if not getattr(dlg, "result", None):
            return

        algo_id, n, order, first_edge = dlg.result
        use_last_run = (str(first_edge or "") == "__LASTRUN__")
        # Si "Même dernier run" : on conserve le dernier edge connu (global) pour lancer l'algo
        first_edge_for_engine = self._simulation_last_first_edge if use_last_run else str(first_edge or "OL").upper().strip()
        if first_edge_for_engine not in ("OL", "BL"):
            first_edge_for_engine = "OL"

        self._simulation_last_order = order
        self._simulation_last_algo_id = algo_id
        self._simulation_last_n = int(n)

        # On ne remplace pas simLastFirstEdge si on a choisi "Même dernier run"
        if not use_last_run:
            self._simulation_last_first_edge = str(first_edge_for_engine or "OL").upper().strip()
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

        engine = MoteurSimulationAssemblage(self)
        engine.firstTriangleEdge = str(first_edge_for_engine or "OL").upper()
        algo_cls = ALGOS.get(algo_id)
        if algo_cls is None:
            raise ValueError(f"Algo inconnu: {algo_id}")
        algo = algo_cls(engine)

        # Snapshot de la carte auto (carte affichée au moment du lancement)
        self.auto_map_state = self._capture_map_state()
        scenarios = algo.run(tri_ids)

        base_idx = len(self.scenarios)
        count_auto = sum(1 for s in self.scenarios if s.source_type == "auto")
        for k, scen in enumerate(scenarios):
            if not isinstance(scen, ScenarioAssemblage):
                continue
            scen.source_type = "auto"
            scen.view_state = self._capture_view_state()
            scen.map_state = {}            
            scen.algo_id = scen.algo_id or algo_id
            scen.tri_ids = scen.tri_ids or list(tri_ids)
            if not scen.name:
                scen.name = f"Auto #{count_auto + k + 1}"
            self.scenarios.append(scen)

        # AUTO: construire géométrie locale + init transform global (commun)
        self._autoInitFromGeneratedScenarios(base_idx, order)

        # Persister le repère auto initial (carte, ordre) après génération
        # (sauf si on a choisi "Même dernier run" : on ne doit pas écraser l’état sauvegardé)
        if not use_last_run:
            self._simulationPersistCurrentAutoPlacement(order=order, save=True)

        # Option "Même dernier run" : appliquer la dernière position/rotation sauvegardée (carte, ordre)
        if use_last_run:
            map_key = self._simulationGetMapKey()
            st = self._simulationGetLastAutoPlacement(map_key, order)
            if isinstance(st, dict):
                self.auto_geom_state = {"ox": float(st.get("ox", 0.0)), "oy": float(st.get("oy", 0.0)), "thetaDeg": float(st.get("thetaDeg", 0.0))}
                self._autoRebuildWorldGeometry(redraw=False)
                self._redraw_from(self._last_drawn)

        self._refresh_scenario_listbox()
        self._set_active_scenario(base_idx)
        self.status.config(text=f"Simulation: {len(scenarios)} scénario(s) généré(s) (algo={algo_id}, n={n})")


    def _rebuild_triangle_file_list_menu(self):
        """
        Reconstruit la portion du menu 'Triangle' listant les fichiers disponibles
        dans self.data_dir. Charge direct au clic.
        """
        m = self.menu_triangle
        last_index = m.index("end")
        if last_index is None:
            return

        # Nouvelle organisation: [Ouvrir][sep][FICHIERS...]
        files_start = 2  # 0:"Ouvrir", 1:"sep", 2..:"fichiers"
        for i in range(last_index, files_start - 1, -1):
            m.delete(i)

        # On prend les fichiers Excel et on eclut les fichiers temporaires
        files = [
            f for f in os.listdir(self.data_dir)
            if f.lower().endswith(".xlsx") and not f.lstrip().startswith("~$")
        ]
        if not files:
            m.insert_command(files_start, label="(aucun fichier trouvé dans data)", state="disabled")
            return

        files.sort(key=lambda s: s.lower())
        for idx, fname in enumerate(files):
            full = os.path.join(self.data_dir, fname)
            m.insert_command(
                files_start + idx,
                label=fname,
                command=lambda p=full: self.open_excel(p)
            )

    # ---------- Icônes ----------  
    def _load_icon(self, filename: str):
        """
        Charge une icône depuis le dossier images.
        Retourne un tk.PhotoImage ou None si échec (fichier manquant, etc.).
        """
        base = self.images_dir
        if not base:
            return None
        path = os.path.join(base, filename)
        if not os.path.isfile(path):
            return None
        return tk.PhotoImage(file=path)


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
                self._redraw_from(self._last_drawn)

        # Radios à droite (icône-only). Fallback texte si l'icône n'est pas dispo.
        rb_kwargs = dict(
            variable=self._ui_triangleContourMode,
            indicatoron=0,
            padx=0,
            pady=0,
            command=_onTriangleContourModeChange,
        )
        if self.iconTriModeContour is not None:
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

        if self.iconTriModeEdges is not None:
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
        self._ui_decrypt_combo.bind("<<ComboboxSelected>>", lambda _e: _applyDecryptorFromUI())

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
                avail = int(pw.winfo_height() or 0)
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
        self.icon_scen_new = self._load_icon("scenario_new.png")
        self.icon_scen_open = self._load_icon("scenario_open.png")
        self.icon_scen_props = self._load_icon("scenario_props.png")
        self.icon_scen_save = self._load_icon("scenario_save.png")
        self.icon_scen_dup = self._load_icon("scenario_duplicate.png")
        self.icon_scen_del = self._load_icon("scenario_delete.png")

        # Icônes de type (affichées dans la Treeview)
        self.icon_scen_manual = self._load_icon("scenario_manual.png")
        self.icon_scen_auto = self._load_icon("scenario_auto.png")

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

        # Liste des scénarios (Treeview) : groupes Manuels / Automatiques
        # + colonnes de propriétés calculées sur l'assemblage final.

        # Spécification extensible des colonnes (ajout facile de nouvelles propriétés)
        # - getter(scen) doit retourner une string affichable ;
        #   en cas de valeur indéfinie, retourner "".
        self._scenario_prop_specs = [
            {
                "id": "dist_km",
                "title": "D(km)",
                "width": 80,
                "anchor": "center",
                "getter": self._scenarioPropDistanceKm,
            },
            {
                "id": "az_deg",
                "title": "Az(°)",
                "width": 70,
                "anchor": "center",
                "getter": self._scenarioPropAzimuthDeltaDeg,
            },
        ]

        columns = tuple([c["id"] for c in self._scenario_prop_specs])

        # IMPORTANT:
        # - ttk ajoute une indentation (~20px) pour les items enfants => gros espace avant l'icône.
        # - en plus, les colonnes (algo/status/n) réduisent #0 => libellés tronqués.
        style = ttk.Style()
        style.configure("Scenario.Treeview", indent=0)  # <- colle les icônes à gauche

        self.scenario_tree = ttk.Treeview(
            scen_lb_frame,
            columns=columns,
            show="tree headings",
            selectmode="browse",
            height=6,
            style="Scenario.Treeview",
        )
        # Colonne principale = ID (libellé)
        # Tri au clic sur en-tête
        self._scenario_tree_sort_col = None
        self._scenario_tree_sort_reverse = False
        self.scenario_tree.heading(
            "#0",
            text="ID",
            command=lambda c="#0": self._scenarioTreeSortBy(c),
        )
        self.scenario_tree.column("#0", width=230, stretch=True, anchor="w")

        # Colonnes propriétés
        for spec in self._scenario_prop_specs:
            cid = spec["id"]
            self.scenario_tree.heading(
                cid,
                text=str(spec.get("title", cid)),
                command=lambda c=cid: self._scenarioTreeSortBy(c),
            )
            self.scenario_tree.column(
                cid,
                width=int(spec.get("width", 70)),
                stretch=False,
                anchor=str(spec.get("anchor", "center")),
            )

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

        # --- Panneau bas : chemins (V1 : UI uniquement, liste vide) ---
        # Même approche que Triangles/Layers/Scénarios : panneau pliable + resize réel de la pane.
        chemins_minsize_expanded = 120

        if not hasattr(self, "_ui_chemins_collapsed"):
            self._ui_chemins_collapsed = tk.BooleanVar(value=False)

        chemins_frame = tk.Frame(pw, bd=0, highlightthickness=0)

        chemins_header = tk.Frame(chemins_frame)
        chemins_header.pack(fill=tk.X, pady=(0, 2))

        # Séparateur visuel sous le header
        ttk.Separator(chemins_frame, orient="horizontal").pack(fill=tk.X, pady=(0, 4))

        self._ui_chemins_content = tk.Frame(chemins_frame)
        if not self._ui_chemins_collapsed.get():
            self._ui_chemins_content.pack(fill=tk.BOTH, expand=True)

        def _calcCheminsExpandedHeightPx():
            """Hauteur 'exacte' pour afficher le contenu du panneau Chemins (hors remplissage)."""
            chemins_frame.update_idletasks()
            hdr_h = int(chemins_header.winfo_reqheight() or 26)
            content_h = int(self._ui_chemins_content.winfo_reqheight() or 0)
            return int(hdr_h + content_h + 22)  # + séparateur/paddings

        def _toggleCheminsPanel():
            collapsed = bool(self._ui_chemins_collapsed.get())
            self._ui_chemins_collapsed.set(not collapsed)
            if self._ui_chemins_collapsed.get():
                self._ui_chemins_content.pack_forget()
                self._ui_chemins_toggle_btn.config(text="▸")
            else:
                self._ui_chemins_content.pack(fill=tk.BOTH, expand=True)
                self._ui_chemins_toggle_btn.config(text="▾")

            chemins_frame.update_idletasks()
            hdr_h = int(chemins_header.winfo_reqheight() or 0)
            chemins_minsize_collapsed = max(28, hdr_h + 10)

            if self._ui_chemins_collapsed.get():
                # Important : si Chemins reste en stretch="always", il reprendra toute la hauteur.
                pw.paneconfigure(chemins_frame, stretch="never")
                pw.paneconfigure(chemins_frame, minsize=chemins_minsize_collapsed, height=chemins_minsize_collapsed)
                # ... et Scénarios absorbe TOUT le reste => plus de vide en bas
                pw.paneconfigure(scen_frame, stretch="always")
            else:
                # Scénarios redevient "fixe" ...
                pw.paneconfigure(scen_frame, stretch="never")

                pw.paneconfigure(chemins_frame, stretch="always")
                target_h = _calcCheminsExpandedHeightPx()
                target_h = max(int(chemins_minsize_expanded), int(target_h))
                pw.paneconfigure(chemins_frame, minsize=chemins_minsize_expanded, height=target_h)

        # Bouton toggle + titre cliquable
        self._ui_chemins_toggle_btn = tk.Button(
            chemins_header,
            text=("▸" if self._ui_chemins_collapsed.get() else "▾"),
            width=2,
            command=_toggleCheminsPanel
        )
        self._ui_chemins_toggle_btn.pack(side=tk.LEFT, padx=(0, 4))

        chemins_title = tk.Label(chemins_header, text="Chemins", font=(None, 9, "bold"))
        chemins_title.pack(side=tk.LEFT, anchor="w")
        chemins_title.bind("<Button-1>", lambda _e: _toggleCheminsPanel())
        chemins_header.bind("<Button-1>", lambda _e: _toggleCheminsPanel())

        # Barre d'actions (éditer, recalculer, supprimer)
        chemins_toolbar = tk.Frame(self._ui_chemins_content, bd=0, highlightthickness=0)
        chemins_toolbar.pack(anchor="w", padx=6, pady=(0, 2), fill="x")
        self.icon_chemin_recalc = self._load_icon("refresh-cw.png")
        self.icon_chemin_engine = self._load_icon("iconCpu24.png")

        def _make_chemin_btn(parent, icon, text, cmd, tooltip_text: str):
            if icon is not None:
                b = tk.Button(parent, image=icon, command=cmd, relief=tk.FLAT)
            else:
                kwargs = {"text": text, "command": cmd, "relief": tk.FLAT}
                if len(str(text)) <= 2:
                    kwargs["width"] = 2
                b = tk.Button(parent, **kwargs)
            self._ui_attach_tooltip(b, tooltip_text)
            return b

        self.chemins_edit_btn = _make_chemin_btn(
            chemins_toolbar,
            self.icon_scen_props,
            "✎",
            self.onEditerChemin,
            "Éditer le chemin",
        )
        self.chemins_edit_btn.pack(side=tk.LEFT, padx=1)

        self.chemins_recalc_btn = _make_chemin_btn(
            chemins_toolbar,
            self.icon_chemin_recalc,
            "Recalculer le chemin",
            self.onRecalculerChemin,
            "Recalculer le chemin",
        )
        self.chemins_recalc_btn.pack(side=tk.LEFT, padx=1)

        self.chemins_engine_btn = _make_chemin_btn(
            chemins_toolbar,
            self.icon_chemin_engine,
            "CPU",
            self.onDecryptageEngine,
            "Moteur de décryptage",
        )
        self.chemins_engine_btn.pack(side=tk.LEFT, padx=1)

        self.chemins_delete_btn = _make_chemin_btn(
            chemins_toolbar,
            self.icon_scen_del,
            "🗑",
            self._chemins_delete_selected,
            "Supprimer le chemin",
        )
        self.chemins_delete_btn.pack(side=tk.LEFT, padx=1)

        chemins_lb_frame = tk.Frame(self._ui_chemins_content, bd=0, highlightthickness=0)
        chemins_lb_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Treeview "Chemins" (vide en V1)
        self.chemins_tree = ttk.Treeview(
            chemins_lb_frame,
            columns=("triplet", "angle"),
            show="headings",
            selectmode="browse",
            height=6,
        )
        self.chemins_tree.heading("triplet", text="Triplet")
        self.chemins_tree.heading("angle", text="Angle")
        self.chemins_tree.column("triplet", width=170, stretch=True, anchor="w")
        self.chemins_tree.column("angle", width=70, stretch=False, anchor="center")
        self.chemins_tree.bind("<<TreeviewSelect>>", self._onCheminsTreeSelect)
            
        self.chemins_scroll = ttk.Scrollbar(
            chemins_lb_frame, orient="vertical", command=self.chemins_tree.yview
        )
        self.chemins_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chemins_tree.configure(yscrollcommand=self.chemins_scroll.set)
        self.chemins_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        pw.add(chemins_frame, minsize=chemins_minsize_expanded)

        # Les panneaux du haut ne doivent pas "aspirer" la hauteur quand la fenêtre grandit.
        # Désormais, "Chemins" est collé en bas et prend la place restante.
        pw.paneconfigure(tri_frame, stretch="never")
        pw.paneconfigure(layer_frame, stretch="never")
        pw.paneconfigure(decrypt_frame, stretch="never")
        pw.paneconfigure(scen_frame, stretch="never")
        pw.paneconfigure(chemins_frame, stretch="always")

        # Remplir la liste des scénarios existants (pour l'instant : le manuel)
        self._refresh_scenario_listbox()
        self.refreshCheminTreeView()

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
        grp_auto = tree.insert("", tk.END, iid="grp_auto",   text="Automatiques", open=True)

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

            # Valeurs des colonnes additionnelles (propriétés calculées)
            values = []
            for spec in (self._scenario_prop_specs or []):
                getter = spec.get("getter")
                if callable(getter):
                    v = getter(scen)
                    values.append("" if v is None else str(v))
                else:
                    values.append("")

            kwargs = {"text": text, "tags": tags, "values": tuple(values)}
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
                    kids = tree.get_children(parent)
                    if kids:
                        tree.selection_set(kids[0])
                        tree.see(kids[0])
                        break

    # =========================
    #  CHEMINS (V1 : UI uniquement)
    # =========================
    def _onCheminsTreeSelect(self, _evt=None) -> None:
        if not hasattr(self, "chemins_tree"):
            return
        tree = self.chemins_tree
        sel = tree.selection()
        if not sel:
            return

        # On récupère le topology chemin et le GroupID de la bordure
        world = self._get_active_scenario().topoWorld
        tc = world.topologyChemins
        groupId = tc.groupId

        iid = sel[0]
        t = self._cheminsTripletByIid[iid]

        # 3 noeuds DSU (IDs)
        nodePrevId = t.nodeA
        nodeCenterId = t.nodeO
        nodeNextId = t.nodeB

        # Construire un snap_target minimal centré sur nodeO
        snapTarget = {
            "nodeDsu": nodeCenterId,
            "topoGroupId": str(groupId),
        }

        # --- world coords du node central (obligatoire pour déplacer la clock)
        xO, yO = world.getConceptNodeWorldXY(nodeCenterId, groupId)
        wO = (float(xO), float(yO))
        sx, sy = self._world_to_screen(wO)

        # on positionne la clock sur le noeud
        self._clock_anchor_world = np.array(wO, dtype=float)
        self._clock_cx, self._clock_cy = float(sx), float(sy)

        # On met à jour les informations d'azimut
        self._clock_arc_auto_from_snap_target(
            snapTarget,
            prevNodeDsu=nodePrevId,
            nextNodeDsu=nodeNextId,
            drag=False,
        )

    def _onCheminsTreeActivate(self, evt=None) -> None:
        # simple alias si tu veux déclencher seulement au double-clic
        self._onCheminsTreeSelect(evt)

    def refreshCheminTreeView(self) -> None:
        """Rafraîchit la TreeView Chemins depuis world.topologyChemins (lecture seule)."""
        if not hasattr(self, "chemins_tree"):
            return
        tree = self.chemins_tree
        self._cheminsTripletByIid = {}
        
        for iid in tree.get_children(""):
            tree.delete(iid)

        scen = self._get_active_scenario()
        world = scen.topoWorld
        tc = world.topologyChemins
        isDefined = bool(tc.isDefined)
        if hasattr(self, "chemins_edit_btn"):
            self.chemins_edit_btn.configure(state=(tk.NORMAL if isDefined else tk.DISABLED))
        if hasattr(self, "chemins_recalc_btn"):
            self.chemins_recalc_btn.configure(state=(tk.NORMAL if isDefined else tk.DISABLED))
        if hasattr(self, "chemins_engine_btn"):
            self.chemins_engine_btn.configure(state=(tk.NORMAL if isDefined else tk.DISABLED))
        if hasattr(self, "chemins_delete_btn"):
            self.chemins_delete_btn.configure(state=(tk.NORMAL if isDefined else tk.DISABLED))
        if not isDefined:
            return

        for t in tc.getTriplets():
            if not t.isGeometrieValide:
                raise RuntimeError("Triplet sans géométrie valide")
            tripletStr = (
                f"{world.getNodeLabel(t.nodeA)} - "
                f"{world.getNodeLabel(t.nodeO)} - "
                f"{world.getNodeLabel(t.nodeB)}"
            )
            angleStr = f"{float(t.angleDeg):.2f}°"
            iid = tree.insert("", tk.END, values=(tripletStr, angleStr))
            self._cheminsTripletByIid[iid] = t

    def onEditerChemin(self) -> None:
        """Édition V6 du chemin: orientationUser + selectionMask (ordre snapshot)."""
        scen = self._get_active_scenario()
        if scen is None:
            return
        world = scen.topoWorld
        if world is None:
            return
        tc = world.topologyChemins
        if not tc.isDefined:
            return

        snapshotNodes = [str(n) for n in list(tc.borderSnapshotNodes)]
        currentMask = [bool(v) for v in list(tc.selectionMask)]
        n = len(snapshotNodes)
        if n != len(currentMask):
            raise RuntimeError("Édition du chemin impossible : mask/snapshot incohérents.")

        groupId = str(tc.groupId)
        boundaryOrientation = str(world.getBoundaryOrientation(groupId)).strip().lower()
        if boundaryOrientation not in ("cw", "ccw"):
            raise RuntimeError(f"Édition du chemin impossible : boundaryOrientation invalide ({boundaryOrientation}).")

        currentOrientation = str(tc.orientationUser).strip().lower()
        if currentOrientation not in ("cw", "ccw"):
            raise RuntimeError(f"Édition du chemin impossible : orientationUser invalide ({currentOrientation}).")

        dlg = tk.Toplevel(self)
        dlg.title("Éditer le chemin")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        root = tk.Frame(dlg, padx=10, pady=10)
        root.pack(fill=tk.BOTH, expand=True)

        tk.Label(root, text="Sens").pack(anchor="w")
        orientationVar = tk.StringVar(value=currentOrientation)
        orientRow = tk.Frame(root)
        orientRow.pack(fill=tk.X, pady=(2, 8))
        tk.Radiobutton(orientRow, text="Sens horaire", value="cw", variable=orientationVar).pack(side=tk.LEFT, padx=(0, 10))
        tk.Radiobutton(orientRow, text="Sens inverse", value="ccw", variable=orientationVar).pack(side=tk.LEFT)

        tk.Label(root, text="Liste des nœuds").pack(anchor="w", pady=(0, 2))
        listFrame = tk.Frame(root, bd=1, relief=tk.GROOVE)
        listFrame.pack(fill=tk.BOTH, expand=True)

        nodesFrame = tk.Frame(listFrame)
        nodesFrame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        viewVars: list[tk.BooleanVar] = []
        viewInverted = (str(orientationVar.get() or "cw").strip().lower() != boundaryOrientation)

        def _snapshotIndexFromView(viewIndex: int, count: int, inverted: bool) -> int:
            if not inverted:
                return int(viewIndex)
            if int(viewIndex) == 0:
                return 0
            return int(count - viewIndex)

        def _syncMaskFromCurrentView(inverted: bool) -> None:
            if len(viewVars) != n:
                return
            for viewIndex, var in enumerate(viewVars):
                snapshotIndex = _snapshotIndexFromView(viewIndex, n, inverted)
                currentMask[snapshotIndex] = bool(var.get())

        def _rebuildView() -> None:
            nonlocal viewInverted
            _syncMaskFromCurrentView(viewInverted)
            for w in nodesFrame.winfo_children():
                w.destroy()
            viewVars.clear()

            inverted = str(orientationVar.get() or "cw").strip().lower() != boundaryOrientation
            viewInverted = inverted
            for viewIndex in range(n):
                snapshotIndex = _snapshotIndexFromView(viewIndex, n, inverted)
                conceptNodeId = str(snapshotNodes[snapshotIndex])
                rawLabel = world.getConceptNodeLabel(conceptNodeId)
                displayLabel = str(rawLabel).strip() if rawLabel is not None else ""
                if not displayLabel:
                    displayLabel = "(sans label)"
                var = tk.BooleanVar(value=bool(currentMask[snapshotIndex]))
                viewVars.append(var)
                tk.Checkbutton(
                    nodesFrame,
                    text=displayLabel,
                    variable=var,
                    anchor="w",
                    justify="left",
                ).pack(anchor="w")

        _rebuildView()
        orientationVar.trace_add("write", lambda *_: _rebuildView())

        actions = tk.Frame(root)
        actions.pack(fill=tk.X, pady=(8, 0))

        def _onOk() -> None:
            newOrientationUser = str(orientationVar.get() or "cw").strip().lower()
            if newOrientationUser not in ("cw", "ccw"):
                raise RuntimeError(f"Édition du chemin impossible : orientationUser invalide ({newOrientationUser}).")

            newSelectionMaskSnapshotOrder = [False] * n
            inverted = newOrientationUser != boundaryOrientation
            for viewIndex, var in enumerate(viewVars):
                snapshotIndex = _snapshotIndexFromView(viewIndex, n, inverted)
                newSelectionMaskSnapshotOrder[snapshotIndex] = bool(var.get())

            if sum(1 for v in newSelectionMaskSnapshotOrder if v) < 3:
                messagebox.showerror("Éditer le chemin", "Au moins 3 nœuds doivent rester sélectionnés.", parent=dlg)
                return

            world.topologyChemins.appliquerEdition(newOrientationUser, newSelectionMaskSnapshotOrder)
            dlg.destroy()
            self.refreshCheminTreeView()

        def _onCancel() -> None:
            dlg.destroy()

        tk.Button(actions, text="Annuler", command=_onCancel).pack(side=tk.RIGHT)
        tk.Button(actions, text="OK", command=_onOk).pack(side=tk.RIGHT, padx=(0, 6))

        dlg.protocol("WM_DELETE_WINDOW", _onCancel)
        dlg.wait_visibility()
        dlg.focus_set()
        dlg.wait_window()

    def onRecalculerChemin(self) -> None:
        """Demande au Core de recalculer le chemin courant puis rafraîchit l'UI."""
        scen = self._get_active_scenario()
        tc = scen.topoWorld.topologyChemins
        if not tc.isDefined:
            return

        tc.recalculerChemin()

        self.refreshCheminTreeView()

    def onDecryptageEngine(self) -> None:
        messagebox.showinfo("Décryptage", "Non implémentée", parent=self)

    def _chemins_edit_selected(self):
        """Compat: redirige vers l'éditeur V6."""
        self.onEditerChemin()

    def _chemins_delete_selected(self):
        """Supprime le chemin courant (Core) après confirmation."""
        scen = self._get_active_scenario()
        if scen is None:
            return
        world = scen.topoWorld
        if world is None:
            return
        tc = world.topologyChemins
        if not tc.isDefined:
            self.refreshCheminTreeView()
            return

        if not messagebox.askokcancel("Supprimer le chemin", "Supprimer le chemin ?"):
            return

        tc.supprimerChemin()
        self.refreshCheminTreeView()

    # =========================
    # Scénarios: propriétés calculées (Treeview)
    # =========================

    def _scenarioTreeSortBy(self, col_id: str):
        """Tri au clic sur une colonne de la Treeview Scénarios.

        - Trie séparément dans chaque groupe (Manuels / Automatiques).
        - Tri numérique pour D(km)/Az(°) quand possible.
        - Les valeurs vides sont toujours envoyées en bas.
        """
        if not hasattr(self, "scenario_tree"):
            return
        tree = self.scenario_tree
 
        # Toggle sens de tri
        if self._scenario_tree_sort_col == col_id:
            self._scenario_tree_sort_reverse = not bool(self._scenario_tree_sort_reverse)
        else:
            self._scenario_tree_sort_col = col_id
            self._scenario_tree_sort_reverse = False
 
        # Helpers de parsing
        def _parseDisplayIdFromText(txt: str):
            s = str(txt or "").strip()
            if s.startswith("★"):
                s = s.lstrip("★ ").strip()
            m = re.match(r"^#(\d+)", s)
            return int(m.group(1)) if m else None
 
        def _coerceKey(val: str):
            s = str(val or "").strip()
            if s == "":
                return (1, 0.0, "")  # vide => en bas
            # numérique si possible
            try:
                return (0, float(s.replace(",", ".")), "")
            except Exception:
                return (0, 0.0, s.lower())
 
        def _itemKey(iid: str):
            if col_id == "#0":
                txt = tree.item(iid, "text")
                did = _parseDisplayIdFromText(txt)
                if did is None:
                    # fallback string (vide en bas)
                    s = str(txt or "").strip()
                    if s == "":
                        return (1, 0.0, "")
                    return (0, 0.0, s.lower())
                return (0, float(did), "")
            else:
                return _coerceKey(tree.set(iid, col_id))
 
        # Trier dans chaque groupe sans casser la structure
        for parent in ("grp_manual", "grp_auto"):
            if not (hasattr(tree, "exists") and tree.exists(parent)):
                continue
            kids = list(tree.get_children(parent))
            if not kids:
                continue
            kids_sorted = sorted(
                kids,
                key=_itemKey,
                reverse=bool(self._scenario_tree_sort_reverse),
            )
            for pos, iid in enumerate(kids_sorted):
                tree.move(iid, parent, pos)
 
        # Optionnel: petit indicateur ▲/▼ sur l'en-tête trié
        base_id_title = "ID"
        tree.heading("#0", text=base_id_title, command=lambda c="#0": self._scenarioTreeSortBy(c))
        for spec in (self._scenario_prop_specs or []):
            cid = spec.get("id")
            if not cid:
                continue
            title = str(spec.get("title", cid))
            tree.heading(cid, text=title, command=lambda c=cid: self._scenarioTreeSortBy(c))

        arrow = " ▼" if bool(self._scenario_tree_sort_reverse) else " ▲"
        if self._scenario_tree_sort_col == "#0":
            tree.heading("#0", text=base_id_title + arrow, command=lambda c="#0": self._scenarioTreeSortBy(c))
        elif self._scenario_tree_sort_col:
            # retrouver le titre de la colonne
            title = None
            for spec in (self._scenario_prop_specs or []):
                if spec.get("id") == self._scenario_tree_sort_col:
                    title = str(spec.get("title", self._scenario_tree_sort_col))
                    break
            if title is None:
                title = str(self._scenario_tree_sort_col)
            tree.heading(self._scenario_tree_sort_col, text=title + arrow,
                            command=lambda c=self._scenario_tree_sort_col: self._scenarioTreeSortBy(c))

    def _scenarioGetFirstLastTriangles(self, scen):
        """Retourne (t_first, t_last) en se basant sur scen.tri_ids (ordre d'assemblage).

        Contrat: en fin d'assemblage, ces triangles doivent exister et contenir les clés 'pts' avec 'L' et 'B'.
        """
        if scen is None:
            return (None, None)

        last_drawn = getattr(scen, "last_drawn", None)
        tri_ids = list(getattr(scen, "tri_ids", None) or [])
        if not last_drawn or not tri_ids:
            return (None, None)

        first_id = int(tri_ids[0])
        last_id = int(tri_ids[-1])

        t_first = None
        t_last = None
        for t in last_drawn:
            tid = t.get("id")
            if tid is None:
                continue
            if int(tid) == first_id:
                t_first = t
            if int(tid) == last_id:
                t_last = t

        if t_first is None or t_last is None:
            raise RuntimeError(
                f"scenarioGetFirstLastTriangles: triangles introuvables dans last_drawn (first={first_id}, last={last_id})"
            )
        return (t_first, t_last)

    def _scenarioPropDistanceKm(self, scen) -> str:
        """Distance (km) entre les 2 points de Lumière libres: L du 1er triangle et L du dernier triangle.

        - Nécessite une carte chargée + calibration Lambert.
        - Affichage: 1 décimale, toujours positive.
        - Si pas de carte: retourne "".
        """
        # Pas de carte => pas de distance (par design)
        if self._bg is None:
            return ""

        (t_first, t_last) = self._scenarioGetFirstLastTriangles(scen)
        if t_first is None or t_last is None:
            return ""

        P1 = (t_first.get("pts") or {}).get("L")
        P2 = (t_last.get("pts") or {}).get("L")
        if P1 is None or P2 is None:
            raise RuntimeError("scenarioPropDistanceKm: point 'L' manquant sur first/last")

        x1_km, y1_km = self._bgWorldToLambertKm(float(P1[0]), float(P1[1]))
        x2_km, y2_km = self._bgWorldToLambertKm(float(P2[0]), float(P2[1]))
        d = math.hypot(x2_km - x1_km, y2_km - y1_km)
        return f"{abs(d):.1f}"

    def _scenarioPropAzimuthDeltaDeg(self, scen) -> str:
        """Différence d'azimut (degrés) entre LB du 1er triangle et LB du dernier triangle.

        - Ordre strict L -> B.
        - 0° = Nord, sens horaire.
        - Normalisé sur [0..360).
        - Si azimut indéterminé (points confondus): retourne "".
        """
        (t_first, t_last) = self._scenarioGetFirstLastTriangles(scen)
        if t_first is None or t_last is None:
            return ""

        P1 = t_first.get("pts") or {}
        P2 = t_last.get("pts") or {}
        if "L" not in P1 or "B" not in P1 or "L" not in P2 or "B" not in P2:
            raise RuntimeError("scenarioPropAzimuthDeltaDeg: points 'L'/'B' manquants sur first/last")

        a1 = self._azimuthDegFromWorld(P1["L"], P1["B"])
        a2 = self._azimuthDegFromWorld(P2["L"], P2["B"])
        if a1 is None or a2 is None:
            return ""
        d = (float(a2) - float(a1)) % 360.0
        return f"{d:.1f}"

    def _azimuthDegFromWorld(self, p1, p2) -> Optional[float]:
        """Azimut absolu en degrés pour le segment p1->p2.

        - p1/p2 sont des points en coordonnées monde.
        - 0° = Nord (axe +Y), 90° = Est (axe +X).
        Retourne None si les points sont confondus.
        """
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return None
        # atan2(E, N) => 0° au Nord, sens horaire
        ang = math.degrees(math.atan2(dx, dy)) % 360.0
        return float(ang)

    def _bgWorldToLambertKm(self, wx: float, wy: float) -> Tuple[float, float]:
        """Convertit un point monde -> Lambert93 (km), en tenant compte du redimensionnement/déplacement du fond.

        Contrat:
        - nécessite une calibration (affineWorldToLambertKm + bgWorldRectAtCalibration).
        - si la calibration est absente alors qu'une carte est chargée, c'est un bug (on lève).
        """
        if self._bg is None:
            raise RuntimeError("bgWorldToLambertKm: pas de carte chargée")
        data = self._bg_calib_data
        if not isinstance(data, dict):
            raise RuntimeError("bgWorldToLambertKm: calibration absente (_bg_calib_data)")
        aff = data.get("affineWorldToLambertKm")
        if not (isinstance(aff, list) and len(aff) == 6):
            raise RuntimeError("bgWorldToLambertKm: affineWorldToLambertKm absent de la calibration")
        rect_cal = data.get("bgWorldRectAtCalibration")
        if not isinstance(rect_cal, dict):
            raise RuntimeError("bgWorldToLambertKm: bgWorldRectAtCalibration absent de la calibration")

        rect_cur = self._bg
        # Similarité entre monde_calibration -> monde_actuel (resize + translation)
        w_cal = float(rect_cal.get("w"))
        h_cal = float(rect_cal.get("h"))
        w_cur = float(rect_cur.get("w"))
        h_cur = float(rect_cur.get("h"))
        if w_cal <= 0 or h_cal <= 0 or w_cur <= 0 or h_cur <= 0:
            raise RuntimeError("bgWorldToLambertKm: rect invalide")

        sx = w_cur / w_cal
        sy = h_cur / h_cal

        x0_cal = float(rect_cal.get("x0"))
        y0_cal = float(rect_cal.get("y0"))
        x0_cur = float(rect_cur.get("x0"))
        y0_cur = float(rect_cur.get("y0"))

        # Inverse: monde_actuel -> monde_calibration
        wx_cal = x0_cal + (float(wx) - x0_cur) / sx
        wy_cal = y0_cal + (float(wy) - y0_cur) / sy

        a, b, c, d, e, f = [float(v) for v in aff]
        x_km = a * wx_cal + b * wy_cal + c
        y_km = d * wx_cal + e * wy_cal + f
        return (float(x_km), float(y_km))

    def _update_triangle_listbox_colors(self):
        """
        Met à jour la couleur des entrées de la listbox des triangles
        en fonction de leur utilisation dans le scénario actif.
        Triangles utilisés → grisés, triangles disponibles → noir.
        """
        if not hasattr(self, "listbox"):
            return
        lb = self.listbox
        size = lb.size()

        for idx in range(size):
            txt = lb.get(idx)

            tri_id = None
            m = re.match(r"\s*(\d+)\.", str(txt))
            if m:
                tri_id = int(m.group(1))

            if tri_id is not None and tri_id in self._placed_ids:
                # Triangle déjà posé dans le scénario → grisé
                lb.itemconfig(idx, fg="gray50")

            else:
                # Triangle disponible
                lb.itemconfig(idx, fg="black")

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

        idx = int(iid.split("_", 1)[1])
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

        Contrat:
        Dans groups[n]["nodes"], chaque node doit fournir explicitement:
          - edge_in  (arête partagée avec le précédent)  : "OB"|"BL"|"LO"|None
          - edge_out (arête partagée avec le suivant)    : "OB"|"BL"|"LO"|None
        """
        sig = {}
        if scen is None:
            return sig

        last_drawn = getattr(scen, "last_drawn", None)
        groups = getattr(scen, "groups", None)

        if not last_drawn or not groups:
            return sig

        # Contrat: groups/nodes doivent fournir edge_in/edge_out (normalisation centralisée)
        self._normalize_all_groups(scen)

        def tri_id_from_tid(tid):
            """tid = index dans last_drawn ; retourne triangle["id"]"""
            tid = int(tid)
            if 0 <= tid < len(last_drawn):
                return int(last_drawn[tid].get("id"))

            return None

        def add_link(tri_a, edge_a, tri_b, edge_b):
            if tri_a is None or tri_b is None or edge_a is None or edge_b is None:
                return
            sig.setdefault(tri_a, {})[edge_a] = (tri_b, edge_b)

        # Parcours de tous les groupes (un groupe = une chaîne de nodes)
        for gid, g in groups.items():
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

                edge_left = left.get("edge_out")
                edge_right = right.get("edge_in")
                if not edge_left or not edge_right:
                    raise RuntimeError(
                        f"scenario_connections_signature: edge manquant gid={gid!r} i={i} "
                        f"edge_out={edge_left!r} edge_in={edge_right!r}"
                    )

                add_link(tri_left, edge_left, tri_right, edge_right)
                add_link(tri_right, edge_right, tri_left, edge_left)

        return sig

    def _normalize_group_nodes(self, scen, gid):
        """Normalise la représentation des nodes d'un groupe.

        Contrat canonique (unique) pour les nodes:
          - tid: int (index dans scen.last_drawn)
          - edge_in:  "OB"|"BL"|"LO"|None
          - edge_out: "OB"|"BL"|"LO"|None

        (Legacy supprimé) : les nodes ne portent plus vkey_in/vkey_out.
        """
        if scen is None:
            raise RuntimeError("normalize_group_nodes: scen is None")

        groups = getattr(scen, "groups", None)
        if not groups or gid not in groups:
            raise RuntimeError(f"normalize_group_nodes: groupe introuvable gid={gid!r}")

        g = groups[gid] or {}
        nodes = g.get("nodes")
        if nodes is None:
            raise RuntimeError(f"normalize_group_nodes: nodes manquant gid={gid!r}")
        if not isinstance(nodes, list):
            raise RuntimeError(f"normalize_group_nodes: nodes doit être une liste gid={gid!r}")

        allowed_edges = {"OB", "BL", "LO"}

        for i, n in enumerate(nodes):
            if not isinstance(n, dict):
                raise RuntimeError(f"normalize_group_nodes: node invalide gid={gid!r} i={i} type={type(n)}")

            if "tid" not in n:
                raise RuntimeError(f"normalize_group_nodes: tid manquant gid={gid!r} i={i}")

            n["tid"] = int(n["tid"])

            # Hygiène: supprimer les champs legacy si présents
            n.pop("vkey_in", None)
            n.pop("vkey_out", None)

            # edge_out / edge_in : doivent être déjà présents (contrat canonique)
            eout = n.get("edge_out")
            ein = n.get("edge_in")

            # Validation stricte du contrat
            ein = n.get("edge_in")
            eout = n.get("edge_out")
            if ein is not None and ein not in allowed_edges:
                raise RuntimeError(f"normalize_group_nodes: edge_in invalide gid={gid!r} i={i} edge_in={ein!r}")
            if eout is not None and eout not in allowed_edges:
                raise RuntimeError(f"normalize_group_nodes: edge_out invalide gid={gid!r} i={i} edge_out={eout!r}")

        # Validation supplémentaire: chaque liaison interne doit être définie
        if len(nodes) >= 2:
            for i in range(len(nodes) - 1):
                left = nodes[i]
                right = nodes[i + 1]
                if not left.get("edge_out"):
                    raise RuntimeError(
                        f"normalize_group_nodes: edge_out manquant gid={gid!r} i={i} (liaison {i}->{i+1})"
                    )
                if not right.get("edge_in"):
                    raise RuntimeError(
                        f"normalize_group_nodes: edge_in manquant gid={gid!r} i={i+1} (liaison {i}->{i+1})"
                    )

        g["nodes"] = nodes
        groups[gid] = g

    def _normalize_all_groups(self, scen):
        """Normalise tous les groups/nodes d'un scénario (contrat edge_in/out obligatoire)."""
        if scen is None:
            raise RuntimeError("normalize_all_groups: scen is None")
        groups = getattr(scen, "groups", None)
        if not groups:
            return
        for gid in list(groups.keys()):
            self._normalize_group_nodes(scen, gid)

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
            tri_id_to_idx[int(tri.get("id"))] = idx

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
                    out[kk] = (int(v[0]), str(v[1]))
                else:
                    out[kk] = int(v)

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

    def _capture_view_state(self) -> dict:
        return {
            "zoom": float(self.zoom or 1.0),
            "offset_x": float(self.offset[0]) if hasattr(self, "offset") else 0.0,
            "offset_y": float(self.offset[1]) if hasattr(self, "offset") else 0.0,
        }
 
    def _apply_view_state(self, vs: dict | None):
        if not vs:
            return
        self.zoom = float(vs.get("zoom", self.zoom or 1.0))
        ox = float(vs.get("offset_x", self.offset[0] if hasattr(self, "offset") else 0.0))
        oy = float(vs.get("offset_y", self.offset[1] if hasattr(self, "offset") else 0.0))
        self.offset = np.array([ox, oy], dtype=float)
 
    def _capture_map_state(self) -> dict:
        bg = self._bg
        state = {
            "path": str(bg.get("path")) if isinstance(bg, dict) and bg.get("path") else "",
            "x0": float(bg.get("x0")) if isinstance(bg, dict) and bg.get("x0") is not None else 0.0,
            "y0": float(bg.get("y0")) if isinstance(bg, dict) and bg.get("y0") is not None else 0.0,
            "w": float(bg.get("w")) if isinstance(bg, dict) and bg.get("w") is not None else 0.0,
            "h": float(bg.get("h")) if isinstance(bg, dict) and bg.get("h") is not None else 0.0,
            "visible": bool(self.show_map_layer.get()) if hasattr(self, "show_map_layer") else True,
            "opacity": int(self.map_opacity.get()) if hasattr(self, "map_opacity") else 100,
            "scale": None,
        }
        if hasattr(self, "_bg_compute_scale_factor"):
            state["scale"] = self._bg_compute_scale_factor()
        if state["scale"] is None:
            state["scale"] = self._bg_scale_factor_override
        return state
 
    def _apply_map_state(self, ms: dict | None, persist: bool = False):
        if ms is None:
            return

        if hasattr(self, "show_map_layer") and "visible" in ms:
            self.show_map_layer.set(bool(ms.get("visible")))
        if hasattr(self, "map_opacity") and "opacity" in ms:
            self.map_opacity.set(int(ms.get("opacity", self.map_opacity.get())))

        if "scale" in ms:
            self._bg_scale_factor_override = ms.get("scale")

        path = str(ms.get("path") or "").strip()
        if not path:
            self._bg_clear(persist=persist)
            return

        if not os.path.isfile(path):
            self._bg_clear(persist=persist)
            self._status_warn(f"Carte introuvable: {path}")
            return
 
        rect = {
            "x0": float(ms.get("x0", 0.0)),
            "y0": float(ms.get("y0", 0.0)),
            "w": float(ms.get("w", 0.0)),
            "h": float(ms.get("h", 0.0)),
        }

        # --- OPTIM: ne pas recharger la carte + calibration si rien n'a changé ---
        bg = self._bg or {}
        samePath = str(bg.get("path") or "") == str(path or "")
        eps = 1e-9
        sameRect = (
            abs(float(bg.get("x0") or 0.0) - float(rect.get("x0") or 0.0)) < eps and
            abs(float(bg.get("y0") or 0.0) - float(rect.get("y0") or 0.0)) < eps and
            abs(float(bg.get("w")  or 0.0) - float(rect.get("w")  or 0.0)) < eps and
            abs(float(bg.get("h")  or 0.0) - float(rect.get("h")  or 0.0)) < eps
        )
        sameScale = (self._bg_scale_factor_override == ms.get("scale"))

        if samePath and sameRect and sameScale:
            # Rien à faire : on a déjà exactement cette carte dans ce repère monde
            return

        self._bg_set_map(path, rect_override=rect, persist=persist)

    # ---------- AUTO (scénarios automatiques): transform géométrique global ----------

    def _get_active_scenario(self):
        if not self.scenarios:
            return None
        idx = int(self.active_scenario_index or 0)
        if idx < 0 or idx >= len(self.scenarios):
            return None
        return self.scenarios[idx]

    def _is_active_auto_scenario(self) -> bool:
        scen = self._get_active_scenario()
        return bool(scen is not None and getattr(scen, "source_type", "manual") == "auto")

    def _autoGetAnchorWorld(self, scen):
        """Retourne l'ancre monde (sommet L du triangle de référence).

        IMPORTANT:
        - Ne pas se baser sur l'ordre de `last_drawn`, qui peut varier.
        - On se base sur `scen.tri_ids` (ordre d'assemblage) et on retrouve le triangle correspondant dans `last_drawn`.
        """
        if scen is None or not getattr(scen, "last_drawn", None):
            return None

        tri_ids = list(getattr(scen, "tri_ids", None) or [])

        tid_ref = None
        if tri_ids:
            tid_ref = tri_ids[0]

        t_ref = None
        if tid_ref is not None:
            for t in scen.last_drawn:
                if int(t.get("id", -1)) == int(tid_ref):
                    t_ref = t
                    break

        if t_ref is None:
            # fallback legacy: premier/dernier de last_drawn
            idx = 0 if str(order) == "forward" else (len(scen.last_drawn) - 1)
            idx = max(0, min(int(idx), len(scen.last_drawn) - 1))
            t_ref = scen.last_drawn[idx]

        P = (t_ref or {}).get("pts", {})
        if "L" not in P:
            return None
        return np.array(P["L"], dtype=float)

    def _autoEnsureLocalGeometry(self, scen):
        """Construit scen.last_drawn_local si absent (coords relatives à l'ancre L_ref)."""
        if scen is None or getattr(scen, "source_type", "manual") != "auto":
            return
        if getattr(scen, "last_drawn_local", None) is not None:
            return
        anchor = self._autoGetAnchorWorld(scen)
        if anchor is None:
            return

        local = []
        for t in (scen.last_drawn or []):
            tt = dict(t)
            P = t.get("pts", {})
            Pl = {}
            for k in ("O", "B", "L"):
                if k in P:
                    Pl[k] = np.array(P[k], dtype=float) - anchor
            tt["pts"] = Pl
            local.append(tt)
        scen.last_drawn_local = local

        # init auto_geom_state si nécessaire (origine = ancre monde, theta=0)
        if self.auto_geom_state is None:
            self.auto_geom_state = {"ox": float(anchor[0]), "oy": float(anchor[1]), "thetaDeg": 0.0}


    def _autoRebuildWorldGeometryScenario(self, scen: ScenarioAssemblage = None) -> None:
        if self.auto_geom_state is None:
            return
        if scen is None:
            scen = self._get_active_scenario()
        if scen.source_type != "auto":
            return

        # garantir la géométrie locale
        self._autoEnsureLocalGeometry(scen)

        local = scen.last_drawn_local
        if not local:
            return

        ox = float(self.auto_geom_state.get("ox", 0.0))
        oy = float(self.auto_geom_state.get("oy", 0.0))
        thetaDeg = float(self.auto_geom_state.get("thetaDeg", 0.0))

        th = math.radians(thetaDeg)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=float)
        origin = np.array([ox, oy], dtype=float)

        world = []
        for tloc in local:
            tt = dict(tloc)
            Ploc = tloc.get("pts", {})
            Pw = {}
            for k in ("O", "B", "L"):
                if k in Ploc:
                    v = np.array(Ploc[k], dtype=float)
                    Pw[k] = origin + (R @ v)
            tt["pts"] = Pw
            world.append(tt)

        scen.last_drawn = world

        # si c'est le scénario actif, on raccorde l'UI
        if scen is self._get_active_scenario():
            self._last_drawn = scen.last_drawn
            self.groups = scen.groups
            for gid in list(self.groups.keys()):
                self._recompute_group_bbox(gid)

    def _autoRebuildWorldGeometry(self, redraw: bool = True) -> None:
        for scen in (self.scenarios or []):
            if scen.source_type != "auto":
                continue
            self._autoRebuildWorldGeometryScenario(scen)
        if redraw:
            self._redraw_from(self._last_drawn)

    def _autoInitFromGeneratedScenarios(self, base_idx: int, order: str):
        """Après génération d'autos, construit les géométries locales + initialise le transform global."""
        if not self.scenarios:
            return
        for i in range(int(base_idx), len(self.scenarios)):
            scen = self.scenarios[i]
            if scen.source_type != "auto":
                continue
            scen.autoOrder = str(order or "forward")
            # ensure local + possibly init auto_geom_state
            self._autoEnsureLocalGeometry(scen)

        # rebuild monde une fois pour tout aligner (no-op visuellement au départ)
        self._autoRebuildWorldGeometry(redraw=False)

    def _convertActiveAutoToManualSnapshot(self):
        """Convertit le scénario auto actif en un nouveau scénario manuel (snapshot monde), et retourne son index."""
        scen = self._get_active_scenario()
        if scen.source_type != "auto":
            return None

        # Snapshot monde courant (déjà transformé par auto_geom_state)
        def clone_last_drawn_world(last_drawn):
            out = []
            for t in (last_drawn or []):
                tt = dict(t)
                P = t.get("pts", {})
                Pw = {}
                for k in ("O", "B", "L"):
                    if k in P:
                        Pw[k] = np.array(P[k], dtype=float).copy()
                tt["pts"] = Pw
                out.append(tt)
            return out

        def clone_groups(groups):
            if not isinstance(groups, dict):
                return {}
            g2 = {}
            for gid, g in groups.items():
                gg = dict(g)
                gg["nodes"] = [dict(n) for n in (g.get("nodes") or [])]
                if "bbox" in gg and gg["bbox"] is not None:
                    try:
                        gg["bbox"] = [float(x) for x in gg["bbox"]]
                    except Exception:
                        pass
                g2[gid] = gg
            return g2

        name = str(getattr(scen, "name", "") or "Snapshot")
        name = name + " (manuel)" if "manuel" not in name.lower() else name

        new_scen = ScenarioAssemblage(name=name, source_type="manual")
        new_scen.last_drawn = clone_last_drawn_world(getattr(scen, "last_drawn", None))
        new_scen.groups = clone_groups(getattr(scen, "groups", None))
        new_scen.topoWorld = scen.topoWorld.clonePhysicalState()

        # copier quelques métadonnées utiles
        for attr in ("algo_id", "tri_ids", "status"):
            if hasattr(scen, attr):
                setattr(new_scen, attr, getattr(scen, attr))

        new_scen.view_state = self._capture_view_state()
        new_scen.map_state = self._capture_map_state()

        self.scenarios.append(new_scen)
        return len(self.scenarios) - 1

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

        # Sauvegarder l'état courant dans l'ancien scénario (vue + carte)
        prev = self.scenarios[self.active_scenario_index]
        prevIsAuto = (getattr(prev, "source_type", "manual") == "auto")

        # Vue: en AUTO, on synchronise => état partagé
        if prevIsAuto:
            self.auto_view_state = self._capture_view_state()
        else:
            prev.view_state = self._capture_view_state()

        # Carte: en AUTO, on synchronise => état partagé
        if prevIsAuto:
            self.auto_map_state = self._capture_map_state()
        else:
            prev.map_state = self._capture_map_state()

        scen = self.scenarios[index]
        self.active_scenario_index = index

        # Rattacher les structures géométriques du scénario courant
        self._last_drawn = scen.last_drawn
        self.groups = scen.groups

        # Restaurer carte + vue (sans écraser la config globale)
        scenIsAuto = (getattr(scen, "source_type", "manual") == "auto")

        if scenIsAuto:
            self._apply_map_state(self.auto_map_state, persist=False)
        else:
            self._apply_map_state(getattr(scen, "map_state", None), persist=False)

        if scenIsAuto:
            # Vue AUTO partagée, fallback si jamais pas encore initialisée
            self._apply_view_state(self.auto_view_state or getattr(scen, "view_state", None))
        else:
            self._apply_view_state(getattr(scen, "view_state", None))

        # --- AUTO: réconcilier les métadonnées de groupe (group_id/group_pos) avec scen.groups ---
        # Certains chemins (redraw / sélection / tools) se basent encore sur les champs portés par les triangles.
        # Si un scénario auto fournit un dictionnaire groups cohérent mais que les triangles n'ont pas été annotés,
        # on peut se retrouver avec un triangle "orphelin" ou des groupes recomposés de travers.

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
                    _tid_i = int(_tid)

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
                self._recompute_group_bbox(_gid)

            # 5) topoGroupId : propager triangle -> groupe UI (sécurisation)
            for _gid, _g in scen.groups.items():
                _nodes = (_g or {}).get("nodes", []) or []
                if not _nodes:
                    continue
                _tid = _nodes[0].get("tid")
                if _tid is None:
                    continue
                _tid_i = int(_tid)
                if not (0 <= _tid_i < len(scen.last_drawn)):
                    continue
                _core_gid = scen.last_drawn[_tid_i].get("topoGroupId", None)
                if _core_gid is not None:
                    scen.groups[_gid]["topoGroupId"] = _core_gid

        # Recalibrer _next_group_id si besoin
        self._next_group_id = (max(self.groups.keys()) + 1) if self.groups else 1

        # Recalcule la liste des triangles déjà utilisés pour ce scénario
        self._placed_ids = {
            int(t["id"]) for t in self._last_drawn
            if t.get("id") is not None
        }

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

        self.refreshCheminTreeView()

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
        self.status.config(text=f"Nom du scénario mis à jour : {scen.name}")

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
        dup.topoWorld = src.topoWorld.clonePhysicalState()

        self.scenarios.append(dup)
        self._refresh_scenario_listbox()
        self._set_active_scenario(new_index)
        self.status.config(text=f"Scénario dupliqué : {dup.name}")

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
        if self.ref_scenario_token is not None and id(scen) == self.ref_scenario_token:
            self.ref_scenario_token = None

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
        self.status.config(text=f"Scénario supprimé : {scen.name}")

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
        self._ctx_menu.add_separator()
        self._ctx_menu.add_command(label="Créer un chemin…", command=self._ctx_CreerChemin)

        # Mémoriser l'index des entrées "OL=0°" / "BL=0°" pour pouvoir les (dés)activer au vol
        self._ctx_idx_ol0 = 4
        self._ctx_idx_bl0 = 5

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

        # Export TopoDump (manuel, snapshot volontaire)
        self.bind_all("<F11>", self._on_export_topodump_key)

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
        cw_now = int(self.canvas.winfo_width() or 0)
        ch_now = int(self.canvas.winfo_height() or 0)

        # Si la taille a réellement changé, on programme un redraw complet.
        # (sinon, le fond de carte reste « figé » jusqu'au prochain pan/zoom)
        if cw_now > 2 and ch_now > 2:
            last_sz = self._last_canvas_size
            if last_sz != (cw_now, ch_now):
                self._last_canvas_size = (cw_now, ch_now)
                if self._resize_redraw_after_id is not None:
                    self.after_cancel(self._resize_redraw_after_id)

                self._resize_redraw_after_id = self.after(40, self._do_resize_redraw)

        if self._bg_defer_redraw and self._bg:
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
        self._redraw_from(self._last_drawn)


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
                if not self.dicoSheet:
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
        r = int(self._clock_radius)
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
        if not self.canvas:
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
        Z = float(self.zoom)
        Ox, Oy = (float(self.offset[0]), float(self.offset[1]))

        def W2S(p):
            # monde -> écran (canvas)
            return (Ox + p[0] * Z, Oy - p[1] * Z)
        for t in self._last_drawn:
            P = t.get("pts", {})
            O = P.get("O")
            B = P.get("B")
            L = P.get("L")
            if O is None or B is None or L is None:
                t["_pick_poly"] = None
                t["_pick_pts"] = {}
                continue
            Os = W2S(O)
            Bs = W2S(B)
            Ls = W2S(L)
            t["_pick_pts"] = {"O": Os, "B": Bs, "L": Ls}
            t["_pick_poly"] = [Os, Bs, Ls]
        self._pick_cache_valid = True


    # centralisation des bindings canvas/clavier --

    def _bind_canvas_handlers(self):
        """(Ré)applique tous les bindings nécessaires au canvas et au clavier.
        À appeler après création du canvas ET après un chargement de scénario."""
        if not self.canvas:
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
        Z = float(self.zoom)
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

        # Si scénario AUTO : le figer en MANUEL (snapshot) avant export
        if self._is_active_auto_scenario():
            new_idx = self._convertActiveAutoToManualSnapshot()
            if new_idx is not None:
                self._refresh_scenario_listbox()
                self._set_active_scenario(int(new_idx))

        self.save_scenario_xml(path)
        self.status.config(text=f"Scénario enregistré dans {path}")

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

        new_index = len(self.scenarios)

        scen = ScenarioAssemblage(
            name=name,
            source_type="manual",
            algo_id=None,
            tri_ids=[],
        )
        # Structures indépendantes pour ce scénario
        scen.last_drawn = []
        scen.view_state = self._capture_view_state()
        scen.map_state = self._capture_map_state()
        scen.groups = {}

        self.scenarios.append(scen)

        # On bascule sur ce nouveau scénario AVANT de charger,
        # ainsi load_scenario_xml() écrit bien dans ses structures.
        self._set_active_scenario(new_index)
        self.load_scenario_xml(path)

        # Succès : rafraîchir la liste et le statut
        self._refresh_scenario_listbox()
        self.status.config(text=f"Scénario importé : {scen.name}")

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
        files = [f for f in os.listdir(self.scenario_dir) if f.lower().endswith(".xml")]

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
            self._tooltip.attributes("-topmost", True)

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
            self._tooltip.destroy()
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

        if not hasattr(self, "appConfig") or self.appConfig is None:
            self.appConfig = {}
        if not self._bg:
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

    def autoLoadBackgroundAtStartup(self):
        """Recharge le fond SVG et sa géométrie depuis la config.

        Important : au moment où _build_ui() s'exécute, le canvas peut encore avoir une taille
        (winfo_width/height) quasi nulle. Dans ce cas, _bg_draw_world_layer() ne peut pas
        rasteriser/afficher correctement et le fond semble "non rechargé".

        Solution : charger le SVG *après* la 1ère passe de layout (after_idle) et forcer
        un redraw complet au 1er <Configure> valide.
        """
        if self._bg_startup_scheduled:
            return
        self._bg_startup_scheduled = True
        self.after_idle(self._autoLoadBackgroundAfterLayout)

    def _autoLoadBackgroundAfterLayout(self):
        """(interne) Effectue le vrai rechargement du fond après layout."""

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
        self.update_idletasks()

        if isinstance(rect, dict) and all(k in rect for k in ("x0", "y0", "w", "h")):
            self._bg_set_map(svg_path, rect_override=rect, persist=False)
        else:
            self._bg_set_map(svg_path, rect_override=None, persist=False)

        # Si le canvas était encore trop petit au moment du redraw, on redessinera une fois
        # dès qu'un <Configure> avec une taille valide arrive.
        self._bg_defer_redraw = True

    def autoLoadTrianglesFileAtStartup(self):
        """Recharge au démarrage le dernier fichier triangles ouvert."""
        # 1) Dernier fichier explicitement chargé
        last = self.getAppConfigValue("lastTriangleExcel", "")
        if last and os.path.isfile(str(last)):
            self.load_excel(str(last))
            return

        # 2) Fallback : data/triangle.xlsx
        default = os.path.join(self.data_dir, "triangle.xlsx")
        if default and os.path.isfile(default):
            self.load_excel(default)

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
        col_B = cmap.get("base")
        col_L = cmap.get("lumiere")
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


    # ---------- Mise en page simple (aperçu brut) ----------
    def _triangle_from_index(self, idx):
        """Construit un triangle 'local' depuis l’élément sélectionné de la listbox.
        IMPORTANT: on parse l'ID affiché (NN.) au lieu d'utiliser df.iloc[idx],
        car la listbox peut avoir des éléments retirés → indices décalés.
        """
        if self.df is None or self.df.empty:
            raise RuntimeError("Pas de données — ouvre d'abord l’Excel.")
        # Récupérer le texte de la listbox et extraire l'id (NN. ...)
        lb_txt = ""
        if 0 <= idx < self.listbox.size():
            lb_txt = self.listbox.get(idx)

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
        ori = str(r.get("orient", "CCW")).upper()

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

    # ====== FRONTIER GRAPH HELPERS (factorisation) ===============================================
    def _ang_of_vec(self, vx, vy):
        import math
        return math.atan2(vy, vx)

    def _ang_diff(self, a, b):
        # plus petit écart absolu d’angle
        return abs(self._ang_wrap(a - b))

    def _refresh_listbox_from_df(self):
        """Recharge la listbox des triangles depuis self.df (sans filtrage)."""
        self.listbox.delete(0, tk.END)
        if self.df is not None and not self.df.empty:
            for _, r in self.df.iterrows():
                self.listbox.insert(tk.END, f"{int(r['id']):02d}. B:{r['B']}  L:{r['L']}")


    def clear_canvas(self):
        """Efface l'affichage après confirmation, et remet à jour la liste des triangles."""
        if not messagebox.askyesno(
            "Effacer l'affichage",
            "Voulez-vous effacer l'affichage et réinitialiser la liste des triangles ?"
        ):
            return
        self.canvas.delete("all")

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
        if not self.canvas:
            return

        self.canvas.delete("clock_overlay")
        self._draw_clock_overlay()

    def _draw_clock_overlay(self):
        """
        Dessine une horloge en haut-gauche du canvas, à taille FIXE (px),
        indépendante du zoom/pan (coordonnées canvas).
        Utilise self._clock_state = {'hour':h, 'minute':m, 'label':str}.
        """
        if not self.canvas:
            return
        # Nettoyer l'ancien overlay
        self.canvas.delete("clock_overlay")

        # Si le compas est masqué via le menu, ne rien dessiner
        if hasattr(self, "show_clock_overlay") and not self.show_clock_overlay.get():
            return
        # Paramètres d'aspect
        margin = 12              # marge par rapport aux bords du canvas (px)
        # rayon (px) — modifiable via UI (min=50)
        R = max(50, int(self._clock_radius))
        # Si un ancrage monde existe (et qu'on ne drag pas), le centre du compas suit pan/zoom
        if self._clock_anchor_world is not None and not self._clock_dragging:
            sx, sy = self._world_to_screen(self._clock_anchor_world)
            self._clock_cx, self._clock_cy = float(sx), float(sy)
        # Si on a déjà une position écran mais pas d'ancrage monde, on l'initialise
        if self._clock_anchor_world is None and self._clock_cx is not None and self._clock_cy is not None and not self._clock_dragging:
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
        ref_az = float(self._clock_ref_azimuth_deg) % 360.0

        # Couleurs
        col_circle = "#b0b0b0"   # gris cercle
        col_ticks = "#707070"
        col_hour = "#0b3d91"   # bleue (petite aiguille)
        col_min = "#000000"   # noire  (grande aiguille)

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
        x3, y3 = _pos(ref_az + 90.0,  r_text)
        x6, y6 = _pos(ref_az + 180.0, r_text)
        x9, y9 = _pos(ref_az + 270.0, r_text)

        def _fmt_mark(v):
            fv = float(v)
            if abs(fv - round(fv)) < 1e-9:
                return str(int(round(fv)))
            # 1 décimale max
            return f"{fv:.1f}".rstrip("0").rstrip(".")

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
            if self._dico_filter_active:
                last = self._clock_arc_last
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
        if self.canvas:
            self.canvas.delete("clock_snap_target")

    def _world_to_screen(self, p):
        x = self.offset[0] + float(p[0]) * self.zoom
        y = self.offset[1] - float(p[1]) * self.zoom
        return x, y

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
        self._update_current_scenario_differences()

        self.canvas.delete("all")
        # Fond carte (si layer visible)
        if self.show_map_layer is None or self.show_map_layer.get():
            self._bg_draw_world_layer()     

        # l'ID de la ligne n'est plus valide après delete("all")
        self._nearest_line_id = None
        # on efface les IDs de surlignage déjà dessinés (mais on conserve le choix et les données)
        self._clear_edge_highlights()

        # Mode "contours uniquement" : même visualisation (noeuds, tags, numéros, mot),
        # mais SANS les arêtes internes. En plus, on trace l'enveloppe extérieure des groupes.
        showContoursMode = bool(
            self.show_only_group_contours is not None
            and self.show_only_group_contours.get()
        )

        onlyContours = False
        v = self.only_group_contours
        if v is not None and hasattr(v, "get"):
            onlyContours = bool(v.get())
        else:
            onlyContours = bool(self._only_group_contours)

        # Si on est en mode "contour only", on force la suppression des arêtes internes.
        if showContoursMode:
            onlyContours = True

        # 1) Triangles (toujours dessinés si layer actif) : on coupe juste les arêtes internes.
        if self.show_triangles_layer is None or self.show_triangles_layer.get():
            for i, t in enumerate(placed):
                labels = t["labels"]
                P = t["pts"]
                tri_id = t.get("id")
                fill = "#ffd6d6" if i in self._comparison_diff_indices else None
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
                if self._edge_highlights:
                    self._redraw_edge_highlights()
                # remet la ligne grise
                idx = self._sel["idx"]
                vkey = self._sel["vkey"]
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
        if not self.groups:
            return

        for gid, g in self.groups.items():
            nodes = g.get("nodes") or []
            if not nodes:
                continue
            outline = self._group_outline_segments_topo(gid)
            for p1, p2 in outline:
                x1, y1 = self._world_to_screen(p1)
                x2, y2 = self._world_to_screen(p2)
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill="#000000",
                    width=3,
                    tags=("group_outline",),
                )

    def _draw_triangle_screen(self, P, 
                              outline="black", width=2, labels=None, inset=0.35, 
                              tri_id=None, tri_mirrored=False, fill=None, diff_outline=False, 
                              drawEdges=True):
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
                value = value.strip()

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
            id_y = sy - (gap // 2)
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
        if self._debug_snap_assist:
            print(msg)

    # --- helpers: mode déconnexion (CTRL) + curseur ---

    def _on_ctrl_down(self, event=None):
        # Pendant une mesure d'azimut du compas, CTRL sert uniquement à désactiver le snap.
        # On évite donc d'activer le mode "déconnexion" des triangles (curseur + aides).
        if self._clock_measure_active:
            self._ctrl_down = True
            return

        if not self._ctrl_down:
            self._ctrl_down = True
            # Masquer tout tooltip en mode déconnexion
            self._hide_tooltip()
            self.canvas.configure(cursor="X_cursor")

            # --- NOUVEAU (étape 2) ---
            # Si CTRL est pressé *après* avoir sélectionné un sommet (assist ON),
            # on applique immédiatement la ROTATION d'alignement (comme au release),
            # mais SANS translation et SANS collage (CTRL empêchera le snap au relâchement).
            if self._sel and (not self._sel.get("suppress_assist")):
                mode = self._sel.get("mode")
                choice = self._edge_choice
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
                                    self._recompute_group_bbox(gid)

                                    self._redraw_from(self._last_drawn)
                                    v_world = np.array(self._last_drawn[anchor_tid]["pts"][anchor_vkey], dtype=float)
                                    self._update_nearest_line(v_world, exclude_idx=anchor_tid, exclude_gid=gid)
                                    self._update_edge_highlights(anchor_tid, anchor_vkey, idx_t, vkey_t)

    def _on_ctrl_up(self, event=None):
        if self._ctrl_down:
            self._ctrl_down = False
            self.canvas.configure(cursor="")

    # ---------- Mouse navigation ----------
    # Drag depuis la liste

    def _on_triangle_list_select(self, event=None):
        """Empêche la sélection d'un triangle déjà utilisé dans le scénario courant."""
        # éviter la récursion quand on modifie la sélection nous-mêmes
        if self._in_triangle_select_guard:
            return
        if not hasattr(self, "listbox"):
            return

        sel = self.listbox.curselection()
        if not sel:
            self._last_triangle_selection = None
            return

        idx = int(sel[0])
        tri = self._triangle_from_index(idx)
        tid = int(tri.get("id"))

        placed = self._placed_ids or set()
        if tid is not None and tid in placed:
            # Triangle déjà posé : on annule la sélection visuelle
            self._in_triangle_select_guard = True
            self.listbox.selection_clear(0, tk.END)
            if (self._last_triangle_selection is not None and
                    0 <= self._last_triangle_selection < self.listbox.size()):
                self.listbox.selection_set(self._last_triangle_selection)
            self._in_triangle_select_guard = False
            if hasattr(self, "status"):
                self.status.config(text=f"Triangle {tid} déjà utilisé dans ce scénario.")
            return

        # Sélection valide
        self._last_triangle_selection = idx

    # ---------- Mouse navigation ----------
    # Drag depuis la liste

    def _on_list_mouse_down(self, event):
        """
        Démarre un drag & drop depuis la listbox,
        sauf si le triangle est déjà utilisé dans le scénario courant.
        """
        # Pas de DF → rien à faire
        if self.df is None or self.df.empty:
            return

        # Index de la ligne cliquée dans la listbox
        i = self.listbox.nearest(event.y)
        if i < 0:
            return

        # Sélection visuelle
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(i)

        # Récupération de la définition du triangle
        tri = self._triangle_from_index(i)

        tri_id = tri.get("id")
        # Si le triangle est déjà posé dans ce scénario, on bloque le drag
        if tri_id is not None and int(tri_id) in self._placed_ids:
            self.status.config(text=f"Triangle {tri_id} déjà utilisé dans ce scénario.")
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
        self.canvas.configure(cursor="hand2")
        self.status.config(text="Glissez le triangle sur le canvas puis relâchez pour le déposer.")

    # ---------- Neighbours for tooltip ----------

    def _point_on_segment(self, P, A, B, eps):
        """Vrai si P appartient au segment [A,B] (colinéarité + projection dans [0,1])."""
        Ax, Ay = float(A[0]), float(A[1])
        Bx, By = float(B[0]), float(B[1])
        Px, Py = float(P[0]), float(P[1])
        ABx, ABy = Bx-Ax, By-Ay
        APx, APy = Px-Ax, Py-Ay
        cross = abs(ABx*APy - ABy*APx)
        if cross > eps:
            return False
        dot = ABx*APx + ABy*APy
        ab2 = ABx*ABx + ABy*ABy
        if ab2 <= eps:
            return False
        t = dot/ab2
        return -eps <= t <= 1.0+eps

    def _display_name(self, key, labels_tuple):
        """Retourne le libellé affiché (sans préfixe) pour O/B/L."""

        if key == "O" :
            raw = labels_tuple[0]
        elif key == "B" :
            raw = labels_tuple[1]
        else:
            raw = labels_tuple[2]

        s = str(raw or "").strip()
        return s

    def _on_canvas_motion_update_drag(self, event):
        # Mode compas : mesure d'un azimut (relatif à la référence)
        if self._clock_measure_active:
            self._clock_measure_update_preview(int(event.x), int(event.y))
            return "break"

        # Mode compas : mesure d'arc d'angle
        if self._clock_arc_active:
            self._clock_arc_update_preview(int(event.x), int(event.y))
            return "break"

        # Mode compas : définition de l'azimut de référence
        if self._clock_setref_active:
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

        # 2) Mode rotation de GROUPE : suivre la souris (sans bouton appuyé)
        if self._sel and self._sel.get("mode") == "rotate_group":
            sel = self._sel
            gid = sel["gid"]
            pivot = sel["pivot"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            cur_angle = math.atan2(wy - pivot[1], wx - pivot[0])
            dtheta = cur_angle - sel["start_angle"]

            # AUTO: rotation globale partagée (doit impacter TOUS les scénarios auto)
            if sel.get("auto_geom"):
                if self.auto_geom_state is None:
                    self._autoEnsureLocalGeometry(self._get_active_scenario())
                if self.auto_geom_state is None:
                    return
                theta0 = float(sel.get("auto_theta0", float(self.auto_geom_state.get("thetaDeg", 0.0))))
                self.auto_geom_state["thetaDeg"] = float(theta0 + math.degrees(dtheta))
                self._autoRebuildWorldGeometryScenario(None)
                self._redraw_from(self._last_drawn)
                self._sel["last_angle"] = cur_angle
                return

            # MANUAL
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
                    for k in ("O", "B", "L"):
                        v = np.array(Orig[k], dtype=float) - pivot
                        Pt[k] = (R @ v) + pivot
            self._recompute_group_bbox(gid)

            self._redraw_from(self._last_drawn)
            self._sel["last_angle"] = cur_angle
            return

        # 3) Pas de drag/rotation : gestion du TOOLTIP (survol de sommet)
        mode, idx, extra = self._hit_test(event.x, event.y)

        # En mode déconnexion (CTRL), ne pas afficher de tooltip
        if self._ctrl_down:
            self._hide_tooltip()
            return

        if mode == "vertex" and idx is not None:
            vkey = extra if isinstance(extra, str) else None
            # position monde du sommet visé
            P0 = self._last_drawn[idx]["pts"]
            v_world = np.array(P0[vkey], dtype=float) if vkey in ("O","B","L") else None

            scen = self._get_active_scenario()
            topoWorld = scen.topoWorld

            tooltip_txt = ""
            if v_world is not None and vkey in ("O", "B", "L"):
                vkey_to_n = {"O": 0, "B": 1, "L": 2}
                triNum = int(self._last_drawn[idx].get("id", idx + 1))
                nodeId = topoWorld.format_node_id(element_id=f"T{triNum:02d}", vertex_index=vkey_to_n[vkey])

                lines = ["Noeuds:"]
                phys_nodes = topoWorld.getPhysicalNodesForConceptNode(nodeId)
                for nid in phys_nodes:
                    lines.append(f"- {topoWorld.getPhysicalNodeName(nid)}")
                lines.append("Liens connectés:")
                rays = topoWorld.getConceptRays(nodeId)
                for ray in rays:
                    other = ray["otherNodeId"]
                    az = ray["azDeg"]
                    lines.append(f"- {topoWorld.getConceptNodeName(other)} @ {float(az):0.2f}°")
                tooltip_txt = "\n".join(lines)
            if tooltip_txt:
                # === Placement robuste par CENTRE du tooltip ===
                # 1) Coordonnées CANVAS du sommet et du barycentre
                sx_v, sy_v = self._world_to_screen(v_world)
                Cx = (P0["O"][0] + P0["B"][0] + P0["L"][0]) / 3.0
                Cy = (P0["O"][1] + P0["B"][1] + P0["L"][1]) / 3.0

                sx_c, sy_c = self._world_to_screen((Cx, Cy))
                # 2) Direction (centroïde -> sommet) en PIXELS CANVAS
                vx = float(sx_v) - float(sx_c)
                vy = float(sy_v) - float(sy_c)
                n = (vx*vx + vy*vy) ** 0.5
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
                cushion_px = float(self._tooltip_cushion_px)
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
            for k in ("O", "B", "L"):
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
            self.canvas.delete(self._nearest_line_id)
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
        _id = self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
        return _id

    # ---------- utilitaires contour de groupe ----------
    def _group_outline_segments_topo(self, gid: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Retourne le contour du groupe gid depuis la topologie (segments monde)."""
        g = self.groups.get(gid) or {}
        core_gid = g.get("topoGroupId", None)
        if not core_gid:
            raise ValueError(f"TopoGroupId manquant pour le groupe {gid}")

        scen = self._get_active_scenario()
        topoWorld = scen.topoWorld

        segments = topoWorld.getBoundarySegments(str(core_gid))
        if not segments:
            return []

        element_map = {}
        for t in (self._last_drawn or []):
            if "topoElementId" in t:
                element_map[str(t["topoElementId"])] = t

        outline = []
        for bs in segments:
            element_id = str(bs.elementId)
            if element_id not in element_map:
                raise ValueError(f"Element manquant dans last_drawn: {element_id}")
            tri = element_map[element_id]
            pts = tri.get("pts", None)
            if not isinstance(pts, dict):
                raise ValueError(f"Points invalides pour {element_id}")

            def vkeyFromNodeId(node_id: str, element_id: str) -> str:
                nid = str(node_id)
                if not nid.startswith(str(element_id) + ":N"):
                    raise ValueError(f"[BOUNDARY][TK] nodeId not in element: node={nid} elem={element_id}")
                if nid.endswith(":N0"):
                    return "O"
                if nid.endswith(":N1"):
                    return "B"
                if nid.endswith(":N2"):
                    return "L"
                raise ValueError(f"[BOUNDARY][TK] unexpected node suffix: {nid}")

            k0 = vkeyFromNodeId(bs.fromNodeId, element_id)
            k1 = vkeyFromNodeId(bs.toNodeId, element_id)
            if k0 not in pts or k1 not in pts:
                raise ValueError(f"Sommet manquant pour {element_id}: {k0}/{k1}")

            pA = np.array(pts[k0], dtype=float)
            pB = np.array(pts[k1], dtype=float)
            t0 = float(bs.t0)
            t1 = float(bs.t1)
            q0 = pA + (pB - pA) * t0
            q1 = pA + (pB - pA) * t1
            outline.append((q0, q1))
        return outline

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
            A = np.array(A, float)
            B = np.array(B, float)
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

        for ln in (lines or []):

            coords = list(getattr(ln, "coords", []))
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

        # Par défaut (eps=None), on calcule un eps "rendu" pour recoller les micro-écarts
        # numériques (auto-assemblage) et éviter que des arêtes internes ressortent comme contours.
        # IMPORTANT : certains appels (ex: snap/assist) ont besoin des sommets exacts ; dans ce cas,
        # passer explicitement eps=EPS_WORLD (ou autre) pour éviter tout regroupement de sommets.
        if eps is None:
            tol_world = max(
                1e-9,
                float(self.stroke_px) / max(self.zoom, 1e-9)
            )
            eps = max(EPS_WORLD, 0.5 * tol_world)
        return self._group_outline_segments(gid, eps=float(eps))

    def _update_edge_highlights(self, mob_idx: int, vkey_m: str, tgt_idx: int, tgt_vkey: str):
        """Version factorisée 'graphe de frontière' (symétrique) :
        1) On récupère l'outline des deux groupes
        2) On construit un graphe de frontière (half-edges) côté mobile & cible
        3) Au sommet cliqué de chaque côté, on prend **les 2 demi-arêtes incidentes**
        4) On sélectionne la paire (mobile×cible) qui **minimise globalement** Δ-angle
        5) Affichage : toujours bleues (incidentes), rouge pour la paire choisie
        """

        def _to_np(p):
            return np.array([float(p[0]), float(p[1])], dtype=float)

        def _azim(a, b):
            # tuple-safe: accepte tuples ou np.array
            return atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))

        def _ang_dist(a, b):
            d = abs(a - b) % (2*pi)
            return d if d <= pi else (2*pi - d)

        def _almost_eq(a, b, eps=EPS_WORLD):
            return abs(a[0]-b[0]) <= eps and abs(a[1]-b[1]) <= eps

        # Sommets & groupes
        tri_m = self._last_drawn[mob_idx]
        Pm = tri_m["pts"]
        vm = _to_np(Pm[vkey_m])
        tri_t = self._last_drawn[tgt_idx]
        Pt = tri_t["pts"]
        vt = _to_np(Pt[tgt_vkey])
        gid_m = tri_m.get("group_id")
        gid_t = tri_t.get("group_id")
        if gid_m is None or gid_t is None:
            self._clear_edge_highlights()
            self._edge_choice = None
            return

        # On récupère la topo
        scen = self._get_active_scenario()
        world = scen.topoWorld
        # On récupère les ID des TopoNodes mA et tA
        tAElementId = str(tri_t["topoElementId"])
        idx = {"O": 0, "B": 1, "L": 2}[tgt_vkey]
        tAId = world.format_node_id(tAElementId, idx)

        mAElementId = str(tri_m["topoElementId"])
        idx = {"O": 0, "B": 1, "L": 2}[vkey_m]
        mAId = world.format_node_id(mAElementId, idx)

        # DEBUG
        self._dbgSnap(
            f"[snap] update_edge_highlights mob={mob_idx}:{vkey_m} (gid={gid_m}) -> tgt={tgt_idx}:{tgt_vkey} (gid={gid_t})"
        )

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
        S_m_base = _group_shape_from_nodes(self._group_nodes(gid_m), self._last_drawn)
        # Helper de pose rigide (vm->vt, alignement (vm→mB) // (vt→tB))

        def _place_union_on_pair(S_base, vm_pt, mB_pt, vt_pt, tB_pt):

            from shapely.affinity import translate, rotate
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

        # Helper pour tester le chevauchement via Algo Topologique
        def _overlap_topo_for_pair(score, me, te) -> bool:
            g_m = self.groups.get(gid_m) or {}
            g_t = self.groups.get(gid_t) or {}
            core_gid_m = g_m.get("topoGroupId", None)
            core_gid_t = g_t.get("topoGroupId", None)
            srcGroup = world.groups.get(str(core_gid_t)) if core_gid_t else None
            dstGroup = world.groups.get(str(core_gid_m)) if core_gid_m else None
            if srcGroup is None or dstGroup is None:
                return False

            mob_tids = [nd.get("tid") for nd in (self._group_nodes(gid_m) or []) if nd.get("tid") is not None]
            tgt_tids = [nd.get("tid") for nd in (self._group_nodes(gid_t) or []) if nd.get("tid") is not None]

            # 1) fabriquer un "best" local, compatible buildEdgeChoiceEptsFromBest
            best_local = (score, me, te)

            res = buildEdgeChoiceEptsFromBest(
                best_local,
                world=world,
                mob_idx=mob_idx,
                tgt_idx=tgt_idx,
                mob_tids=mob_tids,
                tgt_tids=tgt_tids,
                last_drawn=self._last_drawn,
                eps_world=EPS_WORLD,
                mATmpId=mAId,
                tATmpId=tAId,
                debug=False,
            )
            if not res:
                return False

            epts, _meta = res
            attachments = epts.createTopologyAttachments(world=world, debug=False)
            if not attachments:
                return False

            return bool(world.simulateOverlapTopologique(srcGroup, dstGroup, attachments, debug=self.debug_skip_overlap_highlight))

        m_oriented = [(a, b) if _almost_eq(a, vm) else (b, a) for (a, b) in (m_inc_raw or [])]
        t_oriented = [(a, b) if _almost_eq(a, vt) else (b, a) for (a, b) in (t_inc_raw or [])]
        for me in m_oriented:
            azm = _azim(*me)
            for te in t_oriented:
                azt = _azim(*te)
                score = _ang_dist(azm, azt)
                # --- Anti-chevauchement (shrink-only) remis en service ---
                if not self.debug_skip_overlap_highlight:
                    S_m_pose = _place_union_on_pair(S_m_base, me[0], me[1], te[0], te[1])
                    if S_m_pose is not None:
                        if _overlap_topo_for_pair(score, me, te):
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

        # --- NOTE: _edge_choice est la "source de vérité" pour le release ---
        # Contrainte: d'autres appels attendent que choice[4] soit déballable en (m_a,m_b,t_a,t_b).
        self._edge_choice = None
        if best:
            mob_tids = [nd.get("tid") for nd in (self._group_nodes(gid_m) or []) if nd.get("tid") is not None]
            tgt_tids = [nd.get("tid") for nd in (self._group_nodes(gid_t) or []) if nd.get("tid") is not None]
            res = buildEdgeChoiceEptsFromBest(
                best,
                world=world,
                mob_idx=mob_idx,
                tgt_idx=tgt_idx,
                mob_tids=mob_tids,
                tgt_tids=tgt_tids,
                last_drawn=self._last_drawn,
                eps_world=EPS_WORLD,
                mATmpId=mAId,
                tATmpId=tAId,
                debug=bool(self._debug_snap_assist),
            )
            if res:
                epts, _meta = res
                self._edge_choice = (mob_idx, vkey_m, tgt_idx, tgt_vkey, epts)

        # Ajout des contours pour le debug (bleu)
        mob_outline = [(tuple(a), tuple(b)) for (a, b) in self._outline_for_item(mob_idx)]
        tgt_outline = [(tuple(a), tuple(b)) for (a, b) in self._outline_for_item(tgt_idx)]
        self._redraw_edge_highlights()

    def _clear_edge_highlights(self):
        """Efface du canvas les lignes d'aide déjà dessinées.
        N'efface PAS self._edge_choice ni self._edge_highlights (elles peuvent être réutilisées)."""
        if self._edge_highlight_ids:
            for _id in self._edge_highlight_ids:
                self.canvas.delete(_id)

        self._edge_highlight_ids = []

    def _redraw_edge_highlights(self):
        """Redessine les aides à partir de self._edge_highlights :
        - tout le périmètre (segments EXTÉRIEURS) des 2 groupes en BLEU (fin),
        - uniquement les segments INCIDENTS (candidats possibles) en BLEU **épais**,
        - meilleure paire (mobile & cible) en **ROUGE** par-dessus."""
        self._clear_edge_highlights()
        data = self._edge_highlights
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

        # --- Topologie (Core) : créer element + groupe singleton + pose monde au niveau groupe ---
        scen = self._get_active_scenario()

        # 1) Ajout de l'item dans le document
        self._last_drawn.append({
            "labels": tri["labels"],
            "pts": Pw,
            "id": tri.get("id"),
            "mirrored": False,
            })
        new_tid = len(self._last_drawn) - 1

        # Annoter l'objet Tk : elementId topo (si possible)
        scen = self._get_active_scenario()
        world = scen.topoWorld

        tri_rank = tri.get("id", None)
        # tri_rank attendu : 1..32 (rang importé)
        if tri_rank is not None:
            tri_rank_i = int(tri_rank)
        elif (tri_rank_i - 1 < 0) or (tri_rank_i - 1 >= len(self.df)):
            raise ValueError(f"Topology: triRank hors df: {tri_rank_i} (len(df)={len(self.df)})")

        # Construire l'élément topo uniquement si on a un tri_rank valide
        element_id = TopologyWorld.format_element_id(tri_rank_i)

        world.beginTopoTransaction()
        try:
            # ------------------------------------------------------------
            # Intrinsèque : on prend les longueurs + sens (orient) depuis self.df
            # (et non depuis les coords monde Pw).
            # Colonnes attendues dans df : len_OB, len_OL, len_BL, orient, B, L
            # O est fixé à "Bourges" dans ton modèle actuel (voir _triangle_from_index()).
            # ------------------------------------------------------------

            row = self.df.iloc[tri_rank_i - 1]
            len_OB = float(row["len_OB"])
            len_OL = float(row["len_OL"])
            len_BL = float(row["len_BL"])
            orient = str(row.get("orient", "")).strip().upper()

            # --- Orientation source (définition triangle) ---
            # On l'enregistre côté Tk, et on en déduit le miroir de POSE (pas le miroir visuel Tk).
            # Convention: mirrored=False.. utilsé uniquement pour le flip 
            self._last_drawn[new_tid]["orient"] = orient
            self._last_drawn[new_tid]["mirrored"] = False

            # labels/types : O/B/L dans l’ordre topo.
            # (convention projet actuelle : O="Bourges", B=row["B"], L=row["L"])
            v_labels = ["Bourges", str(row["B"]), str(row["L"])]
            v_types = [TopologyNodeType.OUVERTURE, TopologyNodeType.BASE, TopologyNodeType.LUMIERE]

            # Longueurs d’arêtes dans l’ordre du cycle O->B, B->L, L->O
            edge_lengths_km = [len_OB, len_BL, len_OL]

            # element (coordonnées locales canonisées par le core)
            el = TopologyElement(
                element_id=element_id,
                name=f"Triangle {tri_rank_i:02d}",
                vertex_labels=v_labels,
                vertex_types=v_types,
                edge_lengths_km=edge_lengths_km,
                meta={"orient": orient},
            )

            # Ajouter au core (nouveau groupe singleton)
            core_gid = world.add_element_as_new_group(el)

            # Annotation Tk (pont UI↔Core)
            self._last_drawn[new_tid]["topoElementId"] = element_id
            self._last_drawn[new_tid]["topoGroupId"] = core_gid

        finally:
            world.commitTopoTransaction()

        # 2) Création d'un groupe singleton
        self._ensure_group_fields(self._last_drawn[new_tid])
        gid = self._new_group_id()
        self.groups[gid] = {
            "id": gid,
            "nodes": [{"tid": new_tid, "edge_in": None, "edge_out": None}],
            "bbox": None,
        }
        self._last_drawn[new_tid]["group_id"] = gid
        self._last_drawn[new_tid]["group_pos"] = 0
        self._recompute_group_bbox(gid)

        # 2bis) Synchroniser le group avec la topo
        self._sync_group_elements_pose_to_core(gid)

        # Lier group UI -> group Core 
        core_gid = self._last_drawn[new_tid].get("topoGroupId", None)
        self.groups[gid]["topoGroupId"] = core_gid

        # 3) UI
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Triangle déposé → Groupe #{gid} créé.")

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
        return self._last_drawn[idx].get("group_id", None)

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
                xs.append(float(P[k][0]))
                ys.append(float(P[k][1]))
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
            for k in ("O", "B", "L"):
                sx += float(P[k][0])
                sy += float(P[k][1])
                n += 1
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

        opp_vkey_from_edge = {"OB": "L", "BL": "O", "LO": "B"}
        ein = nd.get("edge_in")
        if ein and opp_vkey_from_edge.get(ein) == vkey and pos > 0:
            return (gid, pos, "in")   # lien avec le précédent
        eout = nd.get("edge_out")
        if eout and opp_vkey_from_edge.get(eout) == vkey and pos < len(nodes)-1:
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
                self._last_drawn[tid]["group_id"] = gid
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

        left_nodes = [dict(n) for n in nodes[:split_after_pos+1]]
        right_nodes = [dict(n) for n in nodes[split_after_pos+1:]]

        # Corriger les arêtes aux frontières : tête/queue
        if left_nodes:
            left_nodes[0]["edge_in"] = None
            left_nodes[-1]["edge_out"] = None
        if right_nodes:
            right_nodes[0]["edge_in"] = None
            right_nodes[-1]["edge_out"] = None

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
                # AUTO: rollback du transform global
                if self._sel.get("auto_geom") and self._sel.get("auto_state0") is not None:
                    self.auto_geom_state = dict(self._sel.get("auto_state0") or {})
                    self._autoRebuildWorldGeometryScenario(None)
                    self._sel = None
                    self.status.config(text="Rotation auto annulée (ESC).")
                    self._clear_nearest_line()
                    self._clear_edge_highlights()
                    return
                gid = self._sel.get("gid")
                orig = self._sel.get("orig_group_pts")
                if gid is not None and isinstance(orig, dict):
                    for tid, pts in orig.items():
                        if 0 <= tid < len(self._last_drawn):
                            self._last_drawn[tid]["pts"] = {k: np.array(pts[k].copy()) for k in ("O", "B", "L")}
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
                            self._last_drawn[tid]["pts"] = {k: np.array(pts[k].copy()) for k in ("O", "B", "L")}
                    self._autoRebuildWorldGeometryScenario(None)
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
                self._last_drawn[idx]["pts"] = {k: np.array(orig[k].copy()) for k in ("O", "B", "L")}
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

    def _point_in_tri_screen(self, x: float, y: float, a, b, c) -> bool:
        """True si (x,y) est dans le triangle écran (a,b,c) (inclut bord)."""
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        cx, cy = float(c[0]), float(c[1])

        # Produit vectoriel 2D (p1->p2) x (p1->p3)
        def cross(x1, y1, x2, y2, x3, y3):
            return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

        # Tests de même signe (tolérance légère pour les bords)
        eps = 1e-9
        c1 = cross(ax, ay, bx, by, x, y)
        c2 = cross(bx, by, cx, cy, x, y)
        c3 = cross(cx, cy, ax, ay, x, y)

        has_neg = (c1 < -eps) or (c2 < -eps) or (c3 < -eps)
        has_pos = (c1 > eps) or (c2 > eps) or (c3 > eps)
        return not (has_neg and has_pos)

    def _hit_test(self, x, y):
        """Retourne ('center'|'vertex'|None, idx, extra) selon la zone cliquée.
        - 'vertex' si clic dans un disque autour d'un sommet
        - 'center' si clic à l'intérieur du triangle (hors disques sommets)
        """
        if not self._last_drawn:
            return (None, None, None)
        tol2 = float(self._hit_px) ** 2
        center_tol2 = float(self._center_hit_px) ** 2
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

            # 2) intérieur du triangle (hit réel)
            Os = np.array(self._world_to_screen(P["O"]))
            Bs = np.array(self._world_to_screen(P["B"]))
            Ls = np.array(self._world_to_screen(P["L"]))
            if self._point_in_tri_screen(float(x), float(y), Os, Bs, Ls):
                return ("center", i, None)

            # 3) fallback : clic proche du centre (utile si triangle très petit/degenerate)
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
            existing = self.listbox.get(idx)
            # on parse l'id au début (2 chiffres)
            ex_id = int(str(existing)[:2])
            if ex_id == int(tri_id):
                break

        else:
            # Trouver la position triée par id
            insert_at = self.listbox.size()
            for idx in range(self.listbox.size()):
                ex = self.listbox.get(idx)
                ex_id = int(str(ex)[:2])
                if int(tri_id) < ex_id:
                    insert_at = idx
                    break

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
            self._ctx_clear_chemin_context()
            self._update_compass_ctx_menu_and_dico_state()
            self._ctx_menu_compass.tk_popup(event.x_root, event.y_root)
            self._ctx_menu_compass.grab_release()
            return

        mode, idx, extra = self._hit_test(event.x, event.y)
        if idx is None:
            self._ctx_clear_chemin_context()
            return
        # On ne propose Supprimer que si on est sur un triangle
        if mode in ("center", "vertex"):
            self._ctx_target_idx = idx
            self._ctx_last_rclick = (event.x, event.y)
            self._ctx_nearest_vertex_key = self._ctx_compute_nearest_vertex_key(idx, event.x, event.y)
            try:
                self._ctx_capture_chemin_context(idx, event.x, event.y)
            except (ValueError, RuntimeError):
                self._ctx_clear_chemin_context()

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

    def _ctx_compute_nearest_vertex_key(self, tri_idx: int, sx: float, sy: float) -> str:
        """Retourne 'O'/'B'/'L' du sommet le plus proche du point écran (sx,sy)."""
        if tri_idx is None or not (0 <= int(tri_idx) < len(self._last_drawn)):
            return "L"
        tri = self._last_drawn[int(tri_idx)]
        pts = tri.get("_pick_pts") or {}
        best_k = None
        best_d2 = None
        for k in ("O", "B", "L"):
            p = pts.get(k)
            if not p:
                continue
            dx = float(sx) - float(p[0])
            dy = float(sy) - float(p[1])
            d2 = dx*dx + dy*dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_k = k
        return best_k or "L"

    def _ctx_clear_chemin_context(self) -> None:
        """Réinitialise le contexte minimal de création de chemin."""
        self.ctxGroupId = None
        self.ctxStartNodeId = None

    def _ctx_get_ui_group_from_triangle_index(self, tri_idx: int) -> Optional[int]:
        """Retourne le groupId UI qui contient le triangle tri_idx."""
        idx = int(tri_idx)
        for gid, grp in (self.groups or {}).items():
            for nd in grp.get("nodes", []):
                try:
                    tid = int(nd.get("tid", -1))
                except (TypeError, ValueError):
                    continue
                if tid == idx:
                    return gid
        return None

    def _ctx_capture_chemin_context(self, tri_idx: int, sx: float, sy: float) -> None:
        """
        Mémorise ctxGroupId + ctxStartNodeId depuis le clic droit.

        Règle V3: contexte minimal seulement (groupId Core + startNodeId DSU),
        sans lecture des structures boundary internes.
        """
        scen = self._get_active_scenario()
        if scen is None or scen.topoWorld is None:
            raise RuntimeError("scenario topologique introuvable")
        world = scen.topoWorld

        ui_gid = self._ctx_get_ui_group_from_triangle_index(int(tri_idx))
        if ui_gid is None:
            raise ValueError("groupe UI introuvable")
        g_ui = self.groups.get(ui_gid, {})
        core_gid = g_ui.get("topoGroupId", None)
        if not core_gid:
            raise ValueError(f"topoGroupId manquant pour le groupe UI {ui_gid}")
        gid = str(world.find_group(str(core_gid)))

        world.computeBoundary(gid)
        segments = world.getBoundarySegments(gid)
        if not segments:
            raise ValueError("frontière vide")
        boundary_nodes = sorted(
            {str(seg.fromNodeId) for seg in segments}.union({str(seg.toNodeId) for seg in segments})
        )
        if not boundary_nodes:
            raise ValueError("aucun noeud de frontière")

        wx, wy = self._screen_to_world(float(sx), float(sy))
        best_node = None
        best_d2 = None
        for nid in boundary_nodes:
            px, py = world.getConceptNodeWorldXY(str(nid), gid)
            dx = float(px) - float(wx)
            dy = float(py) - float(wy)
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_node = str(world.find_node(str(nid)))
        if best_node is None:
            raise RuntimeError("impossible de résoudre le noeud de départ")

        self.ctxGroupId = gid
        self.ctxStartNodeId = best_node

    def _ctx_CreerChemin(self) -> None:
        """Crée un chemin Core depuis le contexte du clic droit."""
        gid = self.ctxGroupId
        startNodeId = self.ctxStartNodeId
        if not gid or not startNodeId:
            messagebox.showerror("Créer un chemin", "Création du chemin impossible : contexte invalide.")
            return
        scen = self._get_active_scenario()
        world = scen.topoWorld

        boundaryOrientation = world.getBoundaryOrientation(gid)
        orientationUser = str(boundaryOrientation)

        if bool(world.topologyChemins.isDefined):
            if not messagebox.askokcancel("Créer un chemin", "Un chemin existe déjà. Remplacer ?"):
                return

        world.topologyChemins.creerDepuisGroupe(gid, startNodeId, orientationUser)
        self.refreshCheminTreeView()

    def _is_point_in_clock(self, sx: float, sy: float) -> bool:
        """True si (sx,sy) est dans le disque du compas (coords canvas)."""
        if not self.show_clock_overlay or not self.show_clock_overlay.get():
            return False
        cx = float(self._clock_cx)
        cy = float(self._clock_cy)
        R = float(self._clock_R)
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

        if not self.canvas:
            return
        if not self.show_clock_overlay or not self.show_clock_overlay.get():
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

        if not self.canvas:
            return
        if not self.show_clock_overlay or not self.show_clock_overlay.get():
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
        if not self.show_clock_overlay or not self.show_clock_overlay.get():
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
        if not self.dicoSheet:
            messagebox.showinfo("Filtrer le dictionnaire", "Le dictionnaire n'est pas affiché.")
            return
        ref = self._clock_arc_last_angle_deg
        if ref is None:
            messagebox.showinfo("Filtrer le dictionnaire", "Aucun arc n'a été mesuré.\n\nMesure d'abord un arc d'angle sur le compas.")
            return
        self._dico_filter_active = True
        self._dico_filter_ref_angle_deg = float(ref)
        self._dico_apply_filter_styles()
        self._update_compass_ctx_menu_and_dico_state()
        self.status.config(text=f"Dico filtré (angle ref={float(ref):0.0f}°, tol=±{float(self._dico_filter_tolerance_deg):0.0f}°)")

    def _update_compass_ctx_menu_and_dico_state(self):
        """Synchronise le menu compas et l'état de filtrage du dico selon la dispo de l'arc.

        IMPORTANT:
        - Le dico doit rester sélectionnable même s'il n'y a pas d'arc mesuré.
        - Seule l'action "Filtrer le dictionnaire…" dépend de l'existence d'un arc.
        """
        arc_ok = bool(self._clock_arc_is_available())

        menu = self._ctx_menu_compass
        idx = self._ctx_compass_idx_filter_dico
        if menu is not None and idx is not None:
            menu.entryconfig(idx, state=(tk.NORMAL if arc_ok else tk.DISABLED))

        # Activer/désactiver "Annuler le filtrage" selon l'état courant
        idx_cancel = self._ctx_compass_idx_cancel_dico_filter
        if menu is not None and idx_cancel is not None:
            menu.entryconfig(idx_cancel, state=(tk.NORMAL if bool(self._dico_filter_active) else tk.DISABLED))

        # Le dico reste sélectionnable dans tous les cas.
        self._dico_set_selection_enabled(True)

        # Si on perd l'arc alors qu'un filtrage était actif, on annule le filtrage.
        # IMPORTANT: ne pas appeler _simulation_cancel_dictionary_filter() si aucun filtrage n'est actif,
        # sinon recursion infinie (cancel -> update -> cancel -> ...).
        if (not arc_ok) and (bool(self._dico_filter_active) or (self._dico_filter_ref_angle_deg is not None)):
            if hasattr(self, "_simulation_cancel_dictionary_filter"):
                self._simulation_cancel_dictionary_filter()

    def _azimuth_world_deg(self, a, b) -> float:
        """Azimut absolu en degrés (0°=Nord, 90°=Est) entre 2 points monde."""
        import math
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dx, dy = (bx - ax), (by - ay)
        # dx vers Est, dy vers Nord (monde Lambert-like). Convertir en azimut.
        ang = math.degrees(math.atan2(dx, dy)) % 360.0
        return float(ang)

    def _clock_point_on_circle(self, az_deg: float, radius: float):
        """Point écran (sx,sy) à un azimut donné autour du centre du compas."""
        cx = float(self._clock_cx)
        cy = float(self._clock_cy)
        a = math.radians(float(az_deg) % 360.0)
        sx = cx + float(radius) * math.sin(a)
        sy = cy - float(radius) * math.cos(a)
        return (sx, sy)

    def _clock_apply_optional_snap(self, sx: int, sy: int, *, enable_snap: bool) -> Tuple[int, int]:
        """Applique le snap sur noeud si activé (et si CTRL n'est pas pressé)."""
        sx2, sy2 = int(sx), int(sy)
        if not enable_snap:
            return sx2, sy2
        if self._ctrl_down:
            self._clock_clear_snap_target()
            return sx2, sy2

        self._clock_update_snap_target(float(sx2), float(sy2))
        tgt = self._clock_snap_target
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
        self.status.config(text=f"Azimut mesuré : {az_rel:0.0f}° (ref={float(self._clock_ref_azimuth_deg)%360.0:0.0f}°, abs={az_abs:0.0f}°)")

    def _clock_measure_cancel(self, silent: bool = False):
        if not self._clock_measure_active:
            return
        self._clock_measure_active = False

        if self._clock_measure_line_id is not None:
            self.canvas.delete(self._clock_measure_line_id)

        if self._clock_measure_text_id is not None:
            self.canvas.delete(self._clock_measure_text_id)

        self._clock_measure_line_id = None
        self._clock_measure_text_id = None
        self._clock_measure_last = None
        self._clock_clear_snap_target()

        if not silent:
            self.status.config(text="Mesure d'azimut annulée.")

    def _clock_compute_azimuth_deg(self, sx: float, sy: float) -> float:
        """Azimut (degrés) depuis le centre du compas vers (sx,sy). 0°=Nord, sens horaire."""
        cx = float(self._clock_cx)
        cy = float(self._clock_cy)
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
        if self._ctx_last_rclick:
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
        n_nodes = len(g.get("nodes", []))
        if n_nodes >= 2:
            confirm = messagebox.askyesno(
                "Supprimer le groupe",
                "Voulez-vous supprimer le groupe ?"
            )
            if not confirm:
                return

        # 0) TOPO : capturer les elementId Core AVANT toute modif de _last_drawn
        scen = self._get_active_scenario()
        world = scen.topoWorld
        removed_element_ids = []

        # tids du groupe (dans l'ordre inverse, comme le reste du code)
        _tids = sorted({nd["tid"] for nd in g["nodes"] if "tid" in nd}, reverse=True)
        for tid in _tids:
            if 0 <= tid < len(self._last_drawn):
                eid = self._last_drawn[tid].get("topoElementId")
                if eid:
                    removed_element_ids.append(str(eid))

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

        # 2bis) TOPO : supprimer les éléments Core + purge des attaches + rebuild
        if world is not None and removed_element_ids:
            world.removeElementsAndRebuild(list(removed_element_ids))

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
        self.status.config(text=f"Groupe supprimé (gid={gid}, {len(removed_tids)} triangle(s)).")


    def _ctx_rotate_selected(self):
        """Passe en mode rotation autour du barycentre pour le triangle ciblé."""
        idx = self._ctx_target_idx
        self._ctx_target_idx = None
        if idx is None or not (0 <= idx < len(self._last_drawn)):
            return
        # le triangle fait toujours partie d'un groupe : PIVOTER LE GROUPE
        gid = self._get_group_of_triangle(idx)

        # AUTO: pivot = origine globale (0,0) commune ; MANUAL: barycentre groupe
        auto_geom = self._is_active_auto_scenario()
        if auto_geom:
            if self.auto_geom_state is None:
                self._autoEnsureLocalGeometry(self._get_active_scenario())
            if self.auto_geom_state is None:
                return
            pivot = (float(self.auto_geom_state.get("ox", 0.0)), float(self.auto_geom_state.get("oy", 0.0)))
        else:
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
            "auto_geom": bool(auto_geom),
            "auto_theta0": float(self.auto_geom_state.get("thetaDeg", 0.0)) if auto_geom and self.auto_geom_state else 0.0,
            "auto_state0": dict(self.auto_geom_state) if auto_geom and self.auto_geom_state else None,
        }
        self.status.config(text=f"Mode pivoter GROUPE #{gid} : bouge la souris pour tourner, clic gauche pour valider, ESC pour annuler.")
        return

 
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

        scen = self._get_active_scenario()
        # --- CAS AUTO : rotation globale partagée ---
        if scen.source_type == "auto":
            if self.auto_geom_state is None:
                self._autoEnsureLocalGeometry(scen)
                return

            theta0 = float(self.auto_geom_state.get("thetaDeg", 0.0))
            self.auto_geom_state["thetaDeg"] = float(theta0 + math.degrees(dtheta))

            self._autoRebuildWorldGeometry(redraw=False)   # rebuild monde pour TOUS les autos
            self._autoSyncAllTopoPoses()                   # sync topoWorld pour TOUS les autos

            self._redraw_from(self._last_drawn)
            self.status.config(text=f"Orientation appliquée : AUTO — {status_label} au Nord (0°).")
            return
        
        # --- CAS MANUEL : rotation UI puis sync core ---    
        # Déterminer le groupe (si présent)
        gid = self._get_group_of_triangle(idx)




        c, s = math.cos(dtheta), math.sin(dtheta)
        R = np.array([[c, -s], [s, c]], dtype=float)

        # Pivot = barycentre du triangle cliqué (cohérent avec "Pivoter")
        pivot = self._tri_centroid(P_ref)
        pivot = np.array(pivot, dtype=float)

        def rot_point(pt):
            pt = np.array(pt, dtype=float)
            return (R @ (pt - pivot)) + pivot

        # Appliquer à tout le groupe 
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
        
        # On est en manuel, on alimente la topo Core
        self._sync_group_elements_pose_to_core(gid)
        
        # On raffraichit l'affichage courant
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Orientation appliquée : GROUPE — {status_label} au Nord (0°).")


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
        clicked_tid = int(self._last_drawn[idx].get("id"))


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

        # Contrat: groups/nodes doivent fournir edge_in/edge_out (normalisation centralisée)
        self._normalize_all_groups(scen)

        # prendre la chaîne la plus longue (design actuel = un seul groupe)
        best_nodes = None
        for g in (groups or {}).values():
            nodes = (g or {}).get("nodes") or []
            if best_nodes is None or len(nodes) > len(best_nodes):
                best_nodes = nodes
        nodes = best_nodes or []
        if len(nodes) < upto_index + 1:
            return None

        steps = []
        for i in range(upto_index):
            a = nodes[i] or {}
            b = nodes[i + 1] or {}

            eout = a.get("edge_out")
            ein = b.get("edge_in")
            if not eout or not ein:
                raise RuntimeError(
                    f"scenario_prefix_edge_steps: edge manquant i={i} "
                    f"edge_out={eout!r} edge_in={ein!r}"
                )

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
        if self._is_active_auto_scenario():
            self.status.config(text="Inversion désactivée pour les scénarios automatiques.")
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

        # Alimente la topo Core (poses éléments) depuis l’état graphique legacy (après flip)
        self._sync_group_elements_pose_to_core(gid)

        # BBox groupe puis redraw
        self._recompute_group_bbox(gid)
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Inversion appliquée au groupe #{gid}.")

    def _move_group_world(self, gid, dx_w, dy_w):
        """Translate rigide de tout le groupe en coordonnées monde."""
        # AUTO: on déplace le référentiel global (commun à tous les scénarios auto)
        if self._is_active_auto_scenario():
            if self.auto_geom_state is None:
                self._autoEnsureLocalGeometry(self._get_active_scenario())
                return

            self.auto_geom_state["ox"] = float(self.auto_geom_state.get("ox", 0.0) + float(dx_w))
            self.auto_geom_state["oy"] = float(self.auto_geom_state.get("oy", 0.0) + float(dy_w))
            self._autoRebuildWorldGeometryScenario(None)
            return

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
        # Sync immédiate : Core = vérité persistée (pose monde du groupe)
        self._sync_group_elements_pose_to_core(gid)
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
        self._mouse_world_prev = self._screen_to_world(event.x, event.y)


        # Calibration fond (3 points) : intercepte le clic gauche
        if self._bg_calib_active:
            return self._bg_calibrate_handle_click(event)

        # Horloge : démarrer drag si clic dans le disque (marge 10px)
        if self._is_in_clock(event.x, event.y):
            # ne pas intercepter si un drag de triangle est en cours
            if not self._drag:
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
                self.canvas.configure(cursor="sizing")
                return "break"
            # sinon, en mode redimensionnement : clic maintenu = déplacement du fond
            self._bg_start_move(event.x, event.y)
            self.canvas.configure(cursor="fleur")

            return "break"

        # Validation d'une rotation en cours : le clic gauche sert à COMMIT, pas à re-sélectionner.
        if isinstance(self._sel, dict) and self._sel.get("mode") == "rotate_group":
            if self._sel.get("auto_geom"):
                self._autoRebuildWorldGeometry(redraw=True)
                self._autoSyncAllTopoPoses()
                # Persistance "Même dernier run" (carte, ordre)
                if self.auto_geom_state is not None:
                    self._simulationPersistCurrentAutoPlacement(save=True)
            else:
                # Manuel: sync classique du groupe vers le core
                gid_sync = self._sel.get("gid")
                if gid_sync is not None:
                    self._sync_group_elements_pose_to_core(int(gid_sync))


            self._sel = None
            self._reset_assist()
            self._redraw_from(self._last_drawn)
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
                    self.status.config(text=f"CTRL sans lien : déplacement du groupe #{gid0} par sommet {vkey}.")

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

    def _on_canvas_left_move(self, event):
        # Horloge : drag en cours -> on déplace le centre et on redessine l’overlay
        if self._clock_dragging:
            self._clock_cx = event.x - self._clock_drag_dx
            self._clock_cy = event.y - self._clock_drag_dy
            # Cible snap (sommet le plus proche du CENTRE du compas)
            self._clock_update_snap_target(self._clock_cx, self._clock_cy)
            self._redraw_overlay_only()
            return "break"

        # Mode déplacement fond d'écran (mode resize actif, clic maintenu hors poignée)
        if self._bg_moving:
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


    def _on_canvas_left_up(self, event):
        # Horloge : fin de drag
        if self._clock_dragging:
            # On capture la cible de snap *avant* de la nettoyer pour pouvoir déclencher
            # une éventuelle mesure d'arc automatique.
            snap_tgt = self._clock_snap_target

            # Si CTRL au relâché : on sort du mode sans "snap" (le compas reste où il est)
            if self._ctrl_down:
                wx, wy = self._screen_to_world(self._clock_cx, self._clock_cy)
                self._clock_anchor_world = np.array([wx, wy], dtype=float)

            else:
                # cible snap calculée pendant le drag (capturée au début via snap_tgt)
                tgt = snap_tgt
                if isinstance(tgt, dict) and tgt.get("world") is not None:
                    self._clock_anchor_world = np.array(tgt["world"], dtype=float)
                    sx, sy = self._world_to_screen(tgt["world"])
                    self._clock_cx, self._clock_cy = float(sx), float(sy)
                else:
                    # pas de target : ancrer la position courante en monde
                    wx, wy = self._screen_to_world(self._clock_cx, self._clock_cy)
                    self._clock_anchor_world = np.array([wx, wy], dtype=float)

            self._clock_dragging = False
            self.canvas.configure(cursor="")
            self._clock_clear_snap_target()
            # Spéc : si on déplace le compas et qu'il s'accroche à un noeud,
            # on tente une mesure d'arc automatique (EXT). Sinon on reset.
            measured = False
            if (not self._ctrl_down) and isinstance(snap_tgt, dict):
                measured = bool(self._clock_arc_auto_from_snap_target(snap_tgt, False))
            if not measured:
                self._clock_arc_clear_last()
            self._redraw_overlay_only()
            return "break"

        if self._bg_resizing:
            self._bg_resizing = None
            self.canvas.configure(cursor="")
            self._persistBackgroundConfig()
            self._bg_update_scale_status()
            self._redraw_from(self._last_drawn)
            return "break"

        if self._bg_moving:
            self._bg_moving = None
            self.canvas.configure(cursor="")
            self._persistBackgroundConfig()
            self._redraw_from(self._last_drawn)
            return "break"

        # Pan au clic gauche : fin du pan même si aucun triangle n'est sélectionné
        if self._pan_anchor is not None and not self._sel and not self._drag:
            self._on_pan_end(event)
            return

        """Fin du drag au clic gauche : dépôt de triangle (drag liste) OU fin d'une édition."""

        # 0) Dépôt d'un triangle glissé depuis la liste
        if self._drag:
            # Si le fantôme existe, on le supprime
            if self._drag_preview_id is not None:
                self.canvas.delete(self._drag_preview_id)
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
        if mode == "move_group":
            # Collage du GROUPE quand on l'a déplacé PAR SOMMET (ancre=vertex)
            # et que l'aide de collage était active (pas en déconnexion).
            anchor = self._sel.get("anchor")
            suppress = self._sel.get("suppress_assist")

            # On nettoie l'aide visuelle dans tous les cas
            self._clear_edge_highlights()

            # Si l'aide était active et qu'on a bien un choix d'arête,
            # on applique la même géométrie que pour un triangle seul,
            # mais à TOUS les triangles du groupe.
            choice = self._edge_choice
            self._dbgSnap(
                f"[snap] release(move_group) suppress_assist={self._sel.get('suppress_assist')} choice={'OK' if choice else 'None'}"
            )
            if (not suppress
                and (not self._ctrl_down)
                and anchor
                and anchor.get("type") == "vertex"
                and choice
                and choice[0] == anchor.get("tid")
                and choice[1] == anchor.get("vkey")):

                # Déballage du choix d'arête: (mob_idx,vkey_m,tgt_idx,vkey_t,epts)
                # epts est un objet séquence (mA,mBEdgeVertex,tA,tBEdgeVertex) enrichi pendant l'assist
                (_, _, idx_t, vkey_t, epts) = choice
                (m_a, m_b_edge_vertex, t_a, t_b_edge_vertex) = epts

                R, T = epts.computeRigidTransform()
                # Appliquer p' = R @ p + T à tous
                gid = self._sel.get("gid")
                g = self.groups.get(gid)
                if g:
                    for node in g["nodes"]:
                        tid = node.get("tid")
                        if 0 <= tid < len(self._last_drawn):
                            P = self._last_drawn[tid]["pts"]
                            for k in ("O", "B", "L"):
                                p = np.array(P[k], dtype=float)
                                p_fin = (R @ p) + T
                                P[k][0] = float(p_fin[0])
                                P[k][1] = float(p_fin[1])

                # ------------------------------------------------------------
                # Topologie Core (intention) : edge-edge OU vertex-edge
                # -> décision déjà prise pendant l'assist (epts.kind / epts.tRaw / labels / edges)
                # -> pose Core = sync depuis Tk (pas de compute_pose_* ici)
                # ------------------------------------------------------------
                new_core_gid = None
                scen = self._get_active_scenario()
                world = scen.topoWorld

                tgt_gid = self._last_drawn[idx_t].get("group_id", None)
                mob_gid = self._sel.get("gid")
                if world is not None and mob_gid is not None and tgt_gid is not None and tgt_gid != mob_gid:
                    tri_m = self._last_drawn[int(anchor.get("tid"))]
                    tri_t = self._last_drawn[int(idx_t)]
                    element_id_m = tri_m.get("topoElementId", None)
                    element_id_t = tri_t.get("topoElementId", None)

                    if element_id_m and element_id_t:
                        # La pose Core doit refléter ce que Tk a effectivement appliqué
                        self._sync_group_elements_pose_to_core(mob_gid)

                        kind = str(epts.kind).strip().lower()
                        if kind not in ("edge-edge", "vertex-edge"):
                            raise RuntimeError(f"kind inattendu: {kind}")

                        attachments_to_apply = epts.createTopologyAttachments(
                            world=world,
                            debug=bool(self._debug_snap_assist),
                        )
                        if attachments_to_apply:
                            world.beginTopoTransaction()
                            new_core_gid = world.apply_attachments(attachments_to_apply)
                            world.commitTopoTransaction()

                        if new_core_gid:
                            self.groups[mob_gid]["topoGroupId"] = new_core_gid

                # ====== FUSION DE GROUPE APRÈS COLLAGE (groupe ↔ groupe uniquement) ======
                tgt_gid = self._last_drawn[idx_t].get("group_id", None)
                mob_gid = gid
                if mob_gid is not None and tgt_gid is not None:
                    if tgt_gid != mob_gid:
                        # Fusionner tgt_gid → mob_gid AVEC LIEN EXPLICITE
                        g_src = self.groups.get(mob_gid)
                        g_tgt = self.groups.get(tgt_gid)
                        anchor = self._sel.get("anchor")  # {"type":"vertex","tid":...,"vkey":...}
                        choice = self._edge_choice  # (... idx_t, vkey_t, ...)
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
                                    x, y = float(pt[0]), float(pt[1])

                                    for kk in ("O","B","L"):
                                        if abs(float(P[kk][0]) - x) <= eps and abs(float(P[kk][1]) - y) <= eps:
                                            return kk
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
                            del self.groups[tgt_gid]
                            for i, nd in enumerate(nodes_src):
                                tid2 = nd["tid"]
                                if 0 <= tid2 < len(self._last_drawn):
                                    self._last_drawn[tid2]["group_id"]  = mob_gid
                                    self._last_drawn[tid2]["group_pos"] = i
                                    if new_core_gid:
                                        self._last_drawn[tid2]["topoGroupId"] = new_core_gid
                            if new_core_gid:
                                self.groups[mob_gid]["topoGroupId"] = new_core_gid
                            self._recompute_group_bbox(mob_gid)
                            self.status.config(text=f"Groupes fusionnés avec lien: #{tgt_gid} → #{mob_gid}.")
                    else:
                        # cible déjà dans le même groupe → rien à faire côté structure
                        self.status.config(text=f"Groupe collé (même groupe #{mob_gid}).")
                # ====== /FUSION ======

            # Sync Core : pose par élément (pas de pose groupe)
            gid_sync = self._sel.get("gid") if isinstance(self._sel, dict) else None
            if gid_sync is not None:
                self._sync_group_elements_pose_to_core(int(gid_sync))

            # Fin (pas de snap : simple dépôt à la dernière position)
            # AUTO: persister position/rotation globale (carte, ordre)
            if isinstance(self._sel, dict) and self.auto_geom_state is not None:
                self._autoRebuildWorldGeometry(redraw=False)
                self._autoSyncAllTopoPoses()
                self._simulationPersistCurrentAutoPlacement(save=True)
            self._sel = None
            self._reset_assist()
            self._redraw_from(self._last_drawn)
            return

        # Autres modes : on nettoie juste 
        self._sel = None
        self._reset_assist()
        self._redraw_from(self._last_drawn)
        return

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
        self._offset_anchor = np.array(self.offset, dtype=float).copy()

    def _on_pan_move(self, event):
        if self._pan_anchor is None:
            return
        d = np.array([event.x, event.y], dtype=float) - self._pan_anchor
        self.offset = self._offset_anchor + d
        self._redraw_from(self._last_drawn)

    def _on_pan_end(self, event):
        self._pan_anchor = None
        # après pan -> invalider cache pick (coords écran changent)
        self._invalidate_pick_cache()

    # ---------- Export PDF de l'affichage (A4) ----------
    class _PdfCanvasAdapter:
        """Adaptateur minimal Tk Canvas -> ReportLab.

        Les coordonnées reçues sont en "pixels" avec origine en haut-gauche (comme Tk).
        Elles sont mappées dans une zone A4 en points (ReportLab, origine bas-gauche).
        """

        def __init__(self, rl_canvas, page_w_pt: float, page_h_pt: float,
                     margin_left_pt: float, margin_bottom_pt: float,
                     scale_pt_per_px: float, virtual_h_px: float):
            self._c = rl_canvas
            self._pw = float(page_w_pt)
            self._ph = float(page_h_pt)
            self._ml = float(margin_left_pt)
            self._mb = float(margin_bottom_pt)
            self._s = float(scale_pt_per_px)
            self._vh = float(virtual_h_px)
            self._id = 1

        def _next_id(self):
            i = self._id
            self._id += 1
            return i

        def _xy(self, x, y):
            # Tk: (0,0) en haut-gauche ; RL: (0,0) en bas-gauche
            xp = self._ml + float(x) * self._s
            yp = self._mb + (self._vh - float(y)) * self._s
            return xp, yp

        def _dash(self, dash):
            if not dash:
                return None
            seq = [max(0.0, float(v) * self._s) for v in dash]
            return seq if seq else None


        def _set_stroke(self, color, width=1, dash=None):
            from reportlab.lib import colors
            from PIL import ImageColor
            if color is None or color == "":
                color = "#000000"
            if isinstance(color, str) and color.startswith("#"):
                col = colors.HexColor(color)
            else:
                r, g, b = ImageColor.getrgb(str(color))
                col = colors.Color(r/255.0, g/255.0, b/255.0)

            self._c.setStrokeColor(col)
            self._c.setLineWidth(max(0.1, float(width) * self._s))
            d = self._dash(dash)
            if d:
                self._c.setDash(d)
            else:
                self._c.setDash()

        def _set_fill(self, color):
            from reportlab.lib import colors
            from PIL import ImageColor
            if color is None or color == "":
                self._c.setFillColor(colors.transparent)
                return
            if isinstance(color, str) and color.startswith("#"):
                col = colors.HexColor(color)
            else:
                r, g, b = ImageColor.getrgb(str(color))
                col = colors.Color(r/255.0, g/255.0, b/255.0)

            self._c.setFillColor(col)

        # --- API Tk (subset) ---
        def delete(self, *args, **kwargs): return
        def tag_lower(self, *args, **kwargs): return
        def tag_raise(self, *args, **kwargs): return
        # alias Tk : Canvas.lift(...) fait la même chose que tag_raise(...)
        def coords(self, *args, **kwargs): return
        def itemconfig(self, *args, **kwargs): return

        def create_line(self, x1, y1, x2, y2, **kw):
            fill = kw.get("fill", "#000000")
            width = kw.get("width", 1)
            dash = kw.get("dash", None)
            self._set_stroke(fill, width=width, dash=dash)
            a = self._xy(x1, y1)
            b = self._xy(x2, y2)
            self._c.line(a[0], a[1], b[0], b[1])
            return self._next_id()

        def create_polygon(self, coords, **kw):
            fill = kw.get("fill", None)
            outline = kw.get("outline", None)
            width = kw.get("width", 1)

            pts = list(coords or [])
            if len(pts) < 6:
                return self._next_id()

            path = self._c.beginPath()
            x0, y0 = self._xy(pts[0], pts[1])
            path.moveTo(x0, y0)
            for i in range(2, len(pts), 2):
                xi, yi = self._xy(pts[i], pts[i+1])
                path.lineTo(xi, yi)
            path.close()

            do_fill = bool(fill) and str(fill) not in ("", "none")
            do_stroke = bool(outline) and str(outline) not in ("", "none")
            if do_fill:
                self._set_fill(fill)
            if do_stroke:
                self._set_stroke(outline, width=width)
            self._c.drawPath(path, fill=int(do_fill), stroke=int(do_stroke))
            return self._next_id()

        def create_rectangle(self, x1, y1, x2, y2, **kw):
            outline = kw.get("outline", None)
            fill = kw.get("fill", None)
            width = kw.get("width", 1)
            do_fill = bool(fill) and str(fill) not in ("", "none")
            do_stroke = bool(outline) and str(outline) not in ("", "none")
            if do_fill:
                self._set_fill(fill)
            if do_stroke:
                self._set_stroke(outline, width=width, dash=kw.get("dash"))
            xa, ya = self._xy(min(x1, x2), max(y1, y2))
            xb, yb = self._xy(max(x1, x2), min(y1, y2))
            self._c.rect(xa, yb, xb - xa, ya - yb, stroke=int(do_stroke), fill=int(do_fill))
            return self._next_id()

        def create_oval(self, x1, y1, x2, y2, **kw):
            outline = kw.get("outline", None)
            fill = kw.get("fill", None)
            width = kw.get("width", 1)
            do_fill = bool(fill) and str(fill) not in ("", "none")
            do_stroke = bool(outline) and str(outline) not in ("", "none")
            if do_fill:
                self._set_fill(fill)
            if do_stroke:
                self._set_stroke(outline, width=width, dash=kw.get("dash"))
            xa, ya = self._xy(min(x1, x2), max(y1, y2))
            xb, yb = self._xy(max(x1, x2), min(y1, y2))
            self._c.ellipse(xa, yb, xb, ya, stroke=int(do_stroke), fill=int(do_fill))
            return self._next_id()

        def create_arc(self, x1, y1, x2, y2, **kw):
            outline = kw.get("outline", "#000000")
            width = kw.get("width", 1)
            start = float(kw.get("start", 0.0))
            extent = float(kw.get("extent", 0.0))
            style = str(kw.get("style", "arc"))
            if style != "arc":
                return self._next_id()

            self._set_stroke(outline, width=width, dash=kw.get("dash"))
            xa, ya = self._xy(min(x1, x2), max(y1, y2))
            xb, yb = self._xy(max(x1, x2), min(y1, y2))
            self._c.arc(xa, yb, xb, ya, startAng=start, extent=extent)
            return self._next_id()

        def create_text(self, x, y, **kw):
            txt = str(kw.get("text", ""))
            fill = kw.get("fill", "#000000")
            anchor = str(kw.get("anchor", "center"))
            font = kw.get("font", ("Arial", 10))

            face = "Helvetica"
            size = 10
            style = ""

            if isinstance(font, tuple) and len(font) >= 2:
                size = int(font[1])
                if any(str(x).lower() == "bold" for x in font[2:]):
                    style = "-Bold"
            elif isinstance(font, str):
                parts = font.split()
                for p in parts:
                    if p.isdigit():
                        size = int(p)
                if any(p.lower() == "bold" for p in parts):
                    style = "-Bold"

            face = face + style

            self._set_fill(fill)
            self._c.setFont(face, max(4, float(size)))

            xp, yp = self._xy(x, y)
            yp_adj = yp - 0.35 * float(size)

            if anchor in ("center", "c"):
                self._c.drawCentredString(xp, yp_adj, txt)
            elif anchor in ("w", "west"):
                self._c.drawString(xp, yp_adj, txt)
            elif anchor in ("e", "east"):
                w = self._c.stringWidth(txt, face, max(4, float(size)))
                self._c.drawString(xp - w, yp_adj, txt)
            elif anchor in ("nw", "nwest"):
                self._c.drawString(xp, yp_adj + 0.35 * float(size), txt)
            else:
                self._c.drawCentredString(xp, yp_adj, txt)

            return self._next_id()

    def _export_view_pdf_dialog(self):
        """Boîte de dialogue pour exporter l'affichage courant en PDF A4."""
        from tkinter import filedialog
        scen_name = ""

        if self.scenarios and 0 <= int(self.active_scenario_index) < len(self.scenarios):
            scen_name = str(getattr(self.scenarios[self.active_scenario_index], "name", "") or "")

        tri_file = str(self.triangle_file.get()) if hasattr(self.triangle_file, "get") else ""


        def _safe(s: str) -> str:
            s = str(s or "").strip()
            s = re.sub(r"[^A-Za-z0-9._ -]+", "_", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        base = "affichage"
        if scen_name:
            base += "_" + _safe(scen_name)[:40]
        if tri_file:
            base += "_" + _safe(os.path.splitext(os.path.basename(tri_file))[0])[:30]
        base = base.strip("_ ") + ".pdf"

        path = filedialog.asksaveasfilename(
            title="Exporter en PDF",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=base,
        )
        if not path:
            return
        try:
            self._export_view_pdf(path)
            self.status.config(text=f"PDF généré : {path}")
            messagebox.showinfo("Export PDF", f"Export terminé avec succès.\n\nFichier :\n{path}")            
        except Exception as e:
            messagebox.showerror("Export PDF", f"Impossible de générer le PDF :\n{e}")

    def _export_view_pdf(self, path: str):
        """Exporte l'affichage courant (carte/triangles/compas selon visibilité) en PDF A4."""
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.lib import colors
        import copy

        cw = int(self.canvas.winfo_width() or 0)
        ch = int(self.canvas.winfo_height() or 0)
        if cw <= 2 or ch <= 2:
            self.update_idletasks()
            cw = int(self.canvas.winfo_width() or 0)
            ch = int(self.canvas.winfo_height() or 0)
        if cw <= 2 or ch <= 2:
            cw, ch = 800, 600

        xA, yTop = self._screen_to_world(0, 0)
        xB, yBot = self._screen_to_world(cw, ch)
        vx0 = min(xA, xB); vx1 = max(xA, xB)
        vy0 = min(yBot, yTop); vy1 = max(yBot, yTop)
        w0 = max(1e-9, vx1 - vx0)
        h0 = max(1e-9, vy1 - vy0)
        cx0 = 0.5 * (vx0 + vx1)
        cy0 = 0.5 * (vy0 + vy1)

        margin_lr = 10 * mm
        margin_bottom = 10 * mm
        margin_top = 10 * mm
        title_space = 12 * mm

        def _expand_bbox_to_aspect(w: float, h: float, target_aspect: float):
            cur = w / h
            if cur > target_aspect:
                return (w, w / target_aspect)
            return (h * target_aspect, h)

        def _choose_layout():
            candidates = []
            for is_land in (False, True):
                pw, ph = (landscape(A4) if is_land else A4)
                dw = pw - 2 * margin_lr
                dh = ph - margin_bottom - (margin_top + title_space)
                aspect = dw / max(1e-9, dh)
                w2, h2 = _expand_bbox_to_aspect(w0, h0, aspect)
                factor = (w2 * h2) / max(1e-12, (w0 * h0))
                candidates.append((factor, is_land, pw, ph, dw, dh, aspect, w2, h2))
            candidates.sort(key=lambda t: t[0])
            return candidates[0]

        _, is_land, page_w, page_h, draw_w, draw_h, target_aspect, w2, h2 = _choose_layout()

        vx0e = cx0 - 0.5 * w2
        vx1e = cx0 + 0.5 * w2
        vy0e = cy0 - 0.5 * h2
        vy1e = cy0 + 0.5 * h2

        Z = float(self.zoom)
        virt_w_px = w2 * Z
        virt_h_px = h2 * Z
        if virt_w_px <= 1 or virt_h_px <= 1:
            virt_w_px, virt_h_px = float(cw), float(ch)

        s = min(draw_w / max(1e-9, virt_w_px), draw_h / max(1e-9, virt_h_px))

        c = rl_canvas.Canvas(path, pagesize=(page_w, page_h))
        c.setFillColor(colors.white)
        c.rect(0, 0, page_w, page_h, stroke=0, fill=1)

        scen_name = ""
        if self.scenarios and 0 <= int(self.active_scenario_index) < len(self.scenarios):
            scen_name = str(getattr(self.scenarios[self.active_scenario_index], "name", "") or "")
        tri_name = str(self.triangle_file.get()) if hasattr(self.triangle_file, "get") else ""


        title_y = page_h - margin_top - 10
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin_lr, title_y, f"Scénario : {scen_name or '—'}")
        c.setFont("Helvetica", 10)
        c.drawString(margin_lr, title_y - 14, f"Triangles : {os.path.basename(tri_name) if tri_name else '—'}")

        origin_x_pt = margin_lr
        origin_y_pt = margin_bottom

        # --- Carte raster (si affichée) ---
        show_map = (self.show_map_layer is None) or bool(self.show_map_layer.get())
        op = int(float(self.map_opacity.get())) if hasattr(self, "map_opacity") else 100
        op = max(0, min(100, int(op)))
        if show_map and op > 0 and self._bg and self._bg_base_pil is not None and Image is not None:
            bx0 = float(self._bg.get("x0", 0.0)); by0 = float(self._bg.get("y0", 0.0))
            bw = float(self._bg.get("w", 0.0));  bh = float(self._bg.get("h", 0.0))
            bx1 = bx0 + bw; by1 = by0 + bh

            ix0 = max(vx0e, bx0); ix1 = min(vx1e, bx1)
            iy0 = max(vy0e, by0); iy1 = min(vy1e, by1)
            if ix0 < ix1 and iy0 < iy1 and bw > 1e-9 and bh > 1e-9:
                base = self._bg_base_pil
                baseW, baseH = base.size

                left  = int((ix0 - bx0) / bw * baseW)
                right = int((ix1 - bx0) / bw * baseW)
                upper = int((by1 - iy1) / bh * baseH)
                lower = int((by1 - iy0) / bh * baseH)

                left = max(0, min(baseW - 1, left))
                right = max(left + 1, min(baseW, right))
                upper = max(0, min(baseH - 1, upper))
                lower = max(upper + 1, min(baseH, lower))

                crop = base.crop((left, upper, right, lower))
                if crop.mode != "RGBA":
                    crop = crop.convert("RGBA")

                if op < 100:
                    r, g, b, a = crop.split()
                    a = a.point(lambda p: int(p * op / 100))
                    crop.putalpha(a)

                white = Image.new("RGBA", crop.size, (255, 255, 255, 255))
                white.paste(crop, (0, 0), crop)
                rgb = white.convert("RGB")

                x_px = (ix0 - vx0e) * Z
                y_px_top = (vy1e - iy1) * Z
                w_px = (ix1 - ix0) * Z
                h_px = (iy1 - iy0) * Z

                x_pt = origin_x_pt + x_px * s
                y_pt = origin_y_pt + (virt_h_px - (y_px_top + h_px)) * s
                w_pt = w_px * s
                h_pt = h_px * s

                c.drawImage(ImageReader(rgb), x_pt, y_pt, width=w_pt, height=h_pt, mask=None)

        adapter = self._PdfCanvasAdapter(
            c,
            page_w_pt=page_w,
            page_h_pt=page_h,
            margin_left_pt=origin_x_pt,
            margin_bottom_pt=origin_y_pt,
            scale_pt_per_px=s,
            virtual_h_px=virt_h_px,
        )

        old_canvas = self.canvas
        old_offset = np.array(self.offset, dtype=float).copy()
        old_zoom = float(self.zoom)
        old_nearest = self._nearest_line_id
        old_clock = {
            "_clock_cx": self._clock_cx,
            "_clock_cy": self._clock_cy,
            "_clock_R": self._clock_R,
            "_clock_anchor_world": copy.deepcopy(self._clock_anchor_world),
        }

        try:
            self.canvas = adapter
            self.zoom = old_zoom
            self.offset = np.array([(-vx0e * Z), (vy1e * Z)], dtype=float)
            self._nearest_line_id = None

            self._update_current_scenario_differences()

            showContoursMode = bool(
                self.show_only_group_contours is not None
                and self.show_only_group_contours.get()
            )
            onlyContours = False
            v = self.only_group_contours
            if v is not None and hasattr(v, "get"):
                onlyContours = bool(v.get())
            else:
                onlyContours = bool(self._only_group_contours)
            if showContoursMode:
                onlyContours = True

            if self.show_triangles_layer is None or self.show_triangles_layer.get():
                for i, t in enumerate(self._last_drawn or []):
                    labels = t.get("labels")
                    P = t.get("pts")
                    if not labels or not P:
                        continue
                    tri_id = t.get("id")
                    fill = "#ffd6d6" if i in self._comparison_diff_indices else None
                    self._draw_triangle_screen(
                        P,
                        labels=[f"O:{labels[0]}", f"B:{labels[1]}", f"L:{labels[2]}"],
                        tri_id=tri_id,
                        tri_mirrored=t.get("mirrored", False),
                        fill=fill,
                        diff_outline=bool(fill),
                        drawEdges=(not onlyContours),
                    )

            if showContoursMode:
                self._draw_group_outlines()
            else:
                if self._sel and self._sel.get("mode") == "vertex":
                    if self._edge_highlights:
                        self._redraw_edge_highlights()
                    idx = self._sel.get("idx"); vkey = self._sel.get("vkey")
                    if idx is not None and vkey and 0 <= int(idx) < len(self._last_drawn):
                        P = self._last_drawn[int(idx)]["pts"]
                        v_world = np.array(P[vkey], dtype=float)
                        self._update_nearest_line(v_world, exclude_idx=int(idx))

            self._draw_clock_overlay()

        finally:
            self.canvas = old_canvas
            self.offset = old_offset
            self.zoom = old_zoom
            self._nearest_line_id = old_nearest
            for k, v in old_clock.items():
                setattr(self, k, v)

        c.showPage()
        c.save()

# ---------- Entrée ----------
if __name__ == "__main__":
    app = TriangleViewerManual()
    app.mainloop()
