
import os
import datetime as _dt
import logging
import math
import sys
from math import atan2, pi
import numpy as np
import re
import copy
from typing import Optional, List, Dict, Tuple

from tkinter import filedialog, messagebox, simpledialog, colorchooser
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

# === Modules externalisés (découpage maintenable) ===
from src.assembleur_core import (
    ScenarioAssemblage,
    TopologyWorld, TopologyCheminTriplet, TopologyEdgeEdgeAttachment,
    TopologyConstraintGeometryError,
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
from src.assembleur_dictionary_panel import (
    CFG_KEY_DICO_EXCLURE_MOTS_CODES,
    DictionaryPanel,
)
from src.assembleur_scenario_map_controller import ScenarioMapController
from src.assembleur_compass_controller import CompassController
from src.assembleur_compass_state import (
    CompassState,
    azimuth_world_deg,
    clock_angle_diff_deg,
    clock_theoretical_ref_azimuth_deg,
)
from src.assembleur_background_map_layer import BackgroundMapLayer, format_scale
from src.assembleur_edgechoice import (
    buildManualAttachmentIntentFromBest,
    commitManualAttachment,
    previewManualAttachment,
)
from src.assembleur_beacon_runtime import BeaconWorldResolver
from src.assembleur_logging import configure_logging
from src.assembleur_topology_comparison import (
    build_topology_prefix_steps,
    differing_attachment_element_ids,
)
from src.canvas_objects_collection import CanvasObjectsCollection
from src.assembleur_projection import (
    buildLastDrawnFromTopology,
    getCoreTriangleWorldPoints,
    getManualProjectionElementIds,
)
from src.assembleur_catalogue_window import CatalogueWindow, CitySelectionDialog
from src.assembleur_hypothesis_window import ScenarioHypothesisDialog
from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_io import load_catalogue
from src.assembleur_catalogue_identity import ApplicationContext, load_project_dotenv
from src.assembleur_paths import ApplicationPaths
from src.assembleur_deformation import (
    commit_deformation_copy_on_write,
    simulate_deformation_session,
    simulate_occurrence_deformation,
)
from src.assembleur_deformation_ui import DeformationUiState
from src.assembleur_geometry_reference import GeometryReferenceResolver, ScenarioReference
from src.assembleur_deformation_window import (
    DeformationVertex,
    DeformationWindow,
    derive_assembly_view_rotation_deg,
)
from src.assembleur_tooltip import attach_tooltip
from src.assembleur_catalogue_map_assets import CatalogueMapAssetResolver, load_calibrated_catalogue_map
from src.assembleur_catalogue_geometric_layer_assets import CatalogueGeometricLayerAssetResolver
from src.assembleur_catalogue_book_assets import CatalogueBookAssetResolver
from src.assembleur_geometric_layer_io import load_geometric_layer_document
from src.assembleur_scenario_map import ScenarioMapState
from src.assembleur_scenario import (
    ScenarioHypothesis,
    ScenarioHypothesisChangePlan,
    analyze_hypothesis_change,
    apply_hypothesis_change_to_manual_scenario,
    create_default_scenario_hypothesis,
    materialize_triangle,
)
from src.assembleur_map_print import (
    AssembleurPrintBeacon,
    AssembleurPrintMap,
    AssembleurPrintSettings,
    AssembleurPrintSnapshot,
    AssembleurPrintTriangle,
    AssembleurPrintViewport,
    fit_initial_viewport,
)
from src.assembleur_map_print_dialog import AssembleurMapPrintDialog
from src.assembleur_chemins_export_dialog import CheminsExportDialog
from src.assembleur_chemin_edit_dialog import CheminEditDialog
from src.assembleur_decryptage_window import DecryptageEngineWindow
from src.assembleur_simulation_dialog import AutoOrientationReference, DialogSimulationAssembler


def get_anchor_beacon_candidates(catalogue: Catalogue):
    """Balises actives autorisées pour la création d'un GroupAnchor."""
    return tuple(
        beacon for beacon in catalogue.iter_beacons()
        if not beacon.archived and beacon.usable_as_anchor
    )


def get_geometric_reference_beacon_candidates(catalogue: Catalogue):
    """Balises actives disponibles comme repères géométriques."""
    return tuple(beacon for beacon in catalogue.iter_beacons() if not beacon.archived)


DEFORMATION_DRAG_REFRESH_MS = 40
EPS_WORLD = 1e-6
_LOGGER_NAME = "src.assembleur_tk" if __name__ == "__main__" else __name__
LOGGER = logging.getLogger(_LOGGER_NAME)
MIG_GEO_LOGGER = logging.getLogger(f"{_LOGGER_NAME}.mig_geo")


def resolve_catalogue_base_city_id_for_deformation_triangle(
    resolver: GeometryReferenceResolver,
    triangle_ref_id: str,
) -> str | None:
    """Résout la Base Catalogue d'un TRI ou STRI sans heuristique de secours."""
    triangle = resolver.resolve_triangle(triangle_ref_id)
    base_city = resolver.resolve_city(triangle.base_city_ref_id)
    return base_city.catalogue_source_city_id


def _load_application_catalogue(catalogue_path: str, id_provider) -> Catalogue:
    """Charge le catalogue existant, sans masquer un echec de contrat."""
    if not os.path.exists(catalogue_path):
        return Catalogue(id_provider=id_provider)
    try:
        return load_catalogue(catalogue_path, id_provider=id_provider)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Impossible de charger le catalogue {catalogue_path}: {exc}"
        ) from exc


def createDecryptor(decryptorId: str) -> DecryptorBase:
    """Instancie un decryptor à partir de son id (registre DECRYPTORS)."""
    cls = DECRYPTORS.get(str(decryptorId))
    if cls is None:
        # Fallback explicite
        return ClockDicoDecryptor()
    return cls()


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
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._deformation_state = DeformationUiState()
        instance._deformation_canvas_mode = "select"
        instance._deformation_drag_after_id = None
        instance._deformation_drag_pending_role = None
        instance._deformation_drag_pending_point = None
        instance._deformation_status_text = ""
        instance._deformation_geometric_layer_source_triangle_id = None
        instance.background_map_layer = BackgroundMapLayer(
            instance._world_to_screen,
            instance._screen_to_world,
            instance._on_background_map_geometry_changed,
        )
        return instance

    def __init__(self):
        super().__init__()

        load_project_dotenv()
        self.application_context = ApplicationContext.from_environment()
        self.paths = ApplicationPaths.from_runtime(
            catalogue_mode=self.application_context.mode,
        )
        self.iconbitmap(str(self.paths.app_icon_path))
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
        self._drag_preview_id = None   # compatibilité avec les anciens chemins mono
        self._drag_preview_ids = []    # un fantôme mono, deux pour un quadrilatère
        self._sel = None               # sélection sur canvas: {'mode': 'move'|'vertex', 'idx': int}
        self._hit_px = 12              # tolérance de hit (pixels) pour les sommets
        self._center_hit_px = 12       # même défaut que hit_px historique
        self._marker_px = 6            # rayon des marqueurs (cercles) dessinés aux sommets
        self._pan_anchor = None
        self._offset_anchor = None
        self._last_canvas_size = (0, 0)
        self._resize_redraw_after_id = None
        self.debug_geo_orient = os.environ.get(
            "ASSEMBLEUR_DEBUG_GEO_ORIENT", ""
        ) in ("1", "true", "True")

        # ElementID Core du triangle visé par clic droit (menu contextuel).
        # Un index UI ne vit que pendant le hit-test et n'est jamais persisté.
        self._ctx_target_element_id: Optional[str] = None
        self._ctx_last_rclick = None   # dernière position écran du clic droit (pour pivoter)
        self._ctx_nearest_vertex_key = None  # 'O'|'B'|'L' sommet le plus proche du clic droit
        self._ctx_compass_idx_clear_traits: int | None = None
        self._deformation_state = DeformationUiState()
        self._deformation_canvas_mode = "select"
        self._deformation_drag_after_id = None
        self._deformation_drag_pending_role = None
        self._deformation_drag_pending_point = None
        self._deformation_status_text = ""
        self._deformation_geometric_layer_source_triangle_id: str | None = None
        self._deformation_window: DeformationWindow | None = None
        self._deformation_map_cache_id: str | None = None
        self._deformation_map_cache = None
        self.ctxGroupId = None         # contexte chemin: groupId Core canonique (clic droit)
        self.ctxStartNodeId = None     # contexte chemin: startNodeId DSU canonique (clic droit)
        self._nearest_line_id = None   # trait d'aide "sommet le plus proche"
        self._edge_highlight_ids = []  # surlignage des 2 arêtes (mobile/cible)
        self._attachment_intent = None  # ManualAttachmentIntent produit par le drag
        self._attachment_preview = None  # ManualAttachmentPreview calculé sur clone Core
        self._edge_highlights = None   # données brutes des aides (candidates + best)
        # MIG-ANCHOR-003 : cible gagnante pendant un drag de groupe par sommet.
        # Sommet raccordable et balise sont arbitrés sur leur distance monde.
        self._group_drag_snap_candidate = None
        self._tooltip = None           # tk.Toplevel
        self._tooltip_label = None     # tk.Label

        # --- cache pick (écran) régénéré après load / zoom / pan ---
        self._pick_cache_valid = False
        # chaque item de self._last_drawn recevra:
        #   t["_pick_poly"] : liste de points écran [(x,y),...]
        #   t["_pick_pts"]  : dict {'O':(x,y), 'B':(x,y), 'L':(x,y)}

        # Association triangle -> mot du dictionnaire: { tri_id: {"word": str, "row": int, "col": int} }
        # distance écran supplémentaire pour l'ancrage du tooltip (px)
        self._tooltip_cushion_px = 14

        # Épaisseur de trait (px) — utilisée pour le test de chevauchement "shrink seul"
        self.stroke_px = 2

        # Le dico à créer
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
        self.show_balises_layer = tk.BooleanVar(value=False)
        self._beacon_navigation_index: int | None = None
        self._layerGuidesVisible: bool = True
        self._layerGuidesVisibleVar = tk.BooleanVar(value=True)
        # Opacité du layer "carte" (0..100). 100 = opaque, 0 = invisible.
        self.map_opacity = tk.IntVar(value=70)
        self._map_opacity_redraw_job = None
        self._guidesCurrentColorHex: str = "#0b3d91"
        self._guides_color_btn: tk.Button | None = None
        # Gestion du contour
        self.show_only_group_contours = tk.BooleanVar(value=False)
        self.only_group_contours = None
        self._only_group_contours = False

        # Recentrage automatique (Fit à l'écran) lors de la sélection d'un scénario
        self.auto_fit_scenario_select = tk.BooleanVar(value=False)

        # --- Fond SVG (coordonnées monde) ---
        self.bg_resize_mode = tk.BooleanVar(value=False)

        self._ctrl_down = False

        # === Scénarios d'assemblage ===
        # Liste de scénarios (1 scénario manuel actif + futurs scénarios auto).
        self.scenarios: List[ScenarioAssemblage] = []
        self.active_scenario_index: int = 0

        # Carte partagée pour les scénarios automatiques (snapshot au lancement de la simu)
        self.auto_map_state = None
        self.auto_view_state: dict | None = None

        # Rotation collective supplémentaire des scénarios automatiques ancrés.
        self.auto_rotation_state: dict | None = None  # {'thetaDeg': float}

        # Scénario de référence (pour comparaison des auto)
        self.ref_scenario_token: Optional[int] = None  # id(scen)
        self._comparison_diff_indices: set = set()

        # état IHM
        self.start_index = tk.IntVar(value=1)
        self.num_triangles = tk.IntVar(value=8)
        # Répertoires par défaut
        # Sous-répertoire dédié aux cartes (fond + fichiers de calibration)
        # Exports (artefacts diffables / validation)
        # Répertoire des icônes
        # === Config (persistance des paramètres) ===
        self.paths.ensure_user_data_directories()
        self.scenario_dir = str(self.paths.active_scenarios_dir)
        self.exports_dir = str(self.paths.exports_dir)
        self.topo_xml_dir = str(self.paths.exports_dir / "TopoXML")
        self.images_dir = str(self.paths.images_dir)
        self.config_dir = str(self.paths.config_dir)
        self.config_path = str(self.paths.config_path_for_runtime())
        self.appConfig: Dict = {}
        self.loadAppConfig()
        self.catalogue_path = str(self.paths.catalogue_path_for_mode(self.application_context.mode))
        self.catalogue = _load_application_catalogue(
            self.catalogue_path,
            self.application_context.catalogue_id_provider,
        )
        self.scenario_map_controller = ScenarioMapController(
            self.catalogue,
            self.paths,
            self.background_map_layer,
            self._get_active_scenario,
        )
        self._beacon_world_resolver = BeaconWorldResolver(
            self.catalogue, self.scenario_map_controller.lambert_to_world,
        )
        # === UI : persistance des toggles de visualisation (dico + compas) ===
        # Doit être fait AVANT _build_ui() pour que le checkbutton + le pack initial
        # reflètent correctement l'état sauvegardé.
        self.show_dico_panel.set(bool(self.getAppConfigValue("uiShowDicoPanel", True)))
        self.show_clock_overlay.set(bool(self.getAppConfigValue("uiShowClockOverlay", True)))
        self.show_only_group_contours.set(bool(self.getAppConfigValue("uiShowOnlyGroupContours", False)))
        self.auto_fit_scenario_select.set(bool(self.getAppConfigValue("uiAutoFitScenario", False)))
        self.map_opacity.set(int(self.getAppConfigValue("uiMapOpacity", 70)))
        self.show_balises_layer.set(bool(self.getAppConfigValue("uiShowBalisesLayer", False)))
        self._layerGuidesVisible = bool(self.getAppConfigValue("uiShowGuidesLayer", True))
        self._layerGuidesVisibleVar.set(bool(self._layerGuidesVisible))
        self._clock_auto_ref_sync_enabled = bool(self.getAppConfigValue("uiClockAutoRefSyncEnabled", False))
        self._clock_auto_ref_sync_var = tk.BooleanVar(value=bool(self._clock_auto_ref_sync_enabled))

        # === Simulation : derniers paramètres utilisés (dialog 'Simulation > Assembler…') ===
        self._simulation_last_algo_id = str(self.getAppConfigValue("simLastAlgoId", "") or "").strip()
        self._simulation_last_n = int(self.getAppConfigValue("simLastN", 8) or 8)
        self._simulation_last_order = str(self.getAppConfigValue("simLastOrder", "forward") or "forward").strip().lower()
        if self._simulation_last_order not in ("forward", "reverse"):
            self._simulation_last_order = "forward"
        self._simulation_last_first_edge = str(self.getAppConfigValue("simLastFirstEdge", "OL") or "OL").strip().upper()
        if self._simulation_last_first_edge not in ("OL", "BL"):
            self._simulation_last_first_edge = "OL"
        self._simulation_last_beacon_id = str(
            self.getAppConfigValue("simLastBeaconId", "") or ""
        ).strip()

        # Fond SVG : au démarrage le canvas n'a pas toujours une taille valide.
        self._triangle_list_triangle_ids: list[str] = []
        self.canvas_objects = CanvasObjectsCollection()
        self._last_drawn = self.canvas_objects.entries

        # Crée le scénario manuel "par défaut" qui pointe sur l'état runtime.
        # last_drawn et groups sont partagés par référence : toute modification
        # manuelle met automatiquement à jour ce scénario.
        manual = ScenarioAssemblage(
            name="Scénario manuel",
            source_type="manual",
            algo_id=None,
            hypothesis=self._create_manual_scenario_hypothesis(),
            is_placeholder=True,
        )
        manual.last_drawn = self._last_drawn
        manual.view_state = self._capture_view_state()
        manual.map_state = self.scenario_map_controller.new_default_state()
        manual.book_ref_id = self.catalogue.default_book_id
        self.scenarios.append(manual)
        self._attach_beacon_resolver_to_world(manual.topoWorld)
        self._apply_scenario_map_state(manual.map_state, redraw=False)

        # --- Horloge (overlay fixe) : état par défaut ---
        # hour peut être un float (si l'aiguille des heures avance avec les minutes)
        self.compass_state = CompassState(
            ref_azimuth_deg=float(self.getAppConfigValue("uiClockRefAzimuth", 0.0) or 0.0),
        )
        self.compass_controller = CompassController(
            self.compass_state,
            self._world_to_screen,
            self._screen_to_world,
            self._get_active_scenario,
            lambda: self._last_drawn,
            lambda: self.decryptor,
            lambda: bool(self.show_clock_overlay.get()),
            lambda: self.dictionary_panel.filter_active,
            lambda: bool(self._ctrl_down),
            lambda text: self.status.config(text=text),
            lambda element_id: self.canvas_objects.get_index_by_topology_id(element_id),
            self._simulation_cancel_dictionary_filter,
            self._ctx_filter_dictionary_by_clock_arc,
            lambda: self._guidesCurrentColorHex,
            lambda: self._redraw_from(self._last_drawn),
            self._update_compass_ctx_menu_and_dico_state,
            lambda value: self.setAppConfigValue("uiClockRefAzimuth", value),
            lambda: bool(self._clock_auto_ref_sync_enabled),
            lambda: (
                beacon_id if (beacon_id := self._getCheminsBeaconRefId()) in self.catalogue.beacons else None
            ),
            self._beacon_world_resolver.get_world,
            lambda: [
                {"beaconId": beacon.beacon_id, "label": self.catalogue.get_city(beacon.city_id).name}
                for beacon in self.catalogue.iter_beacons() if not beacon.archived
            ],
            self._clock_refresh_active_preview_under_pointer,
        )

        # Par défaut: mapping "horloge <-> dico" v1.
        self.decryptor: DecryptorBase = ClockDicoDecryptor()

        self._build_ui()
        # Centraliser / garantir les bindings (création initiale)
        self._bind_canvas_handlers()
        self._rebuild_triangle_listbox_from_core()

        # Bind pour annuler avec ESC (drag ou sélection)
        self.bind("<Escape>", self._on_escape_key)

        # === Dictionnaire : livre Catalogue du scénario actif ===
        self._reload_dictionary_for_active_scenario(reset_reference=True)

    # ======================================================================
    #  Topologie (bridge minimal Tk -> Core)
    # ======================================================================

    def _bind_canvas_objects(self, entries) -> None:
        """Attache la collection structurelle aux entrees du scenario actif."""
        for entry in entries or ():
            self._strip_core_duplicates_from_last_drawn_entry(entry)
        self.canvas_objects = CanvasObjectsCollection(entries)
        self._last_drawn = self.canvas_objects.entries

    def _get_canvas_display_world(self) -> TopologyWorld:
        """Retourne le Core qui a produit les objets actuellement affiches."""
        state = self._deformation_state
        if state.active:
            if state.last_accepted_world is not None:
                return state.last_accepted_world
            if state.reference_world is not None:
                return state.reference_world
        return self._get_active_scenario().topoWorld

    def _get_canvas_display_hypothesis(self) -> ScenarioHypothesis | None:
        """Retourne l'hypothese associee au Core actuellement affiche."""
        state = self._deformation_state
        if state.active and state.working_hypothesis is not None:
            return state.working_hypothesis
        return self._get_active_scenario().hypothesis

    @staticmethod
    def _strip_core_duplicates_from_last_drawn_entry(entry: Dict) -> None:
        """Retire les anciennes copies Core d'une entree de projection."""
        if isinstance(entry, dict):
            for key in ("id", "orient", "topoGroupId", "mirrored", "group_id", "labels"):
                entry.pop(key, None)

    def _get_active_core_group_id_for_entry(self, entry: Dict) -> Optional[str]:
        """Résout le ``core_group_id`` canonique d'une entrée projetée."""
        scen = self._get_active_scenario()
        world = scen.topoWorld
        element_id = entry["topoElementId"]
        if not element_id:
            return None

        core_gid = world.get_group_of_element(element_id)
        return core_gid if core_gid else None

    def _get_core_group_id_for_triangle_index(
        self, tri_index: int, world: TopologyWorld | None = None
    ) -> Optional[str]:
        """Resout un triangle projete vers son groupe Core canonique."""
        if not (0 <= int(tri_index) < len(self._last_drawn)):
            return None
        if world is None:
            world = self._get_active_scenario().topoWorld
        topo_element_id = str(self._last_drawn[int(tri_index)].get("topoElementId", "") or "").strip()
        if not topo_element_id:
            return None

        core_group_id = world.get_group_of_element(topo_element_id)
        return None if not core_group_id else core_group_id

    def _get_projected_elements_for_core_group(
        self, core_group_id: str, world: TopologyWorld | None = None
    ) -> Tuple[Dict, ...]:
        """Retourne la projection des membres du groupe Core canonique."""
        if world is None:
            world = self._get_active_scenario().topoWorld
        if not core_group_id:
            return ()

        element_ids = world.getGroupElementIds(core_group_id)
        return self.canvas_objects.get_many_by_topology_ids(element_ids)

    def get_last_drawn_entries_for_core_group(
        self, core_group_id: str, world: TopologyWorld | None = None
    ) -> List[Dict]:
        """Navigue ``groupe Core -> element_ids -> entrées UI``.

        Le groupe est canonisé par ``TopologyWorld``. Les éléments absents du
        cache UI sont simplement omis : le validateur DEBUG est chargé de les
        signaler sans interrompre le comportement existant.
        """
        if world is None:
            world = self._get_active_scenario().topoWorld
        if world is None or not core_group_id:
            return []
        element_ids = world.getGroupElementIds(core_group_id)
        return list(self.canvas_objects.get_many_by_topology_ids(element_ids))

    def _get_core_triangle_world_points(
        self,
        world: TopologyWorld,
        element_id: str,
    ) -> Dict[str, np.ndarray]:
        """Lit les sommets monde d'un triangle exclusivement depuis le Core."""
        return getCoreTriangleWorldPoints(world, element_id)

    def _get_core_group_world_centroid(
        self,
        world: TopologyWorld,
        core_group_id: str,
    ) -> np.ndarray:
        """Retourne le barycentre monde de tous les sommets d'un groupe Core."""
        if not core_group_id:
            raise ValueError("[MIG-CACHE-TRANSFORM-001D] groupe Core absent")
        element_ids = tuple(world.getGroupElementIds(core_group_id))
        if not element_ids:
            raise ValueError(
                f"[MIG-CACHE-TRANSFORM-001D] groupe Core vide ou introuvable: {core_group_id!r}"
            )
        points = [
            point
            for element_id in element_ids
            for point in self._get_core_triangle_world_points(world, element_id).values()
        ]
        centroid = np.mean(np.asarray(points, dtype=float), axis=0)
        if centroid.shape != (2,) or not np.all(np.isfinite(centroid)):
            raise ValueError("[MIG-CACHE-TRANSFORM-001D] barycentre groupe invalide")
        return np.array(centroid, dtype=float, copy=True)

    def _get_core_element_mirrored(self, element_id: str) -> bool:
        """Lit l'état miroir d'affichage dans la pose Core, jamais dans le cache."""
        scen = self._get_active_scenario()
        world = scen.topoWorld
        if not element_id:
            return False
        _, _, mirrored = world.getElementPose(element_id)
        return bool(mirrored)

    def _get_core_element_from_last_drawn_entry(
        self,
        entry: Dict,
        world: TopologyWorld | None = None,
    ):
        """Resout une entree projetee vers son element Core.

        Les informations de groupe, orientation et miroir sont lues depuis le
        Core via ``topoElementId`` ; elles ne sont pas du cache graphique.
        """
        if world is None:
            world = self._get_canvas_display_world()
        element_id = entry["topoElementId"]
        if not element_id:
            return None
        return world.elements.get(element_id)

    def _get_core_vertex_labels(
        self,
        entry: Dict,
        world: TopologyWorld | None = None,
    ) -> tuple[str, str, str]:
        """Lit les trois libelles de sommets depuis l'element Core associe."""
        element = self._get_core_element_from_last_drawn_entry(entry, world)
        element_id = entry["topoElementId"]
        if element is None:
            raise KeyError(
                f"[MIG-CACHE-CLEANUP-002] element Core absent: {element_id!r}"
            )
        labels = tuple(str(label) for label in (element.vertex_labels or ()))
        if len(labels) != 3:
            raise ValueError(
                "[MIG-CACHE-CLEANUP-002] vertex_labels invalides "
                f"element={element_id!r}: {labels!r}"
            )
        return labels

    def _build_triangle_display_label(
        self,
        entry: Dict,
        world: TopologyWorld | None = None,
        hypothesis: ScenarioHypothesis | None = None,
    ) -> str:
        """Construit le label UX depuis l'hypothèse et la pose Core.

        Le rang n'est jamais dupliqué dans la projection : il reste lu dans le Core.
        """
        element = self._get_core_element_from_last_drawn_entry(entry, world)
        element_id = entry["topoElementId"]
        if not element_id:
            raise ValueError("[MIG-UX-LABEL-001] topoElementId absent")
        if element is None:
            raise KeyError(
                f"[MIG-UX-LABEL-001] element Core absent: {element_id!r}"
            )
        if hypothesis is None:
            hypothesis = self._get_canvas_display_hypothesis()
        if hypothesis is None:
            raise ValueError("[MIG-UX-LABEL-001] ScenarioHypothesis absente")
        triangle_id = element.source_triangle_id
        if not triangle_id:
            raise ValueError(
                "[MIG-UX-LABEL-001] source_triangle_id absent pour "
                f"element={element_id!r}"
            )
        try:
            tri_rank = hypothesis.get_rank_for_triangle_ref(triangle_id)
        except ValueError as exc:
            raise ValueError(
                "[MIG-UX-LABEL-001] triangle Catalogue absent de l'hypothèse: "
                f"element={element_id!r}, triangle_id={triangle_id!r}"
            ) from exc
        _rotation, _translation, mirrored = element.get_pose()
        return "T" + str(tri_rank) + ("S" if bool(mirrored) else "")

    def _project_core_element_to_last_drawn(
        self,
        world: TopologyWorld,
        element_id: str,
    ) -> Dict:
        """Projette les sommets monde d'un élément Core dans le cache actif.

        Cette projection ne touche qu'à ``entry["pts"]``. Les coordonnées
        locales et la pose restent l'autorité du Core ; aucune synchronisation
        inverse ne doit être déclenchée depuis ce chemin.
        """
        entry = self.canvas_objects.get_by_topology_id(element_id)
        if entry is None:
            raise KeyError(
                f"[MIG-CACHE-TRANSFORM-001] projection absente pour topoElementId={element_id!r}"
            )

        self._strip_core_duplicates_from_last_drawn_entry(entry)
        entry["pts"] = self._get_core_triangle_world_points(world, element_id)
        return entry

    def _project_core_group_to_last_drawn(
        self,
        world: TopologyWorld,
        core_group_id: str,
    ) -> Tuple[Dict, ...]:
        """Projette les membres déterminés exclusivement par le groupe Core."""
        element_ids = world.getGroupElementIds(core_group_id)
        if not element_ids and not world.hasLiveGroup(core_group_id):
            raise ValueError(
                f"[MIG-CACHE-TRANSFORM-001] groupe Core invalide: {core_group_id!r}"
            )
        return tuple(
            self._project_core_element_to_last_drawn(world, element_id)
            for element_id in element_ids
        )

    def _preview_attachment_rotation_to_last_drawn(self, preview) -> None:
        """Tourne le MOVE courant vers l'orientation V2, sans le translater.

        Le clone V2 sert exclusivement à calculer l'angle compatible avec
        l'attache. La pose de départ reste celle du drag libre, et le pivot est
        le sommet actuellement saisi : aucun point de la projection n'est
        déplacé vers la destination de l'attache.
        """
        if not preview.accepted or preview.world is None:
            raise ValueError("[ATT-003D] preview Attachment V2 non accepté")
        if not isinstance(self._sel, dict) or self._sel.get("mode") != "move_group":
            raise RuntimeError("[ATT-003D] sélection MOVE absente pour la preview")
        core_group_id = self._sel.get("core_group_id")
        if not core_group_id:
            raise RuntimeError("[ATT-003D] core_group_id absent pour la preview")
        anchor = self._sel.get("anchor")
        if not anchor or anchor.get("type") != "vertex":
            return
        intent = self._attachment_intent
        mobile_element_id = intent.mob_element_id
        edge_vertices = {
            "OB": ("O", "B"),
            "BL": ("B", "L"),
            "LO": ("L", "O"),
        }.get(intent.mob_edge)
        if edge_vertices is None:
            raise ValueError("[ATT-003D] arête mobile invalide pour la preview")

        real_world = self._get_active_scenario().topoWorld
        mobile_element_ids = tuple(real_world.getGroupElementIds(str(core_group_id)))
        for element_id in mobile_element_ids:
            if self.canvas_objects.get_by_topology_id(element_id) is None:
                raise KeyError(
                    "[ATT-003D] projection absente "
                    f"pour topoElementId={element_id!r}"
                )
        mobile_entry = self.canvas_objects.get_by_topology_id(mobile_element_id)
        if mobile_entry is None:
            raise KeyError(
                "[ATT-003D] projection absente "
                f"pour topoElementId={mobile_element_id!r}"
            )
        edge_from, edge_to = edge_vertices
        current_points = mobile_entry["pts"]
        preview_points = self._get_core_triangle_world_points(
            preview.world, mobile_element_id
        )
        current_vector = np.asarray(current_points[edge_to], dtype=float) - np.asarray(
            current_points[edge_from], dtype=float
        )
        preview_vector = np.asarray(preview_points[edge_to], dtype=float) - np.asarray(
            preview_points[edge_from], dtype=float
        )
        if np.linalg.norm(current_vector) <= EPS_WORLD or np.linalg.norm(preview_vector) <= EPS_WORLD:
            raise ValueError("[ATT-003D] arête dégénérée pour la preview")
        angle = math.atan2(
            current_vector[0] * preview_vector[1] - current_vector[1] * preview_vector[0],
            float(np.dot(current_vector, preview_vector)),
        )
        anchor_tid = int(anchor["tid"])
        anchor_vkey = anchor["vkey"]
        if not 0 <= anchor_tid < len(self._last_drawn):
            raise IndexError("[ATT-003D] sommet pivot absent pour la preview")
        pivot = np.asarray(self._last_drawn[anchor_tid]["pts"][anchor_vkey], dtype=float)
        free_move_pts = self._capture_move_preview_initial_pts(
            real_world, str(core_group_id)
        )
        self._preview_rotate_group_from_snapshot(free_move_pts, pivot, angle)
        self._replace_mobile_attachment_highlight_with_current_edge(intent)

    def _replace_mobile_attachment_highlight_with_current_edge(self, intent) -> None:
        """Affiche l'arête mobile V2 dans la même pose que le preview CTRL."""
        data = self._edge_highlights
        if not data or data.get("best") is None:
            return
        edge_vertices = {
            "OB": ("O", "B"),
            "BL": ("B", "L"),
            "LO": ("L", "O"),
        }.get(intent.mob_edge)
        if edge_vertices is None:
            raise ValueError("[ATT-003D] arête mobile invalide pour le highlight")
        entry = self.canvas_objects.get_by_topology_id(intent.mob_element_id)
        if entry is None:
            raise KeyError(
                "[ATT-003D] projection absente "
                f"pour topoElementId={intent.mob_element_id!r}"
            )
        edge_from, edge_to = edge_vertices
        target_from, target_to = data["best"][2:]
        points = entry["pts"]
        data["best"] = (
            tuple(points[edge_from]),
            tuple(points[edge_to]),
            target_from,
            target_to,
        )
        # Les anciens contours mobiles seraient dessinés à la pose libre et
        # donneraient l'impression d'un second triangle sous le preview CTRL.
        data["mob_outline"] = []
        data["mob_inc"] = []

    def _project_core_group_to_collection(
        self,
        world: TopologyWorld,
        core_group_id: str,
        collection: CanvasObjectsCollection,
    ) -> Tuple[Dict, ...]:
        """Projette un groupe Core dans la collection Canvas explicitement fournie."""
        element_ids = tuple(world.getGroupElementIds(core_group_id))
        if not element_ids and not world.hasLiveGroup(core_group_id):
            raise ValueError(
                f"[MIG-CACHE-TRANSFORM-001H] groupe Core invalide: {core_group_id!r}"
            )
        projected = []
        for element_id in element_ids:
            entry = collection.get_by_topology_id(element_id)
            if entry is None:
                raise KeyError(
                    "[MIG-CACHE-TRANSFORM-001H] projection absente "
                    f"pour topoElementId={element_id!r}"
                )
            self._strip_core_duplicates_from_last_drawn_entry(entry)
            entry["pts"] = self._get_core_triangle_world_points(world, element_id)
            projected.append(entry)
        return tuple(projected)

    def _build_scenario_projection_from_core(self, scen: ScenarioAssemblage) -> list[dict]:
        """Construit une projection neuve sans lire le cache du scénario."""
        world = scen.topoWorld
        if scen.source_type == "auto":
            element_ids = scen.orderedElementIds
        else:
            element_ids = getManualProjectionElementIds(world)
        return buildLastDrawnFromTopology(
            topologyWorld=world,
            elementIds=element_ids,
        )

    def _rebuild_active_projection_from_core(self) -> None:
        """Remplace le cache actif par une projection entièrement issue du Core."""
        scen = self._get_active_scenario()
        world = scen.topoWorld
        projection = self._build_scenario_projection_from_core(scen)
        self._bind_canvas_objects(projection)
        self.canvas_objects.validate_against_world(world)
        scen.last_drawn = self._last_drawn

    def _project_auto_scenario_from_core(self, scen: ScenarioAssemblage) -> None:
        """Régénère le cache AUTO depuis son monde et son ordre Core explicite."""
        if scen.source_type != "auto":
            return
        if scen is self._get_active_scenario():
            self._rebuild_active_projection_from_core()
            return
        projection = self._build_scenario_projection_from_core(scen)
        CanvasObjectsCollection(projection).validate_against_world(scen.topoWorld)
        scen.last_drawn = projection

    def _project_all_auto_scenarios_from_core(self) -> None:
        """Régénère chaque cache AUTO après un commit global Core-first."""
        for scen in self.scenarios or ():
            self._project_auto_scenario_from_core(scen)

    def _rotate_all_auto_scenarios_around_anchors(self, dtheta: float) -> None:
        """Tourne chaque scénario AUTO autour du pivot imposé par son ancre."""
        for scen in self.scenarios or ():
            if scen.source_type != "auto":
                continue
            world = scen.topoWorld
            ordered_element_ids = scen.orderedElementIds
            core_group_id = world.get_group_of_element(ordered_element_ids[0])
            if core_group_id is None:
                raise RuntimeError("Simulation AUTO: groupe final absent")
            anchor = world.getAnchorForGroup(core_group_id)
            if anchor is None:
                raise RuntimeError("Simulation AUTO: ancre de groupe absente")
            pivot_world = np.asarray(
                world.getBeaconWorldXY(anchor.beacon_id), dtype=float
            )
            world.rotate_group(core_group_id, pivot_world, dtheta)
            self._project_auto_scenario_from_core(scen)

        state = self.auto_rotation_state or {"thetaDeg": 0.0}
        self.auto_rotation_state = {
            "thetaDeg": (float(state["thetaDeg"]) + math.degrees(dtheta)) % 360.0
        }

    def _get_active_scenario(self) -> ScenarioAssemblage | None:
        if not self.scenarios:
            return None
        idx = int(self.active_scenario_index or 0)
        if idx < 0 or idx >= len(self.scenarios):
            return None
        return self.scenarios[idx]

    def _on_export_topodump_key(self, event=None):
        """Export TopoDump_<scenarioId>.xml du scénario actif (F11)."""
        scen = self._get_active_scenario()
        world = scen.topoWorld
        out_name = "TopoDump.xml"
        out_path = os.path.join(self.topo_xml_dir, out_name)
        world.export_topo_dump_xml(
            out_path,
            orientation="cw",
        )
        self.status.config(text=f"TopoDump exporté : {out_name}")

    @staticmethod
    def _geo_orient_from_points(points) -> tuple[str, float | None]:
        """Mesure l'orientation réelle O/B/L sans consulter orient ni mirrored."""
        try:
            O = np.asarray(points["O"], dtype=float)
            B = np.asarray(points["B"], dtype=float)
            L = np.asarray(points["L"], dtype=float)
            cross = float((B[0] - O[0]) * (L[1] - O[1]) - (B[1] - O[1]) * (L[0] - O[0]))
            if not np.isfinite(cross):
                return "<invalide>", None
            if abs(cross) <= 1e-12:
                return "<degénérée>", cross
            return ("CCW" if cross > 0.0 else "CW"), cross
        except Exception:
            return "<absent>", None

    @staticmethod
    def _geo_orient_point_text(point) -> str:
        try:
            value = np.asarray(point, dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                return "<invalide>"
            return f"({float(value[0]):.9g}, {float(value[1]):.9g})"
        except Exception:
            return "<absent>"

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
        self.menu_scenario.add_command(label="Enregistrer", command=self._scenario_save)
        self.menu_scenario.add_command(label="Enregistrer sous…", command=self._scenario_save_as_dialog)
        self.menu_scenario.add_separator()
        self.menu_scenario.add_command(label="Propriétés…", command=self._scenario_edit_properties)
        self.menu_scenario.add_separator()
        # placeholder pour la liste des .xml du dossier 'scenario'
        self._menu_scenario_files_anchor = self.menu_scenario.index("end")
        self.menu_scenario.add_command(label="(scan des scénarios…)", state="disabled")
        self._rebuild_scenario_file_list_menu()

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
        self.menu_visual.add_checkbutton(
            label="Redimensionner et déplacer la carte",
            variable=self.bg_resize_mode,
            command=self._toggle_bg_resize_mode,
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

    def _create_manual_scenario_hypothesis(self, *, report_error: bool = False):
        """Instancie l'hypothèse propriétaire d'un nouveau scénario manuel."""
        try:
            return create_default_scenario_hypothesis(self.catalogue)
        except ValueError as exc:
            if report_error:
                messagebox.showerror("Nouveau scénario", str(exc), parent=self)
            return None

    def _simulation_get_triangle_ids_first_n(self, n: int) -> List[str]:
        """Retourne les IDs Catalogue du scénario, dans leur ordre métier."""
        scen = self._get_active_scenario()
        if scen.hypothesis is None:
            raise ValueError("Simulation: ScenarioHypothesis absente du scénario actif")
        return list(scen.hypothesis.triangle_ids_by_rank[:max(0, int(n))])

    def _simulation_get_triangle_ids_by_order(self, n: int, order: str = "normal") -> List[str]:
        """Retourne les IDs logiques des triangles selon l'ordre choisi:
        - normal : n premiers (début de listbox)
        - inverse : n derniers en partant du dernier (ex: 32,31,30,...)
        """
        triangle_ids = self._simulation_get_triangle_ids_first_n(32)
        n2 = min(max(0, int(n)), len(triangle_ids))
        if str(order).lower() in ("inverse", "reverse"):
            return list(reversed(triangle_ids[-n2:]))
        return triangle_ids[:n2]

    def _simulation_clear_auto_scenarios(self):
        """Supprime tous les scénarios 'auto' (conserve les manuels)."""
        if not self.scenarios:
            return
        kept = [s for s in self.scenarios if getattr(s, "source_type", "manual") == "manual"]
        removed = len(self.scenarios) - len(kept)
        self.scenarios = kept
        if removed:
            self.auto_rotation_state = None
        if not self.scenarios:
            manual = ScenarioAssemblage(
                name="Scénario manuel",
                source_type="manual",
                hypothesis=self._create_manual_scenario_hypothesis(),
            )
            manual.book_ref_id = self.catalogue.default_book_id
            self._attach_beacon_resolver_to_world(manual.topoWorld)
            self.scenarios = [manual]
        self.active_scenario_index = min(self.active_scenario_index, len(self.scenarios) - 1)
        self._set_active_scenario(self.active_scenario_index)
        self._refresh_scenario_listbox()
        self.status.config(text=f"Scénarios auto supprimés : {removed}")

    def _simulation_assemble_dialog(self):
        """Ouvre la boîte de dialogue 'Assembler…' et lance l'algo choisi."""
        active_scenario = self._get_active_scenario()
        if active_scenario is None or active_scenario.hypothesis is None:
            raise ValueError("Simulation: ScenarioHypothesis absente du scénario actif")
        active_scenario.hypothesis.validate(
            GeometryReferenceResolver(self.catalogue, active_scenario.reference)
        )
        n_max = len(active_scenario.hypothesis.triangle_ids_by_rank)
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
        beacon_items = [
            (
                beacon.beacon_id,
                f"{self.catalogue.get_city(beacon.city_id).name} ({beacon.beacon_id})",
            )
            for beacon in get_anchor_beacon_candidates(self.catalogue)
        ]
        if not beacon_items:
            messagebox.showwarning(
                "Assembler", "Impossible de lancer la simulation :\naucune balise disponible."
            )
            return
        beacon_ids = {beacon_id for beacon_id, _label in beacon_items}
        orientation_reference_by_beacon = {
            beacon_id: (
                self._find_orientation_reference_for_beacon(active_scenario, beacon_id)
                if active_scenario is not None
                else None
            )
            for beacon_id in beacon_ids
        }
        default_beacon_id = (
            self._simulation_last_beacon_id
            if self._simulation_last_beacon_id in beacon_ids
            else beacon_items[0][0]
        )

        dlg = DialogSimulationAssembler(
            self,
            algo_items,
            n_max=n_max,
            default_algo_id=default_algo_id,
            default_n=default_n,
            default_order=default_order,
            beacon_items=beacon_items,
            orientation_reference_by_beacon=orientation_reference_by_beacon,
            default_beacon_id=default_beacon_id,
            default_first_edge=default_first_edge,
        )
        self.wait_window(dlg)
        if not getattr(dlg, "result", None):
            return

        algo_id, n, order, beacon_id, initial_orientation = dlg.result
        if initial_orientation.mode == "edge_north":
            first_edge_for_engine = initial_orientation.edge
        else:
            first_edge_for_engine = "OL"

        self._simulation_last_order = order
        self._simulation_last_algo_id = algo_id
        self._simulation_last_n = int(n)

        self._simulation_last_first_edge = str(first_edge_for_engine or "OL").upper().strip()
        if self._simulation_last_first_edge not in ("OL", "BL"):
            self._simulation_last_first_edge = "OL"

        # Persister (app config)
        self.setAppConfigValue("simLastAlgoId", str(self._simulation_last_algo_id or ""))
        self.setAppConfigValue("simLastN", int(self._simulation_last_n or 0))
        self.setAppConfigValue("simLastOrder", str(self._simulation_last_order or "forward"))
        self.setAppConfigValue("simLastFirstEdge", str(self._simulation_last_first_edge or "OL"))
        self._simulation_last_beacon_id = beacon_id
        self.setAppConfigValue("simLastBeaconId", beacon_id)
        self.saveAppConfig()

        # Par design : on détruit systématiquement les scénarios auto existants
        self._simulation_clear_auto_scenarios()

        # --- Construire la liste des triangles selon l'ordre choisi ---
        triangle_ids = self._simulation_get_triangle_ids_by_order(n, order)

        # Sécurité
        if len(triangle_ids) < 2:
            messagebox.showwarning("Assembler", "Impossible de construire la liste des IDs de triangles.")
            return

        # Forcer n pair
        if len(triangle_ids) % 2 == 1:
            triangle_ids = triangle_ids[:-1]

        engine = MoteurSimulationAssemblage(
            self,
            source_hypothesis=active_scenario.hypothesis,
            source_reference=active_scenario.reference,
        )
        engine.firstTriangleEdge = str(first_edge_for_engine or "OL").upper()
        engine.initialTriangleOrientation = initial_orientation
        algo_cls = ALGOS.get(algo_id)
        if algo_cls is None:
            raise ValueError(f"Algo inconnu: {algo_id}")
        algo = algo_cls(engine)

        # Snapshot de la carte auto (carte affichée au moment du lancement)
        self.auto_map_state = self.scenario_map_controller.capture_active_state()
        scenarios = algo.run(triangle_ids)

        base_idx = len(self.scenarios)
        count_auto = sum(1 for s in self.scenarios if s.source_type == "auto")
        for k, scen in enumerate(scenarios):
            if not isinstance(scen, ScenarioAssemblage):
                continue
            scen.source_type = "auto"
            scen.view_state = self._capture_view_state()
            scen.map_state = self.auto_map_state
            scen.book_ref_id = active_scenario.book_ref_id
            scen.algo_id = scen.algo_id or algo_id
            if scen.hypothesis is None:
                raise RuntimeError("Simulation: scénario AUTO sans ScenarioHypothesis")
            scen.traversal_direction = "reverse" if str(order).lower() in ("reverse", "inverse") else "forward"
            self._attach_beacon_resolver_to_world(scen.topoWorld)
            self._anchor_auto_scenario_to_beacon(scen, beacon_id)
            if not scen.name:
                scen.name = f"Auto #{count_auto + k + 1}"
            self.scenarios.append(scen)

        self.auto_rotation_state = {"thetaDeg": 0.0}

        self._refresh_scenario_listbox()
        self._set_active_scenario(base_idx)
        self.status.config(text=f"Simulation: {len(scenarios)} scénario(s) généré(s) (algo={algo_id}, n={n})")

    def _is_deformation_mode_active(self) -> bool:
        window = self._deformation_window
        return bool(
            self._deformation_state.active
            and window is not None
            and window.winfo_exists()
        )

    def _open_deformation_window(self) -> None:
        window = self._deformation_window
        if window is not None and window.winfo_exists():
            window.deiconify()
            window.lift()
            window.focus_force()
            return
        if self._deformation_state.active:
            self._exit_deformation_mode()
        self._deformation_state.enter()
        self._deformation_state.working_reference = (
            self._get_active_scenario().reference.clone()
        )
        self._deformation_state.working_hypothesis = (
            self._get_active_scenario().hypothesis.clone()
            if self._get_active_scenario().hypothesis is not None
            else None
        )
        self._deformation_canvas_mode = "select"
        window = self._ensure_deformation_window()
        window.set_canvas_mode(self._deformation_canvas_mode)
        window.deiconify()
        window.lift()
        window.focus_force()
        self.status.config(text="DEFORM ouvert : sélectionnez un triangle ancré.")

    def _deformation_projection_from_world(self, world: TopologyWorld) -> list[dict]:
        scen = self._get_active_scenario()
        element_ids = (
            scen.orderedElementIds
            if scen.source_type == "auto"
            else getManualProjectionElementIds(world)
        )
        return buildLastDrawnFromTopology(
            topologyWorld=world,
            elementIds=element_ids,
        )

    def _show_deformation_preview(self, world: TopologyWorld) -> None:
        projection = self._deformation_projection_from_world(world)
        self._bind_canvas_objects(projection)
        # La projection candidate remplace _last_drawn avant le redraw : le
        # prochain clic DEFORM doit toujours reconstruire ses polygones écran.
        self._invalidate_pick_cache()
        self.canvas_objects.validate_against_world(world)
        self._redraw_from(self._last_drawn)

    def _restore_deformation_real_projection(self) -> None:
        self._rebuild_active_projection_from_core()
        self._redraw_from(self._last_drawn)

    def _deformation_working_reference(self):
        state = self._deformation_state
        return state.working_reference or self._get_active_scenario().reference

    def _deformation_working_hypothesis(self):
        state = self._deformation_state
        hypothesis = state.working_hypothesis or self._get_active_scenario().hypothesis
        if hypothesis is None:
            raise ValueError("ScenarioHypothesis absente")
        return hypothesis

    def _materialize_deformation_working_reference(
        self,
        world: TopologyWorld,
        *,
        reference=None,
        hypothesis=None,
    ) -> TopologyWorld:
        """Projette les refs TRI/STRI effectives de session dans un world candidat."""
        scenario = self._get_active_scenario()
        if scenario.hypothesis is None:
            raise ValueError("ScenarioHypothesis absente")
        working_reference = reference or self._deformation_working_reference()
        working_hypothesis = hypothesis or self._deformation_working_hypothesis()
        resolver = GeometryReferenceResolver(
            self.catalogue, working_reference
        )
        candidate_world = world.clonePhysicalState()
        for element_id, element in candidate_world.elements.items():
            triangle_ref_id = element.source_triangle_id
            if not triangle_ref_id:
                raise ValueError("Triangle de deformation sans source Catalogue")
            # Un world de base peut encore porter un TRI publie tandis que le
            # draft COW porte deja le STRI du meme rang. Une ref effective
            # STRI doit toutefois toujours etre resolue dans le draft.
            if triangle_ref_id in working_hypothesis.triangle_ids_by_rank:
                rank = working_hypothesis.get_rank_for_triangle_ref(triangle_ref_id)
            elif triangle_ref_id in self._deformation_working_hypothesis().triangle_ids_by_rank:
                rank = self._deformation_working_hypothesis().get_rank_for_triangle_ref(
                    triangle_ref_id
                )
            else:
                rank = scenario.hypothesis.get_rank_for_triangle_ref(triangle_ref_id)
            effective_triangle_ref_id = working_hypothesis.triangle_ids_by_rank[rank - 1]
            if effective_triangle_ref_id != triangle_ref_id:
                candidate_world.replace_element_materialized_definition(
                    element_id,
                    materialize_triangle(resolver, effective_triangle_ref_id),
                )
        return candidate_world

    def _apply_deformation_working_point_names(self, preview_commit) -> None:
        """Rejoue les renommages temporaires sur les SCITY recrees par le COW."""
        state = self._deformation_state
        if not state.working_point_names:
            return
        resolver = GeometryReferenceResolver(self.catalogue, preview_commit.reference)
        renamed_city_ids = set()
        for point_id, name in state.working_point_names.items():
            point = state.working_points.get(point_id)
            if point is None:
                continue
            element_id, role = next(iter(point.occurrences))
            element = preview_commit.world.elements[element_id]
            if not element.source_triangle_id:
                raise ValueError("Triangle de deformation sans source Catalogue")
            city_id = resolver.city_ref_ids_by_role(element.source_triangle_id)[role]
            preview_commit.reference.rename_city(city_id, name)
            renamed_city_ids.add(city_id)

        if not renamed_city_ids:
            return
        resolver = GeometryReferenceResolver(self.catalogue, preview_commit.reference)
        for element_id, element in preview_commit.world.elements.items():
            if not element.source_triangle_id:
                raise ValueError("Triangle de deformation sans source Catalogue")
            city_ids = resolver.city_ref_ids_by_role(element.source_triangle_id)
            if renamed_city_ids.intersection(city_ids.values()):
                preview_commit.world.replace_element_materialized_definition(
                    element_id,
                    materialize_triangle(resolver, element.source_triangle_id),
                )

    def _close_deformation_window(self) -> None:
        window = self._deformation_window
        self._deformation_window = None
        if window is not None and window.winfo_exists():
            window.destroy()

    def _on_deformation_window_closed(self) -> None:
        self._exit_deformation_mode()

    def _deformation_vertices(self) -> dict[str, DeformationVertex]:
        state = self._deformation_state
        if state.element_id is None or state.reference_world is None:
            raise RuntimeError("Triangle de deformation absent")
        world = state.last_accepted_world or state.reference_world
        element = world.elements.get(state.element_id)
        if element is None or not element.source_triangle_id:
            raise ValueError("Triangle de deformation sans source Catalogue")
        resolver = GeometryReferenceResolver(
            self.catalogue, self._deformation_working_reference()
        )
        city_id_by_role = resolver.city_ref_ids_by_role(element.source_triangle_id)
        vertices = {}
        for role, city_id in city_id_by_role.items():
            city = resolver.resolve_city(city_id)
            working_point = state.working_point_for_occurrence(
                (state.element_id, role)
            )
            lambert_xy = (
                working_point.lambert_xy
                if working_point is not None else resolver.get_city_lambert(city_id)
            )
            vertices[role] = DeformationVertex(
                role=role,
                name=city.name,
                lambert_xy=(float(lambert_xy[0]), float(lambert_xy[1])),
            )
        return vertices

    def _deformation_assembly_rotation_deg(self) -> float:
        state = self._deformation_state
        if state.element_id is None:
            raise RuntimeError("Triangle de deformation absent")
        world = state.last_accepted_world or state.reference_world
        if world is None:
            raise RuntimeError("World de deformation absent")
        world_points = self._get_core_triangle_world_points(world, state.element_id)
        vertices = self._deformation_vertices()
        return derive_assembly_view_rotation_deg(
            vertices["O"].lambert_xy,
            vertices["B"].lambert_xy,
            tuple(world_points["O"]),
            tuple(world_points["B"]),
        )

    def _ensure_deformation_window(self) -> DeformationWindow:
        window = self._deformation_window
        if window is not None and window.winfo_exists():
            return window

        calibrated_map = self._resolve_deformation_catalogue_map()
        window = DeformationWindow(
            self,
            calibrated_map=calibrated_map,
            on_vertex_drag_started=self._deformation_window_drag_started,
            on_vertex_dragged=self._deformation_window_dragged,
            on_vertex_drag_released=self._deformation_window_drag_released,
            on_vertex_selected=self._deformation_window_vertex_selected,
            on_occurrence_selected=self._deformation_window_occurrence_selected,
            on_delete_selected=self._deformation_delete_selected,
            on_restore_selected=self._deformation_restore_selected,
            on_pivot_attachment_selected=self._deformation_pivot_attachment_selected,
            on_rename_selected=self._deformation_rename_selected,
            on_map_pin_selected=self._deformation_map_pin_selected,
            on_view_mode_changed=self._deformation_window_view_mode_changed,
            on_geometric_layer_visibility_changed=self._deformation_geometric_layer_visibility_changed,
            on_canvas_mode_changed=self._deformation_canvas_mode_changed,
            on_validate=self._validate_deformation_session,
            on_closed=self._on_deformation_window_closed,
        )
        self._deformation_window = window
        window.set_canvas_mode(self._deformation_canvas_mode)
        return window

    def _resolve_deformation_catalogue_map(self):
        """Résout la référence géographique Catalogue utilisée par DEFORM."""
        map_id = self.catalogue.catalogue_reference_map_id
        if map_id is None:
            raise ValueError("Aucune carte de référence Catalogue n'est définie pour DEFORM.")
        catalogue_map = self.catalogue.get_map(map_id)
        if catalogue_map.archived:
            raise ValueError("La carte de référence Catalogue est archivée.")
        if catalogue_map.projection != "EPSG:2154" or catalogue_map.calibration_file is None:
            raise ValueError("La carte de référence du Catalogue n'est pas calibrée.")
        if self._deformation_map_cache_id != map_id or self._deformation_map_cache is None:
            self._deformation_map_cache = load_calibrated_catalogue_map(
                catalogue_map,
                CatalogueMapAssetResolver(self.paths),
                max_image_dimension=None,
            )
            self._deformation_map_cache_id = map_id
        return self._deformation_map_cache

    def _deformation_canvas_mode_changed(self, mode: str) -> None:
        if mode not in {"select", "move"}:
            raise ValueError(f"Mode canvas DEFORM invalide: {mode!r}")
        if mode == self._deformation_canvas_mode:
            return
        self._cancel_deformation_canvas_interaction()
        self._deformation_canvas_mode = mode

    def _cancel_deformation_canvas_interaction(self) -> None:
        """Annule l'interaction canvas, sans toucher à la session DEFORM."""
        selection = self.__dict__.get("_sel")
        if not isinstance(selection, dict):
            selection = None
        selection_mode = selection.get("mode") if selection is not None else None
        if selection_mode == "move_group":
            if selection.get("auto_geom"):
                self._discard_auto_transform_preview()
            else:
                self._discard_manual_move_preview()
        elif selection_mode in {"rotate_group", "rotate_group_anchor_drag"}:
            if selection.get("auto_geom"):
                self._discard_auto_transform_preview()
            else:
                self._discard_manual_rotate_preview()
        self._sel = None
        self._drag = None
        self._on_pan_end(None)
        self._offset_anchor = None
        self._reset_assist()

    def _refresh_deformation_window(
        self,
        *,
        status_text: str | None = None,
        refresh_occurrences: bool = True,
    ) -> None:
        window = self._ensure_deformation_window()
        state = self._deformation_state
        if status_text is None:
            status_text = self._deformation_status_text
        window.set_validate_enabled(state.dirty)
        if refresh_occurrences:
            window.set_occurrences(
                self._deformation_display_occurrences(),
                state.selected_occurrence,
            )
        pivot_attachment_id = None
        if state.selected_occurrence is not None:
            element_id, role = state.selected_occurrence
            world = state.last_accepted_world or state.reference_world
            if world is not None:
                group_id = world.get_group_of_element(element_id)
                node_id = world.get_element_vertex_node_id_by_type(element_id, role)
                pivot_attachment_id = world.getSingleVertexEdgeAttachmentIdAtNode(
                    group_id, node_id
                )
        window.set_pivot_attachment_enabled(pivot_attachment_id is not None)
        window.set_rename_enabled(self._deformation_selected_occurrence_is_local_city())
        if state.element_id is None:
            return
        window.set_triangle(
            element_id=state.element_id,
            vertices=self._deformation_vertices(),
            assembly_rotation_deg=self._deformation_assembly_rotation_deg(),
            selected_role=(
                state.selected_occurrence[1]
                if state.selected_occurrence is not None
                and state.selected_occurrence[0] == state.element_id
                else None
            ),
            status_text=status_text,
        )
        self._refresh_deformation_geometric_layer()

    def _deformation_display_occurrences(self) -> tuple[tuple[str, str, str, bool, bool], ...]:
        state = self._deformation_state
        world = state.last_accepted_world or state.reference_world
        if world is None:
            return ()
        displays = []
        for element_id, role in sorted(set(state.modified_occurrences)):
            city_id = self._deformation_city_id_for_occurrence(element_id, role, world)
            moved = state.working_point_for_occurrence((element_id, role)) is not None
            group_id = world.get_group_of_element(element_id)
            node_id = world.get_element_vertex_node_id_by_type(element_id, role)
            attachment_id = world.getSingleVertexEdgeAttachmentIdAtNode(group_id, node_id)
            pivoted = attachment_id in state.pivoted_attachment_ids if attachment_id else False
            displays.append((
                element_id, role,
                self._deformation_occurrence_label(element_id, role),
                moved, pivoted,
            ))
        return tuple(displays)

    def _deformation_occurrence_label(self, element_id: str, role: str) -> str:
        state = self._deformation_state
        world = state.last_accepted_world or state.reference_world
        if world is None:
            raise RuntimeError("World de deformation absent")
        element = world.elements.get(element_id)
        if element is None or not element.source_triangle_id:
            raise ValueError("Triangle de deformation sans source Catalogue")
        hypothesis = self._deformation_working_hypothesis()
        resolver = GeometryReferenceResolver(
            self.catalogue, self._deformation_working_reference()
        )
        rank_prefix = "T{}".format(
            hypothesis.get_rank_for_triangle_ref(element.source_triangle_id)
        )
        city_id = self._deformation_city_id_for_occurrence(element_id, role, world)
        return f"{rank_prefix}:{role} - {resolver.resolve_city(city_id).name}"

    def _deformation_selected_occurrence_is_local_city(self) -> bool:
        state = self._deformation_state
        occurrence = state.selected_occurrence
        if occurrence is None:
            return False
        try:
            city_ref_id = self._deformation_city_id_for_occurrence(*occurrence)
        except (RuntimeError, ValueError):
            return False
        return city_ref_id in self._deformation_working_reference().cities

    def _deformation_city_id_for_occurrence(
        self,
        element_id: str,
        role: str,
        world: TopologyWorld | None = None,
    ) -> str:
        if role not in {"O", "B", "L"}:
            raise ValueError(f"Role de deformation inconnu: {role!r}")
        state = self._deformation_state
        source_world = world or state.last_accepted_world or state.reference_world
        if source_world is None:
            raise RuntimeError("World de deformation absent")
        element = source_world.elements.get(element_id)
        if element is None or not element.source_triangle_id:
            raise ValueError("Triangle de deformation sans source Catalogue")
        resolver = GeometryReferenceResolver(
            self.catalogue, self._deformation_working_reference()
        )
        return resolver.city_ref_ids_by_role(element.source_triangle_id)[role]

    def _deformation_occurrences_for_city(
        self,
        city_id: str,
        world: TopologyWorld | None = None,
    ) -> tuple[tuple[str, str], ...]:
        state = self._deformation_state
        source_world = world or state.last_accepted_world or state.reference_world
        if source_world is None:
            raise RuntimeError("World de deformation absent")
        hypothesis = self._deformation_working_hypothesis()
        resolver = GeometryReferenceResolver(
            self.catalogue, self._deformation_working_reference()
        )
        occurrences = []
        for element_id, element in sorted(
            source_world.elements.items(),
            key=lambda item: (
                hypothesis.get_rank_for_triangle_ref(
                    item[1].source_triangle_id
                ),
                item[0],
            ),
        ):
            if not element.source_triangle_id:
                raise ValueError("Triangle de deformation sans source Catalogue")
            for role, occurrence_city_id in resolver.city_ref_ids_by_role(
                element.source_triangle_id
            ).items():
                if occurrence_city_id == city_id:
                    occurrences.append((element_id, role))
        return tuple(occurrences)

    def _deformation_window_drag_started(self, role: str) -> None:
        self._cancel_deformation_drag_tick()
        state = self._deformation_state
        if state.element_id is None:
            raise RuntimeError("Triangle de deformation absent")
        city_id = self._deformation_city_id_for_occurrence(state.element_id, role)
        state.begin_drag(role)
        resolver = GeometryReferenceResolver(
            self.catalogue, self._deformation_working_reference()
        )
        state.ensure_working_point(
            (state.element_id, role),
            resolver.get_city_lambert(city_id),
            self._deformation_occurrences_for_city(city_id),
        )

    def _deformation_window_dragged(self, role: str, lambert_xy: tuple[float, float]) -> None:
        state = self._deformation_state
        if state.dragging_role != role:
            raise RuntimeError("Role de drag DEFORM incoherent")
        if state.reference_world is None or state.element_id is None:
            raise RuntimeError("Etat de deformation incomplet pendant le drag")
        self._deformation_drag_pending_role = role
        self._deformation_drag_pending_point = (float(lambert_xy[0]), float(lambert_xy[1]))
        if self._deformation_drag_after_id is None:
            self._deformation_drag_after_id = self.after(
                DEFORMATION_DRAG_REFRESH_MS,
                self._process_pending_deformation_drag,
            )

    def _process_pending_deformation_drag(self) -> None:
        self._deformation_drag_after_id = None
        role = self._deformation_drag_pending_role
        point = self._deformation_drag_pending_point
        self._deformation_drag_pending_role = None
        self._deformation_drag_pending_point = None
        if role is None or point is None:
            return
        state = self._deformation_state
        if state.dragging_role != role:
            return
        candidate_world = self._apply_deformation_occurrence_overrides(
            state.candidate_occurrence_overrides(point)
        )
        if candidate_world is None:
            self._refresh_deformation_window(
                status_text="Candidat impossible",
                refresh_occurrences=False,
            )
            return
        state.accept_occurrence_candidate(point, candidate_world)
        self._show_deformation_preview(candidate_world)
        self._refresh_deformation_window(refresh_occurrences=False)
        if self._deformation_drag_pending_point is not None:
            self._deformation_drag_after_id = self.after(
                DEFORMATION_DRAG_REFRESH_MS,
                self._process_pending_deformation_drag,
            )

    def _cancel_deformation_drag_tick(self) -> None:
        if self._deformation_drag_after_id is not None:
            self.after_cancel(self._deformation_drag_after_id)
        self._deformation_drag_after_id = None
        self._deformation_drag_pending_role = None
        self._deformation_drag_pending_point = None

    def _deformation_window_drag_released(self, role: str) -> None:
        if self._deformation_state.dragging_role != role:
            raise RuntimeError("Role de release DEFORM incoherent")
        if self._deformation_drag_after_id is not None:
            self.after_cancel(self._deformation_drag_after_id)
            self._deformation_drag_after_id = None
        if self._deformation_drag_pending_point is not None:
            self._process_pending_deformation_drag()
        state = self._deformation_state
        accepted = state.end_occurrence_drag()
        if accepted:
            self.status.config(text="Deformation temporaire conservee.")
        else:
            self.status.config(text="Aucun candidat de deformation valide.")
        self._refresh_deformation_window()

    def _deformation_window_view_mode_changed(self, _mode: str) -> None:
        self._refresh_deformation_window(refresh_occurrences=False)

    def _deformation_geometric_layer_visibility_changed(self, visible: bool) -> None:
        """Charge le calque uniquement à l'activation métier du toggle."""
        self._deformation_geometric_layer_source_triangle_id = None
        window = self._deformation_window
        if not visible:
            if window is not None and window.winfo_exists():
                window.clear_geometric_layer()
            return
        self._refresh_deformation_geometric_layer()

    def _refresh_deformation_geometric_layer(self) -> None:
        """Fournit au widget le document déjà parsé de la Base actuellement affichée."""
        window = self._deformation_window
        if window is None or not window.winfo_exists() or not window.geometric_layer_visible:
            return
        state = self._deformation_state
        world = state.last_accepted_world or state.reference_world
        if state.element_id is None or world is None:
            window.clear_geometric_layer()
            return
        element = world.elements.get(state.element_id)
        if element is None or not element.source_triangle_id:
            raise ValueError("Triangle de deformation sans source Catalogue")
        source_triangle_id = element.source_triangle_id
        if source_triangle_id == self._deformation_geometric_layer_source_triangle_id:
            return
        self._deformation_geometric_layer_source_triangle_id = source_triangle_id
        resolver = GeometryReferenceResolver(self.catalogue, self._deformation_working_reference())
        catalogue_base_city_id = resolve_catalogue_base_city_id_for_deformation_triangle(resolver, source_triangle_id)
        if catalogue_base_city_id is None:
            window.clear_geometric_layer()
            return
        layer = self.catalogue.get_geometric_layer(catalogue_base_city_id)
        if layer is None:
            window.clear_geometric_layer()
            return
        try:
            asset_path = CatalogueGeometricLayerAssetResolver(self.paths).resolve(layer.asset_file)
            document = load_geometric_layer_document(asset_path)
        except (OSError, ValueError) as exc:
            window.clear_geometric_layer()
            messagebox.showerror("Calque géométrique", str(exc), parent=window)
            return
        window.set_geometric_layer(
            document,
            display_overrides=self.catalogue.get_geometric_layer_display_overrides(),
        )

    def _deformation_window_vertex_selected(self, role: str) -> None:
        state = self._deformation_state
        if state.element_id is None:
            return
        state.select_occurrence(state.element_id, role)
        self._refresh_deformation_window()

    def _deformation_window_occurrence_selected(self, element_id: str, role: str) -> None:
        state = self._deformation_state
        if (
            state.selected_occurrence == (element_id, role)
            and state.element_id == element_id
        ):
            return
        state.select_occurrence(element_id, role)
        if state.element_id != element_id:
            if not self._select_deformation_element(element_id):
                return
        self._refresh_deformation_window()

    def _apply_deformation_occurrence_overrides(
        self,
        candidate_overrides: dict[tuple[str, str], tuple[float, float]],
    ) -> TopologyWorld | None:
        """Reconstruit le preview COW depuis le dernier rebase de session."""
        state = self._deformation_state
        if state.reference_world is None:
            raise RuntimeError("World de reference DEFORM absent")
        pivot_result = simulate_deformation_session(
            reference_world=state.reference_world,
            pivoted_attachment_ids=state.pivoted_attachment_ids,
        )
        if not pivot_result.accepted or pivot_result.world is None:
            self._deformation_status_text = ""
            self.status.config(
                text=pivot_result.rejection_reason or "Candidat de pivot impossible"
            )
            return None
        if not candidate_overrides:
            state.working_reference = self._get_active_scenario().reference.clone()
            hypothesis = self._get_active_scenario().hypothesis
            state.working_hypothesis = hypothesis.clone() if hypothesis is not None else None
            self._deformation_status_text = pivot_result.warning_reason or ""
            return pivot_result.world
        scenario = self._get_active_scenario()
        if scenario.hypothesis is None:
            raise ValueError("ScenarioHypothesis absente")
        working_pivot_world = self._materialize_deformation_working_reference(
            pivot_result.world,
            reference=scenario.reference,
            hypothesis=scenario.hypothesis,
        )
        resolver = GeometryReferenceResolver(
            self.catalogue, scenario.reference
        )
        result = simulate_occurrence_deformation(
            resolver=resolver,
            initial_world=working_pivot_world,
            occurrence_lambert_overrides=candidate_overrides,
        )
        if not result.accepted or result.world is None:
            self._deformation_status_text = ""
            self.status.config(text=result.rejection_reason or "Candidat de deformation impossible")
            return None
        candidate_points = {
            point_id: type(working_point)(
                point_id,
                candidate_overrides.get(
                    next(iter(working_point.occurrences)), working_point.lambert_xy
                ),
                set(working_point.occurrences),
            )
            for point_id, working_point in state.working_points.items()
        }
        # Le commit COW est aussi le constructeur Core-first de la reference
        # temporaire : il materialise les STRI/SCITY candidates sans publier le
        # scenario actif. Le world retourne porte donc deja les labels Temp.
        try:
            preview_commit = commit_deformation_copy_on_write(
                catalogue=self.catalogue,
                scenario=scenario,
                preview_world=result.world,
                working_points=candidate_points,
                base_reference=scenario.reference,
                base_hypothesis=scenario.hypothesis,
            )
        except (TypeError, ValueError, TopologyConstraintGeometryError) as exc:
            self._deformation_status_text = ""
            self.status.config(text=f"Candidat de deformation impossible : {exc}")
            return None
        self._apply_deformation_working_point_names(preview_commit)
        state.working_reference = preview_commit.reference
        state.working_hypothesis = preview_commit.hypothesis
        self._deformation_status_text = (
            result.warning_reason or pivot_result.warning_reason or ""
        )
        return preview_commit.world

    def _deformation_delete_selected(self) -> None:
        state = self._deformation_state
        occurrence = state.selected_occurrence
        if occurrence is None:
            return
        restored = state.restore_working_point(occurrence)
        candidate_world = self._apply_deformation_occurrence_overrides(
            state.occurrence_lambert_overrides()
        )
        if candidate_world is None:
            return
        state.last_accepted_world = candidate_world
        state.modified_occurrences = [
            item for item in state.modified_occurrences if item not in restored
        ]
        state.selected_occurrence = None
        self._show_deformation_preview(candidate_world)
        self._refresh_deformation_window()

    def _deformation_restore_selected(self) -> None:
        """Abandonne le WorkingPoint non validé depuis le dernier rebase."""
        state = self._deformation_state
        occurrence = state.selected_occurrence
        if occurrence is None:
            return
        state.restore_working_point(occurrence)
        candidate_world = self._apply_deformation_occurrence_overrides(
            state.occurrence_lambert_overrides()
        )
        if candidate_world is None:
            return
        state.last_accepted_world = candidate_world
        self._show_deformation_preview(candidate_world)
        self._refresh_deformation_window()

    def _deformation_pivot_attachment_selected(self) -> None:
        state = self._deformation_state
        if state.selected_occurrence is None or state.reference_world is None:
            return
        element_id, role = state.selected_occurrence
        world = state.last_accepted_world or state.reference_world
        group_id = world.get_group_of_element(element_id)
        node_id = world.get_element_vertex_node_id_by_type(element_id, role)
        attachment_id = world.getSingleVertexEdgeAttachmentIdAtNode(group_id, node_id)
        if attachment_id is None:
            return
        previous_ids = set(state.pivoted_attachment_ids)
        state.toggle_pivoted_attachment(attachment_id)
        result_world = self._apply_deformation_occurrence_overrides(
            state.occurrence_lambert_overrides()
        )
        if result_world is None:
            state.pivoted_attachment_ids = previous_ids
            state.dirty = bool(state.working_points or state.pivoted_attachment_ids)
            return
        state.last_accepted_world = result_world
        if state.selected_occurrence not in state.modified_occurrences:
            state.modified_occurrences.append(state.selected_occurrence)
        self._show_deformation_preview(result_world)
        self._refresh_deformation_window()

    def _commit_deformation_city_rename(
        self,
        city_ref_id: str,
        new_name: str,
    ) -> None:
        """Publie atomiquement le nouveau nom d'une SCITY et ses labels Core."""
        scenario = self._get_active_scenario()
        candidate_reference = scenario.reference.clone()
        candidate_reference.rename_city(city_ref_id, new_name)
        resolver = GeometryReferenceResolver(self.catalogue, candidate_reference)
        affected_triangle_ref_ids = {
            triangle.triangle_ref_id
            for triangle in candidate_reference.triangles.values()
            if city_ref_id in {
                triangle.opening_city_ref_id,
                triangle.base_city_ref_id,
                triangle.light_city_ref_id,
            }
        }
        candidate_world = scenario.topoWorld.clonePhysicalState()
        for element_id, element in candidate_world.elements.items():
            if element.source_triangle_id in affected_triangle_ref_ids:
                candidate_world.replace_element_materialized_definition(
                    element_id,
                    materialize_triangle(resolver, element.source_triangle_id),
                )
        errors = candidate_world.validate_world()
        if errors:
            raise ValueError(
                "Renommage DEFORM invalide : "
                + " ; ".join(str(error) for error in errors)
            )
        scenario.reference = candidate_reference
        scenario.topoWorld = candidate_world

    def _rename_working_deformation_city(self, city_ref_id: str, new_name: str) -> None:
        """Renomme une SCITY du preview sans publier le scenario actif."""
        state = self._deformation_state
        if state.last_accepted_world is None:
            raise RuntimeError("World de preview DEFORM absent")
        point = (
            state.working_point_for_occurrence(state.selected_occurrence)
            if state.selected_occurrence is not None
            else None
        )
        if point is None:
            resolver = GeometryReferenceResolver(
                self.catalogue, self._deformation_working_reference()
            )
            point = next(
                (
                    working_point
                    for working_point in state.working_points.values()
                    if any(
                        resolver.city_ref_ids_by_role(
                            state.last_accepted_world.elements[element_id].source_triangle_id
                        )[role] == city_ref_id
                        for element_id, role in working_point.occurrences
                    )
                ),
                None,
            )
        if point is None:
            raise RuntimeError("WorkingPoint DEFORM absent")
        candidate_reference = self._deformation_working_reference().clone()
        candidate_reference.rename_city(city_ref_id, new_name)
        resolver = GeometryReferenceResolver(self.catalogue, candidate_reference)
        affected_triangle_ref_ids = {
            triangle.triangle_ref_id
            for triangle in candidate_reference.triangles.values()
            if city_ref_id in {
                triangle.opening_city_ref_id,
                triangle.base_city_ref_id,
                triangle.light_city_ref_id,
            }
        }
        candidate_world = state.last_accepted_world.clonePhysicalState()
        for element_id, element in candidate_world.elements.items():
            if element.source_triangle_id in affected_triangle_ref_ids:
                candidate_world.replace_element_materialized_definition(
                    element_id,
                    materialize_triangle(resolver, element.source_triangle_id),
                )
        errors = candidate_world.validate_world()
        if errors:
            raise ValueError(
                "Renommage DEFORM invalide : "
                + " ; ".join(str(error) for error in errors)
            )
        state.working_reference = candidate_reference
        state.working_point_names[point.point_id] = new_name
        state.last_accepted_world = candidate_world

    def _deformation_rename_selected(self) -> None:
        state = self._deformation_state
        occurrence = state.selected_occurrence
        if occurrence is None:
            return
        scenario = self._get_active_scenario()
        city_ref_id = self._deformation_city_id_for_occurrence(*occurrence)
        city = self._deformation_working_reference().cities.get(city_ref_id)
        if city is None:
            return
        new_name = simpledialog.askstring(
            "Renommer le point",
            "Nouveau nom :",
            initialvalue=city.name,
            parent=self,
        )
        if new_name is None:
            return
        try:
            if state.dirty:
                self._rename_working_deformation_city(city_ref_id, new_name)
            else:
                self._commit_deformation_city_rename(city_ref_id, new_name)
        except ValueError as exc:
            self.status.config(text=f"Renommage DEFORM refusé : {exc}")
            return
        if state.dirty:
            self._show_deformation_preview(state.last_accepted_world)
        else:
            state.working_reference = scenario.reference.clone()
            state.working_hypothesis = (
                scenario.hypothesis.clone() if scenario.hypothesis is not None else None
            )
            self._restore_deformation_real_projection()
        self._refresh_deformation_window()
        self.status.config(text="Point DEFORM renommé.")

    def _deformation_map_pin_selected(self) -> None:
        state = self._deformation_state
        occurrence = state.selected_occurrence
        if occurrence is None:
            return
        source_city_id = self._deformation_city_id_for_occurrence(*occurrence)
        cities = [city for city in self.catalogue.cities.values() if not city.archived]
        target_city_id = CitySelectionDialog(self, cities).show()
        if target_city_id is None:
            return
        occurrences = self._deformation_occurrences_for_city(source_city_id)
        state.set_shared_working_point(
            occurrences, self.catalogue.get_city_lambert(target_city_id)
        )
        working_point = state.working_point_for_occurrence(occurrence)
        if working_point is None:
            raise RuntimeError("WorkingPoint DEFORM absent apres relocalisation")
        target_city = self.catalogue.get_city(target_city_id)
        state.working_point_names[working_point.point_id] = f">> {target_city.name}"
        candidate_world = self._apply_deformation_occurrence_overrides(
            state.occurrence_lambert_overrides()
        )
        if candidate_world is None:
            state.restore_working_point(occurrence)
            return
        state.last_accepted_world = candidate_world
        state.selected_occurrence = occurrence
        self._show_deformation_preview(candidate_world)
        self._refresh_deformation_window()

    def _exit_deformation_mode(self) -> None:
        window = self._deformation_window
        if not self._deformation_state.active and (
            window is None or not window.winfo_exists()
        ):
            return
        self._cancel_deformation_drag_tick()
        self._close_deformation_window()
        self._deformation_state.exit()
        self._sel = None
        self._reset_assist()
        self.canvas.configure(cursor="")
        self._restore_deformation_real_projection()
        self.status.config(text="Mode deformation abandonne.")

    def _validate_deformation_session(self) -> None:
        """Publie le candidat COW puis rebase la session sans fermer DEFORM."""
        state = self._deformation_state
        if not state.dirty:
            self.status.config(text="Aucune modification DEFORM a valider.")
            return
        if state.last_accepted_world is None:
            self.status.config(text="Aucun candidat DEFORM valide.")
            return
        scenario = self._get_active_scenario()
        try:
            if state.working_hypothesis is None:
                committed = commit_deformation_copy_on_write(
                    catalogue=self.catalogue,
                    scenario=scenario,
                    preview_world=state.last_accepted_world,
                    working_points=state.working_points,
                )
                candidate_reference = committed.reference
                candidate_hypothesis = committed.hypothesis
                candidate_world = committed.world
            else:
                candidate_reference = (
                    state.working_reference or scenario.reference
                ).clone()
                candidate_hypothesis = state.working_hypothesis.clone()
                candidate_world = state.last_accepted_world.clonePhysicalState()
                candidate_hypothesis.validate(
                    GeometryReferenceResolver(self.catalogue, candidate_reference)
                )
                errors = candidate_world.validate_world()
                if errors:
                    raise ValueError(
                        "Validation DEFORM invalide : "
                        + " ; ".join(str(error) for error in errors)
                    )
        except (TypeError, ValueError, TopologyConstraintGeometryError) as exc:
            self.status.config(text=f"Validation DEFORM refusee : {exc}")
            return

        # Les trois affectations sont consecutives et ne sont executees qu'apres
        # construction/validation complete des trois candidats ci-dessus.
        scenario.reference = candidate_reference
        scenario.hypothesis = candidate_hypothesis
        scenario.topoWorld = candidate_world
        self._rebuild_triangle_listbox_from_core()
        state.rebase_after_commit(
            scenario.topoWorld, scenario.reference, scenario.hypothesis
        )
        self._show_deformation_preview(scenario.topoWorld)
        self._refresh_deformation_window(status_text="Deformation validee.")
        self.status.config(text="Deformation validee; session rebasee.")

    def _deformation_group_anchor_is_eligible(
        self, world: TopologyWorld, element_id: str
    ) -> tuple[bool, str]:
        """Validate the existing Core GroupAnchor contract without mutating UI state."""
        if not isinstance(world, TopologyWorld):
            # Lightweight controller tests may inject a minimal projection
            # double; production always supplies a TopologyWorld.
            return True, ""
        try:
            group_id = world.get_group_of_element(element_id)
            anchors = [
                anchor for anchor in world.groupAnchors.values()
                if world.find_group(anchor.group_id) == group_id
            ]
            if not anchors:
                return False, (
                    "Déformation impossible : le groupe sélectionné n'est associé "
                    "à aucune balise."
                )
            if len(anchors) != 1:
                return False, (
                    "Déformation impossible : le groupe sélectionné possède une "
                    "ancre ambiguë."
                )
            # applyGroupAnchor is the Core authority for beacon/node resolution.
            # Run it on a physical clone to keep the current preview untouched.
            validation_world = world.clonePhysicalState()
            validation_world.applyGroupAnchor(anchors[0].anchor_id)
        except (KeyError, ValueError, RuntimeError):
            return False, (
                "Déformation impossible : la balise du groupe sélectionné n'est "
                "pas résoluble."
            )
        return True, ""

    def _select_deformation_element(self, element_id: str) -> bool:
        self._cancel_deformation_drag_tick()
        scen = self._get_active_scenario()
        state = self._deformation_state
        current_world = (
            state.last_accepted_world
            or state.reference_world
            or scen.topoWorld.clonePhysicalState()
        )
        element = current_world.elements.get(element_id)
        if element is None:
            self.status.config(text=f"Deformation indisponible : element inconnu {element_id!r}")
            return False
        if not element.source_triangle_id:
            self.status.config(
                text=f"Deformation indisponible : source Catalogue absente pour {element_id!r}"
            )
            return False
        eligible, message = self._deformation_group_anchor_is_eligible(
            current_world, element_id
        )
        if not eligible:
            self.status.config(text=message)
            return False
        state.select(element_id, current_world)
        state.last_accepted_world = current_world
        self._show_deformation_preview(current_world)
        self._refresh_deformation_window()
        self.status.config(text=f"Triangle {element_id} selectionne pour deformation.")
        return True

    def _deformation_refresh_preview_after_rotation(self) -> None:
        state = self._deformation_state
        if state.element_id is None:
            raise RuntimeError("Triangle de deformation absent apres rotation")
        reference_world = state.last_accepted_world or self._get_active_scenario().topoWorld.clonePhysicalState()
        state.replace_reference_world(reference_world)
        result_world = self._apply_deformation_occurrence_overrides(
            state.occurrence_lambert_overrides()
        )
        if result_world is None:
            raise RuntimeError(
                "Les overrides de deformation ne peuvent pas etre rejoues apres rotation"
            )
        state.last_accepted_world = result_world
        self._show_deformation_preview(result_world)
        self._refresh_deformation_window()

    def _deformation_effective_world(self) -> TopologyWorld:
        """Retourne le world candidat affiche, jamais le world publie en priorite."""
        state = self._deformation_state
        return (
            state.last_accepted_world
            or state.reference_world
            or self._get_active_scenario().topoWorld
        )

    def _accept_deformation_main_candidate(
        self, candidate_reference: TopologyWorld, *, changed: bool
    ) -> None:
        """Rejoue DEFORM sur un geste Main sans publier le scenario."""
        state = self._deformation_state
        state.replace_reference_world(candidate_reference)
        result_world = self._apply_deformation_occurrence_overrides(
            state.occurrence_lambert_overrides()
        )
        if result_world is None:
            raise RuntimeError("Le candidat Main DEFORM ne peut pas etre rejoue")
        state.last_accepted_world = result_world
        if changed:
            state.dirty = True
        self._show_deformation_preview(result_world)
        self._refresh_deformation_window()

    def _preview_deformation_group_translation(self, event) -> None:
        """Construit le candidat de translation Main depuis le snapshot DEFORM."""
        selection = self._sel
        if not isinstance(selection, dict):
            raise RuntimeError("Selection MOVE absente en mode deformation")
        core_group_id = selection.get("core_group_id")
        base_world = selection.get("deformation_base_world")
        if core_group_id is None or not isinstance(base_world, TopologyWorld):
            raise RuntimeError("Base DEFORM absente pendant le MOVE")
        delta = self._get_move_drag_delta_world(event)
        candidate_reference = base_world.clonePhysicalState()
        candidate_reference.move_group(
            str(core_group_id), float(delta[0]), float(delta[1])
        )
        self._accept_deformation_main_candidate(
            candidate_reference,
            changed=bool(float(np.linalg.norm(delta)) > 1e-9),
        )

    def _preview_deformation_rotation(self, event) -> None:
        state = self._deformation_state
        if state.element_id is None:
            raise RuntimeError("Etat de deformation incomplet pendant la rotation")
        selection = self._sel
        if not isinstance(selection, dict):
            raise RuntimeError("Selection de rotation absente en mode deformation")
        core_group_id = selection.get("core_group_id")
        if core_group_id is None:
            raise RuntimeError("Groupe Core absent pendant la rotation deformation")
        pivot_world = np.asarray(selection["pivot_world"], dtype=float)
        mouse_world = self._screen_to_world(event.x, event.y)
        angle_delta = self._normalize_rotation_angle(
            self._rotation_angle_from_mouse_world(mouse_world, pivot_world)
            - float(selection["mouse_angle_start"])
        )
        base_world = selection.get("deformation_base_world")
        if not isinstance(base_world, TopologyWorld):
            raise RuntimeError("Base DEFORM absente pendant la rotation")
        preview_reference = base_world.clonePhysicalState()
        preview_reference.rotate_group(core_group_id, pivot_world, angle_delta)
        self._accept_deformation_main_candidate(
            preview_reference,
            changed=bool(abs(angle_delta) > 1e-9),
        )

    def _handle_deformation_left_down(self, event):
        state = self._deformation_state
        self._ensure_pick_cache()
        mode, idx, extra = self._hit_test(event.x, event.y)
        if idx is None:
            self._on_pan_start(event)
            return "break"

        entry = self._last_drawn[idx]
        element_id = str(entry.get("topoElementId", "") or "").strip()
        if not element_id:
            raise ValueError("Triangle projete sans topoElementId")

        if element_id == state.element_id:
            return "break"

        selected = self._select_deformation_element(element_id)
        if selected is not False:
            state.selected_occurrence = None
        return "break"

    def open_catalogue_window(self):
        """Ouvre la fenêtre non modale de gestion du catalogue."""
        self._exit_deformation_mode()
        window = getattr(self, "_catalogue_window", None)
        if window is not None:
            try:
                if window.winfo_exists():
                    window.deiconify()
                    window.lift()
                    window.focus_force()
                    return
            except tk.TclError:
                pass

        window = CatalogueWindow(
            self,
            catalogue=self.catalogue,
            catalogue_path=self.catalogue_path,
            on_catalogue_applied=self._publish_catalogue,
            is_beacon_referenced=self._is_beacon_referenced_by_anchor,
            is_book_referenced=self._is_book_referenced_by_loaded_scenario,
        )
        self._catalogue_window = window

        def _on_close():
            if not window.request_close():
                return
            if getattr(self, "_catalogue_window", None) is window:
                self._catalogue_window = None

        window.protocol("WM_DELETE_WINDOW", _on_close)

    def _update_hypothesis_editor_button(self) -> None:
        """Autorise l'édition uniquement sur le scénario manuel actif."""
        scenario = self._get_active_scenario()
        state = tk.NORMAL if scenario is not None and scenario.source_type == "manual" else tk.DISABLED
        self._ui_hypothesis_editor_button.configure(state=state)

    @staticmethod
    def _format_hypothesis_change_plan(plan: ScenarioHypothesisChangePlan) -> str:
        lines = [
            "Modification de l'hypothèse détectée.",
            "",
            f"Impact : {plan.global_impact.value}",
            f"{len(plan.rank_changes)} rang(s) modifié(s) :",
        ]
        for change in plan.rank_changes[:8]:
            lines.append(
                f"- rang {change.rank} : {change.old_triangle_id} → "
                f"{change.new_triangle_id} — {change.impact.value}"
            )
        if len(plan.rank_changes) > 8:
            lines.append(f"... et {len(plan.rank_changes) - 8} autre(s).")
        lines.extend(("", "La propagation vers la topologie sera prise en charge par la prochaine phase."))
        return "\n".join(lines)

    def _commit_manual_hypothesis_draft(
        self,
        scenario: ScenarioAssemblage,
        draft: ScenarioHypothesis,
        reference: ScenarioReference | None = None,
    ) -> ScenarioHypothesisChangePlan:
        """Analyse puis commit seulement si le monde physique reste cohérent."""
        if scenario.source_type != "manual":
            raise ValueError("Seul un scénario manuel peut recevoir une ScenarioHypothesis modifiée.")
        if scenario.hypothesis is None:
            raise ValueError("ScenarioHypothesis absente du scénario manuel actif")
        candidate_reference = reference or scenario.reference.clone()
        resolver = GeometryReferenceResolver(self.catalogue, candidate_reference)
        draft.validate(resolver)
        if scenario.topoWorld.elements:
            return apply_hypothesis_change_to_manual_scenario(
                self.catalogue, scenario, draft, candidate_reference
            ).plan
        plan = analyze_hypothesis_change(
            resolver,
            scenario.hypothesis,
            draft,
        )
        scenario.reference = candidate_reference.clone()
        scenario.hypothesis = draft.clone()
        return plan

    def open_scenario_hypothesis_dialog(self) -> None:
        """Édite transactionnellement l'hypothèse du manuel actif."""
        self._exit_deformation_mode()
        scenario = self._get_active_scenario()
        if scenario is None:
            raise RuntimeError("Scénario actif absent")
        if scenario.source_type != "manual":
            messagebox.showinfo(
                "Hypothèse du scénario",
                "L'hypothèse d'un scénario AUTO est un snapshot de simulation et ne peut pas être modifiée directement.",
                parent=self,
            )
            return
        if scenario.hypothesis is None:
            raise ValueError("ScenarioHypothesis absente du scénario manuel actif")
        dialog = ScenarioHypothesisDialog(
            self,
            catalogue=self.catalogue,
            hypothesis=scenario.hypothesis,
            resolver=GeometryReferenceResolver(self.catalogue, scenario.reference),
            scenario_reference=scenario.reference,
        )
        result = dialog.show()
        if result is None:
            return
        plan = self._commit_manual_hypothesis_draft(
            scenario, result.hypothesis, result.reference
        )
        self._rebuild_active_projection_from_core()
        self._rebuild_triangle_listbox_from_core()
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Hypothèse du scénario mise à jour ({plan.global_impact.value}).")

    def _publish_catalogue(self, catalogue: Catalogue) -> None:
        """Publie le Catalogue et recale les groupes ancrés via le Core."""
        self._exit_deformation_mode()
        self.catalogue = catalogue
        self.scenario_map_controller.set_catalogue(catalogue)
        self._deformation_map_cache_id = None
        self._deformation_map_cache = None
        self._beacon_world_resolver.set_catalogue(catalogue)
        for scenario in self.scenarios:
            world = scenario.topoWorld
            for anchor in tuple(world.groupAnchors.values()):
                world.applyGroupAnchor(anchor.anchor_id)
            if scenario is self._get_active_scenario():
                self._rebuild_active_projection_from_core()
            elif scenario.source_type == "auto":
                self._project_auto_scenario_from_core(scenario)
        self._refreshCheminsBaliseRefCombo()
        self._redraw_from(self._last_drawn)

    def _is_beacon_referenced_by_anchor(self, beacon_id: str) -> bool:
        """Indique si une balise est encore la cible d'un ancrage runtime."""
        return any(
            anchor.beacon_id == beacon_id
            for scenario in self.scenarios
            for anchor in scenario.topoWorld.groupAnchors.values()
        )

    def _is_book_referenced_by_loaded_scenario(self, book_id: str) -> bool:
        """Indique si un livre est référencé par un scénario runtime chargé."""
        return any(scenario.book_ref_id == book_id for scenario in self.scenarios)

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
        tri_header_separator_height = 5

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
            return int(hdr_h + tri_header_separator_height + lb_h + 18)

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
            tri_minsize_collapsed = max(
                28,
                hdr_h + tri_header_separator_height + 10,
            )
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

        title_lbl = tk.Label(header, text="Catalogue", font=(None, 9, "bold"))
        title_lbl.pack(side=tk.LEFT, anchor="w")
        # Cliquer sur le titre plie/déplie aussi (plus “VS Code”)
        title_lbl.bind("<Button-1>", lambda _e: _toggleTrianglesPanel())
        header.bind("<Button-1>", lambda _e: _toggleTrianglesPanel())

        ttk.Separator(tri_frame, orient="horizontal").pack(
            fill=tk.X,
            pady=(0, 4),
        )

        # Contenu pliable
        self._ui_triangles_content = tk.Frame(tri_frame)
        # (pack conditionnel selon l'état initial)
        if not self._ui_triangles_collapsed.get():
            self._ui_triangles_content.pack(fill=tk.BOTH, expand=True)

        # Barre d'outils du catalogue (extensible avec les futures actions).
        triangles_toolbar = tk.Frame(
            self._ui_triangles_content, bd=0, highlightthickness=0
        )
        triangles_toolbar.pack(anchor="w", padx=6, pady=(0, 2), fill="x")
        self.icon_catalogue = self._load_icon("book.png")
        if self.icon_catalogue is not None:
            catalogue_btn = tk.Button(
                triangles_toolbar,
                image=self.icon_catalogue,
                command=self.open_catalogue_window,
                relief=tk.FLAT,
            )
        else:
            catalogue_btn = tk.Button(
                triangles_toolbar,
                text="C",
                width=2,
                command=self.open_catalogue_window,
                relief=tk.FLAT,
            )
        catalogue_btn.pack(side=tk.LEFT, padx=1)
        self._ui_attach_tooltip(catalogue_btn, "Gestion du catalogue")
        self.icon_hypothesis_props = self._load_icon("props.png")
        hypothesis_btn = tk.Button(
            triangles_toolbar,
            image=self.icon_hypothesis_props,
            command=self.open_scenario_hypothesis_dialog,
            relief=tk.FLAT,
        ) if self.icon_hypothesis_props is not None else tk.Button(
            triangles_toolbar,
            text="H",
            width=2,
            command=self.open_scenario_hypothesis_dialog,
            relief=tk.FLAT,
        )
        hypothesis_btn.pack(side=tk.LEFT, padx=1)
        self._ui_hypothesis_editor_button = hypothesis_btn
        icon_deformation = self._load_icon("vector-triangle.png")
        deformation_btn = tk.Button(
            triangles_toolbar,
            image=icon_deformation,
            text="D" if icon_deformation is None else "",
            width=2 if icon_deformation is None else 0,
            command=self._open_deformation_window,
            relief=tk.FLAT,
        )
        deformation_btn.pack(side=tk.LEFT, padx=1)
        deformation_btn.image = icon_deformation
        self._ui_attach_tooltip(deformation_btn, "Ouvrir DEFORM")
        self._ui_attach_tooltip(hypothesis_btn, "Modifier l'hypothèse du scénario")

        lb_frame = tk.Frame(self._ui_triangles_content, bd=0, highlightthickness=0)
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        # Listbox sans « cadre » (cohérent avec le panneau sans LabelFrame)
        self.listbox = tk.Listbox(
            lb_frame,
            width=34,
            selectmode=tk.EXTENDED,
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

        # Ligne "Guides" : checkbox + swatch couleur
        row_guides = tk.Frame(cb_wrap)
        row_guides.pack(anchor="w", fill="x", pady=(2, 0))
        row_guides.grid_columnconfigure(1, weight=1)

        tk.Checkbutton(
            row_guides, text="Guides",
            variable=self._layerGuidesVisibleVar,
            command=self._on_toggle_guides_layer,
        ).grid(row=0, column=0, sticky="w")

        tk.Frame(row_guides).grid(row=0, column=1, sticky="ew")

        row_guides_right = tk.Frame(row_guides, width=rightColWidth)
        row_guides_right.grid(row=0, column=2, sticky="e")
        row_guides_right.grid_propagate(False)

        self._guides_color_btn = tk.Button(
            row_guides_right,
            text="  ",
            width=2,
            command=self._on_pick_guides_color,
            relief=tk.SUNKEN,
        )
        self._guides_color_btn.pack(side=tk.RIGHT, padx=(0, 4), anchor="e")
        self._update_guides_color_swatch()

        # Ligne "Balises" : visibilité du layer Catalogue.
        row_balises = tk.Frame(cb_wrap)
        row_balises.pack(anchor="w", fill="x", pady=(2, 0))
        row_balises.grid_columnconfigure(1, weight=1)

        def _on_toggle_balises_layer():
            self.setAppConfigValue("uiShowBalisesLayer", bool(self.show_balises_layer.get()))
            self.saveAppConfig()
            self._redraw_from(self._last_drawn)

        tk.Checkbutton(
            row_balises, text="Balises",
            variable=self.show_balises_layer,
            command=_on_toggle_balises_layer,
        ).grid(row=0, column=0, sticky="w")

        # spacer pour pousser la colonne de droite au bord droit
        tk.Frame(row_balises).grid(row=0, column=1, sticky="ew")

        row_balises_right = tk.Frame(row_balises, width=rightColWidth)
        row_balises_right.grid(row=0, column=2, sticky="e")
        row_balises_right.grid_propagate(False)
        tk.Button(
            row_balises_right,
            text=">",
            width=2,
            command=lambda: self._navigate_beacon(+1),
        ).pack(side=tk.RIGHT, padx=(0, 2))

        tk.Button(
            row_balises_right,
            text="<",
            width=2,
            command=lambda: self._navigate_beacon(-1),
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
        self.icon_scen_new = self._load_icon("new.png")
        self.icon_scen_open = self._load_icon("open.png")
        self.icon_scen_props = self._load_icon("props.png")
        self.icon_scen_save = self._load_icon("save.png")
        self.icon_scen_dup = self._load_icon("duplicate.png")
        self.icon_scen_del = self._load_icon("delete.png")

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
        _make_btn(toolbar, self.icon_scen_save,  "S", self._scenario_save,
                  "Enregistrer").pack(side=tk.LEFT, padx=1)
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

        # Liste des scénarios : groupes Manuels / Automatiques et colonne #0.

        # IMPORTANT:
        # - ttk ajoute une indentation (~20px) pour les items enfants => gros espace avant l'icône.
        # - en plus, les colonnes (algo/status/n) réduisent #0 => libellés tronqués.
        style = ttk.Style()
        style.configure("Scenario.Treeview", indent=0)  # <- colle les icônes à gauche

        self.scenario_tree = ttk.Treeview(
            scen_lb_frame,
            show="tree",
            selectmode="browse",
            height=6,
            style="Scenario.Treeview",
        )
        self.scenario_tree.column("#0", width=230, stretch=True, anchor="w")

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

        # Barre d'actions (editer, exporter, recalculer, moteur, supprimer)
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

        self.chemins_export_btn = _make_chemin_btn(
            chemins_toolbar,
            self.icon_scen_save,
            "S",
            self.onExporterCheminsExcel,
            "Exporter en Excel",
        )
        self.chemins_export_btn.pack(side=tk.LEFT, padx=1)

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

        self.chemins_balise_ref_var = tk.StringVar(value="")
        self.chemins_balise_ref_combo = ttk.Combobox(
            chemins_toolbar,
            textvariable=self.chemins_balise_ref_var,
            state="disabled",
            width=24,
            values=[],
        )
        self.chemins_balise_ref_combo.pack(side=tk.RIGHT, padx=(6, 0))
        self.chemins_balise_ref_combo.bind("<<ComboboxSelected>>", self._onCheminsBaliseRefSelected)

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

        iid = sel[0]
        t = self._cheminsTripletByIid[iid]

        # 3 noeuds DSU (IDs) et le groupe ID associé
        nodePrevId = t.nodeA
        nodeCenterId = t.nodeO
        nodeNextId = t.nodeB
        groupId = world.getGroupIdFromConceptNode(nodeCenterId)

        # Construire un snap_target minimal centré sur nodeO
        snapTarget = {
            "nodeId": nodeCenterId,
            "nodeDsu": nodeCenterId,
            "topoGroupId": groupId,
        }

        # --- world coords du node central (obligatoire pour déplacer la clock)
        xO, yO = world.getConceptNodeWorldXY(nodeCenterId, groupId)
        wO = (float(xO), float(yO))
        sx, sy = self._world_to_screen(wO)

        # on positionne la clock sur le noeud
        self.compass_state.anchor_world = np.array(wO, dtype=float)
        self.compass_state.cx, self.compass_state.cy = float(sx), float(sy)
        self._clock_bind_anchor_to_node(
            node_id=nodeCenterId,
            topo_group_id=groupId,
            world_pos=wO,
        )

        # On met à jour les informations d'azimut
        self._clock_arc_auto_from_snap_target(
            snapTarget,
            prevNodeDsu=nodePrevId,
            nextNodeDsu=nodeNextId,
            drag=False,
        )
        self._clock_apply_auto_ref_sync()

    def _onCheminsTreeActivate(self, evt=None) -> None:
        # simple alias si tu veux déclencher seulement au double-clic
        self._onCheminsTreeSelect(evt)

    def _refreshCheminsBaliseRefCombo(self) -> None:
        if not hasattr(self, "chemins_balise_ref_combo"):
            return

        beacons = list(get_geometric_reference_beacon_candidates(self.catalogue))
        option_to_id = {
            f"{self.catalogue.get_city(beacon.city_id).name} ({beacon.beacon_id})": beacon.beacon_id
            for beacon in beacons
        }
        self._chemins_beacon_option_ids = option_to_id
        options = list(option_to_id)
        self.chemins_balise_ref_combo.configure(values=options)
        if not options:
            self.chemins_balise_ref_var.set("")
            self.chemins_balise_ref_combo.configure(state="disabled")
            return

        saved = str(self.getAppConfigValue(_assembleur_io.CFG_KEY_CHEMINS_BEACON_REF, "") or "").strip()
        selected_id = saved if saved in option_to_id.values() else beacons[0].beacon_id
        selected = next(option for option, beacon_id in option_to_id.items() if beacon_id == selected_id)

        self.chemins_balise_ref_var.set(selected)
        self.chemins_balise_ref_combo.configure(state="readonly")

    def _getCheminsBeaconRefId(self) -> str:
        if not hasattr(self, "chemins_balise_ref_var"):
            return ""
        return str(getattr(self, "_chemins_beacon_option_ids", {}).get(
            str(self.chemins_balise_ref_var.get() or "").strip(), ""
        ))

    def _recalculerCheminFromSelection(self, beacon_id: str) -> None:
        scen = self._get_active_scenario()
        tc = scen.topoWorld.topologyChemins
        if not tc.isDefined:
            return

        tc.recalculerChemin(str(beacon_id or ""))
        self.refreshCheminTreeView()

    def _onCheminsBaliseRefSelected(self, _evt=None) -> None:
        beacon_id = self._getCheminsBeaconRefId()
        if not beacon_id:
            return
        self.setAppConfigValue(_assembleur_io.CFG_KEY_CHEMINS_BEACON_REF, beacon_id)
        self.saveAppConfig()
        self._recalculerCheminFromSelection(beacon_id)
        self._clock_apply_auto_ref_sync()

    def refreshCheminTreeView(self) -> None:
        """Rafraîchit la TreeView Chemins depuis world.topologyChemins (lecture seule)."""
        if not hasattr(self, "chemins_tree"):
            return
        self._refreshCheminsBaliseRefCombo()
        tree = self.chemins_tree
        self._cheminsTripletByIid = {}

        # Vider
        for iid in tree.get_children(""):
            tree.delete(iid)

        scen = self._get_active_scenario()
        world = scen.topoWorld
        tc = world.topologyChemins

        self.chemins_edit_btn.configure(state=(tk.NORMAL if tc.isDefined else tk.DISABLED))
        self.chemins_export_btn.configure(state=(tk.NORMAL if tc.isDefined else tk.DISABLED))
        self.chemins_recalc_btn.configure(state=(tk.NORMAL if tc.isDefined else tk.DISABLED))
        self.chemins_engine_btn.configure(state=(tk.NORMAL if tc.isDefined else tk.DISABLED))
        self.chemins_delete_btn.configure(state=(tk.NORMAL if tc.isDefined else tk.DISABLED))
        if not tc.isDefined:
            return

        # --- Colonnes dynamiques (UI) ---
        mesuresSpecs = TopologyCheminTriplet.getMesuresSpecs()
        specsByKey = {str(s.get("key")): dict(s) for s in (mesuresSpecs or []) if s.get("key")}

        allowed = list(specsByKey.keys())
        selected = self.getAppConfigValue("cheminsMeasures", ["angle"]) or ["angle"]
        if isinstance(selected, str):
            selected = [selected]
        selected = [k for k in list(selected) if k in allowed]
        if not selected:
            selected = ["angle"]

        columns = ["triplet"] + selected
        tree.configure(columns=tuple(columns), show="headings")

        # Headings
        tree.heading("triplet", text="Triplet")
        tree.column("triplet", anchor="w", width=280, stretch=True)

        for k in selected:
            lab = str(specsByKey.get(k, {}).get("label", k))
            tree.heading(k, text=lab)
            tree.column(k, anchor="center", width=90, stretch=False)

        def _fmt_angle(v: float) -> str:
            return f"{float(v):.2f}°"

        def _fmt_dist(v: float) -> str:
            return f"{float(v):.2f} km"

        # Remplir
        for t in tc.getTriplets():
            if not t.isGeometrieValide:
                raise RuntimeError("Triplet sans géométrie valide")

            tripletStr = (
                f"{world.getNodeLabel(t.nodeA)} - "
                f"{world.getNodeLabel(t.nodeO)} - "
                f"{world.getNodeLabel(t.nodeB)}"
            )

            values = [tripletStr]
            for k in selected:
                if k == "azOA":
                    values.append(_fmt_angle(t.azOA))
                elif k == "azOB":
                    values.append(_fmt_angle(t.azOB))
                elif k == "angle":
                    values.append(_fmt_angle(t.angleDeg))
                elif k == "distOA":
                    values.append(_fmt_dist(t.distOA_km))
                elif k == "distOB":
                    values.append(_fmt_dist(t.distOB_km))
                else:
                    values.append("")

            iid = tree.insert("", tk.END, values=tuple(values))
            self._cheminsTripletByIid[iid] = t

    def onExporterCheminsExcel(self) -> None:
        scen = self._get_active_scenario()
        if scen is None:
            return
        world = scen.topoWorld
        chemins = world.topologyChemins
        if not chemins.isDefined:
            return
        if not os.path.isdir(self.exports_dir):
            raise FileNotFoundError(f"Répertoire d'export introuvable: {self.exports_dir}")

        scenario_name = str(scen.name or "").strip() or "Scenario"
        beacons = list(get_geometric_reference_beacon_candidates(self.catalogue))
        beacon_options = [
            (beacon.beacon_id, f"{self.catalogue.get_city(beacon.city_id).name} ({beacon.beacon_id})")
            for beacon in beacons
        ]
        resolved_map = self.scenario_map_controller.resolved_map
        CheminsExportDialog(
            self,
            world=world,
            chemins=chemins,
            scenario_name=scenario_name,
            exports_dir=self.exports_dir,
            catalogue=self.catalogue,
            beacon_options=beacon_options,
            selected_beacon_id=self._getCheminsBeaconRefId(),
            map_transform=None if resolved_map is None else resolved_map.transform,
            map_name=None if resolved_map is None else resolved_map.catalogue_map.name,
            beacon_world_resolver=self._beacon_world_resolver.get_world,
            on_success=lambda path: self.status.config(text=f"Chemins exportés : {path}"),
        )

    def onEditerChemin(self) -> None:
        """Prépare l'édition puis applique son résultat au Core."""
        scen = self._get_active_scenario()
        if scen is None:
            return
        world = scen.topoWorld
        chemins = world.topologyChemins
        if not chemins.isDefined:
            return
        snapshot_nodes = tuple(str(node) for node in chemins.borderSnapshotNodes)
        selection_mask = tuple(bool(value) for value in chemins.selectionMask)
        if len(snapshot_nodes) != len(selection_mask):
            raise RuntimeError("Édition du chemin impossible : mask/snapshot incohérents.")
        boundary_orientation = str(world.getBoundaryOrientation(chemins.groupId)).strip().lower()
        if boundary_orientation not in ("cw", "ccw"):
            raise RuntimeError(f"Édition du chemin impossible : boundaryOrientation invalide ({boundary_orientation}).")
        current_orientation = str(chemins.orientationUser).strip().lower()
        if current_orientation not in ("cw", "ccw"):
            raise RuntimeError(f"Édition du chemin impossible : orientationUser invalide ({current_orientation}).")
        allowed_measures = [str(spec.get("key")) for spec in TopologyCheminTriplet.getMesuresSpecs()]
        selected_measures = self.getAppConfigValue("cheminsMeasures", ["angle"]) or ["angle"]
        if isinstance(selected_measures, str):
            selected_measures = [selected_measures]
        selected_measures = [key for key in selected_measures if key in allowed_measures] or ["angle"]
        result = CheminEditDialog(
            self,
            snapshot_nodes=snapshot_nodes,
            selection_mask=selection_mask,
            boundary_orientation=boundary_orientation,
            current_orientation=current_orientation,
            measures_specs=TopologyCheminTriplet.getMesuresSpecs(),
            selected_measures=selected_measures,
            node_label_provider=world.getConceptNodeLabel,
        ).show()
        if result is None:
            return
        self.setAppConfigValue("cheminsMeasures", list(result.selected_measures))
        world.topologyChemins.appliquerEdition(
            result.orientation_user,
            result.selection_mask,
            self._getCheminsBeaconRefId(),
        )
        self.refreshCheminTreeView()

    def onRecalculerChemin(self) -> None:
        """Demande au Core de recalculer le chemin courant puis rafraîchit l'UI."""
        beacon_id = self._getCheminsBeaconRefId()
        self._recalculerCheminFromSelection(beacon_id)

    def onDecryptageEngine(self) -> None:
        """Ouvre ou réactive le déchiffreur, avec ses dépendances explicites."""
        win = getattr(self, "_decryptage_engine_win", None)
        if win is not None:
            try:
                if win.winfo_exists():
                    win.deiconify()
                    win.lift()
                    win.focus_force()
                    return
            except tk.TclError:
                pass

        def _clear_closed_window(closed_window) -> None:
            if self._decryptage_engine_win is closed_window:
                self._decryptage_engine_win = None

        self._decryptage_engine_win = DecryptageEngineWindow(
            self,
            get_config=self.getAppConfigValue,
            set_config=self.setAppConfigValue,
            scenario_provider=self._get_active_scenario,
            dico_provider=lambda: self.dictionary_panel.dictionary,
            decryptor_provider=lambda: self.decryptor,
            icon_loader=self._load_icon,
            icons={
                "new": self.icon_scen_new,
                "props": self.icon_scen_props,
                "delete": self.icon_scen_del,
            },
            on_close=_clear_closed_window,
        )

    def _chemins_edit_selected(self):
        """Compat: redirige vers l'éditeur V6."""
        self.onEditerChemin()

    def _chemins_delete_selected(self):
        """Supprime le chemin courant (Core) après confirmation."""
        scen = self._get_active_scenario()
        if scen is None:
            return
        world = scen.topoWorld
        tc = world.topologyChemins
        if not tc.isDefined:
            self.refreshCheminTreeView()
            return

        if not messagebox.askokcancel("Supprimer le chemin", "Supprimer le chemin ?"):
            return

        tc.supprimerChemin()
        self.refreshCheminTreeView()

    def _update_triangle_listbox_colors(self):
        """
        Met à jour la couleur des entrées de la listbox des triangles
        en fonction de leur utilisation dans le scénario actif.
        Triangles utilisés → grisés, triangles disponibles → noir.
        """
        if not hasattr(self, "listbox"):
            return
        scen = self._get_active_scenario()
        world = scen.topoWorld
        if scen.hypothesis is None:
            raise ValueError("ScenarioHypothesis absente du scénario actif")
        used_ids = world.get_used_source_triangle_ids()
        for idx, triangle_id in enumerate(self._triangle_list_triangle_ids):
            if idx >= self.listbox.size():
                break
            self.listbox.itemconfig(
                idx,
                fg="gray50" if triangle_id in used_ids else "black",
            )

    def _attach_beacon_resolver_to_world(self, world: TopologyWorld | None) -> None:
        """Injecte le résolveur World des balises Catalogue dans un monde Core."""
        if world is None:
            return
        world.attachBeaconResolver(self._beacon_world_resolver)

    def _reapply_scenario_group_anchors(self, scenario: ScenarioAssemblage) -> None:
        """Recale les ancres après la restauration du repère cartographique runtime."""
        world = scenario.topoWorld
        self._attach_beacon_resolver_to_world(world)
        for anchor in world.groupAnchors.values():
            world.applyGroupAnchor(anchor.anchor_id)

    def _rebuild_triangle_listbox_from_core(self) -> None:
        """Projette la selection du scenario dans la listbox, sans lire ``last_drawn``."""
        if not hasattr(self, "listbox"):
            return

        scroll_position = self.listbox.yview()[0]

        scen = self._get_active_scenario()
        if scen.hypothesis is None:
            raise ValueError("ScenarioHypothesis absente du scénario actif")

        self._triangle_list_triangle_ids = list(
            scen.hypothesis.triangle_ids_by_rank
        )

        self.listbox.delete(0, tk.END)
        resolver = GeometryReferenceResolver(self.catalogue, scen.reference)

        for rank, triangle_id in enumerate(
            self._triangle_list_triangle_ids,
            start=1,
        ):
            triangle = resolver.resolve_triangle(triangle_id)
            base = resolver.resolve_city(triangle.base_city_ref_id)
            light = resolver.resolve_city(triangle.light_city_ref_id)

            self.listbox.insert(
                tk.END,
                f"{rank:02d}. B:{base.name}  L:{light.name}",
            )

        self._update_triangle_listbox_colors()

        self.listbox.yview_moveto(scroll_position)

    def _get_triangle_id_from_listbox_index(self, idx: int) -> str:
        if not 0 <= int(idx) < len(self._triangle_list_triangle_ids):
            raise IndexError(f"ScenarioHypothesis: index listbox invalide: {idx}")
        return self._triangle_list_triangle_ids[int(idx)]

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

    def _update_current_scenario_differences(self):
        """Marque les éléments dont les attachments diffèrent de la référence."""
        self._comparison_diff_indices = set()

        ref_scen = self._get_reference_scenario()
        if ref_scen is None:
            return

        if not (0 <= self.active_scenario_index < len(self.scenarios)):
            return

        cur_scen = self.scenarios[self.active_scenario_index]
        if cur_scen is ref_scen:
            return

        new_element_ids = differing_attachment_element_ids(ref_scen.topoWorld, cur_scen.topoWorld)
        self._comparison_diff_indices = {
            idx for idx, tri in enumerate(cur_scen.last_drawn)
            if str(tri.get("topoElementId", "") or "") in new_element_ids
        }

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

    # ---------- AUTO (scénarios automatiques): transform géométrique global ----------

    def _is_active_auto_scenario(self) -> bool:
        scen = self._get_active_scenario()
        return bool(scen is not None and getattr(scen, "source_type", "manual") == "auto")

    def _find_orientation_reference_for_beacon(
        self, scenario: ScenarioAssemblage, beacon_id: str
    ) -> AutoOrientationReference | None:
        """Résout le triangle ancré par L de plus petit rang pour une balise.

        La sélection est intégralement topologique : aucune projection Canvas
        ni proximité géométrique ne participe à la décision.
        """
        world = scenario.topoWorld
        hypothesis = scenario.hypothesis
        if hypothesis is None:
            raise ValueError(
                "Simulation: ScenarioHypothesis absente pour la référence d'orientation"
            )

        candidates: list[AutoOrientationReference] = []
        for anchor in world.groupAnchors.values():
            if anchor.beacon_id != beacon_id:
                continue
            for element_id in world.getGroupElementIds(anchor.group_id):
                element = world.elements[element_id]
                node_l = world.get_element_vertex_node_id_by_type(element_id, "L")
                if node_l != anchor.node_id:
                    continue
                triangle_id = element.source_triangle_id
                if not triangle_id:
                    raise ValueError(
                        "Simulation: source_triangle_id absent pour le triangle de "
                        f"référence {element_id}"
                    )
                try:
                    tri_rank = hypothesis.get_rank_for_triangle_ref(
                        triangle_id
                    )
                except ValueError as exc:
                    raise ValueError(
                        "Simulation: triangle Catalogue absent de l'hypothèse pour la "
                        f"référence {element_id}: {triangle_id!r}"
                    ) from exc
                R, _T, _mirrored = world.getElementPose(element_id)
                candidates.append(
                    AutoOrientationReference(
                        beacon_id=beacon_id,
                        element_id=element_id,
                        tri_rank=tri_rank,
                        theta_rad=math.atan2(float(R[1, 0]), float(R[0, 0])),
                    )
                )
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.tri_rank)
        seen_ranks: set[int] = set()
        for candidate in candidates:
            if candidate.tri_rank in seen_ranks:
                raise RuntimeError(
                    f"Simulation: référence d'orientation ambiguë pour la balise {beacon_id!r}"
                )
            seen_ranks.add(candidate.tri_rank)
        return candidates[0]

    def _anchor_auto_scenario_to_beacon(
        self, scen: ScenarioAssemblage, beacon_id: str
    ) -> None:
        """Ancre le groupe final AUTO par le sommet L de son premier élément."""
        if scen is None or scen.source_type != "auto":
            raise ValueError("Simulation AUTO: scénario invalide pour l'ancrage")
        world = scen.topoWorld
        if world is None:
            raise RuntimeError("Simulation AUTO: TopologyWorld absent")
        self._attach_beacon_resolver_to_world(world)
        if beacon_id not in self.catalogue.beacons:
            raise ValueError(f"Simulation AUTO: balise inconnue {beacon_id!r}")
        beacon = self.catalogue.get_beacon(beacon_id)
        if beacon.archived:
            raise ValueError(f"Simulation AUTO: balise archiv\u00c3\u00a9e {beacon_id!r}")
        if not beacon.usable_as_anchor:
            raise ValueError(f"Simulation AUTO: balise non utilisable comme ancrage {beacon_id!r}")
        ordered_element_ids = scen.orderedElementIds
        if not ordered_element_ids:
            raise ValueError("Simulation AUTO: orderedElementIds vide")
        first_element_id = ordered_element_ids[0]
        if first_element_id not in world.elements:
            raise KeyError(f"Simulation AUTO: premier élément absent {first_element_id!r}")
        core_group_id = world.get_group_of_element(first_element_id)
        if core_group_id is None or not world.hasLiveGroup(core_group_id):
            raise RuntimeError("Simulation AUTO: groupe final canonique absent")
        if any(
            world.get_group_of_element(element_id) != core_group_id
            for element_id in ordered_element_ids
        ):
            raise RuntimeError("Simulation AUTO: plusieurs groupes finaux vivants")
        concept_node_l = world.get_element_vertex_node_id_by_type(first_element_id, "L")
        existing_anchor = world.getAnchorForGroup(core_group_id)
        if existing_anchor is not None:
            raise RuntimeError("Simulation AUTO: ancre concurrente inattendue")
        anchor = world.createGroupAnchor(core_group_id, beacon_id, concept_node_l)
        world.applyGroupAnchor(anchor.anchor_id)
        beacon_world = np.asarray(world.getBeaconWorldXY(beacon_id), dtype=float)
        node_world = np.asarray(
            world.getConceptNodeWorldXY(concept_node_l, core_group_id), dtype=float
        )
        final_anchor = world.getAnchorForGroup(core_group_id)
        if (
            final_anchor is not anchor
            or final_anchor.beacon_id != beacon_id
            or final_anchor.node_id != concept_node_l
            or not np.allclose(node_world, beacon_world, rtol=0.0, atol=1e-9)
        ):
            raise RuntimeError("Simulation AUTO: invariant d'ancrage final rompu")
        self._project_auto_scenario_from_core(scen)

    def _convertActiveAutoToManualSnapshot(self):
        """Convertit le scénario auto actif en un nouveau scénario manuel (snapshot monde), et retourne son index."""
        scen = self._get_active_scenario()
        if scen.source_type != "auto":
            return None

        # Snapshot monde courant, y compris son ancre Core.
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

        name = str(getattr(scen, "name", "") or "Snapshot")
        name = name + " (manuel)" if "manuel" not in name.lower() else name
        if scen.hypothesis is None:
            raise ValueError(
                "Scénario AUTO sans ScenarioHypothesis lors de la conversion"
            )

        new_scen = ScenarioAssemblage(
            name=name,
            source_type="manual",
            hypothesis=scen.hypothesis.clone(),
        )
        new_scen.last_drawn = clone_last_drawn_world(getattr(scen, "last_drawn", None))
        new_scen.topoWorld = scen.topoWorld.clonePhysicalState()
        new_scen.clockRefEdgeId = scen.clockRefEdgeId
        new_scen.clockRefNodeId = scen.clockRefNodeId
        new_scen.clockRefTopoGroupId = scen.clockRefTopoGroupId
        new_scen.clockAzimuthTraits = copy.deepcopy(scen.clockAzimuthTraits)

        # copier quelques métadonnées utiles
        for attr in ("algo_id", "status"):
            if hasattr(scen, attr):
                setattr(new_scen, attr, getattr(scen, attr))

        new_scen.view_state = self._capture_view_state()
        new_scen.map_state = self.scenario_map_controller.capture_active_state()
        new_scen.book_ref_id = scen.book_ref_id

        self.scenarios.append(new_scen)
        return len(self.scenarios) - 1

    def _set_active_scenario(self, index: int):
        """
        Bascule vers le scénario d'index donné.
        Pour l'instant, on n'a qu'un scénario manuel qui partage les mêmes
        structures _last_drawn / groups, mais cette méthode sera utilisée
        plus tard pour les scénarios automatiques (copies séparées).
        """
        self._exit_deformation_mode()
        if index < 0 or index >= len(self.scenarios):
            return
        if index == self.active_scenario_index:
            return

        # MIG-CACHE-TRANSFORM-001B : un aperçu MOVE manuel n'est jamais un
        # état validé ; il doit être restauré avant de quitter son scénario.
        self._discard_manual_move_preview()
        self._discard_manual_rotate_preview()
        self._discard_auto_transform_preview()

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
            self.auto_map_state = self.scenario_map_controller.capture_active_state()
        else:
            prev.map_state = self.scenario_map_controller.capture_active_state()

        scen = self.scenarios[index]
        self.active_scenario_index = index
        self._attach_beacon_resolver_to_world(scen.topoWorld)
        self._reload_dictionary_for_active_scenario(reset_reference=True)

        # Restaurer carte + vue (sans écraser la config globale)
        scenIsAuto = (getattr(scen, "source_type", "manual") == "auto")

        if scenIsAuto:
            self._apply_scenario_map_state(self.auto_map_state, redraw=False)
        else:
            self._apply_scenario_map_state(
                getattr(scen, "map_state", None), redraw=False
            )

        if scenIsAuto:
            # Vue AUTO partagée, fallback si jamais pas encore initialisée
            self._apply_view_state(self.auto_view_state or getattr(scen, "view_state", None))
        else:
            self._apply_view_state(getattr(scen, "view_state", None))

        self._reapply_scenario_group_anchors(scen)

        # MIG-CACHE-REBUILD-003 : l'ancien cache est volontairement ignoré.
        # Les ancres ont été recalées après la restauration de la carte.
        self._rebuild_active_projection_from_core()

        self._rebuild_triangle_listbox_from_core()

        # Invalider le cache de pick et redessiner
        self._invalidate_pick_cache()

        # Fit à l'écran optionnel lors de la sélection d'un scénario
        if self._last_drawn and bool(self.auto_fit_scenario_select.get()):
            # _fit_to_view redessine déjà via _redraw_from()
            self._fit_to_view(self._last_drawn)
        else:
            self._redraw_from(self._last_drawn)

        self._redraw_overlay_only()
        self._update_hypothesis_editor_button()

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
        self._update_compass_ctx_menu_and_dico_state()

        self.status.config(text=f"Scénario actif : {scen.name}")

    def _new_empty_scenario(self):
        """
        Crée un nouveau scénario *vide* (sans triangles assemblés),
        l'ajoute à la liste et le rend actif.
        Les triangles sources (dans la listbox) restent évidemment disponibles.
        """
        # Nom par défaut : "Scénario N" (N = nombre total de scénarios après ajout)
        self._exit_deformation_mode()
        new_index = len(self.scenarios)  # l'index qu'il prendra une fois append
        name = f"Scénario {new_index + 1}"
        hypothesis = self._create_manual_scenario_hypothesis(report_error=True)
        if hypothesis is None:
            return

        scen = ScenarioAssemblage(
            name=name,
            source_type="manual",
            algo_id=None,
            hypothesis=hypothesis,
        )
        self._attach_beacon_resolver_to_world(scen.topoWorld)
        # Scénario vide : nouvelles structures indépendantes
        scen.last_drawn = []
        scen.view_state = self._capture_view_state()
        scen.map_state = self.scenario_map_controller.new_default_state()
        scen.book_ref_id = self.catalogue.default_book_id

        self.scenarios.append(scen)
        # Bascule sur ce nouveau scénario
        self._set_active_scenario(new_index)
        # Rafraîchir la liste visible
        self._refresh_scenario_listbox()

        self.status.config(text=f"Nouveau scénario créé : {scen.name}")

    def _scenario_property_book_choices(self, scen: ScenarioAssemblage) -> tuple[tuple[str, str], ...]:
        """Retourne les livres sélectionnables, avec le livre archivé courant conservé."""
        book_ref_id = scen.book_ref_id
        if book_ref_id is None:
            raise ValueError("Le scénario ne référence aucun livre Catalogue.")
        try:
            current = self.catalogue.get_book(book_ref_id)
        except KeyError as exc:
            raise ValueError(
                f"Le livre {book_ref_id} référencé par ce scénario est absent du Catalogue."
            ) from exc
        choices = [book for book in self.catalogue.iter_books() if not book.archived]
        if current.archived:
            choices.append(current)
        return tuple((book.book_id, book.name) for book in choices)

    def _apply_scenario_book_selection(self, scen: ScenarioAssemblage, book_id: str) -> bool:
        """Valide et applique un livre ; reconstruit le dictionnaire du scénario actif."""
        book = self.catalogue.get_book(book_id)
        if book.archived and book.book_id != scen.book_ref_id:
            raise ValueError(f"Le livre Catalogue {book_id} est archivé.")
        if book_id == scen.book_ref_id:
            return False
        scen.book_ref_id = book_id
        if scen is self._get_active_scenario():
            self._reload_dictionary_for_active_scenario(reset_reference=True)
        return True

    def _scenario_edit_properties(self):
        """Edite transactionnellement le nom, la carte et le livre du scénario actif."""
        if not self.scenarios:
            return
        idx = self.active_scenario_index
        if idx < 0 or idx >= len(self.scenarios):
            return
        scen = self.scenarios[idx]
        state = (
            scen.map_state
            if isinstance(scen.map_state, ScenarioMapState)
            else self.scenario_map_controller.new_default_state()
        )
        maps = [item for item in self.catalogue.iter_maps() if not item.archived or item.map_id == state.map_ref_id]
        labels = {
            f"{item.name}{' (archivée)' if item.archived else ''}": item.map_id
            for item in maps
        }
        if not labels:
            return
        current_label = next((label for label, map_id in labels.items() if map_id == state.map_ref_id), None)
        book_choices = self._scenario_property_book_choices(scen)
        book_labels = {name: book_id for book_id, name in book_choices}
        current_book_label = next(
            (name for book_id, name in book_choices if book_id == scen.book_ref_id),
            None,
        )
        if current_book_label is None:
            raise ValueError(
                f"Le livre {scen.book_ref_id} référencé par ce scénario est absent du Catalogue."
            )
        dialog = tk.Toplevel(self)
        dialog.title("Propriétés du scénario")
        dialog.transient(self)
        dialog.resizable(False, False)
        name_var = tk.StringVar(value=scen.name)
        map_var = tk.StringVar(value=current_label or next(iter(labels)))
        book_var = tk.StringVar(value=current_book_label)
        ttk.Label(dialog, text="Nom").grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=34)
        name_entry.grid(row=0, column=1, padx=(0, 12), pady=(12, 6))
        ttk.Label(dialog, text="Carte").grid(row=1, column=0, padx=12, pady=6, sticky="w")
        combo = ttk.Combobox(dialog, textvariable=map_var, values=tuple(labels), state="readonly", width=31)
        combo.grid(row=1, column=1, padx=(0, 12), pady=6)
        ttk.Label(dialog, text="Livre").grid(row=2, column=0, padx=12, pady=6, sticky="w")
        book_combo = ttk.Combobox(
            dialog,
            textvariable=book_var,
            values=tuple(book_labels),
            state="readonly",
            width=31,
        )
        book_combo.grid(row=2, column=1, padx=(0, 12), pady=6)
        result = {"ok": False}

        def accept():
            cleaned = name_var.get().strip()
            if not cleaned:
                messagebox.showerror("Propriétés du scénario", "Le nom du scénario ne peut pas être vide.", parent=dialog)
                return
            selected_book_id = book_labels.get(book_var.get())
            if selected_book_id is None:
                messagebox.showerror("Propriétés du scénario", "Le livre sélectionné est invalide.", parent=dialog)
                return
            try:
                selected_book = self.catalogue.get_book(selected_book_id)
            except ValueError as exc:
                messagebox.showerror("Propriétés du scénario", str(exc), parent=dialog)
                return
            if selected_book.archived and selected_book.book_id != scen.book_ref_id:
                messagebox.showerror("Propriétés du scénario", "Le livre sélectionné est archivé.", parent=dialog)
                return
            result.update(
                ok=True,
                name=cleaned,
                map_id=labels[map_var.get()],
                book_id=selected_book_id,
            )
            dialog.destroy()
        buttons = ttk.Frame(dialog)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", padx=12, pady=(8, 12))
        ttk.Button(buttons, text="Annuler", command=dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="OK", command=accept).pack(side=tk.RIGHT, padx=(0, 6))
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.grab_set()
        name_entry.focus_set()
        self.wait_window(dialog)
        if not result["ok"]:
            return
        scen.name = result["name"]
        if result["map_id"] != state.map_ref_id:
            state = ScenarioMapState(map_ref_id=result["map_id"], visible=state.visible)
            scen.map_state = state
            self._apply_scenario_map_state(state)
        self._apply_scenario_book_selection(scen, result["book_id"])
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
        if src.hypothesis is None:
            raise ValueError("Scénario: ScenarioHypothesis absente lors de la duplication")

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
            hypothesis=src.hypothesis.clone(),
        )
        # copies indépendantes
        dup.reference = src.reference.clone()
        dup.last_drawn = copy.deepcopy(src.last_drawn)
        dup.groups = copy.deepcopy(src.groups)
        dup.status = src.status
        dup.traversal_direction = getattr(src, "traversal_direction", None)
        dup.topoWorld = src.topoWorld.clonePhysicalState()
        dup.clockRefEdgeId = src.clockRefEdgeId
        dup.clockRefNodeId = src.clockRefNodeId
        dup.clockRefTopoGroupId = src.clockRefTopoGroupId
        dup.clockAzimuthTraits = copy.deepcopy(src.clockAzimuthTraits)
        # Le duplicat devient actif immédiatement : capturer le contexte
        # runtime courant plutôt que les snapshots potentiellement périmés de src.
        dup.view_state = self._capture_view_state()
        dup.map_state = self.scenario_map_controller.capture_active_state()
        dup.book_ref_id = src.book_ref_id

        self.scenarios.append(dup)
        self._refresh_scenario_listbox()
        self._set_active_scenario(new_index)
        self.status.config(text=f"Scénario dupliqué : {dup.name}")

    def _scenario_delete(self):
        """
        Supprime le scénario actif.
        On interdit la suppression du scénario manuel de base pour garder un point d'appui.
        """
        self._exit_deformation_mode()
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

        manual_count = sum(
            1
            for scenario in self.scenarios
            if scenario.source_type == "manual"
        )
        if scen.source_type == "manual" and manual_count <= 1:
            messagebox.showinfo("Supprimer le scénario",
                                "Impossible de supprimer le dernier scénario manuel.")
            return

        if not messagebox.askyesno(
            "Supprimer le scénario",
            f"Supprimer le scénario « {scen.name} » ?",
            parent=self,
        ):
            return

        # Activer le remplaçant tant que le scénario courant est encore dans
        # la collection. Cela évite tout état observable où l'index actif
        # désigne un scénario déjà supprimé.
        replacement_index = idx - 1 if idx == len(self.scenarios) - 1 else idx + 1
        self._set_active_scenario(replacement_index)

        # Anticiper le décalage avant la mutation de liste : après ``pop``,
        # cet index désignera encore le remplaçant, y compris s'il était situé
        # après le scénario supprimé.
        self.active_scenario_index = (
            replacement_index - 1 if replacement_index > idx else replacement_index
        )
        self.scenarios.pop(idx)
        self._refresh_scenario_listbox()
        self.status.config(text=f"Scénario supprimé : {scen.name}")

    def _build_canvas(self, parent):
        # Conteneur de droite : canvas (haut, expansible) + panel dico (bas, hauteur fixe)
        self.rightPane = tk.Frame(parent)
        self.rightPane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas d’affichage des triangles
        self.canvas = tk.Canvas(self.rightPane, bg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.background_map_layer.attach_canvas(self.canvas)
        self.compass_controller.attach_canvas(self.canvas)

        # Redessiner l'overlay si la taille du canvas change
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Panel dico (placeholder à hauteur fixe, prêt pour intégrer la grille)
        exclude_coded_words = self.getAppConfigValue(CFG_KEY_DICO_EXCLURE_MOTS_CODES, False)
        if not isinstance(exclude_coded_words, bool):
            raise ValueError(
                f"Invalid config type for {CFG_KEY_DICO_EXCLURE_MOTS_CODES}: {type(exclude_coded_words).__name__}"
            )
        self.dictionary_panel = DictionaryPanel(
            self.rightPane,
            exclude_coded_words=exclude_coded_words,
            icon_loader=self._load_icon,
            on_exclude_coded_words_changed=self._on_dictionary_exclusion_changed,
            clock_state_resolver=self._resolve_clock_state_from_dictionary_cell,
            on_clock_state_changed=self._apply_clock_state_from_dictionary,
            on_status=lambda text: self.status.config(text=text),
            height=self.dico_panel_height,
        )
        if self.show_dico_panel.get():
            self.dictionary_panel.pack(side=tk.BOTTOM, fill=tk.X)

        # Menu contextuel COMPAS (clic droit sur le compas)
        self._ctx_menu_compass = tk.Menu(self, tearoff=0)
        self._ctx_menu_compass.add_command(label="Définir l'azimut de ref…",
                                           command=self._ctx_define_clock_ref_azimuth)
        self._ctx_menu_compass.add_checkbutton(
            label="Synchroniser auto l'azimut de ref",
            variable=self._clock_auto_ref_sync_var,
            command=self._ctx_toggle_clock_auto_ref_sync,
        )
        self._ctx_menu_compass.add_command(label="Tracer un azimut…",
                                           command=self._ctx_trace_clock_azimuth)
        self._ctx_menu_compass.add_command(label="Mesurer un arc d'angle…",
                                           command=self._ctx_measure_clock_arc_angle)
        self._ctx_menu_compass.add_separator()
        self._ctx_menu_compass.add_command(label="Filtrer le dictionnaire…",
                                           command=self._ctx_filter_dictionary_by_clock_arc,
                                           state=tk.DISABLED)
        self._ctx_compass_idx_filter_dico = self._ctx_compass_find_entry_index("Filtrer le dictionnaire…")
        self._ctx_menu_compass.add_command(label="Annuler le filtrage",
                                           command=self._simulation_cancel_dictionary_filter,
                                           state=tk.DISABLED)
        self._ctx_compass_idx_cancel_dico_filter = self._ctx_compass_find_entry_index("Annuler le filtrage")
        self._ctx_compass_idx_clear_traits = None
        self._update_compass_ctx_menu_and_dico_state()

        # Menu contextuel
        self._ctx_menu = tk.Menu(self, tearoff=0)
        self._ctx_degrouper_label = "Dégrouper"
        self._ctx_pivot_attachment_label = "Pivoter l'attache"
        self._ctx_menu.add_command(label="Supprimer",
                                   command=self._ctx_delete_group)
        self._ctx_menu.add_command(label="Pivoter",
                                   command=self._ctx_rotate_selected)
        self._ctx_menu.add_command(label="Inverser",
                                   command=self._ctx_flip_selected)
        self._ctx_menu.add_command(label="Filtrer les scénarios…",
                                   command=self._ctx_filter_scenarios)
        self._ctx_menu.add_command(label="OL=0°",
                                   command=self._ctx_orient_OL_north)
        self._ctx_menu.add_command(label="BL=0°",
                                   command=self._ctx_orient_BL_north)
        self._ctx_menu.add_separator()
        self._ctx_menu.add_command(label="Créer un chemin…",
                                   command=self._ctx_CreerChemin)

        # Les index sont dynamiques (l'entrée "Dégrouper" peut apparaître/disparaître).
        self._ctx_idx_ol0 = None
        self._ctx_idx_bl0 = None

        # Séparateur et zone dynamique pour les actions "mot"
        self._ctx_refresh_menu_runtime_indexes()

        # Export TopoDump (manuel, snapshot volontaire) et toggle diagnostic.
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
        """Affiche ou masque le panneau dictionnaire."""
        if bool(self.show_dico_panel.get()):
            if not self.dictionary_panel.winfo_ismapped():
                self.dictionary_panel.pack(side=tk.BOTTOM, fill=tk.X)
        else:
            self.dictionary_panel.pack_forget()
        self.setAppConfigValue("uiShowDicoPanel", bool(self.show_dico_panel.get()))

    def _resolve_active_scenario_book_path(self) -> str:
        scenario = self._get_active_scenario()
        if scenario.book_ref_id is None:
            raise ValueError("Le scénario actif ne référence aucun livre Catalogue.")
        return str(CatalogueBookAssetResolver(self.paths).resolve(self.catalogue.get_book(scenario.book_ref_id)))

    def _reload_dictionary_for_active_scenario(self, *, reset_reference: bool) -> None:
        self.dictionary_panel.load_book(self._resolve_active_scenario_book_path(), reset_reference=reset_reference)

    def _on_dictionary_exclusion_changed(self, value: bool) -> None:
        self.setAppConfigValue(CFG_KEY_DICO_EXCLURE_MOTS_CODES, value)
        self.saveAppConfig()

    def _resolve_clock_state_from_dictionary_cell(self, *, row, col, word, mode):
        return self.decryptor.clockStateFromDicoCell(row=row, col=col, word=word, mode=mode)

    def _apply_clock_state_from_dictionary(self, state) -> None:
        self.compass_state.clock.update({"hour": float(state.hour), "minute": state.minute, "label": state.label})
        self._redraw_overlay_only()

    def _toggle_only_group_contours(self):
        """Toggle: afficher uniquement les contours des groupes."""
        self.setAppConfigValue("uiShowOnlyGroupContours", bool(self.show_only_group_contours.get()))
        # Un mode purement visuel -> redraw complet
        self._redraw_from(self._last_drawn)

    def _clock_change_radius(self, delta: int):
        """Modifie le rayon du compas (min=50) et redessine l'overlay."""
        self.compass_controller.change_radius(delta)
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
            self.background_map_layer.cancel_interaction()
            self.canvas.configure(cursor="")
            self._update_background_map_scale_status()

        self._redraw_from(self._last_drawn)

    def _update_background_map_scale_status(self) -> None:
        if not bool(self.bg_resize_mode.get()) or not self.background_map_layer.has_map:
            return
        self.status.config(
            text=f"Échelle carte : {format_scale(self.scenario_map_controller.scale_factor)}"
        )

    def _apply_scenario_map_state(
        self, state: ScenarioMapState, *, redraw: bool = True
    ) -> None:
        self.scenario_map_controller.apply_state(state)
        self.show_map_layer.set(state.visible)
        if redraw:
            self._redraw_from(self._last_drawn)

    def _on_background_map_geometry_changed(self) -> None:
        self.scenario_map_controller.sync_active_state_from_background()

    def _toggle_layers(self):
        """Redessine le canvas suite à un changement de visibilité d'un layer."""
        self.scenario_map_controller.set_active_visibility(bool(self.show_map_layer.get()))
        self._redraw_from(self._last_drawn)

    def _navigate_beacon(self, direction: int) -> None:
        """Centre la vue sur la balise Catalogue précédente ou suivante."""
        if direction not in (-1, 1):
            raise ValueError(f"Direction de navigation balise invalide: {direction!r}")

        beacons = [
            beacon
            for beacon in self.catalogue.iter_beacons()
            if not beacon.archived
        ]
        if not beacons:
            self.status.config(text="Aucune balise disponible.")
            return

        if self._beacon_navigation_index is None:
            index = 0 if direction > 0 else len(beacons) - 1
        else:
            index = (self._beacon_navigation_index + direction) % len(beacons)

        beacon = beacons[index]
        wx, wy = self._beacon_world_resolver.get_world(beacon.beacon_id)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        self.offset = np.array(
            [
                canvas_width / 2.0 - float(wx) * self.zoom,
                canvas_height / 2.0 + float(wy) * self.zoom,
            ],
            dtype=float,
        )

        self._beacon_navigation_index = index

        self._invalidate_pick_cache()
        self._redraw_from(self._last_drawn)

        city = self.catalogue.get_city(beacon.city_id)
        self.status.config(
            text=f"Balise {city.name} ({beacon.beacon_id}) centrée."
        )

    def _on_toggle_guides_layer(self):
        self._layerGuidesVisible = bool(self._layerGuidesVisibleVar.get())
        self.setAppConfigValue("uiShowGuidesLayer", bool(self._layerGuidesVisible))
        self._redraw_from(self._last_drawn)

    def _update_guides_color_swatch(self):
        if self._guides_color_btn is None:
            return
        self._guides_color_btn.configure(
            bg=self._guidesCurrentColorHex,
            activebackground=self._guidesCurrentColorHex,
        )

    def _on_pick_guides_color(self):
        _rgb, hex_color = colorchooser.askcolor(color=self._guidesCurrentColorHex, parent=self)
        if not hex_color:
            return
        self._guidesCurrentColorHex = str(hex_color)
        self._update_guides_color_swatch()

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

    def _ctx_toggle_clock_auto_ref_sync(self):
        self._clock_auto_ref_sync_enabled = bool(self._clock_auto_ref_sync_var.get())
        self.setAppConfigValue("uiClockAutoRefSyncEnabled", bool(self._clock_auto_ref_sync_enabled))
        if self._clock_auto_ref_sync_enabled:
            self._clock_apply_auto_ref_sync()

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
    # --- conversions utilitaires (utilisées par le pick) ---

    def _screen_to_world(self, x, y):
        Z = float(self.zoom)
        Ox, Oy = (float(self.offset[0]), float(self.offset[1]))
        return ((x - Ox) / Z, (Oy - y) / Z)

    # =========================
    #  SCENARIO: SAVE / LOAD
    # =========================

    def _save_active_scenario_to_path(self, path: str) -> None:
        """Sauvegarde le scénario actif et publie son identité de fichier runtime."""
        self.save_scenario_xml(path)
        scenario = self._get_active_scenario()
        if scenario is None:
            raise RuntimeError("Sauvegarde scénario: scénario actif absent après export")
        scenario.file_path = str(path)
        scenario.name = os.path.splitext(os.path.basename(str(path)))[0] or scenario.name
        scenario.is_placeholder = False
        self._refresh_scenario_listbox()
        self.status.config(text=f"Scénario enregistré dans {path}")

    def _scenario_save(self):
        """Enregistre directement le scénario associé à un fichier, sinon délègue à « sous »."""
        scenario = self._get_active_scenario()
        if scenario is None:
            return
        if scenario.file_path:
            self._save_active_scenario_to_path(scenario.file_path)
            return
        self._scenario_save_as_dialog()

    def _scenario_save_as_dialog(self):
        """Boîte de dialogue « Enregistrer sous » pour le scénario actif."""
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

        self._save_active_scenario_to_path(path)

    # Compatibilité interne pour les anciens appels non UI.
    _scenario_save_dialog = _scenario_save_as_dialog

    def _load_scenario_into_new_scenario(self, path: str):
        """
        Charge un fichier scénario XML dans un *nouveau* scénario.
        - Le scénario courant n'est pas modifié.
        - Le nouveau scénario porte le nom du fichier XML (sans extension).
        """
        # Nom lisible dérivé du fichier
        self._exit_deformation_mode()
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        if not name:
            name = base

        previous_scenario = self._get_active_scenario()
        new_index = len(self.scenarios)

        scen = ScenarioAssemblage(
            name=name,
            source_type="manual",
            algo_id=None,
            hypothesis=self._create_manual_scenario_hypothesis(report_error=True),
        )
        # Structures indépendantes pour ce scénario
        scen.last_drawn = []
        scen.view_state = self._capture_view_state()
        loaded = _assembleur_io._parse_loaded_scenario_xml(self, path)
        _assembleur_io._publish_loaded_scenario_xml(
            self, scen, loaded, publish_ui=False,
        )
        self.scenarios.append(scen)

        # Le scénario est intégralement publié avant toute résolution/redraw.
        self._set_active_scenario(new_index)
        _assembleur_io._publish_loaded_scenario_xml(
            self, scen, loaded, publish_ui=True,
        )
        scen.file_path = str(path)
        scen.name = name
        scen.is_placeholder = False

        # Le canevas initial ne doit pas rester comme scénario utilisateur
        # après le chargement réussi d'un fichier. Les autres scénarios sont
        # conservés sans exception.
        if previous_scenario is not None and previous_scenario.is_placeholder:
            self.scenarios.remove(previous_scenario)
            # ``scen`` est déjà affiché : corriger l'index sans repasser par
            # _set_active_scenario(), qui lirait l'ancien index devenu hors
            # bornes après le retrait du placeholder.
            self.active_scenario_index = self.scenarios.index(scen)

        # On se ajoute un fit to screen
        self._fit_to_view(self._last_drawn)

        # On repositionne le compas au cntre de la figure
        canvasW = self.canvas.winfo_width()
        canvasH = self.canvas.winfo_height()
        cxScreen = canvasW / 2.0
        cyScreen = canvasH / 2.0
        cxWorld, cyWorld = self._screen_to_world(cxScreen, cyScreen)
        self._clock_clear_anchor_binding()
        self.compass_state.anchor_world = np.array([cxWorld, cyWorld], dtype=float)
        self.compass_state.cx = float(cxScreen)
        self.compass_state.cy = float(cyScreen)

        self._draw_clock_overlay()
        self._redraw_overlay_only()

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
            m.add_command(label="(aucun scénario utilisateur)", state="disabled")
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

    def _is_in_clock(self, x: float, y: float, pad: float = 10) -> bool:
        return self.compass_controller.contains_point(x, y, pad=pad)
    def _ui_attach_tooltip(self, widget, text: str):
        if widget is None:
            return None
        return attach_tooltip(widget, text)

    def _ensure_canvas_tooltip(self, text: str) -> None:
        if self._tooltip is None or not self._tooltip.winfo_exists():
            self._tooltip = tk.Toplevel(self)
            self._tooltip.withdraw()
            self._tooltip.wm_overrideredirect(True)
            self._tooltip.attributes("-topmost", True)

            self._tooltip_label = tk.Label(
                self._tooltip,
                text=text,
                bg="#ffffe0",
                relief="solid",
                borderwidth=1,
                font=("Arial", 9),
                justify="left",
                anchor="w",
            )
            self._tooltip_label.pack(ipadx=4, ipady=2)

        self._tooltip_label.config(text=text, justify="left", anchor="w")
        self._tooltip.update_idletasks()

    def _show_tooltip_at_center(self, text: str, cx_canvas: float, cy_canvas: float):
        """Affiche/MAJ le tooltip en le **centrant** sur (cx_canvas, cy_canvas) (coords CANVAS)."""
        if not text:
            self._hide_tooltip()
            return

        self._ensure_canvas_tooltip(text)
        # Mesurer la taille réelle du tooltip
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
        self._tooltip.deiconify()

    def _isTooltipRectValidForNode(
        self,
        x: int,
        y: int,
        tipW: int,
        tipH: int,
        nodeCx: float,
        nodeCy: float,
        rExcl: int,
    ) -> bool:
        cw = int(self.canvas.winfo_width())
        ch = int(self.canvas.winfo_height())

        if x < 0 or y < 0:
            return False
        if x + tipW > cw or y + tipH > ch:
            return False

        exLeft = float(nodeCx) - float(rExcl)
        exTop = float(nodeCy) - float(rExcl)
        exRight = float(nodeCx) + float(rExcl)
        exBottom = float(nodeCy) + float(rExcl)

        tipLeft = float(x)
        tipTop = float(y)
        tipRight = float(x + tipW)
        tipBottom = float(y + tipH)

        intersects = not (
            tipRight <= exLeft
            or tipLeft >= exRight
            or tipBottom <= exTop
            or tipTop >= exBottom
        )
        return not intersects

    def _computeNodeTooltipCanvasPosition(
        self,
        nodeCx: float,
        nodeCy: float,
        tipW: int,
        tipH: int,
    ) -> tuple[int, int] | None:
        cw = int(self.canvas.winfo_width())
        ch = int(self.canvas.winfo_height())
        rExcl = int(self._marker_px)
        anchorGap = int(self._marker_px + self._tooltip_cushion_px)

        if tipW > cw or tipH > ch:
            return None

        candidates = [
            (int(nodeCx - anchorGap - tipW), int(nodeCy - tipH // 2)),
            (int(nodeCx - tipW // 2), int(nodeCy - anchorGap - tipH)),
            (int(nodeCx + anchorGap), int(nodeCy - anchorGap - tipH)),
            (int(nodeCx - anchorGap - tipW), int(nodeCy + anchorGap)),
            (int(nodeCx - tipW // 2), int(nodeCy + anchorGap)),
            (int(nodeCx + anchorGap), int(nodeCy + anchorGap)),
        ]

        for x, y in candidates:
            if self._isTooltipRectValidForNode(x, y, tipW, tipH, nodeCx, nodeCy, rExcl):
                return x, y

        return None

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

    def _triangle_from_index(self, idx):
        """Construit l'aperçu local depuis le modèle du scénario actif."""
        scen = self._get_active_scenario()
        if scen.hypothesis is None:
            raise ValueError("ScenarioHypothesis absente du scénario actif")
        triangle_id = self._get_triangle_id_from_listbox_index(idx)
        element = materialize_triangle(
            GeometryReferenceResolver(self.catalogue, scen.reference), triangle_id
        )
        points = element.vertex_local_xy
        return {
            "labels": tuple(element.vertex_labels),
            "pts": {
                "O": np.array(points[0], dtype=float),
                "B": np.array(points[1], dtype=float),
                "L": np.array(points[2], dtype=float),
            },
            "triangle_id": triangle_id,
            "mirrored": False,
        }

    def _build_drag_world_points(
        self,
        triangle_id: str,
        mouse_world: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        """Construit le preview temporaire depuis la référence effective."""
        scen = self._get_active_scenario()
        element = materialize_triangle(
            GeometryReferenceResolver(self.catalogue, scen.reference), triangle_id
        )
        local_points = element.vertex_local_xy
        origin = np.asarray(local_points[0], dtype=float)
        delta = np.asarray(mouse_world, dtype=float) - origin
        return {
            "O": np.asarray(local_points[0], dtype=float) + delta,
            "B": np.asarray(local_points[1], dtype=float) + delta,
            "L": np.asarray(local_points[2], dtype=float) + delta,
        }

    # ====== FRONTIER GRAPH HELPERS (factorisation) ===============================================
    def _ang_of_vec(self, vx, vy):
        import math
        return math.atan2(vy, vx)

    def _ang_diff(self, a, b):
        # plus petit écart absolu d’angle
        return abs(self._ang_wrap(a - b))

    def _refresh_listbox_from_df(self):
        """Compatibilité : la listbox est désormais projetée depuis le Core."""
        self._rebuild_triangle_listbox_from_core()

    def clear_canvas(self):
        """Efface l'affichage après confirmation, et remet à jour la liste des triangles."""
        if not messagebox.askyesno(
            "Effacer l'affichage",
            "Voulez-vous effacer l'affichage et réinitialiser la liste des triangles ?"
        ):
            return
        self._discard_manual_move_preview()
        self.canvas.delete("all")

        # Réinitialiser l'état d'affichage du scénario actif
        # (vidage en place pour garder le lien scen.last_drawn)
        self.canvas_objects.clear()
        self._nearest_line_id = None
        self._clear_edge_highlights()
        self._attachment_intent = None
        self._attachment_preview = None
        self._edge_highlights = None
        self._rebuild_triangle_listbox_from_core()

        self.status.config(text="Affichage effacé")
        self._hide_tooltip()

        # L’horloge reste visible (overlay)
        self._draw_clock_overlay()

    # ---------- Overlay Horloge (indépendant du zoom/pan) ----------
    def _redraw_overlay_only(self):
        """Efface/redessine uniquement l'overlay (horloge)."""
        self.compass_controller.redraw()
    def _draw_clock_overlay(self):
        """Délègue le rendu du socle du compas au contrôleur dédié."""
        self.compass_controller.draw_overlay()
    def _clock_clear_snap_target(self):
        """Efface le marqueur visuel du sommet ciblé pendant le drag."""
        self.compass_controller.clear_snap_target()
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
        canvas_world = self._get_canvas_display_world()
        canvas_hypothesis = self._get_canvas_display_hypothesis()

        self.canvas.delete("all")
        # Fond carte (si layer visible)
        if self.show_map_layer is None or self.show_map_layer.get():
            self.background_map_layer.draw(opacity=int(self.map_opacity.get()))

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
                labels = self._get_core_vertex_labels(t, canvas_world)
                P = t["pts"]
                fill = "#ffd6d6" if i in self._comparison_diff_indices else None
                deform_outline = (
                    self._deformation_state.active
                    and str(t.get("topoElementId", "") or "")
                    == self._deformation_state.element_id
                )
                self._draw_triangle_screen(
                    P,
                    labels=[f"O:{labels[0]}", f"B:{labels[1]}", f"L:{labels[2]}"],
                    tri_label=self._build_triangle_display_label(
                        t, canvas_world, canvas_hypothesis
                    ),
                    fill=fill,
                    diff_outline=bool(fill),
                    drawEdges=(not onlyContours),
                    deform_outline=deform_outline,
                )
                if self._deformation_state.active:
                    self._draw_deformation_vertex_overlays(
                        P,
                        str(t.get("topoElementId", "") or ""),
                    )

        # 2) Contour des groupes par-dessus (lisible), si demandé.
        if showContoursMode:
            self._draw_group_outlines()

        if not showContoursMode:
            # — Recrée l'aide visuelle UNIQUEMENT si une sélection 'vertex' est encore active —
            if self._sel and self._sel.get("mode") in ("vertex", "move_group"):
                # redessine candidates + best si on a des données
                if self._edge_highlights:
                    self._redraw_edge_highlights()
                if self._sel.get("mode") == "vertex":
                    # remet la ligne grise
                    idx = self._sel["idx"]
                    vkey = self._sel["vkey"]
                    P = self._last_drawn[idx]["pts"]
                    v_world = np.array(P[vkey], dtype=float)
                    self._update_nearest_line(v_world, exclude_idx=idx)
            else:
                # pas de sélection active -> pas d'aides persistantes
                self._edge_highlights = None
                self._attachment_intent = None
                self._attachment_preview = None

        self._draw_clock_azimuth_traits_layer()

        self.canvas.delete("balises_layer")
        if self.show_balises_layer is None or self.show_balises_layer.get():
            self._draw_balises_layer()

        # Poignées de redimensionnement fond (overlay UI)
        self.background_map_layer.draw_resize_handles(bool(self.bg_resize_mode.get()))
        # Redessiner l'horloge (overlay indépendant)
        self._draw_clock_overlay()
        # Après tout redraw, le cache de pick n'est plus valide
        self._invalidate_pick_cache()

    def _draw_balises_layer(self):
        """Dessine les balises Catalogue actives, résolues dans le repère World."""
        for beacon in self.catalogue.iter_beacons():
            if beacon.archived:
                continue
            wx, wy = self._beacon_world_resolver.get_world(beacon.beacon_id)
            city = self.catalogue.get_city(beacon.city_id)

            sx, sy = self._world_to_screen((wx, wy))
            r = self._marker_px
            color = self.catalogue.get_beacon_group_color(beacon.group) if beacon.group else None
            self.canvas.create_oval(
                sx - r, sy - r, sx + r, sy + r,
                fill=color or "#000000",
                outline=color or "#000000",
                tags=("balises_layer",),
            )
            self.canvas.create_text(
                sx,
                sy + r + 2,
                text=city.name,
                anchor="n",
                font=("Arial", 8),
                fill="#000000",
                tags=("balises_layer",),
            )

    def _draw_group_outlines(self):
        """Dessine uniquement le contour de chaque groupe (enveloppe extérieure)."""
        self.canvas.delete("group_outline")

        scen = self._get_active_scenario()
        world = scen.topoWorld

        for core_group_id in world.getLiveGroupIds():
            boundary_segments = world.getBoundarySegments(core_group_id)
            outline = self._project_boundary_segments(core_group_id, boundary_segments)
            for p1, p2 in outline:
                x1, y1 = self._world_to_screen(p1)
                x2, y2 = self._world_to_screen(p2)
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill="#000000",
                    width=3,
                    tags=("group_outline",),
                )

    def _clock_clip_ray_to_viewport(self, sx0: float, sy0: float, azDeg: float) -> tuple[float, float] | None:
        return self.compass_controller.clip_ray_to_viewport(sx0, sy0, azDeg)
    def _draw_clock_azimuth_traits_layer(self):
        self.canvas.delete("clock_azimuth_traits")
        if not self._layerGuidesVisible:
            return
        scen = self._get_active_scenario()
        if scen is None:
            return
        traits = scen.clockAzimuthTraits
        if not traits:
            return

        world = scen.topoWorld
        for trait in traits:
            nodeId = str(trait["nodeId"])
            topoGroupId = str(trait["topoGroupId"])
            deltaAz = float(trait["deltaAzDeg"]) % 360.0
            if "colorHex" in trait:
                colorHex = str(trait["colorHex"])
            else:
                colorHex = "#0b3d91"

            try:
                nodeWorld = np.array(world.getConceptNodeWorldXY(nodeId, topoGroupId), dtype=float)
            except Exception:
                continue
            azTraitAbs = (float(self.compass_state.ref_azimuth_deg) + deltaAz) % 360.0

            sx0, sy0 = self._world_to_screen(nodeWorld)
            end = self._clock_clip_ray_to_viewport(float(sx0), float(sy0), float(azTraitAbs))
            if end is None:
                continue
            sx1, sy1 = end
            self.canvas.create_line(
                float(sx0), float(sy0), float(sx1), float(sy1),
                width=2,
                fill=colorHex,
                tags=("clock_azimuth_traits",),
            )
            sxMid = (float(sx0) + float(sx1)) / 2.0
            syMid = (float(sy0) + float(sy1)) / 2.0
            self.canvas.create_text(
                float(sxMid), float(syMid),
                text=self._clock_delta_display_text(deltaAz),
                font=("Arial", 12, "bold"),
                fill=colorHex,
                tags=("clock_azimuth_traits",),
            )

    def _draw_triangle_screen(self, P,
                              outline="black", width=2, labels=None, inset=0.35,
                              tri_label=None, fill=None, diff_outline=False,
                              drawEdges=True, deform_outline=False):
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
        if deform_outline:
            self.canvas.create_line(Ox, Oy, Bx, By, Lx, Ly, Ox, Oy,
                                    fill="#d00000", width=3)

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
        if tri_label is not None:
            sx, sy = self._world_to_screen((cx, cy))
            num_txt = str(tri_label)
            self.canvas.create_text(
                sx, sy, text=num_txt,
                anchor="center", font=("Arial", 10, "bold"),
                fill="red", tags="tri_num"
            )
            self.canvas.tag_raise("tri_num")

    def _draw_deformation_vertex_overlays(self, P, element_id: str) -> None:
        """Signale les occurrences modifiees sans modifier la geometrie Core."""
        state = self._deformation_state
        roles = state.modified_roles_for_element(element_id)
        if not roles:
            return
        for role in roles:
            x, y = self._world_to_screen(P[role])
            selected = state.selected_occurrence == (element_id, role)
            radius = 11 if selected else 8
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="",
                outline="#d00000" if selected else "#f59e0b",
                width=3 if selected else 2,
            )

    # --- helpers: mode déconnexion (CTRL) + curseur ---
    def _clock_get_pointer_canvas_xy(self) -> Tuple[int, int]:
        if self.canvas is None:
            return (0, 0)
        sx = int(self.canvas.winfo_pointerx() - self.canvas.winfo_rootx())
        sy = int(self.canvas.winfo_pointery() - self.canvas.winfo_rooty())
        return (sx, sy)

    def _clock_refresh_active_preview_under_pointer(self):
        if self.canvas is None:
            return
        sx, sy = self._clock_get_pointer_canvas_xy()
        if self.compass_state.trace.active:
            self._clock_trace_update_preview(int(sx), int(sy))
        elif self.compass_state.measure.active:
            self._clock_measure_update_preview(int(sx), int(sy))
        elif self.compass_state.arc.active:
            self._clock_arc_update_preview(int(sx), int(sy))
        elif self.compass_state.set_ref.active:
            self._clock_setref_update_preview(int(sx), int(sy))

    def _on_ctrl_down(self, event=None):
        # Pendant les modes interactifs du compas, CTRL sert uniquement à désactiver le snap.
        # On évite donc d'activer le mode "déconnexion" des triangles (curseur + aides).
        if (
            self.compass_state.measure.active
            or self.compass_state.arc.active
            or self.compass_state.set_ref.active
            or self.compass_state.trace.active
        ):
            self._ctrl_down = True
            self._clock_refresh_active_preview_under_pointer()
            return

        if not self._ctrl_down:
            self._ctrl_down = True
            # Masquer tout tooltip en mode déconnexion
            self._hide_tooltip()
            self.canvas.configure(cursor="X_cursor")

            selection = self._sel if isinstance(self._sel, dict) else None
            if (
                selection is not None
                and selection.get("mode") == "move_group"
                and not self._is_active_auto_scenario()
            ):
                core_group_id = selection.get("core_group_id")
                if not core_group_id:
                    raise RuntimeError("[ATT-003D] core_group_id absent pour la preview")
                anchor = selection.get("anchor")
                if anchor and anchor.get("type") == "vertex":
                    anchor_tid = int(anchor["tid"])
                    anchor_vkey = anchor["vkey"]
                    if 0 <= anchor_tid < len(self._last_drawn):
                        anchor_world = np.asarray(
                            self._last_drawn[anchor_tid]["pts"][anchor_vkey],
                            dtype=float,
                        )
                        self._update_group_drag_snap_assist(
                            anchor_world,
                            anchor_tid,
                            anchor_vkey,
                            str(core_group_id),
                        )
                        if (
                            self._attachment_preview is not None
                            and self._attachment_preview.accepted
                        ):
                            self._preview_attachment_rotation_to_last_drawn(
                                self._attachment_preview
                            )
                            self._redraw_from(self._last_drawn)

    def _on_ctrl_up(self, event=None):
        if self._ctrl_down:
            self._ctrl_down = False
            if (
                self.compass_state.measure.active
                or self.compass_state.arc.active
                or self.compass_state.set_ref.active
                or self.compass_state.trace.active
            ):
                self._clock_refresh_active_preview_under_pointer()
                return
            selection = self._sel if isinstance(self._sel, dict) else None
            if (
                selection is not None
                and selection.get("mode") == "move_group"
                and self._attachment_preview is not None
                and self._attachment_preview.accepted
                and not self._is_active_auto_scenario()
            ):
                self._restore_manual_move_group_preview()
                anchor = selection.get("anchor")
                if anchor and anchor.get("type") == "vertex":
                    anchor_tid = int(anchor["tid"])
                    anchor_vkey = anchor["vkey"]
                    if 0 <= anchor_tid < len(self._last_drawn):
                        self._update_group_drag_snap_assist(
                            np.asarray(
                                self._last_drawn[anchor_tid]["pts"][anchor_vkey],
                                dtype=float,
                            ),
                            anchor_tid,
                            anchor_vkey,
                            str(selection["core_group_id"]),
                        )
                self._redraw_from(self._last_drawn)
            if self.canvas is not None:
                self.canvas.configure(cursor="")

    # ---------- Mouse navigation ----------
    # Drag depuis la liste

    def _on_triangle_list_select(self, event=None):
        """Valide la sélection Catalogue (zéro, un ou deux triangles)."""
        # éviter la récursion quand on modifie la sélection nous-mêmes
        if self._in_triangle_select_guard:
            return
        if not hasattr(self, "listbox"):
            return

        sel = tuple(sorted(int(index) for index in self.listbox.curselection()))
        valid, message = self._validate_triangle_list_selection(sel)

        if not valid:
            self._set_triangle_list_selection(getattr(self, "_last_triangle_selection", ()) or ())
            if hasattr(self, "status"):
                self.status.config(text=message)
            return
        self._last_triangle_selection = sel

    def _validate_triangle_list_selection(self, selection: tuple[int, ...]) -> tuple[bool, str]:
        """Validation pure de la sélection Catalogue contrôlée par l'UI."""
        normalized = tuple(sorted(set(int(index) for index in selection)))
        if len(normalized) > 2:
            return False, "Sélection multiple impossible : deux triangles maximum."
        try:
            triangle_ids = tuple(self._get_triangle_id_from_listbox_index(index) for index in normalized)
        except IndexError:
            return False, "Sélection Catalogue invalide."
        used_ids = self._get_active_scenario().topoWorld.get_used_source_triangle_ids()
        if any(triangle_id in used_ids for triangle_id in triangle_ids):
            return False, "Triangle déjà utilisé dans ce scénario."
        if len(triangle_ids) == 2:
            scenario = self._get_active_scenario()
            resolver = GeometryReferenceResolver(self.catalogue, scenario.reference)
            first, second = (
                resolver.resolve_triangle(triangle_id) for triangle_id in triangle_ids
            )
            if first.base_city_ref_id != second.base_city_ref_id:
                return False, "Sélection multiple impossible : les triangles doivent avoir la même base."
            if first.opening_city_ref_id != second.opening_city_ref_id:
                return False, "Sélection multiple impossible : les triangles doivent avoir la même ouverture."
        return True, ""

    def _set_triangle_list_selection(self, selection: tuple[int, ...]) -> None:
        """Projette atomiquement l'état validé du contrôleur dans la Listbox."""
        normalized = tuple(sorted(set(int(index) for index in selection)))
        self._in_triangle_select_guard = True
        self.listbox.selection_clear(0, tk.END)
        for index in normalized:
            if 0 <= index < self.listbox.size():
                self.listbox.selection_set(index)
        self._last_triangle_selection = normalized
        self._in_triangle_select_guard = False

    # ---------- Mouse navigation ----------
    # Drag depuis la liste

    def _on_list_mouse_down(self, event):
        if self._deformation_state.active:
            self.status.config(text="Ajout de triangle indisponible en mode deformation.")
            return "break"
        """
        Démarre un drag & drop depuis la listbox,
        sauf si le triangle est déjà utilisé dans le scénario courant.
        """
        scen = self._get_active_scenario()
        if scen.hypothesis is None:
            raise ValueError("ScenarioHypothesis absente du scénario actif")

        # Index de la ligne cliquée dans la listbox
        i = self.listbox.nearest(event.y)
        if i < 0:
            return

        triangle_id = self._get_triangle_id_from_listbox_index(i)
        used = triangle_id in scen.topoWorld.get_used_source_triangle_ids()
        # Si le triangle est déjà posé dans ce scénario, on bloque le drag
        if used:
            self.status.config(text=f"Triangle {triangle_id} déjà utilisé dans ce scénario.")
            self._prepare_list_placement((), i)
            return "break"

        ctrl_pressed = bool(getattr(event, "state", 0) & 0x0004)
        current = tuple(getattr(self, "_last_triangle_selection", ()) or ())
        if ctrl_pressed:
            requested = tuple(index for index in current if index != i) if i in current else current + (i,)
            requested = tuple(sorted(requested))
            valid, message = self._validate_triangle_list_selection(requested)
            if valid:
                self._set_triangle_list_selection(requested)
                self._prepare_list_placement(requested, i)
            else:
                self._set_triangle_list_selection(current)
                self.status.config(text=message)
            return "break"

        # Un clic sur un membre d'une paire validée conserve la paire.
        if not (len(current) == 2 and i in current):
            drag_selection = (i,)
        else:
            drag_selection = current
        valid, message = self._validate_triangle_list_selection(drag_selection)
        if not valid:
            self._set_triangle_list_selection(current)
            self.status.config(text=message)
            return "break"
        self._set_triangle_list_selection(drag_selection)
        self._prepare_list_placement(drag_selection, i)
        return "break"

    def _prepare_list_placement(self, selection: tuple[int, ...], clicked_index: int) -> None:
        """Attache immédiatement la sélection Catalogue validée au curseur."""
        self._delete_drag_previews()
        if not selection:
            self._drag = None
            self.canvas.configure(cursor="")
            return
        triangle_ids = tuple(self._get_triangle_id_from_listbox_index(index) for index in selection)
        self._drag = {
            "from": "list",
            "kind": "triangle" if len(selection) == 1 else "quadrilateral",
            "triangle_ids": triangle_ids,
            "triangle_id": self._get_triangle_id_from_listbox_index(clicked_index),
            "list_index": clicked_index,
        }
        if self._drag["kind"] == "quadrilateral":
            self._drag.update(self._build_quadrilateral_drag_geometry(triangle_ids))
        self.canvas.configure(cursor="hand2")
        message = (
            "Déplacez le quadrilatère puis cliquez pour le déposer."
            if len(selection) == 2
            else "Déplacez le triangle puis cliquez pour le déposer."
        )
        self.status.config(text=message)

    def _commit_list_placement_at_canvas_event(self, event) -> None:
        """Valide, au clic Canvas, l'objet Catalogue attaché au curseur."""
        if not self._drag or self._drag.get("from") != "list":
            return
        self._update_list_drag_preview_at_canvas_xy(event.x, event.y)
        kind = self._drag["kind"]
        self._delete_drag_previews()
        if kind == "quadrilateral":
            self._place_dragged_quadrilateral()
        else:
            self._place_dragged_triangle()
        self._drag = None
        self._set_triangle_list_selection(())
        self.canvas.configure(cursor="")

    def _delete_drag_previews(self) -> None:
        # ``__new__``-based tests intentionally do not initialise Tk itself;
        # read the instance dictionary so Tk.__getattr__ is never involved.
        preview_ids = list(self.__dict__.get("_drag_preview_ids", []))
        for item_id in preview_ids:
            self.canvas.delete(item_id)
        self._drag_preview_ids = []
        preview_id = self.__dict__.get("_drag_preview_id")
        if preview_id is not None and preview_id not in preview_ids:
            self.canvas.delete(preview_id)
        self._drag_preview_id = None

    def _build_quadrilateral_drag_geometry(self, triangle_ids: tuple[str, str]) -> dict:
        """Construit la pose relative exclusivement via le resolver Core."""
        first_id, second_id = triangle_ids
        preview_world = TopologyWorld()
        scen = self._get_active_scenario()
        resolver = GeometryReferenceResolver(self.catalogue, scen.reference)
        first = materialize_triangle(resolver, first_id)
        second = materialize_triangle(resolver, second_id)
        preview_world.add_element_as_new_group(first)
        preview_world.add_element_as_new_group(second)
        preview_world.setElementPose(str(first.element_id), np.eye(2), np.zeros(2), mirrored=False)
        group_id = preview_world.apply_attachment(TopologyEdgeEdgeAttachment(
            attachment_id=preview_world.new_attachment_id(),
            mob_element_id=str(second.element_id), mob_edge="OB",
            dest_element_id=str(first.element_id), dest_edge="OB",
        ))
        preview_world.replay_group_attachment_poses(group_id, str(first.element_id))
        return {
            "relative_world_pts": {
                first_id: getCoreTriangleWorldPoints(preview_world, str(first.element_id)),
                second_id: getCoreTriangleWorldPoints(preview_world, str(second.element_id)),
            },
            "reference_triangle_id": first_id,
            "grab_offset": None,
        }

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

    def _resolve_hover_vertex_node_id(self, *, entry: Dict, vertex_key: str, world) -> str | None:
        """Resout un sommet UI vers son node physique Core, sans format d'ID local."""
        element_id = str((entry or {}).get("topoElementId", "") or "").strip()
        if not element_id or world is None:
            return None
        try:
            return world.get_element_vertex_node_id_by_type(element_id, str(vertex_key))
        except (KeyError, IndexError, ValueError):
            return None

    def _on_canvas_motion_update_drag(self, event):
        if self.compass_state.trace.active:
            self._clock_trace_update_preview(int(event.x), int(event.y))
            return "break"

        # Mode compas : mesure d'un azimut (relatif à la référence)
        if self.compass_state.measure.active:
            self._clock_measure_update_preview(int(event.x), int(event.y))
            return "break"

        # Mode compas : mesure d'arc d'angle
        if self.compass_state.arc.active:
            self._clock_arc_update_preview(int(event.x), int(event.y))
            return "break"

        # Mode compas : définition de l'azimut de référence
        if self.compass_state.set_ref.active:
            self._clock_setref_update_preview(int(event.x), int(event.y))
            return "break"

        # Toujours garantir un pick-cache à jour avant tout hit/tooltip
        self._ensure_pick_cache()

        # 1) Drag & drop depuis la liste → fantôme
        if self._drag:
            self._update_list_drag_preview_at_canvas_xy(event.x, event.y)
            return

        # 2) Mode rotation de GROUPE : suivre la souris (sans bouton appuyé)
        if self._sel and self._sel.get("mode") == "rotate_group":
            sel = self._sel
            pivot = sel["pivot"] if sel.get("auto_geom") else sel["pivot_world"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            cur_angle = math.atan2(wy - pivot[1], wx - pivot[0])
            start_angle = sel["start_angle"] if sel.get("auto_geom") else sel["mouse_angle_start"]
            dtheta_raw = cur_angle - start_angle
            dtheta = dtheta_raw if sel.get("auto_geom") else self._normalize_rotation_angle(dtheta_raw)

            # AUTO: rotation globale partagée (doit impacter TOUS les scénarios auto)
            if sel.get("auto_geom"):
                self._preview_auto_rotation_from_snapshot(
                    sel["auto_preview_initial_pts"],
                    sel["pivot"],
                    dtheta,
                )
                self._redraw_from(self._last_drawn)
                self._sel["last_angle"] = cur_angle
                return

            # MANUAL: preview UI transactionnel depuis le snapshot immuable.
            self._preview_rotate_group_from_snapshot(
                sel["rotate_preview_initial_pts"],
                sel["pivot_world"],
                dtheta,
            )

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
            v_world = np.array(P0[vkey], dtype=float) if vkey in ("O", "B", "L") else None

            topoWorld = self._get_canvas_display_world()

            tooltip_txt = ""
            if v_world is not None and vkey in ("O", "B", "L"):
                nodeId = self._resolve_hover_vertex_node_id(
                    entry=self._last_drawn[idx],
                    vertex_key=vkey,
                    world=topoWorld,
                )
                if nodeId is not None:
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
                sx_v, sy_v = self._world_to_screen(v_world)
                self._ensure_canvas_tooltip(tooltip_txt)
                tw = max(1, int(self._tooltip.winfo_width()))
                th = max(1, int(self._tooltip.winfo_height()))

                pos = self._computeNodeTooltipCanvasPosition(sx_v, sy_v, tw, th)
                if pos is None:
                    self._hide_tooltip()
                else:
                    x_tip, y_tip = pos
                    base_x = self.canvas.winfo_rootx()
                    base_y = self.canvas.winfo_rooty()
                    self._tooltip.wm_geometry(f"+{int(base_x + x_tip)}+{int(base_y + y_tip)}")
                    self._tooltip.deiconify()
            else:
                self._hide_tooltip()
        else:
            # pas de sommet → masquer tooltip
            self._hide_tooltip()

        # sécurité : si le cache n’est pas prêt (post-load), le régénérer pour les tooltips
        if not self._pick_cache_valid:
            self._rebuild_pick_cache()

    # ---------- Lien + surlignage faces candidates ----------
    def _find_nearest_vertex(self, v_world, exclude_idx=None, exclude_core_group_id=None):
        """Retourne (idx_triangle, key('O'|'B'|'L'), pos_world) du sommet d'un AUTRE triangle le plus proche.
        On peut exclure un triangle précis (exclude_idx) et/ou tout un groupe Core."""
        best = None
        best_d2 = None
        for j, t in enumerate(self._last_drawn):
            if j == exclude_idx:
                continue
            candidate_core_group_id = self._get_active_core_group_id_for_entry(t)
            if (
                exclude_core_group_id
                and candidate_core_group_id == exclude_core_group_id
            ):
                continue
            P = t["pts"]
            for k in ("O", "B", "L"):
                w = np.array(P[k], dtype=float)
                d2 = float((w[0]-v_world[0])**2 + (w[1]-v_world[1])**2)
                if (best_d2 is None) or (d2 < best_d2):
                    best_d2 = d2
                    best = (j, k, w)
        return best

    def _find_nearest_vertex_candidate(
        self, v_world, exclude_idx=None, exclude_core_group_id=None
    ):
        """Expose le candidat sommet avec sa distance, sans changer l'API historique."""
        found = self._find_nearest_vertex(
            v_world,
            exclude_idx=exclude_idx,
            exclude_core_group_id=exclude_core_group_id,
        )
        if found is None:
            return None
        triangle_idx, vkey, world_pos = found
        origin = np.asarray(v_world, dtype=float)
        target = np.asarray(world_pos, dtype=float)
        return {
            "type": "vertex",
            "triangle_idx": triangle_idx,
            "vkey": vkey,
            "world": target,
            "distance2": float(np.sum((target - origin) ** 2)),
        }

    def _find_nearest_beacon_candidate(self, v_world):
        """Retourne la balise monde la plus proche du sommet monde donné."""
        origin = np.asarray(v_world, dtype=float)
        best = None
        best_d2 = None
        for beacon in get_anchor_beacon_candidates(self.catalogue):
            world_pos = np.asarray(self._beacon_world_resolver.get_world(beacon.beacon_id), dtype=float)
            if world_pos.shape != (2,) or not np.all(np.isfinite(world_pos)):
                continue
            d2 = float(np.sum((world_pos - origin) ** 2))
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = {
                    "type": "beacon",
                    "beacon_id": beacon.beacon_id,
                    "world": world_pos,
                    "distance2": d2,
                }
        return best

    def _update_assist_line_to_world(self, source_world, target_world, color="#888888"):
        """Dessine la ligne d'aide commune aux cibles sommet et balise."""
        x1, y1 = self._world_to_screen(source_world)
        x2, y2 = self._world_to_screen(target_world)
        if self._nearest_line_id is None:
            self._nearest_line_id = self.canvas.create_line(
                x1, y1, x2, y2, fill=color, width=1
            )
        else:
            self.canvas.coords(self._nearest_line_id, x1, y1, x2, y2)
            self.canvas.itemconfig(self._nearest_line_id, fill=color)

    def _group_drag_clear_beacon_target(self):
        """Efface uniquement l'anneau de snap balise du drag de groupe."""
        self.canvas.delete("group_drag_beacon_target")

    def _clear_anchor_rotation_pivot_highlight(self):
        """Efface l'anneau jaune signalant le pivot d'une rotation ancrée."""
        self.canvas.delete("anchor_rotation_pivot_highlight")

    def _draw_anchor_rotation_pivot_highlight(self, pivot_world) -> None:
        """Dessine l'anneau jaune du pivot-balise d'une rotation ancrée."""
        self._clear_anchor_rotation_pivot_highlight()
        px, py = self._world_to_screen(pivot_world)
        radius = 12
        self.canvas.create_oval(
            px - radius,
            py - radius,
            px + radius,
            py + radius,
            outline="#FFD700",
            width=3,
            fill="",
            tags="anchor_rotation_pivot_highlight",
        )
        self.canvas.tag_raise("anchor_rotation_pivot_highlight")

    def _group_drag_update_beacon_target(self, candidate):
        """Affiche le même anneau rouge que le snap compas, sans son état."""
        self._group_drag_clear_beacon_target()
        if not isinstance(candidate, dict) or candidate.get("type") != "beacon":
            return
        world_pos = candidate.get("world")
        if world_pos is None:
            return
        px, py = self._world_to_screen(world_pos)
        radius = 10
        self.canvas.create_oval(
            px - radius,
            py - radius,
            px + radius,
            py + radius,
            outline="#FF0000",
            width=3,
            fill="",
            tags="group_drag_beacon_target",
        )
        self.canvas.tag_raise("group_drag_beacon_target")

    def _update_group_drag_snap_assist(
        self, v_world, anchor_idx, anchor_vkey, core_group_id
    ):
        """Actualise l'unique moteur de cible du drag sommet.

        Un sommet n'est admissible qu'après construction effective d'un
        ``_attachment_intent``. Il est ensuite arbitré avec la balise la plus proche
        sur la distance géométrique ; à distance égale, le sommet gagne.
        """
        self._reset_assist()

        vertex_candidate = self._find_nearest_vertex_candidate(
            v_world,
            exclude_idx=anchor_idx,
            exclude_core_group_id=core_group_id,
        )
        if vertex_candidate is not None:
            self._update_assist_line_to_world(
                v_world, vertex_candidate["world"]
            )
            self._update_edge_highlights(
                anchor_idx,
                anchor_vkey,
                vertex_candidate["triangle_idx"],
                vertex_candidate["vkey"],
            )
            if (
                self._attachment_intent is None
                or self._attachment_preview is None
                or not self._attachment_preview.accepted
            ):
                vertex_candidate = None

        beacon_candidate = None
        if self.show_balises_layer is None or self.show_balises_layer.get():
            beacon_candidate = self._find_nearest_beacon_candidate(v_world)
        if beacon_candidate is not None and (
            vertex_candidate is None
            or beacon_candidate["distance2"] < vertex_candidate["distance2"]
        ):
            self._clear_edge_highlights()
            self._attachment_intent = None
            self._attachment_preview = None
            self._update_assist_line_to_world(
                v_world, beacon_candidate["world"], color="#2b78e4"
            )
            self._group_drag_update_beacon_target(beacon_candidate)
            self._group_drag_snap_candidate = beacon_candidate
            return beacon_candidate

        if vertex_candidate is not None:
            self._group_drag_snap_candidate = vertex_candidate
            return vertex_candidate

        self._reset_assist()
        return None

    def _update_nearest_line(self, v_world, exclude_idx=None, exclude_core_group_id=None):
        """Dessine (ou MAJ) un trait fin entre v_world et le sommet le plus proche d'un AUTRE triangle."""
        found = self._find_nearest_vertex(
            v_world,
            exclude_idx=exclude_idx,
            exclude_core_group_id=exclude_core_group_id,
        )

        if found is None:
            self._clear_nearest_line()
            return
        _, _, best = found
        self._update_assist_line_to_world(v_world, best)

    def _clear_nearest_line(self):
        if self._nearest_line_id is not None:
            self.canvas.delete(self._nearest_line_id)
            self._nearest_line_id = None

    def _reset_assist(self):
        """Nettoie TOUTES les aides visuelles et l'état associé."""
        self._clear_nearest_line()
        self._clear_edge_highlights()
        self._edge_highlights = None
        self._attachment_intent = None
        self._attachment_preview = None
        self._group_drag_snap_candidate = None
        self._group_drag_clear_beacon_target()
        self._clear_anchor_rotation_pivot_highlight()

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
    def _project_boundary_segments(self, core_group_id: str, segments) -> list[tuple[np.ndarray, np.ndarray]]:
        """Convertit des `BoundarySegment` Core en segments de rendu temporaires."""
        if not segments:
            return []

        element_map = {
            entry["topoElementId"]: entry
            for entry in self._get_projected_elements_for_core_group(core_group_id)
            if entry.get("topoElementId")
        }

        outline = []
        for bs in segments:
            element_id = bs.elementId
            if element_id not in element_map:
                raise ValueError(f"Element manquant dans last_drawn: {element_id}")
            tri = element_map[element_id]
            pts = tri.get("pts", None)
            if not isinstance(pts, dict):
                raise ValueError(f"Points invalides pour {element_id}")

            def vkeyFromNodeId(node_id: str, element_id: str) -> str:
                nid = node_id
                if not nid.startswith(element_id + ":N"):
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

    @staticmethod
    def _select_first_accepted_manual_attachment_candidate(
        candidates,
        *,
        build_intent,
        preview_intent,
    ):
        """Retient le premier candidat, par score croissant, accepté par le Core."""
        for candidate in sorted(candidates, key=lambda item: item[0]):
            intent = build_intent(candidate)
            if intent is None:
                continue
            preview = preview_intent(intent)
            if preview.accepted:
                return candidate, intent, preview
        return None, None, None

    def _update_edge_highlights(self, mob_idx: int, vkey_m: str, tgt_idx: int, tgt_vkey: str):
        """Construit l'assistance de snap depuis les segments Boundary fournis par le Core."""

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
        # On récupère la topo
        scen = self._get_active_scenario()
        world = scen.topoWorld
        # Les nodes physiques sont resolus par le Core; l'UI ne connait ni
        # leur format ni l'ordre physique des sommets.
        tAElementId = str(tri_t.get("topoElementId", "") or "").strip()
        mAElementId = str(tri_m.get("topoElementId", "") or "").strip()
        try:
            tAId = world.get_element_vertex_node_id_by_type(tAElementId, tgt_vkey)
            mAId = world.get_element_vertex_node_id_by_type(mAElementId, vkey_m)
        except (KeyError, IndexError, ValueError):
            self._clear_edge_highlights()
            self._attachment_intent = None
            self._attachment_preview = None
            return
        core_gid_m = self._get_core_group_id_for_triangle_index(mob_idx)
        core_gid_t = self._get_core_group_id_for_triangle_index(tgt_idx)

        # MIG-TOPO-SERVICE-001 : le Core décide du contour extérieur et de
        # l'incidence. L'UI ne fait que projeter ces segments puis les classe.
        if not core_gid_m or not core_gid_t:
            self._clear_edge_highlights()
            self._attachment_intent = None
            self._attachment_preview = None
            return
        mob_boundary_segments = world.getBoundarySegments(core_gid_m)
        tgt_boundary_segments = world.getBoundarySegments(core_gid_t)
        mob_incident_boundary_segments = world.getIncidentBoundarySegments(core_gid_m, mAId)
        tgt_incident_boundary_segments = world.getIncidentBoundarySegments(core_gid_t, tAId)

        mob_outline = self._project_boundary_segments(core_gid_m, mob_boundary_segments)
        tgt_outline = self._project_boundary_segments(core_gid_t, tgt_boundary_segments)
        m_inc = self._project_boundary_segments(core_gid_m, mob_incident_boundary_segments)
        t_inc = self._project_boundary_segments(core_gid_t, tgt_incident_boundary_segments)

        # 4) sélection globale : minimiser l’écart d’azimut + anti-chevauchement
        candidates = []  # (score, m_edge, t_edge)
        mob_projected_elements = self._get_projected_elements_for_core_group(core_gid_m)
        tgt_projected_elements = self._get_projected_elements_for_core_group(core_gid_t)
        mob_tids = [index for index, _entry in self.canvas_objects.get_indexed_by_topology_ids(
            entry.get("topoElementId") for entry in mob_projected_elements
        )]
        tgt_tids = [index for index, _entry in self.canvas_objects.get_indexed_by_topology_ids(
            entry.get("topoElementId") for entry in tgt_projected_elements
        )]

        # ATT-003A : la sélection visuelle ne lance plus de simulation legacy.
        # La preview Core V2 (dont l'overlap) sera introduite en ATT-003B.

        m_oriented = [(a, b) if _almost_eq(a, vm) else (b, a) for (a, b) in (m_inc or [])]
        t_oriented = [(a, b) if _almost_eq(a, vt) else (b, a) for (a, b) in (t_inc or [])]
        for me in m_oriented:
            azm = _azim(*me)
            for te in t_oriented:
                azt = _azim(*te)
                score = _ang_dist(azm, azt)
                candidates.append((score, me, te))

        best, self._attachment_intent, self._attachment_preview = (
            self._select_first_accepted_manual_attachment_candidate(
                candidates,
                build_intent=lambda candidate: buildManualAttachmentIntentFromBest(
                    candidate,
                    world=world,
                    mob_idx=mob_idx,
                    tgt_idx=tgt_idx,
                    mob_tids=mob_tids,
                    tgt_tids=tgt_tids,
                    last_drawn=self._last_drawn,
                    eps_world=EPS_WORLD,
                    mATmpId=mAId,
                    tATmpId=tAId,
                ),
                preview_intent=lambda intent: previewManualAttachment(world, intent),
            )
        )

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

        self._redraw_edge_highlights()

    def _clear_edge_highlights(self):
        """Efface du canvas les lignes d'aide déjà dessinées.
        N'efface PAS self._attachment_intent ni self._edge_highlights."""
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
        # 3) Meilleure : verte si la preview Core V2 est acceptée, rouge sinon.
        best = data.get("best")
        if best:
            preview = self._attachment_preview
            best_color = "#178C3A" if preview is not None and preview.accepted else "#FF0000"
            # best = (m_a, m_b, t_a, t_b)
            m_a, m_b, t_a, t_b = best
            # Côté cible (épais)
            _id = self._draw_temp_edge_world(np.array(t_a, float), np.array(t_b, float), color=best_color, width=4)
            if _id:
                self._edge_highlight_ids.append(_id)
            # Côté mobile (épais également)
            _id = self._draw_temp_edge_world(np.array(m_a, float), np.array(m_b, float), color=best_color, width=4)
            if _id:
                self._edge_highlight_ids.append(_id)

            # côté mobile (fin) — l'arête entrée en contact
            _id2 = self._draw_temp_edge_world(np.array(m_a, float), np.array(m_b, float), color=best_color, width=3)
            if _id2:
                self._edge_highlight_ids.append(_id2)

    def _place_dragged_triangle(self):
        if self._deformation_state.active:
            raise RuntimeError("Ajout de triangle interdit en mode deformation")
        """Dépose un triangle manuel en créant d'abord sa géométrie Core."""
        if not self._drag or "world_pts" not in self._drag:
            return
        Pw = self._drag["world_pts"]
        scen = self._get_active_scenario()
        world = scen.topoWorld
        triangle_id = self._drag["triangle_id"]
        hypothesis = scen.hypothesis
        if hypothesis is None:
            raise ValueError("ScenarioHypothesis absente du scénario actif")
        if triangle_id not in hypothesis.triangle_ids_by_rank:
            raise ValueError(f"ScenarioHypothesis: triangle outside hypothesis: {triangle_id}")
        if triangle_id in world.get_used_source_triangle_ids():
            raise ValueError(f"Catalogue: triangle already used: {triangle_id}")

        world.beginTopoTransaction()
        try:
            el = materialize_triangle(
                GeometryReferenceResolver(self.catalogue, scen.reference), triangle_id
            )
            core_gid = world.add_element_as_new_group(el)
            element_id = el.element_id
            if not element_id:
                raise RuntimeError("Topology: le Core n'a pas attribue d'elementId")
            world.setElementPose(
                str(element_id),
                R=np.eye(2),
                T=np.asarray(Pw["O"], dtype=float),
                mirrored=False,
            )
        finally:
            world.commitTopoTransaction()

        self.canvas_objects.add({"topoElementId": str(element_id)})
        self._project_core_element_to_last_drawn(world, str(element_id))
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Triangle placed: Core group {core_gid} created.")
        self._rebuild_triangle_listbox_from_core()
        self._in_triangle_select_guard = True
        if hasattr(self, "listbox"):
            self.listbox.selection_clear(0, tk.END)
        self._last_triangle_selection = None
        self._in_triangle_select_guard = False
        scen.is_placeholder = False
        return

    def _place_dragged_quadrilateral(self):
        """Publie atomiquement deux triangles reliés OB<->OB dans le Core."""
        if not self._drag or self._drag.get("kind") != "quadrilateral":
            return
        if "world_pts_by_triangle" not in self._drag:
            return
        first_id, second_id = self._drag["triangle_ids"]
        scen = self._get_active_scenario()
        if first_id in scen.topoWorld.get_used_source_triangle_ids() or second_id in scen.topoWorld.get_used_source_triangle_ids():
            raise ValueError("Catalogue: triangle already used")
        resolver = GeometryReferenceResolver(self.catalogue, scen.reference)
        first_triangle = resolver.resolve_triangle(first_id)
        second_triangle = resolver.resolve_triangle(second_id)
        if first_triangle.base_city_ref_id != second_triangle.base_city_ref_id:
            raise ValueError("Quadrilatère: bases Catalogue différentes")
        if first_triangle.opening_city_ref_id != second_triangle.opening_city_ref_id:
            raise ValueError("Quadrilatère: ouvertures Catalogue différentes")

        candidate_world = scen.topoWorld.clonePhysicalState()
        first = materialize_triangle(resolver, first_id)
        second = materialize_triangle(resolver, second_id)
        candidate_world.add_element_as_new_group(first)
        candidate_world.add_element_as_new_group(second)
        candidate_world.setElementPose(
            str(first.element_id), np.eye(2),
            np.asarray(self._drag["world_pts_by_triangle"][first_id]["O"], dtype=float),
            mirrored=False,
        )
        group_id = candidate_world.apply_attachment(TopologyEdgeEdgeAttachment(
            attachment_id=candidate_world.new_attachment_id(),
            mob_element_id=str(second.element_id), mob_edge="OB",
            dest_element_id=str(first.element_id), dest_edge="OB",
        ))
        candidate_world.replay_group_attachment_poses(group_id, str(first.element_id))
        if candidate_world.get_group_of_element(str(first.element_id)) != candidate_world.get_group_of_element(str(second.element_id)):
            raise RuntimeError("Quadrilatère: groupage Core incomplet")

        # Aucun état réel n'est touché avant cette unique publication.
        scen.topoWorld = candidate_world
        scen.is_placeholder = False
        self._rebuild_active_projection_from_core()
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Quadrilatère placé : groupe Core {group_id} créé.")
        self._rebuild_triangle_listbox_from_core()

    # =============   Groupes : helpers   ====================
    def _group_nodes(self, gid: int) -> List[Dict]:
        g = self.groups.get(gid)
        return [] if not g else g["nodes"]

    def _group_centroid(
        self, core_group_id: str, world: TopologyWorld | None = None
    ) -> Optional[np.ndarray]:
        """Barycentre des elements projetes d'un groupe Core."""
        projected_elements = self._get_projected_elements_for_core_group(
            core_group_id, world
        )
        if not projected_elements:
            return None
        sx = sy = 0.0
        n = 0
        for element in projected_elements:
            P = element.get("pts")
            if not isinstance(P, dict):
                continue
            for k in ("O", "B", "L"):
                sx += float(P[k][0])
                sy += float(P[k][1])
                n += 1
        if n == 0:
            return None
        return np.array([sx/n, sy/n], dtype=float)

    def _cancel_drag(self):
        # Mémoriser la source du drag avant de le remettre à zéro
        drag_info = self._drag
        self._drag = None

        self._delete_drag_previews()

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
        if self.compass_state.trace.active:
            self._clock_trace_cancel()
            return

        if self.compass_state.arc.active:
            self._clock_arc_cancel()
            return

        # Annule le mode de mesure d'azimut du compas
        if self.compass_state.measure.active:
            self._clock_measure_cancel()
            return

        # Annule le mode de définition d'azimut du compas
        if self.compass_state.set_ref.active:
            self._clock_setref_cancel()
            return

        if self.compass_state.dragging:
            wx, wy = self._screen_to_world(self.compass_state.cx, self.compass_state.cy)
            self._clock_clear_anchor_binding()
            self.compass_state.anchor_world = np.array([wx, wy], dtype=float)
            self.compass_state.dragging = False
            self.canvas.configure(cursor="")
            self._clock_clear_snap_target()
            self._clock_arc_clear_last()
            self._redraw_overlay_only()
            self._update_compass_ctx_menu_and_dico_state()
            self.status.config(text="D\u00e9placement du compas interrompu (ESC).")
            return "break"

        if self._drag:
            self._cancel_drag()
            self.status.config(text="Drag annulé (ESC).")
            return

        if self._sel:
            # rollback rotation de GROUPE
            if self._sel.get("mode") in (
                "rotate_group",
                "rotate_group_anchor_drag",
            ):
                if self._discard_manual_rotate_preview():
                    self._redraw_from(self._last_drawn)
                    self.status.config(text="Rotation de groupe annulée (ESC).")
                    self._reset_assist()
                    return "break"
                if self._discard_auto_transform_preview():
                    self._redraw_from(self._last_drawn)
                    self.status.config(text="Rotation auto annulée (ESC).")
                    self._reset_assist()
                    return "break"
                core_gid = self._sel.get("core_group_id")
                orig = self._sel.get("orig_group_pts")
                if core_gid and isinstance(orig, dict):
                    for tid, pts in orig.items():
                        if 0 <= tid < len(self._last_drawn):
                            self._last_drawn[tid]["pts"] = {k: np.array(pts[k].copy()) for k in ("O", "B", "L")}
                    self._redraw_from(self._last_drawn)
                self._sel = None
                self.status.config(text="Rotation de groupe annulée (ESC).")
                self._reset_assist()
                return
            # rollback déplacement de GROUPE
            if self._sel.get("mode") == "move_group":
                if not self._is_active_auto_scenario() and self._discard_manual_move_preview():
                    self._redraw_from(self._last_drawn)
                    self.status.config(text="Déplacement de groupe annulé (ESC).")
                    self._reset_assist()
                    return "break"
                if self._discard_auto_transform_preview():
                    self._redraw_from(self._last_drawn)
                    self.status.config(text="Déplacement de groupe annulé (ESC).")
                    self._reset_assist()
                    return "break"
                core_gid = self._sel.get("core_group_id")
                orig = self._sel.get("orig_group_pts")
                if core_gid and isinstance(orig, dict):
                    for tid, pts in orig.items():
                        if 0 <= tid < len(self._last_drawn):
                            self._last_drawn[tid]["pts"] = {k: np.array(pts[k].copy()) for k in ("O", "B", "L")}
                self._redraw_from(self._last_drawn)
                self._sel = None
                self.status.config(text="Déplacement de groupe annulé (ESC).")
                self._reset_assist()
                return
            # rollback déplacement/édition d'un triangle seul
            idx = self._sel.get("idx")
            orig = self._sel.get("orig_pts")
            if idx is not None and orig is not None and 0 <= idx < len(self._last_drawn):
                self._last_drawn[idx]["pts"] = {k: np.array(orig[k].copy()) for k in ("O", "B", "L")}
                self._redraw_from(self._last_drawn)
            self._sel = None
            self.status.config(text="Action annulée (rollback).")
            self._reset_assist()

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

    def _ctx_find_menu_index_by_label(self, label: str) -> Optional[int]:
        menu = getattr(self, "_ctx_menu", None)
        if menu is None:
            return None
        end = menu.index("end")
        if end is None:
            return None
        for i in range(int(end) + 1):
            try:
                if menu.type(i) != "command":
                    continue
                if str(menu.entrycget(i, "label")) == str(label):
                    return int(i)
            except tk.TclError:
                continue
        return None

    def _ctx_refresh_menu_runtime_indexes(self) -> None:
        menu = getattr(self, "_ctx_menu", None)
        if menu is None:
            self._ctx_idx_ol0 = None
            self._ctx_idx_bl0 = None
            return

        self._ctx_idx_ol0 = self._ctx_find_menu_index_by_label("OL=0°")
        self._ctx_idx_bl0 = self._ctx_find_menu_index_by_label("BL=0°")

    def _ctx_update_degrouper_menu_entry(self) -> None:
        menu = getattr(self, "_ctx_menu", None)
        if menu is None:
            return

        label = str(getattr(self, "_ctx_degrouper_label", "Dégrouper"))
        idx = self._ctx_find_menu_index_by_label(label)

        show_degrouper = False
        scen = self._get_active_scenario()
        world = scen.topoWorld
        gid = str(getattr(self, "ctxGroupId", "") or "")
        nid = str(getattr(self, "ctxStartNodeId", "") or "")
        if gid and nid:
            try:
                show_degrouper = bool(world.canDegrouperAtNode(gid, nid))
            except (ValueError, AssertionError):
                show_degrouper = False

        if show_degrouper and idx is None:
            menu.insert_command(1, label=label, command=self._ctx_degrouper)
        elif (not show_degrouper) and idx is not None:
            menu.delete(idx)

        self._ctx_refresh_menu_runtime_indexes()

    def _ctx_update_pivot_attachment_menu_entry(self) -> None:
        menu = getattr(self, "_ctx_menu", None)
        if menu is None:
            return

        label = str(
            getattr(
                self,
                "_ctx_pivot_attachment_label",
                "Pivoter l'attache",
            )
        )
        idx = self._ctx_find_menu_index_by_label(label)

        show_pivot = False

        scen = self._get_active_scenario()
        world = scen.topoWorld

        gid = str(getattr(self, "ctxGroupId", "") or "")
        nid = str(getattr(self, "ctxStartNodeId", "") or "")

        if gid and nid:
            show_pivot = bool(
                world.canPivotVertexEdgeAtNode(
                    gid,
                    nid,
                )
            )

        if show_pivot and idx is None:
            degrouper_idx = self._ctx_find_menu_index_by_label(
                self._ctx_degrouper_label
            )

            # Doit apparaître juste après "Dégrouper".
            insert_idx = (
                int(degrouper_idx) + 1
                if degrouper_idx is not None
                else 1
            )

            menu.insert_command(
                insert_idx,
                label=label,
                command=self._ctx_pivot_attachment,
            )

        elif (not show_pivot) and idx is not None:
            menu.delete(idx)

        self._ctx_refresh_menu_runtime_indexes()

    def _ctx_update_beacon_detach_menu_entry(self) -> None:
        """Affiche l'action de décrochage seulement pour un groupe ancré."""
        menu = self._ctx_menu
        label = "Décrocher le groupe de la balise"
        idx = self._ctx_find_menu_index_by_label(label)

        anchored = False
        element_id = self._ctx_target_element_id
        scen = self._get_active_scenario()
        world = scen.topoWorld
        if element_id:
            core_group_id = world.get_group_of_element(element_id)
            anchored = (
                core_group_id is not None
                and world.getAnchorForGroup(core_group_id) is not None
            )

        if anchored and idx is None:
            menu.insert_command(1, label=label, command=self._ctx_detach_group_from_beacon)
        elif not anchored and idx is not None:
            menu.delete(idx)
        self._ctx_refresh_menu_runtime_indexes()

    def _ctx_update_anchor_rotation_menu_state(self) -> None:
        """Applique les états contextuels des rotations pour un groupe ancré."""
        element_id = self._ctx_target_element_id
        scen = self._get_active_scenario()
        world = scen.topoWorld
        anchored = False
        if element_id:
            core_group_id = world.get_group_of_element(element_id)
            anchored = (
                core_group_id is not None
                and world.getAnchorForGroup(core_group_id) is not None
            )
        states = {
            "Pivoter": tk.DISABLED if anchored else tk.NORMAL,
            "Inverser": tk.DISABLED if anchored else tk.NORMAL,
            "OL=0°": tk.NORMAL,
            "BL=0°": tk.NORMAL,
        }
        for label, state in states.items():
            idx = self._ctx_find_menu_index_by_label(label)
            if idx is not None:
                self._ctx_menu.entryconfig(idx, state=state)

    def _ctx_detach_group_from_beacon(self) -> None:
        """Supprime l'ancre active puis décale le groupe depuis le Core."""
        element_id = self._ctx_target_element_id
        scen = self._get_active_scenario()
        world = scen.topoWorld
        if not element_id:
            raise RuntimeError("Décrochage: contexte Core absent")

        core_group_id = world.get_group_of_element(element_id)
        if core_group_id is None:
            raise RuntimeError("Décrochage: groupe Core introuvable")
        anchor = world.getAnchorForGroup(core_group_id)
        if anchor is None:
            raise RuntimeError("Décrochage: groupe non ancré")

        center_world = self._get_core_group_world_centroid(world, core_group_id)
        beacon_world = np.asarray(world.getBeaconWorldXY(anchor.beacon_id), dtype=float)
        center_x, center_y = self._world_to_screen(center_world)
        beacon_x, beacon_y = self._world_to_screen(beacon_world)
        direction = np.array((center_x - beacon_x, center_y - beacon_y), dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            dx_screen, dy_screen = 0.0, 30.0
        else:
            dx_screen, dy_screen = 30.0 * direction / norm

        world.removeGroupAnchor(anchor.anchor_id)
        self._move_core_group_by_screen_delta(
            world, core_group_id, float(dx_screen), float(dy_screen)
        )
        self._ctx_target_element_id = None
        self._sel = None
        self._reset_assist()
        self._redraw_from(self._last_drawn)
        self.refreshCheminTreeView()
        self.status.config(text="Groupe décroché de la balise.")

    # ---------- Clic droit / menu contextuel ----------
    def _on_canvas_right_click(self, event):
        if self._deformation_state.active:
            self.status.config(text="Actions structurelles indisponibles en mode deformation.")
            return "break"
        """Affiche le menu contextuel si un triangle est cliqué (intérieur ou sommet)."""
        self._hide_tooltip()
        # Pas de menu si on est en train de drag depuis la liste
        if self._drag:
            return
        # Si clic droit sur le compas : menu dédié (même si aucun triangle)
        if self._is_point_in_clock(event.x, event.y):
            self._ctx_target_element_id = None
            self._ctx_last_rclick = (event.x, event.y)
            self._ctx_clear_chemin_context()
            self._update_compass_ctx_menu_and_dico_state()
            self._ctx_menu_compass.tk_popup(event.x_root, event.y_root)
            self._ctx_menu_compass.grab_release()
            return

        mode, idx, extra = self._hit_test(event.x, event.y)
        if idx is None:
            self._ctx_target_element_id = None
            self._ctx_clear_chemin_context()
            return
        # On ne propose Supprimer que si on est sur un triangle
        if mode in ("center", "vertex"):
            entry = self._last_drawn[idx]
            element_id = str(entry.get("topoElementId", "") or "").strip()
            scen = self._get_active_scenario()
            world = scen.topoWorld
            if not element_id or world is None or element_id not in world.elements:
                self._ctx_target_element_id = None
                self._ctx_clear_chemin_context()
                return
            self._ctx_target_element_id = element_id
            self._ctx_last_rclick = (event.x, event.y)
            self._ctx_nearest_vertex_key = self._ctx_compute_nearest_vertex_key(element_id, event.x, event.y)
            try:
                self._ctx_capture_chemin_context(element_id, event.x, event.y)
            except (ValueError, RuntimeError):
                self._ctx_clear_chemin_context()

            # Actions topologiques conditionnelles selon le node cliqué.
            self._ctx_update_degrouper_menu_entry()
            self._ctx_update_pivot_attachment_menu_entry()
            self._ctx_update_beacon_detach_menu_entry()
            self._ctx_update_anchor_rotation_menu_state()

            # (ré)construire la section "mot" du menu selon le triangle visé + selection dico
            self._ctx_menu.tk_popup(event.x_root, event.y_root)
            self._ctx_menu.grab_release()

    def _ctx_take_target_element_id(self) -> Optional[str]:
        """Consomme l'ElementID Core mémorisé par le menu contextuel."""
        element_id = str(getattr(self, "_ctx_target_element_id", "") or "").strip()
        self._ctx_target_element_id = None
        return element_id or None

    def _ctx_compute_nearest_vertex_key(self, element_id: str, sx: float, sy: float) -> str:
        """Retourne 'O'/'B'/'L' du sommet le plus proche du point écran (sx,sy)."""
        tri = self.canvas_objects.get_by_topology_id(str(element_id or "").strip())
        if tri is None:
            return "L"
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

    def _ctx_capture_chemin_context(self, element_id: str, sx: float, sy: float) -> None:
        """
        Mémorise ctxGroupId + ctxStartNodeId depuis le clic droit.

        Règle V3: contexte minimal seulement (groupId Core + startNodeId DSU),
        sans lecture des structures boundary internes.
        """
        scen = self._get_active_scenario()
        world = scen.topoWorld

        topo_element_id = str(element_id or "").strip()
        if not topo_element_id or topo_element_id not in world.elements:
            raise ValueError("topoElementId manquant pour le triangle de contexte")

        # MIG-GEO-011-A : triangle projete -> element Core -> groupe DSU.
        # Le contexte de chemin ne consulte ni self.groups ni la chaine nodes.
        try:
            core_gid = world.get_group_of_element(topo_element_id)
        except Exception as exc:
            raise ValueError(f"groupe Core introuvable pour {topo_element_id}") from exc

        segments = world.getBoundarySegments(core_gid)
        if not segments:
            raise ValueError("frontière vide")
        boundary_nodes = sorted(
            {seg.fromNodeId for seg in segments}.union({seg.toNodeId for seg in segments})
        )
        if not boundary_nodes:
            raise ValueError("aucun noeud de frontière")

        wx, wy = self._screen_to_world(float(sx), float(sy))
        best_node = None
        best_d2 = None
        for nid in boundary_nodes:
            px, py = world.getConceptNodeWorldXY(nid, core_gid)
            dx = float(px) - float(wx)
            dy = float(py) - float(wy)
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_node = world.find_node(nid)
        if best_node is None:
            raise RuntimeError("impossible de résoudre le noeud de départ")

        self.ctxGroupId = core_gid
        self.ctxStartNodeId = best_node

    def _ctx_CreerChemin(self) -> None:
        if self._deformation_state.active:
            self.status.config(text="Creation de chemin indisponible en mode deformation.")
            return
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

        if world.topologyChemins.isDefined:
            if not messagebox.askokcancel("Créer un chemin", "Un chemin existe déjà. Remplacer ?"):
                return
        world.topologyChemins.creerDepuisGroupe(
            gid,
            startNodeId,
            orientationUser,
            self._getCheminsBeaconRefId(),
        )
        self.refreshCheminTreeView()

    def _ctx_degrouper(self) -> None:
        if self._deformation_state.active:
            self.status.config(text="Degroupage indisponible en mode deformation.")
            return
        core_gid = self.ctxGroupId
        nodeId = self.ctxStartNodeId
        if not core_gid or not nodeId:
            messagebox.showerror("Dégrouper", "Dégrouper impossible : contexte invalide.")
            return

        scen = self._get_active_scenario()
        world = scen.topoWorld

        res = world.degrouperAtNode(core_gid, nodeId)
        self._applyDegrouperResultToTk(res)

    def _ctx_pivot_attachment(self) -> None:
        if self._deformation_state.active:
            self.status.config(
                text="Pivot d'attache indisponible en mode deformation."
            )
            return

        core_gid = self.ctxGroupId
        node_id = self.ctxStartNodeId

        if not core_gid or not node_id:
            raise RuntimeError(
                "Pivot d'attache impossible : contexte de clic droit invalide"
            )

        scen = self._get_active_scenario()
        world = scen.topoWorld

        new_world = world.pivotVertexEdgeAtNode(
            core_gid,
            node_id,
        )

        scen.topoWorld = new_world
        world = new_world

        new_group_id = world.get_group_of_element(
            self._ctx_target_element_id
        )

        # Le Core est l'autorité : on reconstruit intégralement la projection.
        self._rebuild_active_projection_from_core()

        self._sel = None
        self._reset_assist()
        self._invalidate_pick_cache()
        self._redraw_from(self._last_drawn)
        self.refreshCheminTreeView()

        self.status.config(
            text=f"Attache Vertex-Edge pivotée ({new_group_id})."
        )

    def _degrouperGroupScreenBBox(self, core_group_id: str) -> Optional[Tuple[float, float, float, float]]:
        projected_elements = self._get_projected_elements_for_core_group(core_group_id)
        if not projected_elements:
            return None

        xs: list[float] = []
        ys: list[float] = []
        for element in projected_elements:
            P = element.get("pts")
            if not isinstance(P, dict):
                continue
            for k in ("O", "B", "L"):
                sx, sy = self._world_to_screen(P[k])
                xs.append(float(sx))
                ys.append(float(sy))
        if not xs or not ys:
            return None
        return (min(xs), min(ys), max(xs), max(ys))

    def _screen_delta_to_world_delta(self, dx_screen: float, dy_screen: float) -> np.ndarray:
        """Convertit un delta écran en delta monde sans toucher la projection.

        Le layout post-dégroupage reste exprimé en pixels pour préserver son
        comportement historique. Sa mutation, elle, appartient désormais au
        Core : l'appelant transmet ce delta à ``TopologyWorld.move_group``.
        """
        dxs = float(dx_screen)
        dys = float(dy_screen)
        if not np.isfinite(dxs) or not np.isfinite(dys):
            raise ValueError("Dégrouper: delta écran non fini")

        wx0, wy0 = self._screen_to_world(0.0, 0.0)
        wx1, wy1 = self._screen_to_world(dxs, dys)
        return np.array([float(wx1 - wx0), float(wy1 - wy0)], dtype=float)

    def _move_core_group_by_screen_delta(
        self,
        world: TopologyWorld,
        core_group_id: str,
        dx_screen: float,
        dy_screen: float,
    ) -> None:
        """Déplace un groupe Core selon un delta écran, puis le reprojette."""
        delta_world = self._screen_delta_to_world_delta(dx_screen, dy_screen)
        world.move_group(
            core_group_id,
            float(delta_world[0]),
            float(delta_world[1]),
        )
        self._project_core_group_to_last_drawn(world, core_group_id)

    def _applyDegrouperResultToTk(self, res: dict) -> None:
        if not isinstance(res, dict):
            raise ValueError("Dégrouper: résultat Core invalide (dict attendu)")

        scen = self._get_active_scenario()
        world = scen.topoWorld

        main_core_gid = str(res.get("mainGroupId", "") or "").strip()
        new_core_gids = [
            str(x)
            for x in list(res.get("newGroupIds", []) or [])
            if str(x or "").strip()
        ]

        if not main_core_gid:
            raise ValueError("Dégrouper: mainGroupId absent")

        # Vérifier que les groupes retournés existent réellement dans le Core.
        if not world.getGroupElementIds(main_core_gid):
            raise ValueError(
                f"Dégrouper: groupe Core principal introuvable ({main_core_gid})"
            )

        valid_new_core_gids = []
        for core_gid in new_core_gids:
            if world.getGroupElementIds(core_gid):
                valid_new_core_gids.append(core_gid)

        # MIG-CACHE-TRANSFORM-001F: le Core vient de scinder la topologie.
        # Repartir systématiquement de sa géométrie avant le layout écran.
        affected_core_gids = [main_core_gid] + valid_new_core_gids
        for core_gid in affected_core_gids:
            self._project_core_group_to_last_drawn(world, core_gid)

        # Décalage visuel des groupes nouvellement détachés.
        bbox_main = self._degrouperGroupScreenBBox(main_core_gid)

        if bbox_main is not None:
            min_x_m, min_y_m, max_x_m, max_y_m = bbox_main

            ordered_new_core_gids = sorted(
                valid_new_core_gids,
                key=lambda core_gid: len(world.getGroupElementIds(core_gid)),
            )

            for core_gid in ordered_new_core_gids:
                if world.getAnchorForGroup(core_gid) is not None:
                    continue

                bbox_group = self._degrouperGroupScreenBBox(core_gid)
                if bbox_group is None:
                    continue

                min_x_g, min_y_g, max_x_g, max_y_g = bbox_group

                dx_screen = 0.0
                dy_screen = 0.0

                if max_x_g <= min_x_m:
                    dx_screen = 30.0
                elif max_y_g <= min_y_m:
                    dy_screen = 30.0
                elif min_x_g >= max_x_m:
                    dx_screen = -30.0
                else:
                    dy_screen = 30.0

                self._move_core_group_by_screen_delta(
                    world, core_gid, dx_screen, dy_screen
                )

        self._sel = None
        self._reset_assist()
        self._redraw_from(self._last_drawn)
        self.refreshCheminTreeView()

        self.status.config(
            text=(
                f"Dégrouper : {1 + len(valid_new_core_gids)} groupes Core "
                f"({len(valid_new_core_gids)} nouveau(x))."
            )
        )

    def _is_point_in_clock(self, sx: float, sy: float) -> bool:
        """True si (sx,sy) est dans le disque du compas (coords canvas)."""
        return bool(self.show_clock_overlay and self.show_clock_overlay.get()) and self.compass_controller.contains_point(sx, sy, pad=6)
    def _ctx_define_clock_ref_azimuth(self):
        """Entrée de menu du mode de définition d'azimut de référence."""
        self.compass_controller.start_set_ref(*self._clock_get_initial_cursor_xy())
    def _ctx_trace_clock_azimuth(self):
        """Entrée de menu du mode de tracé d'azimut."""
        self._clock_arc_cancel(silent=True)
        self.compass_controller.start_trace(*self._clock_get_initial_cursor_xy())
    def _ctx_clear_clock_azimuth_traits(self):
        scen = self._get_active_scenario()
        if scen is None:
            return
        hit = self._clock_get_anchor_node_hit()
        if hit is None:
            return

        nodeId = hit["nodeId"]
        topoGroupId = hit["groupId"]
        before = len(scen.clockAzimuthTraits)
        scen.clockAzimuthTraits = [
            g for g in scen.clockAzimuthTraits
            if not (g["nodeId"] == nodeId and g["topoGroupId"] == topoGroupId)
        ]
        after = len(scen.clockAzimuthTraits)
        if before == after:
            return
        self._redraw_from(self._last_drawn)
        self._update_compass_ctx_menu_and_dico_state()
        self.status.config(text="Guides effacés pour ce nœud.")

    def _clock_trace_update_preview(self, sx: int, sy: int):
        self.compass_controller.update_trace(sx, sy)
    def _clock_trace_confirm(self):
        self.compass_controller.confirm_trace()
    def _clock_trace_cancel(self, silent: bool = False):
        self.compass_controller.cancel_trace(silent=silent)
    def _ctx_measure_clock_azimuth(self):
        """Entrée de menu du mode de mesure d'azimut."""
        self.compass_controller.start_measure(*self._clock_get_initial_cursor_xy())
    def _ctx_measure_clock_arc_angle(self):
        """Entrée de menu du mode de mesure d'arc."""
        self.compass_controller.start_arc()
    def _ctx_filter_dictionary_by_clock_arc(self):
        """Filtre visuellement le dictionnaire selon l'angle mesuré."""
        if not self.dictionary_panel.is_loaded:
            messagebox.showinfo("Filtrer le dictionnaire", "Le dictionnaire n'est pas affiché.")
            return
        ref = self.compass_state.arc.last_angle_deg
        if ref is None:
            messagebox.showinfo("Filtrer le dictionnaire", "Aucun arc n'a été mesuré.\n\nMesure d'abord un arc d'angle sur le compas.")
            return
        self.dictionary_panel.apply_angle_filter(float(ref))
        self._update_compass_ctx_menu_and_dico_state()
        self.status.config(text=f"Dico filtré (angle ref={float(ref):0.0f}°, tol=±2°)")

    def _simulation_cancel_dictionary_filter(self):
        if self.dictionary_panel.clear_angle_filter():
            self.status.config(text="Dico: filtrage annule")
        self._update_compass_ctx_menu_and_dico_state()

    def _ctx_compass_find_entry_index(self, label: str) -> int | None:
        menu = self._ctx_menu_compass
        if menu is None:
            return None
        end = menu.index("end")
        if end is None:
            return None
        for i in range(int(end) + 1):
            if menu.type(i) != "command":
                continue
            if str(menu.entrycget(i, "label")) == str(label):
                return int(i)
        return None

    def _update_compass_ctx_menu_traits_state(self):
        menu = self._ctx_menu_compass
        if menu is None:
            return

        clear_label = "Effacer les traits"
        clear_idx = self._ctx_compass_find_entry_index(clear_label)
        if clear_idx is None:
            trace_idx = self._ctx_compass_find_entry_index("Tracer un azimut…")
            arc_idx = self._ctx_compass_find_entry_index("Mesurer un arc d'angle…")
            if trace_idx is not None:
                insert_idx = int(trace_idx) + 1
            elif arc_idx is not None:
                insert_idx = int(arc_idx)
            else:
                insert_idx = 1
            menu.insert_command(
                insert_idx,
                label=clear_label,
                command=self._ctx_clear_clock_azimuth_traits,
                state=tk.DISABLED,
            )
            clear_idx = self._ctx_compass_find_entry_index(clear_label)

        enabled = False
        scen = self._get_active_scenario()
        hit = self._clock_get_anchor_node_hit()
        if scen is not None and hit is not None:
            nodeId = str(hit["nodeId"])
            topoGroupId = str(hit["groupId"])
            for guide in scen.clockAzimuthTraits:
                if guide["nodeId"] == nodeId and guide["topoGroupId"] == topoGroupId:
                    enabled = True
                    break

        if clear_idx is not None:
            menu.entryconfig(int(clear_idx), state=(tk.NORMAL if enabled else tk.DISABLED))

        self._ctx_compass_idx_clear_traits = self._ctx_compass_find_entry_index(clear_label)
        self._ctx_compass_idx_filter_dico = self._ctx_compass_find_entry_index("Filtrer le dictionnaire…")
        self._ctx_compass_idx_cancel_dico_filter = self._ctx_compass_find_entry_index("Annuler le filtrage")

    def _update_compass_ctx_menu_and_dico_state(self):
        """Synchronise le menu compas et l'état de filtrage du dico selon la dispo de l'arc.

        IMPORTANT:
        - Le dico doit rester sélectionnable même s'il n'y a pas d'arc mesuré.
        - Seule l'action "Filtrer le dictionnaire…" dépend de l'existence d'un arc.
        """
        self._clock_auto_ref_sync_var.set(bool(self._clock_auto_ref_sync_enabled))
        has_trace_ref = bool(self._clock_get_anchor_node_hit() is not None)
        menu = self._ctx_menu_compass
        trace_idx = self._ctx_compass_find_entry_index("Tracer un azimut…")
        if menu is not None and trace_idx is not None:
            menu.entryconfig(trace_idx, state=(tk.NORMAL if has_trace_ref else tk.DISABLED))

        arc_ok = bool(self._clock_arc_is_available())

        idx = self._ctx_compass_find_entry_index("Filtrer le dictionnaire…")
        if menu is not None and idx is not None:
            menu.entryconfig(idx, state=(tk.NORMAL if arc_ok else tk.DISABLED))

        # Activer/désactiver "Annuler le filtrage" selon l'état courant
        idx_cancel = self._ctx_compass_find_entry_index("Annuler le filtrage")
        if menu is not None and idx_cancel is not None:
            menu.entryconfig(idx_cancel, state=(tk.NORMAL if self.dictionary_panel.filter_active else tk.DISABLED))

        # Le dico reste sélectionnable dans tous les cas.
        self.dictionary_panel.set_selection_enabled(True)

        # Si on perd l'arc alors qu'un filtrage était actif, on annule le filtrage.
        # IMPORTANT: ne pas appeler _simulation_cancel_dictionary_filter() si aucun filtrage n'est actif,
        # sinon recursion infinie (cancel -> update -> cancel -> ...).
        if not arc_ok and self.dictionary_panel.filter_active:
            self._simulation_cancel_dictionary_filter()

        self._update_compass_ctx_menu_traits_state()

    def _azimuth_world_deg(self, a, b) -> float:
        return azimuth_world_deg(a, b)
    def _clock_delta_display_deg(self, deltaAzDeg: float) -> float:
        return self.compass_controller.delta_display_deg(deltaAzDeg)
    def _clock_delta_display_text(self, deltaAzDeg: float) -> str:
        return self.compass_controller.delta_display_text(deltaAzDeg)
    def _clock_point_on_circle(self, az_deg: float, radius: float):
        """Point écran (sx,sy) à un azimut donné autour du centre du compas."""
        return self.compass_controller.point_on_circle(az_deg, radius)
    def _clock_apply_optional_snap(self, sx: int, sy: int, *, enable_snap: bool) -> Tuple[int, int]:
        return self.compass_controller.apply_optional_snap(sx, sy, enable_snap=enable_snap)

    def _clock_update_snap_target(self, sx: float, sy: float):
        self.compass_controller.update_snap_target(sx, sy)

    def _clock_arc_auto_from_snap_target(self, snap_target: dict, drag: bool, prevNodeDsu=None, nextNodeDsu=None):
        return self.compass_controller.auto_arc_from_snap_target(snap_target, drag, prevNodeDsu, nextNodeDsu)

    def _clock_arc_is_available(self) -> bool:
        return self.compass_controller.arc_is_available()

    def _clock_arc_handle_click(self, sx: int, sy: int):
        self.compass_controller.handle_arc_click(sx, sy)

    def _clock_arc_update_preview(self, sx: int, sy: int):
        self.compass_controller.update_arc_preview(sx, sy)

    def _clock_arc_cancel(self, silent: bool = False):
        self.compass_controller.cancel_arc(silent=silent)

    def _clock_arc_clear_last(self):
        self.compass_controller.clear_arc_last()
    def _clock_measure_update_preview(self, sx: int, sy: int):
        self.compass_controller.update_measure(sx, sy)
    def _clock_measure_confirm(self):
        self.compass_controller.confirm_measure()
    def _clock_measure_cancel(self, silent: bool = False):
        self.compass_controller.cancel_measure(silent=silent)
    def _clock_compute_azimuth_deg(self, sx: float, sy: float) -> float:
        """Azimut (degrés) depuis le centre du compas vers (sx,sy)."""
        return self.compass_controller.compute_azimuth_deg(sx, sy)
    def _clock_angle_diff_deg(self, a: float, b: float) -> float:
        return clock_angle_diff_deg(a, b)
    def _clock_clear_anchor_binding(self):
        self.compass_controller.clear_anchor_binding()
    def _clock_bind_anchor_to_node(
        self,
        *,
        node_id: str,
        topo_group_id: str,
        idx: int | None = None,
        vkey: str | None = None,
        world_pos=None,
    ):
        self.compass_controller.bind_anchor_to_node(
            node_id=node_id, topo_group_id=topo_group_id, idx=idx, vkey=vkey, world_pos=world_pos,
        )
    def _clock_bind_anchor_from_snap_target(self, snap_target: dict | None):
        self.compass_controller.bind_anchor_from_snap_target(snap_target)
    def _clock_refresh_anchor_world_from_binding(self):
        self.compass_controller.refresh_anchor_world_from_binding()
    def _clock_compute_ref_azimuth_from_balise(self) -> float | None:
        return self.compass_controller.compute_ref_azimuth_from_selected_beacon()
    def _clock_apply_auto_ref_sync(self):
        self.compass_controller.apply_auto_ref_sync()
    def _clock_get_center_world(self) -> Optional[np.ndarray]:
        return self.compass_controller.get_center_world()
    def _clock_get_anchor_node_hit(self) -> dict | None:
        return self.compass_controller.get_anchor_node_hit()
    def _clock_collect_setref_candidates(self) -> list[dict]:
        return self.compass_controller.collect_set_ref_candidates()
    def _clock_pick_setref_snap_candidate(self, azMouseAbs: float, candidates: list[dict]) -> dict | None:
        return self.compass_controller.pick_set_ref_snap_candidate(azMouseAbs, candidates)
    def _clock_clear_setref_snap_target(self):
        self.compass_controller.clear_set_ref_snap_target()
    def _clock_draw_setref_snap_target(self, snap_target: dict | None):
        self.compass_controller.draw_set_ref_snap_target(snap_target)
    def _clock_build_setref_preview(self, sx: int, sy: int) -> dict | None:
        return self.compass_controller.build_set_ref_preview(sx, sy)
    def _clock_compute_theoretical_ref_azimuth_deg(
        self,
        *,
        az1: float,
        az2: float,
        ang_hour_0: float,
        ang_min_0: float,
    ) -> float:
        return clock_theoretical_ref_azimuth_deg(
            az1=az1, az2=az2, ang_hour_0=ang_hour_0, ang_min_0=ang_min_0,
        )
    def _clock_get_initial_cursor_xy(self) -> Tuple[int, int]:
        """Point de départ pour les modes compas: clic droit si dispo, sinon position souris."""
        if self._ctx_last_rclick:
            sx, sy = self._ctx_last_rclick
            return int(sx), int(sy)
        # coords canvas sous la souris
        sx = int(self.canvas.winfo_pointerx() - self.canvas.winfo_rootx())
        sy = int(self.canvas.winfo_pointery() - self.canvas.winfo_rooty())
        return sx, sy

    def _clock_clamp_preview_text_xy(self, sx: int, sy: int) -> tuple[int, int]:
        return self.compass_controller.clamp_preview_text_xy(sx, sy)
    def _clock_update_azimuth_preview(
        self, sx: int, sy: int, *, line_id: Optional[int], text_id: Optional[int],
        preview_tag: str, relative_to_ref: bool, enable_snap: bool, draw_line: bool = True,
        label_text: str | None = None, line_fill: str = "#202020",
        line_dash: tuple[int, int] | None = (4, 3), text_fill: str = "#202020",
    ):
        return self.compass_controller.update_azimuth_preview(
            sx, sy, line_id=line_id, text_id=text_id, preview_tag=preview_tag,
            relative_to_ref=relative_to_ref, enable_snap=enable_snap, draw_line=draw_line,
            label_text=label_text, line_fill=line_fill, line_dash=line_dash, text_fill=text_fill,
        )
    def _clock_setref_update_preview(self, sx: int, sy: int):
        self.compass_controller.update_set_ref(sx, sy)
    def _clock_setref_confirm(self, sx: int, sy: int):
        self.compass_controller.confirm_set_ref(sx, sy)
    def _clock_setref_cancel(self, silent: bool = False):
        self.compass_controller.cancel_set_ref(silent=silent)
    def _ctx_delete_group(self):
        if self._deformation_state.active:
            self.status.config(text="Suppression indisponible en mode deformation.")
            return
        """Supprime **tout le groupe** du triangle ciblé, réinsère les triangles dans la liste,
        puis remappe les tids restants."""
        selected_element_id = self._ctx_take_target_element_id()
        if selected_element_id is None:
            return

        world = self._get_active_scenario().topoWorld
        if world is None or selected_element_id not in world.elements:
            return
        core_group_id = world.get_group_of_element(selected_element_id)
        removed_element_ids = [
            str(element_id)
            for element_id in world.getGroupElementIds(core_group_id)
        ]
        if not removed_element_ids:
            return
        removed_tids = sorted(
            {
                tid
                for element_id in removed_element_ids
                for tid in (self.canvas_objects.get_index_by_topology_id(element_id),)
                if tid is not None
            },
            reverse=True,
        )
        if not removed_tids:
            return

        # --- Confirmation si le groupe comporte au moins 2 triangles ---
        if len(removed_tids) >= 2:
            confirm = messagebox.askyesno(
                "Supprimer le groupe",
                "Voulez-vous supprimer le groupe ?"
            )
            if not confirm:
                return

        # 0) La collection conserve le remappage structurel old tid -> new tid.
        self.canvas_objects.remove_many(removed_tids)

        # 2bis) TOPO : supprimer les éléments Core + purge des attaches + rebuild
        if world is not None and removed_element_ids:
            world.removeElementsAndRebuild(list(removed_element_ids))
        self._rebuild_triangle_listbox_from_core()

        # 4) Fin : purge sélection/assist et redraw
        self._sel = None
        self._reset_assist()
        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Groupe supprimé (gid={core_group_id}, {len(removed_tids)} triangle(s)).")

    def _ctx_rotate_selected(self):
        """Passe en mode rotation autour du barycentre pour le triangle ciblé."""
        element_id = self._ctx_take_target_element_id()
        if element_id is None:
            return
        scen = self._get_active_scenario()
        world = scen.topoWorld
        if element_id not in world.elements:
            return
        # le triangle fait toujours partie d'un groupe : PIVOTER LE GROUPE
        core_group_id = world.get_group_of_element(element_id)
        if not core_group_id:
            return

        # AUTO: pivot imposé par l'ancre ; MANUAL: barycentre groupe.
        auto_geom = self._is_active_auto_scenario()
        if auto_geom:
            anchor = world.getAnchorForGroup(core_group_id)
            if anchor is None:
                raise RuntimeError("Simulation AUTO: ancre absente pour la rotation")
            pivot = np.asarray(world.getBeaconWorldXY(anchor.beacon_id), dtype=float)
        else:
            pivot = self._group_centroid(core_group_id)
            if pivot is None:
                return
        # angle de départ = angle (pivot -> curseur au clic droit)
        if self._ctx_last_rclick:
            sx, sy = self._ctx_last_rclick
        else:
            sx, sy = self._world_to_screen(pivot)
        wx = (sx - self.offset[0]) / self.zoom
        wy = (self.offset[1] - sy) / self.zoom
        start_angle = math.atan2(wy - pivot[1], wx - pivot[0])
        if auto_geom:
            # MIG-CACHE-TRANSFORM-001H: aperçu exclusivement dans le cache
            # actif ; Core et état partagé restent inchangés jusqu'au commit.
            self._sel = {
                "mode": "rotate_group",
                "core_group_id": core_group_id,
                "pivot": np.array(pivot, dtype=float),
                "start_angle": start_angle,
                "auto_geom": True,
                "auto_state0": dict(self.auto_rotation_state or {"thetaDeg": 0.0}),
                "auto_preview_initial_pts": self._capture_active_auto_preview_pts(),
            }
        else:
            self._sel = {
                "mode": "rotate_group",
                "core_group_id": core_group_id,
                "pivot_world": np.array(pivot, dtype=float),
                "mouse_angle_start": start_angle,
                "rotate_preview_initial_pts": self._capture_move_preview_initial_pts(
                    world, str(core_group_id)
                ),
                "auto_geom": False,
            }
        self.status.config(
            text=(
                f"Mode pivoter GROUPE #{core_group_id} : bouge la souris pour tourner, "
                "clic gauche pour valider, ESC pour annuler."
            )
        )
        return

    def _ctx_orient_segment_north(self, from_key: str, to_key: str, status_label: str):
        """
        Oriente automatiquement le TRIANGLE ou le GROUPE pour que l'azimut du segment
        (from_key -> to_key) du triangle cliqué soit 0° = vers le Nord (axe +Y en coords monde).

        - Triangle seul : rotation autour du barycentre du triangle cliqué.
        - Groupe : rotation RIGIDE de tout le groupe autour du barycentre du triangle cliqué.
        """
        element_id = self._ctx_take_target_element_id()
        if element_id is None:
            return

        scen = self._get_active_scenario()
        world = scen.topoWorld
        if element_id not in world.elements:
            raise ValueError("[MIG-GEO-001] ElementID contextuel invalide")
        # --- CAS AUTO : rotation globale partagée ---
        if scen.source_type == "auto":
            points_world = self._get_core_triangle_world_points(world, element_id)
            segment = points_world[to_key] - points_world[from_key]
            if float(np.hypot(segment[0], segment[1])) < 1e-12:
                return
            dtheta = (math.pi / 2.0) - math.atan2(segment[1], segment[0])
            self._rotate_all_auto_scenarios_around_anchors(dtheta)

            self._redraw_from(self._last_drawn)
            self.status.config(text=f"Orientation appliquée : AUTO — {status_label} au Nord (0°).")
            return

        # --- CAS MANUEL : lecture et commit exclusivement depuis le Core ---
        points_world = self._get_core_triangle_world_points(world, element_id)
        segment_start = points_world[from_key]
        segment_end = points_world[to_key]
        dx = float(segment_end[0] - segment_start[0])
        dy = float(segment_end[1] - segment_start[1])
        if math.hypot(dx, dy) <= 1e-12:
            raise ValueError(
                f"[MIG-CACHE-TRANSFORM-001C2] segment dégénéré: {from_key}->{to_key}"
            )
        current_angle_deg = math.degrees(math.atan2(dy, dx))
        target_angle_deg = 90.0
        angle_deg = (target_angle_deg - current_angle_deg + 180.0) % 360.0 - 180.0
        angle_rad = math.radians(angle_deg)
        if not math.isfinite(angle_rad):
            raise ValueError("[MIG-CACHE-TRANSFORM-001C2] angle de rotation invalide")

        core_group_id = world.get_group_of_element(element_id)
        if core_group_id is None:
            raise ValueError(
                f"[MIG-CACHE-TRANSFORM-001C2] groupe Core introuvable element={element_id!r}"
            )
        anchor = world.getAnchorForGroup(core_group_id)
        if anchor is None:
            pivot = (
                points_world["O"] + points_world["B"] + points_world["L"]
            ) / 3.0
        else:
            pivot = np.asarray(
                world.getBeaconWorldXY(anchor.beacon_id), dtype=float
            )
        if pivot.shape != (2,) or not np.all(np.isfinite(pivot)):
            raise ValueError("[MIG-CACHE-TRANSFORM-001C2] pivot monde invalide")

        world.rotate_group(core_group_id, pivot, angle_rad)
        final_core_group_id = world.get_group_of_element(element_id)
        if final_core_group_id is None:
            raise ValueError(
                f"[MIG-CACHE-TRANSFORM-001C2] groupe Core final introuvable element={element_id!r}"
            )
        self._project_core_group_to_last_drawn(world, str(final_core_group_id))
        self._invalidate_pick_cache()
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
        clicked_element_id = self._ctx_take_target_element_id()
        if clicked_element_id is None:
            return

        ok = messagebox.askyesno(
            "Filtrer les scénarios",
            "Cette action va supprimer définitivement les scénarios automatiques incompatibles.\n\nContinuer ?"
        )
        if not ok:
            return

        self._filter_auto_scenarios_by_prefix_edges(clicked_element_id)

    def _scenario_prefix_edge_steps(self, scen, upto_index: int):
        """Étapes Core du préfixe, depuis ``orderedElementIds`` et les attachments.

        Chaque etape contient toutes les contraintes canoniques reliant deux
        triangles consecutifs. Aucun champ ``groups/nodes`` ou ``edge_in/out``
        n'est consulte dans ce chemin fonctionnel.
        """
        world = scen.topoWorld
        element_ids = [
            str(element_id or "").strip()
            for element_id in (getattr(scen, "orderedElementIds", None) or [])
        ]
        if (
            not element_ids
            or any(not element_id for element_id in element_ids)
            or len(set(element_ids)) != len(element_ids)
            or any(element_id not in world.elements for element_id in element_ids)
            or upto_index < 0
            or upto_index >= len(element_ids)
        ):
            return None
        return build_topology_prefix_steps(world, element_ids, upto_index)

    def _filter_auto_scenarios_by_prefix_edges(self, clicked_element_id: str):
        """
        Filtre les scénarios automatiques en conservant uniquement ceux
        qui commencent par le même préfixe de triangles ET les mêmes contraintes
        Attachments V2 entre triangles consécutifs, jusqu'au triangle cliqué.

        Le scénario actif est la référence absolue et ne peut jamais être supprimé.
        """
        if not self.scenarios or not (0 <= self.active_scenario_index < len(self.scenarios)):
            return

        active = self.scenarios[self.active_scenario_index]
        if getattr(active, "source_type", "manual") != "auto":
            return  # filtrage auto uniquement

        active_order = [
            str(element_id or "").strip()
            for element_id in (getattr(active, "orderedElementIds", None) or [])
        ]
        if not clicked_element_id or clicked_element_id not in active_order:
            # Impossible par design → bug
            raise RuntimeError(f"[FILTER] ElementID {clicked_element_id!r} absent du scénario actif")

        upto_index = active_order.index(clicked_element_id)
        reference_prefix = active_order[: upto_index + 1]

        ref_steps = self._scenario_prefix_edge_steps(active, upto_index)
        if ref_steps is None:
            raise RuntimeError("[FILTER] Données topologiques manquantes dans le scénario actif")

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
            candidate_order = [
                str(element_id or "").strip()
                for element_id in (getattr(scen, "orderedElementIds", None) or [])
            ]
            if candidate_order[: len(reference_prefix)] != reference_prefix:
                removed += 1
                continue

            # 2) mêmes contraintes d'attachment V2 dans le préfixe
            steps = self._scenario_prefix_edge_steps(scen, upto_index)
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
        if self._deformation_state.active:
            self.status.config(text="Inversion indisponible en mode deformation.")
            return
        """
        Inverse **tout le GROUPE** par symétrie axiale rigide.
        Axe = direction (O→L) du triangle ciblé ; la droite passe par le **barycentre du groupe**.
        La transformation et l'état ``mirrored`` sont autoritaires dans le Core.
        """
        element_id = self._ctx_take_target_element_id()
        if element_id is None:
            return
        if self._is_active_auto_scenario():
            self.status.config(text="Inversion désactivée pour les scénarios automatiques.")
            return
        scen = self._get_active_scenario()
        world = scen.topoWorld

        if element_id not in world.elements:
            raise ValueError("[MIG-CACHE-TRANSFORM-001D] topoElementId absent")
        core_group_id = world.get_group_of_element(element_id)
        if core_group_id is None:
            raise ValueError(
                f"[MIG-CACHE-TRANSFORM-001D] groupe Core introuvable element={element_id!r}"
            )

        # Axe = O→L du triangle ciblé, lu depuis le Core.
        points_world = self._get_core_triangle_world_points(world, element_id)
        axis = np.asarray(points_world["L"] - points_world["O"], dtype=float)
        nrm = float(np.hypot(axis[0], axis[1]))
        if nrm < 1e-12:
            raise ValueError("[MIG-CACHE-TRANSFORM-001D] axe O->L dégénéré")
        pivot = self._get_core_group_world_centroid(world, str(core_group_id))

        world.flip_group(str(core_group_id), pivot, axis)
        final_core_group_id = world.get_group_of_element(element_id)
        if final_core_group_id is None:
            raise ValueError(
                f"[MIG-CACHE-TRANSFORM-001D] groupe Core final introuvable element={element_id!r}"
            )
        self._project_core_group_to_last_drawn(world, str(final_core_group_id))
        self._invalidate_pick_cache()

        self._redraw_from(self._last_drawn)
        self.status.config(text=f"Inversion appliquée au groupe #{final_core_group_id}.")

    def _commit_move_group_to_core(self, core_group_id: str, dx_w, dy_w) -> None:
        """Valide une translation manuelle unique dans le Core puis la projette."""
        if not core_group_id:
            return
        scen = self._get_active_scenario()
        world = scen.topoWorld

        world.move_group(core_group_id, float(dx_w), float(dy_w))
        self._project_core_group_to_last_drawn(world, core_group_id)

    def _move_group_world(self, core_group_id: str, dx_w, dy_w, move_member_entries=None):
        """Compatibilité : commit d'une translation, jamais un aperçu manuel."""
        if self._is_active_auto_scenario():
            raise RuntimeError("Simulation AUTO: la translation est interdite pour un groupe ancré")

        self._commit_move_group_to_core(core_group_id, dx_w, dy_w)

    def _prepare_core_group_operation_members(
        self,
        operation: str,
        tri_index: int,
        world: TopologyWorld | None = None,
    ) -> Dict:
        """Prepare les membres projetes d'une operation depuis le seul Core."""
        operation_name = str(operation or "MIG-GEO").upper()
        result = {"core_group_id": None, "entries": []}
        if not (0 <= int(tri_index) < len(self._last_drawn)):
            MIG_GEO_LOGGER.warning("[%s] index triangle invalide: %s", operation_name, tri_index)
            return result
        triangle = self._last_drawn[int(tri_index)]
        topo_element_id = str(triangle.get("topoElementId", "") or "").strip()
        effective_world = world or self._get_active_scenario().topoWorld
        core_group_id = self._get_core_group_id_for_triangle_index(
            int(tri_index), effective_world
        )
        if not core_group_id:
            MIG_GEO_LOGGER.warning("[%s] groupe Core introuvable topoElementId=%s", operation_name, topo_element_id or "(absent)")
            return result
        entries = list(
            self._get_projected_elements_for_core_group(core_group_id, effective_world)
        )

        if not entries:
            MIG_GEO_LOGGER.warning("[%s] aucun membre projete CoreGroupId=%s", operation_name, core_group_id)
        return {"core_group_id": core_group_id, "entries": entries}

    def _resolve_core_vertex_move_members(
        self,
        tri_index: int,
        vkey: str,
        world: TopologyWorld | None = None,
    ) -> Dict:
        """Résout le groupe Core du sommet sélectionné, sans lire les groupes UI."""
        result = {
            "core_group_id": None,
            "entries": [],
            "element_id": None,
            "node_id": None,
            "node_canon": None,
        }
        if not (0 <= int(tri_index) < len(self._last_drawn)):
            return result

        vertex_type = str(vkey)
        if vertex_type not in ("O", "B", "L"):
            return result

        world = world or self._get_active_scenario().topoWorld
        element_id = str(self._last_drawn[int(tri_index)].get("topoElementId", "") or "").strip()
        if not element_id:
            return result

        node_id = world.get_element_vertex_node_id_by_type(element_id, vertex_type)
        node_canon = world.find_node(node_id)
        core_group_id = str(world.get_group_of_element(element_id))
        entries = self.get_last_drawn_entries_for_core_group(core_group_id, world)

        result.update({
            "core_group_id": core_group_id,
            "entries": entries,
            "element_id": element_id,
            "node_id": str(node_id),
            "node_canon": str(node_canon),
        })

        return result

    def _snapshot_mig_geo_entries(self, entries: List[Dict]) -> Dict[int, Dict]:
        """Capture les poses des entrées effectivement concernées par l'opération."""
        snapshots: Dict[int, Dict] = {}
        for entry in entries:
            topo_element_id = str(entry.get("topoElementId", "") or "").strip()
            tid = self.canvas_objects.get_index_by_topology_id(topo_element_id)
            if tid is None:
                continue
            points = entry.get("pts")
            if points is None:
                continue
            snapshots[tid] = {
                key: np.array(points[key], dtype=float, copy=True)
                for key in ("O", "B", "L")
            }
        return snapshots

    def _capture_move_preview_initial_pts(
        self,
        world: TopologyWorld,
        core_group_id: str,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Capture la projection initiale des seuls membres du groupe Core."""
        if world is None:
            raise RuntimeError("[MIG-CACHE-TRANSFORM-001B] TopologyWorld absent")
        snapshots: Dict[str, Dict[str, np.ndarray]] = {}
        for element_id in world.getGroupElementIds(core_group_id):
            key = element_id
            entry = self.canvas_objects.get_by_topology_id(key)
            if entry is None:
                raise KeyError(
                    "[MIG-CACHE-TRANSFORM-001B] projection absente "
                    f"pour topoElementId={key!r}"
                )
            points = entry.get("pts")
            if not isinstance(points, dict):
                raise ValueError(
                    "[MIG-CACHE-TRANSFORM-001B] points absents "
                    f"pour topoElementId={key!r}"
                )
            snapshots[key] = {
                vertex: np.array(points[vertex], dtype=float, copy=True)
                for vertex in ("O", "B", "L")
            }
        return snapshots

    def _capture_active_auto_preview_pts(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Capture le cache actif pour un aperçu AUTO, sans toucher au Core."""
        snapshots: Dict[str, Dict[str, np.ndarray]] = {}
        for entry in self.canvas_objects:
            element_id = str(entry.get("topoElementId", "") or "").strip()
            points = entry.get("pts")
            if not element_id or not isinstance(points, dict):
                continue
            snapshots[element_id] = {
                vertex: np.array(points[vertex], dtype=float, copy=True)
                for vertex in ("O", "B", "L")
            }
        return snapshots

    def _preview_auto_translation_from_snapshot(
        self,
        initial_pts: Dict[str, Dict[str, np.ndarray]],
        delta_world,
    ) -> None:
        """Aperçu MOVE AUTO limité au cache du scénario actif."""
        delta = np.asarray(delta_world, dtype=float)
        if delta.shape != (2,) or not np.all(np.isfinite(delta)):
            raise ValueError("[MIG-CACHE-TRANSFORM-001H] delta preview AUTO invalide")
        for element_id, points0 in initial_pts.items():
            entry = self.canvas_objects.get_by_topology_id(element_id)
            if entry is None:
                raise KeyError(
                    f"[MIG-CACHE-TRANSFORM-001H] projection absente: {element_id!r}"
                )
            entry["pts"] = {
                vertex: np.array(points0[vertex], dtype=float, copy=True) + delta
                for vertex in ("O", "B", "L")
            }
        self._invalidate_pick_cache()

    def _preview_auto_rotation_from_snapshot(
        self,
        initial_pts: Dict[str, Dict[str, np.ndarray]],
        pivot_world,
        angle_rad: float,
    ) -> None:
        """Aperçu ROTATE AUTO limité au cache du scénario actif."""
        pivot = np.asarray(pivot_world, dtype=float)
        angle = float(angle_rad)
        if pivot.shape != (2,) or not np.all(np.isfinite(pivot)) or not np.isfinite(angle):
            raise ValueError("[MIG-CACHE-TRANSFORM-001H] rotation preview AUTO invalide")
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array(((c, -s), (s, c)), dtype=float)
        for element_id, points0 in initial_pts.items():
            entry = self.canvas_objects.get_by_topology_id(element_id)
            if entry is None:
                raise KeyError(
                    f"[MIG-CACHE-TRANSFORM-001H] projection absente: {element_id!r}"
                )
            entry["pts"] = {
                vertex: rotation @ (np.asarray(points0[vertex], dtype=float) - pivot) + pivot
                for vertex in ("O", "B", "L")
            }
        self._invalidate_pick_cache()

    def _discard_auto_transform_preview(self) -> bool:
        """Abandonne un aperçu AUTO en reprojectant le Core actif inchangé."""
        selection = self._sel if isinstance(getattr(self, "_sel", None), dict) else None
        scen = self._get_active_scenario()
        if (
            selection is None
            or not selection.get("auto_geom")
            or scen is None
            or getattr(scen, "source_type", "manual") != "auto"
        ):
            return False
        self._project_auto_scenario_from_core(scen)
        self._sel = None
        return True

    def _preview_move_group_from_snapshot(
        self,
        initial_pts: Dict[str, Dict[str, np.ndarray]],
        dx_total: float,
        dy_total: float,
    ) -> None:
        """Met à jour l'aperçu UI d'un MOVE sans modifier le Core."""
        delta = np.array((float(dx_total), float(dy_total)), dtype=float)
        for element_id, points0 in initial_pts.items():
            entry = self.canvas_objects.get_by_topology_id(str(element_id))
            if entry is None:
                raise KeyError(
                    "[MIG-CACHE-TRANSFORM-001B] projection absente "
                    f"pour topoElementId={element_id!r}"
                )
            entry["pts"] = {
                vertex: np.array(points0[vertex], dtype=float, copy=True) + delta
                for vertex in ("O", "B", "L")
            }
        self._invalidate_pick_cache()

    def _restore_manual_move_group_preview(self) -> None:
        """Restaure la translation libre courante après une preview ATT-003D."""
        if not isinstance(self._sel, dict) or self._sel.get("mode") != "move_group":
            raise RuntimeError("[ATT-003D] sélection MOVE absente à la restauration")
        initial_pts = self._sel.get("move_preview_initial_pts")
        start = self._sel.get("mouse_world_start")
        current = self._sel.get("last_mouse_world", start)
        if not isinstance(initial_pts, dict) or start is None or current is None:
            raise RuntimeError("[ATT-003D] état MOVE absent à la restauration")
        delta = np.asarray(current, dtype=float) - np.asarray(start, dtype=float)
        self._preview_move_group_from_snapshot(
            initial_pts,
            float(delta[0]),
            float(delta[1]),
        )

    @staticmethod
    def _normalize_rotation_angle(angle_rad: float) -> float:
        """Normalise un angle dans l'intervalle [-pi, pi[."""
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    def _preview_rotate_group_from_snapshot(
        self,
        initial_pts: Dict[str, Dict[str, np.ndarray]],
        pivot_world,
        angle_rad: float,
    ) -> None:
        """Met à jour un aperçu de rotation sans modifier le Core."""
        pivot = np.asarray(pivot_world, dtype=float)
        angle = self._normalize_rotation_angle(angle_rad)
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array(((c, -s), (s, c)), dtype=float)
        for element_id, points0 in initial_pts.items():
            entry = self.canvas_objects.get_by_topology_id(str(element_id))
            if entry is None:
                raise KeyError(
                    "[MIG-CACHE-TRANSFORM-001C] projection absente "
                    f"pour topoElementId={element_id!r}"
                )
            entry["pts"] = {
                vertex: rotation @ (np.asarray(points0[vertex], dtype=float) - pivot) + pivot
                for vertex in ("O", "B", "L")
            }
        self._invalidate_pick_cache()

    def _discard_manual_rotate_preview(self) -> bool:
        """Abandonne un aperçu ROTATE manuel et restaure la projection Core."""
        selection = self._sel if isinstance(getattr(self, "_sel", None), dict) else None
        if (
            selection is None
            or selection.get("mode") not in (
                "rotate_group",
                "rotate_group_anchor_drag",
            )
            or selection.get("auto_geom")
        ):
            return False
        core_group_id = selection.get("core_group_id")
        scen = self._get_active_scenario()
        world = scen.topoWorld
        if core_group_id:
            self._project_core_group_to_last_drawn(world, str(core_group_id))
        self._sel = None

        return True

    def _discard_manual_move_preview(self) -> bool:
        """Abandonne un aperçu MOVE manuel et restaure la projection Core."""
        selection = self._sel if isinstance(getattr(self, "_sel", None), dict) else None
        if (
            selection is None
            or selection.get("mode") != "move_group"
            or self._is_active_auto_scenario()
        ):
            return False
        core_group_id = selection.get("core_group_id")
        scen = self._get_active_scenario()
        world = scen.topoWorld
        if core_group_id:
            self._project_core_group_to_last_drawn(world, str(core_group_id))
        self._sel = None
        return True

    def _get_move_drag_delta_world(self, event) -> np.ndarray:
        """Retourne le delta monde total du MOVE transactionnel courant."""
        if not isinstance(self._sel, dict):
            raise RuntimeError("[MIG-CACHE-TRANSFORM-001E1-FIX] sélection MOVE absente")
        start = self._sel.get("mouse_world_start")
        if start is None:
            raise RuntimeError(
                "[MIG-CACHE-TRANSFORM-001E1-FIX] point de départ du MOVE absent"
            )
        end = np.asarray(self._screen_to_world(event.x, event.y), dtype=float)
        start_world = np.asarray(start, dtype=float)
        delta = end - start_world
        if delta.shape != (2,) or not np.all(np.isfinite(delta)):
            raise ValueError(
                "[MIG-CACHE-TRANSFORM-001E1-FIX] delta monde du MOVE invalide"
            )
        return np.array(delta, dtype=float, copy=True)

    @staticmethod
    def _rotation_angle_from_mouse_world(mouse_world, pivot_world) -> float:
        """Angle souris/pivot, avec direction déterministe au point du pivot."""
        delta = np.asarray(mouse_world, dtype=float) - np.asarray(pivot_world, dtype=float)
        if float(np.linalg.norm(delta)) <= 1e-9:
            return 0.0
        return math.atan2(float(delta[1]), float(delta[0]))

    def _begin_anchored_group_rotation_drag(self, world, core_group_id, anchor, event):
        """Démarre la rotation transactionnelle d'un groupe autour de sa balise."""
        pivot_world = np.asarray(world.getBeaconWorldXY(anchor.beacon_id), dtype=float)
        mouse_world = self._screen_to_world(event.x, event.y)
        self._sel = {
            "mode": "rotate_group_anchor_drag",
            "core_group_id": core_group_id,
            "anchor_id": anchor.anchor_id,
            "beacon_id": anchor.beacon_id,
            "pivot_world": pivot_world,
            "mouse_angle_start": self._rotation_angle_from_mouse_world(
                mouse_world, pivot_world
            ),
            "rotate_preview_initial_pts": self._capture_move_preview_initial_pts(
                world, core_group_id
            ),
            "auto_collective": self._is_active_auto_scenario(),
        }
        if self._deformation_state.active:
            self._sel["deformation_base_world"] = world.clonePhysicalState()
        self._draw_anchor_rotation_pivot_highlight(pivot_world)
        self.status.config(
            text=f"Rotation du groupe Core {core_group_id} autour de la balise {anchor.beacon_id}."
        )
        return "break"

    def _on_canvas_left_down(self, event):
        # Mode compas : arc d'angle (clic pour P1/P2)
        if self.compass_state.arc.active:
            self._clock_arc_handle_click(int(event.x), int(event.y))
            return "break"

        if self.compass_state.trace.active:
            self._clock_trace_confirm()
            return "break"

        # Mode compas : clic pour valider une mesure d'azimut
        if self.compass_state.measure.active:
            self._clock_measure_confirm()
            return "break"

        # Mode compas : clic pour valider l'azimut de référence
        if self.compass_state.set_ref.active:
            self._clock_setref_confirm(int(event.x), int(event.y))
            return "break"

        # Catalogue : le clic Canvas valide l'objet qui suit déjà le curseur.
        if self._drag and self._drag.get("from") == "list":
            self._commit_list_placement_at_canvas_event(event)
            return "break"

        # garantir un cache pick à jour avant tout hit-test
        self._ensure_pick_cache()
        # mémoriser l'ancre monde de la souris pour des déplacements "delta"
        self._mouse_world_prev = self._screen_to_world(event.x, event.y)

        # Horloge : démarrer drag si clic dans le disque (marge 10px)
        if self._is_in_clock(event.x, event.y):
            # ne pas intercepter si un drag de triangle est en cours
            if not self._drag:
                self.compass_state.dragging = True
                self.compass_state.drag_dx = event.x - (self.compass_state.cx or event.x)
                self.compass_state.drag_dy = event.y - (self.compass_state.cy or event.y)
                # Mode "snap compas" : dès le mouse-down, viser le sommet le plus proche
                self._clock_update_snap_target(event.x, event.y)
                self.canvas.configure(cursor="fleur")
                return "break"  # on court-circuite la logique des triangles

        # Fond SVG : si mode resize et clic sur poignée -> on capture et on court-circuite le reste
        if self.bg_resize_mode.get() and self.background_map_layer.has_map:
            h = self.background_map_layer.hit_test_handle(event.x, event.y)
            if h:
                self.background_map_layer.start_resize(h, event.x, event.y)
                self.canvas.configure(cursor="sizing")
                return "break"
            # sinon, en mode redimensionnement : clic maintenu = déplacement du fond
            self.background_map_layer.start_move(event.x, event.y)
            self.canvas.configure(cursor="fleur")

            return "break"

        # Validation d'une rotation en cours : le clic gauche sert à COMMIT, pas à re-sélectionner.
        if (
            self._deformation_state.active
            and self._deformation_canvas_mode == "select"
        ):
            return self._handle_deformation_left_down(event)

        if isinstance(self._sel, dict) and self._sel.get("mode") == "rotate_group":
            if self._sel.get("auto_geom"):
                pivot = np.asarray(self._sel["pivot"], dtype=float)
                wx, wy = self._screen_to_world(event.x, event.y)
                current_angle = math.atan2(float(wy - pivot[1]), float(wx - pivot[0]))
                angle_delta = current_angle - float(self._sel["start_angle"])
                self._rotate_all_auto_scenarios_around_anchors(angle_delta)
            else:
                # MIG-CACHE-TRANSFORM-001C : commit Core unique depuis le
                # clic de validation, jamais depuis le preview UI.
                core_group_id = self._sel.get("core_group_id")
                pivot = np.asarray(self._sel["pivot_world"], dtype=float)
                start_angle = float(self._sel["mouse_angle_start"])
                wx, wy = self._screen_to_world(event.x, event.y)
                final_angle = math.atan2(float(wy - pivot[1]), float(wx - pivot[0]))
                angle_total = self._normalize_rotation_angle(final_angle - start_angle)
                scen = self._get_active_scenario()
                world = scen.topoWorld
                if core_group_id is None:
                    raise RuntimeError(
                        "[MIG-CACHE-TRANSFORM-001C] groupe Core ou monde absent au commit"
                    )
                world.rotate_group(str(core_group_id), pivot, angle_total)
                element_ids = tuple(self._sel["rotate_preview_initial_pts"])
                if not element_ids:
                    raise RuntimeError(
                        "[MIG-CACHE-TRANSFORM-001C] groupe vide au commit"
                    )
                final_core_group_id = world.get_group_of_element(str(element_ids[0]))
                self._project_core_group_to_last_drawn(world, str(final_core_group_id))

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
        deformation_world = (
            self._deformation_effective_world()
            if self._deformation_state.active
            else None
        )
        if mode == "center":
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            # Groupe obligatoire : on démarre toujours un move_group
            move_members = self._prepare_core_group_operation_members(
                "MOVE", idx, deformation_world
            )
            core_group_id = move_members["core_group_id"]
            move_member_entries = move_members["entries"]
            if not core_group_id or not move_member_entries:
                return
            group_centroid = self._group_centroid(core_group_id, deformation_world)
            if group_centroid is None:
                return
            is_auto_move = self._is_active_auto_scenario()
            world = deformation_world or self._get_active_scenario().topoWorld
            anchor = world.getAnchorForGroup(core_group_id)
            if anchor is not None:
                return self._begin_anchored_group_rotation_drag(
                    world, core_group_id, anchor, event
                )
            preview_initial_pts = (
                self._capture_move_preview_initial_pts(world, str(core_group_id))
                if not is_auto_move else {}
            )
            self._sel = {
                "mode": "move_group",
                "core_group_id": core_group_id,
                "mouse_world_start": np.array([wx, wy], dtype=float),
                "move_preview_initial_pts": preview_initial_pts,
            }
            if self._deformation_state.active:
                self._sel["deformation_base_world"] = world.clonePhysicalState()
            if is_auto_move:
                self._sel.update({
                    "auto_geom": True,
                    "auto_state0": dict(self.auto_rotation_state or {"thetaDeg": 0.0}),
                    "auto_move_preview_pts0": self._capture_active_auto_preview_pts(),
                })
            self.status.config(text=f"Déplacement du groupe Core {core_group_id}.")
        elif mode == "vertex":
            # déplacement par sommet (translation comme 'center', mais calée sur le sommet choisi)
            vkey = extra or "O"
            P = self._last_drawn[idx]["pts"]
            wx = (event.x - self.offset[0]) / self.zoom
            wy = (self.offset[1] - event.y) / self.zoom
            # NOTE: si on est en déconnexion, on désactivera toute aide au collage

            vertex_move_members = self._resolve_core_vertex_move_members(
                idx, vkey, deformation_world
            )
            if not vertex_move_members["entries"]:
                return
            world = deformation_world or self._get_active_scenario().topoWorld
            core_group_id = vertex_move_members["core_group_id"]
            anchor = world.getAnchorForGroup(core_group_id)
            if anchor is not None:
                return self._begin_anchored_group_rotation_drag(
                    world, core_group_id, anchor, event
                )

            # CTRL active la preview géométrique V2 pendant le MOVE du groupe.
            if self._ctrl_down:
                # Les membres proviennent du groupe résolu par le sommet Core.
                move_members = vertex_move_members
                is_auto_move = self._is_active_auto_scenario()
                world = deformation_world or self._get_active_scenario().topoWorld
                preview_initial_pts = (
                    self._capture_move_preview_initial_pts(world, str(move_members["core_group_id"]))
                    if not is_auto_move else {}
                )

                self._sel = {
                    "mode": "move_group",
                    "core_group_id": move_members["core_group_id"],
                    "mouse_world_start": np.array([wx, wy], dtype=float),
                    "move_preview_initial_pts": preview_initial_pts,
                    "anchor": {"type": "vertex", "tid": idx, "vkey": vkey},
                    "suppress_assist": False,
                }
                if self._deformation_state.active:
                    self._sel["deformation_base_world"] = world.clonePhysicalState()
                if is_auto_move:
                    self._sel.update({
                        "auto_geom": True,
                        "auto_state0": dict(self.auto_rotation_state or {"thetaDeg": 0.0}),
                        "auto_move_preview_pts0": self._capture_active_auto_preview_pts(),
                    })
                # nettoyer toute aide existante avant de calculer le candidat.
                self._reset_assist()
                self.status.config(text=f"Déplacement du groupe Core {move_members['core_group_id']} par sommet {vkey}.")

                self._redraw_from(self._last_drawn)
                return

            # Si CTRL n'est pas tenu, le MOVE reste une translation libre.
            if not self._ctrl_down:
                # Les membres proviennent du groupe résolu par le sommet Core.
                move_members = vertex_move_members
                is_auto_move = self._is_active_auto_scenario()
                world = deformation_world or self._get_active_scenario().topoWorld
                preview_initial_pts = (
                    self._capture_move_preview_initial_pts(world, str(move_members["core_group_id"]))
                    if not is_auto_move else {}
                )

                self._sel = {
                    "mode": "move_group",
                    "core_group_id": move_members["core_group_id"],
                    "mouse_world_start": np.array([wx, wy], dtype=float),
                    "move_preview_initial_pts": preview_initial_pts,
                    "anchor": {"type": "vertex", "tid": idx, "vkey": vkey},
                    # on veut l'aide de collage active
                    "suppress_assist": False,
                }
                if self._deformation_state.active:
                    self._sel["deformation_base_world"] = world.clonePhysicalState()
                if is_auto_move:
                    self._sel.update({
                        "auto_geom": True,
                        "auto_state0": dict(self.auto_rotation_state or {"thetaDeg": 0.0}),
                        "auto_move_preview_pts0": self._capture_active_auto_preview_pts(),
                    })
                self.status.config(text=f"Déplacement du groupe Core {move_members['core_group_id']} par sommet {vkey}.")
                # Aide immédiate : viser un sommet d'un AUTRE triangle (exclure le groupe lui-même)
                v_world = np.array(P[vkey], dtype=float)
                self._update_group_drag_snap_assist(
                    v_world, idx, vkey, move_members["core_group_id"]
                )
                return

            orig_pts = {k: np.array(P[k].copy()) for k in ("O", "B", "L")}
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
        if self.compass_state.dragging:
            self.compass_state.cx = event.x - self.compass_state.drag_dx
            self.compass_state.cy = event.y - self.compass_state.drag_dy
            # Cible snap (sommet le plus proche du CENTRE du compas)
            self._clock_update_snap_target(self.compass_state.cx, self.compass_state.cy)
            self._redraw_overlay_only()
            return "break"

        # Mode déplacement fond d'écran (mode resize actif, clic maintenu hors poignée)
        if self.background_map_layer.is_moving:
            self.background_map_layer.update_move(event.x, event.y)
            self._redraw_from(self._last_drawn)
            return "break"

        # Mode resize fond d'écran
        if self.background_map_layer.is_resizing:
            self.background_map_layer.update_resize(event.x, event.y)
            self._update_background_map_scale_status()
            self._redraw_from(self._last_drawn)
            return "break"

        if self._drag:
            return  # le drag liste gère déjà le mouvement
        if not self._sel:
            self._on_pan_move(event)
            return

        # --- Rotation ancrée : aperçu transactionnel autour de la balise ---
        if (
            self._deformation_state.active
            and self._sel["mode"] == "rotate_group_anchor_drag"
        ):
            try:
                self._preview_deformation_rotation(event)
            except (ValueError, RuntimeError) as exc:
                self._exit_deformation_mode()
                messagebox.showerror("Deformation", str(exc), parent=self)
            return "break"

        if self._sel["mode"] == "rotate_group_anchor_drag":
            pivot_world = np.asarray(self._sel["pivot_world"], dtype=float)
            mouse_world = self._screen_to_world(event.x, event.y)
            current_angle = self._rotation_angle_from_mouse_world(
                mouse_world, pivot_world
            )
            angle_delta = self._normalize_rotation_angle(
                current_angle - float(self._sel["mouse_angle_start"])
            )
            self._preview_rotate_group_from_snapshot(
                self._sel["rotate_preview_initial_pts"],
                pivot_world,
                angle_delta,
            )
            self._redraw_from(self._last_drawn)
            self._draw_anchor_rotation_pivot_highlight(pivot_world)
            return

        # --- Déplacement de GROUPE ---
        if self._sel["mode"] == "move_group":
            if self._deformation_state.active:
                try:
                    self._preview_deformation_group_translation(event)
                except (ValueError, RuntimeError) as exc:
                    self._exit_deformation_mode()
                    messagebox.showerror("Deformation", str(exc), parent=self)
                return "break"
            # Pendant une déconnexion, ne jamais montrer l'aide de collage
            if self._sel.get("suppress_assist"):
                self._reset_assist()
            core_group_id = self._sel.get("core_group_id")
            if not core_group_id:
                return
            # MIG-CACHE-TRANSFORM-001B : en manuel, l'aperçu est calculé
            # depuis le snapshot du mouse-down ; le Core reste inchangé.
            wx, wy = self._screen_to_world(event.x, event.y)
            if self._is_active_auto_scenario():
                start = self._sel.get("mouse_world_start")
                initial_pts = self._sel.get("auto_move_preview_pts0")
                if start is None or not isinstance(initial_pts, dict):
                    raise RuntimeError(
                        "[MIG-CACHE-TRANSFORM-001H] état initial du MOVE AUTO absent"
                    )
                self._preview_auto_translation_from_snapshot(
                    initial_pts,
                    np.array([wx, wy], dtype=float) - np.asarray(start, dtype=float),
                )
            else:
                start = self._sel.get("mouse_world_start")
                initial_pts = self._sel.get("move_preview_initial_pts")
                if start is None or not isinstance(initial_pts, dict):
                    raise RuntimeError(
                        "[MIG-CACHE-TRANSFORM-001B] état initial du MOVE absent"
                    )
                dx, dy = float(wx - start[0]), float(wy - start[1])
                self._preview_move_group_from_snapshot(initial_pts, dx, dy)
                self._sel["last_mouse_world"] = np.array([wx, wy], dtype=float)
            self._redraw_from(self._last_drawn)
            # L'assistance Attachment est permanente pendant un MOVE manuel.
            # CTRL ne contrôle que la projection temporaire du preview V2.
            if not self._sel.get("suppress_assist"):
                anchor = self._sel.get("anchor")
                if anchor and anchor.get("type") == "vertex":
                    anchor_tid = anchor.get("tid")
                    anchor_vkey = anchor.get("vkey")
                    if 0 <= anchor_tid < len(self._last_drawn):
                        Panchor = self._last_drawn[anchor_tid]["pts"]
                        v_world = np.array(Panchor[anchor_vkey], dtype=float)
                        self._update_group_drag_snap_assist(
                            v_world,
                            anchor_tid,
                            anchor_vkey,
                            self._sel.get("core_group_id"),
                        )
                        if (
                            self._ctrl_down
                            and self._attachment_preview is not None
                            and self._attachment_preview.accepted
                        ):
                            self._preview_attachment_rotation_to_last_drawn(
                                self._attachment_preview
                            )
                            self._redraw_from(self._last_drawn)
                    else:
                        self._reset_assist()
            self._clock_apply_auto_ref_sync()

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
            self._clock_apply_auto_ref_sync()

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
            self._clock_apply_auto_ref_sync()

    def _update_list_drag_preview_at_canvas_xy(self, canvas_x: float, canvas_y: float) -> None:
        """Met à jour le fantôme actif depuis des coordonnées Canvas."""
        wx = (canvas_x - self.offset[0]) / self.zoom
        wy = (self.offset[1] - canvas_y) / self.zoom
        if self._drag.get("kind") == "quadrilateral":
            mouse_world = np.array([wx, wy], dtype=float)
            reference_id = self._drag["reference_triangle_id"]
            relative = self._drag["relative_world_pts"]
            reference_point = np.asarray(relative[reference_id]["O"], dtype=float)
            delta = mouse_world - reference_point
            self._drag["world_pts_by_triangle"] = {
                triangle_id: {key: np.asarray(point, dtype=float) + delta for key, point in points.items()}
                for triangle_id, points in relative.items()
            }
            while len(self._drag_preview_ids) < 2:
                item_id = self.canvas.create_polygon(
                    0, 0, 0, 0, 0, 0, outline="gray50", dash=(4, 2), fill="", width=2,
                )
                self._drag_preview_ids.append(item_id)
            for item_id, triangle_id in zip(self._drag_preview_ids, self._drag["triangle_ids"]):
                coords = []
                for key in ("O", "B", "L"):
                    sx, sy = self._world_to_screen(self._drag["world_pts_by_triangle"][triangle_id][key])
                    coords += [sx, sy]
                self.canvas.coords(item_id, *coords)
            return
        triangle_id = self._drag["triangle_id"]
        self._drag["world_pts"] = self._build_drag_world_points(triangle_id, (wx, wy))
        coords = []
        for key in ("O", "B", "L"):
            sx, sy = self._world_to_screen(self._drag["world_pts"][key])
            coords += [sx, sy]
        if self._drag_preview_id is None:
            self._drag_preview_id = self.canvas.create_polygon(*coords, outline="gray50", dash=(4, 2), fill="", width=2)
            self._drag_preview_ids = [self._drag_preview_id]
        else:
            self.canvas.coords(self._drag_preview_id, *coords)

    def _on_canvas_left_up(self, event):
        # Horloge : fin de drag
        if self.compass_state.dragging:
            # On capture la cible de snap *avant* de la nettoyer pour pouvoir déclencher
            # une éventuelle mesure d'arc automatique.
            snap_tgt = self.compass_state.snap_target

            # Si CTRL au relâché : on sort du mode sans "snap" (le compas reste où il est)
            if self._ctrl_down:
                wx, wy = self._screen_to_world(self.compass_state.cx, self.compass_state.cy)
                self._clock_clear_anchor_binding()
                self.compass_state.anchor_world = np.array([wx, wy], dtype=float)

            else:
                # cible snap calculée pendant le drag (capturée au début via snap_tgt)
                tgt = snap_tgt
                if isinstance(tgt, dict) and tgt.get("world") is not None:
                    self._clock_bind_anchor_from_snap_target(tgt)
                    self.compass_state.anchor_world = np.array(tgt["world"], dtype=float)
                    sx, sy = self._world_to_screen(tgt["world"])
                    self.compass_state.cx, self.compass_state.cy = float(sx), float(sy)
                else:
                    # pas de target : ancrer la position courante en monde
                    wx, wy = self._screen_to_world(self.compass_state.cx, self.compass_state.cy)
                    self._clock_clear_anchor_binding()
                    self.compass_state.anchor_world = np.array([wx, wy], dtype=float)

            self.compass_state.dragging = False
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
            self._update_compass_ctx_menu_and_dico_state()
            return "break"

        if self.background_map_layer.is_resizing:
            self.background_map_layer.finish_resize()
            self.canvas.configure(cursor="")
            self._update_background_map_scale_status()
            self._redraw_from(self._last_drawn)
            return "break"

        if self.background_map_layer.is_moving:
            self.background_map_layer.finish_move()
            self.canvas.configure(cursor="")
            self._redraw_from(self._last_drawn)
            return "break"

        # Pan au clic gauche : fin du pan même si aucun triangle n'est sélectionné
        if self._pan_anchor is not None and not self._sel and not self._drag:
            self._on_pan_end(event)
            return

        # Le placement Catalogue est committé au ButtonPress Canvas. Un éventuel
        # ButtonRelease ultérieur ne doit jamais déclencher un second commit.
        if self._drag and self._drag.get("from") == "list":
            return "break"

        # 1) Le reste est ton comportement existant (fin de drag/rotation/snap)
        if not self._sel:
            return

        mode = self._sel.get("mode")
        if mode == "rotate_group_anchor_drag":
            if self._deformation_state.active:
                try:
                    self._preview_deformation_rotation(event)
                except (ValueError, RuntimeError) as exc:
                    self._exit_deformation_mode()
                    messagebox.showerror("Deformation", str(exc), parent=self)
                    return "break"
                self._sel = None
                self._reset_assist()
                return "break"
            core_group_id = self._sel.get("core_group_id")
            pivot_world = np.asarray(self._sel["pivot_world"], dtype=float)
            scen = self._get_active_scenario()
            world = scen.topoWorld
            if core_group_id is None:
                raise RuntimeError(
                    "[MIG-ANCHOR-007] groupe Core absent au commit"
                )
            mouse_world = self._screen_to_world(event.x, event.y)
            final_angle_delta = self._normalize_rotation_angle(
                self._rotation_angle_from_mouse_world(mouse_world, pivot_world)
                - float(self._sel["mouse_angle_start"])
            )
            if self._sel.get("auto_collective"):
                self._rotate_all_auto_scenarios_around_anchors(final_angle_delta)
            else:
                world.rotate_group(core_group_id, pivot_world, final_angle_delta)
                self._project_core_group_to_last_drawn(world, core_group_id)
            self._sel = None
            self._reset_assist()
            if self._deformation_state.active:
                try:
                    self._deformation_refresh_preview_after_rotation()
                except (ValueError, RuntimeError) as exc:
                    self._exit_deformation_mode()
                    messagebox.showerror("Deformation", str(exc), parent=self)
            else:
                self._redraw_from(self._last_drawn)
            return

        if mode == "move_group":
            if self._deformation_state.active:
                try:
                    self._preview_deformation_group_translation(event)
                except (ValueError, RuntimeError) as exc:
                    self._exit_deformation_mode()
                    messagebox.showerror("Deformation", str(exc), parent=self)
                    return "break"
                self._sel = None
                self._reset_assist()
                return "break"
            # Collage du GROUPE quand on l'a déplacé PAR SOMMET (ancre=vertex)
            # et que l'aide de collage était active (pas en déconnexion).
            anchor = self._sel.get("anchor")
            suppress = self._sel.get("suppress_assist")
            # MIG-GROUP-017 : les données métier du geste sont Core. Les IDs
            # UI ne restent nécessaires que pour projeter la fusion legacy.
            mobile_core_group_id = self._sel.get("core_group_id")
            core_snap_transform_applied = False
            beacon_anchor_applied = False
            snap_world = None
            mobile_element_id = None
            beacon_candidate = self._group_drag_snap_candidate
            attachment_intent = self._attachment_intent
            attachment_preview = self._attachment_preview
            manual_attachment_attempted = False
            manual_attachment_intent_pending = False

            # On nettoie l'aide visuelle dans tous les cas
            self._clear_edge_highlights()

            # ATT-003D : seul l'intent V2 validé par son preview peut produire
            # un raccord au release ; aucune donnée EdgeChoice n'est consultée.
            manual_attachment_attempted = bool(
                not suppress
                and not beacon_anchor_applied
                and attachment_intent is not None
            )
            if (
                manual_attachment_attempted
                and attachment_preview is not None
                and attachment_preview.accepted
            ):
                manual_attachment_intent_pending = True

            # MIG-ANCHOR-003 : sommet raccordable et balise sont recherchés
            # indépendamment ; le candidat admissible le plus proche gagne.
            # L'ancre traduit le groupe depuis le Core, jamais depuis le preview.
            if (
                not suppress
                and not self._ctrl_down
                and not self._is_active_auto_scenario()
                and anchor
                and anchor.get("type") == "vertex"
                and isinstance(beacon_candidate, dict)
                and beacon_candidate.get("type") == "beacon"
            ):
                scen = self._get_active_scenario()
                world = scen.topoWorld
                anchor_tid = int(anchor["tid"])
                if mobile_core_group_id is None:
                    raise RuntimeError(
                        "[MIG-ANCHOR-003] groupe Core mobile absent"
                    )
                mobile_entry = self._last_drawn[anchor_tid]
                mobile_element_id = str(
                    mobile_entry.get("topoElementId", "") or ""
                ).strip()
                if not mobile_element_id:
                    raise RuntimeError(
                        "[MIG-ANCHOR-003] topoElementId mobile absent"
                    )
                node_id = world.get_element_vertex_node_id_by_type(
                    mobile_element_id, anchor["vkey"]
                )
                group_anchor = world.createGroupAnchor(
                    group_id=mobile_core_group_id,
                    node_id=node_id,
                    beacon_id=beacon_candidate["beacon_id"],
                )
                world.applyGroupAnchor(group_anchor.anchor_id)
                final_core_group_id = world.get_group_of_element(mobile_element_id)
                if final_core_group_id is None:
                    raise RuntimeError(
                        "[MIG-ANCHOR-003] groupe Core final introuvable après ancrage"
                    )
                self._project_core_group_to_last_drawn(world, str(final_core_group_id))
                beacon_anchor_applied = True
                self.status.config(
                    text=f"Groupe ancré sur la balise {beacon_candidate['beacon_id']}."
                )

            # Commit exclusif : soit snap Core-first, soit translation
            # libre Core-first (un move_group), jamais les deux.
            if beacon_anchor_applied:
                pass
            elif manual_attachment_intent_pending:
                scen = self._get_active_scenario()
                world = scen.topoWorld
                commitManualAttachment(world, attachment_intent)
                self._rebuild_active_projection_from_core()
            elif manual_attachment_attempted:
                self._rebuild_active_projection_from_core()
            elif (
                self._is_active_auto_scenario()
                and not core_snap_transform_applied
                and mobile_core_group_id is not None
            ):
                raise RuntimeError(
                    "Simulation AUTO: translation impossible sans ancre de rotation"
                )
            elif core_snap_transform_applied and mobile_core_group_id is not None:
                if not mobile_element_id:
                    raise RuntimeError(
                        "[MIG-CACHE-TRANSFORM-001B] topoElementId mobile absent après snap"
                    )
                final_core_group_id = snap_world.get_group_of_element(
                    str(mobile_element_id)
                )
                self._project_core_group_to_last_drawn(
                    snap_world, str(final_core_group_id)
                )
            elif not self._is_active_auto_scenario() and mobile_core_group_id is not None:
                drag_delta = self._get_move_drag_delta_world(event)
                self._commit_move_group_to_core(
                    str(mobile_core_group_id),
                    float(drag_delta[0]),
                    float(drag_delta[1]),
                )

            if manual_attachment_intent_pending:
                self.status.config(text="Raccord manuel V2 appliqué.")
            elif manual_attachment_attempted:
                self.status.config(text="Raccord manuel refusé.")
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

    # ---------- Impression / export PDF indépendant ----------
    def _build_print_snapshot(self) -> AssembleurPrintSnapshot:
        """Capture le contenu métier imprimable sans dépendre du canvas Tk."""
        scen = self._get_active_scenario()
        if scen is None:
            raise ValueError("Aucun scénario actif à imprimer.")
        world = scen.topoWorld
        triangles = []
        element_ids = (
            tuple(scen.orderedElementIds)
            if scen.source_type == "auto"
            else tuple(getManualProjectionElementIds(world))
        )
        for element_id in element_ids:
            element = world.elements[element_id]
            points = getCoreTriangleWorldPoints(world, element_id)
            labels = tuple(str(value) for value in element.vertex_labels)
            if len(labels) != 3:
                raise ValueError(f"Labels de sommet invalides pour {element_id!r}.")
            if scen.hypothesis is None:
                raise ValueError("ScenarioHypothesis absente pour l'impression.")
            rank = scen.hypothesis.get_rank_for_triangle_ref(element.source_triangle_id)
            _rotation, _translation, mirrored = world.getElementPose(element_id)
            triangles.append(
                AssembleurPrintTriangle(
                    element_id=str(element_id),
                    o=tuple(float(value) for value in points["O"]),
                    b=tuple(float(value) for value in points["B"]),
                    l=tuple(float(value) for value in points["L"]),
                    labels=labels,
                    display_label="T" + str(rank) + ("S" if mirrored else ""),
                )
            )
        beacons = []
        for beacon in self.catalogue.iter_beacons():
            if beacon.archived:
                continue
            city = self.catalogue.get_city(beacon.city_id)
            beacons.append(
                AssembleurPrintBeacon(
                    beacon_id=beacon.beacon_id,
                    position=self._beacon_world_resolver.get_world(beacon.beacon_id),
                    name=city.name,
                )
            )
        map_snapshot = None
        map_rect = self.background_map_layer.world_rect
        if map_rect is not None and self.background_map_layer.base_image is not None:
            map_snapshot = AssembleurPrintMap(
                image=self.background_map_layer.base_image,
                x0=map_rect.x0, y0=map_rect.y0,
                width=map_rect.w, height=map_rect.h,
            )
        contour_only = bool(self.show_only_group_contours.get())
        boundaries = []
        if contour_only:
            for group_id in world.getLiveGroupIds():
                for segment in world.getBoundarySegments(group_id):
                    points = getCoreTriangleWorldPoints(world, segment.elementId)
                    keys = {"N0": "O", "N1": "B", "N2": "L"}
                    from_key = keys[segment.fromNodeId.rsplit(":", 1)[-1]]
                    to_key = keys[segment.toNodeId.rsplit(":", 1)[-1]]
                    p0, p1 = points[from_key], points[to_key]
                    q0 = p0 + (p1 - p0) * float(segment.t0)
                    q1 = p0 + (p1 - p0) * float(segment.t1)
                    boundaries.append((tuple(float(value) for value in q0), tuple(float(value) for value in q1)))
        return AssembleurPrintSnapshot(
            scenario_name=str(scen.name or "Assemblage"),
            map_snapshot=map_snapshot,
            triangles=tuple(triangles), beacons=tuple(beacons),
            contour_only=contour_only, boundary_segments=tuple(boundaries),
        )

    def _print_fallback_viewport(self) -> AssembleurPrintViewport:
        canvas_width = max(1, int(self.canvas.winfo_width() or 1))
        canvas_height = max(1, int(self.canvas.winfo_height() or 1))
        x0, y1 = self._screen_to_world(0, 0)
        x1, y0 = self._screen_to_world(canvas_width, canvas_height)
        return AssembleurPrintViewport(
            min(float(x0), float(x1)), min(float(y0), float(y1)),
            max(1e-6, abs(float(x1) - float(x0))),
            max(1e-6, abs(float(y1) - float(y0))),
        )

    def _export_view_pdf_dialog(self):
        """Ouvre le workflow d'impression, sans modifier la vue principale."""
        snapshot = self._build_print_snapshot()
        title = snapshot.scenario_name.strip() or "Assemblage"
        settings = AssembleurPrintSettings(
            title=title,
            selected_layers=tuple(
                layer for layer, visible in (
                    ("map", bool(self.show_map_layer.get())),
                    ("assembly", bool(self.show_triangles_layer.get())),
                    ("beacons", bool(self.show_balises_layer.get())),
                ) if visible
            ),
            map_opacity=int(self.map_opacity.get()),
        )
        viewport = fit_initial_viewport(snapshot, settings, self._print_fallback_viewport())
        AssembleurMapPrintDialog(self, snapshot, viewport, settings)


# ---------- Entrée ----------
if __name__ == "__main__":
    load_project_dotenv()
    application_context = ApplicationContext.from_environment()
    paths = ApplicationPaths.from_runtime(catalogue_mode=application_context.mode)
    configure_logging(paths)
    LOGGER.info(
        "Demarrage AssembleurTriangles mode=%s python=%s installation=%s userdata=%s catalogue=%s scenarios=%s",
        application_context.mode,
        sys.version.split()[0],
        paths.installation_root,
        paths.user_data_root,
        paths.active_catalogue_dir,
        paths.active_scenarios_dir,
    )
    try:
        app = TriangleViewerManual()
        app.mainloop()
    except (OSError, RuntimeError, ValueError, tk.TclError):
        LOGGER.exception("Erreur pendant le demarrage ou l'execution de l'application")
        raise
