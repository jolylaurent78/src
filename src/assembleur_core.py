"""assembleur_core.py
Noyau 'pur' (sans Tk) : géométrie + structures de base.
"""

import math
import datetime as _dt
from typing import Optional, List, Dict, Tuple
import re

import numpy as np
from dataclasses import dataclass

from shapely.geometry import Polygon as _ShPoly, LineString as _ShLine, Point as _ShPoint
from shapely.ops import unary_union as _sh_union


def _overlap_shrink(S_m, S_t, stroke_px: float, zoom: float) -> bool:
    """
    Anti-chevauchement unique et partagé : SHRINK-ONLY sur les deux solides.
    - Rétrécit S_m et S_t de 2× (stroke_px / zoom) en unités monde,
      puis teste l'intersection des solides réduits.
    - Retourne True si chevauchement détecté, False sinon.
    """
    shrink_world = 2.0 * (float(stroke_px) / max(zoom, 1e-9))
    S_t_shrunk = S_t.buffer(-shrink_world).buffer(0)
    S_m_shrunk = S_m.buffer(-shrink_world).buffer(0)
    if S_t_shrunk.is_empty or S_m_shrunk.is_empty:
        return False
    return S_m_shrunk.intersects(S_t_shrunk)


def _tri_shape(P: Dict[str, np.ndarray]):
    """Triangle → Polygon (brut, sans rétrécissement)."""
    poly = _ShPoly([tuple(map(float, P["O"])),
                    tuple(map(float, P["B"])),
                    tuple(map(float, P["L"]))])
    # 'buffer(0)' nettoie les très petites imprécisions numériques
    return poly.buffer(0)


def _group_shape_from_nodes(nodes, last_drawn):
    """Union BRUTE des triangles listés dans 'nodes' (sans rétrécissement)."""
    polys = []
    for nd in nodes:
        tid = nd.get("tid")
        if tid is None or not (0 <= tid < len(last_drawn)):
            continue
        polys.append(_tri_shape(last_drawn[tid]["pts"]))
    if not polys:
        return None
    # nettoie et unionne
    return _sh_union([p.buffer(0) for p in polys])


def _build_local_triangle(OB: float, OL: float, BL: float) -> dict:
    """
    Construit un triangle local avec O=(0,0), B=(OB,0), L=(x,y) à partir des 3 longueurs.
    """
    x = (OL*OL - BL*BL + OB*OB) / (2*OB)
    y2 = max(0.0, OL*OL - x*x)
    y = math.sqrt(y2)
    return {
        "O": np.array([0.0,     0.0], dtype=float),
        "B": np.array([float(OB), 0.0], dtype=float),
        "L": np.array([float(x),  float(y)], dtype=float),
    }

# --- Simulation d'une pose (pour filtrer le chevauchement après collage) ---
def _apply_R_T_on_P(P: Dict[str,np.ndarray], R: np.ndarray, T: np.ndarray, pivot: np.ndarray) -> Dict[str,np.ndarray]:
    """Applique une rotation R autour de 'pivot' puis une translation T aux points P."""
    out = {}
    for k in ("O","B","L"):
        v = np.array(P[k], dtype=float)
        out[k] = (R @ (v - pivot)) + pivot + T
    return out

def _pose_params(Pm: Dict[str,np.ndarray], am: str, bm: str, Vm: np.ndarray,
                 Pt: Dict[str,np.ndarray], at: str, bt: str, Vt: np.ndarray):
    """Retourne (R, T, pivot) pour la pose 'am->bm' alignée sur 'at->bt' avec Vm→Vt."""
    vm_dir = np.array(Pm[bm], float) - np.array(Pm[am], float)
    vt_dir = np.array(Pt[bt], float) - np.array(Pt[at], float)
    ang_m = math.atan2(vm_dir[1], vm_dir[0])
    ang_t = math.atan2(vt_dir[1], vt_dir[0])
    dtheta = ang_t - ang_m
    c, s = math.cos(dtheta), math.sin(dtheta)
    R = np.array([[c, -s],[s, c]], float)
    pivot = np.array(Vm, float)
    Vm_rot = (R @ (np.array(Pm[am], float) - pivot)) + pivot
    T = np.array(Vt, float) - Vm_rot
    return R, T, pivot

class ScenarioAssemblage:
    """
    Représente un scénario d'assemblage (manuel ou automatique).
    Un scénario contient :
      - un nom lisible (ex. 'Scénario manuel', 'Auto #1'),
      - un type de source ('manual' / 'auto'),
      - éventuellement l'identifiant de l'algorithme qui l'a généré,
      - la liste des triangles d'origine (ids),
      - l'état géométrique : triangles dessinés + groupes.

    Pour le scénario manuel, last_drawn / groups sont des références
    directes vers les structures runtime du viewer (pas une copie).
    Pour les scénarios automatiques, on utilisera plutôt des copies.
    """
    def __init__(self, name: str, source_type: str = "manual", algo_id: Optional[str] = None, tri_ids: Optional[List[int]] = None):
        self.name: str = name
        self.source_type: str = source_type      # "manual" ou "auto"
        self.algo_id: Optional[str] = algo_id    # id de l'algo (pour les auto)
        self.tri_ids: List[int] = list(tri_ids) if tri_ids is not None else []

        # État géométrique associé au scénario
        self.last_drawn: List[Dict] = []         # même structure que viewer._last_drawn
        self.groups: Dict[int, Dict] = {}        # même structure que viewer.groups

        # --- Vue & Carte ---
        # Vue: zoom + offset (pan). Stockée pour retrouver exactement l'affichage au switch.
        # Carte: état du fond (fichier + worldRect + visibilité/opacité + scale affiché).
        # Pour les scénarios automatiques, la carte peut être partagée au niveau du viewer.
        self.view_state: Dict[str, float] = {}   # {"zoom":..., "offset_x":..., "offset_y":...}
        self.map_state: Dict[str, object] = {}   # {"path":..., "x0":..., "y0":..., "w":..., "h":..., "visible":..., "opacity":..., "scale":...}
 
        # Statut global du scénario (utile pour les simulations auto)
        self.status: str = "complete"            # "complete", "pruned", etc.
        self.created_at: _dt.datetime = _dt.datetime.now()



# =============================================================================
# TopologyModel v4.3 (Core) – Modèle objet V1 (sans Tk) – IDs lisibles (Phase 2)
# =============================================================================
#
# Convention d’identifiants lisibles :
# - Element (triangle instance) : "<scenarioId>:T<triRank:02d>"         ex: "S1:T01"
# - Node atomique (par sommet) : "<scenarioId>:T<triRank:02d}:N<idx>"   ex: "S1:T01:N0"
# - Group (DSU)                : "<scenarioId>:G<k:03d>"                ex: "S1:G001"
# - Attachment                 : "<scenarioId>:A<k:03d>"                ex: "S1:A003"
#
# Remarques :
# - Les IDs sont des strings pour une lisibilité maximale (TopoDump XML).
# - La règle “plus récent gagne” utilise createdOrder (int monotone interne),
#   et PAS l’ordre lexicographique des IDs.
# - t est fourni par l’UI en référentiel monde (km).
# - fusionDistanceKm (km) pilote la fusion de points sur une arête :
#       abs(t1 - t2) * edgeLengthKm <= fusionDistanceKm  => fusion
#   et on conserve le t du point canonique (référence contour).
#
# Couverture V1 :
# - vertex↔vertex
# - vertex↔edge (t obligatoire)
# - edge↔edge (mapping "direct"|"reverse") + coverage total [0,1] sur les deux arêtes
#
# Dégrouper/Undo (MVP) :
# - suppression d’attaches + rebuild complet depuis la liste des attachments.
#

import re
import xml.etree.ElementTree as _ET


class TopologyNodeType:
    """Types de nœuds topologiques (métier) et priorité associée.

    Rôle :
    - Représenter le 'type' métier d’un nœud : Ouverture (O), Base (B), Lumière (L).
    - Fournir une fonction de rang pour la canonisation DSU : L > B > O.

    Ne fait PAS :
    - Ne stocke aucune donnée d’instance (simple namespace de constantes).

    V1 :
    - La règle de priorité L > B > O est figée par la spec v4.2.
    """
    OUVERTURE = "O"
    BASE = "B"
    LUMIERE = "L"

    @staticmethod
    def rank(node_type: str) -> int:
        if node_type == TopologyNodeType.LUMIERE:
            return 3
        if node_type == TopologyNodeType.BASE:
            return 2
        return 1


class TopologyFeatureType:
    """Types de features référencées par TopologyFeatureRef.

    Rôle :
    - Distinguer explicitement les deux natures de cible manipulées par la topologie :
      - vertex (sommet d’un élément)
      - edge (arête d’un élément)

    Ne fait PAS :
    - Ne contient aucune logique topo ; c’est un namespace de constantes.

    V1 :
    - Suffisant pour vertex↔vertex, vertex↔edge et edge↔edge.
    """
    VERTEX = "vertex"
    EDGE = "edge"


class TopologyFeatureRef:
    """Référence typée et sérialisable vers une feature topologique.

    Rôle :
    - Pointer de manière stable vers une feature (Vertex ou Edge) dans un élément donné :
      (feature_type, element_id, index).
    - Permettre de stocker/rejouer des TopologyAttachment sans dépendre d’objets Python en mémoire.
    - Servir directement à la sérialisation (TopoDump XML) : export lisible et diffable.

    Ne fait PAS :
    - Ne porte aucune décision topologique (pas de fusion, pas de merge).
    - Ne dépend pas du canvas Tk ; aucun calcul géométrique.

    Note :
    - Le champ optionnel t est une commodité de debug/trace. Dans la spec v4.2,
      t est fourni par l’UI (référentiel monde, km) et consommé par le Core.

    V1 :
    - Utilisée comme brique de base pour les attachments et l’export TopoDump.
    """
    def __init__(self, feature_type: str, element_id: str, index: int, t: float | None = None):
        self.feature_type = str(feature_type)
        self.element_id = str(element_id)
        self.index = int(index)
        self.t = float(t) if t is not None else None

    def to_string(self) -> str:
        if self.t is None:
            return f"{self.feature_type}({self.element_id},{self.index})"
        return f"{self.feature_type}({self.element_id},{self.index},t={self.t:.6f})"


class TopologyAttachment:
    """Trace métier d’une intention topologique binaire (action rejouable).

    Rôle :
    - Représenter UNE action topologique atomique entre deux features (A ↔ B).
    - Constituer la source de vérité pour reconstruire la topologie (rebuild depuis la liste d’attaches).
    - Être sérialisable (TopoDump XML) et diffable (par scénario).

    Ne fait PAS :
    - Ne modifie pas directement le modèle : l’application est effectuée par TopologyWorld.
    - Ne stocke pas de coordonnées (la géométrie est hors topo ; t est fourni par l’UI).

    V1 :
    - Stockage + export.
    - Supporte vertex↔vertex, vertex↔edge et edge↔edge (mapping direct|reverse).
    - La reconstruction (rebuild) depuis la liste d’attaches est la stratégie de référence.
    """
    def __init__(self, attachment_id: str, kind: str,
                 feature_a: TopologyFeatureRef, feature_b: TopologyFeatureRef,
                 params: dict | None = None, source: str = "manual"):
        self.attachment_id = str(attachment_id)
        self.kind = str(kind)
        self.feature_a = feature_a
        self.feature_b = feature_b
        self.params = dict(params) if params is not None else {}
        self.source = str(source)


class TopologyVertex:
    """Sommet d’un élément (polygone), labellisé (ville), rattaché à un node atomique.

    Rôle :
    - Porter le label (nom de ville) utilisé pour l’affichage et les clés de matching.
    - Référencer le node atomique d’origine (node_id) et son type (O/B/L) à la création.
    - Fournir une identité stable : (element_id, vertex_index).

    Ne fait PAS :
    - Ne calcule pas la topologie ; il ne fait que référencer le node d’origine.
    - Ne dépend pas de la géométrie (pas de coordonnées).

    V1 :
    - Les unions (DSU) opèrent au niveau TopologyWorld via node_id (atomique) -> node canonique.
    """
    def __init__(self, element_id: str, vertex_index: int, label: str, node_id: str, node_type: str):
        self.element_id = str(element_id)
        self.vertex_index = int(vertex_index)
        self.label = str(label)
        self.node_id = str(node_id)
        self.node_type = str(node_type)


class TopologyEdgePoint:
    """Point repéré sur une arête paramétrique (Option B).

    Rôle :
    - Représenter un point 'sur l’arête' via un paramètre t ∈ [0,1] et un node atomique associé.
    - Servir à la construction du contour (boundary) et à la validation des inserts (tri par t).

    Ne fait PAS :
    - Ne contient aucune géométrie : t est un paramètre adimensionnel, référencé monde côté UI.
    - Ne fait aucune canonisation : TopologyWorld résout node atomique -> node canonique.

    V1 :
    - Les points (0.0 et 1.0) existent toujours (extrémités).
    - Les insertions respectent la politique de fusion en km.
    """
    def __init__(self, t: float, node_id: str):
        self.t = float(t)
        self.node_id = str(node_id)


class TopologyCoverageInterval:
    """Intervalle de couverture sur une arête (portion interne), exprimé en t.

    Rôle :
    - Modéliser des portions d’arêtes devenues internes (collage edge↔edge, etc.),
      sous forme d’intervalles [t0, t1].
    - Permettre le calcul du boundary comme complément des coverages.

    Ne fait PAS :
    - Ne décide pas quand une couverture apparaît : cela dépend des attachments et des règles métier.

    V1 :
    - Structure utilisée.
    - Un collage edge↔edge produit un coverage total [0.0, 1.0] sur chacune des deux arêtes collées.
    - Le boundary sera calculé comme le complément de l’union des coverages sur [0,1].
    """
    def __init__(self, t0: float, t1: float):
        self.t0 = float(t0)
        self.t1 = float(t1)


class TopologyEdge:
    """Arête paramétrique d’un élément (Option B) : points internes + coverages.

    Rôle :
    - Représenter une arête orientée (v_start -> v_end) avec :
      - une liste de pointsOnEdge triés par t (incluant les extrémités),
      - des coverages (intervalles internes) pour le calcul du boundary.
    - Fournir des clés de matching (edgeLabelKey) à partir des labels de sommets.
    - Appliquer la politique de fusion des points sur l’arête (sélection du candidat).

    Ne fait PAS :
    - Ne fait pas de DSU : la canonisation des nodes est dans TopologyWorld.
    - Ne dépend pas du canvas Tk ; aucune logique d’affichage.

    V1 :
    - Implémente la recherche de candidat de fusion à partir de fusionDistanceKm (km) et edgeLengthKm.
    - Le calcul complet du boundary sera implémenté ultérieurement.
    """
    def __init__(self, element_id: str, edge_index: int, v_start: TopologyVertex, v_end: TopologyVertex,
                 edge_length_km: float):
        self.element_id = str(element_id)
        self.edge_index = int(edge_index)
        self.v_start = v_start
        self.v_end = v_end
        self.edge_length_km = float(edge_length_km)

        self.points_on_edge: list[TopologyEdgePoint] = [
            TopologyEdgePoint(0.0, v_start.node_id),
            TopologyEdgePoint(1.0, v_end.node_id),
        ]
        self.coverages: list[TopologyCoverageInterval] = []

    def edge_id(self) -> str:
        return f"{self.element_id}:E{self.edge_index}"

    def edge_label_key(self) -> tuple[str, str]:
        a = self.v_start.label
        b = self.v_end.label
        return (a, b) if a <= b else (b, a)

    def labels_display(self) -> str:
        return f"{self.v_start.label}–{self.v_end.label}"

    def find_fusion_candidate_index(self, t_new: float, fusion_distance_km: float, node_rank_fn) -> int | None:
        if self.edge_length_km <= 0:
            return None
        candidates = []
        for i, p in enumerate(self.points_on_edge):
            delta_km = abs(float(t_new) - float(p.t)) * self.edge_length_km
            if delta_km <= float(fusion_distance_km):
                candidates.append(i)
        if not candidates:
            return None
        best_i = candidates[0]
        best_key = node_rank_fn(self.points_on_edge[best_i].node_id)
        for i in candidates[1:]:
            key_i = node_rank_fn(self.points_on_edge[i].node_id)
            if key_i > best_key:
                best_key = key_i
                best_i = i
        return best_i

    def insert_point_sorted(self, t_new: float, node_id: str):
        self.points_on_edge.append(TopologyEdgePoint(float(t_new), str(node_id)))
        self.points_on_edge.sort(key=lambda p: p.t)


class TopologyPose2D:
    """Pose 2D d'un élément dans le référentiel monde (rotation + translation).

    Rôle :
    - Porter la transformation Monde <- Local, sous forme (R, T) :
        p_world = R @ p_local + T
    - Être la SEULE représentation des coordonnées monde persistées dans le modèle.
      Les coordonnées monde des sommets/points sont toujours dérivées via cette pose.

    Ne fait PAS :
    - Ne modifie pas la géométrie intrinsèque (longueurs / angles) de l'élément.
    - Ne décide pas de la pose (elle est calculée/ajustée par l'UI ou les règles d'attachments).
    """
    def __init__(self, R: np.ndarray | None = None, T: np.ndarray | None = None):
        self.R = np.array(R, float) if R is not None else np.eye(2, dtype=float)
        self.T = np.array(T, float) if T is not None else np.zeros(2, dtype=float)

    def theta_rad(self) -> float:
        # R = [[c, -s],[s, c]]
        return float(math.atan2(self.R[1, 0], self.R[0, 0]))

    def to_dict(self) -> dict:
        return {
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "thetaRad": self.theta_rad(),
        }

    def compose(self, other: "TopologyPose2D") -> "TopologyPose2D":
        """Compose deux poses : self ∘ other.
        Si p' = other(p) et p'' = self(p'), alors compose() retourne la pose p'' = (self ∘ other)(p).
        """
        R = self.R @ other.R
        T = (self.R @ other.T) + self.T
        return TopologyPose2D(R=R, T=T)

    @staticmethod
    def identity() -> "TopologyPose2D":
        return TopologyPose2D(R=np.eye(2, dtype=float), T=np.zeros(2, dtype=float))


class TopologyElementPose2D:
    """Pose 2D d'un élément : rotation (det=+1) + translation + réflexion locale.

    Décision de modèle (session) :
    - Les coordonnées locales des éléments (vertex_local_xy) ne sont jamais modifiées.
    - La pose monde est portée par l'élément (pas par les groupes).

    Convention stable (local -> world) :
      - si mirrored == False : p_world = R @ p_local + T
      - si mirrored == True  : p_world = R @ (M @ p_local) + T
        avec M = diag(-1, +1) dans le repère local.

    Note : on impose R ~ rotation pure (det ~ +1) et on stocke toute réflexion
    dans le flag mirrored.
    """

    def __init__(self, R: np.ndarray | None = None, T: np.ndarray | None = None, mirrored: bool = False):
        self.R = np.array(R, float) if R is not None else np.eye(2, dtype=float)
        self.T = np.array(T, float) if T is not None else np.zeros(2, dtype=float)
        self.mirrored = bool(mirrored)

    @staticmethod
    def identity() -> "TopologyElementPose2D":
        return TopologyElementPose2D(R=np.eye(2, dtype=float), T=np.zeros(2, dtype=float), mirrored=False)

    @staticmethod
    def mirror_matrix() -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)

    def local_to_world(self, p_local: np.ndarray) -> np.ndarray:
        p = np.array(p_local, dtype=float)
        if self.mirrored:
            p = self.mirror_matrix() @ p
        return (self.R @ p) + self.T

class TopologyElement:
    """Élément topologique : polygone (triangle = cas particulier n=3).

    Rôle :
    - Regrouper les labels de sommets (villes) et leurs types (O/B/L) dans un cycle ordonné.
    - Porter les longueurs d’arêtes en km (monde) nécessaires à la politique de fusion sur t.
    - Servir de conteneur pour les TopologyVertex et TopologyEdge instanciés par TopologyWorld.

    Ne fait PAS :
    - Ne crée pas lui-même les nodes : TopologyWorld crée un node atomique par sommet.
    - Ne dépend pas du canvas Tk (pixels/zoom/pan).

    Notes (v4.3) :
    - La 'vérité' géométrique du triangle est intrinsèque (longueurs / angles).
    - La pose monde (rotation + translation) permet de dériver des coordonnées monde si nécessaire.
    - Les coordonnées monde ne sont pas une vérité métier ; elles sont dérivées.

    V1 :
    - edge_lengths_km peut être fourni (recommandé). À défaut, une valeur par défaut est posée
      et remplacée ensuite par la géométrie monde.
    """
    def __init__(self, element_id: str, name: str,
                 vertex_labels: list[str], vertex_types: list[str],
                 edge_lengths_km: list[float],
                 intrinsic_sides_km: dict[str, float] | None = None,
                 local_frame: dict | None = None,
                 vertex_local_xy: dict[int, tuple[float, float]] | None = None,
                 meta: dict | None = None):
        self.element_id = str(element_id)
        self.name = str(name)
        self.meta = dict(meta) if meta is not None else {}

        if len(vertex_labels) < 3:
            raise ValueError("TopologyElement: un polygone doit avoir au moins 3 sommets")
        if len(vertex_labels) != len(vertex_types):
            raise ValueError("TopologyElement: vertex_types doit avoir la même taille que vertex_labels")
        if len(edge_lengths_km) != len(vertex_labels):
            raise ValueError("TopologyElement: edge_lengths_km doit avoir la même taille que les sommets")

        self.vertex_labels = [str(x) for x in vertex_labels]
        self.vertex_types = [str(x) for x in vertex_types]
        self.edge_lengths_km = [float(x) for x in edge_lengths_km]

        self.intrinsic_sides_km: dict[str, float] = dict(intrinsic_sides_km) if intrinsic_sides_km is not None else {}

        # --- Repère local (coordonnées relatives) ---
        # Convention recommandée (v4.3+):
        # - origine au sommet Ouverture (O) -> (0,0)
        # - axe X aligné sur le segment O->B (si présent)
        # Les coordonnées locales sont déterministes et dérivées de la géométrie intrinsèque.
        self.local_frame: dict = dict(local_frame) if local_frame is not None else {
            "origin": TopologyNodeType.OUVERTURE,
            "xAxis": "O->B",
            "units": "km",
        }

        # vertex_local_xy : coordonnées relatives des sommets (par index), exprimées dans local_frame.
        # Ex: { idxO:(0,0), idxB:(OB,0), idxL:(x,y) }
        self.vertex_local_xy: dict[int, tuple[float, float]] = dict(vertex_local_xy) if vertex_local_xy is not None else {}

        if not self.vertex_local_xy:
            self._try_build_default_local_coords()

        self.vertexes: list[TopologyVertex] = []
        self.edges: list[TopologyEdge] = []

        # --- Pose monde (par élément) ---
        # IMPORTANT: les coordonnées locales (vertex_local_xy) restent figées.
        # La position/orientation en monde est exclusivement portée par cette pose.
        self.pose: TopologyElementPose2D = TopologyElementPose2D.identity()

    # --- Pose API (élément) ---
    def get_pose(self) -> tuple[np.ndarray, np.ndarray, bool]:
        return (np.array(self.pose.R, float), np.array(self.pose.T, float), bool(self.pose.mirrored))

    def set_pose(self, R: np.ndarray, T: np.ndarray, mirrored: bool | None = None) -> None:
        self.pose.R = np.array(R, float)
        self.pose.T = np.array(T, float)
        if mirrored is not None:
            self.pose.mirrored = bool(mirrored)

    def localToWorld(self, p_local: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self.pose.local_to_world(np.array(p_local, dtype=float))

    def _try_build_default_local_coords(self):
        """Construit des coordonnées locales canoniques si l'élément est un triangle O/B/L.

        Stratégie V1 :
        - On détecte les indices des sommets de type O, B, L.
        - On reconstruit les longueurs OB, OL, BL depuis edge_lengths_km (cycle).
        - On pose O=(0,0), B=(OB,0), L=(x,y) (y>=0) via la loi des cosinus.

        Si l'élément n'est pas un triangle O/B/L (hors-scope V1), la méthode ne fait rien.
        Si l'élément EST un triangle O/B/L mais que les longueurs nécessaires sont absentes,
        c'est un problème de données et on lève une exception explicite (pas de masquage).
        """
        if len(self.vertex_labels) != 3:
            return

        # Hors-scope V1 : si on n'a pas exactement un O, un B et un L, on ne force pas.
        if (TopologyNodeType.OUVERTURE not in self.vertex_types) or \
           (TopologyNodeType.BASE not in self.vertex_types) or \
           (TopologyNodeType.LUMIERE not in self.vertex_types):
            return

        # indices par type (première occurrence)
        idxO = self.vertex_types.index(TopologyNodeType.OUVERTURE)
        idxB = self.vertex_types.index(TopologyNodeType.BASE)
        idxL = self.vertex_types.index(TopologyNodeType.LUMIERE)

        # map des longueurs par paire non orientée (triangle)
        n = 3
        pair_len: dict[frozenset, float] = {}
        for i in range(n):
            j = (i + 1) % n
            pair_len[frozenset((i, j))] = float(self.edge_lengths_km[i])

        def d(i, j) -> float:
            return float(pair_len.get(frozenset((i, j)), 0.0))

        OB = d(idxO, idxB)
        OL = d(idxO, idxL)
        BL = d(idxB, idxL)
        if OB <= 0 or OL <= 0 or BL <= 0:
            raise ValueError(
                f"TopologyElement({self.element_id}): longueurs invalides pour repère local O/B/L "
                f"(OB={OB}, OL={OL}, BL={BL})"
            )

        P = _build_local_triangle(OB=OB, OL=OL, BL=BL)

        # Convention import historique : si orient == "CW", miroir par rapport à l'axe X local.
        # (équivalent à P["L"][1] *= -1 dans l'ancien code Tk)
        orient = str(self.meta.get("orient", "")).strip().upper()
        if orient == "CW":
            P["L"][1] = -float(P["L"][1])

        # Remap vers les indices réels
        self.vertex_local_xy[idxO] = (float(P["O"][0]), float(P["O"][1]))
        self.vertex_local_xy[idxB] = (float(P["B"][0]), float(P["B"][1]))
        self.vertex_local_xy[idxL] = (float(P["L"][0]), float(P["L"][1]))

        return

    def n_vertices(self) -> int:
        return len(self.vertex_labels)


class TopologyGroup:
    """Groupe topologique : composante connexe d’éléments (unité de manipulation).

    Rôle :
    - Représenter un ensemble d’éléments reliés par des attachments (groupe d’assemblage).
    - Porter la liste des elementIds et des attachmentIds associés au groupe canonique.

    Ne fait PAS :
    - Ne calcule pas la connectivité : TopologyWorld gère l’union-find des groupes.
    - Ne stocke pas de géométrie.

    V1 :
    - Structure légère : l’objectif est d’avoir un conteneur stable pour l’export TopoDump.
    """
    def __init__(self, group_id: str):
        self.group_id = str(group_id)
        self.element_ids: list[str] = []
        self.attachment_ids: list[str] = []

@dataclass
class ConceptNodeInfo:
    concept_id: str
    members: set[str]
    sumx: float = 0.0
    sumy: float = 0.0
    count: int = 0

    def add_occ(self, x: float, y: float) -> None:
        self.sumx += float(x)
        self.sumy += float(y)
        self.count += 1

    def world_xy(self) -> tuple[float, float]:
        if self.count <= 0:
            return (0.0, 0.0)
        return (self.sumx / self.count, self.sumy / self.count)


@dataclass
class ConceptEdgeInfo:
    a: str
    b: str
    occurrences: list[dict]

@dataclass
class ConceptGroupCache:
    """Cache du modèle conceptuel pour un groupe canonique."""
    valid: bool = False
    nodes: dict[str, ConceptNodeInfo] = None
    edges: dict[tuple[str, str], ConceptEdgeInfo] = None

    def __post_init__(self) -> None:
        if self.nodes is None:
            self.nodes = {}
        if self.edges is None:
            self.edges = {}

class TopologyWorld:
    """Racine métier du modèle topologique (Core, sans Tk).

    Rôle :
    - Détenir l’ensemble des éléments, groupes, nodes et attachments.
    - Implémenter les DSU non destructifs :
      - nodes : canonisation métier (L > B > O, puis plus récent ID max),
      - groupes : canonisation par plus récent (ID max).
    - Appliquer la politique de fusion des points sur une arête en km :
      abs(t1 - t2) * edgeLengthKm <= fusionDistanceKm  => fusion
      et conserver le t du point canonique (référence contour).
    - Fournir un export TopoDump XML manuel (par scénario), lisible et diffable.

    Ne fait PAS :
    - Ne dépend jamais du canvas Tk.
    - Ne calcule pas t (fourni par l’UI en référentiel monde, km).

    V1 :
    - Pose les fondations (classes + DSU + export TopoDump).
    - Applique vertex↔vertex, vertex↔edge et edge↔edge (mapping direct|reverse) au niveau DSU + coverages.
    - Dégrouper/undo = suppression d’attaches puis rebuild complet (petits scénarios).
    """
    def __init__(self, scenario_id: str):
        self.scenario_id = str(scenario_id)
        self.fusion_distance_km: float = 1.0

        # Repère "monde" pour les calculs d'azimut (0°=Nord, sens horaire).
        # - True  : Y augmente vers le bas (repère image/écran)  -> dyNord = -dy
        # - False : Y augmente vers le Nord (repère carto)        -> dyNord =  dy
        self.worldYAxisDown: bool = False

        # --- cache topo conceptuelle (PAR GROUPE) ---
        self._concept_by_gid: dict[str, ConceptGroupCache] = {}

        self.groups: dict[str, TopologyGroup] = {}
        self.elements: dict[str, TopologyElement] = {}
        self.element_to_group: dict[str, str] = {}
        self.attachments: dict[str, TopologyAttachment] = {}

        self._node_parent: dict[str, str] = {}
        self._node_type: dict[str, str] = {}
        self._node_members: dict[str, list[str]] = {}
        self._node_created_order: dict[str, int] = {}

        self._group_parent: dict[str, str] = {}
        self._group_members: dict[str, list[str]] = {}
        self._group_created_order: dict[str, int] = {}

        self._created_counter_nodes = 0
        self._created_counter_groups = 0
        self._created_counter_attachments = 0

    # ------------------------------------------------------------------
    # Topologie conceptuelle (MVP)
    # - concept node = find_node(canon)
    # - position concept node = moyenne des occurrences monde (points_on_edge)
    # - concept edge = segment entre 2 concept nodes consécutifs sur une arête
    # ------------------------------------------------------------------
    def invalidateConceptCache(self, group_id: str | None = None) -> None:
        """Invalide le cache conceptuel.
        - group_id=None : tous les groupes
        - group_id!=None : uniquement le groupe canonique correspondant
        """
        if group_id is None:
            for c in self._concept_by_gid.values():
                c.valid = False
            return
        gid = self.find_group(str(group_id))
        c = self._concept_by_gid.get(gid)
        if c is not None:
            c.valid = False

    def toConceptNodeId(self, node_id: str) -> str:
        return self.find_node(str(node_id))

    def _concept_cache(self, group_id: str) -> ConceptGroupCache:
        gid = self.find_group(str(group_id))
        c = self._concept_by_gid.get(gid)
        if c is None:
            c = ConceptGroupCache()
            self._concept_by_gid[gid] = c
        return c

    @staticmethod
    def _element_id_from_atomic_node_id(node_id: str) -> str | None:
        s = str(node_id)
        if ":N" not in s:
            return None
        return s.rsplit(":N", 1)[0]

    def _infer_group_for_node(self, node_id: str) -> str | None:
        cn = self.find_node(str(node_id))
        mem = self.node_members(cn)
        if not mem:
            return None
        eid = self._element_id_from_atomic_node_id(mem[0])
        if not eid:
            return None
        return self.element_to_group.get(str(eid))

    def _iter_group_elements(self, group_id: str) -> list["TopologyElement"]:
        gid = self.find_group(str(group_id))
        g = self.groups.get(gid)
        if g is None:
            return []
        out: list[TopologyElement] = []
        for eid in g.element_ids:
            el = self.elements.get(str(eid))
            if el is not None:
                out.append(el)
        return out

    def _buildConceptCacheForGroup(self, group_id: str) -> None:
        gid = self.find_group(str(group_id))
        c = self._concept_cache(gid)

        nodes: dict[str, ConceptNodeInfo] = {}
        edges: dict[tuple[str, str], ConceptEdgeInfo] = {}

        for el in self._iter_group_elements(gid):
            for edge in getattr(el, "edges", []):
                pts = list(getattr(edge, "points_on_edge", []))
                if not pts:
                    continue
                pts.sort(key=lambda p: float(p.t))

                # nodes : occurrences monde
                for p in pts:
                    cn = self.find_node(str(p.node_id))
                    info = nodes.get(cn)
                    if info is None:
                        info = ConceptNodeInfo(concept_id=cn, members=set(self.node_members(cn)))
                        nodes[cn] = info
                    p_local = self._localPointOnEdge(el, edge, float(p.t))
                    p_world = el.localToWorld(p_local)
                    info.add_occ(p_world[0], p_world[1])

                # edges : segments consécutifs canonisés
                canon_seq: list[tuple[float, str]] = []
                last_cn = None
                for p in pts:
                    cn = self.find_node(str(p.node_id))
                    if cn == last_cn:
                        continue
                    canon_seq.append((float(p.t), cn))
                    last_cn = cn

                for (t0, n0), (t1, n1) in zip(canon_seq[:-1], canon_seq[1:]):
                    if n0 == n1:
                        continue
                    k = (n0, n1) if n0 < n1 else (n1, n0)
                    ce = edges.get(k)
                    if ce is None:
                        ce = ConceptEdgeInfo(a=k[0], b=k[1], occurrences=[])
                        edges[k] = ce
                    ce.occurrences.append({
                        "elementId": str(el.element_id),
                        "edgeId": str(edge.edge_id()),
                        "t0": float(t0),
                        "t1": float(t1),
                    })

        c.nodes = nodes
        c.edges = edges
        c.valid = True

    def buildConceptCacheIfNeeded(self, group_id: str) -> None:
        gid = self.find_group(str(group_id))
        c = self._concept_cache(gid)
        if not c.valid:
            self._buildConceptCacheForGroup(gid)

    def getConceptNodeWorldXY(self, node_id: str, group_id: str | None = None) -> tuple[float, float]:
        gid = group_id if group_id is not None else self._infer_group_for_node(node_id)
        if gid is None:
            return (0.0, 0.0)
        self.buildConceptCacheIfNeeded(gid)
        cn = self.find_node(str(node_id))
        c = self._concept_cache(gid)
        info = c.nodes.get(cn)
        if info is None:
            return (0.0, 0.0)
        return info.world_xy()


    def getConceptRays(self, node_id: str, group_id: str | None = None) -> list[dict]:
        """Rays conceptuels au node canonique, LIMITÉS AU GROUPE."""
        gid = group_id if group_id is not None else self._infer_group_for_node(node_id)
        if gid is None:
            return []
        self.buildConceptCacheIfNeeded(gid)
        cn = self.find_node(str(node_id))
        p0 = self.getConceptNodeWorldXY(cn, gid)
        c = self._concept_cache(gid)
        out: list[dict] = []
        for (a, b), ce in c.edges.items():
            if cn != a and cn != b:
                continue
            other = b if cn == a else a
            p1 = self.getConceptNodeWorldXY(other, gid)
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            out.append({
                "fromNode": cn,
                "toNode": other,
                "fromLabel": self.getNodeLabel(cn),
                "toLabel": self.getNodeLabel(other),
                "dx": dx,
                "dy": dy,
                "azimutDeg": self.azimutDegFromDxDy(dx, dy),
                "occurrences": list(ce.occurrences),
            })
        out.sort(key=lambda r: (float(r.get("azimutDeg", 0.0)), str(r.get("toNode", ""))))
        return out


    # --- Parcours du graphe conceptuel (par groupe) ---
    def getConceptEdgesForNode(self, node_id: str, group_id: str | None = None) -> list[dict]:
        """Liste des edges conceptuels incidentes à un concept node (par groupe)."""
        gid = group_id if group_id is not None else self._infer_group_for_node(node_id)
        if gid is None:
            return []
        self.buildConceptCacheIfNeeded(gid)
        cn = self.find_node(str(node_id))
        c = self._concept_cache(gid)
        out: list[dict] = []
        for (a, b), ce in c.edges.items():
            if cn != a and cn != b:
                continue
            other = b if cn == a else a
            out.append({
                "a": a, "b": b,
                "this": cn,
                "other": other,
                "occurrences": list(ce.occurrences),
            })
        return out

    def getConceptNeighborNodes(self, node_id: str, group_id: str | None = None) -> list[str]:
        """Liste des concept nodes voisins (ids canoniques) connectés à node_id (par groupe)."""
        edges = self.getConceptEdgesForNode(node_id, group_id)
        neigh = [e["other"] for e in edges]
        # unique + stable
        seen = set()
        out = []
        for n in neigh:
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
        return out

    def getConceptNodeType(self, node_id: str) -> str:
        """Type conceptuel: 'M' si mélange de types, sinon type unique."""
        cn = self.find_node(str(node_id))
        mem = self.node_members(cn)
        if not mem:
            return str(self.node_type(cn))
        types = {str(self.node_type(m)) for m in mem}
        return "M" if len(types) > 1 else next(iter(types))

    # --- IDs helpers ---
    @staticmethod
    def format_element_id(scenario_id: str, tri_rank_1based: int) -> str:
        return f"{scenario_id}:T{int(tri_rank_1based):02d}"

    @staticmethod
    def parse_tri_rank_from_element_id(element_id: str) -> int | None:
        m = re.search(r":T(\d+)$", str(element_id))
        if not m:
            return None
        return int(m.group(1))


    # Azimut (C7) – Référence unique
    def azimutDegFromDxDy(self, dx: float, dy: float) -> float:
        """Calcule l'azimut standard (0°=Nord, sens horaire) à partir d'un delta.

        Convention :
        - dx = Est  (x2 - x1)
        - dy = variation sur l'axe Y du repère monde courant.

        IMPORTANT : si `self.worldYAxisDown` est True (repère image/écran), alors
        `dy` pointe vers le Sud quand il est positif. On convertit donc en dyNord = -dy
        avant d'appliquer la formule de référence.

        Retour : angle normalisé dans [0..360).
        """
        dx = float(dx)
        dy = float(dy)
        dy_north = -dy if bool(getattr(self, "worldYAxisDown", True)) else dy
        a = math.degrees(math.atan2(dx, dy_north))  # atan2(E, N)
        return float((a + 360.0) % 360.0)

    def azimutDegFromPoints(self, p0_world: np.ndarray, p1_world: np.ndarray) -> float:
        """Azimut standard de p0 vers p1 (0°=Nord, sens horaire)."""
        dx = float(p1_world[0] - p0_world[0])
        dy = float(p1_world[1] - p0_world[1])
        return self.azimutDegFromDxDy(dx, dy)

    def format_node_id(self, element_id: str, vertex_index: int) -> str:
        tri = self.parse_tri_rank_from_element_id(element_id)
        if tri is None:
            return f"{self.scenario_id}:{element_id}:N{int(vertex_index)}"
        return f"{self.scenario_id}:T{tri:02d}:N{int(vertex_index)}"

    def new_group_id(self) -> str:
        self._created_counter_groups += 1
        gid = f"{self.scenario_id}:G{self._created_counter_groups:03d}"
        self._group_created_order[gid] = self._created_counter_groups
        return gid

    def new_attachment_id(self) -> str:
        self._created_counter_attachments += 1
        return f"{self.scenario_id}:A{self._created_counter_attachments:03d}"

    # --- DSU nodes ---
    def create_node_atomic(self, node_id: str, node_type: str) -> str:
        node_id = str(node_id)
        self._created_counter_nodes += 1
        self._node_parent[node_id] = node_id
        self._node_type[node_id] = str(node_type)
        self._node_members[node_id] = [node_id]
        self._node_created_order[node_id] = self._created_counter_nodes
        return node_id

    def find_node(self, node_id: str) -> str:
        node_id = str(node_id)
        parent = self._node_parent.get(node_id, node_id)
        if parent != node_id:
            self._node_parent[node_id] = self.find_node(parent)
        return self._node_parent.get(node_id, node_id)

    def _node_rank_key(self, node_id: str) -> tuple[int, int, str]:
        c = self.find_node(node_id)
        t = self._node_type.get(c, TopologyNodeType.OUVERTURE)
        created = self._node_created_order.get(c, 0)
        return (TopologyNodeType.rank(t), created, c)

    def union_nodes(self, a: str, b: str) -> str:
        ra = self.find_node(a)
        rb = self.find_node(b)
        if ra == rb:
            return ra
        canonical = ra if self._node_rank_key(ra) >= self._node_rank_key(rb) else rb
        other = rb if canonical == ra else ra
        self._node_parent[other] = canonical
        mem = self._node_members.get(canonical, [canonical])
        mem_other = self._node_members.get(other, [other])
        self._node_members[canonical] = mem + [x for x in mem_other if x not in mem]
        self.invalidateConceptCache(None)
        return canonical

    def node_members(self, node_id: str) -> list[str]:
        c = self.find_node(node_id)
        return list(self._node_members.get(c, [c]))

    def node_type(self, node_id: str) -> str:
        c = self.find_node(node_id)
        return self._node_type.get(c, TopologyNodeType.OUVERTURE)

    # --- DSU groups ---
    def create_group_atomic(self) -> str:
        gid = self.new_group_id()
        self._group_parent[gid] = gid
        self._group_members[gid] = [gid]
        self.groups[gid] = TopologyGroup(gid)
        return gid

    def rebuildGroupElementLists(self) -> None:
        for g in self.groups.values():
            g.element_ids = []
        for eid, gid in self.element_to_group.items():
            gidc = self.find_group(gid)
            if gidc in self.groups:
                if eid not in self.groups[gidc].element_ids:
                    self.groups[gidc].element_ids.append(eid)

    def find_group(self, group_id: str) -> str:
        group_id = str(group_id)
        parent = self._group_parent.get(group_id, group_id)
        if parent != group_id:
            self._group_parent[group_id] = self.find_group(parent)
        return self._group_parent.get(group_id, group_id)

    def _group_rank_key(self, group_id: str) -> tuple[int, str]:
        c = self.find_group(group_id)
        created = self._group_created_order.get(c, 0)
        return (created, c)

    def union_groups(self, a: str, b: str) -> str:
        ra = self.find_group(a)
        rb = self.find_group(b)
        if ra == rb:
            return ra
        canonical = ra if self._group_rank_key(ra) >= self._group_rank_key(rb) else rb
        other = rb if canonical == ra else ra
        self._group_parent[other] = canonical
        mem = self._group_members.get(canonical, [canonical])
        mem_other = self._group_members.get(other, [other])
        self._group_members[canonical] = mem + [x for x in mem_other if x not in mem]
        if canonical in self.groups and other in self.groups:
            self.groups[canonical].element_ids.extend(self.groups[other].element_ids)
            self.groups[canonical].attachment_ids.extend(self.groups[other].attachment_ids)
        self.invalidateConceptCache(canonical)
        return canonical

    def group_members(self, group_id: str) -> list[str]:
        c = self.find_group(group_id)
        return list(self._group_members.get(c, [c]))

    # --- Elements ---
    def add_element_as_new_group(self, element: TopologyElement) -> str:
        if element.element_id in self.elements:
            raise ValueError(f"TopologyWorld: elementId déjà présent: {element.element_id}")
        gid = self.create_group_atomic()
        self.elements[element.element_id] = element
        self.element_to_group[element.element_id] = gid
        self.groups[gid].element_ids.append(element.element_id)

        element.vertexes = []
        for i, (lab, typ) in enumerate(zip(element.vertex_labels, element.vertex_types)):
            nid = self.format_node_id(element.element_id, i)
            self.create_node_atomic(nid, typ)
            element.vertexes.append(TopologyVertex(element.element_id, i, lab, nid, typ))

        n = element.n_vertices()
        element.edges = []
        for i in range(n):
            v0 = element.vertexes[i]
            v1 = element.vertexes[(i + 1) % n]
            element.edges.append(TopologyEdge(element.element_id, i, v0, v1, element.edge_lengths_km[i]))

        self.invalidateConceptCache(gid)
        return gid

    def get_edge(self, element_id: str, edge_index: int) -> TopologyEdge:
        return self.elements[str(element_id)].edges[int(edge_index)]

    def get_vertex(self, element_id: str, vertex_index: int) -> TopologyVertex:
        return self.elements[str(element_id)].vertexes[int(vertex_index)]

    def get_group_of_element(self, element_id: str) -> str:
        gid0 = self.element_to_group.get(str(element_id))
        if gid0 is None:
            raise ValueError(f"Element sans groupe: {element_id}")
        return self.find_group(gid0)

    # --- API pose par élément (session) ---
    def getElementPose(self, element_id: str) -> tuple[np.ndarray, np.ndarray, bool]:
        """Retourne (R, T, mirrored) pour un élément."""
        el = self.elements.get(str(element_id))
        if el is None:
            raise ValueError(f"Element inconnu: {element_id}")
        return el.get_pose()

    def setElementPose(self, element_id: str, R: np.ndarray, T: np.ndarray, mirrored: bool | None = None) -> None:
        """Assigne la pose d'un élément (R,T,mirrored)."""
        el = self.elements.get(str(element_id))
        if el is None:
            raise ValueError(f"Element inconnu: {element_id}")
        el.set_pose(R=R, T=T, mirrored=mirrored)
        # pose change -> positions monde changent -> cache conceptuel invalide
        self.invalidateConceptCache(self.element_to_group.get(str(element_id)))

    # ------------------------------------------------------------------
    # Flip (symétrie axiale) au niveau GROUPE (mais appliquée aux poses d'éléments)
    # - groupes = topologiques
    # - LocalCoords intacts
    # - on met à jour (R,T,mirrored) pour chaque élément du groupe
    # ------------------------------------------------------------------
    @staticmethod
    def _closest_rotation(R: np.ndarray) -> np.ndarray:
        """Projette une matrice 2x2 proche d'une rotation sur SO(2) (det=+1)."""
        U, _S, Vt = np.linalg.svd(np.array(R, float))
        R2 = U @ Vt
        # Forcer det=+1
        if float(np.linalg.det(R2)) < 0.0:
            U[:, -1] *= -1.0
            R2 = U @ Vt
        return np.array(R2, float)

    @staticmethod
    def _reflection_matrix(axis_dir: np.ndarray) -> np.ndarray:
        """Matrice de réflexion (2D) par rapport à une droite passant par l'origine, dirigée par axis_dir."""
        u = np.array(axis_dir, float)
        n = float(np.hypot(u[0], u[1]))
        if n < 1e-12:
            raise ValueError("flipGroup: axisDir dégénéré")
        u /= n
        # S = 2*u*u^T - I
        I = np.eye(2, dtype=float)
        uuT = np.outer(u, u)
        return (2.0 * uuT) - I

    def flipGroup(self, group_id: str, axisPoint: np.ndarray, axisDir: np.ndarray) -> None:
        """Applique une symétrie axiale à tous les éléments du groupe.

        Contrat:
        - géométrie monde: x' = S*(x - p) + p  (p = axisPoint, S = reflection(axisDir))
        - mise à jour des poses éléments uniquement
        - LocalCoords inchangés
        """
        gid = self.find_group(str(group_id))
        g = self.groups.get(gid)
        if g is None:
            return

        p = np.array(axisPoint, float).reshape((2,))
        S = self._reflection_matrix(axisDir)
        b = p - (S @ p)  # translation affine pour une droite passant par p

        M = TopologyElementPose2D.mirror_matrix()

        for eid in list(g.element_ids):
            el = self.elements.get(str(eid))
            if el is None:
                continue
            R, T, mirrored = el.get_pose()
            R = np.array(R, float)
            T = np.array(T, float).reshape((2,))

            # A = rotation * (miroir local si mirrored)
            A = R @ (M if mirrored else np.eye(2, dtype=float))

            # Après réflexion monde: x' = S*(A p + T) + b = (S A) p + (S T + b)
            A2 = S @ A
            T2 = (S @ T) + b

            # Re-factorisation A2 en (R', mirrored') avec R' rotation pure det=+1
            detA2 = float(np.linalg.det(A2))
            if detA2 < 0.0:
                mirrored2 = True
                R2_raw = A2 @ M  # car A2 = R2 * M  => R2 = A2 * M
            else:
                mirrored2 = False
                R2_raw = A2

            R2 = self._closest_rotation(R2_raw)
            el.set_pose(R=R2, T=T2, mirrored=mirrored2)


    def elementLocalToWorld(self, element_id: str, p_local: np.ndarray | tuple[float, float]) -> np.ndarray:
        el = self.elements.get(str(element_id))
        if el is None:
            raise ValueError(f"Element inconnu: {element_id}")
        return el.localToWorld(p_local)

    def get_element_world_pose(self, element_id: str) -> TopologyPose2D:
        """Pose monde dérivée d'un élément.

        NOTE compat : TopologyPose2D ne porte pas mirrored.
        - si mirrored==False : retourne (R,T) tel quel
        - si mirrored==True  : retourne (R@M, T) (det négatif)
        """
        R, T, mirrored = self.getElementPose(str(element_id))
        if mirrored:
            R = np.array(R, float) @ TopologyElementPose2D.mirror_matrix()
        return TopologyPose2D(R=R, T=T)

    # ------------------------------------------------------------------
    # Pose Edge↔Edge (V1) : calcule la pose ABSOLUE (R,T) du groupe mobile
    # pour aligner une arête mobile sur une arête cible.
    # - mapping="direct" : (start_m -> start_t) et (end_m -> end_t)
    # - mapping="reverse": (start_m -> end_t) et (end_m -> start_t)
    # Contrat :
    #   - retourne (R_abs, T_abs) : Monde <- GroupeMobile
    #   - aucune modif de state (pur calcul)
    # ------------------------------------------------------------------
    def compute_pose_edge_to_edge(
        self,
        group_id_m: str,
        element_id_m: str,
        edge_index_m: int,
        group_id_t: str,
        element_id_t: str,
        edge_index_t: int,
        mapping: str = "direct",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcule une pose ABSOLUE (R,T) pour l'ÉLÉMENT mobile afin d'aligner une arête mobile sur une arête cible.

        NOTE: les groupes sont topologiques (pas de pose groupe).
        - Cette fonction sert uniquement de helper géométrique éventuel côté UI/tests.
        - Elle ne modifie aucun state.

        Contrat:
        - retourne (R_abs, T_abs) tel que:
            world = (R_abs @ (M? @ p_local)) + T_abs
            avec mirrored supposé False (si mirrored est nécessaire, c'est à l'UI de le gérer).

        mapping:
        - "direct"  : start_m -> start_t  et end_m -> end_t
        - "reverse" : start_m -> end_t    et end_m -> start_t
        """
        # On ignore group_id_* (compat API), tout est porté par les poses d'éléments.
        e_m = self.get_edge(str(element_id_m), int(edge_index_m))
        e_t = self.get_edge(str(element_id_t), int(edge_index_t))

        el_m = self.elements[str(element_id_m)]
        el_t = self.elements[str(element_id_t)]

        def _v_local(el: TopologyElement, vidx: int) -> np.ndarray:
            xy = el.vertex_local_xy.get(int(vidx))
            if xy is None:
                raise ValueError(
                    f"compute_pose_edge_to_edge: vertex_local_xy manquant (element={el.element_id} idx={vidx})"
                )
            return np.array([float(xy[0]), float(xy[1])], dtype=float)

        # Endpoints LOCAL (mobile)
        p0m = _v_local(el_m, e_m.v_start.vertex_index)
        p1m = _v_local(el_m, e_m.v_end.vertex_index)

        # Endpoints WORLD (cible) via pose élément
        p0t = _v_local(el_t, e_t.v_start.vertex_index)
        p1t = _v_local(el_t, e_t.v_end.vertex_index)
        w0t = self.elementLocalToWorld(str(element_id_t), p0t)
        w1t = self.elementLocalToWorld(str(element_id_t), p1t)

        if str(mapping).lower() == "direct":
            W0, W1 = w0t, w1t
        elif str(mapping).lower() == "reverse":
            W0, W1 = w1t, w0t
        else:
            raise ValueError(f"compute_pose_edge_to_edge: mapping invalide: {mapping}")

        vL = (p1m - p0m)
        vW = (W1 - W0)

        nL = float(np.linalg.norm(vL))
        nW = float(np.linalg.norm(vW))
        if nL <= 1e-12 or nW <= 1e-12:
            raise ValueError("compute_pose_edge_to_edge: arête dégénérée (norme ~0)")

        angL = float(np.arctan2(vL[1], vL[0]))
        angW = float(np.arctan2(vW[1], vW[0]))
        dtheta = angW - angL

        c = float(np.cos(dtheta)); s = float(np.sin(dtheta))
        R_abs = np.array([[c, -s],
                        [s,  c]], dtype=float)

        # T : amener p0m sur W0
        T_abs = np.array(W0, dtype=float) - (R_abs @ np.array(p0m, dtype=float))
        return (R_abs, T_abs)

    def apply_attachments(self, attachments: list[TopologyAttachment]) -> str:
        """Applique une liste d'attachments (transaction simple V1) et retourne le gid canonique final."""
        gid_last: str | None = None
        for att in list(attachments or []):
            gid_last = self.apply_attachment(att)
        return str(gid_last) if gid_last is not None else ""


    def _edge_endpoints_atomic_nodes(self, edge: TopologyEdge) -> tuple[str, str]:
        return (edge.v_start.node_id, edge.v_end.node_id)

    # --- points on edge ---
    def add_or_fuse_point_on_edge(self, edge_ref: TopologyFeatureRef, t_new: float, node_id_new: str) -> tuple[str, float]:
        edge = self.get_edge(edge_ref.element_id, edge_ref.index)
        idx = edge.find_fusion_candidate_index(
            t_new=float(t_new),
            fusion_distance_km=self.fusion_distance_km,
            node_rank_fn=lambda nid: self._node_rank_key(self.find_node(nid))
        )
        if idx is None:
            edge.insert_point_sorted(float(t_new), str(node_id_new))
            self.invalidateConceptCache(self.element_to_group.get(str(edge_ref.element_id)))
            return (self.find_node(str(node_id_new)), float(t_new))

        existing = edge.points_on_edge[idx]
        canon = self.union_nodes(existing.node_id, str(node_id_new))
        t_keep = float(existing.t)  # garder t du point canonique choisi
        existing.node_id = canon
        edge.points_on_edge.sort(key=lambda p: p.t)
        self.invalidateConceptCache(self.element_to_group.get(str(edge_ref.element_id)))
        return (canon, t_keep)

    # --- attachments ---
    def record_attachment(self, attachment: TopologyAttachment, group_id: str | None = None):
        self.attachments[attachment.attachment_id] = attachment
        if group_id is not None:
            self.groups[self.find_group(group_id)].attachment_ids.append(attachment.attachment_id)

    def apply_attachment(self, attachment: TopologyAttachment) -> str:
        gA = self.get_group_of_element(attachment.feature_a.element_id)
        gB = self.get_group_of_element(attachment.feature_b.element_id)
        gC = self.union_groups(gA, gB) if gA != gB else gA

        kind = str(attachment.kind)
        if kind == "vertex-vertex":
            vA = self.get_vertex(attachment.feature_a.element_id, attachment.feature_a.index)
            vB = self.get_vertex(attachment.feature_b.element_id, attachment.feature_b.index)
            self.union_nodes(vA.node_id, vB.node_id)

        elif kind == "vertex-edge":
            if attachment.feature_a.feature_type == TopologyFeatureType.VERTEX and attachment.feature_b.feature_type == TopologyFeatureType.EDGE:
                vRef, eRef = attachment.feature_a, attachment.feature_b
            elif attachment.feature_a.feature_type == TopologyFeatureType.EDGE and attachment.feature_b.feature_type == TopologyFeatureType.VERTEX:
                vRef, eRef = attachment.feature_b, attachment.feature_a
            else:
                raise ValueError("vertex-edge attend un vertexRef et un edgeRef")

            t = attachment.params.get("t", None)
            if t is None:
                t = vRef.t if vRef.t is not None else eRef.t
            if t is None:
                raise ValueError("vertex-edge: paramètre t manquant")

            v = self.get_vertex(vRef.element_id, vRef.index)
            canon_node, _ = self.add_or_fuse_point_on_edge(eRef, float(t), v.node_id)
            self.union_nodes(v.node_id, canon_node)

        elif kind == "edge-edge":
            eA = self.get_edge(attachment.feature_a.element_id, attachment.feature_a.index)
            eB = self.get_edge(attachment.feature_b.element_id, attachment.feature_b.index)

            mapping = str(attachment.params.get("mapping", "direct"))
            a0, a1 = self._edge_endpoints_atomic_nodes(eA)
            b0, b1 = self._edge_endpoints_atomic_nodes(eB)

            if mapping == "direct":
                self.union_nodes(a0, b0); self.union_nodes(a1, b1)
            elif mapping == "reverse":
                self.union_nodes(a0, b1); self.union_nodes(a1, b0)
            else:
                raise ValueError(f"edge-edge: mapping invalide: {mapping}")

            eA.coverages.append(TopologyCoverageInterval(0.0, 1.0))
            eB.coverages.append(TopologyCoverageInterval(0.0, 1.0))

        else:
            raise ValueError(f"Attachment kind non supporté: {kind}")

        self.record_attachment(attachment, group_id=gC)
        return gC

    def rebuild_from_attachments(self, attachments: list[TopologyAttachment]):
        self.attachments = {}
        for g in self.groups.values():
            g.attachment_ids = []

        for nid in list(self._node_parent.keys()):
            self._node_parent[nid] = nid
            self._node_members[nid] = [nid]

        for gid in list(self._group_parent.keys()):
            self._group_parent[gid] = gid
            self._group_members[gid] = [gid]

        for el in self.elements.values():
            for edge in el.edges:
                edge.points_on_edge = [
                    TopologyEdgePoint(0.0, edge.v_start.node_id),
                    TopologyEdgePoint(1.0, edge.v_end.node_id),
                ]
                edge.coverages = []

        for att in sorted(attachments, key=lambda a: a.attachment_id):
            self.apply_attachment(att)

    def removeElementsAndRebuild(self, element_ids: list[str]) -> None:
        """Supprime des éléments du world (et tout ce qui les référence), puis rebuild depuis les attachments restants.

        - Purge les attachments dont featureA/featureB référence un élément supprimé.
        - Reconstruit un world propre avec les éléments restants.
        - Rejoue les attachments conservés (IDs conservés).
        """
        removed = {str(eid) for eid in (element_ids or []) if str(eid).strip()}
        if not removed:
            return

        # 1) éléments conservés (snapshot complet)
        kept_elements: list[TopologyElement] = []
        kept_poses: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
        for eid, el in list(self.elements.items()):
            if str(eid) in removed:
                continue
            kept_elements.append(el)
            kept_poses[str(eid)] = el.get_pose()

        # 2) attachments conservés
        kept_atts: list[TopologyAttachment] = []
        for att in self.attachments.values():
            a = str(att.feature_a.element_id)
            b = str(att.feature_b.element_id)
            if a in removed or b in removed:
                continue
            kept_atts.append(att)

        # 3) rebuild dans un nouveau world (propre)
        neww = TopologyWorld(self.scenario_id)
        neww.fusion_distance_km = float(getattr(self, "fusion_distance_km", 1.0))

        # éviter collisions si on recrée des attachments plus tard
        # (on se cale sur le max Axxx existant)
        max_att_num = 0
        for att in kept_atts:
            m = re.search(r":A(\d+)$", str(att.attachment_id))
            if m:
                max_att_num = max(max_att_num, int(m.group(1)))
        neww._created_counter_attachments = max_att_num

        # re-créer les éléments (ne pas réutiliser les instances : on veut un état clean)
        for old in kept_elements:
            eid = str(old.element_id)
            R, T, mirrored = kept_poses.get(eid, (np.eye(2), np.zeros(2), False))

            clone = TopologyElement(
                element_id=eid,
                name=str(old.name),
                vertex_labels=list(old.vertex_labels),
                vertex_types=list(old.vertex_types),
                edge_lengths_km=list(old.edge_lengths_km),
                intrinsic_sides_km=dict(getattr(old, "intrinsic_sides_km", {}) or {}),
                local_frame=dict(getattr(old, "local_frame", {}) or {}),
                vertex_local_xy=dict(getattr(old, "vertex_local_xy", {}) or {}),
                meta=dict(getattr(old, "meta", {}) or {}),
            )
            neww.add_element_as_new_group(clone)
            neww.setElementPose(eid, R=R, T=T, mirrored=mirrored)

        # rejouer les attachments conservés (IDs conservés)
        for att in sorted(kept_atts, key=lambda a: str(a.attachment_id)):
            neww.apply_attachment(att)

        # on reconstruit les références aux groupes (sur le nouveau world)
        neww.rebuildGroupElementLists()
        neww.invalidateConceptCache(None)

        # 4) swap in-place (les refs Tk vers scen.topoWorld restent valides)
        self.__dict__.clear()
        self.__dict__.update(neww.__dict__)

    # ------------------------------------------------------------------
    # Helpers "Tooltip" (Core)
    # ------------------------------------------------------------------
    def _parseElementAndVertexIndexFromNodeId(self, node_id: str) -> tuple[str, int] | None:
        """Parse un node_id atomique '<scenario>:Txx:Nn' -> (element_id, vertex_index)."""
        m = re.search(r"^(.*):N(\d+)$", str(node_id))
        if not m:
            return None
        return (str(m.group(1)), int(m.group(2)))

    def getAtomicNodeLabel(self, node_id: str) -> str:
        """Retourne le label *métier* (ville) d'un node atomique.
        Fallback: 'N0', 'N1', ...
        """
        parsed = self._parseElementAndVertexIndexFromNodeId(str(node_id))
        if parsed is None:
            return str(node_id).split(":")[-1]
        element_id, vidx = parsed
        el = self.elements.get(element_id)
        if el is None:
            return f"N{vidx}"
        # v4.3 : labels portés par TopologyVertex / vertex_labels
        if hasattr(el, "vertexes") and el.vertexes and 0 <= vidx < len(el.vertexes):
            return str(el.vertexes[vidx].label)
        if hasattr(el, "vertex_labels") and 0 <= vidx < len(el.vertex_labels):
            return str(el.vertex_labels[vidx])
        return f"N{vidx}"

    def getNodeLabel(self, node_id: str) -> str:
        """Label métier d'un node (canonique ou atomique).
        - si node_id est canonique (DSU), agrège les labels de ses membres.
        """
        nid = str(node_id)
        if nid not in self._node_parent:
            return self.getAtomicNodeLabel(nid)
        c = self.find_node(nid)
        uniq = sorted({self.getAtomicNodeLabel(m) for m in self.node_members(c)})
        if len(uniq) == 0:
            return self.getAtomicNodeLabel(nid)
        if len(uniq) == 1:
            return uniq[0]
        return "|".join([str(x) for x in uniq])

    def _parseTriangleIdFromNodeId(self, node_id: str) -> str:
        """Retourne 'Txx' à partir d'un node_id atomique '<scenario>:Txx:Nn'."""
        m = re.search(r":T(\d+):N\d+$", str(node_id))
        if m:
            return f"T{int(m.group(1)):02d}"
        return "T??"

    def getNodeDisplayTriplet(self, node_id: str) -> tuple[str, str, str]:
        """Triplet (type, triangleId, label) pour debug/tooltip.
        - Pour un node canonique, triangleId peut être ambigu -> 'T*' si >1.
        """
        nid = str(node_id)
        typ = str(self.node_type(nid))
        if nid in self._node_parent:
            c = self.find_node(nid)
            tris = sorted({self._parseTriangleIdFromNodeId(m) for m in self.node_members(c)})
            tri = tris[0] if len(tris) == 1 else "T*"
            return (typ, tri, self.getNodeLabel(c))
        return (typ, self._parseTriangleIdFromNodeId(nid), self.getAtomicNodeLabel(nid))

    def getConnectedPointsTriplets(self, node_id: str) -> list[tuple[str, str, str]]:
        """Liste (type, triangleId, nodeLabel) pour tous les points connectés via ATTACHMENTS.

        IMPORTANT: nodeLabel = label métier (ville), pas 'N0/N1/N2'.
        """
        c = self.find_node(str(node_id))
        out: list[tuple[str, str, str]] = []
        for nid in sorted(self.node_members(c)):
            typ = str(self.node_type(str(nid)))
            tri_id = self._parseTriangleIdFromNodeId(str(nid))
            lab = self.getAtomicNodeLabel(str(nid))
            out.append((typ, tri_id, lab))
        return out

    def _localPointOnEdge(self, el: "TopologyElement", edge: "TopologyEdge", t: float) -> np.ndarray:
        """Point local sur une arête (interpolation linéaire) via vertex_local_xy."""
        v0 = int(edge.v_start.vertex_index)
        v1 = int(edge.v_end.vertex_index)
        p0 = el.vertex_local_xy.get(v0) if hasattr(el, "vertex_local_xy") else None
        p1 = el.vertex_local_xy.get(v1) if hasattr(el, "vertex_local_xy") else None
        if p0 is None or p1 is None:
            raise ValueError(
                f"_localPointOnEdge: vertex_local_xy manquant (element={el.element_id} v0={v0} v1={v1})"
            )
        t = float(t)
        x = (1.0 - t) * float(p0[0]) + t * float(p1[0])
        y = (1.0 - t) * float(p0[1]) + t * float(p1[1])
        return np.array([x, y], dtype=float)

    def getNodeHalfRaysAzimuts(self, node_id: str) -> list[dict]:
        """Demi-droites incidentes à un node topo + azimut.

        Ajoute aussi: fromLabel/toLabel (labels métier) pour debug.
        """
        c = self.find_node(str(node_id))
        rays: list[dict] = []

        for el in self.elements.values():
            if not hasattr(el, "edges"):
                continue
            for edge in el.edges:
                pts = []
                for p in getattr(edge, "points_on_edge", []):
                    tn = float(p.t)
                    cn = self.find_node(str(p.node_id))
                    pts.append((tn, cn))
                pts.sort(key=lambda x: x[0])

                for (t0, n0), (t1, n1) in zip(pts[:-1], pts[1:]):
                    if n0 == n1:
                        continue
                    p0_local = self._localPointOnEdge(el, edge, t0)
                    p1_local = self._localPointOnEdge(el, edge, t1)
                    p0_world = el.localToWorld(p0_local)
                    p1_world = el.localToWorld(p1_local)

                    if n0 == c:
                        dx = float(p1_world[0] - p0_world[0])
                        dy = float(p1_world[1] - p0_world[1])
                        rays.append({
                            "elementId": str(el.element_id),
                            "edgeId": str(edge.edge_id()),
                            "fromNode": str(n0),
                            "toNode": str(n1),
                            "fromLabel": self.getNodeLabel(n0),
                            "toLabel": self.getNodeLabel(n1),
                            "tFrom": float(t0),
                            "tTo": float(t1),
                            "dx": dx,
                            "dy": dy,
                            "azimutDeg": self.azimutDegFromDxDy(dx, dy),
                        })
                    if n1 == c:
                        dx = float(p0_world[0] - p1_world[0])
                        dy = float(p0_world[1] - p1_world[1])
                        rays.append({
                            "elementId": str(el.element_id),
                            "edgeId": str(edge.edge_id()),
                            "fromNode": str(n1),
                            "toNode": str(n0),
                            "fromLabel": self.getNodeLabel(n1),
                            "toLabel": self.getNodeLabel(n0),
                            "tFrom": float(t1),
                            "tTo": float(t0),
                            "dx": dx,
                            "dy": dy,
                            "azimutDeg": self.azimutDegFromDxDy(dx, dy),
                        })

        seen = set()
        uniq: list[dict] = []
        for r in rays:
            key = (r["edgeId"], round(float(r["tFrom"]), 9), round(float(r["tTo"]), 9), r["toNode"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        uniq.sort(key=lambda r: (float(r.get("azimutDeg", 0.0)), str(r.get("edgeId", ""))))
        return uniq


    # --- export ---
    def validate_world(self) -> list[str]:
        errors: list[str] = []
        for el in self.elements.values():
            for edge in el.edges:
                last_t = -1.0
                for p in sorted(edge.points_on_edge, key=lambda p: p.t):
                    if not (0.0 <= p.t <= 1.0):
                        errors.append(f"Edge {edge.edge_id()}: t hors bornes: {p.t}")
                    if p.t < last_t:
                        errors.append(f"Edge {edge.edge_id()}: pointsOnEdge non triés")
                    last_t = p.t
                for c in edge.coverages:
                    if c.t0 >= c.t1:
                        errors.append(f"Edge {edge.edge_id()}: coverage dégénéré [{c.t0},{c.t1}]")
        return errors

    def export_topo_dump_xml(self, scenario_id: str, path: str, orientation: str = "cw") -> str:
        root = _ET.Element("TopoDump", {
            "version": "4.3",
            "units": "km",
            "tFrame": "world",
            "scenarioId": str(scenario_id),
        })

        groups_el = _ET.SubElement(root, "Groups")
        for gid in sorted(self.groups.keys()):
            if gid != self.find_group(gid):
                continue
            g = self.groups[gid]
            grp = _ET.SubElement(groups_el, "Group", {
                "id": gid,
                "elements": str(len(g.element_ids)),
                "attachments": str(len(g.attachment_ids)),
                "originGroups": ",".join(sorted(self.group_members(gid))),
            })
            errs = self.validate_world()
            _ET.SubElement(grp, "Validate", {"errors": str(len(errs)), "warnings": "0"})

        nodes_el = _ET.SubElement(root, "Nodes")
        canonical_nodes = sorted({self.find_node(nid) for nid in self._node_parent.keys()})
        for cn in canonical_nodes:
            _ET.SubElement(nodes_el, "Node", {
                "id": cn,
                "type": self.node_type(cn),
                "origins": ",".join(sorted(self.node_members(cn))),
                "createdOrder": str(self._node_created_order.get(self.find_node(cn), 0)),
            })

        # --- ConceptModels (debug) : un par GROUPE ---
        # DSU nodes = global, mais on exporte un sous-graphe conceptuel par groupe canonique,
        # indispensable pour calculer/valider un contour de groupe.
        cm_all = _ET.SubElement(root, "ConceptModels")
        for gid in sorted(self.groups.keys()):
            if gid != self.find_group(gid):
                continue
            self.buildConceptCacheIfNeeded(gid)
            ccache = self._concept_cache(gid)

            cm = _ET.SubElement(cm_all, "ConceptModel", {"version": "1.0", "group": str(gid)})
            cns = _ET.SubElement(cm, "ConceptNodes")
            for cid in sorted(ccache.nodes.keys()):
                xy = ccache.nodes[cid].world_xy()
                cn_el = _ET.SubElement(cns, "ConceptNode", {
                    "id": str(cid),
                    "type": str(self.getConceptNodeType(cid)),
                    "x": f"{float(xy[0]):.6f}",
                    "y": f"{float(xy[1]):.6f}",
                })
                mem_el = _ET.SubElement(cn_el, "Members")
                for mid in sorted(self.node_members(cid)):
                    _ET.SubElement(mem_el, "M", {
                        "id": str(mid),
                        "type": str(self.node_type(mid)),
                        "tri": str(self._parseTriangleIdFromNodeId(str(mid))),
                        "label": str(self.getAtomicNodeLabel(str(mid))),
                    })

            ces = _ET.SubElement(cm, "ConceptEdges")
            for (a, b) in sorted(ccache.edges.keys()):
                p0 = ccache.nodes[a].world_xy() if a in ccache.nodes else (0.0, 0.0)
                p1 = ccache.nodes[b].world_xy() if b in ccache.nodes else (0.0, 0.0)
                dx = float(p1[0] - p0[0])
                dy = float(p1[1] - p0[1])
                dist = float(math.hypot(dx, dy))
                az = self.azimutDegFromDxDy(dx, dy)
                ce_el = _ET.SubElement(ces, "ConceptEdge", {
                    "a": str(a),
                    "b": str(b),
                    "distanceKm": f"{dist:.6f}",
                    "azimutDeg": f"{az:.6f}",
                })
                occ_el = _ET.SubElement(ce_el, "Occurrences")
                for occ in ccache.edges[(a, b)].occurrences:
                    _ET.SubElement(occ_el, "O", {
                        "element": str(occ.get("elementId", "")),
                        "edge": str(occ.get("edgeId", "")),
                        "t0": f"{float(occ.get('t0', 0.0)):.6f}",
                        "t1": f"{float(occ.get('t1', 0.0)):.6f}",
                    })


        elements_el = _ET.SubElement(root, "Elements")
        for eid in sorted(self.elements.keys()):
            el = self.elements[eid]
            # Lien explicite Element -> Groupe (dump-only, info-only)
            gid = self.get_group_of_element(el.element_id)
            elx = _ET.SubElement(elements_el, "Element", {
                "id": el.element_id,
                "name": el.name,
                "n": str(el.n_vertices()),
                "group": str(gid),
            })

            # --- Pose monde (par élément) ---
            # Fallback implicite: identité si non renseigné.
            R, T, mirrored = el.get_pose()
            ep = _ET.SubElement(elx, "ElementPose", {
                "mirrored": "1" if mirrored else "0",
                "tx": f"{float(T[0]):.6f}",
                "ty": f"{float(T[1]):.6f}",
            })
            _ET.SubElement(ep, "R", {
                "r00": f"{float(R[0,0]):.9f}", "r01": f"{float(R[0,1]):.9f}",
                "r10": f"{float(R[1,0]):.9f}", "r11": f"{float(R[1,1]):.9f}",
            })

            # --- Coordonnées locales (relatives) ---
            orient = str(el.meta.get("orient", "")).strip().upper()
            lc = _ET.SubElement(elx, "LocalCoords", {
                "frameOrigin": str(el.local_frame.get("origin", TopologyNodeType.OUVERTURE)),
                "xAxis": str(el.local_frame.get("xAxis", "O->B")),
                "units": str(el.local_frame.get("units", "km")),
                "orient": orient if orient else "CCW",
            })
            for vidx in sorted(el.vertex_local_xy.keys()):
                x, y = el.vertex_local_xy[vidx]
                _ET.SubElement(lc, "P", {"idx": str(vidx), "x": f"{float(x):.6f}", "y": f"{float(y):.6f}"})

            vx = _ET.SubElement(elx, "Vertices")
            for v in el.vertexes:
                _ET.SubElement(vx, "V", {
                    "idx": str(v.vertex_index),
                    "label": v.label,
                    "node": self.find_node(v.node_id),
                    "nodeOrigin": v.node_id,
                    "type": v.node_type,
                })

            ex = _ET.SubElement(elx, "Edges")
            for e in el.edges:
                a, b = e.edge_label_key()
                edge_el = _ET.SubElement(ex, "Edge", {
                    "id": e.edge_id(),
                    "idx": str(e.edge_index),
                    "from": str(e.v_start.vertex_index),
                    "to": str(e.v_end.vertex_index),
                    "labels": e.labels_display(),
                    "edgeLabelKey": f"{a}|{b}",
                    "lengthKm": f"{e.edge_length_km:.3f}",
                })
                pts = _ET.SubElement(edge_el, "Points")
                for pnt in sorted(e.points_on_edge, key=lambda p: p.t):
                    _ET.SubElement(pts, "P", {
                        "t": f"{pnt.t:.6f}",
                        "node": self.find_node(pnt.node_id),
                        "nodeOrigin": pnt.node_id,
                    })
                covs = _ET.SubElement(edge_el, "Coverages")
                for c in sorted(e.coverages, key=lambda c: (c.t0, c.t1)):
                    _ET.SubElement(covs, "C", {"t0": f"{c.t0:.6f}", "t1": f"{c.t1:.6f}"})

        atts_el = _ET.SubElement(root, "Attachments")
        for aid in sorted(self.attachments.keys()):
            a = self.attachments[aid]
            att = _ET.SubElement(atts_el, "Attachment", {
                "id": a.attachment_id,
                "kind": a.kind,
                "source": a.source,
                "featureA": a.feature_a.to_string(),
                "featureB": a.feature_b.to_string(),
            })
            params_el = _ET.SubElement(att, "Params")
            for k in sorted(a.params.keys()):
                _ET.SubElement(params_el, "Param", {"name": str(k), "value": str(a.params[k])})

        boundary_el = _ET.SubElement(root, "Boundary", {"orientation": str(orientation)})
        _ET.SubElement(boundary_el, "Note").text = "Boundary non implémentée en V1 (IDs lisibles) (sera ajoutée en itération suivante)."

        _ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        return path
