"""assembleur_core.py
Noyau 'pur' (sans Tk) : géométrie + structures de base.
"""

import math
import datetime as _dt
from typing import Optional, List, Dict, Tuple

import numpy as np

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
# TopologyModel v4.3 (Core) – Modèle objet V1 (sans Tk)
# =============================================================================
#
# Objectifs de cette V1 (v4.3):
# - Introduire les classes du modèle (World/Group/Element/Vertex/Edge/Node/Attachment).
# - Implémenter la fusion non destructive (DSU) des nodes et des groupes.
# - Implémenter la politique de fusion des points sur une arête en km:
#       abs(t1 - t2) * edgeLengthKm <= fusionDistanceKm  => fusion
#   et conserver le t du point canonique (référence contour).
# - Implémenter l'export manuel TopoDump XML par scénario.
#
# Hors périmètre V1:
# - Calcul complet du boundary (frontière). Une API est posée, mais le calcul
#   sera itéré ensuite. La règle "mono-cycle sinon bug" est néanmoins posée
#   dans validate() via un placeholder.
#
# Notes:
# - Le paramètre t est toujours fourni côté UI et exprimé en référentiel monde.
# - Les unités monde sont les kilomètres.
#

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
    # Types de nodes (métier). Priorité: L > B > O.
    OUVERTURE = "O"
    BASE = "B"
    LUMIERE = "L"

    @staticmethod
    def rank(node_type: str) -> int:
        if node_type == TopologyNodeType.LUMIERE:
            return 3
        if node_type == TopologyNodeType.BASE:
            return 2
        return 1  # O par défaut


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
    # Référence typée vers un Vertex ou une Edge (et éventuellement un t).
    def __init__(self, feature_type: str, element_id: int, index: int, t: Optional[float] = None):
        self.feature_type = str(feature_type)
        self.element_id = int(element_id)
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
    # Trace métier d'une intention binaire.
    def __init__(self, attachment_id: int, kind: str,
                 feature_a: TopologyFeatureRef, feature_b: TopologyFeatureRef,
                 params: Optional[Dict[str, object]] = None, source: str = "manual"):
        self.attachment_id = int(attachment_id)
        self.kind = str(kind)  # "vertex-edge" | "vertex-vertex" | "edge-edge"
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
    def __init__(self, element_id: int, vertex_index: int, label: str, node_id: int, node_type: str):
        self.element_id = int(element_id)
        self.vertex_index = int(vertex_index)
        self.label = str(label)
        self.node_id = int(node_id)          # node atomique (origine)
        self.node_type = str(node_type)      # O/B/L (type métier du node initial)

    def vertex_id(self) -> Tuple[int, int]:
        return (self.element_id, self.vertex_index)


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
    def __init__(self, t: float, node_id: int):
        self.t = float(t)
        self.node_id = int(node_id)  # node atomique (origine)

    def to_tuple(self):
        return (self.t, self.node_id)


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
    # Arête paramétrique (Option B): points internes + intervalles de couverture.
    def __init__(self, element_id: int, edge_index: int, v_start: TopologyVertex, v_end: TopologyVertex,
                 edge_length_km: float):
        self.element_id = int(element_id)
        self.edge_index = int(edge_index)
        self.v_start = v_start
        self.v_end = v_end
        self.edge_length_km = float(edge_length_km)

        # pointsOnEdge triés par t: inclure implicitement les extrémités
        self.points_on_edge: List[TopologyEdgePoint] = [
            TopologyEdgePoint(0.0, v_start.node_id),
            TopologyEdgePoint(1.0, v_end.node_id),
        ]
        self.coverages: List[TopologyCoverageInterval] = []

    def edge_id(self) -> Tuple[int, int]:
        return (self.element_id, self.edge_index)

    def edge_label_key(self) -> Tuple[str, str]:
        a = self.v_start.label
        b = self.v_end.label
        return (a, b) if a <= b else (b, a)

    def labels_display(self) -> str:
        return f"{self.v_start.label}–{self.v_end.label}"

    def find_fusion_candidate_index(self, t_new: float, fusion_distance_km: float,
                                    node_rank_fn) -> Optional[int]:
        # Retourne l'index d'un point existant à fusionner, si deltaKm <= fusionDistanceKm.
        # En cas de multiples candidats, choisit le point dont le node canonique est prioritaire
        # (type puis plus récent) via node_rank_fn(canonicalNodeId)->(rankType, canonicalId).
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

    def insert_point_sorted(self, t_new: float, node_id: int):
        t_new = float(t_new)
        for p in self.points_on_edge:
            if p.t == t_new and p.node_id == int(node_id):
                return
        self.points_on_edge.append(TopologyEdgePoint(t_new, int(node_id)))
        self.points_on_edge.sort(key=lambda p: p.t)


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
    # Polygone (triangle = n=3). Les sommets sont ordonnés (cycle).
    def __init__(self, element_id: int, name: str, vertex_labels: List[str], vertex_types: List[str],
                 edge_lengths_km: Optional[List[float]] = None, meta: Optional[Dict] = None,
                 intrinsic_sides_km: Optional[Dict[str, float]] = None,
                 pose_R: Optional[np.ndarray] = None, pose_T: Optional[np.ndarray] = None):
        self.element_id = int(element_id)
        self.name = str(name)
        self.meta = dict(meta) if meta is not None else {}

        # --- v4.3: géométrie intrinsèque + pose monde (sans Tk) ---
        # intrinsic_sides_km (optionnel, triangle) : {"OB":..., "OL":..., "BL":...} en km.
        # Si fourni, peut servir à reconstruire un repère local (O=(0,0), B=(OB,0), L=(x,y)).
        self.intrinsic_sides_km: Dict[str, float] = dict(intrinsic_sides_km) if intrinsic_sides_km is not None else {}

        # Pose monde (rotation + translation) appliquée au repère local pour obtenir des coordonnées monde.
        # Ces paramètres sont optionnels et servent uniquement à dériver des coordonnées monde si besoin.
        self.pose_R = np.array(pose_R, float) if pose_R is not None else None
        self.pose_T = np.array(pose_T, float) if pose_T is not None else None

        if len(vertex_labels) < 3:
            raise ValueError("TopologyElement: un polygone doit avoir au moins 3 sommets")

        if len(vertex_types) != len(vertex_labels):
            raise ValueError("TopologyElement: vertex_types doit avoir la même taille que vertex_labels")

        self.vertex_labels = [str(x) for x in vertex_labels]
        self.vertex_types = [str(x) for x in vertex_types]

        n = len(vertex_labels)
        if edge_lengths_km is None:
            # V1: longueur inconnue => 1km par défaut (sera remplacé par la géométrie)
            self.edge_lengths_km = [1.0] * n
        else:
            if len(edge_lengths_km) != n:
                raise ValueError("TopologyElement: edge_lengths_km doit avoir la même taille que les sommets")
            self.edge_lengths_km = [float(x) for x in edge_lengths_km]

        # Remplies par le World lors de l'ajout (création des nodes atomiques)
        self.vertexes: List[TopologyVertex] = []
        self.edges: List[TopologyEdge] = []

    def n_vertices(self) -> int:
        return len(self.vertex_labels)

    def build_local_points_triangle(self) -> Optional[Dict[str, np.ndarray]]:
        """Construit un repère local O/B/L à partir des longueurs intrinsèques (triangle uniquement).

        Retourne un dict {"O":(x,y), "B":(x,y), "L":(x,y)} en km, ou None si non applicable.
        """
        if self.n_vertices() != 3:
            return None
        if not self.intrinsic_sides_km:
            return None
        OB = float(self.intrinsic_sides_km.get("OB", 0.0))
        OL = float(self.intrinsic_sides_km.get("OL", 0.0))
        BL = float(self.intrinsic_sides_km.get("BL", 0.0))
        if OB <= 0 or OL <= 0 or BL <= 0:
            return None
        return _build_local_triangle(OB=OB, OL=OL, BL=BL)

    def build_world_points_triangle(self) -> Optional[Dict[str, np.ndarray]]:
        """Dérive des coordonnées monde (km) depuis le repère local et la pose monde.

        - Nécessite build_local_points_triangle() et (pose_R, pose_T).
        - Ne dépend pas de Tk.
        """
        local_pts = self.build_local_points_triangle()
        if local_pts is None:
            return None
        if self.pose_R is None or self.pose_T is None:
            return None
        R = np.array(self.pose_R, float)
        T = np.array(self.pose_T, float)
        out = {}
        for k in ("O", "B", "L"):
            out[k] = (R @ np.array(local_pts[k], float)) + T
        return out


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
    def __init__(self, group_id: int):
        self.group_id = int(group_id)
        self.element_ids: List[int] = []
        self.attachment_ids: List[int] = []


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
    # Racine métier: détient les éléments, groups, nodes et DSU (nodes & groupes).
    def __init__(self):
        # Paramètres métier globaux (v4.2)
        self.fusion_distance_km: float = 1.0

        # Indices
        self.groups: Dict[int, TopologyGroup] = {}
        self.elements: Dict[int, TopologyElement] = {}
        self.element_to_group: Dict[int, int] = {}  # elementId -> groupId atomique
        self.attachments: Dict[int, TopologyAttachment] = {}

        # DSU nodes (non destructif)
        self._node_parent: Dict[int, int] = {}
        self._node_type: Dict[int, str] = {}
        self._node_members: Dict[int, List[int]] = {}

        # DSU groups (non destructif)
        self._group_parent: Dict[int, int] = {}
        self._group_members: Dict[int, List[int]] = {}

        # Id factory monotone (plus grand = plus récent)
        self._next_group_id: int = 1
        self._next_node_id: int = 1
        self._next_attachment_id: int = 1

    # --- Id generation ---
    def new_group_id(self) -> int:
        gid = self._next_group_id
        self._next_group_id += 1
        return gid

    def new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def new_attachment_id(self) -> int:
        aid = self._next_attachment_id
        self._next_attachment_id += 1
        return aid

    # --- DSU nodes ---
    def _node_rank_key(self, node_id: int) -> Tuple[int, int]:
        n0 = int(node_id)
        c = self.find_node(n0)
        t = self._node_type.get(c, TopologyNodeType.OUVERTURE)
        return (TopologyNodeType.rank(t), c)

    def create_node_atomic(self, node_type: str) -> int:
        nid = self.new_node_id()
        self._node_parent[nid] = nid
        self._node_type[nid] = str(node_type)
        self._node_members[nid] = [nid]
        return nid

    def find_node(self, node_id: int) -> int:
        node_id = int(node_id)
        parent = self._node_parent.get(node_id, node_id)
        if parent != node_id:
            self._node_parent[node_id] = self.find_node(parent)
        return self._node_parent.get(node_id, node_id)

    def union_nodes(self, a: int, b: int) -> int:
        ra = self.find_node(int(a))
        rb = self.find_node(int(b))
        if ra == rb:
            return ra

        key_a = self._node_rank_key(ra)
        key_b = self._node_rank_key(rb)
        canonical = ra if key_a >= key_b else rb
        other = rb if canonical == ra else ra

        self._node_parent[other] = canonical

        mem = self._node_members.get(canonical, [canonical])
        mem_other = self._node_members.get(other, [other])
        merged = mem + [x for x in mem_other if x not in mem]
        self._node_members[canonical] = merged

        return canonical

    def node_members(self, node_id: int) -> List[int]:
        c = self.find_node(int(node_id))
        return list(self._node_members.get(c, [c]))

    def node_type(self, node_id: int) -> str:
        c = self.find_node(int(node_id))
        return self._node_type.get(c, TopologyNodeType.OUVERTURE)

    # --- DSU groups ---
    def create_group_atomic(self) -> int:
        gid = self.new_group_id()
        self._group_parent[gid] = gid
        self._group_members[gid] = [gid]
        self.groups[gid] = TopologyGroup(gid)
        return gid

    def find_group(self, group_id: int) -> int:
        group_id = int(group_id)
        parent = self._group_parent.get(group_id, group_id)
        if parent != group_id:
            self._group_parent[group_id] = self.find_group(parent)
        return self._group_parent.get(group_id, group_id)

    def union_groups(self, a: int, b: int) -> int:
        ra = self.find_group(int(a))
        rb = self.find_group(int(b))
        if ra == rb:
            return ra

        canonical = ra if ra >= rb else rb  # plus récent (max)
        other = rb if canonical == ra else ra

        self._group_parent[other] = canonical

        mem = self._group_members.get(canonical, [canonical])
        mem_other = self._group_members.get(other, [other])
        merged = mem + [x for x in mem_other if x not in mem]
        self._group_members[canonical] = merged

        if canonical in self.groups and other in self.groups:
            self.groups[canonical].element_ids.extend(self.groups[other].element_ids)
            self.groups[canonical].attachment_ids.extend(self.groups[other].attachment_ids)

        return canonical

    def group_members(self, group_id: int) -> List[int]:
        c = self.find_group(int(group_id))
        return list(self._group_members.get(c, [c]))

    # --- Elements / construction ---
    def add_element_as_new_group(self, element: TopologyElement) -> int:
        if element.element_id in self.elements:
            raise ValueError(f"TopologyWorld: elementId déjà présent: {element.element_id}")

        gid = self.create_group_atomic()
        group = self.groups[gid]

        self.elements[element.element_id] = element
        self.element_to_group[element.element_id] = gid
        group.element_ids.append(element.element_id)

        element.vertexes = []
        for i, (lab, typ) in enumerate(zip(element.vertex_labels, element.vertex_types)):
            nid = self.create_node_atomic(typ)
            element.vertexes.append(TopologyVertex(element.element_id, i, lab, nid, typ))

        n = element.n_vertices()
        element.edges = []
        for i in range(n):
            v0 = element.vertexes[i]
            v1 = element.vertexes[(i + 1) % n]
            elen = element.edge_lengths_km[i]
            element.edges.append(TopologyEdge(element.element_id, i, v0, v1, elen))

        return gid

    def get_edge(self, element_id: int, edge_index: int) -> TopologyEdge:
        return self.elements[int(element_id)].edges[int(edge_index)]

    def get_vertex(self, element_id: int, vertex_index: int) -> TopologyVertex:
        return self.elements[int(element_id)].vertexes[int(vertex_index)]

    def get_group_of_element(self, element_id: int) -> int:
        """Retourne le groupId canonique auquel appartient un élément."""
        gid0 = self.element_to_group.get(int(element_id))
        if gid0 is None:
            raise ValueError(f"Element sans groupe: {element_id}")
        return self.find_group(int(gid0))

    def _edge_endpoints_atomic_nodes(self, edge: TopologyEdge) -> Tuple[int, int]:
        """Retourne (nodeStartAtomic, nodeEndAtomic) pour une arête orientée."""
        return (int(edge.v_start.node_id), int(edge.v_end.node_id))

        # --- PointsOnEdge insertion with fusion policy (km) ---
        def add_or_fuse_point_on_edge(self, edge_ref: TopologyFeatureRef, t_new: float, node_id_new: int) -> Tuple[int, float]:
            if edge_ref.feature_type != TopologyFeatureType.EDGE:
                raise ValueError("add_or_fuse_point_on_edge: edgeRef attendu")

            edge = self.get_edge(edge_ref.element_id, edge_ref.index)
            t_new = float(t_new)

            idx = edge.find_fusion_candidate_index(
                t_new=t_new,
                fusion_distance_km=self.fusion_distance_km,
                node_rank_fn=lambda nid: self._node_rank_key(self.find_node(nid))
            )

            if idx is None:
                edge.insert_point_sorted(t_new, int(node_id_new))
                return (self.find_node(int(node_id_new)), t_new)

            existing = edge.points_on_edge[idx]
            canon = self.union_nodes(existing.node_id, int(node_id_new))

            # garder le t du point canonique (le point existant choisi)
            t_keep = float(existing.t)
            existing.node_id = canon
            edge.points_on_edge.sort(key=lambda p: p.t)

            return (canon, t_keep)

    # --- Attachments : application + stockage (v4.3) ---
    def apply_attachment(self, attachment: TopologyAttachment) -> int:
        """Applique UNE attache au modèle (DSU + pointsOnEdge + coverages).

        Retourne le groupId canonique résultant.
        """
        kind = str(attachment.kind)

        # Déterminer les groupes impliqués (canonique) à partir des éléments référencés
        gA = self.get_group_of_element(attachment.feature_a.element_id)
        gB = self.get_group_of_element(attachment.feature_b.element_id)
        gC = self.union_groups(gA, gB) if gA != gB else gA

        if kind == "vertex-vertex":
            if attachment.feature_a.feature_type != TopologyFeatureType.VERTEX or attachment.feature_b.feature_type != TopologyFeatureType.VERTEX:
                raise ValueError("vertex-vertex attend deux vertexRef")
            vA = self.get_vertex(attachment.feature_a.element_id, attachment.feature_a.index)
            vB = self.get_vertex(attachment.feature_b.element_id, attachment.feature_b.index)
            self.union_nodes(vA.node_id, vB.node_id)

        elif kind == "vertex-edge":
            # Le paramètre t est fourni par l’UI (référentiel monde). On le prend dans params["t"] ou featureRef.t
            if attachment.feature_a.feature_type == TopologyFeatureType.VERTEX and attachment.feature_b.feature_type == TopologyFeatureType.EDGE:
                vRef = attachment.feature_a
                eRef = attachment.feature_b
            elif attachment.feature_a.feature_type == TopologyFeatureType.EDGE and attachment.feature_b.feature_type == TopologyFeatureType.VERTEX:
                vRef = attachment.feature_b
                eRef = attachment.feature_a
            else:
                raise ValueError("vertex-edge attend un vertexRef et un edgeRef")

            t = attachment.params.get("t", None)
            if t is None:
                t = vRef.t if vRef.t is not None else eRef.t
            if t is None:
                raise ValueError("vertex-edge: paramètre t manquant (UI doit fournir t)")

            v = self.get_vertex(vRef.element_id, vRef.index)
            # Ajout/fusion du point sur l’arête (politique km). Puis union du node du vertex avec celui du point.
            canon_node, _t_keep = self.add_or_fuse_point_on_edge(edge_ref=eRef, t_new=float(t), node_id_new=v.node_id)
            self.union_nodes(v.node_id, canon_node)

        elif kind == "edge-edge":
            # mapping : "direct" | "reverse"
            if attachment.feature_a.feature_type != TopologyFeatureType.EDGE or attachment.feature_b.feature_type != TopologyFeatureType.EDGE:
                raise ValueError("edge-edge attend deux edgeRef")
            eA = self.get_edge(attachment.feature_a.element_id, attachment.feature_a.index)
            eB = self.get_edge(attachment.feature_b.element_id, attachment.feature_b.index)

            mapping = str(attachment.params.get("mapping", "direct"))
            a0, a1 = self._edge_endpoints_atomic_nodes(eA)
            b0, b1 = self._edge_endpoints_atomic_nodes(eB)

            if mapping == "direct":
                self.union_nodes(a0, b0)
                self.union_nodes(a1, b1)
            elif mapping == "reverse":
                self.union_nodes(a0, b1)
                self.union_nodes(a1, b0)
            else:
                raise ValueError(f"edge-edge: mapping invalide: {mapping} (attendu 'direct'|'reverse')")

            # Coverage total sur les deux arêtes : elles deviennent internes (exclusion du boundary)
            eA.coverages.append(TopologyCoverageInterval(0.0, 1.0))
            eB.coverages.append(TopologyCoverageInterval(0.0, 1.0))

        else:
            raise ValueError(f"Attachment kind non supporté: {kind}")

        # Stocker l'attache dans le monde et l'associer au groupe canonique
        self.record_attachment(attachment, group_id=gC)
        return gC

    def rebuild_from_attachments(self, attachments: List[TopologyAttachment]):
        """Reconstruit la topologie depuis une liste d’attaches (stratégie MVP).

        Note: dans cette V1, on suppose que les éléments ont déjà été ajoutés au world.
        La reconstruction réinitialise uniquement l'état topo mutable (DSU, pointsOnEdge, coverages, attachments).
        """
        # Reset DSU nodes/groups + attachments + edge points/coverages
        self.attachments = {}
        for g in self.groups.values():
            g.attachment_ids = []

        # Reset DSU nodes (chaque node atomique redevient canonique) et membres
        for nid in list(self._node_parent.keys()):
            self._node_parent[nid] = nid
            self._node_members[nid] = [nid]

        # Reset DSU groups
        for gid in list(self._group_parent.keys()):
            self._group_parent[gid] = gid
            self._group_members[gid] = [gid]

        # Reset edges pointsOnEdge/coverages
        for el in self.elements.values():
            for edge in el.edges:
                edge.points_on_edge = [
                    TopologyEdgePoint(0.0, edge.v_start.node_id),
                    TopologyEdgePoint(1.0, edge.v_end.node_id),
                ]
                edge.coverages = []

        # Re-apply
        for att in sorted(attachments, key=lambda a: a.attachment_id):
            self.apply_attachment(att)

    # --- Attachments (V1: stockage) ---
    def record_attachment(self, attachment: TopologyAttachment, group_id: Optional[int] = None):
        self.attachments[attachment.attachment_id] = attachment
        if group_id is not None:
            g = self.groups[self.find_group(int(group_id))]
            g.attachment_ids.append(attachment.attachment_id)

    # --- Validation (V1) ---
    def validate_world(self) -> List[str]:
        errors: List[str] = []

        for el in self.elements.values():
            for edge in el.edges:
                last_t = -1.0
                for p in edge.points_on_edge:
                    if not (0.0 <= p.t <= 1.0):
                        errors.append(f"Edge {edge.edge_id()}: t hors bornes: {p.t}")
                    if p.t < last_t:
                        errors.append(f"Edge {edge.edge_id()}: pointsOnEdge non triés")
                    last_t = p.t

                for c in edge.coverages:
                    if not (0.0 <= c.t0 <= 1.0 and 0.0 <= c.t1 <= 1.0):
                        errors.append(f"Edge {edge.edge_id()}: coverage hors bornes [{c.t0},{c.t1}]")
                    if c.t0 >= c.t1:
                        errors.append(f"Edge {edge.edge_id()}: coverage dégénéré [{c.t0},{c.t1}]")

        # Boundary mono-cycle: placeholder (sera validé quand computeBoundary sera implémentée)
        return errors

    # --- TopoDump XML export (manuel, par scénario) ---
    def export_topo_dump_xml(self, scenario_id: str, path: str, orientation: str = "cw") -> str:
        root = _ET.Element("TopoDump", {
            "version": "4.3",
            "units": "km",
            "tFrame": "world",
            "scenarioId": str(scenario_id),
        })

        groups_el = _ET.SubElement(root, "Groups")
        for gid in sorted(self.groups.keys()):
            gcanon = self.find_group(gid)
            if gid != gcanon:
                continue
            g = self.groups[gid]
            grp = _ET.SubElement(groups_el, "Group", {
                "id": str(gid),
                "elements": str(len(g.element_ids)),
                "attachments": str(len(g.attachment_ids)),
                "originGroups": ",".join(map(str, sorted(self.group_members(gid))))
            })
            errs = self.validate_world()
            _ET.SubElement(grp, "Validate", {"errors": str(len(errs)), "warnings": "0"})

        nodes_el = _ET.SubElement(root, "Nodes")
        canonical_nodes = sorted({self.find_node(nid) for nid in self._node_parent.keys()})
        for cn in canonical_nodes:
            members = sorted(self.node_members(cn))
            _ET.SubElement(nodes_el, "Node", {
                "id": str(cn),
                "type": str(self.node_type(cn)),
                "origins": ",".join(map(str, members)),
            })

        elements_el = _ET.SubElement(root, "Elements")
        for eid in sorted(self.elements.keys()):
            el = self.elements[eid]
            elx = _ET.SubElement(elements_el, "Element", {
                "id": str(el.element_id),
                "name": str(el.name),
                "n": str(el.n_vertices()),
            })

            vx = _ET.SubElement(elx, "Vertices")
            for v in el.vertexes:
                _ET.SubElement(vx, "V", {
                    "idx": str(v.vertex_index),
                    "label": v.label,
                    "node": str(self.find_node(v.node_id)),
                    "nodeOrigin": str(v.node_id),
                    "type": str(v.node_type),
                })

            ex = _ET.SubElement(elx, "Edges")
            for e in el.edges:
                a, b = e.edge_label_key()
                edge = _ET.SubElement(ex, "Edge", {
                    "id": f"({e.element_id},{e.edge_index})",
                    "idx": str(e.edge_index),
                    "from": str(e.v_start.vertex_index),
                    "to": str(e.v_end.vertex_index),
                    "labels": e.labels_display(),
                    "edgeLabelKey": f"{a}|{b}",
                    "lengthKm": f"{e.edge_length_km:.3f}",
                })
                pts = _ET.SubElement(edge, "Points")
                for pnt in sorted(e.points_on_edge, key=lambda p: p.t):
                    _ET.SubElement(pts, "P", {
                        "t": f"{pnt.t:.6f}",
                        "node": str(self.find_node(pnt.node_id)),
                        "nodeOrigin": str(pnt.node_id),
                    })
                covs = _ET.SubElement(edge, "Coverages")
                for c in sorted(e.coverages, key=lambda c: (c.t0, c.t1)):
                    _ET.SubElement(covs, "C", {"t0": f"{c.t0:.6f}", "t1": f"{c.t1:.6f}"})

        atts_el = _ET.SubElement(root, "Attachments")
        for aid in sorted(self.attachments.keys()):
            a = self.attachments[aid]
            att = _ET.SubElement(atts_el, "Attachment", {
                "id": str(a.attachment_id),
                "kind": a.kind,
                "source": a.source,
                "featureA": a.feature_a.to_string(),
                "featureB": a.feature_b.to_string(),
            })
            params_el = _ET.SubElement(att, "Params")
            for k in sorted(a.params.keys()):
                _ET.SubElement(params_el, "Param", {"name": str(k), "value": str(a.params[k])})

        boundary_el = _ET.SubElement(root, "Boundary", {"orientation": str(orientation)})
        _ET.SubElement(boundary_el, "Note").text = "Boundary non implémentée en V1 (v4.3) (sera ajoutée en itération suivante)."

        tree = _ET.ElementTree(root)
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return path

