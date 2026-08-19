"""assembleur_sim.py
Moteur + algorithmes d'assemblage automatique (sans dépendance Tk).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Tuple, Type, Optional
import numpy as np
import math
import copy

from src.assembleur_core import (
    _tri_shape,
    _group_shape_from_nodes,
    ScenarioAssemblage,
    TopologyEdgeEdgeAttachment,
    TopologyElement,
    TopologyAttachmentResolutionError,
    TopologyAttachmentValidationError,
    TopologyConstraintGeometryError,
    TopologyVertexEdgeAttachment,
    TopologyWorld,
    compute_vertex_edge_attachment_orientation,
)
from src.assembleur_edge_mapping import (
    apply_edge_edge_pose,
    compute_edge_edge_pose,
)
from src.assembleur_projection import buildLastDrawnFromTopology
from src.assembleur_scenario import ScenarioHypothesis, materialize_catalogue_triangle

EPS_WORLD = 1e-6


@dataclass(frozen=True)
class InitialTriangleOrientation:
    """Stratégie d'orientation du premier triangle d'une simulation.

    Cette valeur ne transporte aucune référence UI : le moteur reçoit soit la
    convention historique d'une arête au Nord, soit un angle Core à atteindre.
    """
    mode: str
    edge: Optional[str] = None
    reference_element_id: Optional[str] = None
    reference_tri_rank: Optional[int] = None
    target_theta_rad: Optional[float] = None

    @classmethod
    def edge_north(cls, edge: str) -> "InitialTriangleOrientation":
        normalized_edge = edge.upper().strip()
        if normalized_edge not in ("OL", "BL"):
            raise ValueError(f"InitialTriangleOrientation: arête invalide {edge!r}")
        return cls(mode="edge_north", edge=normalized_edge)

    @classmethod
    def reference(
        cls, element_id: str, tri_rank: int, theta_rad: float
    ) -> "InitialTriangleOrientation":
        if not element_id:
            raise ValueError("InitialTriangleOrientation: element de référence absent")
        return cls(
            mode="reference",
            reference_element_id=element_id,
            reference_tri_rank=int(tri_rank),
            target_theta_rad=float(theta_rad),
        )


@dataclass(eq=False)
class _BranchNode:
    parent: Optional["_BranchNode"]
    children: List["_BranchNode"]
    branchTriangleId: Optional[str] = None


@dataclass
class PlacedTriangle:
    """Triangle projeté manipulé par le moteur de simulation."""
    triangleId: str
    points: Dict[str, np.ndarray]
    labels: tuple | list | None = None
    mirrored: bool = False
    topologyElementId: Optional[str] = None

    @classmethod
    def fromLegacyDict(cls, entry: Dict) -> "PlacedTriangle":
        if not isinstance(entry, dict):
            raise TypeError("PlacedTriangle.fromLegacyDict: dictionnaire attendu")
        return cls(
            triangleId=entry["id"],
            points=entry["pts"],
            labels=entry.get("labels"),
            mirrored=bool(entry.get("mirrored", False)),
            topologyElementId=entry.get("topoElementId"),
        )

    def toLegacyDict(self) -> Dict:
        entry = {"pts": self.points}
        if self.topologyElementId is not None:
            entry["topoElementId"] = self.topologyElementId
        return entry


class PlacedTriangles:
    """Conteneur minimal des triangles projetés d'une branche."""

    def __init__(self, entries: Optional[List[PlacedTriangle]] = None):
        self._entries = list(entries) if entries is not None else []
        if not all(isinstance(entry, PlacedTriangle) for entry in self._entries):
            raise TypeError("PlacedTriangles: objets PlacedTriangle attendus")

    def append(self, entry: PlacedTriangle) -> None:
        if not isinstance(entry, PlacedTriangle):
            raise TypeError("PlacedTriangles.append: objet PlacedTriangle attendu")
        self._entries.append(entry)

    def clone(self) -> "PlacedTriangles":
        return PlacedTriangles(copy.deepcopy(self._entries))

    def last(self) -> PlacedTriangle:
        return self._entries[-1]

    def count(self) -> int:
        return len(self._entries)

    def items(self) -> List[PlacedTriangle]:
        return self._entries

    def toLegacyList(self) -> List[Dict]:
        return [entry.toLegacyDict() for entry in self._entries]

    def findByTopologyElementId(self, element_id: str) -> Optional[PlacedTriangle]:
        target = str(element_id)
        for entry in self._entries:
            if str(entry.topologyElementId or "") == target:
                return entry
        return None

    def findByTriangleId(self, triangle_id: str) -> Optional[PlacedTriangle]:
        for entry in self._entries:
            if entry.triangleId == triangle_id:
                return entry
        return None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, index):
        return self._entries[index]


@dataclass
class BranchState:
    """État explicite d'une branche de simulation automatique."""
    node: _BranchNode
    topoWorld: TopologyWorld
    placedTriangles: PlacedTriangles
    orderedElementIds: List[str]
    poly_occ: object
    tailElementId: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.orderedElementIds = list(self.orderedElementIds)
        self.tailElementId = self.orderedElementIds[-1] if self.orderedElementIds else None


# ============================================================
# Décryptage (générique) — sans dépendance UI
# ============================================================


@dataclass
class ClockState:
    """État minimal du compas/horloge pour le décryptage."""
    hour: float = 0.0
    minute: float = 0
    label: str = ""
    # Conserver la provenance (utile pour debug / synchro)
    dicoRow: Optional[int] = None
    dicoCol: Optional[int] = None
    word: str = ""


class DecryptorBase:
    """Contrat minimal pour brancher différents types de décryptage."""
    id: str = "decrypt_base"
    label: str = "Décryptage (base)"

    def __init__(self):
        super().__init__()
        # Paramètres génériques (communs pour l’instant)
        self.hourMovesWithMinutes = True
        # Bases du cadran
        self.minutesBase: int = 60
        self.hoursBase: int = 12

    def getMinutesBase(self) -> int:
        """Base minutes du cadran (ex: 60 ou 100)."""
        return self.minutesBase

    def getHoursBase(self) -> int:
        """Base heures du cadran (ex: 12 ou 10)."""
        return self.hoursBase

    def setMinutesBase(self, base: int):
        b = int(base)
        if b not in (60, 100):
            raise ValueError(f"minutesBase invalide: {base}")
        self.minutesBase = b

    def setHoursBase(self, base: int):
        b = int(base)
        if b not in (12, 10):
            raise ValueError(f"hoursBase invalide: {base}")
        self.hoursBase = b

    def degreesPerMinute(self) -> float:
        base = max(1, int(self.getMinutesBase()))
        return 360.0 / float(base)

    def degreesPerHour(self) -> float:
        base = max(1, int(self.getHoursBase()))
        return 360.0 / float(base)

    def anglesFromClock(self, *, hour: float, minute: int) -> Tuple[float, float]:
        """Retourne (angleHourDeg, angleMinuteDeg) dans [0..360).

        Convention:
          - 0° = 12h
          - sens horaire
        """
        hBase = max(1, int(self.getHoursBase()))
        mBase = max(1, int(self.getMinutesBase()))
        h = float(hour) % float(hBase)
        m = int(minute) % int(mBase)
        ang_min = (m * self.degreesPerMinute()) % 360.0
        ang_hour = (h * self.degreesPerHour()) % 360.0
        return (ang_hour, ang_min)

    def clockStateFromDicoCell(self, *, row: int, col: int, word: str = "", mode: str = "delta") -> ClockState:
        """Convertit une cellule (row,col) en état d'horloge.
        Par défaut: non supporté.
        """
        raise NotImplementedError


class ClockDicoDecryptor(DecryptorBase):
    """Décryptage Horloge ↔ Dictionnaire.

    Modes supportés:
      - ABS:
        * row = 1..10 (pas de 0)
        * col = … -2, -1, 1, 2, … (pas de 0)
        * mapping compas: hour=row (1..10), minute=col en base 1
          (col=-1 => 60, col=-2 => 59, etc.)
        * Row 1 (premiere) = 1 heure ; Row 10 (derniere) = 10 heures

      - DELTA:
        * row = 0..9 (0 autorisé), col = delta signé (0 autorisé)
        * mapping compas: hour=row, minute=col mod 60  (ex: -5 => 55')
        * Pour le filtrage d'angle, la ligne a 2 interprétations horaires
          possibles (h et h+10 mod 12) — géré dans deltaAngleFromDicoCell.
    """
    id = "clock_dico_v1"
    label = "Horloge ↔ Dictionnaire (v1)"

    def clockStateFromDicoCell(self, *, row: int, col: int, word: str = "", mode: str = "delta") -> ClockState:
        m = str(mode or "delta").strip().lower()

        hBase = max(1, self.getHoursBase())
        mBase = max(1, self.getMinutesBase())

        if m.startswith("abs"):
            # --- ABS ---
            # Le référentiel est (1, 1) row = [1.10] ligne =  [-100.. 1] ou [1..100] ] la cellule [0,0] n'existe pas
            # On associe des heures de 1 à 10 ==> OUVERTURE = 1H
            rowDisp = ((row - 1) % 10) + 1
            hourDisp = rowDisp
            hour = hourDisp % hBase

            # col: pas de 0. Convertir en minute 1..60.
            # On suppose que la 1ere colonne = 1mn ==> OUVERTURE = 1mn
            # Pour le dictionnaire symétrique col = -1 ==> DEVIN = 60mn
            if col > 0:
                minute = col
            else:
                minute = mBase + col + 1
            minute = minute % mBase
            if minute == 0:
                minute = mBase

        else:
            # --- DELTA ---
            # Le référentiel est (0,0) donc la cellule [0,0] veut dire même mot
            hour = row % hBase
            hourDisp = hour
            # En DELTA, une colonne négative se lit comme une minute "avant":
            #   -5 => 55' (comme 10h - 5' = 9h55)
            # Donc: minute = col mod 60 (0..59)
            minute = col % mBase

        # Option : l’aiguille des heures avance avec les minutes
        if self.hourMovesWithMinutes:
            minuteFloat = 0.0 if minute == mBase else float(minute)
            hourFloat = (float(hour) + minuteFloat / float(mBase)) % float(hBase)
        else:
            hourFloat = float(hour)

        # Label
        w = str(word or "").strip()
        if w:
            label = f"{w} — ({hourDisp}h, {minute}')"
        else:
            label = f"({hourDisp}h, {minute}')"

        return ClockState(
            hour=float(hourFloat),
            minute=float(minute),
            label=label,
            dicoRow=row,
            dicoCol=col,
            word=w,
        )


# Petit registre (optionnel) pour brancher d’autres décryptages
DECRYPTORS: Dict[str, Type[DecryptorBase]] = {
    ClockDicoDecryptor.id: ClockDicoDecryptor,
}


class AlgorithmeAssemblage:
    """Contrat minimal pour un algo d'assemblage automatique."""
    id: str = "algo_base"
    label: str = "Algorithme (base)"

    def __init__(self, engine: "MoteurSimulationAssemblage"):
        self.engine = engine

    def run(self, triangle_ids: List[str]) -> List["ScenarioAssemblage"]:
        """Lance la simulation et retourne une liste de scénarios."""
        raise NotImplementedError


def createTopoQuadrilateral(
    *,
    world: TopologyWorld,
    triangleMobFromId: str,
    triangleMobToId: str,
    triangleMobFrom: dict,
    triangleMobTo: dict,
    triangleMobFrom_PtsLocal: Dict[str, np.ndarray],
    triangleMobTo_PtsLocal: Dict[str, np.ndarray],
    triangleMobFromPts: Dict[str, np.ndarray],
    triangleMobToPts: Dict[str, np.ndarray],
    entryOdd: PlacedTriangle | None = None,
    entryEven: PlacedTriangle | None = None,
    element_factory: Callable[[str], TopologyElement],
) -> tuple[str, str, str, str, str]:
    """
    Crée un quadrilatère topologique cohérent (toujours) :
      - 2 TopologyElement (odd/even) dans `world`
      - pose des deux éléments via setElementPoseFromWorldPts()
      - attachement interne edge-edge (arête commune classifiée par le Core)
      - commit topo
      - retourne (topoGroupId, elementIdOdd, elementIdEven, src_edge, dst_edge)

    entryOdd/entryEven (si fournis) reçoivent l'identifiant topologique de leur
    élément. Le groupe est projeté à la finalisation du scénario.
    """

    def _ensure_element_from_local(
        *,
        triangle_id: str,
    ) -> str:
        el = element_factory(triangle_id)
        # IMPORTANT: crée un nouveau groupe topo pour cet élément
        world.add_element_as_new_group(el)
        if not el.element_id:
            raise RuntimeError("createTopoQuadrilateral: le Core n'a pas attribue d'elementId")
        return el.element_id

    world.beginTopoTransaction()
    try:
        # --- 1) Creer les deux instances et conserver les IDs attribues par le Core. ---
        elementIdOdd = _ensure_element_from_local(
            triangle_id=triangleMobFromId,
        )
        elementIdEven = _ensure_element_from_local(
            triangle_id=triangleMobToId,
        )

        # --- 3) Poser les 2 éléments (monde) ---
        setElementPoseFromWorldPts(world, elementIdOdd, triangleMobFromPts, mirrored=False)
        setElementPoseFromWorldPts(world, elementIdEven, triangleMobToPts, mirrored=False)

        # Injecter topoElementId dans les entrées graphiques si fournies
        if entryOdd is not None:
            entryOdd.topologyElementId = elementIdOdd
        if entryEven is not None:
            entryEven.topologyElementId = elementIdEven

        # --- 4) Détecter l'unique arête métier commune dans le Core. ---
        common_edges = [
            (odd_edge, even_edge)
            for odd_edge in ("OB", "BL", "LO")
            for even_edge in ("OB", "BL", "LO")
            if world.are_same_business_edge(
                elementIdOdd,
                odd_edge,
                elementIdEven,
                even_edge,
            )
        ]
        if not common_edges:
            raise ValueError(
                "createTopoQuadrilateral: aucune arête métier commune"
            )
        if len(common_edges) != 1:
            raise ValueError(
                "createTopoQuadrilateral: arêtes métier communes ambiguës"
            )
        src_edge, dst_edge = common_edges[0]

        # --- 5) Créer le raccord interne edge-edge V2 ---
        # Le Resolver V2 détermine seul la correspondance directe/inversée
        # des extrémités : AUTO ne transporte aucun mapping d'endpoints.
        attachment = TopologyEdgeEdgeAttachment(
            attachment_id=world.new_attachment_id(),
            mob_element_id=elementIdOdd,
            mob_edge=src_edge,
            dest_element_id=elementIdEven,
            dest_edge=dst_edge,
        )
        group_mob_id = world.get_group_of_element(attachment.mob_element_id)
        group_dest_id = world.get_group_of_element(attachment.dest_element_id)
        if world.simulate_topological_overlap(
            group_dest_id,
            group_mob_id,
            attachment,
        ):
            raise TopologyConstraintGeometryError(
                "createTopoQuadrilateral: chevauchement topologique interne"
            )
        topoGroupId = world.apply_attachment(attachment)

        # Le premier triangle du couple reste la référence géométrique du
        # scénario. Le rejeu place uniquement l'élément pair depuis le EE V2.
        world.replay_group_attachment_poses(topoGroupId, elementIdOdd)
    finally:
        world.commitTopoTransaction()

    # --- 7) Groupe topo résultant (odd/even doivent maintenant être dans le même groupe) ---
    topoGroupId = world.get_group_of_element(elementIdOdd)

    return (topoGroupId, elementIdOdd, elementIdEven, src_edge, dst_edge)


def setElementPoseFromWorldPts(
    world: TopologyWorld,
    elementId: str,
    Pw: dict,
    mirrored: bool = False,
) -> None:
    eps = 1e-12
    if world is None or elementId not in world.elements:
        raise ValueError(f"setElementPoseFromWorldPts: elementId inconnu: {elementId}")
    if not isinstance(Pw, dict):
        raise ValueError("setElementPoseFromWorldPts: Pw invalide")
    for k in ("O", "B", "L"):
        if k not in Pw:
            raise ValueError("setElementPoseFromWorldPts: Pw incomplet")

    el = world.elements[elementId]
    pO = np.array(el.vertex_local_xy.get(0, (0.0, 0.0)), dtype=float)
    pB = np.array(el.vertex_local_xy.get(1, (0.0, 0.0)), dtype=float)
    pL = np.array(el.vertex_local_xy.get(2, (0.0, 0.0)), dtype=float)

    Ow = np.array(Pw["O"], dtype=float)
    Bw = np.array(Pw["B"], dtype=float)
    Lw = np.array(Pw["L"], dtype=float)
    if Ow.shape != (2,) or Bw.shape != (2,) or Lw.shape != (2,):
        raise ValueError("setElementPoseFromWorldPts: Pw dimension invalide")
    if not np.isfinite(Ow).all() or not np.isfinite(Bw).all() or not np.isfinite(Lw).all():
        raise ValueError("setElementPoseFromWorldPts: Pw non fini")

    if mirrored:
        M = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
        pO = (M @ pO)
        pB = (M @ pB)
        pL = (M @ pL)

    X = np.stack([pO, pB, pL], axis=0)
    Y = np.stack([Ow, Bw, Lw], axis=0)
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    if np.linalg.norm(Xc) <= eps or np.linalg.norm(Yc) <= eps:
        raise ValueError("setElementPoseFromWorldPts: points degeneres")
    H = Xc.T @ Yc
    U, _S, Vt = np.linalg.svd(H)
    R = (Vt.T @ U.T)
    if np.linalg.det(R) < 0.0:
        Vt[1, :] *= -1.0
        R = (Vt.T @ U.T)
    T = Y.mean(axis=0) - (R @ X.mean(axis=0))

    world.setElementPose(elementId, R=R, T=T, mirrored=bool(mirrored))


class AlgoQuadrisParPaires(AlgorithmeAssemblage):
    id = "quadris_par_paires"
    label = "Quadrilatères par paires (bases communes) [WIP]"

    def run(self, triangle_ids: List[str]) -> List["ScenarioAssemblage"]:
        """Étape 1+2 :
        - Si n=2 : assemble uniquement le 1er couple (2 triangles) en un quadrilatère.
        - Si n>=4 : assemble le 1er couple (tri1,tri2), puis assemble le 2e couple (tri3,tri4)
          et pose le quad2 en le connectant au quad1 via L2↔L3 avec EXACTEMENT 2 essais :
            - aligner (L3→O3) sur (L2→O2)
            - aligner (L3→O3) sur (L2→B2)

        - Applique la convention: le triangle 1 est orienté pour que l'azimut O→L soit 0° (Nord, axe +Y).
        - Détecte automatiquement la longueur commune entre les 2 triangles (OB / OL / BL).
        - Tente 2 poses (direction directe / inversée) et conserve les poses sans chevauchement (shrink-only).
        """
        if not triangle_ids or len(triangle_ids) < 2:
            return []
        if (len(triangle_ids) % 2) != 0:
            # On ne gère que des n pairs pour l'instant
            return []

        # --- DEBUG init ---
        engine = self.engine
        engine.debugReset(triangle_ids)

        tri1_id = triangle_ids[0]
        tri2_id = triangle_ids[1]

        v = engine.viewer

        topoScenarioId = "SA_AUTO"
        topoWorld0 = TopologyWorld()

        t1 = engine.build_local_triangle(tri1_id)
        t2 = engine.build_local_triangle(tri2_id)

        # ---- 1) Orientation : OL ou BL au Nord (+Y) pour le 1er triangle
        P1 = {k: np.array(t1["pts"][k], dtype=float) for k in ("O", "B", "L")}

        initial_orientation = engine.initialTriangleOrientation
        if initial_orientation is None:
            initial_orientation = InitialTriangleOrientation.edge_north(
                engine.firstTriangleEdge
            )
        if initial_orientation.mode not in ("edge_north", "reference"):
            raise ValueError(
                f"Simulation: mode d'orientation initiale invalide {initial_orientation.mode!r}"
            )
        edge = (
            initial_orientation.edge
            if initial_orientation.mode == "edge_north"
            else "OL"
        )
        src = "B" if edge == "BL" else "O"

        vOL = P1["L"] - P1[src]
        if float(np.hypot(vOL[0], vOL[1])) > 1e-12:
            cur = math.atan2(vOL[1], vOL[0])
            target = math.pi / 2.0  # Nord = +Y
            dtheta = target - cur
            c, s = math.cos(dtheta), math.sin(dtheta)
            R = np.array([[c, -s], [s, c]], dtype=float)
            pivot = P1[src]
            for k in ("O", "B", "L"):
                P1[k] = (R @ (P1[k] - pivot)) + pivot

        # ---- 2) Classification topologique de l'arête commune.
        def _shared_business_edges(triangle_a_id, triangle_b_id):
            probe = TopologyWorld()
            element_a = engine.materialize_triangle(triangle_a_id)
            element_b = engine.materialize_triangle(triangle_b_id)
            probe.add_element_as_new_group(element_a)
            probe.add_element_as_new_group(element_b)
            matches = [
                (edge_a, edge_b)
                for edge_a in ("OB", "BL", "LO")
                for edge_b in ("OB", "BL", "LO")
                if probe.are_same_business_edge(
                    element_a.element_id, edge_a, element_b.element_id, edge_b
                )
            ]
            if not matches:
                raise ValueError("AUTO: aucune arête métier commune")
            if len(matches) != 1:
                raise ValueError("AUTO: arêtes métier communes ambiguës")
            return matches[0]

        edge_vertices = {"OB": ("O", "B"), "BL": ("B", "L"), "LO": ("L", "O")}
        edge_1, edge_2 = _shared_business_edges(tri1_id, tri2_id)
        a1, b1 = edge_vertices[edge_1]
        a2, b2 = edge_vertices[edge_2]
        P2_local = {k: np.array(t2["pts"][k], dtype=float) for k in ("O", "B", "L")}

        def _bootstrap_topo_first_pair(
            *,
            world: TopologyWorld,
            tri1_id: str,
            tri2_id: str,
            t1: dict,
            t2: dict,
            P1_local: dict,
            P2_local: dict,
            P1_world: dict,
            P2_world: dict,
            base_placed_triangles: PlacedTriangles,
        ):
            """
            Bootstrap du premier quadrilatère topo.
            Délègue intégralement à createTopoQuadrilateral().
            """

            # Entrées placées du premier quadrilatère : tri1 puis tri2.
            entryOdd = base_placed_triangles.findByTriangleId(tri1_id)
            entryEven = base_placed_triangles.findByTriangleId(tri2_id)
            if entryOdd is None or entryEven is None:
                raise RuntimeError("Simulation: projection absente pour le premier quadrilatère")

            topoGroupId, elementIdOdd, elementIdEven, _, _ = createTopoQuadrilateral(
                world=world,
                triangleMobFromId=tri1_id,
                triangleMobToId=tri2_id,
                triangleMobFrom=t1,
                triangleMobTo=t2,
                triangleMobFrom_PtsLocal=P1_local,
                triangleMobTo_PtsLocal=P2_local,
                triangleMobFromPts=P1_world,
                triangleMobToPts=P2_world,
                entryOdd=entryOdd,
                entryEven=entryEven,
                element_factory=engine.materialize_triangle,
            )

            return topoGroupId, [elementIdOdd, elementIdEven]

        def _apply_reference_initial_orientation(
            world: TopologyWorld,
            ordered_element_ids: list[str],
            placed_triangles: PlacedTriangles,
        ) -> None:
            if initial_orientation.mode != "reference":
                return
            first_element_id = ordered_element_ids[0]
            first_group_id = world.get_group_of_element(first_element_id)
            first_l_node_id = world.get_element_vertex_node_id_by_type(
                first_element_id, "L"
            )
            pivot_world = np.asarray(
                world.getConceptNodeWorldXY(first_l_node_id, first_group_id), dtype=float
            )
            R_current, _T_current, mirrored_current = world.getElementPose(first_element_id)
            current_theta = math.atan2(float(R_current[1, 0]), float(R_current[0, 0]))
            target_theta = initial_orientation.target_theta_rad
            if target_theta is None:
                raise RuntimeError("Simulation: angle de référence absent")
            dtheta = math.atan2(
                math.sin(float(target_theta) - current_theta),
                math.cos(float(target_theta) - current_theta),
            )
            world.rotate_group(first_group_id, pivot_world, dtheta)
            _R_final, _T_final, mirrored_final = world.getElementPose(first_element_id)
            if mirrored_final != mirrored_current:
                raise RuntimeError("Simulation: l'orientation de référence a modifié mirrored")
            projected_by_element_id = {
                entry["topoElementId"]: entry["pts"]
                for entry in buildLastDrawnFromTopology(
                    topologyWorld=world, elementIds=ordered_element_ids
                )
            }
            for placed_triangle in placed_triangles:
                placed_triangle.points = projected_by_element_id[placed_triangle.topologyElementId]

        def _pose_triangle2_with_mapping(engine, v, Pm_local, am, bm, Pt, at, bt, poly_dest):
            pose = compute_edge_edge_pose(Pm_local, am, bm, Pt, at, bt)
            mapping = pose.mapping
            Pmw = apply_edge_edge_pose(Pm_local, pose)
            return Pmw, mapping, False

        # ---- 3) Pose du triangle 2 : 2 essais (direct / inversé)
        poly1 = _tri_shape(P1)
        poses = []
        # si tu veux garder mapping pour debug:
        # engine.debugInfo(... mapping=mapping)
        P2w, mapping, is_overlap = _pose_triangle2_with_mapping(
            engine, v,
            P2_local, a2, b2,     # arête mobile
            P1,      a1, b1,     # arête destination
            poly1
        )

        if is_overlap or P2w is None:
            # échec propre : ce couple de triangles ne peut pas former un quad valide
            return []   # ou raise / skip scénario selon ta logique actuelle

        poses.append(P2w)

        # ---- 4) Si n=2 : même comportement qu'avant (1 ou 2 scénarios possibles)
        if len(triangle_ids) <= 2:
            out: List[ScenarioAssemblage] = []
            poses_short = list(poses[:2])
            for i, P2 in enumerate(poses_short):
                scen = ScenarioAssemblage(
                    name=(f"#1" if i == 0 else f"#{i+1}=#1+({tri2_id})"),
                    source_type="auto",
                    algo_id=self.id,
                    hypothesis=engine.source_hypothesis.clone(),
                )
                scen.status = "complete"
                scen.topoScenarioId = topoScenarioId

                placed_triangles = PlacedTriangles()
                placed_triangles.append(PlacedTriangle(
                    triangleId=tri1_id,
                    points=P1,
                    labels=t1.get("labels"),
                    mirrored=bool(t1.get("mirrored", False)),
                ))
                placed_triangles.append(PlacedTriangle(
                    triangleId=tri2_id,
                    points=P2,
                    labels=t2.get("labels"),
                    mirrored=bool(t2.get("mirrored", False)),
                ))
                topoWorld_scen = TopologyWorld()
                _topo_group_id, ordered_element_ids = _bootstrap_topo_first_pair(
                    world=topoWorld_scen,
                    tri1_id=tri1_id,
                    tri2_id=tri2_id,
                    t1=t1,
                    t2=t2,
                    P1_local={k: np.array(t1["pts"][k], dtype=float) for k in ("O", "B", "L")},
                    P2_local={k: np.array(t2["pts"][k], dtype=float) for k in ("O", "B", "L")},
                    P1_world=P1,
                    P2_world=P2,
                    base_placed_triangles=placed_triangles,
                )
                _apply_reference_initial_orientation(
                    topoWorld_scen, ordered_element_ids, placed_triangles
                )
                scen.topoWorld = topoWorld_scen
                scen.orderedElementIds = list(ordered_element_ids)

                scen.last_drawn = buildLastDrawnFromTopology(
                    topologyWorld=topoWorld_scen,
                    elementIds=ordered_element_ids,
                )
            out.append(scen)
            return out

        # ---- 5) Étape 2 : chaîner les quadrilatères (tri3,tri4), (tri5,tri6), ... via les sommets Lumière
        # Convention : on connecte L(2) ↔ L(3), puis L(4) ↔ L(5), etc.
        # À chaque connexion : EXACTEMENT 2 essais (aligner (Lodd→Oodd) sur (Leven→Oeven) puis sur (Leven→Beven)).
        #
        # IMPORTANT :
        # - On fige l'assemblage interne de chaque paire (A,B) à la 1ère pose valide (shrink-only)
        #   pour éviter l'explosion combinatoire.
        # - Tous les triangles sont réunis par le même groupe Core afin que les
        #   tests de chevauchement et la manipulation de groupe restent cohérents.

        if len(triangle_ids) < 4:
            return []

        # On fige le 1er quad sur la pose retenue (déjà pruné par overlap shrink-only)
        P2 = poses[0]

        # Fabrique la projection placée de base (quad1)
        base_placed_triangles = PlacedTriangles([
            PlacedTriangle(
                triangleId=tri1_id,
                points=P1,
                labels=t1.get("labels"),
                mirrored=bool(t1.get("mirrored", False)),
            ),
            PlacedTriangle(
                triangleId=tri2_id,
                points=P2,
                labels=t2.get("labels"),
                mirrored=bool(t2.get("mirrored", False)),
            ),
        ])

        def _orient_O_to_L_north(P: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            P = {k: np.array(P[k], dtype=float) for k in ("O", "B", "L")}
            vOL = P["L"] - P["O"]
            if float(np.hypot(vOL[0], vOL[1])) > 1e-12:
                cur = math.atan2(vOL[1], vOL[0])
                target = math.pi / 2.0
                dtheta = target - cur
                c, s = math.cos(dtheta), math.sin(dtheta)
                R = np.array([[c, -s], [s, c]], dtype=float)
                pivot = P["O"]
                for k in ("O", "B", "L"):
                    P[k] = (R @ (P[k] - pivot)) + pivot
            return P

        def _edge_len(P: Dict[str, np.ndarray], a: str, b: str) -> float:
            vv = np.array(P[b], float) - np.array(P[a], float)
            return float(np.hypot(vv[0], vv[1]))

        def _build_quad_local(triA_id: int, triB_id: int) -> Tuple[Dict, Dict, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
            """Construit un quad (A,B) en repère local, en figeant B à la 1ère pose valide."""
            tA = engine.build_local_triangle(triA_id)
            tB = engine.build_local_triangle(triB_id)

            PA = _orient_O_to_L_north({k: np.array(tA["pts"][k], dtype=float) for k in ("O", "B", "L")})
            PB_local = {k: np.array(tB["pts"][k], dtype=float) for k in ("O", "B", "L")}

            edge_a, edge_b = _shared_business_edges(triA_id, triB_id)
            aA, bA = edge_vertices[edge_a]
            aB, bB = edge_vertices[edge_b]

            polyA = _tri_shape(PA)

            PB, mapping, is_overlap = _pose_triangle2_with_mapping(
                engine, v,
                PB_local, aB, bB,      # mobile
                PA,       aA, bA,      # dest
                polyA
            )

            if PB is None:
                raise ValueError("Aucune pose valide trouvée pour assembler le quadrilatère (A,B).")

            return tA, tB, PA, PB

        # État de recherche : liste de branches (scénarios partiels).
        poly_occ0 = _group_shape_from_nodes(
            [{"tid": 0}, {"tid": 1}],
            base_placed_triangles.toLegacyList()
        )

        _topo_group_id, base_ordered_element_ids = _bootstrap_topo_first_pair(
            world=topoWorld0,
            tri1_id=tri1_id,
            tri2_id=tri2_id,
            t1=t1,
            t2=t2,
            P1_local={k: np.array(t1["pts"][k], dtype=float) for k in ("O", "B", "L")},
            P2_local={k: np.array(t2["pts"][k], dtype=float) for k in ("O", "B", "L")},
            P1_world=P1,
            P2_world=P2,
            base_placed_triangles=base_placed_triangles,
        )

        if initial_orientation.mode == "reference":
            _apply_reference_initial_orientation(
                topoWorld0, base_ordered_element_ids, base_placed_triangles
            )
            poly_occ0 = _group_shape_from_nodes(
                [{"tid": 0}, {"tid": 1}], base_placed_triangles.toLegacyList()
            )

        rootNode = _BranchNode(parent=None, children=[], branchTriangleId=None)

        # État de recherche : liste de branches (scénarios partiels)
        # Chaque état conserve sa propre chronologie de construction, indépendante
        # de la topologie et de sa projection graphique.
        states = [BranchState(
            node=rootNode,
            topoWorld=topoWorld0,
            placedTriangles=base_placed_triangles,
            orderedElementIds=base_ordered_element_ids,
            poly_occ=poly_occ0,
        )]

        # Boucle sur les paires suivantes : (tri3,tri4), (tri5,tri6), ...
        for pair_start in range(2, len(triangle_ids), 2):
            if pair_start + 1 >= len(triangle_ids):
                break

            tri_odd_id = triangle_ids[pair_start]       # tri3, tri5, ...
            tri_even_id = triangle_ids[pair_start + 1]  # tri4, tri6, ...

            # On construit le quad local dans l'ordre courant (odd->even)
            tOdd, tEven, Podd, Peven = _build_quad_local(tri_odd_id, tri_even_id)

            def tryAttachMobQuadToDestChain(triangleMobFrom, triangleMobTo,
                                            triangleMobFromPts, triangleMobToPts,
                                            triangleMobFromId, triangleMobToId
                                            ):
                new_states = []
                dbg_try = 0
                dbg_overlap = 0
                dbg_added = 0

                for state in states:
                    node_prev = state.node
                    placed_triangles_prev = state.placedTriangles
                    poly_occ_prev = state.poly_occ
                    topoWorld_prev = state.topoWorld
                    ordered_element_ids_prev = state.orderedElementIds
                    baseKey = getattr(node_prev, "debugKey", "")    # Une cle de debug pour tracer les scénarios
                    candidates = []

                    # Le parent reste immuable : le quadrilatère mobile interne
                    # V2 est construit une seule fois dans son clone de base.
                    topo_world_candidate_base = topoWorld_prev.clonePhysicalState()
                    _, element_id_odd, element_id_even, _, _ = createTopoQuadrilateral(
                        world=topo_world_candidate_base,
                        triangleMobFromId=triangleMobFromId,
                        triangleMobToId=triangleMobToId,
                        triangleMobFrom=triangleMobFrom,
                        triangleMobTo=triangleMobTo,
                        triangleMobFrom_PtsLocal={k: np.array(triangleMobFrom["pts"][k], float) for k in ("O", "B", "L")},
                        triangleMobTo_PtsLocal={k: np.array(triangleMobTo["pts"][k], float) for k in ("O", "B", "L")},
                        triangleMobFromPts=triangleMobFromPts,
                        triangleMobToPts=triangleMobToPts,
                        element_factory=engine.materialize_triangle,
                    )
                    ordered_element_ids_base = [
                        *ordered_element_ids_prev,
                        element_id_odd,
                        element_id_even,
                    ]
                    if state.tailElementId is None:
                        raise RuntimeError("Simulation: élément final absent de la branche")

                    # Les quatre combinaisons métier sont indépendantes :
                    # LO/LO, LO/BL, BL/LO, BL/BL.
                    for mobEdgeAtL in ("LO", "BL"):
                        for destEdgeAtL in ("LO", "BL"):
                            dbg_try += 1
                            candidate_world = topo_world_candidate_base.clonePhysicalState()
                            attachment = TopologyVertexEdgeAttachment(
                                attachment_id=candidate_world.new_attachment_id(),
                                mob_element_id=element_id_odd,
                                mob_vertex="L",
                                creation_mob_edge=mobEdgeAtL,
                                dest_element_id=state.tailElementId,
                                dest_vertex="L",
                                creation_dest_edge=destEdgeAtL,
                                mob_orientation=compute_vertex_edge_attachment_orientation(
                                    candidate_world, element_id_odd, "L", mobEdgeAtL
                                ),
                                dest_orientation=compute_vertex_edge_attachment_orientation(
                                    candidate_world, state.tailElementId, "L", destEdgeAtL
                                ),
                            )
                            try:
                                group_mob_id = candidate_world.get_group_of_element(
                                    attachment.mob_element_id
                                )
                                group_dest_id = candidate_world.get_group_of_element(
                                    attachment.dest_element_id
                                )
                                if candidate_world.simulate_topological_overlap(
                                    group_dest_id,
                                    group_mob_id,
                                    attachment,
                                ):
                                    dbg_overlap += 1
                                    continue
                                candidate_world.beginTopoTransaction()
                                try:
                                    group_id = candidate_world.apply_attachment(attachment)
                                    candidate_world.replay_group_attachment_poses(
                                        group_id,
                                        state.tailElementId,
                                    )
                                finally:
                                    candidate_world.commitTopoTransaction()
                            except (
                                TopologyAttachmentValidationError,
                                TopologyAttachmentResolutionError,
                                TopologyConstraintGeometryError,
                            ):
                                continue

                            projection = buildLastDrawnFromTopology(
                                topologyWorld=candidate_world,
                                elementIds=ordered_element_ids_base,
                            )
                            projected_points = {
                                entry["topoElementId"]: entry["pts"]
                                for entry in projection
                            }
                            placed_triangles_new = placed_triangles_prev.clone()
                            for placed_triangle in placed_triangles_new:
                                placed_triangle.points = projected_points[
                                    placed_triangle.topologyElementId
                                ]
                            placed_triangles_new.append(PlacedTriangle(
                                triangleId=triangleMobFromId,
                                points=projected_points[element_id_odd],
                                labels=triangleMobFrom.get("labels"),
                                mirrored=bool(triangleMobFrom.get("mirrored", False)),
                                topologyElementId=element_id_odd,
                            ))
                            placed_triangles_new.append(PlacedTriangle(
                                triangleId=triangleMobToId,
                                points=projected_points[element_id_even],
                                labels=triangleMobTo.get("labels"),
                                mirrored=bool(triangleMobTo.get("mirrored", False)),
                                topologyElementId=element_id_even,
                            ))
                            poly_new = _group_shape_from_nodes(
                                [{"tid": 0}, {"tid": 1}],
                                [
                                    {"pts": projected_points[element_id_odd]},
                                    {"pts": projected_points[element_id_even]},
                                ],
                            )
                            candKey = f"{baseKey}|{triangleMobFromId}:{mobEdgeAtL}->{destEdgeAtL}"
                            candidates.append((
                                candidate_world,
                                placed_triangles_new,
                                poly_occ_prev.union(poly_new),
                                candKey,
                            ))
                            dbg_added += 1

                    # Si au moins 2 candidats existent *à cette étape*, on enregistre une bifurcation.
                    # La bifurcation ne devient "réelle" que si les 2 sous-branches mènent à des feuilles survivantes,
                    # ce qui sera résolu après pruning (sur l'arbre survivant).
                    if candidates:
                        node_prev.children = []
                        if len(candidates) >= 2:
                            # IMPORTANT (naming): la bifurcation correspond au triangle
                            # connecté au bloc précédent via le point de Lumière,
                            # c'est triOddId (triEvenId est "collé" au triOddId via BO).
                            node_prev.branchTriangleId = triangleMobFromId
                        else:
                            node_prev.branchTriangleId = None

                        for topo_new, placed_triangles_new, poly_u, cand_key in candidates:
                            child = _BranchNode(parent=node_prev, children=[], branchTriangleId=None)
                            node_prev.children.append(child)
                            child.debugKey = cand_key
                            new_states.append(BranchState(
                                node=child,
                                topoWorld=topo_new,
                                placedTriangles=placed_triangles_new,
                                orderedElementIds=ordered_element_ids_base,
                                poly_occ=poly_u,
                            ))

                return new_states, dbg_try, dbg_overlap, dbg_added

            # 1) Tentative standard : mobile = odd (tri_odd_id)
            new_states, dbg_try, dbg_overlap, dbg_added = tryAttachMobQuadToDestChain(
                tOdd, tEven, Podd, Peven, tri_odd_id, tri_even_id
            )

            states = new_states
            if not states:
                engine.debugFail(
                    step="chain_connect",
                    pair=(tri_odd_id, tri_even_id),
                    reason="Aucune connexion valide (chaînage)",
                    detail=f"essais={dbg_try}, prunes_overlap={dbg_overlap}, ajoutés={dbg_added}",
                )
                return []

        # --- Post-traitement : construire une numérotation COHÉRENTE sur l'arbre survivant ---
        # Objectif : pouvoir "pruner mentalement" par plages (#1..#96 / #97..#117, etc.).
        leafData = {state.node: state for state in states}

        kept = set()
        for leaf in leafData.keys():
            n = leaf
            while n is not None and n not in kept:
                kept.add(n)
                n = n.parent

        def _keptChildren(n):
            return [c for c in (n.children or []) if c in kept]

        # Collecte des feuilles survivantes dans l'ordre gauche→droite (DFS)
        leaves = []

        def _collectLeaves(n):
            if n not in kept:
                return
            ch = _keptChildren(n)
            if not ch:
                leaves.append(n)
                return
            for c in ch:
                _collectLeaves(c)

        _collectLeaves(rootNode)

        leafIndex = {leaf: (i + 1) for i, leaf in enumerate(leaves)}

        def _leftMostLeaf(n):
            cur = n
            while True:
                ch = _keptChildren(cur)
                if not ch:
                    return cur
                cur = ch[0]

        # Par défaut : "#k"
        labels = {leaf: f"#{leafIndex[leaf]}" for leaf in leaves}

        # Pour chaque bifurcation survivante, on étiquette UNIQUEMENT le "start" du sous-arbre droit :
        #   #startR = #startL + (triEvenId)
        for n in kept:
            ch = _keptChildren(n)
            if n.branchTriangleId is None:
                continue
            if len(ch) < 2:
                continue
            leftLeaf = _leftMostLeaf(ch[0])
            idxL = leafIndex.get(leftLeaf)
            if idxL is None:
                continue
            for j in range(1, len(ch)):
                rightLeaf = _leftMostLeaf(ch[j])
                idxR = leafIndex.get(rightLeaf)
                if idxR is None:
                    continue
                labels[rightLeaf] = (
                    f"#{idxR}=#{idxL}+({engine.get_hypothesis_rank(n.branchTriangleId)})"
                )

        # Finalisation : créer les scénarios complets
        out: List[ScenarioAssemblage] = []
        for leaf in leaves:
            state = leafData[leaf]
            placed_triangles = state.placedTriangles
            topoWorld_leaf = state.topoWorld
            idx = int(leafIndex.get(leaf, 0) or 0)
            scen = ScenarioAssemblage(
                name=labels.get(leaf, f"#{idx}"),
                source_type="auto",
                algo_id=self.id,
                hypothesis=engine.source_hypothesis.clone(),
            )
            scen.status = "complete"
            scen.topoWorld = topoWorld_leaf
            scen.topoScenarioId = topoScenarioId
            scen.orderedElementIds = list(state.orderedElementIds)

            scen.last_drawn = buildLastDrawnFromTopology(
                topologyWorld=topoWorld_leaf,
                elementIds=state.orderedElementIds,
            )

            out.append(scen)

        return out


# Registre des algorithmes disponibles pour la boîte de dialogue.
# IMPORTANT : conserver l'ordre pour un affichage stable dans la combo.
ALGOS: Dict[str, Type[AlgorithmeAssemblage]] = {
    AlgoQuadrisParPaires.id: AlgoQuadrisParPaires,
}


class MoteurSimulationAssemblage:
    """Wrapper 'pur data' autour des briques géométriques du viewer (manuel)."""

    def __init__(
        self,
        viewer: "TriangleViewerManual",
        source_hypothesis: ScenarioHypothesis,
    ):
        self.viewer = viewer
        self.source_hypothesis = source_hypothesis.clone()
        self.firstTriangleEdge = "OL"
        self.initialTriangleOrientation: InitialTriangleOrientation | None = None
        # --- DEBUG (instrumentation minimaliste, sans console spam) ---
        # Rempli par les algos si un run() échoue et retourne [].
        self.debug_last: Dict | None = None

    def get_hypothesis_rank(self, triangle_id: str) -> int:
        try:
            return self.source_hypothesis.triangle_ids_by_rank.index(triangle_id) + 1
        except ValueError as exc:
            raise ValueError(
                f"Simulation: triangle absent de l'hypothèse: {triangle_id!r}"
            ) from exc

    # --- DEBUG helpers ---
    def debugReset(self, triangle_ids: List[str] | None = None):
        self.debug_last = {
            "triangle_ids": list(triangle_ids or []),
            "step": None,
            "pair": None,
            "anchor": None,
            "reason": None,
            "detail": None,
        }

    def debugFail(self, step: str, pair=None, reason: str | None = None, detail: str | None = None, anchor=None):
        if self.debug_last is None:
            self.debugReset([])
        self.debug_last.update({
            "step": step,
            "pair": pair,
            "anchor": anchor,
            "reason": reason,
            "detail": detail,
        })

    def materialize_triangle(self, triangle_id: str):
        return materialize_catalogue_triangle(self.viewer.catalogue, triangle_id)

    def build_local_triangle(self, triangle_id: str) -> Dict:
        """Construit la géométrie locale depuis la factory Catalogue/Core unique."""
        element = self.materialize_triangle(triangle_id)
        return {
            "labels": tuple(element.vertex_labels),
            "triangle_id": triangle_id,
            "mirrored": False,
            "orient": element.local_frame["orient"],
            "pts": {
                "O": np.array(element.vertex_local_xy[0], dtype=float),
                "B": np.array(element.vertex_local_xy[1], dtype=float),
                "L": np.array(element.vertex_local_xy[2], dtype=float),
            },
        }
