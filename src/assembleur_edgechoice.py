import math
import numpy as np

# === Modules externalisés (découpage maintenable) ===
from src.assembleur_core import (
    TopologyAttachment, TopologyFeatureRef, TopologyFeatureType,
)


class EdgeChoiceEpts:
    __slots__ = ("mA", "mBEdgeVertex", "tA", "tBEdgeVertex",
                 "src_owner_tid", "src_edge", "dst_owner_tid", "dst_edge",
                 "src_vkey_at_mA", "src_vkey_at_mB", "dst_vkey_at_tA", "dst_vkey_at_tB",
                 "src_edge_labels", "dst_edge_labels", "kind", "tRaw",
                 "elementIdSrc", "elementIdDst")

    def __init__(self, mA, mBEdgeVertex, tA, tBEdgeVertex,
                 src_owner_tid=None, src_edge=None, dst_owner_tid=None, dst_edge=None,
                 src_vkey_at_mA=None, src_vkey_at_mB=None, dst_vkey_at_tA=None, dst_vkey_at_tB=None,
                 src_edge_labels=None, dst_edge_labels=None, kind=None, t_raw=None,
                 elementIdSrc=None, elementIdDst=None):
        self.mA = tuple(mA)
        self.mBEdgeVertex = tuple(mBEdgeVertex)
        self.tA = tuple(tA)
        self.tBEdgeVertex = tuple(tBEdgeVertex)
        self.src_owner_tid = src_owner_tid
        self.src_edge = src_edge
        self.dst_owner_tid = dst_owner_tid
        self.dst_edge = dst_edge
        self.src_vkey_at_mA = src_vkey_at_mA
        self.src_vkey_at_mB = src_vkey_at_mB
        self.dst_vkey_at_tA = dst_vkey_at_tA
        self.dst_vkey_at_tB = dst_vkey_at_tB
        self.src_edge_labels = src_edge_labels
        self.dst_edge_labels = dst_edge_labels
        self.kind = kind
        self.tRaw = t_raw
        self.elementIdSrc = elementIdSrc
        self.elementIdDst = elementIdDst

    def __len__(self):
        return 4

    def __iter__(self):
        yield self.mA
        yield self.mBEdgeVertex
        yield self.tA
        yield self.tBEdgeVertex

    def __getitem__(self, i):
        if i == 0:
            return self.mA
        if i == 1:
            return self.mBEdgeVertex
        if i == 2:
            return self.tA
        if i == 3:
            return self.tBEdgeVertex
        raise IndexError(i)

    def computeRigidTransform(self, *, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Compute la pose rigide 2D telle que p_dest = R @ p_mob + T.

        Points utilisÃ©s :
        - mA, mBEdgeVertex : segment mobile (direction)
        - tA, tBEdgeVertex : segment cible (direction)

        Retourne None si un segment est dÃ©gÃ©nÃ©rÃ© (norme <= eps) ou si les points sont invalides.
        """
        p0m = np.array([float(self.mA[0]), float(self.mA[1])], dtype=float)
        p1m = np.array([float(self.mBEdgeVertex[0]), float(self.mBEdgeVertex[1])], dtype=float)
        p0d = np.array([float(self.tA[0]), float(self.tA[1])], dtype=float)
        p1d = np.array([float(self.tBEdgeVertex[0]), float(self.tBEdgeVertex[1])], dtype=float)

        vm = p1m - p0m
        vd = p1d - p0d

        nm = math.hypot(float(vm[0]), float(vm[1]))
        nd = math.hypot(float(vd[0]), float(vd[1]))
        if nm <= eps or nd <= eps:
            return None

        ang_m = math.atan2(float(vm[1]), float(vm[0]))
        ang_d = math.atan2(float(vd[1]), float(vd[0]))
        dtheta = ang_d - ang_m

        c = math.cos(dtheta)
        s = math.sin(dtheta)
        R = np.array([[float(c), float(-s)], [float(s), float(c)]], dtype=float)

        T = p0d - R @ p0m
        return (R, T)

    def createTopologyAttachments(self, *, world, debug: bool = False):
        if world is None:
            raise ValueError("createTopologyAttachments: world manquant")
        elementIdSrc = self.elementIdSrc
        elementIdDst = self.elementIdDst

        edge_code_to_index = {"OB": 0, "BL": 1, "LO": 2}
        vkey_to_index = {"O": 0, "B": 1, "L": 2}
        incident_edges = {
            elementIdSrc: self.src_edge.upper(),
            elementIdDst: self.dst_edge.upper(),
        }

        kind = str(self.kind or "").strip().lower()
        if kind not in ("edge-edge", "vertex-edge"):
            raise ValueError(f"createTopologyAttachments: kind inattendu: {kind}")

        atts: list[TopologyAttachment] = []

        if kind == "edge-edge":
            em = edge_code_to_index[self.src_edge.upper()]
            et = edge_code_to_index[self.dst_edge.upper()]
            if (self.src_vkey_at_mA == self.dst_vkey_at_tA) and (self.src_vkey_at_mB == self.dst_vkey_at_tB):
                mapping = "direct"
            elif (self.src_vkey_at_mA == self.dst_vkey_at_tB) and (self.src_vkey_at_mB == self.dst_vkey_at_tA):
                mapping = "reverse"
            else:
                msg = (
                    "edge-edge: mapping indéterminable "
                    f"(mA={self.src_vkey_at_mA} mB={self.src_vkey_at_mB} "
                    f"tA={self.dst_vkey_at_tA} tB={self.dst_vkey_at_tB})"
                )
                if debug:
                    raise RuntimeError(msg)
                print(f"[ATTACH][edge-edge] {msg}")
                return []
            atts.append(
                TopologyAttachment(
                    attachment_id=None,
                    kind="edge-edge",
                    feature_a=TopologyFeatureRef(TopologyFeatureType.EDGE, elementIdSrc, int(em)),
                    feature_b=TopologyFeatureRef(TopologyFeatureType.EDGE, elementIdDst, int(et)),
                    params={"mapping": mapping, "incident_edge_by_element": incident_edges},
                    source="manual",
                )
            )
            return atts

        return materialize_vertex_edge_attachments(
            world=world,
            element_id_src=elementIdSrc,
            src_edge=self.src_edge,
            src_anchor_vkey=self.src_vkey_at_mA,
            element_id_dst=elementIdDst,
            dst_edge=self.dst_edge,
            dst_anchor_vkey=self.dst_vkey_at_tA,
        )


def _to_np(p):
    return np.array([float(p[0]), float(p[1])], dtype=float)


def _edge_code_from_vkeys(a, b):
    if not a or not b or a == b:
        return None
    s = {a, b}
    if s == {"O", "B"}:
        return "OB"
    if s == {"B", "L"}:
        return "BL"
    if s == {"L", "O"}:
        return "LO"
    return None


def _edge_vkeys_from_code(edge_code: str):
    e = str(edge_code or "").upper().strip()
    if e == "OB":
        return ("O", "B")
    if e == "BL":
        return ("B", "L")
    if e == "LO":
        return ("L", "O")
    return (None, None)


def _edge_labels_for(owner_tid: int, edge_code: str, last_drawn):
    if owner_tid is None or not (0 <= int(owner_tid) < len(last_drawn)):
        return None
    tri = last_drawn[int(owner_tid)]
    labels = tri.get("labels", None)
    if labels is None:
        return None
    a, b = _edge_vkeys_from_code(edge_code)
    if not a or not b:
        return None
    idx = {"O": 0, "B": 1, "L": 2}
    ia, ib = idx.get(a, None), idx.get(b, None)
    if ia is None or ib is None:
        return None
    la = labels[ia] if ia < len(labels) else ""
    lb = labels[ib] if ib < len(labels) else ""
    return (la.strip(), lb.strip())


def _find_owner_edge_for_segment(group_tids, A, B, eps_world, last_drawn):
    if not group_tids:
        return (None, None)

    Ax, Ay = float(A[0]), float(A[1])
    Bx, By = float(B[0]), float(B[1])

    def _proj_t_and_d2(P0x, P0y, P1x, P1y, Qx, Qy):
        vx, vy = (P1x - P0x), (P1y - P0y)
        wx, wy = (Qx - P0x), (Qy - P0y)
        vv = vx*vx + vy*vy
        if vv <= 1e-12:
            return (0.0, (Qx - P0x)**2 + (Qy - P0y)**2)
        t = (wx*vx + wy*vy) / vv
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        px, py = (P0x + t*vx), (P0y + t*vy)
        dx, dy = (Qx - px), (Qy - py)
        return (t, dx*dx + dy*dy)

    best = None  # (score, tid, edge_code)
    for tid in group_tids:
        if tid is None or not (0 <= tid < len(last_drawn)):
            continue
        P = last_drawn[tid].get("pts")
        if not P:
            continue
        for (a, b) in (("O", "B"), ("B", "L"), ("L", "O")):
            P0x, P0y = float(P[a][0]), float(P[a][1])
            P1x, P1y = float(P[b][0]), float(P[b][1])

            tA, d2A = _proj_t_and_d2(P0x, P0y, P1x, P1y, Ax, Ay)
            tB, d2B = _proj_t_and_d2(P0x, P0y, P1x, P1y, Bx, By)

            if d2A <= eps_world*eps_world and d2B <= eps_world*eps_world:
                score = d2A + d2B
                if best is None or score < best[0]:
                    best = (score, tid, _edge_code_from_vkeys(a, b))

    if best is None:
        return (None, None)
    return (best[1], best[2])


def _edge_code_to_vkeys(edge_code: str):
    c = str(edge_code or "").upper()
    if c == "OB":
        return ("O", "B")
    if c == "BL":
        return ("B", "L")
    if c == "LO":
        return ("L", "O")
    return (None, None)


def _assign_edge_endpoints_by_cross_distance(src_edge_code, dst_edge_code, Pm, Pt):
    """
    Retourne:
      (srcA, srcB), (dstA, dstB)
    où A est le couple minimisant la distance croisée, et B est déduit (autre endpoint).
    """
    sv0, sv1 = _edge_code_to_vkeys(src_edge_code)
    dv0, dv1 = _edge_code_to_vkeys(dst_edge_code)
    if sv0 is None or sv1 is None or dv0 is None or dv1 is None:
        return ((None, None), (None, None))

    pairs = [(sv0, dv0), (sv0, dv1), (sv1, dv0), (sv1, dv1)]
    best_pair = None
    best_d2 = None
    for sa, da in pairs:
        p = _to_np(Pm[sa])
        q = _to_np(Pt[da])
        d2 = float((p[0] - q[0])**2 + (p[1] - q[1])**2)
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_pair = (sa, da)

    if best_pair is None:
        return ((None, None), (None, None))

    srcA, dstA = best_pair
    srcB = sv1 if srcA == sv0 else sv0
    dstB = dv1 if dstA == dv0 else dv0
    return ((srcA, srcB), (dstA, dstB))


def _compute_t_by_edge_ratio(mA, mB, tA, tB):
    mAx, mAy = float(mA[0]), float(mA[1])
    mBx, mBy = float(mB[0]), float(mB[1])
    tAx, tAy = float(tA[0]), float(tA[1])
    tBx, tBy = float(tB[0]), float(tB[1])
    dx_m = mBx - mAx
    dy_m = mBy - mAy
    dx_t = tBx - tAx
    dy_t = tBy - tAy
    L_src = math.hypot(dx_m, dy_m)
    L_dst = math.hypot(dx_t, dy_t)
    if L_dst <= 1e-12:
        raise ValueError("Degenerate target edge length")
    t = L_src / L_dst
    return float(t)


def materialize_vertex_edge_attachments(
    *,
    world,
    element_id_src: str,
    src_edge: str,
    src_anchor_vkey: str,
    element_id_dst: str,
    dst_edge: str,
    dst_anchor_vkey: str,
    vertex_edge_attachment_id: str | None = None,
    vertex_vertex_attachment_id: str | None = None,
    vertex_edge_source: str = "manual",
    vertex_vertex_source: str = "manual",
) -> list[TopologyAttachment]:
    """Matérialise le raccord atomique vertex-edge depuis ses arêtes métier.

    L'ordre ``src`` / ``dst`` est structurel : il conserve l'orientation de la
    décision de groupement initiale. Les longueurs, elles, sont toujours lues
    dans le ``TopologyWorld`` courant afin que le même contrat serve aussi aux
    rematérialisations après déformation.
    """
    if world is None:
        raise ValueError("materialize_vertex_edge_attachments: world manquant")

    edge_code_to_index = {"OB": 0, "BL": 1, "LO": 2}
    vkey_to_index = {"O": 0, "B": 1, "L": 2}
    source_edge = str(src_edge or "").strip().upper()
    destination_edge = str(dst_edge or "").strip().upper()
    source_anchor = str(src_anchor_vkey or "").strip().upper()
    destination_anchor = str(dst_anchor_vkey or "").strip().upper()
    if source_edge not in edge_code_to_index or destination_edge not in edge_code_to_index:
        raise ValueError("materialize_vertex_edge_attachments: arête incidente invalide")
    if source_anchor not in vkey_to_index or destination_anchor not in vkey_to_index:
        raise ValueError("materialize_vertex_edge_attachments: sommet d'ancrage invalide")
    if element_id_src not in world.elements or element_id_dst not in world.elements:
        raise ValueError("materialize_vertex_edge_attachments: élément incident inconnu")

    source_edge_vkeys = _edge_code_to_vkeys(source_edge)
    destination_edge_vkeys = _edge_code_to_vkeys(destination_edge)
    if source_anchor not in source_edge_vkeys or destination_anchor not in destination_edge_vkeys:
        raise ValueError(
            "materialize_vertex_edge_attachments: sommet d'ancrage hors de son arête"
        )

    source_other = (
        source_edge_vkeys[1]
        if source_anchor == source_edge_vkeys[0]
        else source_edge_vkeys[0]
    )
    destination_other = (
        destination_edge_vkeys[1]
        if destination_anchor == destination_edge_vkeys[0]
        else destination_edge_vkeys[0]
    )
    source_element = world.elements[element_id_src]
    destination_element = world.elements[element_id_dst]
    source_anchor_index = vkey_to_index[source_anchor]
    source_other_index = vkey_to_index[source_other]
    destination_anchor_index = vkey_to_index[destination_anchor]
    destination_other_index = vkey_to_index[destination_other]
    try:
        source_anchor_point = source_element.vertex_local_xy[source_anchor_index]
        source_other_point = source_element.vertex_local_xy[source_other_index]
        destination_anchor_point = destination_element.vertex_local_xy[destination_anchor_index]
        destination_other_point = destination_element.vertex_local_xy[destination_other_index]
    except KeyError as exc:
        raise ValueError(
            "materialize_vertex_edge_attachments: géométrie intrinsèque incomplète"
        ) from exc

    t_raw = _compute_t_by_edge_ratio(
        source_anchor_point,
        source_other_point,
        destination_anchor_point,
        destination_other_point,
    )
    if t_raw < 0.0:
        raise ValueError(
            f"materialize_vertex_edge_attachments: tRaw négatif ({t_raw})"
        )

    incident_edges = {
        element_id_src: source_edge,
        element_id_dst: destination_edge,
    }
    if t_raw <= 1.0:
        vertex_edge = TopologyAttachment(
            attachment_id=vertex_edge_attachment_id,
            kind="vertex-edge",
            feature_a=TopologyFeatureRef(
                TopologyFeatureType.VERTEX, element_id_src, source_other_index
            ),
            feature_b=TopologyFeatureRef(
                TopologyFeatureType.EDGE,
                element_id_dst,
                edge_code_to_index[destination_edge],
            ),
            params={
                "t": float(t_raw),
                "edgeFrom": world.get_element_vertex_node_id(
                    element_id_dst, destination_anchor_index
                ),
                "incident_edge_by_element": incident_edges,
            },
            source=vertex_edge_source,
        )
        vertex_vertex = TopologyAttachment(
            attachment_id=vertex_vertex_attachment_id,
            kind="vertex-vertex",
            feature_a=TopologyFeatureRef(
                TopologyFeatureType.VERTEX, element_id_src, source_anchor_index
            ),
            feature_b=TopologyFeatureRef(
                TopologyFeatureType.VERTEX,
                element_id_dst,
                destination_anchor_index,
            ),
            params={"incident_edge_by_element": incident_edges},
            source=vertex_vertex_source,
        )
    else:
        vertex_edge = TopologyAttachment(
            attachment_id=vertex_edge_attachment_id,
            kind="vertex-edge",
            feature_a=TopologyFeatureRef(
                TopologyFeatureType.VERTEX,
                element_id_dst,
                destination_other_index,
            ),
            feature_b=TopologyFeatureRef(
                TopologyFeatureType.EDGE,
                element_id_src,
                edge_code_to_index[source_edge],
            ),
            params={
                "t": float(1.0 / t_raw),
                "edgeFrom": world.get_element_vertex_node_id(
                    element_id_src, source_anchor_index
                ),
                "incident_edge_by_element": incident_edges,
            },
            source=vertex_edge_source,
        )
        vertex_vertex = TopologyAttachment(
            attachment_id=vertex_vertex_attachment_id,
            kind="vertex-vertex",
            feature_a=TopologyFeatureRef(
                TopologyFeatureType.VERTEX,
                element_id_dst,
                destination_anchor_index,
            ),
            feature_b=TopologyFeatureRef(
                TopologyFeatureType.VERTEX, element_id_src, source_anchor_index
            ),
            params={"incident_edge_by_element": incident_edges},
            source=vertex_vertex_source,
        )
    return [vertex_edge, vertex_vertex]


def rematerialize_vertex_edge_attachments_for_element(
    world,
    element_id: str,
) -> list[TopologyAttachment]:
    """Retourne les attachments rematérialisés touchant ``element_id``.

    Une liaison vertex-edge est un raccord atomique : son vertex-vertex frère
    est reconstruit avec le même sens et les mêmes arêtes incidentes. Les deux
    identifiants d'attachment restent stables.
    """
    attachments = list(world.attachments.values())
    replacements: dict[str, TopologyAttachment] = {}
    for vertex_edge in attachments:
        if vertex_edge.kind != "vertex-edge":
            continue
        endpoint_ids = {
            vertex_edge.feature_a.element_id,
            vertex_edge.feature_b.element_id,
        }
        if element_id not in endpoint_ids:
            continue
        if len(endpoint_ids) != 2:
            raise ValueError(
                f"Attachment {vertex_edge.attachment_id}: vertex-edge sans deux éléments distincts"
            )
        incident_edges = vertex_edge.params.get("incident_edge_by_element")
        if not isinstance(incident_edges, dict):
            raise ValueError(
                f"Attachment {vertex_edge.attachment_id}: incident_edge_by_element absent"
            )
        incident_items = list(incident_edges.items())
        if len(incident_items) != 2:
            raise ValueError(
                f"Attachment {vertex_edge.attachment_id}: incident_edge_by_element incohérent"
            )
        (source_element_id, source_edge), (destination_element_id, destination_edge) = incident_items
        if {source_element_id, destination_element_id} != endpoint_ids:
            raise ValueError(
                f"Attachment {vertex_edge.attachment_id}: arêtes incidentes incohérentes"
            )

        siblings = [
            attachment
            for attachment in attachments
            if attachment.kind == "vertex-vertex"
            and {
                attachment.feature_a.element_id,
                attachment.feature_b.element_id,
            } == endpoint_ids
        ]
        if len(siblings) != 1:
            raise ValueError(
                f"Attachment {vertex_edge.attachment_id}: raccord atomique vertex-vertex ambigu"
            )
        vertex_vertex = siblings[0]
        sibling_incident_edges = vertex_vertex.params.get("incident_edge_by_element")
        if not isinstance(sibling_incident_edges, dict):
            raise ValueError(
                f"Attachment {vertex_edge.attachment_id}: incident_edge_by_element absent dans le raccord atomique"
            )
        if list(sibling_incident_edges.items()) != incident_items:
            raise ValueError(
                f"Attachment {vertex_edge.attachment_id}: arêtes incidentes incohérentes dans le raccord atomique"
            )

        vertex_by_element = {
            vertex_vertex.feature_a.element_id: vertex_vertex.feature_a,
            vertex_vertex.feature_b.element_id: vertex_vertex.feature_b,
        }
        if set(vertex_by_element) != endpoint_ids:
            raise ValueError(
                f"Attachment {vertex_edge.attachment_id}: vertex-vertex incohérent"
            )
        source_vertex = vertex_by_element[source_element_id]
        destination_vertex = vertex_by_element[destination_element_id]
        if (
            source_vertex.feature_type != TopologyFeatureType.VERTEX
            or destination_vertex.feature_type != TopologyFeatureType.VERTEX
        ):
            raise ValueError(
                f"Attachment {vertex_edge.attachment_id}: vertex-vertex invalide"
            )

        rematerialized = materialize_vertex_edge_attachments(
            world=world,
            element_id_src=source_element_id,
            src_edge=source_edge,
            src_anchor_vkey=world.get_vertex(
                source_element_id, source_vertex.index
            ).node_type,
            element_id_dst=destination_element_id,
            dst_edge=destination_edge,
            dst_anchor_vkey=world.get_vertex(
                destination_element_id, destination_vertex.index
            ).node_type,
            vertex_edge_attachment_id=vertex_edge.attachment_id,
            vertex_vertex_attachment_id=vertex_vertex.attachment_id,
            vertex_edge_source=vertex_edge.source,
            vertex_vertex_source=vertex_vertex.source,
        )
        replacements[vertex_edge.attachment_id] = rematerialized[0]
        replacements[vertex_vertex.attachment_id] = rematerialized[1]

    return [
        replacements.get(attachment.attachment_id, attachment)
        for attachment in attachments
    ]


def buildEdgeChoiceEptsFromBest(
    best,
    *,
    world,
    mob_idx: int,
    tgt_idx: int,
    mob_tids: list,
    tgt_tids: list,
    last_drawn: list,
    eps_world: float,
    mATmpId: str,
    tATmpId: str,
    debug: bool = False,
):
    if not best:
        return None

    if last_drawn is None or len(last_drawn) == 0:
        raise ValueError("buildEdgeChoiceEptsFromBest: last_drawn manquant")

    (mA, mB), (tA, tB) = best[1], best[2]

    src_owner_tid, src_edge = _find_owner_edge_for_segment(mob_tids, mA, mB, eps_world, last_drawn)
    dst_owner_tid, dst_edge = _find_owner_edge_for_segment(tgt_tids, tA, tB, eps_world, last_drawn)

    if src_owner_tid is None:
        src_owner_tid = mob_idx
    if dst_owner_tid is None:
        dst_owner_tid = tgt_idx

    src_owner_tid = int(src_owner_tid)
    dst_owner_tid = int(dst_owner_tid)
    tri_src = last_drawn[src_owner_tid]
    tri_dst = last_drawn[dst_owner_tid]
    if not isinstance(tri_src, dict) or not isinstance(tri_dst, dict):
        raise ValueError("buildEdgeChoiceEptsFromBest: tri_src/tri_dst invalides")

    elementIdSrc = tri_src.get("topoElementId", None)
    elementIdDst = tri_dst.get("topoElementId", None)
    if not elementIdSrc or not elementIdDst:
        raise ValueError("buildEdgeChoiceEptsFromBest: topoElementId manquant (src/dst)")

    # On récupère le noeud équivalent dans le bon référentiel de triangle
    def getEquivalentNodeInElement(world, nodeIdSrc: str, elementId: str) -> str | None:
        for nid in world.node_members(nodeIdSrc):
            eid, _ = world._parseElementAndVertexIndexFromNodeId(nid)
            if eid == elementId:
                return nid
        return None
    mAId = getEquivalentNodeInElement(world, mATmpId, elementIdSrc)
    tAId = getEquivalentNodeInElement(world, tATmpId, elementIdDst)
    if mAId is None or tAId is None:
        if debug:
            print(
                "[EDGECHOICE] candidate rejected: concept anchor has no physical "
                "vertex occurrence in boundary owner element "
                f"mATmpId={mATmpId!r} elementIdSrc={elementIdSrc!r} mAId={mAId!r} "
                f"tATmpId={tATmpId!r} elementIdDst={elementIdDst!r} tAId={tAId!r}"
            )
        return None

    src_anchor_vkey = world.getNodeType(mAId)
    dst_anchor_vkey = world.getNodeType(tAId)

    Psrc = tri_src.get("pts", None)
    Pdst = tri_dst.get("pts", None)
    if Psrc is None or Pdst is None:
        raise ValueError(
            f"Topo pts manquants (src_tid={src_owner_tid}, dst_tid={dst_owner_tid}, src_edge={src_edge}, dst_edge={dst_edge})"
        )
    if not all(k in Psrc for k in ("O", "B", "L")) or not all(k in Pdst for k in ("O", "B", "L")):
        raise ValueError(
            f"Topo pts incomplets (src_tid={src_owner_tid}, dst_tid={dst_owner_tid}, src_edge={src_edge}, dst_edge={dst_edge})"
        )

    edge_code_set = {"OB", "BL", "LO"}
    src_edge = str(src_edge or "").upper().strip()
    dst_edge = str(dst_edge or "").upper().strip()

    if src_edge not in edge_code_set or dst_edge not in edge_code_set:
        raise ValueError("buildEdgeChoiceEptsFromBest: src_edge/dst_edge invalide")

    src_edge_labels = _edge_labels_for(src_owner_tid, src_edge, last_drawn)
    dst_edge_labels = _edge_labels_for(dst_owner_tid, dst_edge, last_drawn)

    kind = "vertex-edge"
    if src_edge_labels and dst_edge_labels:
        if (src_edge_labels[0] == dst_edge_labels[1]) and (src_edge_labels[1] == dst_edge_labels[0]):
            kind = "edge-edge"
        if (src_edge_labels[0] == dst_edge_labels[0]) and (src_edge_labels[1] == dst_edge_labels[1]):
            kind = "edge-edge"
    if kind not in ("edge-edge", "vertex-edge"):
        raise ValueError("buildEdgeChoiceEptsFromBest: kind inattendu")

    sv0, sv1 = _edge_code_to_vkeys(src_edge)
    dv0, dv1 = _edge_code_to_vkeys(dst_edge)
    src_vkey_at_mA = src_vkey_at_mB = None
    dst_vkey_at_tA = dst_vkey_at_tB = None

    if kind == "vertex-edge":
        if not (sv0 and sv1 and dv0 and dv1):
            raise ValueError("buildEdgeChoiceEptsFromBest: vkeys manquants")
        if src_anchor_vkey not in (sv0, sv1):
            raise ValueError("buildEdgeChoiceEptsFromBest: vkeys manquants")
        if dst_anchor_vkey not in (dv0, dv1):
            raise ValueError("buildEdgeChoiceEptsFromBest: vkeys manquants")
        src_vkey_at_mA = src_anchor_vkey
        dst_vkey_at_tA = dst_anchor_vkey
        src_vkey_at_mB = sv1 if src_vkey_at_mA == sv0 else sv0
        dst_vkey_at_tB = dv1 if dst_vkey_at_tA == dv0 else dv0
    else:
        if sv0 and sv1 and dv0 and dv1:
            (srcA, srcB), (dstA, dstB) = _assign_edge_endpoints_by_cross_distance(src_edge, dst_edge, Psrc, Pdst)
            src_vkey_at_mA = srcA
            src_vkey_at_mB = srcB
            dst_vkey_at_tA = dstA
            dst_vkey_at_tB = dstB
        if not (src_vkey_at_mA and src_vkey_at_mB and dst_vkey_at_tA and dst_vkey_at_tB):
            raise ValueError("buildEdgeChoiceEptsFromBest: vkeys manquants")

        mapping_direct = (src_vkey_at_mA == dst_vkey_at_tA) and (src_vkey_at_mB == dst_vkey_at_tB)
        mapping_reverse = (src_vkey_at_mA == dst_vkey_at_tB) and (src_vkey_at_mB == dst_vkey_at_tA)
        if not (mapping_direct or mapping_reverse):
            raise ValueError("buildEdgeChoiceEptsFromBest: mapping indeterminable")

    t_raw = None
    if kind == "vertex-edge":
        if not (src_vkey_at_mB and dst_vkey_at_tB):
            raise ValueError("buildEdgeChoiceEptsFromBest: vkeys manquants")
        if src_vkey_at_mB not in Psrc or dst_vkey_at_tB not in Pdst:
            raise ValueError("buildEdgeChoiceEptsFromBest: vkeys manquants")
        mB_edge_vertex = Psrc[src_vkey_at_mB]
        tB_edge_vertex = Pdst[dst_vkey_at_tB]
        t_raw = _compute_t_by_edge_ratio(mA, mB_edge_vertex, tA, tB_edge_vertex)

        mB_edge_vertex_out = mB_edge_vertex
        tB_edge_vertex_out = tB_edge_vertex
    else:
        mB_edge_vertex_out = mB
        tB_edge_vertex_out = tB
    if kind == "vertex-edge" and t_raw is None:
        raise ValueError("buildEdgeChoiceEptsFromBest: tRaw manquant")

    epts = EdgeChoiceEpts(
        tuple(mA), tuple(mB_edge_vertex_out), tuple(tA), tuple(tB_edge_vertex_out),
        src_owner_tid=src_owner_tid, src_edge=src_edge,
        dst_owner_tid=dst_owner_tid, dst_edge=dst_edge,
        src_vkey_at_mA=src_vkey_at_mA, src_vkey_at_mB=src_vkey_at_mB,
        dst_vkey_at_tA=dst_vkey_at_tA, dst_vkey_at_tB=dst_vkey_at_tB,
        src_edge_labels=src_edge_labels, dst_edge_labels=dst_edge_labels,
        kind=kind, t_raw=t_raw,
        elementIdSrc=elementIdSrc, elementIdDst=elementIdDst
    )

    meta = {
        "src_owner_tid": src_owner_tid,
        "dst_owner_tid": dst_owner_tid,
        "src_edge": src_edge,
        "dst_edge": dst_edge,
        "kind": kind,
    }

    return (epts, meta)


def buildEdgeChoiceEptsForAutoChain(
    *,
    world,
    mobile_entry,
    destination_entry,
    src_edge: str,      # "LO" ou "BL" (côté mobile)
    dst_edge: str,      # "LO" ou "BL" (côté dest)
    src_vkey: str = "L",
    dst_vkey: str = "L",
    kind: str = "vertex-edge",
    debug: bool = False,
):
    def _require_projection_entry(role: str, entry):
        required = ("triangleId", "points", "topologyElementId")
        missing = [name for name in required if not hasattr(entry, name)]
        if missing:
            raise ValueError(
                f"buildEdgeChoiceEptsForAutoChain: entrée {role} invalide "
                f"(attributs manquants: {', '.join(missing)})"
            )
        if not isinstance(entry.points, dict):
            raise ValueError(f"buildEdgeChoiceEptsForAutoChain: entrée {role} invalide (pts attendu)")
        if entry.triangleId is None or not entry.topologyElementId:
            raise ValueError(f"buildEdgeChoiceEptsForAutoChain: entrée {role} invalide (identité absente)")
        return entry

    # --- helpers ---
    def _other_vkey_on_edge(edge_code: str, vkey_anchor: str) -> str:
        e = str(edge_code).upper()
        if e == "LO":
            return "O" if vkey_anchor == "L" else "L"
        if e == "BL":
            return "B" if vkey_anchor == "L" else "L"
        if e == "OB":
            return "B" if vkey_anchor == "O" else "O"
        raise ValueError(f"edge_code invalide: {edge_code}")

    mob = _require_projection_entry("mobile", mobile_entry)
    dst = _require_projection_entry("destination", destination_entry)

    elementIdSrc = mob.topologyElementId
    elementIdDst = dst.topologyElementId
    src_owner_tid = mob.triangleId
    dst_owner_tid = dst.triangleId

    Pm = mob.points
    Pt = dst.points

    # Ancre = L des deux côtés (Phase 3)
    mA = Pm[src_vkey]
    tA_v = dst_vkey
    tA = Pt[tA_v]

    # Endpoint "edge vertex" : l’autre sommet de l’arête
    mB_v = _other_vkey_on_edge(src_edge, src_vkey)
    tB_v = _other_vkey_on_edge(dst_edge, dst_vkey)

    mBEdgeVertex = Pm[mB_v]
    tBEdgeVertex = Pt[tB_v]

    # Cas auto actuel : on attache l’endpoint mobile (O ou B) sur l’arête dest (LO ou BL)
    # avec edgeFrom = dst_vkey (L). => t=1 (endpoint opposé).
    edge_code_to_index = {"OB": 0, "BL": 1, "LO": 2}

    el_src = world.elements[elementIdSrc]
    el_dst = world.elements[elementIdDst]

    i_src = edge_code_to_index[src_edge.upper()]
    i_dst = edge_code_to_index[dst_edge.upper()]

    L_src = float(el_src.edge_lengths_km[i_src])
    L_dst = float(el_dst.edge_lengths_km[i_dst])
    if L_dst <= 1e-12:
        raise ValueError("buildEdgeChoiceEptsForAutoChain: arête cible dégénérée")

    t_raw = L_src / L_dst

    epts = EdgeChoiceEpts(
        mA, mBEdgeVertex,
        tA, tBEdgeVertex,
        src_owner_tid=src_owner_tid, src_edge=src_edge,
        dst_owner_tid=dst_owner_tid, dst_edge=dst_edge,
        src_vkey_at_mA=src_vkey, src_vkey_at_mB=mB_v,
        dst_vkey_at_tA=dst_vkey, dst_vkey_at_tB=tB_v,
        kind=kind,
        t_raw=t_raw,
        elementIdSrc=elementIdSrc,
        elementIdDst=elementIdDst,
    )

    meta = {
        "kind": kind,
        "src_edge": src_edge,
        "dst_edge": dst_edge,
        "src_owner_tid": src_owner_tid,
        "dst_owner_tid": dst_owner_tid,
        "src_vkey_at_mA": src_vkey,
        "src_vkey_at_mB": mB_v,
        "dst_vkey_at_tA": dst_vkey,
        "dst_vkey_at_tB": tB_v,
        "tRaw": t_raw,
    }

    return (epts, meta) if debug else (epts, meta)
