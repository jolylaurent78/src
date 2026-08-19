from dataclasses import dataclass

# === Modules externalisés (découpage maintenable) ===
from src.assembleur_core import (
    TopologyAttachmentResolutionError,
    TopologyAttachmentValidationError,
    TopologyConstraintGeometryError,
    TopologyEdgeEdgeAttachment,
    TopologyVertexEdgeAttachment,
    compute_vertex_edge_attachment_orientation,
)


@dataclass(frozen=True)
class ManualAttachmentIntent:
    """Choix manuel exprimé uniquement avec des identités métier Core.

    La géométrie effective du raccord reste délibérément hors de cet objet :
    elle sera résolue par le Core lors des étapes de preview puis de commit.
    """

    kind: str
    mob_element_id: str
    mob_vertex: str
    mob_edge: str
    dest_element_id: str
    dest_vertex: str
    dest_edge: str


@dataclass(frozen=True)
class ManualAttachmentPreview:
    """Résultat éphémère de l'assemblage manuel calculé sur un clone Core."""

    accepted: bool
    attachment: TopologyVertexEdgeAttachment | TopologyEdgeEdgeAttachment
    world: object | None
    rejection_reason: str | None = None


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


def buildManualAttachmentIntentFromBest(
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
) -> ManualAttachmentIntent | None:
    """Traduit le candidat Boundary du drag en intention métier légère.

    Les indices Canvas ne servent qu'à retrouver immédiatement les éléments
    propriétaires des segments. Ils ne sortent jamais de cette fonction.
    """
    if not best:
        return None
    if last_drawn is None or len(last_drawn) == 0:
        raise ValueError("buildManualAttachmentIntentFromBest: last_drawn manquant")

    (mA, mB), (tA, tB) = best[1], best[2]
    src_owner_tid, src_edge = _find_owner_edge_for_segment(
        mob_tids, mA, mB, eps_world, last_drawn
    )
    dst_owner_tid, dst_edge = _find_owner_edge_for_segment(
        tgt_tids, tA, tB, eps_world, last_drawn
    )
    if src_owner_tid is None:
        src_owner_tid = mob_idx
    if dst_owner_tid is None:
        dst_owner_tid = tgt_idx

    tri_src = last_drawn[int(src_owner_tid)]
    tri_dst = last_drawn[int(dst_owner_tid)]
    if not isinstance(tri_src, dict) or not isinstance(tri_dst, dict):
        raise ValueError(
            "buildManualAttachmentIntentFromBest: propriétaires Boundary invalides"
        )
    mob_element_id = str(tri_src.get("topoElementId", "") or "").strip()
    dest_element_id = str(tri_dst.get("topoElementId", "") or "").strip()
    if not mob_element_id or not dest_element_id:
        raise ValueError(
            "buildManualAttachmentIntentFromBest: topoElementId manquant"
        )

    def _equivalent_node(node_id: str, element_id: str) -> str | None:
        for candidate_node_id in world.node_members(node_id):
            candidate_element_id, _ = world._parseElementAndVertexIndexFromNodeId(
                candidate_node_id
            )
            if candidate_element_id == element_id:
                return candidate_node_id
        return None

    mob_anchor_node_id = _equivalent_node(mATmpId, mob_element_id)
    dest_anchor_node_id = _equivalent_node(tATmpId, dest_element_id)
    if mob_anchor_node_id is None or dest_anchor_node_id is None:
        return None

    mob_vertex = str(world.getNodeType(mob_anchor_node_id) or "").upper().strip()
    dest_vertex = str(world.getNodeType(dest_anchor_node_id) or "").upper().strip()
    mob_edge = str(src_edge or "").upper().strip()
    dest_edge = str(dst_edge or "").upper().strip()
    if mob_vertex not in {"O", "B", "L"} or dest_vertex not in {"O", "B", "L"}:
        raise ValueError("buildManualAttachmentIntentFromBest: vertex métier invalide")
    if mob_edge not in {"OB", "BL", "LO"} or dest_edge not in {"OB", "BL", "LO"}:
        raise ValueError("buildManualAttachmentIntentFromBest: edge métier invalide")

    kind = (
        "edge-edge"
        if world.are_same_business_edge(
            mob_element_id,
            mob_edge,
            dest_element_id,
            dest_edge,
        )
        else "vertex-edge"
    )

    intent = ManualAttachmentIntent(
        kind=kind,
        mob_element_id=mob_element_id,
        mob_vertex=mob_vertex,
        mob_edge=mob_edge,
        dest_element_id=dest_element_id,
        dest_vertex=dest_vertex,
        dest_edge=dest_edge,
    )
    return intent


def buildTopologyAttachmentFromManualIntent(
    intent: ManualAttachmentIntent,
    *,
    attachment_id: str,
    world,
) -> TopologyVertexEdgeAttachment | TopologyEdgeEdgeAttachment:
    """Matérialise une intention manuelle sans aucune résolution géométrique."""
    if not isinstance(intent, ManualAttachmentIntent):
        raise TypeError("buildTopologyAttachmentFromManualIntent: intent invalide")
    normalized_id = str(attachment_id or "").strip()
    if not normalized_id:
        raise ValueError("buildTopologyAttachmentFromManualIntent: attachment_id vide")
    if intent.kind == "vertex-edge":
        return TopologyVertexEdgeAttachment(
            attachment_id=normalized_id,
            mob_element_id=intent.mob_element_id,
            mob_vertex=intent.mob_vertex,
            creation_mob_edge=intent.mob_edge,
            dest_element_id=intent.dest_element_id,
            dest_vertex=intent.dest_vertex,
            creation_dest_edge=intent.dest_edge,
            mob_orientation=compute_vertex_edge_attachment_orientation(
                world, intent.mob_element_id, intent.mob_vertex, intent.mob_edge
            ),
            dest_orientation=compute_vertex_edge_attachment_orientation(
                world, intent.dest_element_id, intent.dest_vertex, intent.dest_edge
            ),
        )
    if intent.kind == "edge-edge":
        return TopologyEdgeEdgeAttachment(
            attachment_id=normalized_id,
            mob_element_id=intent.mob_element_id,
            mob_edge=intent.mob_edge,
            dest_element_id=intent.dest_element_id,
            dest_edge=intent.dest_edge,
        )
    raise ValueError(
        "buildTopologyAttachmentFromManualIntent: kind inattendu "
        f"{intent.kind!r}"
    )


def previewManualAttachment(
    real_world,
    intent: ManualAttachmentIntent,
) -> ManualAttachmentPreview:
    """Prévisualise une intention via le pipeline V2 sur un clone du Core."""
    preview_world = real_world.clonePhysicalState()
    attachment_id = "PREVIEW_MANUAL"
    suffix = 1
    while attachment_id in preview_world.attachments:
        attachment_id = f"PREVIEW_MANUAL_{suffix}"
        suffix += 1
    attachment = buildTopologyAttachmentFromManualIntent(
        intent,
        attachment_id=attachment_id,
        world=real_world,
    )

    mob_group_id = preview_world.get_group_of_element(attachment.mob_element_id)
    dest_group_id = preview_world.get_group_of_element(attachment.dest_element_id)
    if mob_group_id == dest_group_id:
        return ManualAttachmentPreview(
            accepted=False,
            attachment=attachment,
            world=None,
            rejection_reason="Les éléments mobile et destination appartiennent déjà au même groupe.",
        )
    try:
        overlap = preview_world.simulate_topological_overlap(
            dest_group_id,
            mob_group_id,
            attachment,
        )
    except (
        TopologyAttachmentValidationError,
        TopologyAttachmentResolutionError,
        TopologyConstraintGeometryError,
    ) as exc:
        return ManualAttachmentPreview(
            accepted=False,
            attachment=attachment,
            world=None,
            rejection_reason=str(exc),
        )

    if overlap:
        return ManualAttachmentPreview(
            accepted=False,
            attachment=attachment,
            world=None,
            rejection_reason="Chevauchement géométrique incompatible.",
        )

    try:
        merged_group_id = preview_world.apply_attachment(attachment)
        preview_world.getResolvedAttachment(attachment.attachment_id)
        preview_world.replay_group_attachment_poses(
            merged_group_id,
            intent.dest_element_id,
        )
    except (
        TopologyAttachmentValidationError,
        TopologyAttachmentResolutionError,
        TopologyConstraintGeometryError,
    ) as exc:
        return ManualAttachmentPreview(
            accepted=False,
            attachment=attachment,
            world=None,
            rejection_reason=str(exc),
        )

    return ManualAttachmentPreview(
        accepted=True,
        attachment=attachment,
        world=preview_world,
    )


def commitManualAttachment(
    real_world,
    intent: ManualAttachmentIntent,
) -> tuple[TopologyVertexEdgeAttachment | TopologyEdgeEdgeAttachment, str]:
    """Commit la même intention que la preview dans le vrai World Core.

    La preview acceptée garantit que cette application déterministe a été
    validée sur un clone physique identique. Aucun état du clone n'est copié
    ou installé dans ``real_world``.
    """
    attachment = buildTopologyAttachmentFromManualIntent(
        intent,
        attachment_id=real_world.new_attachment_id(),
        world=real_world,
    )
    merged_group_id = real_world.apply_attachment(attachment)
    real_world.replay_group_attachment_poses(
        merged_group_id,
        intent.dest_element_id,
    )
    return (attachment, merged_group_id)
