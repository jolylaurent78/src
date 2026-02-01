# assembleur_debug.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons




def listTopoGroups(topoWorld, rebuild: bool = True):
    """
    Debug: liste les groupes d'un TopologyWorld.

    rebuild=True force la reconstruction des listes group.element_ids à partir
    de topoWorld.element_to_group (sinon elles peuvent être vides même si le world est OK).

    USAGE:
    from assembleur_debug import listTopoGroups
    listTopoGroups(topoWorld_prev)

    """
    if rebuild and hasattr(topoWorld, "rebuildGroupElementLists"):
        topoWorld.rebuildGroupElementLists()

    print("=== TopologyWorld Groups ===")
    if not getattr(topoWorld, "groups", None):
        print("  (no groups)")
        return

    for gid in sorted(topoWorld.groups.keys()):
        gidc = topoWorld.find_group(gid)

        g = topoWorld.groups.get(gidc)
        if g is None:
            continue

        nb_elems = len(getattr(g, "element_ids", []))
        nb_att = len(getattr(g, "attachment_ids", []))

        # concept cache (peut être stale si graphValid=False, mais utile)
        try:
            c = topoWorld._concept_cache(gidc)
            nb_nodes = len(getattr(c, "nodes", {}))
            graph_ok = getattr(c, "graphValid", None)
        except Exception:
            nb_nodes = "?"
            graph_ok = None

        print(f"- {gidc} | elements={nb_elems} | attachments={nb_att} | conceptNodes={nb_nodes} | graphValid={graph_ok}")


def plotLastDrawn(lastDrawn: List[Dict[str, Any]], showIds: bool = True, ax=None):
    """
    Trace un last_drawn (liste de triangles) avec matplotlib.

    Usage
    from assembleur_debug import plotLastDrawn
    plotLastDrawn(last_drawn_base)
    
    Attendu par item:
      - d["pts"]["O"], d["pts"]["B"], d["pts"]["L"] : (x,y)
      - d["id"] optionnel
      - d["topoElementId"] optionnel
    """
    created = False
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        created = True
    else:
        ax.set_aspect("equal", adjustable="box")

    artists = []

    for d in lastDrawn:
        pts = d["pts"]
        O = np.asarray(pts["O"], dtype=float)
        B = np.asarray(pts["B"], dtype=float)
        L = np.asarray(pts["L"], dtype=float)

        poly = np.vstack([O, B, L, O])
        line, = ax.plot(poly[:, 0], poly[:, 1], linewidth=2)
        artists.append(line)

        sc = ax.scatter([O[0], B[0], L[0]], [O[1], B[1], L[1]], s=20)
        artists.append(sc)

        if showIds:
            c = (O + B + L) / 3.0
            label = d.get("topoElementId") or f'T{d.get("id","?")}'
            txt = ax.text(c[0], c[1], label, fontsize=8)
            artists.append(txt)

    ax.grid(True)
    return artists



def plotTopoWorldBoundaries(topoWorld, groupIds=None, showNodeIds=False, ax=None):
    """
    Usage
    from assembleur_debug import plotTopoWorldBoundaries
    plotTopoWorldBoundaries(topoWorld_prev)

    Ouvre une fenêtre matplotlib interactive avec les boundaries des groupes.
    Nécessite que le programme continue à tourner (ou au moins un plt.pause()).
    """
    created = False
    if ax is None:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        created = True
    else:
        ax.set_aspect("equal", adjustable="box")
    
    artists = []

    gids = groupIds
    if gids is None:
        gids = list(getattr(topoWorld, "groups", {}).keys())

    for gid in gids:
        gid = topoWorld.find_group(str(gid))

        topoWorld.ensureConceptGeom(gid)
        topoWorld.computeBoundary(gid)

        c = topoWorld._concept_cache(gid)
        cycle = c.boundaryCycle or []
        if len(cycle) < 3:
            continue

        pts = []
        for cn in cycle:
            x, y = topoWorld.getConceptNodeWorldXY(str(cn), gid)
            pts.append(np.array([float(x), float(y)], dtype=float))

        pts_closed = pts + [pts[0]]
        X = [p[0] for p in pts_closed]
        Y = [p[1] for p in pts_closed]
        line, = ax.plot(X, Y, linewidth=2)
        artists.append(line)

        if showNodeIds:
            for cn, p in zip(cycle, pts):
                txt = ax.text(p[0], p[1], str(cn), fontsize=7)
                artists.append(txt)

        cx = float(sum(p[0] for p in pts) / len(pts))
        cy = float(sum(p[1] for p in pts) / len(pts))
        txtg = ax.text(cx, cy, str(gid), fontsize=9)
        artists.append(txtg)

    ax.grid(True)
    if created:
        plt.show()
        return fig, ax, artists
    return None, ax, artists


def plotConceptGraph(topoWorld, groupId: str, ax=None, showNodeIds: bool = True):
    """
    Trace le graphe concept d'un groupe:
      - ConceptNodes: position monde via getConceptNodeWorldXY
      - ConceptEdges: segments entre nodes (a->b)

    Usage
    from assembleur_debug import plotConceptGraph
    plotConceptGraph(topoWorld_prev, "SA_AUTO:G010")    

        Retour:
      - (fig, ax, artists)
    """
    created = False
    if ax is None:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        created = True
    else:
        fig = ax.figure
        ax.set_aspect("equal", adjustable="box")

    artists = []

    gid = topoWorld.find_group(str(groupId))

    # important: construit le cache concept (nodes + edges)
    c = topoWorld.ensureConceptGraph(gid)

    # ---- Nodes (positions monde)
    nodePos = {}
    for nid in c.nodes.keys():
        x, y = topoWorld.getConceptNodeWorldXY(str(nid), gid)
        nodePos[str(nid)] = (float(x), float(y))

    xs = [p[0] for p in nodePos.values()]
    ys = [p[1] for p in nodePos.values()]
    sc = ax.scatter(xs, ys, s=25)
    artists.append(sc)

    if showNodeIds:
        for nid, (x, y) in nodePos.items():
            # label compact: "T09:N1" plutôt que "SA_AUTO:T09:N1"
            short = nid.split(":")[-2] + ":" + nid.split(":")[-1] if ":" in nid else nid
            txt = ax.text(x, y, short, fontsize=7)
            artists.append(txt)

    # ---- Edges
    for (_a, _b), e in c.edges.items():
        a = str(e.a)
        b = str(e.b)
        if a not in nodePos or b not in nodePos:
            continue
        x1, y1 = nodePos[a]
        x2, y2 = nodePos[b]
        line, = ax.plot([x1, x2], [y1, y2], linewidth=1)
        artists.append(line)

    ax.grid(True)

    if created:
        plt.show(block=False)
        plt.pause(0.05)

    return fig, ax, artists



def plotScenarioWithToggles(
    lastDrawn,
    topoWorld,
    groupIds=None,
    showIds=True,
    showNodeIds=False,
):
    """
    USAGE
    from assembleur_debug import plotScenarioWithToggles
    plotScenarioWithToggles(last_drawn_base, topoWorld_prev)

    Console friendly:
      from assembleur_debug import plotScenarioWithToggles
      plotScenarioWithToggles(last_drawn_base, topoWorld_prev)
    """
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")

    triArtists = plotLastDrawn(lastDrawn, showIds=showIds, ax=ax)

    _, _, bndArtists = plotTopoWorldBoundaries(
        topoWorld, groupIds=groupIds, showNodeIds=showNodeIds, ax=ax
    )

    # Toggle panel
    rax = fig.add_axes([0.01, 0.75, 0.22, 0.20])
    checks = CheckButtons(rax, ["Triangles", "Boundaries"], [True, True])

    def setVisible(artists, vis):
        for a in artists:
            a.set_visible(vis)

    def onToggle(label):
        triOn, bndOn = checks.get_status()
        setVisible(triArtists, triOn)
        setVisible(bndArtists, bndOn)
        fig.canvas.draw_idle()

    checks.on_clicked(onToggle)

    ax.grid(True)

    # Important en mode debug: force un rendu initial
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)
    plt.pause(0.05)

    return fig, ax, triArtists, bndArtists

"""
sid = "SA_AUTO"          # ou topoScenarioId réel si tu l’as
out_path = r"C:\temp\TopoDump_SA_AUTO_debug.xml"  # ou un chemin existant
topoWorld_prev.export_topo_dump_xml(sid, out_path, orientation="cw")

"""