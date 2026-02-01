"""
TriangleViewerFrontierGraphMixin

Ce module est généré pour découper assembleur_tk.py.
"""

from __future__ import annotations
import os, re, math, json, copy
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from tksheet import Sheet

EPS_WORLD = 1e-6

class TriangleViewerFrontierGraphMixin:
    """Mixin: méthodes extraites de assembleur_tk.py."""
    pass

    def _edge_dir(self, a, b):
        return (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


    def _ang_wrap(self, x):
        import math
        # wrap sur [-pi, pi]
        x = (x + math.pi) % (2*math.pi) - math.pi
        return x


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


    # ---------- CHEMIN : lissage du boundary graph (post-traitement) ----------

    def _smooth_boundary_graph_for_chemin(self, graph, epsSnap: float):
        """
        Lisse un boundary graph *après* _build_boundary_graph(), sans modifier la fonction partagée.
        Objectif: fusionner des sommets quasi-identiques (pollution shapely) pour restaurer un cycle (degré=2).

        - graph: {"adj": {k:[pt,...]}, "pts": {k:(x,y)}}
        - epsSnap: tolérance monde pour fusion (typiquement 1e-5..1e-4 dans tes cas)
        """
        if not graph or "pts" not in graph or "adj" not in graph:
            return graph
        eps = float(epsSnap) if epsSnap is not None else 0.0
        if eps <= 0.0:
            return graph

        pts_map = graph.get("pts", {}) or {}
        adj = graph.get("adj", {}) or {}
        if len(pts_map) <= 1:
            return graph

        keys = list(pts_map.keys())
        coords = {k: (float(pts_map[k][0]), float(pts_map[k][1])) for k in keys}
        e2 = eps * eps

        # --- bucketing sur grille eps (réduit drastiquement les comparaisons) ---
        def _cell(p):
            return (int(math.floor(float(p[0]) / eps)), int(math.floor(float(p[1]) / eps)))

        buckets = {}
        for k in keys:
            c = _cell(coords[k])
            buckets.setdefault(c, []).append(k)

        # --- union-find minimal ---
        parent = {k: k for k in keys}

        def _find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def _union(a, b):
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        # compare dans cellule + voisines
        neighCells = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        for (cx, cy), lst in buckets.items():
            if not lst:
                continue
            # comparer la liste avec celles des cellules voisines (évite doublons massifs)
            for dx, dy in neighCells:
                lst2 = buckets.get((cx + dx, cy + dy))
                if not lst2:
                    continue
                for a in lst:
                    pa = coords[a]
                    for b in lst2:
                        if a == b:
                            continue
                        pb = coords[b]
                        dx2 = pa[0] - pb[0]
                        dy2 = pa[1] - pb[1]
                        if (dx2*dx2 + dy2*dy2) <= e2:
                            _union(a, b)

        # --- construire clusters -> point canonique (centroïde) ---
        clusters = {}
        for k in keys:
            r = _find(k)
            clusters.setdefault(r, []).append(k)

        oldToNew = {}
        newPts = {}
        for r, lst in clusters.items():
            # centroïde monde
            sx = 0.0; sy = 0.0
            for k in lst:
                sx += coords[k][0]
                sy += coords[k][1]
            cx = sx / float(len(lst))
            cy = sy / float(len(lst))
            newKey = self._pt_key_eps((cx, cy), eps=eps)
            newPts.setdefault(newKey, (float(cx), float(cy)))
            for k in lst:
                oldToNew[k] = newKey

        # --- rebuild adj (en dédupliquant) ---
        newAdj = {}
        newAdjKeys = {}
        for ku, neighPts in (adj or {}).items():
            if ku not in oldToNew:
                continue
            nu = oldToNew[ku]
            newAdj.setdefault(nu, [])
            newAdjKeys.setdefault(nu, set())
            for vpt in (neighPts or []):
                # retrouver la key d'origine du voisin (même granularité que _build_boundary_graph)
                kv = self._pt_key_eps(vpt)
                if kv not in oldToNew:
                    continue
                nv = oldToNew[kv]
                if nv == nu:
                    continue
                if nv in newAdjKeys[nu]:
                    continue
                newAdjKeys[nu].add(nv)
                newAdj[nu].append(newPts.get(nv, (float(vpt[0]), float(vpt[1]))))

        return {"adj": newAdj, "pts": newPts}


    def _incident_half_edges_at_vertex(self, graph, v, eps=EPS_WORLD):
        """
        Renvoie les demi-arêtes qui partent de 'v' le long de la frontière.
        Retour: liste (0..2) éléments, chaque élément est ((ax,ay),(bx,by))
        NOTE: on travaille en coordonnées *canoniques* du graphe (graph["pts"]).
        """
        key = self._pt_key_eps(v, eps=eps)

        # point canonique du sommet (clé quantifiée -> coord monde "réelle")
        a = graph.get("pts", {}).get(key, (float(v[0]), float(v[1])))

        neighs = graph.get("adj", {}).get(key, [])
        out = []
        pts_map = graph.get("pts", {})

        for w in neighs:
            wk = self._pt_key_eps(w, eps=eps)
            b = pts_map.get(wk, (float(w[0]), float(w[1])))
            out.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))

        # garder au maximum 2 ; si plus, prendre 2 extrêmes en angle autour de a
        if len(out) > 2:
            import math
            def ang(e):
                (p, q) = e
                vx, vy = self._edge_dir(p, q)
                return math.atan2(vy, vx)

            out = sorted(out, key=ang)
            a0 = out[0]; ang0 = ang(a0)
            a1 = max(out[1:], key=lambda e: self._ang_diff(ang(e), ang0))
            out = [a0, a1]

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
         


