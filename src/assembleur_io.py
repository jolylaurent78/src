"""assembleur_io.py
Persistance (config JSON + scénario XML) isolée du GUI.
Les fonctions prennent 'viewer' en paramètre (duck-typing) pour éviter les imports circulaires.
"""

import os
import json
import datetime as _dt
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET

import numpy as np

def loadAppConfig(viewer):
    """Charge la config JSON (best-effort)."""
    viewer.appConfig = {}
    try:
        path = getattr(viewer, "config_path", "")
        if not path:
            return
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            viewer.appConfig = data
    except Exception:
        # Jamais bloquant : si la config est corrompue on repart de zéro.
        viewer.appConfig = {}

def saveAppConfig(viewer):
    """Sauvegarde la config JSON (best-effort)."""
    try:
        path = getattr(viewer, "config_path", "")
        if not path:
            return
        cfg_dir = os.path.dirname(path)
        if cfg_dir:
            os.makedirs(cfg_dir, exist_ok=True)
        # écriture atomique (évite un fichier vide si un souci survient)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(getattr(viewer, "appConfig", {}) or {}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass

def getAppConfigValue(viewer, key: str, default=None):
    try:
        return (getattr(viewer, "appConfig", {}) or {}).get(key, default)
    except Exception:
        return default

def setAppConfigValue(viewer, key: str, value):
    try:
        if not hasattr(viewer, "appConfig") or viewer.appConfig is None:
            viewer.appConfig = {}
        viewer.appConfig[key] = value
        viewer.saveAppConfig()
    except Exception:
        pass

def saveScenarioXml(viewer, path: str):
    """
    Sauvegarde 'robuste' (v1) :
      - source excel, view (zoom/offset), clock (pos + hm),
      - ids restants (listbox),
      - triangles affichés (id, mirrored, points monde O/B/L, group),
      - groupes (membres best-effort),
      - associations triangle→mot (word,row,col).
    """
    root = ET.Element("scenario", {
        "version": "1",
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds")
    })
    # source
    ET.SubElement(root, "source", {
        "excel": os.path.abspath(viewer.excel_path) if getattr(viewer, "excel_path", None) else ""
    })
    # view
    view = ET.SubElement(root, "view", {
        "zoom": f"{float(getattr(viewer, 'zoom', 1.0)):.6g}",
        "offset_x": f"{float(viewer.offset[0]) if hasattr(viewer,'offset') else 0.0:.6g}",
        "offset_y": f"{float(viewer.offset[1]) if hasattr(viewer,'offset') else 0.0:.6g}",
    })
    # clock
    ET.SubElement(root, "clock", {
        "x": f"{float(viewer._clock_cx) if viewer._clock_cx is not None else 0.0:.6g}",
        "y": f"{float(viewer._clock_cy) if viewer._clock_cy is not None else 0.0:.6g}",
        "hour": f"{int(viewer._clock_state.get('hour', 0))}",
        "minute": f"{int(viewer._clock_state.get('minute', 0))}",
        "label": str(viewer._clock_state.get("label","")),
    })
    # listbox: ids restants
    lb = ET.SubElement(root, "listbox")
    try:
        for i in range(viewer.listbox.size()):
            txt = viewer.listbox.get(i)
            m = re.match(r"\s*(\d+)\.", str(txt))
            if m:
                ET.SubElement(lb, "tri", {"id": m.group(1)})
    except Exception:
        pass
    # groupes (best-effort)
    groups_xml = ET.SubElement(root, "groups")
    try:
        for gid, gdata in (viewer.groups or {}).items():
            g_el = ET.SubElement(groups_xml, "group", {"id": str(gid)})
            # si une API _group_nodes existe, on l’utilise
            members = []
            try:
                members = [nd.get("tid") for nd in viewer._group_nodes(gid) if nd.get("tid") is not None]
            except Exception:
                # fallback: scanner _last_drawn via le champ runtime correct 'group_id'
                for idx, t in enumerate(viewer._last_drawn or []):
                    if int(t.get("group_id", 0)) == int(gid):
                        members.append(idx)
            for tid in members:
                if tid is None:
                    continue
                ET.SubElement(g_el, "member", {"tid": str(tid)})
    except Exception:
        pass
    # triangles posés
    tris_xml = ET.SubElement(root, "triangles")
    for t in (viewer._last_drawn or []):
        tri_el = ET.SubElement(tris_xml, "triangle", {
            "id": str(t.get("id", "")),
            "mirrored": "1" if t.get("mirrored", False) else "0",
            # on sérialise la valeur runtime correcte
            "group": str(t.get("group_id", 0)),
        })
        P = t.get("pts", {})
        for key in ("O","B","L"):
            if key in P:
                ET.SubElement(tri_el, key).text = viewer._pt_to_xml(P[key])
    # mots associés
    words_xml = ET.SubElement(root, "words")
    for tri_id, info in (viewer._tri_words or {}).items():
        ET.SubElement(words_xml, "w", {
            "tri_id": str(tri_id),
            "row": str(info.get("row","")),
            "col": str(info.get("col","")),
            "text": str(info.get("word","")),
        })
    # écrire
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)

def loadScenarioXml(viewer, path: str):
    """
    Recharge un scénario v1 :
      - tente de recharger le fichier Excel source,
      - restaure vue, horloge, listbox, triangles posés (+mots), groupes (best-effort),
      - redessine.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "scenario":
        raise ValueError("Fichier scénario invalide (balise racine).")
    # 1) Excel source
    src = root.find("source")
    excel = src.get("excel") if src is not None else ""
    if excel and os.path.isfile(excel):
        viewer.load_excel(excel)
    # 2) vue (restaurer AVANT toute reconstruction pour que les conversions monde<->écran soient cohérentes)
    view = root.find("view")
    if view is not None:
        try:
            viewer.zoom = float(view.get("zoom", viewer.zoom))
        except Exception: pass
        try:
            ox = float(view.get("offset_x", viewer.offset[0] if hasattr(viewer,"offset") else 0.0))
            oy = float(view.get("offset_y", viewer.offset[1] if hasattr(viewer,"offset") else 0.0))
            viewer.offset = np.array([ox, oy], dtype=float)
        except Exception:
            pass
    # État d’interaction propre (purge complète UI)
    viewer._sel = {"mode": None}
    viewer._clear_nearest_line()
    viewer._clear_edge_highlights()
    viewer._hide_tooltip()
    viewer._ctx_target_idx = None
    viewer._edge_choice = None
    viewer._drag_preview_id = None
    viewer.canvas.delete("preview")

    # 3) horloge
    clock = root.find("clock")
    if clock is not None:
        try: viewer._clock_cx = float(clock.get("x", "0"))
        except Exception: pass
        try: viewer._clock_cy = float(clock.get("y", "0"))
        except Exception: pass
        try:
            h = int(clock.get("hour", "0")); m = int(clock.get("minute", "0"))
            lbl = clock.get("label","")
            viewer._clock_state.update({"hour": h, "minute": m, "label": lbl})
        except Exception:
            pass
    # 4) listbox (ids restants)
    lb = root.find("listbox")
    if lb is not None:
        try:
            # ids des triangles encore présents dans la listbox au moment de la sauvegarde
            remain_ids = [int(e.get("id")) for e in lb.findall("tri") if e.get("id")]
            viewer.listbox.delete(0, tk.END)
            # reconstruire la liste depuis df en conservant l'ordre original
            if getattr(viewer, "df", None) is not None and not viewer.df.empty:
                if remain_ids:
                    wanted = set(remain_ids)
                    for _, r in viewer.df.iterrows():
                        tid = int(r["id"])
                        if tid in wanted:
                            viewer.listbox.insert(
                                tk.END, f"{tid:02d}. B:{r['B']}  L:{r['L']}"
                            )
                else:
                    # listbox vide dans le XML (ancien scénario ou bug) :
                    # on retombe sur le comportement par défaut = tous les triangles.
                    for _, r in viewer.df.iterrows():
                        tid = int(r["id"])
                        viewer.listbox.insert(
                            tk.END, f"{tid:02d}. B:{r['B']}  L:{r['L']}"
                        )
        except Exception:
            pass
    # 5) triangles posés (monde)
    # On reconstruit la liste en vidant d'abord celle du scénario actif,
    # sans casser la référence partagée avec scen.last_drawn.
    viewer._last_drawn.clear()
    tris_xml = root.find("triangles")
    if tris_xml is not None:
        for t_el in tris_xml.findall("triangle"):
            try:
                tid = int(t_el.get("id"))
            except Exception:
                continue
            mirrored = (t_el.get("mirrored","0") == "1")
            group_id = int(t_el.get("group","0"))
            P = {}
            for k in ("O","B","L"):
                n = t_el.find(k)
                if n is not None and n.text:
                    P[k] = viewer._xml_to_pt(n.text)
            # Ne plus écrire 'group' (inconnu du runtime) ; initialiser les champs officiels
            item = {"id": tid, "pts": P, "mirrored": mirrored}
            item["group_id"] = None
            item["group_pos"] = None
            viewer._last_drawn.append(item)

    # 5bis) compléter les 'labels' manquants (compat v1: non stockés dans le XML)
    try:
        if getattr(viewer, "df", None) is not None and not viewer.df.empty:
            # dictionnaire id -> (B, L) sous forme de chaînes
            _by_id = {int(r["id"]): (str(r["B"]), str(r["L"])) for _, r in viewer.df.iterrows()}
            for t in viewer._last_drawn:
                if "labels" not in t or not t["labels"]:
                    b, l = _by_id.get(int(t.get("id", -1)), ("", ""))
                    t["labels"] = ("Bourges", b, l)
        else:
            # DF absent : fallback neutre
            for t in viewer._last_drawn:
                if "labels" not in t or not t["labels"]:
                    t["labels"] = ("Bourges", "", "")
    except Exception:
        # sécurité : ne jamais laisser 'labels' absent
        for t in viewer._last_drawn:
            if "labels" not in t or not t["labels"]:
                t["labels"] = ("Bourges", "", "")

    # 5ter) tenir à jour les ids déjà posés (pour cohérence listbox / drag)
    try:
        viewer._placed_ids = {int(t["id"]) for t in viewer._last_drawn}
    except Exception:
        viewer._placed_ids = set()
    # Mettre à jour l'affichage de la listbox en fonction des triangles déjà posés
    viewer._update_triangle_listbox_colors()

    # 6) mots associés
    viewer._tri_words = {}
    words_xml = root.find("words")
    if words_xml is not None:
        for w in words_xml.findall("w"):
            try:
                tid = int(w.get("tri_id"))
            except Exception:
                continue
            viewer._tri_words[tid] = {
                "word": w.get("text",""),
                "row": int(w.get("row","0")) if w.get("row") else 0,
                "col": int(w.get("col","0")) if w.get("col") else 0,
            }

    # 7) groupes (reconstruction complète: nodes + bboxes)
    # Même logique : on réutilise le dict existant pour préserver
    # le lien avec manual.groups.
    viewer.groups.clear()
    groups_xml = root.find("groups")
    if groups_xml is not None:
        for g_el in groups_xml.findall("group"):
            try:
                gid = int(g_el.get("id"))
            except Exception:
                continue
            nodes = []
            for mem in g_el.findall("member"):
                # compat: on accepte 'tid' (index dans _last_drawn) OU 'id' (id triangle)
                tidx = None
                if mem.get("tid") is not None:
                    try:
                        tidx = int(mem.get("tid"))
                    except Exception:
                        tidx = None
                elif mem.get("id") is not None:
                    # recherche de l’index correspondant à l’id
                    try:
                        tri_id = int(mem.get("id"))
                        for k, t in enumerate(viewer._last_drawn):
                            if int(t.get("id")) == tri_id:
                                tidx = k
                                break
                    except Exception:
                        tidx = None
                if tidx is None or not (0 <= tidx < len(viewer._last_drawn)):
                    continue
                # Nodes au format runtime + marquage triangle: group_id/group_pos
                nodes.append({"tid": tidx, "vkey_in": None, "vkey_out": None})
                viewer._last_drawn[tidx]["group_id"]  = gid
                viewer._last_drawn[tidx]["group_pos"] = len(nodes) - 1
                # Hygiène: supprimer l’ancien champ 'group' s’il subsiste
                if "group" in viewer._last_drawn[tidx]:
                    del viewer._last_drawn[tidx]["group"]
            viewer.groups[gid] = {"id": gid, "nodes": nodes, "bbox": None}
            viewer._recompute_group_bbox(gid)

    # Nettoyage global de compatibilité : purger toute trace résiduelle de 'group'
    for _t in viewer._last_drawn:
        if "group" in _t:
            del _t["group"]
    # sécurité: prochain id de groupe
    try:
        viewer._next_group_id = (max(viewer.groups.keys()) + 1) if viewer.groups else 1
    except Exception:
        viewer._next_group_id = 1

    # 8) sélection et aides reset
    viewer._sel = {"mode": None}
    viewer._clear_nearest_line()
    viewer._clear_edge_highlights()

    # 9) réappliquer les bindings (utile si le canvas a été recréé ou si Tk a perdu des liaisons)
    viewer._bind_canvas_handlers()

    # 10) redraw complet avec la vue restaurée + overlay
    viewer._redraw_from(viewer._last_drawn)
    viewer._redraw_overlay_only()
    # [H6] reconstruire le pick-cache avec la vue effectivement restaurée
    try:
        viewer._rebuild_pick_cache()
        viewer._pick_cache_valid = True
    except Exception:
        viewer._pick_cache_valid = False

    # focus + rebind défensif (tooltips/drag)
    viewer.canvas.focus_set()
    viewer._bind_canvas_handlers()

# ---------- Horloge : test de hit ----------
