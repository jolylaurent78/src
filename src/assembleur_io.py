"""assembleur_io.py
Persistance (config JSON + scénario XML) isolée du GUI.
Les fonctions prennent 'viewer' en paramètre (duck-typing) pour éviter les imports circulaires.
"""

import os
import json
import datetime as _dt
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET
import re
import traceback
import numpy as np

def _ioWarn(viewer, where: str, exc: Exception):
    """
    Best-effort logging (console) sans casser l'IHM.
    Active si:
      - viewer.debug_io == True
      - ou variable d'env ASSEMBLEUR_DEBUG_IO=1
    """
    try:
        if getattr(viewer, "debug_io", False) or os.environ.get("ASSEMBLEUR_DEBUG_IO", "") in ("1", "true", "True"):
            msg = f"[IO][WARN] {where}: {type(exc).__name__}: {exc}"
            print(msg)
            # trace utile en dev
            print(traceback.format_exc())
    except Exception:
        # dernier filet: on ne casse jamais sur le logger
        return

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
    except (OSError, json.JSONDecodeError) as e:
        # Jamais bloquant : si la config est corrompue on repart de zéro.
        _ioWarn(viewer, f"loadAppConfig(path={getattr(viewer,'config_path','')})", e)
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
    except OSError as e:
        _ioWarn(viewer, f"saveAppConfig(path={getattr(viewer,'config_path','')})", e)

def getAppConfigValue(viewer, key: str, default=None):
    return (getattr(viewer, "appConfig", {}) or {}).get(key, default)


def setAppConfigValue(viewer, key: str, value):
    try:
        if not hasattr(viewer, "appConfig") or viewer.appConfig is None:
            viewer.appConfig = {}
        viewer.appConfig[key] = value
        viewer.saveAppConfig()
    except Exception as e:
        # Ici on loggue: si ça casse, tu veux le savoir en dev.
        _ioWarn(viewer, f"setAppConfigValue(key={key!r})", e)

def saveScenarioXml(viewer, path: str):
    """
    Sauvegarde 'robuste' (v4) :
      - source excel, view (zoom/offset), clock (pos + hm),
      - ids restants (listbox),
      - triangles affichés (id, mirrored, points monde O/B/L, group),
      - groupes (nodes complets : tid + edge_in/out),
      - associations triangle→mot (word,row,col).
    """
    scen = viewer._get_active_scenario()
    world = scen.topoWorld

    topo_tx_orientation = world._topoTxOrientation

    snapshot = world._exportPhysicalSnapshot()
    if not isinstance(snapshot, dict):
        raise ValueError(
            f"saveScenarioXml: snapshot topo invalide ({type(snapshot).__name__}), dict attendu."
        )

    root = ET.Element("scenario", {
        # v2 : contrat canonique groups/nodes = edge_in/edge_out
        "version": "4",
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "topo_tx_orientation": topo_tx_orientation,
    })
    topo_snapshot_el = ET.SubElement(root, "topoSnapshot", {"encoding": "json"})
    topo_snapshot_el.text = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    world.topologyChemins._saveToXml(root)
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

    # map (fond) : fichier + worldRect + opacité/visibilité + scale
    bg = getattr(viewer, "_bg", None)
    map_path = str(bg.get("path")) if isinstance(bg, dict) and bg.get("path") else ""
    # scale affiché (best-effort)
    map_scale = viewer._bg_compute_scale_factor() if hasattr(viewer, "_bg_compute_scale_factor") else None
    if map_scale is None:
        map_scale = getattr(viewer, "_bg_scale_factor_override", None)
 
    ET.SubElement(root, "map", {
        "path": os.path.abspath(map_path) if map_path else "",
        "x0": f"{float(bg.get('x0')) if isinstance(bg, dict) and bg.get('x0') is not None else 0.0:.6g}",
        "y0": f"{float(bg.get('y0')) if isinstance(bg, dict) and bg.get('y0') is not None else 0.0:.6g}",
        "w":  f"{float(bg.get('w'))  if isinstance(bg, dict) and bg.get('w')  is not None else 0.0:.6g}",
        "h":  f"{float(bg.get('h'))  if isinstance(bg, dict) and bg.get('h')  is not None else 0.0:.6g}",
        "visible": "1" if (getattr(viewer, "show_map_layer", None) and viewer.show_map_layer.get()) else "0",
        "opacity": f"{int(viewer.map_opacity.get()) if hasattr(viewer, 'map_opacity') else 100}",
        "scale": f"{float(map_scale):.6g}" if map_scale is not None else "",
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
    except Exception as e:
        _ioWarn(viewer, "saveScenarioXml(listbox)", e)
    # groupes (best-effort)
    groups_xml = ET.SubElement(root, "groups")
    try:
        for gid, gdata in (viewer.groups or {}).items():
            g_el = ET.SubElement(groups_xml, "group", {"id": str(gid)})
            # on s'appuie sur l'API viewer._group_nodes(gid) (contrat = nodes complets)
            nodes = viewer._group_nodes(gid)
            if nodes is None:
                raise RuntimeError(f"saveScenarioXml: _group_nodes({gid}) a retourné None")
            for nd in nodes:
                if not isinstance(nd, dict):
                    raise RuntimeError(f"saveScenarioXml: node invalide gid={gid} type={type(nd)}")
                tid = nd.get("tid")
                if tid is None:
                    raise RuntimeError(f"saveScenarioXml: node tid manquant gid={gid}")
                ET.SubElement(g_el, "node", {
                    "tid": str(int(tid)),
                    "edge_in": str(nd.get("edge_in") or ""),
                    "edge_out": str(nd.get("edge_out") or ""),
                })
    except Exception as e:
        _ioWarn(viewer, "saveScenarioXml(groups)", e)
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
    Recharge un scenario v4:
      - tente de recharger le fichier Excel source (mode degrade si absent),
      - restaure vue, horloge, listbox, triangles poses (+mots), groupes,
      - restaure la topologie core depuis <topoSnapshot encoding="json">,
      - redessine.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "scenario":
        raise ValueError("Fichier scenario invalide (balise racine).")

    ver = str(root.get("version", "") or "").strip()
    if ver != "4":
        raise ValueError(f"Unsupported scenario version: expected 4, got {ver}.")

    topo_tx_orientation = str(root.get("topo_tx_orientation", "") or "").strip().lower()
    if topo_tx_orientation not in {"cw", "ccw"}:
        raise ValueError("Missing or invalid topo_tx_orientation (expected cw|ccw).")

    topo_snapshot_el = root.find("topoSnapshot")
    topo_snapshot_txt = ""
    if topo_snapshot_el is not None:
        topo_snapshot_txt = str(topo_snapshot_el.text or "").strip()
    if (
        topo_snapshot_el is None
        or str(topo_snapshot_el.get("encoding", "") or "").strip().lower() != "json"
        or not topo_snapshot_txt
    ):
        raise ValueError("Missing topoSnapshot (encoding=json) in scenario v4.")

    snapshot = json.loads(topo_snapshot_txt)
    if not isinstance(snapshot, dict):
        raise ValueError("Missing topoSnapshot (encoding=json) in scenario v4.")

    scen = viewer._get_active_scenario()
    if scen is None:
        raise ValueError("loadScenarioXml: active scenario is required for v4 load.")

    # 1) Excel source (degraded mode allowed)
    src = root.find("source")
    excel = str(src.get("excel", "") if src is not None else "").strip()
    excel_loaded = False
    if excel:
        if os.path.isfile(excel):
            try:
                viewer.load_excel(excel)
                excel_loaded = True
            except Exception as e:
                _ioWarn(viewer, "loadScenarioXml(excel)", e)
                if hasattr(viewer, "_status_warn"):
                    viewer._status_warn(
                        "Excel source not found; scenario loaded in degraded mode (no full triangle catalog)."
                    )
        else:
            if hasattr(viewer, "_status_warn"):
                viewer._status_warn(
                    "Excel source not found; scenario loaded in degraded mode (no full triangle catalog)."
                )

    # 2) vue
    view = root.find("view")
    if view is not None:
        viewer.zoom = float(view.get("zoom", viewer.zoom))
        ox = float(view.get("offset_x", viewer.offset[0] if hasattr(viewer, "offset") else 0.0))
        oy = float(view.get("offset_y", viewer.offset[1] if hasattr(viewer, "offset") else 0.0))
        viewer.offset = np.array([ox, oy], dtype=float)

    # 2bis) map (fond)
    map_el = root.find("map")
    if map_el is not None:
        map_path = str(map_el.get("path", "") or "").strip()
        if map_path and os.path.isfile(map_path):
            rect = {
                "x0": float(map_el.get("x0", "0") or 0.0),
                "y0": float(map_el.get("y0", "0") or 0.0),
                "w": float(map_el.get("w", "0") or 0.0),
                "h": float(map_el.get("h", "0") or 0.0),
            }
            try:
                viewer._bg_set_map(map_path, rect_override=rect, persist=False)
            except Exception as e:
                _ioWarn(viewer, "loadScenarioXml(map)", e)
                viewer._bg_clear(persist=False)
        else:
            if map_path and hasattr(viewer, "_status_warn"):
                viewer._status_warn(f"Carte introuvable: {map_path}")
            viewer._bg_clear(persist=False)

        if hasattr(viewer, "show_map_layer"):
            viewer.show_map_layer.set(str(map_el.get("visible", "1")) not in ("0", "false", "False"))
        if hasattr(viewer, "map_opacity"):
            viewer.map_opacity.set(int(map_el.get("opacity", "100") or 100))
        sc = str(map_el.get("scale", "") or "").strip()
        if sc:
            viewer._bg_scale_factor_override = float(sc)

    # Etat d'interaction propre
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
        viewer._clock_cx = float(clock.get("x", "0"))
        viewer._clock_cy = float(clock.get("y", "0"))
        h = int(clock.get("hour", "0"))
        m = int(clock.get("minute", "0"))
        lbl = clock.get("label", "")
        viewer._clock_state.update({"hour": h, "minute": m, "label": lbl})

    # 4) listbox (ids restants)
    lb = root.find("listbox")
    if lb is not None:
        remain_ids = [int(e.get("id")) for e in lb.findall("tri") if e.get("id")]
        viewer.listbox.delete(0, "end")

        can_use_df = (
            getattr(viewer, "df", None) is not None
            and not viewer.df.empty
            and (excel_loaded or not excel)
        )
        if can_use_df:
            if remain_ids:
                wanted = set(remain_ids)
                for _, r in viewer.df.iterrows():
                    tid = int(r["id"])
                    if tid in wanted:
                        viewer.listbox.insert("end", f"{tid:02d}. B:{r['B']}  L:{r['L']}")
            else:
                for _, r in viewer.df.iterrows():
                    tid = int(r["id"])
                    viewer.listbox.insert("end", f"{tid:02d}. B:{r['B']}  L:{r['L']}")
        else:
            for tid in remain_ids:
                viewer.listbox.insert("end", f"{int(tid):02d}.")

    # 5) triangles poses (monde)
    viewer._last_drawn.clear()
    tris_xml = root.find("triangles")
    if tris_xml is not None:
        for t_el in tris_xml.findall("triangle"):
            tid = int(t_el.get("id"))
            mirrored = (t_el.get("mirrored", "0") == "1")
            _group_id_unused = int(t_el.get("group", "0"))
            P = {}
            for k in ("O", "B", "L"):
                n = t_el.find(k)
                if n is not None and n.text:
                    P[k] = viewer._xml_to_pt(n.text)

            item = {
                "id": tid,
                "pts": P,
                "mirrored": mirrored,
                "group_id": None,
                "group_pos": None,
            }
            topo_element_id = str(t_el.get("topoElementId", "") or "").strip()
            topo_group_id = str(t_el.get("topoGroupId", "") or "").strip()
            if topo_element_id:
                item["topoElementId"] = topo_element_id
            if topo_group_id:
                item["topoGroupId"] = topo_group_id
            viewer._last_drawn.append(item)

    # 5bis) labels
    try:
        if getattr(viewer, "df", None) is not None and not viewer.df.empty and (excel_loaded or not excel):
            by_id = {int(r["id"]): (str(r["B"]), str(r["L"])) for _, r in viewer.df.iterrows()}
            for t in viewer._last_drawn:
                if "labels" not in t or not t["labels"]:
                    b, l = by_id.get(int(t.get("id", -1)), ("", ""))
                    t["labels"] = ("Bourges", b, l)
        else:
            for t in viewer._last_drawn:
                if "labels" not in t or not t["labels"]:
                    t["labels"] = ("Bourges", "", "")
    except Exception:
        for t in viewer._last_drawn:
            if "labels" not in t or not t["labels"]:
                t["labels"] = ("Bourges", "", "")

    # 5ter) ids deja poses
    viewer._placed_ids = {int(t["id"]) for t in viewer._last_drawn}
    viewer._update_triangle_listbox_colors()

    # 6) mots associes
    viewer._tri_words = {}
    words_xml = root.find("words")
    if words_xml is not None:
        for w in words_xml.findall("w"):
            tid = int(w.get("tri_id"))
            viewer._tri_words[tid] = {
                "word": w.get("text", ""),
                "row": int(w.get("row", "0")) if w.get("row") else 0,
                "col": int(w.get("col", "0")) if w.get("col") else 0,
            }

    # 7) groupes
    viewer.groups.clear()
    groups_xml = root.find("groups")
    if groups_xml is not None:
        for g_el in groups_xml.findall("group"):
            gid_attr = g_el.get("id")
            if gid_attr is None:
                raise ValueError("Scenario XML invalide: group sans 'id'")
            gid = int(gid_attr)

            nodes = []
            allowed_edges = {"OB", "BL", "LO"}
            for nd_i, nd_el in enumerate(g_el.findall("node")):
                tid_attr = nd_el.get("tid")
                if tid_attr is None:
                    raise ValueError(f"Scenario XML invalide: node sans 'tid' (gid={gid})")

                tid = int(tid_attr)
                if not (0 <= tid < len(viewer._last_drawn)):
                    raise ValueError(
                        f"Scenario XML invalide: tid hors bornes (gid={gid} i={nd_i} tid={tid})"
                    )

                nd = {
                    "tid": tid,
                    "edge_in": (nd_el.get("edge_in") or "").strip() or None,
                    "edge_out": (nd_el.get("edge_out") or "").strip() or None,
                }

                ein = nd.get("edge_in")
                eout = nd.get("edge_out")
                if ein is not None and ein not in allowed_edges:
                    raise ValueError(f"Scenario XML invalide: edge_in={ein!r} (gid={gid} i={nd_i})")
                if eout is not None and eout not in allowed_edges:
                    raise ValueError(f"Scenario XML invalide: edge_out={eout!r} (gid={gid} i={nd_i})")
                # v4 : edge_in utilisé par la simulation d'assemblage automatique
                # on laisse None 
                if nd_i > 0 and not ein:
                    nd["edge_in"] = None

                nodes.append(nd)
                viewer._last_drawn[tid]["group_id"] = gid
                viewer._last_drawn[tid]["group_pos"] = len(nodes) - 1
                if "group" in viewer._last_drawn[tid]:
                    del viewer._last_drawn[tid]["group"]

            grp = {"id": gid, "nodes": nodes, "bbox": None}
            grp_topo_gid = str(g_el.get("topoGroupId", "") or "").strip()
            if grp_topo_gid:
                grp["topoGroupId"] = grp_topo_gid
            viewer.groups[gid] = grp
            viewer._recompute_group_bbox(gid)

            if len(nodes) >= 2:
                for i in range(len(nodes) - 1):
                    if not nodes[i].get("edge_out"):
                        # V2.1: edge_out est un champ UI non bloquant en v4.
                        nodes[i]["edge_out"] = None

    # 7bis) restore core world from snapshot + strict UI/core relink
    from src.assembleur_core import TopologyWorld

    sid = str(getattr(scen, "topoScenarioId", "") or "").strip() or "SCENARIO"
    scen.topoScenarioId = sid
    scen.topoWorld = TopologyWorld()
    world = scen.topoWorld
    world._topoTxOrientation = topo_tx_orientation
    world._importPhysicalSnapshot(snapshot)
    world.topologyChemins._loadFromXml(root.find("chemins"))

    # On construit une table elem_id_by_rank qui contient la liste des Triangles Txx du core
    elem_id_by_rank = {}
    for _eid in world.elements.keys():
        _rank = world.__class__.parse_tri_rank_from_element_id(_eid)
        if _rank is None:
            continue
        if _rank not in elem_id_by_rank:
            elem_id_by_rank[_rank] = str(_eid)

    # On prend tous les elements de lastdrawn et on vérifie qu'ils sont bien mappés avec le core
    for t in viewer._last_drawn:
        # On remappe chaque triangle
        tid = t.get("id", "?")
        topo_element_id = t.get("topoElementId", None)
        if topo_element_id in (None, ""):
            topo_element_id = elem_id_by_rank.get(int(tid), None)
            if topo_element_id is not None:
                t["topoElementId"] = topo_element_id

        topo_element_id = str(topo_element_id)
        if topo_element_id not in world.elements:
            raise ValueError(f"Triangle {tid} references missing topoElementId {topo_element_id}.")

        gid = world.get_group_of_element(topo_element_id)
        if t.get("topoGroupId") in (None, ""):
            t["topoGroupId"] = gid

    for gid, g in (viewer.groups or {}).items():
        nodes = list((g or {}).get("nodes", []) or [])
        if not nodes:
            continue

        canonical_gid = None
        for nd in nodes:
            tid = int(nd.get("tid"))
            tri = viewer._last_drawn[tid]
            topo_element_id = tri.get("topoElementId", None)
            if topo_element_id in (None, "") or str(topo_element_id) not in world.elements:
                raise ValueError(
                    f"Triangle {tri.get('id', tid)} references missing topoElementId {topo_element_id}."
                )
            cgid = world.get_group_of_element(str(topo_element_id))
            if canonical_gid is None:
                canonical_gid = cgid
            elif str(canonical_gid) != str(cgid):
                raise ValueError(f"Group {gid} topoGroupId mismatch with core DSU canonical.")

        if canonical_gid is None or str(canonical_gid) not in world.groups:
            raise ValueError(f"Group {gid} topoGroupId mismatch with core DSU canonical.")

        ui_topo_gid = g.get("topoGroupId", None)
        if ui_topo_gid in (None, ""):
            g["topoGroupId"] = canonical_gid
        elif str(ui_topo_gid) != str(canonical_gid):
            raise ValueError(f"Group {gid} topoGroupId mismatch with core DSU canonical.")

        for nd in nodes:
            tid = int(nd.get("tid"))
            viewer._last_drawn[tid]["topoGroupId"] = canonical_gid

    # Nettoyage global de compatibilite: purger toute trace residuelle de 'group'
    for _t in viewer._last_drawn:
        if "group" in _t:
            del _t["group"]
    viewer._next_group_id = (max(viewer.groups.keys()) + 1) if viewer.groups else 1

    # 8) selection et aides reset
    viewer._sel = {"mode": None}
    viewer._clear_nearest_line()
    viewer._clear_edge_highlights()

    # 9) re-appliquer les bindings
    viewer._bind_canvas_handlers()

    # 10) redraw complet
    viewer._redraw_from(viewer._last_drawn)
    viewer._redraw_overlay_only()
    viewer._rebuild_pick_cache()
    viewer._pick_cache_valid = True
    if hasattr(viewer, "refreshCheminTreeView"):
        viewer.refreshCheminTreeView()

    viewer.canvas.focus_set()
    viewer._bind_canvas_handlers()

# ---------- Horloge : test de hit ----------
