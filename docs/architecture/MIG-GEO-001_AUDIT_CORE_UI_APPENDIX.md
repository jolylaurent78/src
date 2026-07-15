# MIG-GEO-001 Audit Core ↔ UI — appendice de traçabilité

Cet appendice complète le rapport principal. Il inventorie les producteurs, consommateurs et mécanismes de persistance observés sans proposer de changement.

## A. Entrées `_last_drawn` : producteurs et variantes

| Variante | Fichier, classe/méthode | Champs construits ou modifiés | Notes |
|---|---|---|---|
| triangle catalogue temporaire | `assembleur_tk.py`, `TriangleViewerManual._triangle_from_index()` (`5664-5705`) | `labels`, `pts`, `id`, `mirrored` | `pts` est local au preview ; CW inverse le Y et rend initialement `mirrored=True` |
| placement manuel | `assembleur_tk.py`, `_place_dragged_triangle()` (`7247-7349`) | `labels`, `pts`, `id`, `mirrored`, puis `orient`, `topoElementId`, `topoGroupId`, `group_id`, `group_pos` | append UI avant création/sync Core ; `mirrored` est fixé à False |
| chargement XML | `assembleur_io.py`, `loadScenarioXml()` (`732-758`, `760-779`, `793-965`) | `id`, `pts`, `mirrored`, `group_id`, `group_pos`, labels ajoutés, IDs topologiques relinkés | points UI lus avant import Core ; `topoElementId` rétabli par rang `Txx` si absent |
| simulation auto, première paire | `assembleur_sim.py`, `AlgoQuadrisParPaires.run()` (`740-794`) | `labels`, `pts`, `id`, `mirrored`, `group_id`, `group_pos`, IDs topo ensuite | groupe UI 1 et bbox construits séparément |
| simulation auto, chaîne | `assembleur_sim.py` (`962-1116`, `1266-1310`) | mêmes champs, plus `_chain_edge_in/out` | les clés de chaîne sont transférées vers `groups.nodes` |
| auto, cache local | `assembleur_tk.py`, `_autoEnsureLocalGeometry()` (`4350-4375`) | copie de chaque entrée avec `pts` relatifs | stockée dans `scen.last_drawn_local`, pas dans `_last_drawn` actif |
| auto, reconstruction monde | `assembleur_tk.py`, `_autoRebuildWorldGeometryScenario()` (`4376-4419`) | nouvelle liste, même métadonnées, `pts` monde recalculés | construite depuis `last_drawn_local` + `auto_geom_state`, non directement depuis `TopologyWorld` |
| duplication | `assembleur_tk.py`, `_scenario_duplicate()` (`4695-4735`) | deepcopy complet des entrées | le Core est cloné séparément |
| cache de pick | `assembleur_tk.py`, `_rebuild_pick_cache()` (`5033-5068`) | `_pick_pts`, `_pick_poly` | mutation éphémère dans les entrées UI |

## B. Matrice détaillée par champ

| Champ | Écritures principales | Lectures principales | Persistance | Catégorie cible |
|---|---|---|---|---|
| `labels` | `_triangle_from_index`; `_place_dragged_triangle`; simulations; restauration Excel dans `loadScenarioXml` | `_redraw_from`, `_draw_triangle_screen`, `_reinsert_triangle_to_list`, simulation | pas dans `<triangle>` UI ; labels portés par `topoSnapshot` Core ; XML recharge depuis Excel ou `("", "", "")` | C, copie de labels A Core/catalogue |
| `pts` | preview drag, placement, move, rotation, flip, collage, dégroupement visuel, XML, simulations, auto rebuild | rendu, hit-test, groupes/bbox, collision/edge choice, contours, compas fallback, XML | `<triangles>/<triangle>/<O,B,L>` ; redondance avec local+pose du `topoSnapshot` | C ; projection, mais autorité opérationnelle actuelle |
| `id` | catalogue, XML, simulations | liste, suppression/réinsertion, `_placed_ids`, XML relink, comparaison de scénarios | attribut XML `triangle@id`; élément Core généralement `Txx` | B ; identité de catalogue reliée à Core |
| `mirrored` | catalogue preview, placement, flip, XML, simulation, pose import | rendu, sync UI→Core, simulation | attribut XML UI ; `pose.mirrored` dans snapshot Core | C pour copie UI ; A pour pose Core |
| `orient` | placement manuel depuis `df`; entrée source simulation | simulation et création Core via `meta` | pas dans triangle XML UI ; `meta` dans snapshot Core | A dans Core ; copie/entrée transitionnelle UI |
| `topoElementId` | placement, simulation, load XML par inférence | sync pose, contours topo UI, collage, suppression, groupes, XML validation | pas écrit dans `<triangle>` par `saveScenarioXml`; lu si présent puis reconstruit par rang | B |
| `topoGroupId` | placement, simulation, load/relink, collage, dégroupement | contours topo, compas, groupes UI, XML validation | snapshot Core porte groupes ; attribut UI lu mais non écrit par sauvegarde des groupes/triangles courante | B dérivée |
| `group_id` | placement, load XML, fusion, split, deletion remap | déplacement, rotation, flip, groupe ciblé, XML | `triangle@group` écrit ; groupe effectivement reconstitué de `<groups>` | E / structure UI |
| `group_pos` | placement, load, `_apply_group_meta_after_split_`, fusion | ordre de groupes, opérations de chaîne | non persisté directement ; dérivé de `<groups>/<node tid>` | E / structure UI |
| `_pick_pts` | `_rebuild_pick_cache` | `_ctx_compute_nearest_vertex_key` | non | D |
| `_pick_poly` | `_rebuild_pick_cache` | aucun consommateur direct trouvé hors cache prévu | non | D |
| `_chain_edge_in/out` | simulation de chaîne | finalisation auto vers `nodes.edge_in/out` | non directement | E ; A candidate si ordre de chaîne devient métier |

## C. Lecteurs transversaux de géométrie UI

| Sous-système | Fichier | Consommation de `_last_drawn` | Signification architecturale |
|---|---|---|---|
| Rendu | `assembleur_tk.py:_redraw_from`, `_draw_triangle_screen` | O/B/L, labels, id, mirrored | projection graphique directe |
| Hit-test | `assembleur_tk.py:_hit_test`, `_ctx_compute_nearest_vertex_key` | O/B/L ou `_pick_pts` | interaction UI légitime, ne doit pas créer une autorité métier |
| Bbox/groupe | `assembleur_tk.py:_recompute_group_bbox`, `_group_centroid` | O/B/L, `group_id`, nodes | cache/UI de manipulation |
| Contours non topo | `assembleur_tk.py:_group_outline_segments` | O/B/L et Shapely | traitement graphique/géométrique UI |
| Contours topo | `assembleur_tk.py:_group_outline_segments_topo` | `topoElementId`, O/B/L | mélange Core de segments et UI de points ; frontière hybride |
| Collage/choix arêtes | `assembleur_tk.py`, `assembleur_edgechoice.py` | points, IDs, groupes | utilise la projection comme base de calcul d'intention |
| Synchronisation pose | `_sync_group_elements_pose_to_core`; `assembleur_sim.py:setElementPoseFromWorldPts` | O/B/L, mirrored, elementId | flux inverse de compatibilité |
| XML | `assembleur_io.py` | champs de triangle et groupes | double représentation persistée |
| Compas | `assembleur_tk.py` | ancre UI avec fallback `TopologyWorld.getConceptNodeWorldXY` | montre les deux sources de coordonnées monde |
| Auto | `assembleur_sim.py`, `_autoRebuildWorldGeometryScenario` | copies, chaînes, local relative | sous-modèle de projection autonome |

## D. Groupes UI : structure inventoriée

```python
viewer.groups[ui_gid] = {
    "id": int,
    "nodes": [
        {
            "tid": int,
            "edge_in": "OB"|"BL"|"LO"|None,
            "edge_out": "OB"|"BL"|"LO"|None,
            # variantes automatiques / collage :
            "vkey_in": "O"|"B"|"L"|None,
            "vkey_out": "O"|"B"|"L"|None,
        },
    ],
    "bbox": (xmin, ymin, xmax, ymax)|None,
    "topoGroupId": str,  # optionnel en construction, attendu après liaison
}
```

Écrivains notables : placement (`assembleur_tk.py:7329-7346`), chargement XML (`assembleur_io.py:793-844`), collage (`assembleur_tk.py:10101-10221`), dégroupement (`assembleur_tk.py:7997-8152`), simulations (`assembleur_sim.py:778-793`, `:1291-1310`).

Persistant : XML `<groups>/<group id>/<node tid edge_in edge_out>`. `vkey_in/out`, `bbox` et `topoGroupId` sont lus comme compatibilité si présents, mais la sauvegarde courante ne sérialise explicitement que l'id UI et les nœuds `tid`/`edge_in`/`edge_out`.

## E. Snapshot Core et XML : séparation constatée

| Représentation | Sauvegarde | Chargement | Contrôle effectué |
|---|---|---|---|
| Core `TopologyWorld` | `saveScenarioXml` appelle `_exportPhysicalSnapshot` (`assembleur_io.py:398-417`) | `_importPhysicalSnapshot` (`:852-861`) | validité structurelle interne ; éléments, poses, attachements |
| UI triangles | `saveScenarioXml` écrit `id`, `mirrored`, `group`, O/B/L (`:512-524`) | points recréés (`:732-758`) | pas de comparaison avec pose Core |
| UI groupes | `<groups>` via `viewer.groups` (`:490-511`) | groupes/nodes recréés (`:793-844`) | relink strict vers groupes DSU (`:890-948`) |
| mots dictionnaire | `<words>` | `_tri_words` | indépendant de `TopologyWorld` |

Le relink XML impose qu'un triangle UI trouve un élément Core et qu'un groupe UI n'agrège pas plusieurs groupes DSU. Il ne vérifie pas que les O/B/L UI égaux aux coordonnées obtenues par la pose Core. C'est le point exact où deux géométries peuvent coexister après chargement.

## F. Exceptions et limites de l'inventaire

- L'audit est statique : il ne conclut pas sur la fréquence des variantes dans les fichiers XML historiques.
- Certains champs peuvent être ajoutés par extensions Python non déclenchées dans les chemins inspectés ; seuls les champs construits ou lus dans le dépôt ont été classés.
- L'absence de test automatisé trouvé sous `tests/` pour `last_drawn`, pose, miroir ou XML UI/Core ne prouve pas l'absence d'essais manuels ; elle limite seulement la preuve automatique disponible.
- Les caches conceptuels Core sont des caches internes du Core : ils ne sont pas classés comme `_last_drawn` et ne doivent pas être confondus avec le cache UI.
