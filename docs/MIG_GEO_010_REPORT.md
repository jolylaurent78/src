# MIG-GEO-010 — Remplacement des lecteurs passifs de `groups[*]["nodes"]`

Date : 16 juillet 2026  
Portée : migration fonctionnelle limitée aux lecteurs géométriques passifs.

## Résultat

Les lecteurs passifs ciblés ne déterminent plus les membres d'un groupe par la
chaîne historique `groups[gid]["nodes"]`. Ils suivent maintenant le flux :

```text
groupe UI (clé de projection uniquement)
  -> topoGroupId
  -> TopologyWorld / TopologyGroup canonique
  -> element_ids
  -> index topoElementId -> entrée _last_drawn
  -> calcul géométrique UI
```

Le modèle Core reste la source des membres. `_last_drawn` reste exclusivement
la projection géométrique nécessaire au canvas.

## API introduite

### `build_last_drawn_index(last_drawn)`

Construit le cache central `topoElementId -> (tid, entrée _last_drawn)`. Il
conserve la première occurrence en cas de doublon, conformément au comportement
du cache MIG-GEO-001 ; le validateur existant conserve la responsabilité de
signaler les doublons.

### `get_projected_elements(topology_group, last_drawn_index)`

Résout les `TopologyGroup.element_ids` vers les entrées projetées. Cette
fonction ne consulte ni `group_id` UI, ni `groups[*]["nodes"]`. Les éléments
Core absents de la projection sont ignorés sans mutation.

`TriangleViewerManual` expose également les ponts privés nécessaires aux
consommateurs UI : résolution d'un groupe Core à partir de la clé de projection,
résolution d'un groupe depuis une entrée projetée et conversion indexée des
entrées vers leurs `tid` lorsqu'une API historique les exige encore.

## Fonctions migrées

| Fonction | Avant | Après |
| --- | --- | --- |
| `_recompute_group_bbox()` | itération de `groups[gid]["nodes"]`, puis `tid -> _last_drawn` | `TopologyGroup.element_ids -> get_projected_elements()` |
| `_group_centroid()` | idem | `TopologyGroup.element_ids -> get_projected_elements()` |
| `_degrouperGroupScreenBBox()` | bbox écran obtenue depuis les `nodes` de la projection résultante | éléments projetés du groupe Core |
| `_group_outline_segments()` | union Shapely des triangles listés dans `nodes` | union Shapely des éléments projetés du groupe Core |
| `_group_outline_segments_topo()` | frontière Core mais index local reconstruit par parcours de `_last_drawn` | frontière Core et entrées obtenues par l'index central |
| `_draw_group_outlines()` | condition de présence de `nodes` avant le rendu | rendu depuis la frontière du groupe Core |
| `_find_nearest_vertex()` | excluait les candidats ayant le même `last_drawn[*]["group_id"]` | exclut les candidats dont le groupe Core est identique |
| `_update_edge_highlights()` | forme de collision et listes de tids obtenues via `_group_nodes()` | éléments projetés Core ; tids résolus depuis l'index |

## Lectures legacy supprimées

Les chemins ci-dessus n'effectuent plus les lectures suivantes :

```python
nodes = self.groups[gid]["nodes"]
for node in nodes:
    triangle = self._last_drawn[node["tid"]]
```

Ils n'utilisent plus `_group_shape_from_nodes()` dans l'UI. La forme de
collision est construite directement depuis les éléments projetés résolus par le
groupe Core.

## Lectures legacy restantes, hors périmètre

| Famille | Emplacements principaux | Statut |
| --- | --- | --- |
| Collage manuel | `_on_canvas_left_up()` : insertion, rotation et concaténation des chaînes `nodes` ; écriture de `edge_in/out` et `vkey_in/out` | hors périmètre, inchangé |
| XML historique | `saveScenarioXml()` / `loadScenarioXml()` dans `assembleur_io.py` | hors périmètre, inchangé |
| Projection de chaîne UI | activation de scénario, clonage de snapshot, suppression, maintenance de `group_id` et application des résultats de dégrouper | non migré : ces flux écrivent ou maintiennent la représentation legacy |
| Synchronisation de poses | `_sync_group_elements_pose_to_core()` | non migré : écriture Core depuis une projection UI, à traiter séparément |
| Fallback et diagnostic MIG-GEO | `_prepare_mig_geo_operation_members()` | non migré : comparaison / secours temporaire des migrations MOVE, ROTATE et FLIP |

Aucun de ces lecteurs n'a été modifié dans MIG-GEO-010.

## Validation

Les tests de caractérisation ajoutés vérifient que :

* `get_projected_elements()` renvoie les entrées dictées par les membres du
  groupe Core et l'index ;
* bbox, centroïde et contour incluent tous les membres Core même lorsqu'une
  chaîne UI de test est volontairement incomplète ;
* les tests existants des ponts Core/UI, MOVE, ROTATE, FLIP et de comparaison
  topologique restent verts.

Commandes exécutées :

```text
python -m compileall -q src
python -m pytest tests/test_mig_geo001_group_linking.py tests/test_topology_comparison.py tests/test_logging_utils.py -q
```

Résultat : `23 passed`.

## Limites volontairement conservées

Cette migration ne remplace pas `self.groups` comme structure de projection et
ne modifie pas la logique de collage. En particulier, elle ne change ni
`_on_canvas_left_up()`, ni `edge_in`, `edge_out`, `vkey_in`, `vkey_out`,
`saveScenarioXml()` ou `loadScenarioXml()`.

Le gain de MIG-GEO-010 est donc ciblé : les lecteurs purement géométriques ne
tirent plus leurs membres de la chaîne legacy, sans préjuger de la future
décision sur l'ordre de collage et la compatibilité XML.
