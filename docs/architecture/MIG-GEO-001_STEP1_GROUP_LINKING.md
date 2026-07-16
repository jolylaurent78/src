# MIG-GEO-001 Step 1 — Infrastructure de liaison Core ↔ `_last_drawn`

**Statut :** infrastructure de compatibilité, sans changement fonctionnel visible.  
**Périmètre :** scénario actif du `TriangleViewerManual`.  
**Hors périmètre :** suppression ou migration de `group_id`, XML métier, persistance, gestes move/rotate/flip/collage/dégroupement et modification des groupes UI.

## Objectif réalisé

Le Step 1 centralise désormais le chemin :

```text
topoElementId
    ↓ index UI
entrée _last_drawn

topoGroupId Core
    ↓ TopologyWorld.groups[canonical].element_ids
    ↓ index UI
entrées _last_drawn correspondantes
```

`group_id` reste l'unique mécanisme utilisé par les comportements existants. Les nouvelles API ne le remplacent encore dans aucun flux utilisateur.

## Architecture retenue

L'index vit dans `TriangleViewerManual` et concerne la liste active `self._last_drawn` :

```python
self._last_drawn_topo_index: dict[str, tuple[int, dict]]
# topoElementId -> (tid UI actuel, entrée _last_drawn)
```

Il est construit paresseusement et réutilisé tant que la référence de liste et sa longueur ne changent pas. Un changement de scénario ou une reconstruction automatique affecte une autre liste et force donc naturellement sa reconstruction au prochain appel. Le chargement XML invalide explicitement l'index dès le vidage de `_last_drawn`, puis le reconstruit après le relink strict UI/Core.

La valeur indexée conserve le `tid` pour la compatibilité avec les lecteurs historiques, mais les helpers exposent surtout l'entrée elle-même. Une suppression qui renumérote les `tid` fait varier la longueur de la liste et entraîne une reconstruction paresseuse. Le contrat d'invalidation explicite reste disponible pour toute future mutation *in-place* de `topoElementId`.

En cas de doublon de `topoElementId`, l'index conserve la première occurrence sans modifier les données. Le validateur DEBUG signale ce défaut.

## Helpers créés

Implémentés dans `src/assembleur_tk.py`, dans le bridge topologique du viewer :

| Helper | Rôle |
|---|---|
| `_invalidate_last_drawn_topo_index()` | invalide le cache de liaison sans effet sur l'état fonctionnel |
| `_rebuild_last_drawn_topo_index()` | reconstruit `topoElementId → (tid, entrée)` |
| `get_last_drawn_entry_by_topo_id(element_id)` | accès indexé à une entrée UI |
| `get_last_drawn_entries_by_topo_ids(element_ids)` | accès indexé à plusieurs entrées, dans l'ordre Core demandé |
| `get_last_drawn_entries_for_core_group(core_group_id)` | navigation groupe Core canonique → `element_ids` → entrées UI |
| `getTidForTopoElementId(topoElementId)` | API historique conservée, désormais alimentée par l'index |
| `debug_validate_core_ui_group_linking(ui_group_id=None)` | comparateur passif Core/UI de diagnostic |

Les helpers de groupe passent par `TopologyWorld.find_group()` et `TopologyWorld.groups[gid].element_ids`. Ils ne consultent pas `group_id` pour obtenir la liste Core.

## Stratégie de validation DEBUG

`debug_validate_core_ui_group_linking()` compare, pour chaque groupe UI demandé :

1. les entrées obtenues par parcours des valeurs `entry["group_id"]` ;
2. le groupe Core associé (`groups[ui_gid]["topoGroupId"]`, ou le groupe déduit du premier élément) ;
3. les `element_ids` de ce groupe Core ;
4. les entrées obtenues à travers l'index.

Il journalise, avec le préfixe `[MIG-GEO-001]`, les écarts suivants :

- groupe UI absent ;
- groupe Core absent/non déductible ;
- IDs UI seulement ou Core seulement ;
- `topoElementId` dupliqué dans `_last_drawn` ;
- incohérence entre l'index et les IDs effectivement présents.

Le validateur retourne une liste de dictionnaires de diagnostic. Il ne modifie aucune structure et ne lève pas d'exception de validation ; les erreurs de lecture du Core sont converties en résultat vide ou log DEBUG. Il n'est pas activé automatiquement dans les gestes UI afin de préserver strictement leur comportement.

## Inventaire synthétique des accès actuels

### Accès `topoElementId`

- `getTidForTopoElementId()` réalisait un parcours complet de `_last_drawn` ; il utilise maintenant l'index.
- `_sync_group_elements_pose_to_core()` relie une entrée UI à `TopologyWorld.elements` pour recalculer une pose.
- `_group_outline_segments_topo()` construit aujourd'hui localement une map `topoElementId → entrée` pour dessiner les contours Core.
- L'assistance de collage lit les IDs source/cible puis crée des attachements Core (`_on_canvas_left_up`, chemins autour de `assembleur_tk.py:10078`).
- Suppression et dégroupement utilisent les IDs pour agir sur les éléments/groupes Core.
- `loadScenarioXml()` reconstruit/valide les liens UI/Core ; les simulations les injectent dans les entrées générées.

### Accès `group_id`

`group_id` est encore employé par tous les gestes de groupe et reste intact :

- `_get_group_of_triangle()` et `_group_nodes()` ;
- déplacement, rotation, miroir, orientation et drag de groupe ;
- prévisualisation/collision/collage ;
- suppression, dégroupement et recomposition des groupes UI ;
- chargement XML, changement de scénario automatique et persistance de compatibilité.

### Parcours `_last_drawn` de recherche

Les principaux parcours concernent le rendu, hit-test, bboxes, contours, collision, groupes et XML. Le Step 1 ne les remplace pas, sauf le chemin historique de recherche `topoElementId → tid`. Les futurs lecteurs qui veulent partir du Core disposent désormais d'une API sans avoir à reconstruire chacun une recherche linéaire.

## Futurs lecteurs de `group_id` à migrer

La migration est volontairement différée. Les candidats, à traiter un par un avec leur contrat de comportement, sont :

1. les sélections de groupe depuis un triangle (`_get_group_of_triangle`) pour les opérations Core-first ;
2. les boucles `_group_nodes(gid)` de move/rotate/flip ;
3. la préparation de collision et de collage (`_build_edge_choice`, formes de groupe et listes mobiles/cibles) ;
4. les contours, en commençant par `_group_outline_segments_topo`, déjà relié à `topoGroupId` ;
5. la suppression et le dégroupement après validation des règles d'identité UI ;
6. les groupes de scénarios automatiques, une fois l'ordre UI et la composante Core explicitement séparés.

Ces éléments ne sont pas modifiés dans ce Step 1.

## Vérification

Le test ciblé `tests/test_mig_geo001_group_linking.py` couvre :

- lookup d'une entrée par `topoElementId` ;
- compatibilité `getTidForTopoElementId()` ;
- navigation d'un groupe Core vers l'entrée UI ;
- validateur sans diagnostic dans un cas cohérent ;
- invalidation/reconstruction après relink in-place.

Exécution : `python -m pytest tests/test_mig_geo001_group_linking.py -q` — **2 passed**.
