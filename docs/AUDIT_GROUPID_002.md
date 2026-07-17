# AUDIT-GROUPID-002

## 1. Résumé exécutif

L'invariant MIG-GEO-016 est atteint pour la production : aucune couche hors `src/assembleur_core.py` n'appelle `find_group()` ni n'accède directement à `TopologyWorld.groups`.

Le reliquat n'est plus une fuite du DSU. Il est constitué de la projection UI historique : `last_drawn[*]["group_id"] <-> viewer.groups[ui_gid]["nodes"]`, avec `topoGroupId` comme miroir du Core.

Cette projection reste active dans XML, les scénarios automatiques, le dégroupage UI et le commit du collage manuel. Le point principal est le release de collage, qui transforme encore les triangles via `groups[ui_gid]["nodes"]` avant la synchronisation Core.

## 2. État des invariants

| Invariant | État | Preuve |
|---|---|---|
| `find_group()` hors Core en production | Conforme | 0 occurrence; les 49 appels de production sont dans `assembleur_core.py`. |
| `TopologyWorld.groups` hors Core en production | Conforme | 0 occurrence; les 9 accès trouvés sont dans les tests. |
| Re-canonicalisation DSU hors Core | Conforme | aucune lecture de `_group_parent`, `_group_members`, boucle de parents ou table d'alias. |
| Itération de groupes vivants | Conforme | `getLiveGroupIds()` est utilisé en UI, simulation et debug. |
| Lecture de membres Core | Conforme dans les flux migrés | `getGroupElementIds()` puis index `_last_drawn`. |
| Suppression de l'identité UI | Non atteinte | `group_id`, `self.groups` et `nodes` restent des structures historiques. |

## 3. Résultats chiffrés

| Motif | Occurrences dans `src/` + `tests/` | Interprétation |
|---|---:|---|
| `find_group(` | 54 | 49 Core; 4 assertions DSU; 1 double de test. |
| `group_id` | 172 | UI legacy, `TopologyGroup` et contrats Core. |
| `groupId` | 69 | Core historique, chemin et compas. |
| `gid` | 285 | variable locale de plusieurs sémantiques. |
| `topoGroupId` | 84 | miroir/projection d'un groupe Core. |
| `core_group_id` | 131 | contrat migré, majoritairement canonique. |
| `topo_group_id` | 24 | variante IO/compas. |
| `world.groups` / `topoWorld.groups` hors Core | 9 | tests uniquement. |

Les variantes `groupID`, `group_idx`, `groupIndex`, `coreGroupId`, `source_group_id`, `target_group_id`, `mobile_group_id` et `fixed_group_id` ne sont pas présentes.

## 4. Inventaire détaillé

### 4.1 `find_group`

| Fichier:ligne | Fonction/couche | Classe | Justification |
|---|---|---|---|
| `src/assembleur_core.py:677`, `1124`, `1634-4757` | Boundary, chemins, `TopologyWorld` | A | DSU, invalidation, fusion, clone, dégroupage et export. |
| `tests/test_audit_group_cleanup.py:100,106,118,128` | tests fusion/clone | C | vérifie explicitement les alias DSU. |
| `tests/test_mig_geo001_group_linking.py:312` | double `_World` | C | double de test, non production. |

Il n'existe ni import de `find_group`, ni alias d'import, ni appel indirect évident hors Core. L'objectif « 0 usage production de `find_group()` hors Core » est satisfait.

### 4.2 `world.groups` et équivalents

| Fichier:ligne | Intention | Classe | Remplacement ou constat |
|---|---|---|---|
| `tests/test_audit_group_cleanup.py:71,101,107,108,116,121` | inspecter/supprimer un alias | C | dette de test acceptable; aucune API publique ne doit cacher ce mécanisme DSU. |
| `tests/test_mig_geo001_group_linking.py:73` | fixture d'un `TopologyGroup` | C | non prioritaire. |
| `tests/test_topology_comparison.py:241,251` | snapshot d'intégrité | C | vérifie précisément l'absence de mutation. |
| `src/assembleur_debug.py:48,133` | lecture de `_concept_cache` | D | outil debug : fuite faible de cache privé, pas du registre de groupes. |

`self.groups` dans Tk et `viewer.groups` dans IO désignent les groupes UI historiques, non `TopologyWorld.groups`. Ils sont le reliquat fonctionnel décrit ci-dessous.

### 4.3 Variables et stockages significatifs

| Emplacement | Identifiant/structure | Sémantique | Canonique ? | Risque |
|---|---|---|---|---|
| `assembleur_core.py:563` | `TopologyGroup.group_id` | ID atomique interne DSU | non requis | légitime Core. |
| `assembleur_core.py:3041-3073` | `core_group_id` | représentant vivant public | oui | faible. |
| `assembleur_tk.py:768-784` | `ui_gid`, `topoGroupId` | pont UI vers Core | validé par `hasLiveGroup` | moyen. |
| `assembleur_tk.py:786-815` | groupe depuis `topoElementId` | groupe de l'élément projeté | oui | faible. |
| `assembleur_tk.py:867-933` | `entry["group_id"]`, `self.groups` | validateur F8 UI/Core | UI | faible, debug volontaire. |
| `assembleur_tk.py:9319-9341`, `9788-9866` | `_sel["core_group_id"]` | groupe mobile rotate/move | oui au stockage | moyen si fusion pendant geste. |
| `assembleur_tk.py:9829,9856,9967,10146` | `_sel["ui_group_id"]` | exclusion/snap et commit legacy | UI | élevé au collage. |
| `assembleur_tk.py:672,8304-8436` | `_clock_trace_preview_topoGroupId` | ancre Core compas | issue du Core | moyen après mutation. |
| `assembleur_tk.py:8108-8206` | `groups[*]["topoGroupId"]`, `last_drawn[*]["topoGroupId"]` | miroirs UI | normalement oui | moyen, synchronisation multiple. |
| `assembleur_io.py:453-621` | références compas/guides XML | ID Core persistant | supposé oui | moyen. |
| `assembleur_sim.py:364-371` | retour `topoGroupId` | résultat auto | oui | faible. |
| `assembleur_sim.py:746-818,962-1067` | `group_id: 1`, `groups[1]` | projection UI auto unique | UI | moyen. |

Les `gid` de `_new_group_id`, `_get_group_of_triangle`, `_group_nodes`, `_ctx_delete_group` et de la fusion de collage (`assembleur_tk.py:7565-7605,9209-9292,10146-10314`) sont des IDs UI. Les `gid` de `assembleur_core.py` sont internes au Core.

### 4.4 API exposant des group IDs

| API, fichier:ligne | Entrée/sortie | Contrat | Alias ? |
|---|---|---|---|
| `get_group_of_element`, `assembleur_core.py:3106` | élément -> groupe | A, sortie canonique | non en sortie. |
| `getGroupIdFromConceptNode`, `3112` | noeud -> groupe | A | non en sortie. |
| `getLiveGroupIds`, `3041` | monde -> groupes vivants | A | non, masque les alias. |
| `hasLiveGroup`, `3058` | Core ID -> bool | A | non, faux pour alias. |
| `getGroupElementIds`, `3063` | Core ID -> éléments | A, explicitement canonique | non, retourne `[]` sinon. |
| `ensureBoundary` et `getBoundary*`, `2028-2088,2179` | Core ID -> frontière | B cohérent | non, alias rejeté. |
| `computeBoundary`, `1990` | `group_id` générique | D, réservé Core | oui, Core canonise. |
| `simulateOverlapTopologique`, `2612` | deux groupes -> bool | D, réservé Core | oui, Core canonise. |
| `degrouperAtNode`, `3819` | groupe -> résultat | D entrée; A sortie | oui à l'entrée; sorties assertées canoniques. |
| `_get_active_core_group_id_for_ui_gid`, `assembleur_tk.py:768` | UI gid -> Core | B | non, validé par `hasLiveGroup`. |
| `_get_projected_elements_for_ui_group`, `832` | UI gid -> projection | B | compatibilité dégroupage. |
| `_sync_group_elements_pose_to_core`, `953` | Core ID -> poses | A implicite | non en pratique. |
| `createTopoQuadrilateral`, `assembleur_sim.py:229-371` | retour `topoGroupId` | A | non, obtenu par `get_group_of_element`. |

### 4.5 Re-canonicalisations

Le seul code qui lit ou modifie `_group_parent` ou `_group_members` est `src/assembleur_core.py:1606-1610,2987-3061,3598-3817`. Aucune résolution manuelle (`while parent`, `aliases`, `merged_into`, `resolve_group`) n'a été trouvée hors Core.

`assembleur_io.py:912-947` compare des valeurs déjà retournées par `get_group_of_element` et appelle `hasLiveGroup`. C'est une validation de projection, pas une résolution d'alias.

## 5. Cartographie des flux

### 5.1 Drag & drop

`mouse down -> topoElementId -> Core group canonique -> _sel[core_group_id] -> éléments projetés -> preview/highlights Core -> release/attachment Core -> projection UI`.

Le clic sommet utilise `_resolve_core_vertex_move_members` (`assembleur_tk.py:9637-9684`) : noeud logique, `find_node` (DSU de noeud), `get_group_of_element`, puis `get_last_drawn_entries_for_core_group`. Le groupe stocké est canonique.

Les highlights utilisent les Boundary Services via `core_gid_m/core_gid_t` (`assembleur_tk.py:7249-7395`) et ne lisent plus `group_id` UI.

Exception : le release du collage (`assembleur_tk.py:10142-10324`) applique encore `ui_group_id -> self.groups[gid] -> nodes -> _last_drawn`, puis réécrit `group_id`, concatène les `nodes` et met à jour `topoGroupId` après le commit Core. Criticité élevée.

### 5.2 Transformations

| Opération | Résolution des membres | État |
|---|---|---|
| MOVE centre | `_prepare_core_group_operation_members`, `_move_group_world`, `9586-9635` | conforme Core. |
| MOVE sommet | `_resolve_core_vertex_move_members`, `9637-9684` | conforme, sans split. |
| ROTATE | membres/pivot/sync Core, `9301-9412` | conforme; `gid` restant est UI/statut. |
| FLIP | helpers Core, `9546-9584` | conforme. |
| Recentrage/pose | `_sync_group_elements_pose_to_core`, `953-987` | conforme. |

### 5.3 Rendu et contours

- `_draw_group_outlines`, `assembleur_tk.py:6507-6530`, itère `getLiveGroupIds()`.
- `_update_edge_highlights`, `7249-7395`, est indépendant de `group_id` UI.
- `_find_nearest_vertex`, `7135-7162`, convertit `exclude_gid` UI en Core via l'adaptateur prévu.
- `_degrouperGroupScreenBBox`, `8044-8062`, est le lecteur restant de `_get_projected_elements_for_ui_group`.

### 5.4 Assemblage automatique

`assembleur_sim.py:229-371` renvoie `get_group_of_element(elementIdOdd)`, donc un ID canonique. L'overlap obtient ses deux groupes de la même API (`1015-1018`) et la reconstruction utilise `getLiveGroupIds()` (`982-983`).

`group_id: 1` et `groups[1]["topoGroupId"]` subsistent (`746-818`, `962-1067`) comme projection UI auto. Ce constat ne justifie aucun chantier EdgeChoiceEpts, BoundarySegment ou algorithme de snap.

### 5.5 Simulation, clonage et persistance

`TopologyWorld.clonePhysicalState()` reconstruit ses groupes depuis les éléments et attachments dans le Core. Le snapshot manuel clone séparément `last_drawn` et `groups` (`assembleur_tk.py:4863-4896`) : risque de miroir `topoGroupId` périmé, non fuite DSU.

Le chargement XML crée les groupes UI (`assembleur_io.py:802-843`), déduit le groupe Core depuis `topoElementId`, vérifie l'unicité et complète le miroir (`912-947`). La sauvegarde conserve `viewer.groups/nodes` (`493-519`) : dette fonctionnelle, pas code mort.

## 6. Constats classés par criticité

| Criticité | Constat | Références | Correction probable |
|---|---|---|---|
| ÉLEVÉE | Commit manuel encore piloté par `ui_group_id`, `group_id`, `nodes`. | `assembleur_tk.py:10142-10314` | membres par Core, projection UI séparée. |
| ÉLEVÉE | XML persiste le graphe UI ordonné et ses liens. | `assembleur_io.py:493-519,802-947` | étude de format/projection, pas suppression mécanique. |
| MOYENNE | Bascule auto reconstruit `group_id` depuis `scen.groups`. | `assembleur_tk.py:4956-4990` | isoler la réconciliation UI. |
| MOYENNE | Compas/guides conservent `topoGroupId` à travers mutations possibles. | `assembleur_tk.py:672,8304-8897`; `assembleur_io.py:453-621` | validation/invalidation après transaction. |
| MOYENNE | Adaptateurs UI -> Core pour snap/bbox dégroupage. | `assembleur_tk.py:768-784,7135-7162,8044-8062` | transmettre un `core_group_id` existant. |
| FAIBLE | F8 lit volontairement les deux modèles. | `assembleur_tk.py:867-933,1047-1210` | aucune : validateur. |
| FAIBLE | Outils/tests accèdent au privé ou aux alias. | `assembleur_debug.py:48,133`; tests | API diagnostic éventuelle. |

Aucun constat CRITIQUE : aucun alias ne sort du Core et aucun consommateur externe de production ne dépend du DSU.

## 7. Dette acceptable

- Les tests DSU peuvent inspecter les alias et `world.groups`.
- Le validateur F8 doit lire `group_id` UI tant que la projection existe.
- `assembleur_debug.py` est un outil, pas un chemin applicatif.
- XML et scénarios auto portent une dette fonctionnelle; ils ne sont pas du code mort.

## 8. Migrations recommandées

### MIG-GROUP-017 — Commit de collage manuel depuis le groupe Core

- **Problème :** `assembleur_tk.py:10142-10314` détermine encore les triangles transformés par `nodes`.
- **Fichiers/périmètre :** `src/assembleur_tk.py` et tests drag/snap; remplacer seulement la résolution des membres par `core_group_id -> getGroupElementIds -> _last_drawn`, puis conserver la projection UI après commit.
- **Gain/risque :** dernier flux géométrique manuel indépendant de `nodes` / moyen à élevé.
- **Tests :** collage groupe/groupe, même groupe, edge-edge, vertex-edge, annulation, sauvegarde/rechargement.

### MIG-GROUP-018 — Réconciliation auto comme projection Core

- **Problème :** `_set_active_scenario` recompose `group_id` depuis `scen.groups` (`4956-4990`).
- **Fichiers/périmètre :** `src/assembleur_tk.py`, `src/assembleur_sim.py`; isoler la reconstruction UI sans modifier l'algorithme auto.
- **Gain/risque :** moins d'écritures legacy / moyen.
- **Tests :** bascule manuel/auto, forward/reverse, comparaison et XML auto.

### MIG-GROUP-019 — Durée de vie des références `topoGroupId`

- **Problème :** compas, guides et caches peuvent conserver un ID devenu alias après fusion.
- **Fichiers/périmètre :** `src/assembleur_tk.py`, `src/assembleur_io.py`; Core seulement si une API de validation manque réellement.
- **Gain/risque :** références Core robustes / moyen.
- **Tests :** guide/trace, fusion, dégroupage, sauvegarde/rechargement.

### MIG-GROUP-020 — Projection XML des groupes UI

- **Problème :** XML sauvegarde/recharge `groups/nodes` comme compatibilité.
- **Fichiers/périmètre :** `src/assembleur_io.py`, `src/assembleur_tk.py`, fixtures; nécessite une décision sur XML cible.
- **Gain/risque :** prérequis à la disparition de `group_id` UI / élevé, données persistées.

### MIG-GROUP-021 — Hygiène outils et tests Core

- **Problème :** outils debug consultent `_concept_cache` privé.
- **Périmètre :** uniquement si une API de diagnostic Core est justifiée.
- **Gain/risque :** frontière plus lisible / faible.

## 9. Résultat des tests

Commande exécutée : `python -m pytest tests -q --ignore=tests/test_results_panel.py`.

Résultat : **145 tests exécutés, 139 réussis, 6 échecs** : `test_checkpointpolicy_progress_cadence`, `test_angle180_matching_positive`, `test_engine_abs_mirroring_total_6_solutions`, `test_engine_abs_extended_total_15_solutions`, `test_engine_abs_extended_total_joker_solutions` et `test_engine_rel_extended_total_count_smoke`.

Ces échecs concernent la cadence runtime et les résultats du décryptage Angle180, non les groupes Core/UI. L'audit ne les corrige pas.

## 10. Conclusion

**Peut-on imposer que tout ID de groupe métier manipulé hors Core soit déjà canonique ?** Oui. `get_group_of_element`, `getLiveGroupIds`, `hasLiveGroup` et `getGroupElementIds` fournissent déjà ce contrat; MOVE, ROTATE, FLIP, contours, highlights et simulation topologique l'utilisent.

**Quels obstacles subsistent ?** L'identité `ui_group_id` historique et `nodes`, encore actives dans le commit de collage manuel, la réconciliation auto, le dégroupage UI et XML. Le prochain chantier utile est MIG-GROUP-017, pas une nouvelle encapsulation du DSU.
