# AUDIT-CORE-ID-001 — Fausses dépendances aux identifiants de groupes UI

Date : 17 juillet 2026  
Nature : audit statique ; aucune modification de code.

## Résumé exécutif

Le paramètre `gid` de nombreuses méthodes Tk ne désigne plus le groupe métier.
Dans les lecteurs géométriques migrés, il sert seulement à retrouver le lien
de projection `topoGroupId`, puis le calcul est effectué sur un
`TopologyGroup` et les éléments de `_last_drawn` correspondants.

Les fonctions de catégorie A sont donc des candidates immédiates à une
signature Core. Elles ne doivent pas recevoir un entier UI dans leur API
métier. Les fonctions de catégorie B restent réellement liées au modèle
historique parce qu'elles lisent, écrivent ou ordonnent `nodes`. Les fonctions
de catégorie C n'ont qu'un rôle de cache/rendu Tk et pourront disparaître avec
la projection UI, mais n'exigent pas de migration fonctionnelle immédiate.

Les API de `assembleur_core.py` qui prennent `group_id` sont exclues : il s'agit
de l'identifiant canonique du DSU (`TopologyGroup.group_id`), donc d'une
dépendance Core valide.

## Méthode et vocabulaire

L'audit a recherché les signatures contenant `gid`, `ui_gid`, `group_id` ou
équivalent dans `src/` et `tests/`, puis a suivi les appels et les premières
lectures de chaque méthode concernée.

| Nom | Signification |
| --- | --- |
| `core_group_id` | identifiant canonique résolu par `TopologyWorld.find_group()` |
| `TopologyGroup` | entité métier contenant `element_ids` et attachments |
| `ui_gid` | clé numérique locale de `self.groups` et de `last_drawn.group_id` |
| `nodes` | chaîne historique ordonnée de triangles dans la projection UI |

## Inventaire et classification

### A — Fausse dépendance UI : migrable vers Core Group ID

| Fonction actuelle | Où le gid intervient | Travail réel | Signature cible | Priorité |
| --- | --- | --- | --- | --- |
| `_get_active_topology_group_for_ui_gid(ui_gid)` | lit `self.groups[ui_gid].topoGroupId` | résout le groupe DSU canonique | à terme supprimer ; les appelants reçoivent `core_group_id` | haute |
| `_get_projected_elements_for_ui_group(ui_gid)` | délègue au helper précédent | `TopologyGroup -> element_ids -> _last_drawn` | `_get_projected_elements_for_core_group(topology_group)` déjà disponible | haute |
| `_find_nearest_vertex(..., exclude_gid)` | adapte le gid exclu vers son groupe Core | exclusion de candidats par `TopologyGroup.group_id` | `exclude_core_group_id` ou `excluded_group` | haute |
| `_group_outline_segments_topo(gid)` | adapte gid vers `TopologyGroup` | frontière depuis `TopologyWorld.getBoundarySegments()` | `_group_outline_segments_topo(core_group_id)` | haute |
| `_group_outline_segments(gid, eps)` | appelle la projection Core | union géométrique des éléments projetés | `_group_outline_segments(core_group_id, eps)` | haute |
| `_outline_for_item(idx, eps)` | lit le `group_id` UI de l'entrée | choisit le contour du groupe de l'élément | résoudre directement `topoElementId -> groupe Core` | haute |
| `_recompute_group_bbox(gid)` | conserve seulement `g["bbox"]` après résolution Core | bbox des éléments de `TopologyGroup` | `_compute_group_bbox(core_group_id)` ; écriture éventuelle dans un cache UI séparé | haute |
| `_group_centroid(gid)` | passe par `_get_projected_elements_for_ui_group()` | barycentre des éléments projetés | `_group_centroid(core_group_id)` | haute |
| `_degrouperGroupScreenBBox(ui_gid)` | passe par la projection Core | bbox écran des éléments de composante Core | `_degrouperGroupScreenBBox(core_group_id)` | moyenne |
| `_update_edge_highlights(...)` | reçoit des indices puis obtient des `gid` UI | collisions et tids construits depuis éléments projetés Core | porter les groupes Core dans l'état d'assistance | moyenne |

Ces dix fonctions ne consultent pas `nodes` pour déterminer les membres. Leur
`gid` est un adaptateur de compatibilité, pas une donnée métier.

### B — Dépendance legacy forte

| Fonction actuelle | Dépendance `nodes` / ordre | Pourquoi elle n'est pas une simple adaptation | Signature cible éventuelle |
| --- | --- | --- | --- |
| `_group_nodes(gid)` | retourne directement la chaîne | façade de compatibilité XML/collage | aucune avant migration C/D/E |
| `_apply_group_meta_after_split_(gid)` | réécrit `group_id` pour chaque node | maintient la projection UI après mutation | recevoir des `element_ids` Core lors de la refonte de projection |
| `_degrouperTranslateGroupByScreen(ui_gid, ...)` | déplace les nodes de la vue post-dégrouper | mutation géométrique UI active | `core_group_id` + éléments projetés, dans un chantier dédié |
| `_applyDegrouperResultToTk(res)` | reconstruit et trie des chaînes UI | crée les vues et conserve une convention d'ordre | résultat Core + contrat de projection explicite |
| `_ctx_delete_group()` | collecte/remappe les `tid` de `nodes` | suppression et réparation de chaînes UI | suppression Core suivie d'une reconstruction de projection |
| `_move_group_world(gid, ...)` fallback | parcourt `g["nodes"]` si les membres Core manquent | secours legacy encore actif | supprimer le fallback après caractérisation |
| `_prepare_mig_geo_operation_members(..., legacy_group_id)` | produit `LegacyMembers` avec `_group_nodes()` | oracle/fallback de migration MOVE/ROTATE/FLIP | conserver temporairement puis retirer le chemin legacy |
| `_on_ctrl_down()`, rotation `move_group` | applique une rotation aux nodes | flux actif d'interaction, pas un lecteur passif | membres Core stockés dans `_sel` |
| `_on_canvas_left_up()` | insertion, rotation et concaténation de `nodes` | collage manuel, ordre et jonctions historiques | hors périmètre : nécessite contrat de chaîne explicite |
| `assembleur_sim.py`, génération/enrichissement | construit voisins successifs et `edge_in/out` | ordre de génération et conventions de chaîne | décider si cet ordre devient une métadonnée Core |
| `assembleur_io.py`, save/load v4 | écrit/lit l'ordre et edges des nodes | contrat de compatibilité XML | définir XML cible avant migration |

Ces onze fonctions sont les vrais consommateurs du modèle historique. Les
convertir mécaniquement en `core_group_id` ferait perdre soit une mutation UI,
soit un ordre de chaîne, soit la compatibilité XML.

### C — UI pure / cache de projection

| Fonction/famille | Rôle | Sort à terme |
| --- | --- | --- |
| `_new_group_id()` et `_ensure_group_fields()` | allocation et présence du cache `last_drawn.group_id` | supprimables quand la projection n'adressera plus de groupe UI |
| placement singleton | crée `self.groups[gid]` pour l'UI | remplacer par une projection dérivée du groupe Core créé |
| `_draw_group_outlines()` | énumère les clés UI uniquement pour lancer le rendu de frontières Core | peut itérer directement les groupes Core canoniques |
| `_set_active_scenario()` et clone de snapshot | rattache, copie et réconcilie le cache UI historique | reconstruction déterministe lorsque le XML cible sera défini |
| validateur MIG-GEO-001 / F8 | compare Core et projection UI | à supprimer lorsque la projection legacy n'est plus produite |

## Cas particuliers et signatures recommandées

### Calculs géométriques

Le meilleur contrat pour bbox, centroïde et contour est un identifiant Core
canonisé, car la méthode résout elle-même les éléments projetés :

```python
_compute_group_bbox(core_group_id: str)
_group_centroid(core_group_id: str)
_group_outline_segments(core_group_id: str, eps: float = EPS_WORLD)
_group_outline_segments_topo(core_group_id: str)
```

Si l'appelant détient déjà l'objet, le contrat le plus direct est :

```python
get_projected_elements(topology_group, last_drawn_index)
```

Il évite une seconde résolution DSU et convient aux boucles de rendu.

### Exclusion de snap et assistance de collage

`_find_nearest_vertex()` doit recevoir `exclude_core_group_id` plutôt que
`exclude_gid`. `_outline_for_item()` devrait obtenir le groupe depuis le
`topoElementId` de l'entrée cliquée. L'état `_sel` contient déjà
`core_group_id` pour les gestes migrés ; son usage doit être étendu aux aides
visuelles avant de supprimer les adaptateurs UI.

### Bbox UI mémorisée

`_recompute_group_bbox()` est une fausse dépendance seulement pour le calcul.
Son écriture actuelle dans `self.groups[gid]["bbox"]` reste un cache UI. Une
migration sûre sépare donc :

```text
calcul Core/projection -> bbox
cache UI optionnel <- bbox
```

Il ne faut pas réintroduire un `gid` UI dans l'API de calcul uniquement parce
que l'ancien cache le stocke sous cette clé.

## Liste priorisée des migrations immédiates

1. `_find_nearest_vertex`, `_outline_for_item`, `_group_outline_segments` et
   `_group_outline_segments_topo` : mêmes invariants, uniquement lecture Core
   et géométrie projetée.
2. `_recompute_group_bbox`, `_group_centroid`, `_degrouperGroupScreenBBox` :
   séparer calcul Core de l'écriture éventuelle dans le cache UI.
3. `_update_edge_highlights` : faire transiter les ids Core dans l'état de
   l'assistance plutôt que de reconstituer les ids UI.
4. Après caractérisation : supprimer le fallback de `_move_group_world` et
   l'oracle legacy de `_prepare_mig_geo_operation_members`.

Les éléments de catégorie B liés au collage, à l'ordre et au XML ne doivent pas
être inclus dans ce lot.

## Effort restant avant suppression des UI Group IDs métier

| Étape | Effort relatif | Dépendance |
| --- | --- | --- |
| Migrer les dix adaptateurs A | moyen | tests de caractérisation géométrique et snap |
| Retirer fallbacks opérationnels B | moyen | validation Core/UI prolongée |
| Refaire projection post-dégrouper et suppression | élevé | contrat de vue dérivée |
| Migrer collage manuel et ordre de chaîne | très élevé | décision produit sur ordre/insertion |
| Définir XML cible et compatibilité | très élevé | stratégie de versionnement et de migration |

Ainsi, les UI Group IDs peuvent cesser rapidement d'être des identifiants
**métier** dans les calculs passifs, mais leur suppression complète comme clés
de cache et comme contrat XML nécessite encore les deux derniers chantiers.

## Conclusion

La priorité architecturale n'est pas de supprimer toutes les clés UI d'un coup.
Elle est d'éliminer les signatures trompeuses des lecteurs déjà Core-driven.
Les catégories A constituent le prochain lot sûr : elles réduisent la fausse
autorité du `gid` sans toucher au collage ni au XML. Les catégories B et C
documentent précisément pourquoi la structure UI existe encore et dans quel
ordre elle pourra être retirée.
