# MIG-GEO-003 — ROTATE et FLIP : membres résolus depuis le Core

## Objet et périmètre

MIG-GEO-003 étend le contrat introduit par MIG-GEO-002 aux opérations manuelles
de rotation et d'inversion. La seule migration concerne la détermination des
triangles ciblés : la liste des membres est désormais construite depuis le
`TopologyWorld`, puis projetée vers `_last_drawn`.

```text
triangle sélectionné
    -> topoElementId
    -> TopologyWorld.get_group_of_element()
    -> CoreGroup.element_ids
    -> get_last_drawn_entries_for_core_group()
    -> triangles ROTATE / FLIP
```

Ne sont pas modifiés : pivots, matrices de rotation ou de réflexion, delta ou
angle utilisateur, synchronicité de rendu, synchronisation de pose, XML,
collage, dégroupement, MOVE et groupes UI legacy.

## Points d'entrée analysés

| Opération | Point d'entrée | Résolution des membres |
|---|---|---|
| ROTATE interactif | `_ctx_rotate_selected()` puis le mode `rotate_group` | Core au démarrage, liste conservée dans `_sel["rotate_member_entries"]` |
| ROTATE d'orientation | `_ctx_orient_OL_north()` / `_ctx_orient_BL_north()` via `_ctx_orient_segment_north()` | Core pour le chemin manuel |
| FLIP | `_ctx_flip_selected()` | Core lors de l'exécution directe |

Les scénarios automatiques appliquent un référentiel global partagé. Ils ne
parcourent pas `group_id` ou `groups[gid]["nodes"]` pour sélectionner les
membres d'un groupe ; ils ne relevaient donc pas de cette migration.

## Implémentation

`_prepare_mig_geo_operation_members(operation, tri_index, legacy_group_id)`
est le préparateur commun aux opérations `MOVE`, `ROTATE` et `FLIP`.

Il conserve le groupe UI seulement pour comparer l'état historique, puis
résout le groupe canonique et ses éléments via le Core. Si la projection UI
est indisponible, il produit un avertissement et applique le fallback legacy
préexistant afin de ne pas modifier le comportement d'un état incohérent.

Pour ROTATE interactif, le snapshot de rollback et la prévisualisation par
souris utilisent `rotate_member_entries`. Pour FLIP et les orientations
manuelles, la boucle qui applique la transformation parcourt directement les
entrées Core résolues.

Les groupes UI restent utilisés après la transformation pour les contrats non
migrés : recalcul de bbox et synchronisation des poses vers le Core.

## Instrumentation

Chaque opération écrit dans `logs/mig_geo.log` :

```text
[ROTATE] Triangle sélectionné=T17
[ROTATE] topoElementId=T17
[ROTATE] LegacyGroupId=1
[ROTATE] CoreGroupId=G1
[ROTATE] LegacyMembers=T17,T18
[ROTATE] CoreMembers=T17,T18
[ROTATE] LegacyMembers == CoreMembers: OK

[FLIP] Triangle sélectionné=T17
[FLIP] topoElementId=T17
[FLIP] LegacyGroupId=1
[FLIP] CoreGroupId=G1
[FLIP] LegacyMembers=T17,T18
[FLIP] CoreMembers=T17,T18
[FLIP] LegacyMembers == CoreMembers: OK
```

`KO` signale une divergence. `[ROTATE] Fallback legacy` ou `[FLIP] Fallback
legacy` signale une absence de projection exploitable. Ces événements ne
modifient aucune donnée et doivent être rapprochés du rapport F8.

## Usages legacy restant volontairement hors périmètre

Les usages de `group_id` et `groups[gid]["nodes"]` qui servent à la sélection,
au snapping, au collage, à la déconnexion, à la fusion, au dégroupement, aux
groupes UI, aux bboxes et à la synchronisation de pose ne sont pas supprimés.
Ils ne servent plus à déterminer les membres géométriquement transformés par
les chemins ROTATE et FLIP migrés ici.

## Vérification automatisée

`test_rotate_and_flip_prepare_members_from_core_group` vérifie que le mode
ROTATE conserve le groupe Core résolu et que FLIP s'exécute sur la projection
Core. Les tests précédents couvrent l'index Core -> UI, MOVE et l'audit F8.
