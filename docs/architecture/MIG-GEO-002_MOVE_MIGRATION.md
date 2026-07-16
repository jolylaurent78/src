# MIG-GEO-002 — MOVE : membres résolus depuis le Core

## Objet

Cette étape remplace la source utilisée pour déterminer les triangles
effectivement translatés par un geste `MOVE` manuel. Elle ne modifie ni les
calculs de translation, ni l'interface, ni le comportement de collage,
rotation, miroir ou XML.

Le chemin actif est désormais :

```text
triangle sélectionné
    -> topoElementId
    -> TopologyWorld.get_group_of_element()
    -> CoreGroup.element_ids
    -> get_last_drawn_entries_for_core_group()
    -> entrées _last_drawn translatées
```

`group_id` et `groups[gid]["nodes"]` restent présents car des contrats legacy
restent actifs (aide au collage, bbox, synchronisation de pose et structures
de groupe UI). Ils ne servent plus à choisir les entrées translatées pendant
un `move_group` normal.

## Point d'entrée et état du drag

Les quatre entrées dans `TriangleViewerManual._on_canvas_left_down` qui créent
un état `mode="move_group"` appellent
`_prepare_mig_geo_move_members(tri_index, legacy_group_id)`.

Le résultat mémorisé dans `_sel` contient :

- `core_group_id` : identifiant canonique du groupe dans `TopologyWorld` ;
- `move_member_entries` : entrées `_last_drawn` obtenues par le Core ;
- `orig_group_pts` : snapshot des mêmes entrées, utilisé par le rollback.

`_on_canvas_left_move` transmet ensuite uniquement
`move_member_entries` à `_move_group_world`. La boucle de translation applique
le delta monde inchangé à ces entrées. Les formules et le rythme de redraw ne
changent pas.

## Contrôle d'équivalence et logs

Chaque démarrage de MOVE écrit dans `logs/mig_geo.log` :

```text
[MOVE] Triangle sélectionné=T17
[MOVE] topoElementId=T17
[MOVE] LegacyGroupId=1
[MOVE] CoreGroupId=G1
[MOVE] LegacyMembers=T17,T18
[MOVE] CoreMembers=T17,T18
[MOVE] LegacyMembers == CoreMembers: OK
```

La comparaison porte sur les ensembles d'identifiants effectivement résolus
dans `_last_drawn`. L'ordre des listes n'est pas une condition métier de la
translation rigide.

## Cas de liaison Core indisponible

Si le triangle ne possède pas de `topoElementId`, si le monde actif est absent,
ou si aucun membre UI ne peut être résolu depuis le groupe Core, le code écrit
un avertissement `[MOVE] Fallback legacy`. Il réutilise alors les entrées
legacy pour ne pas modifier le comportement d'un scénario incohérent.

Ce fallback est une protection de compatibilité, pas une seconde source de
vérité prévue pour les scénarios validés. Un audit F8 en erreur ou un
`MISMATCH` dans ce log doit être traité comme un diagnostic à analyser avant
de poursuivre les migrations ; aucune correction automatique n'est effectuée.

## Périmètre explicitement non modifié

- rotation et miroir ;
- collage, fusion, déconnexion et dégroupement ;
- représentation ou persistance XML ;
- calcul des poses, delta monde, centroid, snapping et rendu ;
- suppression des champs `group_id` et des groupes UI.

## Vérification automatisée

Le test `test_move_members_are_resolved_from_core_group` vérifie que le
préparateur de MOVE utilise le groupe canonique du `TopologyWorld` et renvoie
les entrées `_last_drawn` associées. Les tests MIG-GEO-001 existants continuent
de couvrir l'index Core -> UI et les écarts de composition signalés par F8.
