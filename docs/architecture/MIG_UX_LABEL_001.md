# MIG-UX-LABEL-001 — Label visuel Core-first

## Résultat

Le texte rouge central est maintenant une projection UX unique :

```text
TopologyElement.meta["triRank"] + TopologyElement.pose.mirrored
    -> "T17" ou "T17S"
```

Il ne dépend plus de `last_drawn["id"]`.

## Helper unique

`TriangleViewerManual._build_triangle_display_label(entry, world=None)` :

1. résout l'élément depuis `entry["topoElementId"]` ;
2. exige `element.meta["triRank"]` ;
3. lit le miroir via `element.get_pose()` ;
4. retourne `T<triRank>` avec le suffixe `S` si nécessaire.

L'absence de `topoElementId`, d'élément Core ou de `triRank` provoque une
erreur explicite. Il n'existe aucun fallback vers `entry["id"]`.

## Lecteurs migrés

- `_redraw_from()` (Canvas) ;
- le chemin de dessin utilisé pour l'export PDF.

`_draw_triangle_screen()` reçoit désormais `tri_label`, soit le texte UX déjà
construit. Police, couleur, taille, position et suffixe restent inchangés.

## Usages restants de `id`

`id` reste dans la projection uniquement pour les structures de compatibilité :

- index catalogue de `CanvasObjectsCollection` ;
- listbox, mots associés et filtrage de scénarios ;
- lecture/écriture XML historique ;
- diagnostics et tests.

Aucun de ces usages ne construit le label rouge.

## Validation

Les tests vérifient un triangle normal (`T1`) et miroir (`T1S`) à partir de la
pose Core, en plus des parcours de création, projection, XML et
transformations déjà couverts par les tests ciblés.
