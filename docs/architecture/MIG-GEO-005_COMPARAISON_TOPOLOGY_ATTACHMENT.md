# MIG-GEO-005 — Comparaison de scénarios via `TopologyAttachment`

## Résultat

La comparaison dynamique référence/scénario actif ne dépend plus de
`groups[*]["nodes"]`, `edge_in` ou `edge_out` pour décider les triangles à
surligner. Elle compare maintenant les attachments des deux `TopologyWorld`.

Le calcul legacy est conservé temporairement comme oracle. Une divergence entre
les ensembles d'indices marqués produit un warning `[TOPO-COMPARE]` dans
`logs/mig_geo.log`; il n'est ni bloquant ni correctif.

## Fichiers modifiés

- `src/assembleur_topology_comparison.py` : signatures et différences Core
  pures, sans Tk ni structures UI.
- `src/assembleur_core.py` : métadonnées minimales de parcours sur
  `ScenarioAssemblage`.
- `src/assembleur_tk.py` : nouveau calcul, double calcul, logs et alimentation
  des métadonnées à la génération/duplication.
- `tests/test_topology_comparison.py` : signatures des trois types et scénario
  identique.

## Signature canonique

Une feature est représentée par :

```text
(feature_type, element_id, index)
```

La signature est :

```text
(kind, endpoints_triés, semantic_params)
```

Les endpoints sont triés, donc le résultat est indépendant du côté mobile,
du côté statique, de l'ordre de création et de `attachment_id`/`source`.

- `edge-edge` conserve `mapping` (`direct` ou `reverse`) ;
- `vertex-edge` conserve `t` arrondi à 12 décimales et `edgeFrom`, car ils
  distinguent le point effectivement attaché sur l'arête ;
- `vertex-vertex` ne porte pas de paramètre sémantique supplémentaire.

## Comparaison et surlignage

Les maps `signature -> element_ids` sont construites depuis
`TopologyWorld.attachments`. Les signatures exclusives de référence ou de
courant donnent les `topoElementId` des triangles impliqués. Ces IDs sont
résolus vers `last_drawn` afin de conserver le surlignage : triangle modifié
et triangle voisin impliqué.

Le mode `DEBUG_TOPOLOGY_COMPARISON = True` journalise les signatures des deux
mondes, les différences et les éléments concernés. Il est volontairement
temporaire et local à MIG-GEO.

## Métadonnées de parcours

`ScenarioAssemblage` reçoit :

- `first_triangle_id` ;
- `traversal_direction`.

`tri_ids` reste l'ordre effectif principal. À chaque génération auto, le
viewer renseigne les deux champs depuis cet ordre et l'option de dialogue. Ces
métadonnées n'appartiennent pas au Core et ne sont pas encore sérialisées XML,
conformément au périmètre de ce chantier.

## Limites connues

- Les libellés de fork `#5 = #4 + (6)` ne sont pas modifiés.
- La persistance XML reste inchangée.
- `edge_in/out` est toujours écrit et lu par les fonctionnalités hors
  comparaison (simulation, filtrage, XML et normalisation).
- Les scénarios legacy sans attachments produisent une comparaison Core vide ;
  le double calcul le signale explicitement.

## Vérification

Tests ciblés :

```text
13 passed
```

Ils couvrent le scénario identique et les signatures `edge-edge`,
`vertex-edge`, `vertex-vertex`, y compris l'indépendance à l'ordre des
endpoints.
