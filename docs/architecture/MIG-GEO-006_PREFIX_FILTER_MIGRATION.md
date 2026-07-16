# MIG-GEO-006 — Migration du filtrage de préfixe

## Résultat

`_scenario_prefix_edge_steps()` est maintenant le chemin fonctionnel Core.
Il utilise exclusivement `ScenarioAssemblage.tri_ids` et
`TopologyWorld.attachments`.

Il ne lit pas `groups`, `nodes`, `edge_in`, `edge_out`, `group_pos` ou
`group_id`. Il n'appelle pas les normaliseurs de groupes.

La représentation legacy de préfixe a été purgée. Ce flux ne comporte plus
d'oracle ni de lecture des normaliseurs legacy.

## Signature d'une étape

Pour chaque paire ordonnée de `tri_ids`, le code construit un tuple trié de
toutes les signatures d'attachments reliant exactement les deux éléments Core.

```text
(kind, endpoint_triangle_deja_assemble,
       endpoint_triangle_ajoute, semantic_params)
```

Un endpoint contient `(feature_type, element_id, index)`. Les endpoints sont
réorientés selon l'étape `triangle_a -> triangle_b`, même si l'attachment est
stocké dans le sens inverse. Le tri rend le résultat indépendant de l'ordre de
stockage.

`edge-edge`, `vertex-edge` et `vertex-vertex` sont pris en charge. Une étape
peut donc contenir plusieurs contraintes complémentaires.

## Décision fonctionnelle

Le filtre décide exclusivement avec les signatures Core. Il compare le préfixe
ordonné `tri_ids`, puis les signatures d'étapes construites depuis les
attachments. Aucun calcul legacy ni log de divergence temporaire ne subsiste.

`tri_ids` reste la source de vérité de l'ordre historique. Une étape sans
attachment entre deux triangles renvoie `None`.

## Purge des normaliseurs

MIG-GEO-007 a supprimé le dernier consommateur des normaliseurs de groupes.
Le filtre de préfixe et la comparaison visuelle ne dépendent plus de cette
représentation legacy.

## Vérification

```text
python -m pytest tests/test_topology_comparison.py \
  tests/test_mig_geo001_group_linking.py tests/test_logging_utils.py -q
```

Résultat : `19 passed` avec les contrôles finaux de comparaison Core.
