# MIG-GEO-006 — Migration du filtrage de préfixe

## Résultat

`_scenario_prefix_edge_steps()` est maintenant le chemin fonctionnel Core.
Il utilise exclusivement `ScenarioAssemblage.tri_ids` et
`TopologyWorld.attachments`.

Il ne lit pas `groups`, `nodes`, `edge_in`, `edge_out`, `group_pos` ou
`group_id`. Il n'appelle pas les normaliseurs de groupes.

L'ancienne implémentation est conservée sans refactoring sous le nom
`_scenario_prefix_edge_steps_legacy()`. Elle reste l'oracle temporaire et est
le seul chemin de ce flux à appeler les normaliseurs legacy.

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

## Double calcul

Le filtre décide avec les signatures Core. Le calcul legacy est encore exécuté
pour le scénario de référence et les candidats comparables. Les décisions
`keep/reject` sont confrontées.

- Équivalence : `[MIG-GEO-006] Prefix filter legacy/core: OK`
- Divergence : référence, candidat, préfixe, étapes legacy, étapes Core et
  décisions sont journalisés.
- Échec de l'oracle : journalisé sans empêcher la décision Core.

`tri_ids` reste la source de vérité de l'ordre historique. Une étape sans
attachment entre deux triangles renvoie `None`.

## Vérification

```text
python -m pytest tests/test_topology_comparison.py \
  tests/test_mig_geo001_group_linking.py tests/test_logging_utils.py -q
```

Résultat : `16 passed`.
