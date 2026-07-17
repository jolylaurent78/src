# MIG-GEO-012C1 — Convention applicative `core_group_id`

## Signatures migrées

```python
_get_core_group_id_for_triangle_index(tri_index) -> Optional[str]
_get_projected_elements_for_core_group(core_group_id) -> Tuple[Dict, ...]
_group_centroid(core_group_id) -> Optional[np.ndarray]
```

Le premier helper suit exclusivement `tri_index -> topoElementId ->
TopologyWorld.get_group_of_element() -> find_group()`. Il ne lit aucun groupe
UI. Le second résout localement le `TopologyGroup` canonique avant d'appeler la
fonction pure `get_projected_elements(topology_group, index)`.

## Appelants de `_group_centroid` migrés

* `_ctx_rotate_selected()` : `idx -> core_group_id` avant le calcul du pivot ;
  le gid UI ne reste utilisé que pour l'oracle/fallback de préparation existant.
* `_ctx_flip_selected()` : `idx -> core_group_id` pour le pivot.
* sélection centre / MOVE : réutilise `move_members["core_group_id"]`.

## Ponts legacy conservés

`_get_projected_elements_for_ui_group(ui_gid)` reste un adaptateur explicite
et délègue désormais à `_get_projected_elements_for_core_group(core_group_id)`.
Ses lecteurs restants sont les contours géométriques legacy, l'assistance de
collision/snap et la bbox écran de dégrouper. `_get_group_of_triangle()` reste
utilisé par suppression, dégrouper, collage et les oracles/fallbacks des
opérations géométriques.

Aucun objet `TopologyGroup` ne circule désormais entre méthodes applicatives
migrées : il reste local à la résolution de l'index ou à la fonction pure de
projection.

## Validation

```text
python -m compileall -q .
python -m pytest tests/test_mig_geo001_group_linking.py tests/test_topology_comparison.py tests/test_chemins_balise_ref_core.py -q
```

Résultat : `26 passed`. Le test de géométrie de groupe vérifie désormais le
barycentre par `canonical_group_id`, sans passer de gid UI à Core.
