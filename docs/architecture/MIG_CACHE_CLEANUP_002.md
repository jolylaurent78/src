# MIG-CACHE-CLEANUP-002 — Retrait de `labels` de `last_drawn`

## Résultat

Les entrées runtime de `CanvasObjectsCollection.entries`,
`TriangleViewerManual._last_drawn` et `ScenarioAssemblage.last_drawn` ne
contiennent plus `labels`. Le contrat de projection est maintenant :

```python
{
    "id": 28,
    "topoElementId": "T28",
    "pts": {"O": ..., "B": ..., "L": ...},
    # caches de pick UI facultatifs : _pick_pts, _pick_poly
}
```

## Source de vérité et helper

`TopologyElement.vertex_labels` est l'unique source de vérité des noms de
sommets. `TriangleViewerManual._get_core_vertex_labels(entry, world=None)`
résout l'élément à partir de `entry["topoElementId"]`, vérifie que les trois
labels existent, puis les retourne dans l'ordre O/B/L.

Le Canvas et le PDF utilisent ce helper avant d'appeler
`_draw_triangle_screen`. Le texte, la police, l'inset et l'ordre restent donc
identiques ; seule la provenance change.

## Producteurs retirés

- `PlacedTriangle.toLegacyDict()` ne projette plus `PlacedTriangle.labels`.
- `_place_dragged_triangle()` ne construit plus de copie `labels`.
- `_bind_canvas_objects()` et les projections Core retirent aussi `labels`
  d'une entrée historique déjà présente, comme les autres doublons Core.
- `loadScenarioXml()` ne reconstruit plus les labels à partir du DataFrame.
  Le `topoSnapshot` restaure les `TopologyElement.vertex_labels` avant que la
  projection ne soit liée au Canvas.

## Occurrences conservées

- `PlacedTriangle.labels` reste une donnée de travail interne au simulateur,
  utilisée pour créer les `TopologyElement`. Elle n'est plus exportée vers
  `last_drawn`.
- Les paramètres locaux `labels` de `_draw_triangle_screen` et d'EdgeChoice
  ne sont pas des clés de projection persistante.
- `TopologyElement.vertex_labels` est la donnée Core autoritaire.
- Les formats legacy peuvent encore être lus par `PlacedTriangle.fromLegacyDict`;
  aucune sortie runtime moderne ne propage cette clé.

## Validation

Les tests couvrent la projection automatique, le dépôt manuel, le chargement
XML, les transformations Core-first et le split. Ils vérifient que les entrées
ne contiennent pas `labels` et que les libellés restent accessibles depuis le
Core. La recherche source ne laisse aucun lecteur/écrivain de
`entry["labels"]` ou `entry.get("labels")` dans les projections runtime.
