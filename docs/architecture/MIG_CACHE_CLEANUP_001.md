# MIG-CACHE-CLEANUP-001 — Retrait des doublons Core de `last_drawn`

## Résultat

Les entrées runtime de `CanvasObjectsCollection.entries`,
`TriangleViewerManual._last_drawn` et `ScenarioAssemblage.last_drawn` ne
produisent plus les clés suivantes :

- `orient` ;
- `topoGroupId` ;
- `mirrored`.

Le contrat normal est désormais :

```python
{
    "topoElementId": "T28",
    "pts": {"O": ..., "B": ..., "L": ...},
    "id": 28,
    "labels": (...),
    # caches UI paresseux éventuels : _pick_pts, _pick_poly
}
```

## Changements

- `PlacedTriangle.toLegacyDict()` ne sérialise plus `mirrored`.
- `buildLegacyLastDrawnFromTopology()` ne projette plus `topoGroupId`.
- La création manuelle ne crée plus `orient`, `mirrored` ni `topoGroupId`.
- Le chargeur XML conserve le contrôle du snapshot `TopologyWorld` : il ne
  recopie plus `mirrored` ni `topoGroupId` vers l'entrée. L'attribut XML
  `mirrored` est comparé à la pose restaurée et une divergence est signalée.
- Le writer XML écrit `mirrored` à partir de
  `TopologyWorld.getElementPose(topoElementId)`.
- Le diagnostic F11/GEO-ORIENT résout désormais élément, orientation, miroir
  et groupe directement depuis le Core. Ses attributs XML de diagnostic sont
  explicitement nommés `coreGroupId` lorsque nécessaire.

## Compatibilité et occurrences volontairement conservées

- Le format XML v4 garde l'attribut `mirrored` pour compatibilité. Il ne
  devient jamais une donnée de projection : le snapshot Core est autoritaire.
- `PlacedTriangle.fromLegacyDict()` accepte encore une clé historique
  `mirrored`; aucune sortie moderne de `toLegacyDict()` ne la produit.
- Les clés `topoGroupId` des guides, de l'horloge et des traits sont hors
  `last_drawn` et restent des références Core propres à ces structures.
- L'ancien cache local AUTO est explicitement hors périmètre.

## Validation

Les tests ciblés couvrent la projection automatique, la création manuelle,
MOVE, ROTATE, FLIP, XML et le diagnostic F11. Un test de round-trip XML
vérifie en particulier que le miroir est écrit et restauré via la pose Core,
alors que l'entrée projetée reste dépourvue de `mirrored`.
