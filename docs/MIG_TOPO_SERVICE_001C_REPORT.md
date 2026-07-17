# MIG-TOPO-SERVICE-001C — Candidats Boundary non représentables

Date : 2026-07-17  
Statut : correction de sûreté ciblée terminée.

## Fichiers modifiés

| Fichier | Rôle |
|---|---|
| `src/assembleur_edgechoice.py` | garde explicite avant `getNodeType()`. |
| `src/assembleur_tk.py` | `None` de EdgeChoice traité comme rejet de paire. |
| `tests/test_edgechoice_boundary_rejections.py` | cas vertex-edge non représentable, cas normal et poursuite de boucle. |
| `docs/MIG_TOPO_SERVICE_001C_REPORT.md` | présent rapport. |

## Garde finale dans `buildEdgeChoiceEptsFromBest()`

```python
mAId = getEquivalentNodeInElement(world, mATmpId, elementIdSrc)
tAId = getEquivalentNodeInElement(world, tATmpId, elementIdDst)
if mAId is None or tAId is None:
    if debug:
        print("[EDGECHOICE] candidate rejected: concept anchor has no physical ...")
    return None

src_anchor_vkey = world.getNodeType(mAId)
dst_anchor_vkey = world.getNodeType(tAId)
```

La garde couvre exactement l'absence d'une occurrence sommet physique du node conceptuel dans l'élément propriétaire du segment. Les erreurs de structure, de points, de `topoElementId`, d'arête, de mapping ou de géométrie dégénérée continuent à lever leurs erreurs explicites : aucun `except Exception` global n'a été ajouté.

## Convention de rejet dans `_overlap_topo_for_pair()`

Le booléen retourné par `_overlap_topo_for_pair()` signifie déjà « paire à rejeter » : `True` provoque `continue` dans la boucle des candidates. Le résultat `None` de `buildEdgeChoiceEptsFromBest()` retourne donc désormais `True` depuis ce helper local. La boucle reste active et les paires suivantes sont évaluées.

Aucun attachment n'est créé pour cette paire : `createTopologyAttachments()` n'est atteint que lorsque `buildEdgeChoiceEptsFromBest()` a produit un `EdgeChoiceEpts`.

## Tests ajoutés

`tests/test_edgechoice_boundary_rejections.py` couvre :

1. un vertex de T2 attaché au milieu d'une arête de T1 ; le ConceptNode de T2.B n'a aucune occurrence vertex dans T1 et le builder retourne `None` sans exception ;
2. un anchor représentable physiquement dans l'élément porteur ; le builder produit toujours `EdgeChoiceEpts` ;
3. une première paire simulée non représentable, suivie d'une paire valide ; `_update_edge_highlights()` poursuit l'évaluation et conserve une proposition.

## Validation exécutée

```text
python -m py_compile src/assembleur_edgechoice.py src/assembleur_tk.py
python -m pytest tests/test_edgechoice_boundary_rejections.py tests/test_topology_boundary_services.py tests/test_mig_geo001_group_linking.py tests/test_topology_comparison.py tests/test_logging_utils.py -q
```

Résultat : **38 passed**.

Recherche statique : les deux appels `world.getNodeType(mAId)` et `world.getNodeType(tAId)` sont immédiatement précédés de la garde `mAId is None or tAId is None`. Aucune clause `except Exception` n'a été ajoutée autour de cette logique. `git diff --check` est propre.

## Périmètre et suite séparée

Cette passe ne modifie ni `EdgeChoiceEpts`, ni les scores, ni les segments Boundary, ni l'anti-chevauchement, ni les attachments, ni le release. Une évolution future, hors de ce chantier, pourra conserver l'identité complète de `BoundarySegment` jusqu'à EdgeChoice afin de ne plus résoudre le propriétaire d'arête depuis une géométrie projetée.
