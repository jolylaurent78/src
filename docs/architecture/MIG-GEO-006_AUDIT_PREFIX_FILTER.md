# MIG-GEO-006-AUDIT-PREFIX-FILTER

> Statut : audit documentaire, sans modification de code.
>
> Verdict : **PARTIELLEMENT**. Les connexions legacy peuvent être représentées
> par `TopologyAttachment`, mais les attachments ne portent pas l'ordre requis.

## Résumé exécutif

Le filtre conserve les scénarios automatiques partageant, jusqu'au triangle
cliqué, le même préfixe ordonné de `tri_ids` et les mêmes liaisons de chaîne.

Le préfixe `tri_ids` est un historique de parcours. Les liaisons sont une
information topologique locale. La fonctionnalité est donc **C. Hybride
topologie + historique de génération**.

## Cartographie des appels

```text
Menu contextuel « Filtrer les scénarios... »
  -> _ctx_filter_scenarios()
       -> _filter_auto_scenarios_by_prefix_edges(clicked_tid)
            -> _scenario_prefix_edge_steps(active, idx)
            -> _scenario_prefix_edge_steps(candidate, idx)
                 -> _normalize_all_groups()
                      -> _normalize_group_nodes()
```

### `_ctx_filter_scenarios()`

Cette méthode UI lit `_ctx_target_idx`, résout l'id avec
`_last_drawn[idx]["id"]`, demande confirmation puis lance le filtre. Le
triangle cliqué sert à obtenir une position dans `active.tri_ids`, non à
sélectionner directement un groupe Core ou un attachment.

### `_filter_auto_scenarios_by_prefix_edges()`

Le code :

1. s'arrête si le scénario actif n'est pas automatique ;
2. cherche le triangle cliqué dans `active.tri_ids` ;
3. calcule `idx` et `prefix = active.tri_ids[:idx + 1]` ;
4. obtient les liaisons legacy du préfixe actif ;
5. conserve toujours le scénario actif et les scénarios manuels ;
6. conserve un autre scénario auto si son préfixe `tri_ids` est identique et si
   ses liaisons de préfixe sont identiques.

Sens métier : l'utilisateur filtre les branches automatiques ayant suivi le
**même parcours de triangles et les mêmes connexions réalisées** jusqu'au
triangle cliqué. Ce n'est pas un filtre de topologie finale équivalente.

### `_scenario_prefix_edge_steps()`

La valeur retournée est :

```python
[(edge_out_0, edge_in_1), ..., (edge_out_(n-1), edge_in_n)]
```

`n` vaut `upto_index`. L'indice 0 retourne `[]`; l'indice 3 retourne les liens
`0→1`, `1→2` et `2→3`.

Exemple :

```text
nodes : T10 --(OL, BL)-- T12 --(OB, LO)-- T8
_scenario_prefix_edge_steps(scen, 2) == [("OL", "BL"), ("OB", "LO")]
```

Un tuple décrit les deux arêtes engagées dans une connexion entre deux nodes
consécutifs. Il ne contient ni ids de triangles ni date de création ; il dépend
de l'ordre de chaîne retenu.

## Analyse des dépendances

| Donnée legacy | Utilisation réelle | Indispensable aujourd'hui ? | Remplaçable par `TopologyAttachment` ? |
|---|---|---|---|
| `nodes[]` | source de chaîne ; la plus longue est sélectionnée | oui | connexions oui, ordre non |
| `edge_out` | feature de la partie gauche | oui | oui, endpoint `edge` |
| `edge_in` | feature de la partie droite | oui | oui, endpoint `edge` |
| `group_id` | non lu directement | non | sans objet |
| `group_pos` | non lu directement | non | sans objet |
| ordre des groupes | implicite si égalité de longueur | commodité fragile | oui, à supprimer |
| ordre des triangles | `tri_ids`, critère principal | oui | non, pas par attachments seuls |

La méthode choisit le groupe ayant le plus de nodes. Si plusieurs groupes ont
la même longueur, le premier itéré reste retenu. `group_pos` n'est jamais lu.
Le code suppose un seul groupe : l'ordre des groupes est une commodité de
compatibilité, pas une vérité métier stable.

## Reconstruction via `TopologyWorld` / `TopologyAttachment`

### Réponse : PARTIELLEMENT

Un attachment porte `kind`, deux `TopologyFeatureRef` et des paramètres. Avec
une paire ordonnée de triangles **déjà connue**, on peut extraire les signatures
d'attachments reliant ces éléments. Cela remplace `edge_in` / `edge_out` par
une représentation plus générale.

En revanche, `TopologyWorld.attachments` ne porte ni l'ordre des triangles,
ni la position du clic dans cet ordre, ni la séquence des étapes, ni l'affectation
d'un attachment à une étape. La signature MIG-GEO-005 est volontairement
indépendante de l'ordre de création : elle compare un état, pas un historique.

| Type d'attachement | Rapport avec le tuple legacy |
|---|---|
| `edge-edge` | traduction directe par les endpoints d'arête et le mapping |
| `vertex-edge` | aucun équivalent exact dans le tuple legacy |
| `vertex-vertex` | aucun équivalent exact dans le tuple legacy |

## Métadonnées minimales nécessaires

La donnée indispensable est :

```python
tri_ids: list[int]
```

Elle existe déjà dans `ScenarioAssemblage`. `first_triangle_id` et
`traversal_direction` sont utiles au contexte, mais ne suffisent pas à
reconstruire une séquence arbitraire : ils ne remplacent pas `tri_ids`.

Pour une reconstruction exacte, il faut également une règle explicite liant la
transition ordonnée `(tri_ids[i], tri_ids[i + 1])` aux attachments concernés.
Une déduction par endpoints suffit seulement si le modèle est non ambigu.
Sinon, il faut conserver une séquence de signatures d'étapes par scénario.
`algo_id`, l'arête initiale et les options sont utiles au rejeu, mais non lus
par ce filtre.

## Chemin de migration recommandé

1. Conserver `tri_ids` comme source de vérité de l'ordre.
2. Définir et caractériser la signature Core d'une transition entre deux ids
   consécutifs, y compris zéro, un ou plusieurs attachments.
3. Effectuer un double calcul temporaire : préfixe legacy puis préfixe Core.
4. Traiter explicitement les scénarios anciens sans topologie ou métadonnées.
5. Après équivalence démontrée, retirer `groups`, `nodes`, `edge_in`, `edge_out`
   et la normalisation de ce flux.

## Conclusion

La dépendance legacy n'est pas supprimable immédiatement. Les `edge_in/out`
sont remplaçables ; l'historique de parcours ne l'est pas par les attachments
seuls. Verdict final : **PARTIELLEMENT**, catégorie **C. Hybride topologie +
historique de génération**.
