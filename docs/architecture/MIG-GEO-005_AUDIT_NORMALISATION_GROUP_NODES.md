# Audit de suppression des normaliseurs de groupes

> Statut : **DO NOT DELETE** — 16 juillet 2026.
>
> Cet audit est en lecture seule : aucune suppression de code n'a été faite.

## APPELANTS IDENTIFIÉS

Recherche exhaustive effectuée sur les fichiers suivis, ignorés et cachés :

```text
rg -n -uuu -S "_normalize_group_nodes|_normalize_all_groups" .
```

| Ligne observée dans `src/assembleur_tk.py` | Référence | Nature |
|---:|---|---|
| 4553 | `self._normalize_all_groups(scen)` | appel de production : signature legacy des connexions |
| 4597 | `def _normalize_group_nodes(...)` | définition |
| 4665 | `def _normalize_all_groups(...)` | définition |
| 4673 | `self._normalize_group_nodes(scen, gid)` | appel interne direct |
| 9930 | `self._normalize_all_groups(scen)` | appel de production : filtrage de scénarios automatiques |

Les autres occurrences sont documentaires ou un fichier `.pyc` dans
`src/__pycache__`; elles ne sont pas des appelants source.

Une recherche ciblée sur `getattr`, `setattr`, `monkeypatch` et `patch`
contenant ces noms ne retourne aucun résultat. Aucune référence n'est trouvée
dans `tests/`. Il n'y a pas de répertoire `tools/` ou `scripts/` dans cet arbre.
Ces méthodes sont des méthodes d'instance et ne sont pas importées : aucun
indice d'import dynamique n'a été trouvé.

### Appelant 1 : oracle legacy de comparaison MIG-GEO-005

`_scenario_connections_signature_legacy()` appelle le normaliseur à la ligne
4553. Cette signature legacy reste volontairement active comme oracle du
double calcul MIG-GEO-005, alors que la comparaison `TopologyAttachment` est
autoritaire.

```text
_redraw_from() (appel à la ligne 6557)
  -> _update_current_scenario_differences()
       -> _scenario_connections_signature_legacy(reference / courant)
            -> _normalize_all_groups()
                 -> _normalize_group_nodes() pour chaque groupe
```

Le changement de scénario appelle également
`_update_current_scenario_differences()` (ligne 11358). Ce chemin de production
est donc réel tant qu'une comparaison avec référence est active.

### Appelant 2 : filtrage UI des scénarios automatiques

`_scenario_prefix_edge_steps()` appelle le normaliseur à la ligne 9930 afin de
parcourir les paires legacy `(edge_out, edge_in)`.

```text
Action UI « Filtrer les scénarios »
  -> _ctx_filter_scenarios() (ligne 9915)
       -> _filter_auto_scenarios_by_prefix_edges()
            -> _scenario_prefix_edge_steps(active / candidat)
                 -> _normalize_all_groups()
                      -> _normalize_group_nodes() pour chaque groupe
```

Les appels pour le scénario actif et les candidats sont aux lignes 9981 et
10004. Ce flux UI est donc dépendant du contrat legacy de chaîne.

## ANALYSE MÉTIER

### `_normalize_group_nodes(scen, gid)`

Cette méthode valide et **modifie** le contrat legacy
`scen.groups[gid]["nodes"]`. Elle ne consulte pas `TopologyWorld`.

1. Elle vérifie le scénario, le groupe, `nodes`, le type liste et chaque node.
2. Elle exige `tid` et le convertit en `int`.
3. Elle retire `vkey_in` et `vkey_out` s'ils subsistent.
4. Elle valide `edge_in` et `edge_out` contre `OB`, `BL`, `LO` ou `None`.
5. Elle exige, pour chaque paire consécutive, `edge_out` à gauche et `edge_in`
   à droite.
6. Elle réaffecte `nodes` et le groupe dans `scen.groups`.

| Structure | Rôle effectif | Dépendance |
|---|---|---|
| `group["nodes"]` | support validé et réaffecté | directe |
| `tid` | converti en entier ; index de `scen.last_drawn` | directe |
| `edge_in`, `edge_out` | validation de la connectivité de chaîne | directe |
| `vkey_in`, `vkey_out` | suppression de compatibilité legacy | directe |
| `group_pos` | non lu, non écrit | aucune |

Sans cette méthode, l'appel actuel de `_normalize_all_groups()` échouerait par
`AttributeError`. Si les appelants étaient aussi supprimés, les deux flux
perdraient la conversion de `tid`, la purge de `vkey_*` et les diagnostics
précoces sur les liaisons internes incomplètes.

### `_normalize_all_groups(scen)`

Cette méthode est le dispatcher du contrôle précédent : elle vérifie le
scénario, récupère `scen.groups`, retourne s'il est vide, puis appelle
`_normalize_group_nodes(scen, gid)` pour tous les groupes. Elle n'accède pas
directement à `group_pos`, `edge_in`, `edge_out` ou `vkey_*`, mais délègue
explicitement ce traitement.

## VÉRIFICATION DE L'IMPACT RUNTIME

| Workflow | Appel direct | Impact constaté |
|---|---|---|
| Assemblage manuel | non | les groupes produits peuvent être consommés plus tard par la comparaison legacy |
| Génération automatique | non dans le moteur observé | les scénarios sont consommés par le filtrage UI actif |
| Chargement XML | non direct | les scénarios chargés peuvent ensuite être comparés ou filtrés |
| Sauvegarde XML | non trouvé | aucune dépendance directe démontrée |
| Comparaison de scénarios | oui, par l'oracle legacy | dépendance de production réelle |
| Fonctionnalité UI | oui, par le filtrage | dépendance de production réelle |

L'absence d'appel direct dans certains workflows ne démontre pas l'absence de
dépendance indirecte des scénarios qu'ils produisent ou chargent.

## RISQUES DE SUPPRESSION

1. Rupture de l'oracle de validation MIG-GEO-005, qui compare temporairement
   les écarts legacy et Core.
2. Rupture du filtrage de scénarios automatiques, toujours fondé sur
   `groups[*]["nodes"]`, `edge_out` et `edge_in`.
3. Disparition des diagnostics explicites sur `tid` et les arêtes de liaison.
4. Suppression du nettoyage effectif de `vkey_in` et `vkey_out`.

## CONDITIONS POUR UNE SUPPRESSION FUTURE

Réexaminer la suppression seulement après :

1. retrait volontaire du double calcul legacy MIG-GEO-005, après une période
   de validation suffisante de la comparaison `TopologyAttachment` ;
2. migration ou retrait de `_scenario_prefix_edge_steps()` et de son filtrage
   UI fondé sur les champs `groups/nodes` et `edge_in/out`.

Une nouvelle recherche exhaustive devra alors être exécutée avant toute
suppression.

## CONCLUSION

**DO NOT DELETE.**

`_normalize_all_groups()` a deux appelants de production et
`_normalize_group_nodes()` est son appel interne pour chaque groupe. Les deux
méthodes restent atteignables par des workflows UI réels ; aucune suppression,
aucun nettoyage de commentaires ni d'import n'a été appliqué.
