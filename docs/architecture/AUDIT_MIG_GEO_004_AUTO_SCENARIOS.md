# Audit ciblé préalable à la migration de `edge_in` / `edge_out`

## Résumé exécutif

**Recommandation : NO-GO pour supprimer ou désactiver maintenant les lectures
de `edge_in/out`. GO conditionnel pour préparer la migration par étapes.**

Les scénarios automatiques ont déjà un `TopologyWorld` indépendant et les
`TopologyAttachment` constituent bien la vérité topologique. Cependant :

- `edge_in/out` reste le support direct de la comparaison dynamique de
  connexions entre scénarios ;
- il sert aussi au filtrage des scénarios automatiques par préfixe ;
- le XML v4 persiste encore ces données de chaîne ;
- les métadonnées de reproductibilité ne sont pas complètes au niveau du
  scénario : `tri_ids` existe, mais l'ordre (`autoOrder`) n'est pas sérialisé,
  l'arête initiale, la version d'algorithme et les options ne sont pas portées
  par chaque scénario.

La migration est faisable à condition de séparer trois choses :

```text
Topologie finale        = TopologyWorld + TopologyAttachment
Ordre de génération     = métadonnées du scénario automatique
Présentation/compat XML = groups/nodes, temporairement conservés comme cache
```

L'ordre de parcours ne doit pas aller dans `TopologyWorld` : il est un
paramètre de l'algorithme, pas une propriété topologique de la composante
finale.

## Fichiers inspectés

| Fichier | Rôle inspecté |
|---|---|
| `src/assembleur_core.py` | `ScenarioAssemblage`, `TopologyFeatureRef`, `TopologyAttachment`, clonage et APIs Core |
| `src/assembleur_sim.py` | algorithme `AlgoQuadrisParPaires`, arbre de branches, création des mondes et des groupes legacy |
| `src/assembleur_tk.py` | dialogue/lancement simulation, cycle de vie UI, référence, comparaison, affichage et filtrage |
| `src/assembleur_io.py` | sauvegarde/chargement XML v4 des groupes et du snapshot Core |
| `tests/` | suite de non-régression existante |

## Carte des fonctions

| Fonction / zone | Fichier | Responsabilité | Lit `edge_in/out` | Écrit `edge_in/out` | Utilise `TopologyAttachment` | Statut |
|---|---|---|---|---|---|---|
| `TriangleViewerManual._simulation_assemble_dialog` | Tk | collecte paramètres, construit `tri_ids`, lance l'algo, ajoute les scénarios | Non | Non | Indirect | À migrer pour porter les métadonnées sur chaque scénario |
| `_simulation_get_tri_ids_by_order` | Tk | construit la séquence croissante/décroissante depuis la listbox | Non | Non | Non | À conserver ; source des paramètres de parcours |
| `AlgoQuadrisParPaires.run` | Sim | génération, candidats, forks, mondes indépendants, finalisation | Indirect (via groupe final) | Oui, à la finalisation legacy | Oui, crée/applique des attachments | À conserver ; retirer seulement les écritures legacy après migration |
| `_fill_group_vkeys_from_geometry` | Sim | déduit les arêtes communes des voisins de chaîne | Non | Oui | Non | C — écriture legacy post-topologie |
| injection `_chain_edge_in/out` | Sim | encode le raccord inter-quads pour la chaîne UI | Non | Oui | Attachments créés séparément | C — écriture legacy post-topologie |
| `_scenario_connections_signature` | Tk | construit la signature de comparaison à partir des voisins UI | Oui | Non | Non | À migrer en priorité vers attachments |
| `_update_current_scenario_differences` | Tk | compare référence/courant et marque triangles et voisins | Indirect | Non | Non | À conserver, source de signature à remplacer |
| `_scenario_prefix_edge_steps` | Tk | extrait la séquence d'arêtes d'un préfixe de parcours | Oui | Non | Non | À migrer vers métadonnées de parcours + signature attachments |
| `_filter_auto_scenarios_by_prefix_edges` | Tk | filtre les autos compatibles avec le préfixe actif | Indirect | Non | Non | À migrer après les deux précédentes |
| `_normalize_group_nodes` | Tk | valide `nodes`, maintient le contrat et retire `vkey_*` | Oui | Oui (suppression `vkey_*`) | Non | D — maintenance défensive, à nettoyer en dernier |
| `saveScenarioXml` | IO | persiste `groups/nodes` v4 | Oui | Oui, dans XML | snapshot Core séparé | C — compatibilité à maintenir pendant transition |
| `loadScenarioXml` | IO | recharge/valide les nœuds et réconcilie avec le Core | Oui | Oui, dans structures UI | importe snapshot Core | D/C — migration seulement avec format versionné |

## Flux réel de génération automatique

```text
DialogSimulationAssembler
  -> (algo_id, n, order, first_edge)
  -> _simulation_get_tri_ids_by_order(n, order)
  -> MoteurSimulationAssemblage.firstTriangleEdge
  -> AlgoQuadrisParPaires.run(tri_ids)
  -> premiers deux triangles / premier quadrilatère
  -> boucle par paires suivantes
  -> candidats edge-choice et test d'overlap
  -> forks / clones TopologyWorld
  -> feuilles survivantes
  -> ScenarioAssemblage indépendant par feuille
  -> groups/nodes legacy final
  -> ajout à self.scenarios + géométrie auto locale
```

### Paramètres et premier triangle

`_simulation_get_tri_ids_by_order` choisit :

- `forward` : les `n` premiers triangles de la listbox ;
- `reverse` : les `n` derniers, parcourus du dernier vers le premier.

Le premier triangle réel est donc `tri_ids[0]`. Il est aujourd'hui conservé
implicitement dans `ScenarioAssemblage.tri_ids`, mais pas sous un champ nommé
et explicite. `firstTriangleEdge` vaut `OL` ou `BL`, est transmis au moteur et
oriente la première pièce ; il est sauvegardé dans la configuration globale du
viewer, pas dans le scénario produit.

`_autoInitFromGeneratedScenarios` écrit `scen.autoOrder`, mais cet attribut est
ajouté dynamiquement, non déclaré dans `ScenarioAssemblage`, non copié par la
duplication et non sérialisé par XML.

### Construction, candidats, forks et branches

`AlgoQuadrisParPaires.run` est l'algorithme actuellement enregistré dans
`ALGOS`. Il impose un nombre pair de triangles. Il construit le premier quad
avec `tri_ids[0:2]`, puis traite les paires `(2,3)`, `(4,5)`, etc.

Pour chaque état de branche, `tryAttachMobQuadToDestChain` :

1. ajoute le quad mobile au `TopologyWorld` de branche ;
2. essaie quatre raccords locaux (`LO`/`BL` de chaque côté) ;
3. construit les attachments candidats via `EdgeChoiceEpts` ;
4. interroge le Core pour les groupes et rejette les chevauchements ;
5. calcule la pose rigide et conserve les candidats valides ;
6. clone le monde physique si plusieurs candidats doivent diverger ;
7. applique les attachments dans une transaction du clone.

Un fork est détecté quand une étape possède au moins deux candidats valides.
Il ne devient significatif pour le nom que si plusieurs descendants survivent
jusqu'aux feuilles. Une absence de branche valide arrête la simulation avec
`debugFail(step="chain_connect", ...)`.

Chaque feuille possède son propre `last_drawn`, son propre `TopologyWorld` et
son propre `ScenarioAssemblage`. Il n'y a ni fusion ni déduplication
topologique après la génération ; la numérotation est un DFS gauche-droite des
feuilles survivantes.

### Création et clonage des mondes

Le premier monde est construit pendant le bootstrap du premier quad. Durant un
fork, `clonePhysicalState()` isole les candidats. À la finalisation, le monde
de la feuille est affecté directement à `scen.topoWorld`. Il ne s'agit pas
d'un monde partagé entre scénarios.

## Cycle de vie de `ScenarioAssemblage`

`ScenarioAssemblage` est la structure effective d'un scénario. Elle porte
aujourd'hui `name`, `source_type`, `algo_id`, `tri_ids`, `last_drawn`,
`groups`, `topoWorld`, états de vue/carte et statut. Le manuel partage ses
structures runtime ; les autos reçoivent des copies/instances indépendantes.

| Étape | Chemin observé | État des métadonnées auto |
|---|---|---|
| Création auto | `AlgoQuadrisParPaires.run` | `algo_id`, `tri_ids`; pas de version, edge initial ni options |
| Enregistrement UI | `_simulation_assemble_dialog` | `source_type`, vue, carte, `autoOrder` dynamique |
| Activation | `_set_active_scenario` | réaffecte `self._last_drawn` et `self.groups` |
| Duplication | `_scenario_duplicate` | copie `algo_id` et `tri_ids`; ne copie pas explicitement `autoOrder` ni futures options |
| Conversion auto -> manuel | `_convertActiveAutoToManualSnapshot` | copie `algo_id`, `tri_ids`, `status`; change la nature du scénario |
| Suppression | `_scenario_delete`, `_simulation_clear_auto_scenarios` | suppression de l'objet, pas d'archive |
| Sauvegarde XML | `saveScenarioXml` | sérialise topologie/geometry/groups, pas `source_type`, nom, `algo_id`, `tri_ids`, `autoOrder` ni edge initial |
| Chargement XML | `loadScenarioXml` via nouveau scénario | restaure le snapshot Core et legacy, mais aucune métadonnée auto dédiée |

### Emplacement recommandé des futures métadonnées

Le bon propriétaire est `ScenarioAssemblage`, conditionnellement à
`source_type == "auto"`, par exemple une valeur immuable :

```python
@dataclass(frozen=True)
class AutoScenarioParameters:
    first_triangle_rank: int
    traversal_direction: str   # forward | reverse
    algorithm_id: str
    algorithm_version: str
    first_triangle_edge: str
    requested_triangle_count: int
    traversal_order: tuple[int, ...]
    options: tuple[tuple[str, object], ...]
```

Cette proposition n'est pas une demande d'implémentation dans ce chantier.
`traversal_order` est nécessaire : `first_triangle_rank + direction` peut ne
pas suffire si un algorithme ultérieur écarte, permute ou sélectionne des
triangles. Pour les anciens XML/scénarios, le comportement sûr serait
`auto_parameters = None` / provenance inconnue : les scénarios restent
comparables par attachments, mais ne sont pas présentés comme reproductibles.

## Flux de comparaison actuel

```text
double-clic scénario auto
  -> ref_scenario_token = id(scenario)
  -> redraw
  -> _update_current_scenario_differences()
  -> _scenario_connections_signature(reference/current)
  -> comparaison par triangle et arête
  -> _comparison_diff_indices
  -> remplissage rouge + contour des triangles concernés
```

Le double-clic ne change pas de scénario : il désigne ou retire une référence
auto. Le token est l'identité Python de l'objet ; la suppression de la
référence l'efface. À chaque redraw, la comparaison est recalculée, donc elle
est dynamique et ne dépend pas d'un cache de génération.

`_scenario_connections_signature` est le point de dépendance principal : il
parcourt les paires consécutives de `groups[*]["nodes"]` et produit
`triId -> edge -> (neighborTriId, neighborEdge)`. Il normalise au passage les
groupes et exige des `edge_out/edge_in` pour toute liaison interne.

`_update_current_scenario_differences` compare ces signatures. Pour chaque
triangle du courant, il marque le triangle si la map de ses arêtes diffère,
puis marque les voisins impliqués. Le besoin final est topologique et doit être
conservé ; seule la source de signature est legacy.

`_scenario_prefix_edge_steps` et `_filter_auto_scenarios_by_prefix_edges`
répondent à un besoin distinct : filtrer les scénarios automatiques qui ont le
même préfixe d'ordre de parcours et les mêmes liaisons de chaîne jusqu'au
triangle cliqué. Elles mélangent donc paramètres de génération (`tri_ids`) et
topologie (`edge_in/out`).

## Signification exacte de `#5 = #4 + (6)`

Le libellé n'est **pas** une relation parent/enfant entre deux objets
`ScenarioAssemblage`. Les scénarios `#4` et `#5` sont deux feuilles
indépendantes, avec mondes et géométries indépendants.

Lors du post-traitement, l'algorithme construit l'arbre des branches
survivantes. À chaque nœud ayant plusieurs enfants survivants, il prend la
feuille la plus à gauche du premier enfant (`#4`) comme repère et étiquette la
première feuille de chaque sous-arbre droit (`#5`) :

```text
#index_feuille_droite = #index_feuille_gauche + (branchTriId)
```

`branchTriId` est le triangle mobile qui a créé une bifurcation de candidats
valides au raccord du quad suivant. Il n'est ni « le premier triangle dont la
connexion finale diffère », ni un delta entre deux `TopologyWorld`. C'est une
trace minimale de l'arbre de génération, calculée après pruning.

Pour conserver ce libellé sans `edge_in/out`, il faut soit conserver une trace
minimale de génération (recommandé : identité de fork et chemin/branche), soit
le recalculer en rejouant l'algorithme avec les métadonnées complètes. Une
simple différence d'attachments ne permet pas toujours de retrouver le fork
historique qui a engendré une feuille.

## Signature canonique conceptuelle des attachments

`TopologyAttachment` contient `kind`, deux `TopologyFeatureRef`, `params` et
`source`. Une feature stable est :

```text
vertex -> ("vertex", element_id, vertex_index)
edge   -> ("edge", element_id, edge_index)
```

Proposition de signature non implémentée :

```text
endpoint(feature) = (feature_type, element_id, index, normalized_t_if_relevant)
signature(attachment) = (
    kind,
    min(endpoint(feature_a), endpoint(feature_b)),
    max(endpoint(feature_a), endpoint(feature_b)),
    normalized_semantic_params(kind, params),
)
```

Les extrémités sont triées pour être indépendantes du groupe mobile, de l'ordre
de création et de l'ordre `a/b`. `attachment_id` et `source` sont exclus : ce
sont des identifiants/provenances, pas le fait topologique. Les paramètres
sémantiques doivent être définis par type : notamment mapping direct/inverse
pour `edge-edge`, et position normalisée `t` si elle distingue réellement deux
attachments `vertex-edge` sur la même arête. Cette dernière règle est une
ambiguïté à lever avant implémentation ; l'API actuelle accepte `t` dans les
feature refs et des paramètres de validation.

Comparaison conceptuelle de deux mondes :

```text
reference_connections = set(signatures(reference.attachments))
current_connections   = set(signatures(current.attachments))
common_connections    = intersection
only_in_reference     = reference - current
only_in_current       = current - reference
```

Les éléments graphiques à surligner se déduisent des `element_id` présents
dans les endpoints des signatures exclusives, via l'index
`topoElementId -> _last_drawn`. Cela reproduit le principe actuel « triangle
modifié + voisin impliqué », y compris `edge-edge`, `vertex-vertex` et
`vertex-edge`, sans dépendre d'un ordre de chaîne.

## Inventaire `edge_in/out` : simulation et comparaison seulement

| Occurrence / zone | Lecture | Écriture | Classe | Diagnostic |
|---|---:|---:|---|---|
| `_fill_group_vkeys_from_geometry` | Non | Oui | C | déduit les arêtes communes pour les nœuds legacy après la topologie |
| finalisation de `AlgoQuadrisParPaires.run` | Non | Oui | C | injecte `_chain_edge_in/out` dans `groups[1].nodes` |
| `_scenario_connections_signature` | Oui | Non | B | reconstruit une signature de comparaison legacy |
| `_scenario_prefix_edge_steps` | Oui | Non | B | compare le préfixe de génération via chaîne legacy |
| `_filter_auto_scenarios_by_prefix_edges` | Indirect | Non | B | consommateur du résultat précédent |
| `_normalize_group_nodes` | Oui | Non sur `edge_*` | D | validation défensive du contrat ; purge `vkey_*` |
| `saveScenarioXml` | Oui | XML | C | persistance legacy de compatibilité |
| `loadScenarioXml` | XML | Oui | D/C | validation, restauration et compatibilité v4 |

Classes : A = indispensable à la simulation actuelle ; B = reconstruction de
comparaison ; C = écriture legacy après création de la vraie topologie ; D =
normalisation/maintenance défensive. Aucun lecteur `edge_in/out` n'a été
identifié comme nécessaire au calcul Core d'un candidat : la simulation crée
et valide ses `TopologyAttachment` avant la matérialisation finale de la chaîne
legacy. L'ordre de `tri_ids`, lui, reste indispensable à l'algorithme.

## Plan de migration sécurisé

1. Ajouter les paramètres de génération à `ScenarioAssemblage`, pas au Core.
2. Les renseigner au lancement et pour chaque feuille : ordre exact, direction,
   triangle initial, edge initial, algo/version/options.
3. Les sérialiser dans une section XML versionnée ; tolérer leur absence pour
   les anciens scénarios.
4. Définir et tester la signature canonique de `TopologyAttachment`.
5. Migrer `_scenario_connections_signature`, puis
   `_update_current_scenario_differences`, vers cette signature.
6. Traduire les signatures exclusives en `topoElementId` et vérifier que le
   surlignage visuel actuel est conservé.
7. Migrer le filtrage de préfixe : métadonnées de parcours pour le préfixe,
   attachments pour les connexions.
8. Préserver explicitement la trace de fork requise par les libellés, ou
   accepter un nouveau libellé fonctionnel après décision produit.
9. Retirer d'abord les lectures de comparaison devenues mortes, puis les
   écritures de simulation/XML et enfin la normalisation legacy.

## Tests nécessaires avant migration

- même scénario comparé à lui-même ;
- deux scénarios avec une seule connexion différente ;
- différences `edge-edge`, `vertex-edge`, `vertex-vertex` ;
- orientation directe et inverse, avec ordre croissant et décroissant ;
- changement et suppression du scénario de référence ;
- ancien XML sans métadonnées automatiques ;
- nouveau XML avec paramètres complets ;
- conservation du surlignage actuel des triangles et voisins ;
- conservation ou redéfinition validée du libellé de fork ;
- comparaison indépendante de l'ordre de création et des IDs d'attachments ;
- cas avec plusieurs forks, branches prunées et plusieurs composantes.

## Ambiguïtés à trancher

1. La position `t` d'un attachment `vertex-edge` doit-elle faire partie de la
   signature topologique de comparaison, et avec quelle normalisation ?
2. Le mapping direct/inverse d'un `edge-edge` est-il toujours métier ou parfois
   seulement géométrique ?
3. Le filtre par préfixe doit-il comparer l'historique de parcours, la
   topologie finale, ou les deux explicitement ?
4. Le libellé de fork est-il une information utilisateur à préserver à
   l'identique, ou peut-il devenir une métadonnée de diagnostic distincte ?

## Décision GO / NO-GO

**NO-GO pour la suppression immédiate de `edge_in/out`.** Les consommateurs de
comparaison, filtrage et XML ne disposent pas encore de leur remplacement.

**GO pour MIG-GEO-005 préparatoire** : métadonnées de génération au niveau du
scénario, signature canonique en lecture seule, tests de comparaison et logs
de double source. Cette préparation respecte l'architecture cible sans donner
un sens métier à l'ordre des triangles dans `TopologyWorld`.
