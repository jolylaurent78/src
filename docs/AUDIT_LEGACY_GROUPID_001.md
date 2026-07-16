# AUDIT-LEGACY-GROUPID-001 — statut de `group_id` et `groups[*]["nodes"]`

Date d'audit : 16 juillet 2026  
Portée : lecture du code de `src/` et `tests/`, sans modification de comportement.

## Résumé exécutif

La migration vers `TopologyWorld` a déjà déplacé la vérité d'appartenance
topologique des triangles dans le Core. Les opérations MOVE, ROTATE et FLIP,
la comparaison de scénarios, le filtrage par préfixe et la sélection par sommet
peuvent déterminer leurs membres depuis un `topoElementId`, les groupes DSU et
les `TopologyAttachment`.

Il faut donc distinguer deux homonymes :

* le `group_id` du Core (`TopologyGroup.group_id`, DSU, `TopologyWorld`) est
  moderne et autoritaire ;
* le `group_id` numérique de `_last_drawn` et `self.groups` est une identité de
  projection UI historique.

L'ensemble des éléments d'un groupe UI est reconstructible depuis
`TopologyWorld` puis `_last_drawn`. En revanche, `groups[*]["nodes"]` n'est pas
encore un cache jetable : son ordre séquentiel, ses ancres et les métadonnées
`edge_in`, `edge_out`, `vkey_in`, `vkey_out` servent toujours au collage manuel
et au format XML historique. Ces données ne décrivent pas une connectivité
métier supplémentaire par rapport aux `TopologyAttachment`, mais elles
encodent une **représentation de chaîne UI** et un protocole d'insertion que le
Core ne fournit pas actuellement sous cette forme.

Conclusion : un `CoreGroupView` en lecture seule est faisable immédiatement
pour les besoins de membres, bbox, centroïde, contour et sélection. La
suppression complète de `groups[*]["nodes"]` ne sera sûre qu'après une migration
explicite du collage manuel, de la projection de dégrouper et de la persistance
historique. À ce stade, le modèle est une projection UI partiellement
reconstructible, mais contient encore des métadonnées de présentation et de
construction non portées par le Core.

## Méthode et terminologie

Les recherches ont couvert les occurrences de `group_id` et `nodes` dans
`src/` et `tests/`, puis les familles de méthodes qui lisent ou écrivent
`self.groups` / `_last_drawn`. Le terme « groupe » est volontairement qualifié
dans tout le document :

| Terme | Porteur | Nature |
| --- | --- | --- |
| Groupe Core | `TopologyWorld.groups`, `TopologyGroup`, DSU | vérité de connectivité et de membres |
| Groupe UI | `TriangleViewerManual.groups` | projection historique, identifiant numérique local |
| Membre projeté | `_last_drawn[i]["group_id"]` | cache d'appartenance UI par triangle rendu |
| Élément topologique | `_last_drawn[i]["topoElementId"]` | pont entre projection et Core |
| Chaîne UI | `groups[gid]["nodes"]` | liste ordonnée de triangles et métadonnées de jonction historiques |

L'absence totale de `group_pos` dans `src/` et `tests/` a aussi été vérifiée :
MIG-GEO-009 a supprimé cet ancien cache d'index. Aucun résultat ci-dessous ne
suppose son existence.

## 1. Cartographie de `group_id`

### 1.1 `group_id` Core : usages modernes à ne pas migrer comme du legacy

Dans `src/assembleur_core.py`, `TopologyGroup.group_id`, les paramètres
`group_id` des API de `TopologyWorld`, `find_group()`, `group_members()`,
`get_group_of_element()`, les caches conceptuels et les frontières manipulent
des identifiants canoniques du DSU. Ils servent notamment à :

* déterminer les éléments connectés ;
* invalider ou recalculer les caches conceptuels et frontières ;
* résoudre les groupes fusionnés via `find_group()` ;
* exécuter les opérations Core (`flipGroup`, rattachements, dégrouper) ;
* fournir les segments de frontière et les nœuds conceptuels à l'UI.

Ces occurrences sont de catégorie **Core moderne**. Elles ne sont ni des
lectures de `last_drawn[*]["group_id"]`, ni des dépendances au modèle chaîné.
Elles constituent au contraire la source à utiliser pour éliminer le groupe UI
lorsque cela est possible.

### 1.2 `group_id` UI / legacy : inventaire fonctionnel

| Fichier et fonction/famille | Lecture / écriture | Rôle actuel | Classement | Remplaçable depuis Core + projection ? |
| --- | --- | --- | --- | --- |
| `assembleur_sim.py`, matérialisation des scénarios | écriture | initialise `last_drawn[*]["group_id"]` et les groupes UI simulés | projection / simulation | Oui pour l'appartenance ; partiellement pour l'ordre de chaîne |
| `assembleur_io.py`, `saveScenarioXml()` | lecture | écrit l'attribut XML de groupe des triangles | persistance historique | Partiellement : un id de vue peut être reconstruit, mais pas la compatibilité exacte du format existant sans règle de numérotation |
| `assembleur_io.py`, `loadScenarioXml()` | écriture | restaure le groupe UI des triangles à partir des groupes XML | persistance historique | Oui pour une projection après reconstruction Core ; non tant que le chargeur doit accepter l'ancien contrat XML à l'identique |
| `assembleur_tk.py`, audit Core/UI et index `_last_drawn` | lecture | compare l'ancienne projection aux groupes Core et diagnostique les écarts | validation temporaire | Oui : l'audit lui-même deviendra inutile lorsque la projection sera générée depuis le Core |
| `assembleur_tk.py`, activation/réconciliation de scénario | lecture et écriture | aligne les triangles projetés avec `self.groups`, rétablit les singletons et recalcule `_next_group_id` | gestion de projection | Oui pour les membres ; partiellement pour la conservation d'identifiants UI stables |
| `assembleur_tk.py`, `_find_nearest_vertex()` | lecture | exclut les triangles du même groupe UI lors du snap | interaction UI | Oui : comparaison des groupes Core des `topoElementId` |
| `assembleur_tk.py`, `_get_group_of_triangle()` et `_new_group_id()` | lecture / écriture | résout ou attribue l'id de vue numérique | infrastructure UI legacy | Partiellement : une vue a encore besoin d'une clé locale, qui peut être une clé Core ou une clé de projection déterministe |
| `assembleur_tk.py`, placement d'un triangle isolé | écriture | crée un singleton UI et le marque dans `_last_drawn` | projection / interaction | Oui : le Core crée déjà un groupe singleton ; la vue peut être construite depuis lui |
| `assembleur_tk.py`, bbox, centroïde, contours et rendu | lecture | choisit les triangles du groupe à afficher ou mesurer | projection visuelle | Oui, via `group_members()` puis index `topoElementId -> _last_drawn` |
| `assembleur_tk.py`, MOVE / ROTATE / FLIP | lecture de secours et journalisation | construit `LegacyMembers`, conserve une compatibilité/fallback et met à jour la bbox UI | migration en cours | Oui pour les membres : le chemin primaire est déjà Core ; le fallback reste à retirer après caractérisation |
| `assembleur_tk.py`, `_applyDegrouperResultToTk()` | écriture | répartit les triangles dans les groupes UI résultant de `degrouperAtNode()` | projection d'une opération Core | Oui pour les membres ; partiellement pour l'ordre de `nodes` et les identifiants UI attribués |
| `assembleur_tk.py`, collage manuel au relâchement souris | lecture et écriture | choisit les deux groupes UI, fusionne leurs chaînes et propage le nouvel id sur les triangles | interaction UI / assemblage | Partiellement : l'attachement est Core, mais l'algorithme consomme encore la chaîne UI ordonnée |
| tests `test_mig_geo001_group_linking.py` | lecture / écriture | construit des projections minimales et vérifie le pont Core/UI | test de migration | Oui, à adapter lorsque le cache UI sera remplacé |

### 1.3 Lecture importante : identité vs appartenance

Le numéro UI (`1`, `2`, etc.) n'est pas une identité métier stable. Il est
créé par `_new_group_id()`, restauré depuis le XML ou recalibré lors de
l'activation d'un scénario. À l'inverse, le groupe Core est résolu par DSU et
porte un ensemble d'`element_ids` canonique. Ainsi :

```text
topoElementId -> TopologyWorld.get_group_of_element() -> groupe Core
```

est la règle de détermination de l'appartenance. Le champ
`last_drawn[*]["group_id"]` ne doit plus devenir une source de vérité métier.
Il reste cependant nécessaire tant que les lecteurs de la projection historique
ne sont pas migrés.

## 2. Cartographie de `groups[*]["nodes"]`

La structure est une liste ordonnée de dictionnaires, typiquement :

```python
{
    "tid": 17,
    "edge_in": "L",
    "edge_out": "B",
    "vkey_in": "O",
    "vkey_out": "L",
}
```

Les champs de jonction peuvent être absents ou `None` sur un singleton. La
liste est à la fois utilisée comme inventaire de triangles et comme chaîne de
collage. Les usages ci-dessous sont regroupés par finalité, plutôt que par
simple occurrence textuelle.

| Catégorie | Fichier et famille | Finalité concrète | Peut être reconstruit depuis `TopologyWorld` + `TopologyAttachment` + `_last_drawn` ? |
| --- | --- | --- | --- |
| Projection visuelle | `assembleur_tk.py`, `_draw_group_outlines()` | collecte les triangles à dessiner comme contour de groupe | Oui pour les membres ; contour à calculer depuis les éléments projetés ou les segments Core |
| Projection visuelle | `assembleur_tk.py`, `_recompute_group_bbox()`, `_group_centroid()` | bbox et centroïde du groupe | Oui, entièrement : géométrie de `_last_drawn` des membres Core |
| Projection visuelle | `assembleur_core.py`, `_group_shape_from_nodes()` | union géométrique brute des triangles de la chaîne | Oui après remplacement de l'entrée `nodes` par une liste d'entrées `_last_drawn` résolues depuis les membres Core |
| Interaction UI | `assembleur_tk.py`, aperçu de collision / snap | forme mobile, triangle(s) exclus et listes de tids pour le contrôle d'intersection | Oui pour les ensembles ; les calculs géométriques restent dans la projection |
| Interaction UI | `assembleur_tk.py`, suppression et nettoyage de groupes | identifie les triangles à retirer et reconstruit les groupes UI restants | Oui pour le résultat de membres ; l'algorithme de maintenance de chaîne doit être remplacé ou supprimé |
| Dégroupement | `assembleur_tk.py`, `_applyDegrouperResultToTk()` | applique aux groupes UI les composantes renvoyées par le Core | Oui pour les composantes ; seulement partiellement pour l'ordre UI conservé/recréé |
| Déplacement / rotation / miroir | `assembleur_tk.py`, `_prepare_mig_geo_operation_members()` | construit le jeu legacy de comparaison/fallback | Oui : le chemin Core le fait déjà ; cet usage est un reliquat contrôlé |
| Collage manuel | `assembleur_tk.py`, `_on_canvas_left_up()` | trouve les positions d'ancres, fait pivoter / insère / concatène deux chaînes et remplit les jonctions | Partiellement : la connexion est Core, mais l'ordre et la règle d'insertion de la chaîne ne sont pas une API Core |
| Simulation | `assembleur_sim.py`, construction/enrichissement des scénarios | fabrique des chaînes avec `edge_in/out` et `vkey_in/out`, puis les utilise pour des formes et contrôles | Partiellement : membres et attachments sont Core-reconstructibles ; la chaîne simulée reste une convention historique |
| Persistance | `assembleur_io.py`, save/load XML | sérialise / désérialise l'ordre des nodes et les jonctions historiques | Non pour une restitution binaire identique sans définir un nouveau contrat ; oui pour reconstruire la connectivité métier dans le Core |
| Rétablissement d'état UI | `assembleur_tk.py`, activation de scénario, copie/reconciliation | clone les chaînes et attribue les `group_id` projetés | Partiellement : membres oui ; ordre, ids UI et métadonnées legacy non |

## 3. Ce que `groups[*]["nodes"]` transporte encore réellement

### 3.1 Données déjà redondantes avec le Core

Les informations suivantes ne sont pas des vérités propres au modèle historique :

* l'ensemble des triangles reliés ;
* la composante résultant d'un attachement ou d'un dégrouper ;
* la connectivité élémentaire entre triangles ;
* les nœuds logiques partagés ;
* la frontière conceptuelle, lorsque le cache de frontière Core est disponible ;
* les poses affichables des triangles, qui résident dans `_last_drawn` comme
  projection géométrique.

Le Core peut fournir les `element_ids` d'un groupe. L'index introduit par
MIG-GEO-001 permet ensuite de résoudre efficacement ces ids vers les entrées
de `_last_drawn`. Cette chaîne est suffisante pour les usages ensemblistes :

```text
CoreGroup.element_ids
  -> get_last_drawn_entries_for_core_group(...)
  -> bbox / centroïde / contour / membres à déplacer
```

### 3.2 Informations encore uniques dans la projection historique

Les informations ci-dessous ne sont pas, aujourd'hui, accessibles sous une
forme équivalente depuis `TopologyWorld` :

1. **Ordre linéaire de la chaîne UI.** `nodes[0]`, `nodes[-1]` et la position
   d'un `tid` servent à faire une insertion à gauche/droite d'une ancre lors du
   collage. Un groupe topologique est un ensemble DSU, pas une liste ordonnée.
2. **Convention locale d'insertion.** La combinaison de `pos_anchor`,
   `pos_target`, rotation de liste et concaténation encode une politique UI
   historique : quel sous-ordre doit être préservé après un collage.
3. **Métadonnées de chaîne redondantes mais opérantes.** `edge_in`, `edge_out`,
   `vkey_in`, `vkey_out` expriment, pour deux voisins successifs de la chaîne,
   la jonction retenue par l'ancien modèle. Les `TopologyAttachment` expriment
   la connectivité réelle, mais ne fournissent pas une séquence ordonnée de ces
   jonctions telle qu'attendue par le code UI actuel.
4. **Identifiants UI et compatibilité XML.** Les numéros UI, le format
   `<groups>/<group>/<node>` et son ordre sont conservés pour relire et écrire
   les scénarios historiques. Ce n'est pas une vérité topologique, mais c'est
   un contrat de persistance actuel.

Le résultat est important : si `groups[*]["nodes"]` disparaissait aujourd'hui,
la connectivité métier ne serait pas perdue si le `TopologyWorld` correspondant
est présent et cohérent. En revanche, le collage manuel, le round-trip XML
historique et plusieurs projections UI perdraient leur représentation attendue.
Le code ne peut donc pas encore supprimer la structure sans migration dédiée.

## 4. Évaluation de remplaçabilité par usage

| Usage actuel | Réponse | Justification |
| --- | --- | --- |
| Déterminer les membres d'un groupe | Oui | `get_group_of_element()` + `group_members()` + index `_last_drawn` |
| MOVE / ROTATE / FLIP | Oui | migration déjà réalisée ; les lectures legacy ne sont plus que validation/fallback et maintenance UI |
| Clic sommet puis déplacement | Oui | MIG-GEO-008 résout le nœud DSU et le groupe Core |
| Bbox, centroïde, rendu d'ensemble | Oui | agrégation des géométries des entrées projetées membres |
| Contour de groupe | Oui, avec une étape de projection | `getBoundarySegments()` peut fournir la frontière Core ; les points restent à projeter depuis les poses courantes |
| Exclusion lors du snap | Oui | comparer les groupes Core des éléments plutôt que les ids UI |
| Dégroupement : membres résultants | Oui | `degrouperAtNode()` produit les composantes Core |
| Dégroupement : ordre de nouvelle chaîne UI | Partiellement | les composantes sont connues, pas une liste ordonnée équivalente à l'ancien contrat |
| Collage : création de l'attachement | Oui | le Core possède `TopologyAttachment` et les API de rattachement |
| Collage : insertion/concaténation de chaîne UI | Partiellement | le besoin de rendu peut être redéfini, mais la règle exacte actuelle est portée par `nodes` |
| Sauvegarde/rechargement du XML historique | Partiellement | le modèle peut être reconstruit, mais l'ordre et les métadonnées legacy ne sont pas déterministes sans contrat supplémentaire |
| Simulation automatique | Partiellement | les attachments portent la connectivité ; la simulation construit encore explicitement des chaînes legacy |

## 5. Faisabilité de `CoreGroupView`

Une vue de groupe transitoire est techniquement faisable sans faire du cache UI
une autorité. Son contrat minimal serait en lecture seule :

```python
CoreGroupView(
    core_group_id: str,
    element_ids: tuple[str, ...],
    entries: tuple[dict, ...],
    bbox: tuple[float, float, float, float] | None,
    centroid: tuple[float, float] | None,
    boundary_segments: tuple[BoundarySegment, ...],
)
```

La construction suivrait strictement :

```text
TopologyWorld groupe canonique
  -> element_ids
  -> index topoElementId -> _last_drawn
  -> données géométriques de vue
```

Cette structure ne doit ni allouer un `group_id` UI, ni imposer un ordre de
chaîne, ni réécrire `self.groups`. Elle centraliserait les besoins UI dérivés et
éviterait de dupliquer les parcours de `nodes`.

### Migrable immédiatement

* lecture des membres pour les opérations déjà migrées ;
* bbox, centroïde, forme de collision et contours de rendu ;
* exclusion de groupe dans le snap ;
* diagnostics de cohérence Core/UI, puis leur suppression lorsque l'ancienne
  projection ne sera plus produite.

### Nécessitant une étape intermédiaire

* projection après `degrouperAtNode()` : le Core donne les composantes, il faut
  fixer le contrat d'affichage des nouvelles vues ;
* synchronisation de poses : elle doit recevoir une collection d'éléments ou
  d'entrées, pas une liste de `nodes` UI ;
* scénarios simulés : la génération doit produire le monde et la projection,
  non une chaîne comme donnée primaire.

### Bloquant la suppression complète à ce stade

* collage manuel et sa politique d'insertion ordonnée ;
* lecture/écriture XML tant que le format de groupe historique doit être
  conservé à l'identique ;
* toute fonctionnalité dont le résultat visuel dépend expressément de l'ordre
  de `nodes`, plutôt que de la connectivité ou de la frontière Core.

## 6. Proposition de roadmap

### MIG-GEO-010 — Vues de groupe Core en lecture seule

**Chantiers simples.** Introduire et caractériser une projection
`CoreGroupView` obtenue du Core et de `_last_drawn`, sans retirer `self.groups`.
Migrer les consommateurs purement ensemblistes : bbox, centroïde, contour,
forme de collision, exclusion de snap et membres de secours des opérations.

**Pourquoi en premier :** ces usages sont dérivables sans décision métier sur
l'ordre. Ils sont faciles à comparer à l'ancienne projection et réduisent le
nombre de lecteurs de `nodes` sans toucher au collage.

### MIG-GEO-011 — Projection UI des résultats Core

**Chantiers intermédiaires.** Faire de la projection après dégrouper et de la
synchronisation de pose des consommateurs explicites de `element_ids` et
d'entrées `_last_drawn`. Définir la stabilité souhaitée des identifiants de vue
et la stratégie de rendu de la frontière.

**Pourquoi ensuite :** le Core produit déjà les composantes, mais l'UI doit
cesser de les retransformer en chaînes à titre de modèle intermédiaire. Les
tests devront vérifier qu'aucun attachment n'est modifié par le rendu.

### MIG-GEO-012+ — Collage manuel, séquence et XML cible

**Chantiers complexes.** Isoler le besoin fonctionnel réel de la chaîne de
collage : ordre d'affichage, ordre de narration, ou uniquement compatibilité.
Choisir ensuite soit une projection déterministe depuis les attachments, soit
une métadonnée explicite de parcours si cet ordre est bien une exigence produit.
Faire évoluer le XML seulement après cette décision, avec stratégie de lecture
des scénarios historiques.

**Pourquoi en dernier :** ce secteur couple interaction souris, géométrie,
attachements, ordre et persistance. Une migration mécanique de `nodes` vers un
ensemble de membres changerait potentiellement le comportement visible de
collage et invaliderait des fichiers existants.

## Conclusion argumentée

`groups[*]["nodes"]` n'est plus la source de vérité de l'appartenance
topologique : cette vérité est dans `TopologyWorld`, ses groupes DSU et ses
`TopologyAttachment`. Pour les besoins de membre, de sélection, de mouvement et
de géométrie de vue, le modèle historique est reconstructible à partir du Core
et de `_last_drawn`.

Cependant, la structure conserve encore une valeur opérationnelle : elle porte
un **ordre de chaîne et des conventions de jonction** utilisés directement par
le collage manuel et sérialisés par le XML historique. Ce n'est pas une donnée
métier topologique fondamentale, mais ce n'est pas encore un cache UI
entièrement remplaçable dans le code actuel.

La cible est donc nette : `TopologyWorld` autoritaire, `_last_drawn` projection,
et une `CoreGroupView` dérivée pour les consommateurs UI. La suppression de la
chaîne historique doit attendre la migration explicite de son dernier contenu
unique — ordre de collage, représentation de jonction et compatibilité XML —
plutôt qu'une suppression fondée sur la seule équivalence des membres.
