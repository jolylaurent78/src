# ARCHITECTURE CIBLE TOPOLOGIQUE
## Catalogue de variantes, scénarios, reconstruction géométrique et transition depuis le modèle hybride

**Projet :** AssembleurTriangles / La Chouette  
**Statut :** Document de référence conceptuel et architectural  
**Objet :** Formaliser la cible fonctionnelle et technique avant toute évolution majeure du code  
**Portée :** Catalogue des triangles, scénarios, topologie, reconstruction géométrique, cache UI, ancrage, validation transversale, persistance et trajectoire de migration  
**Nature du document :** Référence pour les futures sessions GPT et les travaux Codex  
**Important :** Ce document décrit une architecture cible et les invariants associés. Il ne constitue pas encore un plan d’implémentation détaillé ni une demande de modification immédiate du code.

---

# 1. Finalité du document

Le projet AssembleurTriangles est historiquement né comme un outil de manipulation géométrique de triangles dans une interface graphique. Son évolution a progressivement fait apparaître une réalité métier plus profonde : la géométrie visible n’est pas la définition fondamentale d’un assemblage.

La définition métier d’un assemblage repose sur :

1. l’identité logique des triangles ;
2. la variante — ou hypothèse — sélectionnée pour chaque triangle ;
3. les relations topologiques entre leurs sommets ou leurs arêtes ;
4. les informations d’orientation portées par ces relations ;
5. une politique d’ancrage permettant de positionner l’ensemble dans le plan.

À partir de ces seules informations, la géométrie complète de l’assemblage peut, en théorie, être reconstruite.

Cette conclusion change profondément la lecture de l’architecture du projet :

- la topologie ne doit plus être considérée comme une couche ajoutée à la géométrie ;
- elle constitue la source de vérité du scénario ;
- la géométrie monde est une projection calculée ;
- les structures graphiques historiques, notamment `_last_drawn`, doivent être considérées comme un cache de travail destiné à l’affichage et aux interactions ;
- le canvas constitue une vue de cette projection et non une autorité métier.

Le présent document formalise cette vision cible afin de fournir une référence commune aux futures analyses, aux travaux de stabilisation et aux évolutions du code.

---

# 2. Contexte historique

## 2.1 Modèle initial

Le modèle initial du programme était essentiellement géométrique.

Un triangle était principalement identifié par :

- ses coordonnées ;
- sa position sur le canvas ;
- ses sommets visibles ;
- ses segments ;
- son appartenance éventuelle à un groupe graphique.

L’assemblage était donc appréhendé comme une collection de formes géométriques disposées dans un plan.

Dans ce paradigme, une relation entre deux triangles pouvait être déduite ou matérialisée à partir de la géométrie :

- superposition d’arêtes ;
- proximité de sommets ;
- indices de segments ;
- coordonnées correspondantes ;
- état courant des objets graphiques.

Ce modèle permettait de dessiner, déplacer et assembler des triangles, mais il ne constituait pas une représentation métier suffisamment stable.

## 2.2 Limite du modèle géométrique

Une relation géométrique ne suffit pas à exprimer durablement l’intention topologique.

Une attache telle que :

```text
triangle A, segment d’indice 2
↔
triangle B, segment d’indice 1
```

reste fragile si :

- les sommets sont réordonnés ;
- la forme du triangle change ;
- une nouvelle hypothèse produit une autre géométrie ;
- les objets graphiques sont recréés ;
- la sérialisation modifie l’ordre interne ;
- la géométrie est recalculée ;
- les identifiants temporaires du canvas changent.

Le programme avait donc besoin d’un référentiel qui exprime la relation métier indépendamment de la géométrie courante.

## 2.3 Introduction du référentiel topologique

Le projet a évolué vers une représentation topologique fondée sur une nomenclature stable des sommets et des arêtes.

Chaque triangle logique conserve une structure métier stable :

```text
Sommets : O, B, L
Arêtes  : OB, BL, LO
```

Les dimensions, les angles et la forme du triangle peuvent varier, mais les identités topologiques restent les mêmes.

Une relation peut alors être décrite sous une forme indépendante des coordonnées :

```text
Triangle 7.OB ↔ Triangle 12.BL
```

L’attache peut également contenir l’information d’orientation nécessaire à la reconstruction de la position relative des triangles.

Cette relation ne dépend pas :

- de la longueur actuelle de `OB` ;
- de la position monde des deux triangles ;
- d’un identifiant de canvas ;
- d’un indice d’arête calculé dynamiquement ;
- d’une variante géométrique particulière.

## 2.4 Situation actuelle : un modèle hybride

Le programme contient aujourd’hui deux mondes qui coexistent :

### Monde historique géométrique et graphique

Il comprend notamment :

- `_last_drawn` ;
- les coordonnées monde ;
- les objets du canvas ;
- des structures de groupes UI ;
- des caches de contours ;
- des informations nécessaires au picking et à la manipulation interactive.

### Monde topologique

Il comprend notamment :

- `TopologyWorld` ;
- les groupes topologiques ;
- les éléments topologiques ;
- les attaches ;
- les références aux sommets et arêtes typés ;
- les informations d’orientation ;
- les frontières et chemins dérivés.

Le problème n’est pas l’existence simultanée de ces deux mondes. Une interface graphique performante a légitimement besoin d’une projection géométrique calculée et de caches.

Le problème est l’absence actuelle d’une asymétrie suffisamment stricte entre eux.

Dans le modèle hybride actuel :

- certaines opérations commencent dans l’UI puis sont propagées vers le Core ;
- certaines informations géométriques restent porteuses d’une autorité implicite ;
- les deux représentations peuvent diverger ;
- la persistance peut enregistrer à la fois le modèle topologique et sa projection géométrique ;
- plusieurs fonctions lisent encore la géométrie UI comme s’il s’agissait d’une donnée métier ;
- aucune règle transversale ne garantit partout que la topologie gagne en cas de divergence.

---

# 3. Décision architecturale fondamentale

## 3.1 Principe

La décision architecturale centrale est la suivante :

> **La topologie du scénario constitue la seule source de vérité de l’assemblage.**

La géométrie monde n’est pas une seconde vérité.

Elle est une projection déterministe produite à partir :

1. du catalogue des variantes ;
2. de la sélection de variantes du scénario ;
3. des attaches topologiques ;
4. de leurs orientations ;
5. de l’ancre du scénario.

Le modèle cible n’est donc pas :

```text
TopologyWorld ↔ _last_drawn
```

avec deux représentations de même autorité à synchroniser.

Il est :

```text
Catalogue
   +
Scénario topologique
   +
Ancre
        ↓
Reconstruction géométrique
        ↓
Projection géométrique
        ↓
_last_drawn
        ↓
Canvas
```

## 3.2 Asymétrie obligatoire

Les couches ont des rôles distincts.

### Topologie

Autorité métier.

Elle définit :

- quels triangles logiques existent dans le scénario ;
- quelle variante est sélectionnée pour chacun ;
- quelles features sont attachées ;
- comment elles sont orientées ;
- comment les composants sont reliés ;
- quelle ancre positionne la construction.

### Géométrie calculée

Résultat dérivé.

Elle fournit :

- les poses ;
- les coordonnées monde ;
- les matrices ou transformations ;
- les segments monde ;
- les polygones ;
- les contours géométriques ;
- les bounding boxes ;
- les données nécessaires à l’affichage.

### `_last_drawn`

Cache de travail UI.

Il peut contenir une représentation directement exploitable pour :

- le dessin ;
- le picking ;
- le déplacement interactif ;
- la sélection ;
- le rafraîchissement ;
- l’aperçu ;
- certaines optimisations.

Il ne doit pas porter de décision métier durable.

### Canvas

Vue.

Il affiche la projection courante et transmet les intentions de l’utilisateur vers les commandes du domaine.

## 3.3 Conséquence majeure

En cas de divergence entre la topologie et la géométrie UI :

> **la topologie est réputée correcte et la géométrie doit être reconstruite.**

Il ne doit pas exister de mécanisme général de fusion ou de réconciliation symétrique entre les deux mondes.

---

# 4. Définition du catalogue

## 4.1 Évolution fonctionnelle

Historiquement, le projet manipulait un ensemble de triangles supposés uniques.

Chaque triangle était associé à une énigme ou à un rang déterminé.

La nouvelle réalité fonctionnelle est différente :

- une énigme peut admettre plusieurs hypothèses de lumière ;
- chaque hypothèse produit une géométrie différente ;
- plusieurs triangles géométriques peuvent donc correspondre au même triangle logique ;
- un scénario choisit une hypothèse parmi les variantes disponibles.

Le système doit donc distinguer :

1. le triangle logique ;
2. ses variantes géométriques ;
3. la variante sélectionnée dans un scénario.

## 4.2 Triangle logique : `TriangleSlot`

Le `TriangleSlot` représente l’identité stable d’un triangle dans le référentiel métier.

Exemple :

```text
TriangleSlot 7
```

Il peut être associé à :

- un rang ;
- une énigme ;
- un identifiant métier ;
- des métadonnées descriptives ;
- une nomenclature de features stable.

Le slot ne décrit pas une géométrie monde.

Il représente une place logique dans le problème.

### Invariants du slot

- son identité est stable ;
- il est unique dans le catalogue ;
- il expose une nomenclature topologique connue ;
- ses variantes respectent cette nomenclature ;
- il peut être référencé durablement par les scénarios.

## 4.3 Variante : `TriangleVariant`

Une `TriangleVariant` représente une hypothèse géométrique possible pour un slot.

Exemple :

```text
TriangleSlot 7
 ├─ Variante A
 ├─ Variante B
 └─ Variante C
```

Chaque variante peut contenir :

- un identifiant stable ;
- une référence vers son slot ;
- une description de l’hypothèse ;
- la source ou justification de l’hypothèse ;
- les coordonnées locales des sommets ;
- les longueurs ;
- les angles ;
- les métadonnées de construction ;
- des informations de validité ou de statut ;
- éventuellement une version.

### Invariants d’une variante

- elle appartient à un et un seul slot ;
- elle respecte la nomenclature de features du slot ;
- sa géométrie locale est valide ;
- ses sommets sont nommés de manière stable ;
- ses arêtes sont dérivables sans ambiguïté ;
- son identité ne dépend pas de sa position monde.

## 4.4 Géométrie locale

La géométrie locale d’une variante appartient au catalogue.

Elle ne doit pas être confondue avec la pose monde.

Par exemple :

```text
TriangleVariant 7.B
Sommets locaux :
O = (0, 0)
B = (...)
L = (...)
```

Ces coordonnées définissent la forme intrinsèque du triangle.

La position effective dans un scénario est obtenue par une transformation :

```text
géométrie locale
        +
pose calculée
        ↓
géométrie monde
```

## 4.5 Nomenclature stable

Toutes les variantes d’un même slot conservent la même nomenclature :

```text
O
B
L
```

et les mêmes arêtes sémantiques :

```text
OB
BL
LO
```

Ce point est un invariant fondamental.

Une variante peut modifier :

- les longueurs ;
- les angles ;
- la forme ;
- l’orientation locale conventionnelle, si celle-ci est définie et contrôlée ;
- les métadonnées de construction.

Elle ne doit pas modifier l’identité métier des features.

## 4.6 Indépendance entre identité et géométrie

L’identité d’un triangle est :

```text
TriangleSlot
+
nomenclature stable
```

et non :

```text
coordonnées
+
objet graphique
```

Cette séparation rend possibles :

- le changement d’hypothèse ;
- la conservation des attaches ;
- la reconstruction déterministe ;
- la comparaison de scénarios ;
- la sérialisation durable ;
- les tests métier indépendants du canvas.

---

# 5. Définition du scénario

## 5.1 Nature du scénario

Le scénario ne doit plus être compris comme une collection de triangles déjà positionnés.

Il doit être défini comme une configuration topologique complète.

Conceptuellement :

```text
Scenario
 ├─ sélection de variantes
 ├─ attaches topologiques
 ├─ orientations
 ├─ ancre
 └─ métadonnées
```

## 5.2 Sélection de variantes

Pour chaque slot actif, le scénario sélectionne une variante.

Exemple :

```text
Slot 1  → Variante B
Slot 2  → Variante A
Slot 3  → Variante D
```

Cette sélection constitue une décision métier du scénario.

Changer la variante d’un slot ne change pas nécessairement la topologie.

## 5.3 Attaches

Une attache relie deux features topologiques stables.

Exemple :

```text
Slot 7.OB ↔ Slot 12.BL
```

L’attache est :

- symétrique du point de vue topologique ;
- indépendante de la géométrie monde ;
- indépendante du canvas ;
- indépendante de la variante tant que la nomenclature reste stable.

Elle peut contenir les informations nécessaires à l’orientation géométrique relative.

## 5.4 Orientation

La seule identification des deux arêtes ne suffit pas toujours à déterminer la pose relative.

L’attache doit donc porter, directement ou indirectement, les informations permettant de reconstruire sans ambiguïté :

- le sens de correspondance des extrémités ;
- la face ou le côté relatif ;
- le retournement éventuel ;
- toute convention géométrique nécessaire.

Cette information appartient à la relation topologique du scénario.

Elle ne doit pas être redéduite de manière fragile à partir des coordonnées courantes.

## 5.5 Ancre

L’ancre est une propriété persistante du scénario.

Elle permet de positionner la construction dans le plan.

Le scénario peut être ancré :

- sur un triangle ;
- sur une pose de triangle ;
- sur une arête ;
- sur un sommet ;
- sur un point de référence externe ;
- sur une feature associée à un point fixe.

L’usage métier actuel indique que l’on ancre généralement le premier ou le dernier triangle sur un point de référence, puis que l’on déplace ou reconstruit le reste de la figure.

### Décision actuelle

Le modèle retenu par défaut est :

> l’ancre appartient au scénario.

Aucun besoin concret n’a pour le moment justifié un mécanisme de surcharge temporaire de l’ancre au niveau d’une opération.

Une telle possibilité ne doit donc pas être ajoutée sans cas d’usage réel.

## 5.6 Métadonnées

Le scénario peut également contenir :

- un nom ;
- une description ;
- une version ;
- des notes ;
- une date de création ;
- une date de modification ;
- un statut ;
- des informations de provenance ;
- des options de rendu non métier, à séparer clairement de la définition topologique.

---

# 6. Les attaches comme structure indépendante des variantes

## 6.1 Principe

Les attaches référencent les slots et les features stables.

Elles ne référencent pas directement une géométrie de variante.

Exemple :

```text
7.OB ↔ 12.BL
```

reste vrai lorsque :

```text
Slot 7 : Variante A → Variante B
```

## 6.2 Conséquence

Changer une hypothèse ne doit pas provoquer :

- la suppression des attaches ;
- leur recréation ;
- leur migration ;
- une nouvelle phase de collage manuel ;
- une réinterprétation des indices de segments ;
- une recherche géométrique des relations existantes.

L’opération correcte est :

```text
modifier la sélection de variante
        ↓
invalider la projection géométrique concernée
        ↓
reconstruire les poses
        ↓
rafraîchir le cache UI
```

## 6.3 Portée de la reconstruction

La reconstruction peut concerner :

- tout le scénario ;
- une composante connexe ;
- un sous-arbre induit par l’ancre et le point de modification ;
- un ensemble minimal de triangles dépendants.

La stratégie d’optimisation peut évoluer.

L’invariant fonctionnel reste :

> la topologie et les attaches ne changent pas lorsque seule la variante d’un slot change.

---

# 7. Structure topologique : graphes non orientés et arbres

## 7.1 Symétrie des attaches

Les attaches sont purement symétriques.

Une relation :

```text
A.feature_1 ↔ B.feature_2
```

n’implique pas que :

- A soit le parent ;
- B soit l’enfant ;
- A soit fixe ;
- B soit mobile.

La topologie persistée n’a pas besoin de hiérarchie parent-enfant.

## 7.2 Absence de cycles

Les assemblages considérés sont des arbres.

Cette propriété est fondamentale.

Elle implique :

- aucune boucle topologique ;
- un chemin unique entre deux triangles d’une même composante ;
- une propagation déterministe depuis une ancre ;
- l’absence de contraintes de fermeture de cycle ;
- l’absence de solveur global de type fermeture de boucle ;
- une reconstruction géométrique simple par parcours.

## 7.3 Orientation temporaire lors du calcul

Même si le graphe persisté est non orienté, la reconstruction utilise nécessairement un ordre de propagation.

Une fois l’ancre choisie, le parcours de l’arbre induit temporairement une orientation :

```text
ancre
  ↓
voisins
  ↓
descendants de parcours
```

Cette orientation est :

- algorithmique ;
- temporaire ;
- dérivée de l’ancre ;
- non persistée ;
- sans signification métier parent-enfant.

## 7.4 Plan de reconstruction

Un moteur de reconstruction peut produire un objet temporaire conceptuel :

```text
GeometryRebuildPlan
 ├─ ancre
 ├─ composante concernée
 ├─ ordre de parcours
 ├─ arêtes de propagation
 └─ dépendances à recalculer
```

Cet objet ne modifie pas la topologie.

Il sert uniquement à ordonner le calcul.

---

# 8. Reconstruction géométrique

## 8.1 Entrées

La reconstruction doit pouvoir fonctionner à partir de :

1. l’ancre du scénario ;
2. la sélection de variantes ;
3. les géométries locales des variantes ;
4. les attaches ;
5. les informations d’orientation des attaches.

## 8.2 Sorties

Elle doit produire :

- une pose monde pour chaque triangle actif ;
- les coordonnées monde de chaque sommet ;
- les arêtes monde ;
- les transformations locales-vers-monde ;
- les données nécessaires au dessin ;
- éventuellement des contours et frontières dérivés ;
- un rapport de validation.

## 8.3 Déterminisme

Pour un même :

- catalogue ;
- scénario ;
- ensemble de variantes ;
- graphe d’attaches ;
- ensemble d’orientations ;
- ancre ;

la reconstruction doit produire la même géométrie, à la tolérance numérique près.

## 8.4 Propagation

À partir de la pose de l’élément ancré :

1. parcourir l’arbre ;
2. pour chaque attache entre un triangle déjà posé et un triangle non posé :
   - récupérer la géométrie locale des deux variants ;
   - récupérer les features liées ;
   - appliquer l’orientation définie par l’attache ;
   - calculer la transformation du triangle non posé ;
   - enregistrer sa pose ;
3. poursuivre jusqu’à couverture de la composante.

## 8.5 Cas de plusieurs composantes

Si le scénario peut contenir plusieurs composantes non reliées, chaque composante doit posséder une règle de positionnement.

Plusieurs possibilités devront être arbitrées :

- une ancre par composante ;
- une ancre principale et des poses relatives supplémentaires ;
- interdiction des scénarios non connexes ;
- positionnement libre de certaines composantes.

Ce point reste à analyser dans le code et dans les usages réels.

Il ne faut pas présumer qu’une seule ancre suffit si plusieurs composantes indépendantes sont autorisées.

## 8.6 Changement de variante

Lorsqu’une variante change :

1. modifier la sélection dans le scénario ;
2. invalider les poses dérivées concernées ;
3. déterminer la zone de reconstruction ;
4. recalculer les poses ;
5. reconstruire `_last_drawn` ;
6. rafraîchir le canvas ;
7. valider la cohérence.

### Politique de stabilité

L’ancre reste fixe.

Le reste de la figure se déforme ou se déplace conformément à la nouvelle géométrie locale.

## 8.7 Déplacement global

Un déplacement de la figure peut être interprété de deux manières :

### Modification de l’ancre

La définition topologique ne change pas.

Seule la pose de l’ancre évolue.

La reconstruction repositionne alors l’ensemble.

### Transformation purement visuelle temporaire

Une interaction de drag peut temporairement manipuler le cache UI pour des raisons de fluidité.

Cependant, à la fin de l’opération, l’intention doit être traduite en modification durable de l’ancre, puis la projection doit être régénérée ou validée.

---

# 9. Rôle de `_last_drawn`

## 9.1 Statut cible

`_last_drawn` doit être défini comme :

> un cache géométrique fortement pratique, optimisé pour le canvas et les manipulations interactives, mais dépourvu d’autorité métier.

## 9.2 Contenu possible

Il peut contenir :

- les points monde ;
- les polygones ;
- les objets affichables ;
- les identifiants de canvas ;
- les données de sélection ;
- les bounding boxes ;
- des index spatiaux ;
- des états de survol ;
- des caches de rendu ;
- des références vers les identifiants topologiques.

## 9.3 Ce qu’il ne doit pas définir

`_last_drawn` ne doit pas être la source durable de :

- l’identité d’un triangle ;
- la variante sélectionnée ;
- l’existence d’une attache ;
- l’orientation d’une attache ;
- l’appartenance métier à un scénario ;
- la nomenclature des features ;
- la structure du graphe ;
- l’ancre persistante.

## 9.4 Mutabilité contrôlée

Le cache peut être modifié pendant une interaction.

Exemple :

- déplacement visuel en temps réel ;
- rotation d’aperçu ;
- drag ;
- animation ;
- prévisualisation d’une opération.

Mais toute modification durable doit suivre le cycle :

```text
intention utilisateur
        ↓
commande métier
        ↓
mise à jour de la topologie ou de l’ancre
        ↓
reconstruction ou synchronisation contrôlée
        ↓
rafraîchissement du cache
```

## 9.5 Interdiction de promotion implicite

Une modification de `_last_drawn` ne doit jamais devenir automatiquement une vérité métier sans passer par une commande explicite et validée.

---

# 10. Le canvas comme vue

## 10.1 Responsabilité

Le canvas doit :

- afficher la projection ;
- permettre la sélection ;
- collecter les gestes ;
- présenter les états temporaires ;
- déclencher des intentions.

Il ne doit pas :

- définir l’identité métier ;
- créer directement des relations topologiques sans commande ;
- devenir la seule source d’une pose persistante ;
- porter une logique de reconstruction cachée.

## 10.2 Interactions

Une interaction graphique doit être traduite en commande de domaine.

Exemples :

```text
Déplacer l’assemblage
→ modifier l’ancre
```

```text
Attacher deux arêtes
→ créer une TopologyAttachment
```

```text
Détacher deux triangles
→ supprimer une TopologyAttachment
```

```text
Changer de lumière
→ changer la variante sélectionnée du slot
```

```text
Supprimer un triangle
→ modifier la présence du slot dans le scénario et ses relations
```

---

# 11. Groupes et composantes

## 11.1 Question architecturale

Le terme « groupe » peut actuellement recouvrir plusieurs réalités :

- groupe graphique manipulable ;
- ensemble de triangles sélectionnés ;
- composante connexe du graphe ;
- agrégat topologique ;
- structure persistante possédant sa propre identité ;
- cache d’organisation UI.

Il faut distinguer ces notions.

## 11.2 Hypothèse cible forte

Si les attaches décrivent complètement l’assemblage et si le graphe est un arbre, alors une partie des groupes peut être une propriété dérivée :

> une composante connexe du graphe topologique.

Dans ce cas, il n’est pas nécessaire de persister une appartenance de groupe redondante si elle peut être recalculée à partir des attaches.

## 11.3 Prudence

Il ne faut toutefois pas supprimer ou redéfinir immédiatement les groupes sans audit du code.

Certains groupes peuvent porter :

- une identité fonctionnelle ;
- un ordre ;
- des métadonnées ;
- une ancre ;
- une sémantique utilisateur ;
- une notion de sélection durable ;
- une frontière ;
- des informations non déductibles du graphe.

## 11.4 Travail futur

La documentation et le code devront distinguer explicitement :

1. **Composante topologique**  
   Ensemble connexe dérivé des attaches.

2. **Groupe métier**  
   Agrégat éventuellement persistant possédant une signification propre.

3. **Groupe UI**  
   Structure temporaire de manipulation ou d’affichage.

4. **Sélection**  
   État transitoire de l’interface.

Aucune équivalence ne doit être supposée sans preuve.

---

# 12. Frontières, contours et chemins

## 12.1 Statut dérivé

Les contours et frontières doivent être considérés comme des données dérivées de :

- la topologie ;
- la sélection des variantes ;
- les poses reconstruites.

## 12.2 Invalidation

Ils doivent être invalidés lorsque changent :

- une variante ;
- une attache ;
- une orientation ;
- une ancre ;
- la structure d’une composante ;
- une géométrie locale du catalogue.

## 12.3 Chemins

Les chemins ou triplets qui dépendent de la géométrie ou des frontières ne doivent pas être considérés comme durablement valides après une modification structurelle.

Le projet doit définir :

- leur source ;
- leur portée ;
- leur politique d’invalidation ;
- leur reconstruction ;
- les éventuelles métadonnées persistantes réellement nécessaires.

---

# 13. Persistance

## 13.1 Principe cible

La persistance durable doit enregistrer la source de vérité, pas obligatoirement toutes les projections calculées.

Un scénario devrait pouvoir être reconstruit à partir de :

```text
catalogue ou références de catalogue
+
sélection des variantes
+
attaches
+
orientations
+
ancre
+
métadonnées
```

## 13.2 Géométrie monde

Les poses monde individuelles ne devraient pas être nécessaires pour définir durablement le scénario si elles sont intégralement reconstructibles.

Elles peuvent néanmoins être enregistrées temporairement comme :

- cache d’ouverture ;
- accélérateur ;
- donnée de diagnostic ;
- snapshot de compatibilité ;
- aide à la migration.

Dans ce cas :

- leur statut doit être explicitement marqué comme dérivé ;
- elles doivent être vérifiées à l’ouverture ;
- elles ne doivent jamais gagner sur la topologie ;
- elles doivent pouvoir être supprimées sans perte fonctionnelle.

## 13.3 XML actuel

Le format actuel semble pouvoir enregistrer simultanément :

- des informations UI ou géométriques ;
- un snapshot du Core topologique.

Cette redondance crée un risque de double vérité.

Le travail futur devra :

1. cartographier précisément les données persistées ;
2. identifier leur autorité ;
3. distinguer données métier et caches ;
4. définir les règles de chargement ;
5. définir les règles de migration ;
6. prévoir la compatibilité avec les fichiers existants ;
7. ajouter une version de schéma explicite.

## 13.4 Catalogue et scénarios

Le futur format devra décider si :

- le catalogue est stocké dans le même fichier ;
- le scénario référence un catalogue externe ;
- chaque variante possède un identifiant global stable ;
- une empreinte ou version du catalogue est conservée ;
- les scénarios embarquent les géométries minimales nécessaires à leur reproductibilité.

## 13.5 Reproductibilité

Un scénario sauvegardé doit pouvoir être rouvert avec la même signification métier.

Il faut donc gérer le cas où une variante du catalogue est :

- modifiée ;
- supprimée ;
- renommée ;
- remplacée ;
- versionnée.

La simple référence à un nom humain peut être insuffisante.

---

# 14. Validateur transversal UI/Core

## 14.1 Justification

Avant toute migration importante, le système actuel doit être observable.

Le risque principal du modèle hybride est l’absence de validation globale entre :

- la topologie ;
- la projection géométrique ;
- `_last_drawn` ;
- les groupes UI ;
- les objets du canvas ;
- les caches dérivés.

Le premier outil de stabilisation doit donc être un validateur transversal.

## 14.2 Positionnement

Le validateur n’est pas un correcteur automatique.

Il doit :

- observer ;
- comparer ;
- signaler ;
- produire un rapport exploitable ;
- permettre des tests de caractérisation ;
- identifier les divergences avant qu’elles ne provoquent un plantage ou une corruption silencieuse.

## 14.3 Niveaux de validation

### Niveau 1 — Structure topologique

Vérifier :

- unicité des identifiants ;
- validité des références de slots ;
- validité des variantes ;
- validité des features ;
- absence de références orphelines ;
- symétrie cohérente des attaches ;
- absence de cycles si l’invariant arbre s’applique ;
- cohérence des composantes ;
- orientation valide ;
- absence d’auto-attache interdite ;
- contraintes d’usage d’une feature.

### Niveau 2 — Catalogue

Vérifier :

- existence de la variante sélectionnée ;
- appartenance de la variante au bon slot ;
- nomenclature conforme ;
- géométrie locale valide ;
- valeurs numériques finies ;
- triangles non dégénérés ;
- cohérence des arêtes ;
- stabilité des identifiants.

### Niveau 3 — Reconstruction

Vérifier :

- présence d’une ancre valide ;
- possibilité de parcourir la composante ;
- pose calculable pour chaque élément ;
- absence d’ambiguïté ;
- cohérence des transformations ;
- conformité géométrique des arêtes attachées ;
- respect des orientations ;
- résultat numérique fini.

### Niveau 4 — UI/Core

Vérifier :

- bijection entre éléments topologiques actifs et objets UI attendus ;
- correspondance des identifiants ;
- cohérence des coordonnées monde ;
- cohérence des groupes ;
- absence d’objet UI orphelin ;
- absence d’objet topologique non projeté ;
- cohérence des sélections ;
- cohérence des références vers le canvas.

### Niveau 5 — Caches dérivés

Vérifier :

- validité des contours ;
- validité des frontières ;
- validité des chemins ;
- état d’invalidation ;
- version ou génération de cache ;
- dépendances cohérentes.

## 14.4 Tolérances

Une tolérance unique ne suffit probablement pas.

Le système devra distinguer plusieurs classes :

1. **Tolérance de reconstruction/pose**  
   Pour comparer des transformations et coordonnées métier.

2. **Tolérance d’attache**  
   Pour vérifier la superposition des features liées.

3. **Tolérance d’affichage/picking**  
   Plus permissive et liée à l’ergonomie.

4. **Tolérance de sérialisation**  
   Pour les écarts dus à l’arrondi et au format.

5. **Tolérance de dégénérescence**  
   Pour détecter des triangles ou segments invalides.

## 14.5 Rapport

Le rapport devrait contenir :

- code stable de l’anomalie ;
- niveau de sévérité ;
- couche concernée ;
- identifiants métier ;
- valeurs attendues ;
- valeurs observées ;
- tolérance utilisée ;
- contexte de l’opération ;
- suggestion de reconstruction ou d’investigation.

## 14.6 Modes d’usage

Le validateur devrait pouvoir être utilisé :

- manuellement depuis un outil de diagnostic ;
- après une opération sensible ;
- dans les tests ;
- lors du chargement ;
- avant sauvegarde ;
- dans un mode debug ;
- dans des tests de non-régression.

---

# 15. Transactions et atomicité

## 15.1 Problème du modèle hybride

Une opération peut aujourd’hui modifier successivement :

- le Core ;
- `_last_drawn` ;
- les groupes ;
- le canvas ;
- les caches ;
- la persistance différée.

Si une étape échoue, l’état peut devenir partiellement mis à jour.

## 15.2 Principe cible

Une opération métier doit être atomique du point de vue du domaine.

Exemple pour une attache :

```text
valider la commande
→ modifier la topologie
→ reconstruire la projection
→ valider
→ publier le nouvel état
```

L’UI ne doit pas observer durablement un état intermédiaire incohérent.

## 15.3 Stratégies possibles

### Reconstruction dans un nouvel état

- cloner ou préparer un état de travail ;
- appliquer l’opération ;
- reconstruire ;
- valider ;
- remplacer l’état courant si succès.

### Journal de rollback

- enregistrer les changements ;
- annuler en cas d’échec.

### Commandes immuables ou snapshots

- créer une nouvelle version du scénario ;
- remplacer l’ancienne après validation.

Le choix dépendra des contraintes du code existant et des performances.

---

# 16. Opérations métier dans le modèle cible

## 16.1 Création d’un scénario

1. sélectionner les slots actifs ;
2. sélectionner une variante par slot ;
3. définir les attaches ;
4. définir les orientations ;
5. définir l’ancre ;
6. reconstruire ;
7. valider ;
8. afficher.

## 16.2 Changement de variante

1. vérifier que la variante appartient au slot ;
2. modifier la sélection ;
3. invalider les poses concernées ;
4. reconstruire ;
5. valider ;
6. remplacer le cache ;
7. rafraîchir le canvas.

## 16.3 Création d’une attache

1. identifier les deux slots ;
2. identifier les features ;
3. vérifier leur disponibilité ;
4. définir l’orientation ;
5. vérifier l’absence de cycle ;
6. ajouter l’attache ;
7. reconstruire ;
8. valider ;
9. publier.

## 16.4 Suppression d’une attache

1. supprimer la relation ;
2. recalculer les composantes ;
3. vérifier la politique d’ancrage de chaque composante ;
4. reconstruire ou conserver les poses selon les règles choisies ;
5. valider ;
6. publier.

## 16.5 Déplacement global

1. modifier la pose de l’ancre ;
2. reconstruire la composante ;
3. valider ;
4. mettre à jour le cache.

## 16.6 Suppression d’un triangle

1. déterminer les attaches affectées ;
2. supprimer ou désactiver le slot dans le scénario ;
3. recalculer les composantes ;
4. vérifier les ancres ;
5. reconstruire ;
6. valider.

## 16.7 Dégrouper

Le terme devra être reformulé selon son sens réel :

- suppression d’attaches ;
- séparation d’une composante ;
- dissociation d’un groupe UI ;
- suppression d’un agrégat métier.

L’opération ne peut pas être correctement stabilisée tant que cette ambiguïté n’est pas levée.

---

# 17. Invariants architecturaux

Les invariants suivants doivent servir de référence aux futurs développements.

## 17.1 Catalogue

- Un slot possède une identité stable.
- Une variante appartient à un seul slot.
- Toutes les variantes d’un slot exposent la même nomenclature.
- Les features sont identifiées sémantiquement.
- La géométrie locale est indépendante de la pose monde.

## 17.2 Scénario

- Un scénario sélectionne au plus une variante active par slot.
- Une attache référence des slots et des features stables.
- Une attache ne référence pas un objet graphique.
- Une attache conserve sa validité lors d’un changement de variante.
- L’orientation relative est explicitement disponible.
- L’ancre appartient au scénario.
- La topologie est la source de vérité.

## 17.3 Graphe

- Les attaches sont symétriques.
- Le parent-enfant n’est pas une propriété persistante.
- Les assemblages sont des arbres, sauf évolution métier explicitement décidée.
- Le parcours orienté est temporaire et calculé à partir de l’ancre.
- Un cycle doit être détecté et refusé ou explicitement géré.

## 17.4 Reconstruction

- La géométrie complète est reconstructible.
- La reconstruction est déterministe.
- Une modification de variante ne modifie pas les attaches.
- La pose monde est dérivée.
- Les caches sont invalidables et remplaçables.
- Une divergence UI/Core se résout par reconstruction depuis la topologie.

## 17.5 UI

- `_last_drawn` est un cache.
- Le canvas est une vue.
- Les identifiants de canvas ne sont jamais des identifiants métier.
- Les gestes UI produisent des commandes métier.
- Les modifications durables passent par le domaine.

## 17.6 Persistance

- Le format durable conserve la source de vérité.
- Les caches persistés sont explicitement dérivés.
- Une géométrie dérivée ne gagne jamais sur la topologie.
- Les versions de catalogue et de schéma doivent être maîtrisées.

---

# 18. Architecture conceptuelle cible

## 18.1 Couche Catalogue

Responsabilités :

- gérer les slots ;
- gérer les variantes ;
- garantir la nomenclature ;
- fournir la géométrie locale ;
- gérer les versions et identifiants.

Objets conceptuels :

```text
TriangleCatalog
TriangleSlot
TriangleVariant
FeatureDefinition
LocalTriangleGeometry
```

## 18.2 Couche Scénario

Responsabilités :

- sélectionner les variantes ;
- stocker les attaches ;
- stocker les orientations ;
- stocker l’ancre ;
- porter les métadonnées du scénario.

Objets conceptuels :

```text
Scenario
ScenarioVariantSelection
TopologyAttachment
ScenarioAnchor
ScenarioMetadata
```

## 18.3 Couche Reconstruction

Responsabilités :

- vérifier les préconditions ;
- construire le plan de parcours ;
- calculer les poses ;
- produire la projection ;
- signaler les erreurs.

Objets conceptuels :

```text
GeometryRebuilder
GeometryRebuildPlan
WorldPose
WorldTriangleGeometry
GeometryProjection
RebuildReport
```

## 18.4 Couche Validation

Responsabilités :

- valider le catalogue ;
- valider le scénario ;
- valider la topologie ;
- valider la reconstruction ;
- comparer projection et UI ;
- produire des diagnostics.

Objets conceptuels :

```text
CatalogValidator
ScenarioValidator
TopologyValidator
ProjectionValidator
UiCoreConsistencyValidator
ValidationReport
```

## 18.5 Couche UI

Responsabilités :

- projeter le modèle ;
- gérer les interactions ;
- maintenir les caches ;
- envoyer les commandes ;
- afficher les erreurs.

Objets conceptuels :

```text
CanvasProjectionAdapter
UiGeometryCache
SelectionState
InteractionController
ScenarioCommandDispatcher
```

---

# 19. Relation avec `TopologyWorld`

## 19.1 Point à clarifier

Le nom `TopologyWorld` peut laisser penser qu’il ne contient que des relations abstraites.

Or l’analyse précédente indique qu’il contient probablement aussi :

- des géométries locales ;
- des poses ;
- des informations dérivées ;
- des groupes ;
- des frontières ;
- des chemins.

Il faudra donc distinguer :

- ce qui appartient réellement à la source de vérité ;
- ce qui appartient au catalogue ;
- ce qui appartient au scénario ;
- ce qui appartient à la projection ;
- ce qui appartient aux caches.

## 19.2 Deux trajectoires possibles

### Conserver `TopologyWorld` comme agrégat global

Il pourrait orchestrer plusieurs sous-modèles clairement séparés :

```text
TopologyWorld
 ├─ Catalog
 ├─ Scenario
 ├─ Projection
 └─ DerivedCaches
```

### Éclater les responsabilités

Créer des agrégats distincts :

```text
TriangleCatalog
ScenarioTopology
GeometryProjection
```

Le choix ne doit pas être fait uniquement sur des considérations de nommage.

Il dépendra :

- des dépendances actuelles ;
- du coût de migration ;
- des tests ;
- de la lisibilité ;
- de la stabilité des API.

---

# 20. Architecture actuelle versus architecture cible

## 20.1 Architecture actuelle

Caractéristiques :

- modèle hybride ;
- autorité partagée ou ambiguë ;
- synchronisations bidirectionnelles ;
- opérations parfois initiées dans l’UI ;
- persistance redondante ;
- caches à invalidation partielle ;
- absence de validation transversale ;
- géométrie et topologie imbriquées.

## 20.2 Architecture cible

Caractéristiques :

- topologie autoritaire ;
- catalogue explicite ;
- variantes séparées des slots ;
- attaches indépendantes des variantes ;
- ancre persistante ;
- géométrie reconstruite ;
- `_last_drawn` réduit au rôle de cache ;
- commandes métier explicites ;
- validation transversale ;
- persistance centrée sur la source de vérité.

## 20.3 Transition

La cible ne doit pas être atteinte par une réécriture brutale.

Il faut :

1. documenter ;
2. observer ;
3. caractériser ;
4. valider ;
5. stabiliser les contrats ;
6. introduire les nouvelles abstractions ;
7. migrer progressivement ;
8. supprimer les anciennes autorités seulement après preuve.

---

# 21. Stratégie de stabilisation recommandée

## Phase 0 — Documentation et cartographie

- conserver le document d’état des lieux ;
- conserver `STABILISATION_UI_CORE_PHASE1.md` comme analyse de l’existant ;
- ajouter le présent document comme cible ;
- produire une matrice actuel/cible ;
- identifier les fonctions critiques.

## Phase 1 — Observation

- créer le validateur transversal ;
- ajouter des rapports de cohérence ;
- instrumenter les opérations sensibles ;
- détecter les divergences ;
- ne pas encore corriger automatiquement.

## Phase 2 — Tests de caractérisation

Couvrir :

- dessin initial ;
- collage ;
- détachement ;
- déplacement ;
- groupement ;
- dégroupement ;
- suppression ;
- miroir ;
- rotation ;
- chargement ;
- sauvegarde ;
- changement de scénario ;
- reconstruction ;
- gestion des contours ;
- chemins.

## Phase 3 — Contrats de synchronisation

Pour chaque opération :

- définir l’entrée autoritaire ;
- définir les changements métier ;
- définir les caches invalidés ;
- définir la reconstruction ;
- définir les validations ;
- définir le rollback.

## Phase 4 — Introduction du catalogue

- modéliser `TriangleSlot` ;
- modéliser `TriangleVariant` ;
- préserver la nomenclature ;
- introduire les identifiants stables ;
- adapter progressivement les scénarios.

## Phase 5 — Reconstruction depuis la topologie

- créer un moteur explicite ;
- reconstruire une composante ;
- reconstruire un scénario ;
- comparer au résultat historique ;
- ajouter des tests déterministes.

## Phase 6 — Réduction de l’autorité UI

- empêcher les écritures métier directes dans `_last_drawn` ;
- traduire les interactions en commandes ;
- reconstruire après les commandes ;
- formaliser les caches.

## Phase 7 — Persistance cible

- versionner le format ;
- persister catalogue, sélection, attaches et ancre ;
- rendre la géométrie optionnelle ;
- migrer les anciens fichiers ;
- vérifier la reproductibilité.

## Phase 8 — Nettoyage

- supprimer les synchronisations obsolètes ;
- supprimer les doubles sources ;
- simplifier les groupes ;
- clarifier les responsabilités ;
- réduire la dette de `assembleur_tk.py`.

---

# 22. Risques principaux

## RISK-TARGET-001 — Confondre topologie et projection

**Risque :** continuer à ajouter des fonctions métier dans `_last_drawn`.  
**Impact :** maintien de la double vérité.  
**Réponse :** imposer des commandes métier et une reconstruction descendante.

## RISK-TARGET-002 — Introduire le catalogue sans identifiants stables

**Risque :** scénarios fragiles lors du renommage ou de la modification des variantes.  
**Impact :** impossibilité de rouvrir fidèlement les scénarios.  
**Réponse :** identifiants durables et versionnement.

## RISK-TARGET-003 — Déduire l’orientation depuis la géométrie

**Risque :** perte de déterminisme lors d’un changement de variante.  
**Impact :** reconstruction ambiguë ou retournée.  
**Réponse :** orientation explicitement portée par l’attache.

## RISK-TARGET-004 — Supposer que les groupes sont seulement des composantes

**Risque :** perte de sémantique fonctionnelle.  
**Impact :** régression UI ou métier.  
**Réponse :** audit précis avant simplification.

## RISK-TARGET-005 — Sauvegarder encore deux vérités

**Risque :** divergence entre Core et XML géométrique.  
**Impact :** fichiers incohérents.  
**Réponse :** hiérarchie d’autorité explicite et validation à l’ouverture.

## RISK-TARGET-006 — Migration trop brutale

**Risque :** régression massive.  
**Impact :** perte de fonctionnalités historiques.  
**Réponse :** migration incrémentale appuyée par les tests.

## RISK-TARGET-007 — Reconstruction partielle mal maîtrisée

**Risque :** caches ou poses obsolètes.  
**Impact :** divergences visuelles.  
**Réponse :** commencer par une reconstruction complète fiable avant optimisation.

## RISK-TARGET-008 — Tolérances incohérentes

**Risque :** faux positifs ou incohérences silencieuses.  
**Impact :** validation inutilisable.  
**Réponse :** classes de tolérances nommées et centralisées.

## RISK-TARGET-009 — Cycles introduits accidentellement

**Risque :** l’algorithme de reconstruction d’arbre devient insuffisant.  
**Impact :** géométrie non déterminée ou contradictoire.  
**Réponse :** validation stricte de l’invariant acyclique.

## RISK-TARGET-010 — Confusion entre ancre métier et état de drag

**Risque :** la pose persistante dépend d’un cache temporaire.  
**Impact :** réouverture différente.  
**Réponse :** commit explicite du déplacement vers l’ancre.

---

# 23. Questions encore ouvertes

Les points suivants ne sont pas encore tranchés et doivent être analysés avant implémentation.

## 23.1 Plusieurs composantes

- Un scénario peut-il contenir plusieurs arbres ?
- Chaque arbre possède-t-il une ancre ?
- Les composantes non ancrées conservent-elles une pose libre ?
- Doivent-elles être interdites ?

## 23.2 Granularité de l’ancre

- Pose complète d’un triangle ?
- Sommet + point monde ?
- Arête + orientation ?
- Point de référence externe ?
- Plusieurs types supportés ?

## 23.3 Cycle de vie des variantes

- Une variante peut-elle être modifiée après utilisation ?
- Faut-il la versionner ?
- Un scénario référence-t-il une version précise ?
- Peut-on remplacer une variante sans migration ?

## 23.4 Présence des slots

- Tous les slots sont-ils toujours présents ?
- Un scénario peut-il n’en sélectionner qu’une partie ?
- Existe-t-il un état « non résolu » sans variante ?
- Comment représenter une hypothèse absente ?

## 23.5 Groupes

- Sont-ils dérivés ou persistants ?
- Portent-ils une ancre ?
- Ont-ils une identité métier ?
- `group_pos`, `edge_in`, `edge_out` sont-ils métier ou UI ?

## 23.6 Reconstruction locale

- Peut-on reconstruire seulement un sous-arbre ?
- Comment déterminer le côté qui doit rester fixe ?
- L’ancre globale suffit-elle toujours ?
- Faut-il conserver temporairement une frontière fixe lors de certaines opérations ?

## 23.7 Persistance

- Catalogue interne ou externe ?
- Quelle compatibilité XML ?
- Quel format de migration ?
- Quelle politique de cache géométrique ?

## 23.8 Undo/redo

- Les commandes métier sont-elles journalisées ?
- Un changement de variante doit-il être annulable ?
- Les projections doivent-elles être recalculées lors d’un undo ?

## 23.9 Performance

- Coût d’une reconstruction complète ;
- nécessité d’une reconstruction incrémentale ;
- cache des transformations ;
- indexation des dépendances ;
- comportement sur 32 triangles et plusieurs variantes.

---

# 24. Matrice des sources de vérité

| Information | Source de vérité cible | Projection/cache autorisé | Commentaire |
|---|---|---|---|
| Identité du triangle logique | Catalogue / `TriangleSlot` | UI | Stable |
| Hypothèse disponible | Catalogue / `TriangleVariant` | UI | Stable et versionnable |
| Variante sélectionnée | Scénario | UI | Décision métier |
| Nomenclature O/B/L | Catalogue / définition du slot | UI | Invariant |
| Géométrie locale | Variante | Cache éventuel | Pas une pose monde |
| Attache entre features | Scénario topologique | Représentation graphique | Indépendante des variantes |
| Orientation de l’attache | Scénario topologique | Aperçu UI | Nécessaire à la reconstruction |
| Ancre | Scénario | État de drag temporaire | Persistante |
| Pose monde | Reconstruction | `_last_drawn` | Dérivée |
| Coordonnées monde | Reconstruction | `_last_drawn`, canvas | Dérivées |
| Groupe topologique | À déterminer | UI | Peut être dérivé |
| Sélection UI | UI | — | Temporaire |
| Identifiant canvas | Canvas | — | Non métier |
| Contour | Dérivé | Cache | Invalidation nécessaire |
| Chemin | Dérivé ou métier à clarifier | Cache | À analyser |
| XML géométrique historique | Compatibilité | — | Ne doit pas gagner sur la topologie |

---

# 25. Critères d’acceptation de l’architecture cible

L’architecture cible sera considérée comme effectivement atteinte lorsque les propriétés suivantes seront démontrées.

## 25.1 Reconstruction

- Un scénario peut être reconstruit sans lire les coordonnées monde historiques.
- Le résultat est déterministe.
- Les attaches sont respectées.
- L’ancre reste fixe.
- Les variantes sélectionnées produisent les bonnes formes.

## 25.2 Changement de variante

- Une variante peut être remplacée sans modifier les attaches.
- La figure est reconstruite automatiquement.
- Les références O/B/L restent valides.
- Le canvas est rafraîchi sans corruption.

## 25.3 Autorité

- Une divergence de `_last_drawn` peut être corrigée par reconstruction.
- La suppression du cache ne détruit aucune information métier.
- Les commandes UI modifient le scénario, pas directement la vérité graphique.

## 25.4 Persistance

- Un fichier minimal contient suffisamment d’information pour reconstruire.
- La géométrie persistée éventuelle est vérifiable et jetable.
- Les versions de variantes sont maîtrisées.

## 25.5 Validation

- Les incohérences sont détectées.
- Les erreurs sont localisées.
- Les invariants d’arbre sont vérifiés.
- Les tests couvrent les opérations principales.

---

# 26. Directives pour les futures sessions GPT et Codex

Toute future analyse doit partir des principes suivants :

1. Ne pas traiter `_last_drawn` comme une seconde source de vérité.
2. Ne pas proposer de synchronisation bidirectionnelle permanente.
3. Ne pas reconstruire les attaches lors d’un changement de variante.
4. Ne pas introduire de parent-enfant persistant dans les attaches.
5. Ne pas déduire les features à partir d’indices fragiles.
6. Ne pas ajouter de surcharge d’ancre sans cas d’usage.
7. Ne pas modifier la persistance avant d’avoir cartographié les fichiers existants.
8. Ne pas optimiser la reconstruction partielle avant d’avoir une reconstruction complète correcte.
9. Ne pas supprimer les groupes avant d’avoir établi leur sémantique.
10. Toujours distinguer :
   - état actuel ;
   - cible ;
   - stratégie de transition.
11. Toujours accompagner une évolution structurelle :
   - d’invariants ;
   - de tests ;
   - d’un plan de migration ;
   - d’une validation.
12. Privilégier les changements incrémentaux et réversibles.

---

# 27. Livrables documentaires recommandés après ce document

## 27.1 Mise à jour de `STABILISATION_UI_CORE_PHASE1.md`

Le document doit être enrichi avec :

- la décision de source de vérité ;
- la définition du cache `_last_drawn` ;
- la matrice actuel/cible ;
- la priorité du validateur transversal ;
- les contrats de synchronisation descendante ;
- les implications sur les opérations sensibles.

## 27.2 Spécification du validateur

Créer :

```text
docs/SPEC_VALIDATEUR_UI_CORE.md
```

avec :

- règles ;
- codes d’erreur ;
- tolérances ;
- API ;
- modes d’usage ;
- intégration aux tests.

## 27.3 Spécification du catalogue

Créer :

```text
docs/SPEC_CATALOGUE_TRIANGLES.md
```

avec :

- modèle de données ;
- identifiants ;
- variantes ;
- versions ;
- nomenclature ;
- import/export.

## 27.4 Spécification du scénario topologique

Créer :

```text
docs/SPEC_SCENARIO_TOPOLOGIQUE.md
```

avec :

- sélection ;
- attaches ;
- orientations ;
- ancre ;
- composantes ;
- invariants.

## 27.5 Spécification du moteur de reconstruction

Créer :

```text
docs/SPEC_RECONSTRUCTION_GEOMETRIQUE.md
```

avec :

- algorithme ;
- entrées ;
- sorties ;
- erreurs ;
- tolérances ;
- performances ;
- tests.

## 27.6 Plan de migration

Créer :

```text
docs/PLAN_MIGRATION_MODELE_TOPOLOGIQUE.md
```

avec :

- étapes ;
- dépendances ;
- compatibilité ;
- risques ;
- rollback ;
- critères de sortie.

---

# 28. Conclusion

Le projet a dépassé le stade d’un simple assembleur géométrique.

Son véritable modèle métier est désormais identifiable :

```text
Catalogue de triangles logiques et de variantes
        +
Scénario sélectionnant une variante par slot
        +
Graphe topologique d’attaches symétriques
        +
Informations d’orientation
        +
Ancre persistante
        ↓
Reconstruction géométrique déterministe
        ↓
Cache UI
        ↓
Canvas
```

La topologie n’est pas une information supplémentaire venant compléter la géométrie.

Elle constitue la définition durable de l’assemblage.

La géométrie monde est une conséquence calculée.

Le modèle hybride actuel peut être maintenu pendant la transition, mais uniquement avec une hiérarchie claire :

```text
Topologie
   ↓
Projection géométrique
   ↓
_last_drawn
   ↓
Canvas
```

La prochaine étape ne doit pas être une refonte immédiate.

Elle doit être la stabilisation de l’existant par :

1. la documentation ;
2. la validation transversale ;
3. les tests de caractérisation ;
4. la formalisation des contrats ;
5. l’introduction progressive du catalogue ;
6. la construction d’un moteur explicite de reconstruction.

Ce document constitue désormais la référence conceptuelle de cette trajectoire.
