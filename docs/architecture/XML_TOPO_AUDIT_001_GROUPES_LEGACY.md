# XML-TOPO-AUDIT-001 — Audit préalable à la suppression des groupes legacy dans la persistance XML

## 1. Objet et périmètre

Cet audit, strictement documentaire, vérifie si le topoSnapshot XML v4 permet déjà de restaurer la topologie sans la projection historique viewer.groups / nodes / group_id.

Sources inspectées :

- src/assembleur_io.py : saveScenarioXml() et loadScenarioXml() ;
- src/assembleur_core.py : TopologyWorld._exportPhysicalSnapshot() et _importPhysicalSnapshot() ;
- TopologyChemins._saveToXml() et _loadFromXml() ;
- les trois scénarios XML v4 présents dans scenario/.

Aucun code de production, format XML, test ou comportement n'est modifié par cet audit.

## 2. Résumé exécutif

### Conclusion

Le topoSnapshot est une source suffisante pour restaurer la **topologie Core** : éléments, poses, attachments, groupes canoniques, DSU et frontières dérivées. Les groupes Core ne sont pas enregistrés en parallèle : ils sont reconstruits à partir des éléments puis des attachments. C'est le comportement attendu d'une architecture Core-first.

Le snapshot ne contient pas la projection Canvas (last_drawn) : identifiant catalogue, coordonnées affichées, ordre de projection, labels, état UI. Un document de scénario complet a donc encore besoin d'une projection UI, mais pas des groupes UI comme vérité topologique.

<groups> ne complète le snapshot avec aucune donnée Core. Il reconstruit seulement le miroir legacy : viewer.groups, nodes, last_drawn[*][group_id], edge_in et edge_out.

| Question | Réponse |
|---|---|
| Le snapshot restaure-t-il la topologie Core ? | **Oui.** |
| <groups> apporte-t-il une donnée Core manquante ? | **Non.** |
| Le snapshot restaure-t-il à lui seul le Canvas/UI ? | **Non.** |
| <groups> est-il encore une vérité métier ? | **Non**, seulement une compatibilité UI. |
| Peut-on supprimer <groups> sans autre travail ? | **Non**, car le chargeur courant reconstruit encore le miroir legacy. |
| Verdict | **A pour la topologie ; B pour la duplication legacy ; D pour une incohérence de persistance critique.** |

### Incohérence critique trouvée

Le chargeur exige, après import du snapshot, que chaque triangle chargé possède topoElementId. L'écrivain XML actuel ne sérialise pourtant que id, mirrored et group pour triangle.

Les trois fichiers réels inspectés (Scenario_FrontiereActuelle finale.xml, v5.xml, v6.xml) sont v4, possèdent un topoSnapshot, mais tous leurs triangles ont exactement ces trois attributs et pas de topoElementId. Au regard du chargeur actuel, la lecture atteint donc l'erreur « Triangle … sans topoElementId dans le snapshot Core ». Ce défaut est indépendant du retrait de groups et doit être traité explicitement dans un chantier ultérieur.

## 3. Format v4 réellement écrit et relu

Le document v4 porte : topoSnapshot JSON Core ; chemins ; source, map, clock, clockRef, guides, listbox ; groups ; triangles ; words.

Le chargeur actuel reconstruit d'abord les triangles et viewer.groups, importe ensuite le topoSnapshot, puis vérifie que les groupes UI pointent tous vers un seul groupe Core canonique.

triangles + groups XML
→ CanvasObjectsCollection + viewer.groups legacy
→ topoSnapshot vers TopologyWorld
→ contrôle Core/UI et projection topoGroupId

Ainsi, groups n'est pas une entrée utilisée pour créer ou modifier le Core : c'est une projection validée a posteriori contre lui.

## 4. Audit détaillé de topoSnapshot

### 4.1 Données exportées

TopologyWorld._exportPhysicalSnapshot() exporte :

| Domaine | Données |
|---|---|
| Configuration | fusion_distance_km, worldYAxisDown, éventuellement vertex_edge_endpoint_epsilon |
| Éléments | element_id, nom, labels/types de sommets, longueurs, côtés intrinsèques, repère local, coordonnées locales, meta |
| Pose | matrice R, translation T, booléen mirrored |
| Attachments | attachment_id, kind, les deux TopologyFeatureRef (type, élément, index, t), params, source |

Les params des attachments sont sauvegardés. Les informations observées dans les scénarios, par exemple mapping, t et edgeFrom, sont donc portées par la connexion Core, non par les edge_in/out legacy.

### 4.2 Reconstruction

TopologyWorld._importPhysicalSnapshot() :

1. restaure la configuration ;
2. recrée chaque TopologyElement avec son element_id ;
3. place chaque élément dans un groupe singleton Core ;
4. rétablit sa pose ;
5. recrée et charge chaque TopologyAttachment ;
6. reconstruit les listes de groupes ;
7. recalcule concept et frontière de chaque groupe canonique.

Les tables de DSU et les groupes vivants ne sont donc pas enregistrés comme une seconde vérité XML ; ils sont dérivés à l'import des éléments et attachments. find_group() et les groupes canoniques sont restaurés par le Core.

### 4.3 Données absentes du snapshot

Le snapshot ne porte pas :

- l'ordre de last_drawn ;
- l'identifiant catalogue triangle[id] ;
- les points affichés O/B/L, labels et détails de rendu ;
- group_id UI ;
- viewer.groups[*][nodes] ;
- edge_in / edge_out ;
- l'état de vue, carte, horloge, listbox, mots, sélection ;
- l'état de chemins persistant, sauvegardé séparément dans chemins.

La réponse exacte est donc : snapshot suffisant pour le **monde topologique**, insuffisant pour le **document UI complet**.

## 5. Tableau producteur → source → lecteur

| Champ XML | Producteur | Source réelle | Lecteur | Redondant avec Core ? | Conclusion |
|---|---|---|---|---|---|
| topoSnapshot | saveScenarioXml() | TopologyWorld | _importPhysicalSnapshot() | Non | Vérité Core, indispensable |
| triangle id | last_drawn[id] | catalogue/projection | chargeur, listbox, mots | Non | Projection à conserver tant que le Canvas dépend du catalogue |
| triangle mirrored | last_drawn[mirrored] | projection/pose | chargeur | Partiellement | À auditer avec la future projection Canvas |
| O/B/L | last_drawn[pts] | projection Canvas | chargeur/rendu | Potentiellement dérivable, pas aujourd'hui | Hors retrait des groupes |
| triangle group | last_drawn.get(group_id, 0) | UI legacy | aucun | Oui | Écriture legacy morte |
| triangle topoElementId | **non écrit** | devrait venir du Core/projection | requis par chargeur | Non | Incohérence critique |
| triangle topoGroupId | **non écrit** | get_group_of_element() | optionnel puis recalculé | Oui | Ne pas persister par triangle |
| groups | viewer.groups | miroir legacy | chargeur | Oui pour membres | Compatibilité seulement |
| group id | gid UI | allocateur legacy | chargeur / _next_group_id | Oui | Identité UI, pas métier |
| group topoGroupId | **non écrit** | Core | optionnel puis vérifié | Oui | Copie inutile |
| node tid | groups[*][nodes] | index de projection | réhydrate le miroir | Membres : oui ; index Canvas : non | Projection/compatibilité |
| node edge_in/out | mêmes nodes | chaîne historique | chargeur uniquement | Oui | Reliquat legacy |
| chemins | TopologyChemins | état de chemin Core | _loadFromXml() | Non | À conserver |
| clockRef, guides | scénario | UI référant le Core | chargeur | Non | À conserver |
| words, listbox, carte, horloge, vue | UI | UI | chargeur | Non | Hors sujet |

## 6. groups, nodes, edge_in/out

### 6.1 Effet réel de groups au chargement

Pour chaque group le chargeur :

1. crée viewer.groups[gid] avec id et nodes ;
2. écrit last_drawn[tid][group_id] pour chaque node ;
3. vérifie après import Core que tous les tid du groupe UI résolvent le même groupe Core ;
4. complète ou contrôle topoGroupId depuis world.get_group_of_element().

Aucune de ces étapes ne crée un élément, un attachment, une union DSU, une frontière ou un groupe Core.

### 6.2 Information portée par nodes

| Information | Disponible dans le Core ? | Rôle résiduel |
|---|---|---|
| Membres d'un groupe | Oui : get_group_of_element() / getGroupElementIds() | Dupliqué |
| Connectivité | Oui : attachments | Dupliqué / moins expressif |
| Incidence d'arête | Oui : feature refs et params d'attachments | Dupliqué |
| Ordre de chaîne historique | Pas comme propriété générique du groupe | Historique UI seulement |
| Index dans last_drawn | Non | Projection Canvas |
| gid UI | Non, et ne doit pas être Core | Compatibilité legacy |

nodes ne porte aucune vérité topologique absente du snapshot. Il peut néanmoins porter un ordre de projection historique : si cet ordre doit survivre, il faut le représenter explicitement comme projection, pas le masquer derrière un groupe métier.

### 6.3 edge_in / edge_out

L'écrivain les sérialise ; le chargeur se limite à les valider (OB, BL, LO) et les stocker dans le miroir UI. Il ne les convertit jamais en TopologyAttachment et ne les utilise pas pour reconstruire le snapshot.

Le commentaire du chargeur disant que edge_in est utilisé par la simulation automatique est devenu obsolète : la simulation et la connectivité sont désormais Core-first. Ce rapport le signale, sans modifier le code.

## 7. group_id, topoGroupId, _next_group_id et _group_nodes

### triangle group / group_id

L'écrivain produit triangle group, mais le chargeur ne le lit jamais. Il initialise group_id à None, puis l'alimente exclusivement au parcours de groups/node. Dans le contrat courant, triangle group est une écriture morte ; node tid est le chemin qui reconstruit le miroir UI.

### topoGroupId

Après import, le chargeur appelle world.get_group_of_element(topoElementId) pour chaque triangle. Si topoGroupId manque, il le remplit ; pour un groupe UI, il le dérive ou vérifie l'égalité. Cette valeur dans une projection est une copie non autoritaire.

En revanche, les références Core de clockRef, guides et TopologyChemins.groupId restent légitimes : elles ne sont ni des groupes UI ni des duplications de Scenario.groups.

### _next_group_id

Après lecture, il vaut max(viewer.groups.keys()) + 1, ou 1. C'est exclusivement un allocateur de gid UI historique ; ce n'est pas une donnée Core. Il devra disparaître avec le dernier flux qui crée ou maintient un groupe UI.

### _group_nodes

_group_nodes(gid) lit simplement self.groups[gid][nodes]. Dans le périmètre XML, il sert à l'écrivain pour produire groups. Il ne consulte pas le Core et n'apporte aucune abstraction métier moderne.

## 8. Incohérence writer/reader sur topoElementId

### Preuve dans le code

Le chargeur autorise la lecture facultative de triangle topoElementId, mais après _importPhysicalSnapshot() il refuse toute entrée qui ne le contient pas ou qui ne référence pas world.elements.

L'écrivain construit toutefois les triangles avec seulement : id, mirrored, group. Ni topoElementId ni topoGroupId ne sont écrits.

### Preuve dans les fichiers du dépôt

Les trois scénarios XML analysés sont v4, ont un snapshot et 32 triangles. Chaque triangle possède exactement id, mirrored, group.

L'absence de topoElementId est donc démontrée dans les fichiers réels comme dans l'écrivain. Le chargement courant ne peut pas associer ces entrées de projection à un élément du snapshot et déclenche son contrôle strict.

Cette anomalie ne doit pas être corrigée dans un chantier de retrait de groupes sans décision explicite de contrat XML.

## 9. État de la compatibilité des fichiers v4

| Question | Réponse |
|---|---|
| Le Core peut-il être importé sans groups ? | Oui |
| Le chargeur actuel peut-il reconstruire son miroir viewer.groups sans groups ? | Non |
| groups est-il utilisé pour reconstruire des attachments ? | Non |
| groups est-il utilisé comme source de groupe Core ? | Non, seulement validé contre le Core |
| edge_in/out sont-ils nécessaires au DSU ou aux frontières ? | Non |
| topoGroupId d'un triangle doit-il être persisté ? | Non, il se recalcule depuis le Core |

## 10. Risques et prérequis avant retrait

1. **Résoudre le lien projection → Core.** La persistance de topoElementId doit être cohérente entre écriture, lecture et anciens documents v4.
2. **Définir la projection Canvas minimale.** Décider explicitement du maintien des points, labels, mirrored, ordre et IDs catalogue.
3. **Préserver les états légitimes hors snapshot.** TopologyChemins, clockRef, guides, listbox, mots, vue et carte ne sont pas des groupes legacy.
4. **Retirer les lecteurs de miroir UI avant le miroir.** viewer.groups, _group_nodes et _next_group_id ne doivent pas être supprimés mécaniquement tant qu'un flux XML/compatibilité les utilise.
5. **Caractériser la rétrocompatibilité v4.** La suppression doit définir le comportement voulu pour les documents existants avec groups.

Ce qui **ne constitue pas** un risque topologique : perdre groups ne perd pas les éléments, poses, attachments, DSU, groupes canoniques, frontières ou connectivité ; le snapshot couvre ces données.

## 11. Feuille de route suggérée, sans implémentation

### XML-TOPO-MIG-002 — Contrat de projection Core ↔ Canvas

Définir et caractériser le lien durable entre une entrée Canvas persistée et son TopologyElement. Ce chantier doit traiter l'incohérence topoElementId writer/reader.

- Complexité : moyenne
- Risque : fort (compatibilité des documents)
- Bénéfice : condition nécessaire à un chargement Core-first fiable

### XML-TOPO-MIG-003 — Lecture Core-first et compatibilité v4

Importer le snapshot avant de dériver la projection, rendre groups optionnel de compatibilité et caractériser documents avec/sans bloc legacy.

- Complexité : moyenne
- Risque : moyen à fort
- Bénéfice : une seule source de vérité pendant la réhydratation

### XML-TOPO-MIG-004 — Retrait de l'écriture legacy des groupes

Après migration des consommateurs UI, supprimer la sérialisation de groups, nodes, edge_in/out et triangle group.

- Complexité : faible à moyenne
- Risque : moyen (rétrocompatibilité uniquement)
- Bénéfice : disparition de la projection legacy à l'écriture

### XML-TOPO-MIG-005 — Retrait de la réhydratation legacy

Supprimer viewer.groups, _group_nodes, _next_group_id et les validations Core/UI liées lorsqu'il n'existe plus aucun consommateur runtime.

- Complexité : moyenne
- Risque : moyen
- Bénéfice : XML et runtime entièrement Core-first

## 12. Conclusion

topoSnapshot est déjà l'unique représentation métier nécessaire à la restauration des groupes et connexions. Les groups legacy les dupliquent et ne servent plus à restaurer le Core.

La suppression future de groups est donc justifiée, mais doit être précédée par la définition du contrat de projection Canvas et par la résolution de l'incohérence actuelle topoElementId : **l'écrivain ne le produit pas, le chargeur l'exige**.

> Verdict : **B — groups/nodes dupliquent la topologie Core ; D — le format actuel présente un défaut writer/reader critique à traiter séparément.**

