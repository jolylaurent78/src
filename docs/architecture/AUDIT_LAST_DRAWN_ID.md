# AUDIT — Usages restants de `last_drawn["id"]`

## 1. Objet et périmètre

Cet audit est strictement documentaire. Il décrit les lectures, écritures et
transports de l'identifiant historique de catalogue porté par une entrée de
projection :

```python
{
    "id": 17,
    "topoElementId": "T17",
    "pts": {...},
}
```

Il ne traite pas les autres clés nommées `id` qui ne représentent pas un
triangle projeté : identifiants Tk, colonnes de tables UI, identifiants de
noeuds, d'attachments, de groupes ou de widgets.

Constat de départ : `id` ne construit plus le texte rouge. Celui-ci est
produit par `_build_triangle_display_label()` à partir de
`TopologyElement.meta["triRank"]` et de la pose Core. L'identité topologique
canonique reste `topoElementId`; le rang catalogue est déjà porté par
`TopologyElement.meta["triRank"]`.

## 2. Résumé exécutif

`last_drawn["id"]` n'est plus une donnée géométrique ni topologique. C'est un
identifiant de catalogue, historiquement commun au DataFrame, à la listbox,
aux éléments XML `<triangle id="…">`, à `ScenarioAssemblage.tri_ids` et aux
annotations textuelles historiques, désormais supprimées.

Les lecteurs runtime se répartissent ainsi :

| Domaine | Usages fonctionnels | Risque de migration |
|---|---:|---|
| XML et rétrocompatibilité v4 | 3 | Élevé : le format écrit encore `id` et les fichiers v4 l'utilisent pour retrouver `Txx`. |
| Listbox / disponibilité du catalogue | 5 | Moyen : l'UI catalogue est rangée par numéro, pas par ElementId. |
| Annotations textuelles historiques | 0 | Sans objet : fonctionnalité supprimée. |
| Simulation automatique / préfixes | 4 | Moyen à élevé : `Scenario.tri_ids` est une chronologie de rangs catalogue. |
| Collection / diagnostics | 6 | Faible à moyen : index secondaire, dump, F11 et outil matplotlib. |
| Tests | 3 | Faible : seulement des assertions ou constructions de projections. |

Conclusion : le champ peut disparaître de la projection, mais pas dans une
suppression mécanique. Les deux blocages structurants sont le contrat XML
historique et le modèle de catalogue (`tri_ids`, `_placed_ids`).
Ces flux doivent d'abord consommer soit `topoElementId`, soit `triRank` lu
depuis le Core.

## 3. Cartographie détaillée

Les numéros de ligne sont indicatifs et correspondent à l'état du dépôt lors
de cet audit.

| Fichier : ligne | Fonction / extrait | Rôle réel | Catégorie | Remplaçabilité |
|---|---|---|---|---|
| `src/assembleur_sim.py:51` | `PlacedTriangle.fromLegacyDict(): triangleId=entry["id"]` | Importe une projection historique dans le conteneur de simulation. | Compatibilité / transport | Remplaçable par `topoElementId` pour l'identité; conserver `triRank` si le catalogue est encore requis. |
| `src/assembleur_sim.py:60-66` | `PlacedTriangle.toLegacyDict(): "id": self.triangleId` | Produit la clé de projection aujourd'hui utilisée par les scénarios auto. | Producteur / compatibilité | Suppressible après migration de tous les lecteurs auto et XML. |
| `src/assembleur_sim.py:103-106` | `findByTriangleId()` | Recherche de simulation par rang catalogue. | Catalogue | Remplaçable par `findByTopologyElementId()`; à conserver seulement tant que les appelants possèdent un `triRank`. |
| `src/assembleur_sim.py:1370-1390` | `build_local_triangle(tri_id)` puis `df["id"] == tri_id` | Lit le catalogue afin de construire un triangle local. Ce n'est pas une lecture de `last_drawn`, mais c'est la source du rang historique. | Catalogue | Indispensable tant que l'entrée de simulation est une liste de rangs; Core ne remplace pas le catalogue de départ. |
| `src/canvas_objects_collection.py:83-95` | `_by_triangle_id`, `get_by_triangle_id()`, `get_index_by_triangle_id()` | Index secondaire de la collection projetée. | Catalogue | Remplaçable par l'index `topoElementId`; risque faible si aucun appelant ne garde besoin de rang catalogue. |
| `src/canvas_objects_collection.py:155-181` | `get_many_by_triangle_ids()` et `triangle_ids()` | Résolution / énumération des rangs catalogue présents. | Catalogue | `get_many_by_topology_ids()` existe déjà; `triangle_ids()` doit être remplacé par une source Core de `triRank` ou par des IDs topologiques selon le besoin. |
| `src/canvas_objects_collection.py:258,282` | `dump()` et `_rebuild_indexes(): triangle_id = entry.get("id")` | Diagnostic et maintenance de l'index secondaire. | Diagnostics / catalogue | Probablement supprimable avec l'index triangle. |
| `src/assembleur_tk.py:1117` | F11 / `"triangleId": entry.get("id", "<absent>")` | Affiche l'identifiant historique dans le TopoDump de diagnostic. | Diagnostics | Remplaçable par `topoElementId`, avec `triRank` lu du Core si utile au lecteur humain. |
| `src/assembleur_tk.py:4175-4185` | `_scenarioGetFirstLastTriangles(): tid = t.get("id")` | Relie `Scenario.tri_ids` (ordre catalogue) aux entrées projetées afin de trouver premier et dernier triangles. | Algorithme métier | Remplaçable par `Scenario.orderedElementIds`: premier/dernier ElementId, puis recherche par `topoElementId`. Priorité haute. |
| `src/assembleur_tk.py:4552` | `_autoGetAnchorWorld(): int(t.get("id", -1)) == tid_ref` | Même pont `tri_ids` → projection pour calculer l'ancre AUTO. | Algorithme métier | Même remplacement que ci-dessus. Priorité haute. |
| `src/assembleur_tk.py:6688,6730` | `_on_triangle_list_select`, `_on_list_mouse_down`: `tri.get("id")` | Empêche de reprendre un triangle catalogue déjà placé. La source est un objet catalogue/DF, pas une entrée `last_drawn`. | UI / catalogue | `triRank` est précisément cette identité; pas de remplacement Core à ce stade. |
| `src/assembleur_tk.py:7277-7351` | `_place_dragged_triangle`: `tri_rank = tri.get("id")`, puis projection `{ "id": tri.get("id") }`, `_placed_ids.add(...)` | Création manuelle: `id` est lu depuis le catalogue, enregistré dans `meta.triRank`, puis dupliqué dans le cache et dans l'ensemble d'occupation UI. | Producteur / UI / catalogue | La copie dans `last_drawn` est supprimable. `_placed_ids` peut être alimenté depuis les `triRank` Core. |
| `src/assembleur_tk.py:7567-7595` | `_reinsert_triangle_to_list(): tri_id = tri.get("id")` | Réinsère un triangle supprimé dans la listbox, dans l'ordre catalogue. | UI / catalogue | Remplaçable par `element.meta["triRank"]` via `topoElementId`. Bloqué aussi par l'ancien accès à `tri.get("labels")`, devenu incohérent depuis MIG-CACHE-CLEANUP-002. |
| `src/assembleur_tk.py:9149` | `_ctx_filter_scenarios(): clicked_tid = int(self._last_drawn[idx].get("id"))` | Transforme un clic de projection en rang catalogue pour le filtre de scénarios. | Algorithme métier / UI | Remplaçable par `topoElementId`, puis adaptation de l'API de filtrage. |
| `src/assembleur_tk.py:9177-9189` | `_scenario_prefix_edge_steps(): element_id_by_tri_rank[tri_rank] = element_id` | Pont entre `Scenario.tri_ids` et `TopologyWorld` avant `build_topology_prefix_steps()`. | Algorithme métier | Remplaçable par `Scenario.orderedElementIds`, déjà conçu pour cet ordre. Priorité haute. |
| `src/assembleur_tk.py:9352` | `_prepare_core_group_operation_members()` logue `triangle.get("id", "?")` | Trace MIG-GEO informative; le flux métier travaille déjà avec `topoElementId` et `core_group_id`. | Diagnostics | Probablement supprimable ou remplaçable par `meta.triRank` Core. |
| `src/assembleur_debug.py:65,92` | `plotLastDrawn(): topoElementId or "Triangle catalogue {id}"` | Libellé de secours de l'outil matplotlib. | Diagnostics / legacy fallback | Supprimable: un cache runtime valide doit toujours avoir `topoElementId`. |
| `src/assembleur_io.py:492-510` | `saveScenarioXml(): triangle {t.get('id')} …`, XML `"id": str(t.get("id", ""))` | Écrit l'identifiant historique dans XML et l'emploie dans un message d'erreur. | Compatibilité XML | Le champ XML peut rester temporairement; la lecture métier doit déjà privilégier `topoElementId`. Suppression seulement avec évolution de format. |
| `src/assembleur_io.py:775-815` | `loadScenarioXml(): tid = int(t_el.get("id")); item={"id": tid}; fallback T{tid:02d}` | Lit les fichiers XML v4 et reconstitue `topoElementId` absent depuis l'ID catalogue. | Compatibilité XML / migration | Bloquant pour la compatibilité v4. Une future migration peut garder `tid` local au loader sans le placer dans `last_drawn`. |
| `src/assembleur_io.py:804-814` | `loaded_entries.append(item)` | Propage l'ID XML historique dans la projection runtime. | Producteur / compatibilité | Probablement supprimable après avoir remplacé les consommateurs runtime; ne remet pas en cause la lecture v4 locale. |
| `src/assembleur_io.py:828` | `_placed_ids = {int(triangle_id) for triangle_id in canvas_objects.triangle_ids()}` | Reconstitue la disponibilité catalogue à partir des projections chargées. | UI / catalogue | Remplaçable par lecture de `triRank` depuis les éléments Core. |
| `src/assembleur_io.py:480-487,697-720` | `<listbox><tri id>` et `df["id"]` | Persiste / reconstruit la liste des triangles catalogue disponibles. Ce n'est pas `last_drawn["id"]`. | Compatibilité XML / UI catalogue | À conserver tant que le catalogue est indexé par rang; hors suppression immédiate du cache. |
| `tests/test_topology_element_ids.py:232-233,461` | Assertions sur `entry["id"]` | Vérifie le contrat de projection et les rangs catalogue. | Tests | À mettre à jour avec le contrat futur; non bloquant runtime. |

## 4. Producteurs, transports et lecteurs par flux

### 4.1 Producteurs de la clé dans une projection runtime

1. `PlacedTriangle.toLegacyDict()` (`assembleur_sim.py:60`) produit l'ID des
   scénarios automatiques.
2. `_place_dragged_triangle()` (`assembleur_tk.py:7339`) produit l'ID d'un
   triangle manuel depuis le catalogue glissé.
3. `loadScenarioXml()` (`assembleur_io.py:780`) produit l'ID d'une projection
   chargée.

Ces trois producteurs sont les cibles effectives de la future suppression.

### 4.2 Lecteurs qui bloquent la suppression immédiate

```text
last_drawn["id"]
 ├─ Scenario.tri_ids -> _scenarioGetFirstLastTriangles()
 ├─ Scenario.tri_ids -> _autoGetAnchorWorld()
 ├─ Scenario.tri_ids -> _ctx_filter_scenarios()
 │                    -> _scenario_prefix_edge_steps()
 ├─ _placed_ids      -> listbox, drag & drop, réinsertion
 ├─ XML v4           -> fallback id catalogue -> Txx
 └─ CanvasObjectsCollection -> index de compatibilité et diagnostics
```

### 4.3 Lectures sans portée métier

- F11 (`_geo_orient` / TopoDump) : présentation diagnostique seulement.
- Log MIG-GEO dans `_prepare_core_group_operation_members()`.
- `plotLastDrawn()` : fallback de l'outil matplotlib.
- `CanvasObjectsCollection.dump()` : affichage debug de l'index secondaire.

Ces usages ne doivent pas dicter l'architecture. Ils peuvent passer à
`topoElementId` ou lire `triRank` dans le Core dès que les flux métier auront
été migrés.

## 5. Cas exclus et faux positifs

Les occurrences suivantes ont été relevées mais ne sont pas des lecteurs de
la clé `last_drawn["id"]` :

- `df["id"]` dans `assembleur_tk.py`, `assembleur_sim.py` et
  `assembleur_io.py` : rang de la table catalogue source ;
- `<listbox><tri id="…">` dans XML : disponibilité du catalogue ;
- `id` des noeuds XML, guides, éléments Tk, colonnes de Treeview, groupes et
  attachments : identifiants de leurs propres domaines ;
- `TopologyElement.element_id`, `TopologyAttachment.attachment_id` et les
  `group_id` Core : identifiants modernes, hors sujet.

Ils ne doivent pas être changés dans MIG-CACHE-CLEANUP consacrée au cache.

## 6. Analyse de remplaçabilité

| Besoin actuel | Source historique | Source cible recommandée | Prérequis |
|---|---|---|---|
| Retrouver un élément projeté | `id` | `topoElementId` + `CanvasObjectsCollection.get_by_topology_id()` | Aucun. |
| Premier / dernier élément AUTO | `tri_ids` + recherche `entry["id"]` | `Scenario.orderedElementIds` | Garantir son ordre après XML / activation. |
| Filtrage de préfixe AUTO | clic -> `id` -> `tri_ids` | clic -> `topoElementId`, index dans `orderedElementIds` | Migrer `_filter_auto_scenarios_by_prefix_edges()`. |
| Triangle déjà placé | `_placed_ids` | ensemble de `triRank` dérivés de `TopologyElement.meta` | Helper Core/UI de lecture du rang. |
| Réinsertion listbox | cache `id` + `labels` | `topoElementId` -> élément Core -> `triRank`, `vertex_labels` | Corriger au préalable la dépendance résiduelle à `labels`. |
| XML ancien v4 | `id` -> `Txx` | conserver un `tid` local de compatibilité dans le loader | Ne pas injecter `tid` dans la projection. |
| Diagnostics | cache `id` | `topoElementId`, éventuellement `meta.triRank` | Aucun. |

## 7. Proposition de migrations

### MIG-CACHE-CLEANUP-003 — Ponts AUTO `tri_ids` -> `orderedElementIds`

Migrer `_scenarioGetFirstLastTriangles()`, `_autoGetAnchorWorld()`,
`_ctx_filter_scenarios()` et `_scenario_prefix_edge_steps()` vers les
ElementIds ordonnés. C'est le principal retrait d'un usage métier de
`last_drawn["id"]`.

- Risque : moyen.
- Bénéfice : haut, car le moteur auto dispose déjà de `orderedElementIds`.

### MIG-UX-CATALOG-001 — Disponibilité et réinsertion catalogue

Conserver le rang catalogue dans `TopologyElement.meta["triRank"]` et dériver
`_placed_ids` / la réinsertion listbox depuis le Core. Ne pas confondre cette
évolution avec la suppression du rang dans le DataFrame : le catalogue peut
légitimement rester indexé par rang.

- Risque : moyen.
- Bénéfice : moyen.

### MIG-XML-003 — Retrait de `id` de la projection XML moderne

Le writer peut cesser de sérialiser `triangle@id` uniquement lorsqu'un format
Core-first explicite est choisi. Le loader v4 doit conserver son pont local
`id -> Txx`, sans réintroduire l'ID dans `last_drawn`.

- Risque : élevé (compatibilité des fichiers).
- Bénéfice : moyen.

### MIG-CACHE-CLEANUP-004 — Purge collection et diagnostics

Après migration des flux précédents, supprimer `_by_triangle_id`,
`get_by_triangle_id`, `triangle_ids`, le fallback matplotlib et les champs de
diagnostic basés sur `id`.

- Risque : faible.
- Bénéfice : moyen : la collection n'indexera plus que les références Core.

## 8. Conclusion

La règle cible peut être formulée ainsi :

> Une entrée `last_drawn` ne doit pas porter l'identité catalogue d'un
> triangle. Elle doit porter `topoElementId` et ses coordonnées projetées.

Cette règle n'est pas encore applicable sans migration. Les blocages ne sont
ni géométriques ni topologiques : ils concernent la chronologie AUTO, le
catalogue UI, les mots associés et les fichiers XML historiques. Le retrait du
champ de la projection est donc faisable par étapes, avec un risque global
**moyen**, et sans modifier `TopologyWorld`.
