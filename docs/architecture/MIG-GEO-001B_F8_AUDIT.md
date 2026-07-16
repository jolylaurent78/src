# MIG-GEO-001B — Audit dynamique Core ↔ UI déclenché par F8

## Objectif

MIG-GEO-001B ajoute un audit manuel du scénario actif. Il vérifie la cohérence opérationnelle entre `TopologyWorld`, les liens `topoElementId`, le cache `_last_drawn` et les groupes UI historiques `group_id`.

L'audit ne modifie pas la topologie, les poses, `_last_drawn`, les groupes UI, le XML ou les interactions. Il est volontairement déclenché par l'utilisateur ; il ne s'exécute jamais automatiquement au chargement d'un scénario.

## Raccourci et sortie

- Raccourci : **F8**.
- Binding : `TriangleViewerManual.export_mig_geo001_audit` dans `src/assembleur_tk.py`.
- F8 était disponible : les raccourcis existants F9, F10 et F11 sont préservés ; aucun binding F12 n'a été trouvé.
- Rapports : `exports/audits/MIG-GEO-001_<scenario>_<YYYYMMDD-HHMMSS>.txt`.
- Encodage : UTF-8.

Un appui F8 reconstruit d'abord l'index Step 1, lance les contrôles, écrit un rapport horodaté et affiche la synthèse dans la barre de statut. Chaque invocation produit un fichier distinct lorsque l'horodatage diffère.

## Fonctions principales

| Fonction | Rôle |
|---|---|
| `_collect_mig_geo001_audit(scen, world)` | collecte les métriques et les diagnostics sans écrire de fichier |
| `_render_mig_geo001_audit_report(audit)` | produit le texte lisible du rapport |
| `export_mig_geo001_audit(event=None)` | orchestrateur F8, écriture du fichier et retour utilisateur |
| `get_last_drawn_entry_by_topo_id` | lookup indexé réutilisé depuis Step 1 |
| `get_last_drawn_entries_by_topo_ids` | résolution de la liste Core d'éléments |
| `get_last_drawn_entries_for_core_group` | navigation groupe Core → entrées UI |
| `debug_validate_core_ui_group_linking` | diagnostic Step 1 intégré au rapport |

L'audit ne crée pas un second index : il force `_rebuild_last_drawn_topo_index()` puis réutilise les helpers MIG-GEO-001 Step 1.

## Contrôles réalisés

### Triangles UI et index

- total d'entrées `_last_drawn` ;
- présence, unicité et duplications de `topoElementId` ;
- entrées UI pointant vers un élément Core inexistant ;
- cohérence du dictionnaire d'index avec les entrées présentes ;
- vérification que le helper indexé retrouve l'entrée attendue.

Une entrée UI sans `topoElementId` produit un avertissement explicite. Si elle masque simultanément un élément Core triangulaire attendu, l'absence de projection Core est une erreur structurelle.

### Éléments Core

La règle de projetabilité retenue est celle du viewer actuel : un élément à trois sommets est un triangle attendu dans la projection UI. Les éventuels éléments Core non triangulaires ne sont donc pas automatiquement signalés comme absents de `_last_drawn`.

L'audit compte les éléments projetables, ceux retrouvés dans l'UI et ceux qui y sont absents.

### Groupes

Pour chaque groupe Core canonique, le rapport affiche :

- le groupe Core et ses `ElementId` ;
- les entrées UI résolues par `get_last_drawn_entries_for_core_group` ;
- les `group_id` historiques observés ;
- le résultat `OK`, `INCOMPLET`, `INCOHÉRENT` ou `NON COMPARABLE`.

`NON COMPARABLE` signifie qu'aucun `group_id` UI historique ne permet la comparaison ; c'est un avertissement, non une erreur automatique. Une composition Core différente de la composition UI historique est une erreur.

### Statut final

| Statut | Signification |
|---|---|
| `OK` | aucun écart détecté |
| `WARNING` | donnée UI optionnelle/manquante, groupe non comparable ou scénario vide |
| `ERROR` | ID dupliqué, ID UI absent du Core, projection Core manquante, index incohérent, groupe divergent ou exception d'audit |

## Cas limites gérés

- aucun scénario actif : message `Audit impossible : aucun scénario actif`, aucun rapport créé ;
- `TopologyWorld` temporairement absent : message d'erreur, aucun crash ;
- scénario vide : rapport `WARNING` ;
- entrée UI sans `topoElementId`, ID invalide ou groupe sans projection ;
- scénario manuel comme automatique ;
- erreur d'écriture : exception journalisée et message d'erreur dans la barre de statut ;
- appuis F8 répétés : audit explicite à chaque appui, sans binding supplémentaire.

## Exemple de résultat

```text
MIG-GEO-001 — AUDIT CORE ↔ UI

Scénario : test
Statut global : OK

==================================================
SYNTHÈSE
==================================================
Triangles _last_drawn                    : 1
Triangles avec topoElementId             : 1
Triangles sans topoElementId             : 0
Éléments Core projetables                : 1
Éléments Core absents de l’UI            : 0
Index Core ↔ UI                          : OK
```

## Fichiers modifiés

- `src/assembleur_tk.py` : orchestrateur, contrôles, génération du rapport et binding F8.
- `tests/test_mig_geo001_group_linking.py` : couverture de l'audit et de l'index Step 1.
- `docs/architecture/MIG-GEO-001B_F8_AUDIT.md` : présente documentation.

## Limites connues

L'audit mesure les identités et compositions ; il ne compare pas encore numériquement les points O/B/L UI à la projection locale+pose Core. Cette vérification correspond à l'invariant géométrique GEO-INV-003 et reste un chantier distinct. Il ne modifie pas non plus la règle actuelle selon laquelle `group_id` est encore utilisée par les gestes UI.
