# MIG-GEO-001C — Infrastructure de logging pour les migrations géométriques

## Objectif

Cette étape introduit une journalisation fichier standard, légère et sans dépendance externe. Elle sert à observer les audits MIG-GEO et à préparer les migrations fonctionnelles, en commençant par le déplacement de groupe. Elle ne modifie ni géométrie, ni pose, ni groupes, ni XML, ni logique de sélection.

## Architecture retenue

Le module [logging_utils.py](../../src/utils/logging_utils.py) expose :

```python
get_app_logger()
get_mig_geo_logger()
```

Les fonctions créent le répertoire `logs/` à la racine du projet et retournent des loggers Python `logging` prêts à l'emploi. Leur appel est idempotent : un handler déjà associé au même fichier est réutilisé, sans ajout de handler dupliqué.

| Logger | Fichier | Usage |
|---|---|---|
| `APP` | `logs/app.log` | exceptions et informations applicatives générales |
| `MIG-GEO` | `logs/mig_geo.log` | audits, migrations géométriques/topologiques et validations Core ↔ UI |

Les handlers sont créés lors de l'import du module UI principal, donc au démarrage normal de l'application. Le fichier est ouvert par `RotatingFileHandler` dès cette initialisation.

## Rotation et format

- Implémentation : `logging.handlers.RotatingFileHandler` de la bibliothèque standard.
- Taille maximale : **10 Mio** par fichier.
- Archives conservées : **5** (`.1` à `.5`).
- Encodage : UTF-8.

Exemple :

```text
2026-07-16 10:42:15 [INFO] [MIG-GEO] [MIG-GEO] Audit lancé Scenario=Scenario_FrontiereActuelle v5 Type=manual
2026-07-16 10:42:18 [DEBUG] [MIG-GEO] [MOVE] Triangle sélectionné=T17
2026-07-16 10:42:18 [DEBUG] [MIG-GEO] [MOVE] topoElementId=T17
```

Le nom du logger est fourni par le format ; les préfixes d'événement restent dans le message pour permettre une lecture et une recherche rapides : `[MIG-GEO]`, `[MOVE]`, puis à l'avenir `[ROTATE]`, `[FLIP]`, `[COLLAGE]` ou `[UNGROUP]`.

## Intégration MIG-GEO-001B / F8

`TriangleViewerManual.export_mig_geo001_audit()` écrit dans `MIG-GEO` :

```text
[MIG-GEO] Audit lancé Scenario=<nom> Type=<type>
[MIG-GEO] Audit terminé Status=<OK|WARNING|ERROR> Triangles=<n> Groupes=<n> Errors=<n> Warnings=<n> Rapport=<chemin>
```

Une exception d'audit est écrite dans `mig_geo.log` et dans `app.log`, avec traceback. Le rapport TXT F8 reste inchangé et est toujours créé dans `exports/audits/`.

## Préparation MIG-GEO-002 / MOVE

Le point de départ effectif d'un déplacement de groupe est la création de `self._sel` avec `mode="move_group"` dans `_on_canvas_left_down()`. Les quatre variantes existantes sont instrumentées : clic centre, sommet lié avec CTRL, sommet avec CTRL et sommet sans CTRL.

Depuis MIG-GEO-002, `_prepare_mig_geo_move_members()` est appelée une fois à
ce démarrage. Elle journalise :

- triangle sélectionné ;
- `topoElementId` ;
- `LegacyGroupId` (le `group_id` UI existant) ;
- `LegacyMembers` (IDs des membres UI) ;
- `CoreGroupId` et `CoreMembers` ;
- le résultat de comparaison `LegacyMembers == CoreMembers`.

Elle n'est pas appelée dans `_on_canvas_left_move()` ni dans `_move_group_world()`.
Ainsi, aucun volume de log par pixel de déplacement n'est introduit.

## Fichiers modifiés

- `src/utils/__init__.py` : package d'utilitaires.
- `src/utils/logging_utils.py` : création idempotente des loggers et rotation.
- `src/assembleur_tk.py` : initialisation des loggers, audit F8 et instrumentation MOVE.
- `tests/test_logging_utils.py` : création, absence de handlers dupliqués et rotation.
- `tests/test_mig_geo001_group_linking.py` : reste la couverture de l'audit F8.
- `docs/architecture/MIG-GEO-001C_LOGGING.md` : présente documentation.

## Vérification

Les tests ciblés couvrent :

- création de `app.log` et `mig_geo.log` ;
- absence de handler dupliqué après appels répétés ;
- rotation vers `mig_geo.log.1` avec une taille de test réduite ;
- audit F8 et cas Core/UI existants.

Commande exécutée :

```text
python -m pytest tests/test_logging_utils.py tests/test_mig_geo001_group_linking.py -q
```

Résultat : `8 passed`.

## Limites

Cette infrastructure ne remplace aucun `print()` historique hors des flux MIG-GEO concernés et ne transforme pas encore les opérations rotate/flip/collage/dégroupement. Ces étapes devront uniquement adopter `get_mig_geo_logger()` lorsqu'elles seront traitées par leurs migrations dédiées.
