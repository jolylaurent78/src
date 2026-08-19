## Objectif

Dans l’onglet **Triangles** de la fenêtre `Gestion du catalogue`, rendre fonctionnelle la zone existante :

```text
Référencé par : Hypothèses : 0   [...]
```

Pour chaque triangle sélectionné, afficher le nombre de **scénarios XML sauvegardés** dont la `ScenarioHypothesis` utilise ce triangle.

Le bouton `...` doit ouvrir une petite fenêtre listant les scénarios concernés.

La source de vérité doit rester constituée par les fichiers scénario XML eux-mêmes.

Ne pas persister ces références dans `catalogue.json`.

---

## Contexte actuel

Dans :

```text
src/assembleur_catalogue_window.py
```

l’onglet Triangles contient déjà :

```python
ttk.Label(triangle_usage, text="Référencé par :").pack(...)
ttk.Label(triangle_usage, text="Hypothèses : 0").pack(...)
ttk.Button(triangle_usage, text="...", width=3, state=tk.DISABLED).pack(...)
```

Ces widgets sont actuellement statiques.

Le modèle `Catalogue` possède :

```python
get_templates_referencing_triangle(triangle_id)
```

mais cette API concerne uniquement les **templates du catalogue**.

Elle ne doit pas être utilisée pour cette nouvelle fonctionnalité.

Ici, « Hypothèses » signifie :

```text
ScenarioHypothesis persistée dans un fichier scénario XML
```

---

## Format scénario existant

Les scénarios actuels sont au format XML v5.

La sauvegarde écrit :

```xml
<scenario version="5" ...>
    ...
    <hypothesis sourceTemplateId="...">
        <rank number="1" triangleId="TRI-...."/>
        ...
        <rank number="32" triangleId="TRI-...."/>
    </hypothesis>
</scenario>
```

`saveScenarioXml()` valide déjà la `ScenarioHypothesis` puis sérialise les 32 `triangleId`.

La lecture complète d’un scénario passe actuellement par :

```python
_load_scenario_hypothesis(...)
loadScenarioXml(...)
```

Mais le scan du Catalogue ne doit surtout pas charger un `TopologyWorld`, reconstruire la projection, changer le scénario actif ou produire un quelconque effet runtime.

Le scan doit être **read-only et léger**.

---

## Principe retenu

Ne pas mettre à jour un index lors de la sauvegarde d’un scénario.

À la place :

> le Catalogue scanne les fichiers scénario XML présents sur disque et reconstruit à la demande l’index des utilisations des triangles.

Ainsi :

* scénario ajouté → automatiquement visible ;
* scénario supprimé → automatiquement disparu ;
* scénario écrasé/modifié → référence automatiquement actualisée ;
* aucune donnée dérivée persistée ;
* aucun risque de désynchronisation avec `catalogue.json`.

---

## Architecture souhaitée

Créer un petit service de lecture dédié aux références de scénarios.

Éviter de mettre le parsing XML directement dans les callbacks Tk.

Une API du style est souhaitée :

```python
scan_scenario_triangle_references(...)
```

avec un résultat permettant au minimum :

```text
triangle_id
    -> liste de scénarios
```

Chaque référence scénario devrait contenir au minimum :

```text
nom affichable
path
```

Une petite dataclass immutable peut être utilisée si cohérent avec l’architecture existante, par exemple :

```python
@dataclass(frozen=True)
class ScenarioTriangleReference:
    name: str
    path: Path
```

Le choix exact du module est à déterminer après inspection du projet.

Préférer un module IO/service approprié plutôt qu’un couplage métier du `Catalogue` avec le filesystem.

---

## Répertoire des scénarios

IMPORTANT :

Ne pas inventer un nouveau chemin.

Identifier dans le code existant :

* où `saveScenarioXml()` est appelé ;
* quel répertoire / chemin est proposé lors de la sauvegarde ;
* comment les scénarios XML sont actuellement localisés.

Réutiliser cette convention existante.

Si l’application possède déjà un dossier canonique/configuré pour les scénarios, l’utiliser.

Ne pas hardcoder un chemin arbitraire tel que :

```text
./scenarios
```

sans l’avoir confirmé dans le code.

---

## Scan XML

Pour chaque fichier candidat :

1. parser uniquement le XML ;
2. vérifier que la racine est :

```xml
<scenario>
```

3. retenir uniquement les scénarios exploitables par cette fonctionnalité ;
4. chercher :

```xml
<hypothesis>
```

5. lire ses éléments :

```xml
<rank triangleId="...">
```

6. construire l’ensemble des `triangleId` utilisés par ce scénario.

Il n’est pas nécessaire de charger :

* `topoSnapshot`;
* `TopologyWorld`;
* map;
* clock;
* guides;
* chemins;
* projection.

Le scan doit rester très léger.

---

## Gestion des fichiers non exploitables

Le scan d’un dossier est une opération de consultation.

Un fichier XML invalide ou étranger ne doit pas empêcher l’ouverture du Catalogue.

Mais ne pas masquer aveuglément n’importe quelle erreur.

Gérer explicitement les cas attendus, par exemple selon l’implémentation retenue :

```text
XML mal formé
fichier inaccessible
racine différente de <scenario>
version scénario non supportée
hypothesis absente
rank sans triangleId
```

Un fichier invalide peut être ignoré dans l’index.

Si le projet possède un logger adapté, logguer la raison en debug/warning.

Ne pas utiliser :

```python
except Exception:
    pass
```

ni un large `except (ValueError, RuntimeError)` servant à masquer des bugs.

---

## Validation par rapport au Catalogue courant

Le scan sert à savoir quels scénarios **référencent un identifiant de triangle**.

Il ne faut pas exiger que chaque scénario soit entièrement chargeable par le viewer courant pour découvrir cette référence.

En particulier, éviter d’appeler directement :

```python
_load_scenario_hypothesis(root, catalogue)
```

si cela entraîne une validation complète contre le Catalogue courant et empêche de détecter une référence utile.

Pour ce scan, l’information normative recherchée est simplement :

```text
triangleId persisté dans <hypothesis>
```

Un `triangleId` inconnu du Catalogue courant peut être ignoré pour l’affichage des triangles courants, mais ne doit pas faire échouer tout le scan.

---

## Déduplication

Une hypothèse possède normalement 32 triangles uniques, mais le scanner ne doit pas dépendre de cet invariant pour compter les scénarios.

Pour un scénario donné :

```text
un triangle utilisé plusieurs fois
    => le scénario compte UNE seule fois
```

Le compteur représente donc :

```text
nombre de scénarios/hypothèses utilisant le triangle
```

et non :

```text
nombre de rangs XML utilisant le triangle
```

---

## Nom affiché

Dans la popup, afficher simplement un nom humain du scénario.

Préférence :

1. utiliser une information métier déjà persistée dans le XML si une telle information existe réellement et est normative ;
2. sinon utiliser le nom du fichier sans extension.

Ne pas introduire dans le XML une nouvelle notion de nom uniquement pour ce chantier.

Le chemin complet peut rester disponible dans le modèle de référence mais n’a pas besoin d’être affiché.

---

## Intégration dans CatalogueWindow

Dans :

```text
src/assembleur_catalogue_window.py
```

remplacer les deux widgets anonymes actuels par des attributs, par exemple :

```python
self._triangle_hypothesis_count_label
self._triangle_hypothesis_references_button
```

Le texte doit être :

```text
Hypothèses : N
```

Le bouton `...` :

```text
DISABLED si N == 0
NORMAL   si N > 0
```

---

## Rafraîchissement

Le compteur doit être correct lorsque :

* la fenêtre Catalogue s’ouvre ;
* on change de triangle sélectionné ;
* le catalogue est rafraîchi dans le contexte normal de la fenêtre.

Éviter de rescanner tous les XML à chaque simple clic si le scan peut être construit une seule fois à l’ouverture de `CatalogueWindow`.

Approche souhaitée :

```text
ouverture Catalogue
    -> scan du dossier scénario
    -> construction d'un index runtime
       triangle_id -> références scénario

sélection triangle
    -> simple lookup dans l'index
```

Le volume attendu est faible, donc pas besoin de mécanisme complexe de cache ou de watcher filesystem.

---

## Popup `...`

Au clic sur `...`, ouvrir une petite fenêtre `Toplevel` de consultation.

Rester volontairement simple.

Contenu :

```text
Scénarios utilisant ce triangle

Nom
-------------------------
scenario-reference
scenario-analyse
...
```

Un simple `Treeview` ou une `Listbox` suffit.

Pas de double-clic.

Pas de navigation vers le scénario.

Pas d’édition.

Pas de couleurs.

Pas de boutons supplémentaires complexes.

La croix Windows et éventuellement `Escape` suffisent pour fermer.

---

## Terminologie UX

Conserver le libellé déjà prévu :

```text
Référencé par : Hypothèses : N
```

Même si techniquement le scan porte sur des fichiers scénario, la donnée recherchée est bien leur `ScenarioHypothesis`.

Dans la popup, le titre peut être :

```text
Hypothèses utilisant le triangle
```

ou :

```text
Scénarios utilisant ce triangle
```

Préférer le second si le contenu affiché correspond aux noms de fichiers scénario.

---

## Ne pas confondre avec les Templates

Le Catalogue possède déjà :

```python
get_templates_referencing_triangle()
```

Cette relation sert notamment à empêcher la suppression d’un triangle utilisé dans un template.

Ne pas changer ce comportement.

Ne pas fusionner :

```text
Template
```

et :

```text
ScenarioHypothesis sauvegardée
```

Le compteur demandé dans l’onglet Triangle concerne uniquement les scénarios sauvegardés.

---

## Suppression d’un triangle

Ce chantier est purement informatif.

Ne pas modifier les règles de suppression d’un triangle.

En particulier, ne pas introduire automatiquement :

```text
impossible de supprimer car référencé par un scénario XML
```

sans demande explicite.

La présence d’un ancien scénario sur disque ne doit pas devenir une nouvelle contrainte métier dans ce chantier.

---

## Tests

Ajouter des tests ciblés du scanner.

### Cas 1 — scénario v5 valide

XML avec :

```text
TRI-0001
TRI-0002
...
```

Vérifier :

```text
TRI-0001 -> scénario présent
```

### Cas 2 — deux scénarios

Les deux utilisent `TRI-0001`.

Attendu :

```text
TRI-0001 -> 2 références
```

### Cas 3 — scénarios différents

```text
scenario-A -> TRI-0001
scenario-B -> TRI-0002
```

Les index doivent être séparés.

### Cas 4 — XML invalide

Un fichier mal formé dans le même répertoire ne doit pas empêcher l’indexation des autres scénarios valides.

### Cas 5 — XML qui n’est pas un scénario

Ignorer proprement.

### Cas 6 — scénario sans hypothesis

Ignorer proprement pour cette fonctionnalité.

### Cas 7 — triangleId inconnu du Catalogue courant

Ne pas faire échouer le scan global.

### Cas 8 — triangle répété dans le même XML

Le scénario doit compter une seule fois pour ce triangle.

### Cas 9 — UI sans référence

```text
Hypothèses : 0
bouton désactivé
```

### Cas 10 — UI avec références

```text
Hypothèses : N
bouton activé
popup listant exactement N scénarios
```

---

## Contraintes

Respecter les conventions du projet :

* pas de large `except Exception` pour cacher les erreurs métier ;
* pas de `except (ValueError, RuntimeError)` générique ;
* contrôles de contrat explicites ;
* exceptions attendues ciblées ;
* pas de duplication persistée dans `catalogue.json`;
* pas de modification du format scénario XML ;
* pas de chargement du `TopologyWorld` pour ce scan ;
* pas de dépendance du modèle `Catalogue` au GUI ;
* garder le service de scan indépendant de Tk autant que possible.

---

## Validation manuelle

1. disposer de plusieurs scénarios XML sauvegardés ;
2. ouvrir `Gestion du catalogue`;
3. onglet `Triangles`;
4. sélectionner un triangle utilisé dans un ou plusieurs scénarios.

Attendu :

```text
Référencé par : Hypothèses : N
```

avec `N` correct.

5. cliquer sur `...`.

Attendu :

```text
petite fenêtre
liste des scénarios utilisant le triangle
```

6. sélectionner un triangle absent de tous les scénarios.

Attendu :

```text
Hypothèses : 0
bouton ... désactivé
```

---

## Vérifications finales

Exécuter les tests ciblés puis :

```text
python -m compileall ...
git diff --check
```

Ne pas corriger spontanément des tests historiques hors périmètre.

## Résultat attendu

Le placeholder actuel :

```text
Référencé par : Hypothèses : 0 [...]
```

devient une information réellement calculée à partir des `ScenarioHypothesis` présentes dans les fichiers XML sauvegardés, sans duplication de cette relation dans le Catalogue.
