from itertools import product

import unicodedata

def _normalizeWordLocal(w: str) -> str:
    s = w.strip()
    # Décomposition Unicode
    s = unicodedata.normalize("NFD", s)
    # Suppression des diacritiques (accents)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    # Majuscules
    s = s.upper()
    return s

class LigneCirculaire:
    def __init__(self, ligne, masque=None):
        self.ligne = ligne
        self.masque = masque or [True] * len(ligne)  # True = mot visible

    def __getitem__(self, index):
        n = len(self.ligne)
        if n == 0:
            raise ValueError("Ligne vide")

        index = index % n

        # S'il est masqué, on retourne None
        if not self.masque[index]:
            return ""

        return self.ligne[index]


class DictionnaireEnigmes:
    def __init__(self, cheminFichier, tagExclure: str = None, bonSens: bool = True):
        self.filtrageGlobal = False
        self.chargerFichier(cheminFichier, tagExclure, bonSens)

    def getFiltrageGlobal(self):
        return self.filtrageGlobal

    def setFiltrageGlobal(self, filtrage=True):
        self.filtrageGlobal = filtrage

    def chargerFichier(self, cheminFichier, tagExclure: str = None, bonSens: bool = True):
        self.dictionnaire = []
        self.indexCategories = {
            "action": [False, []],
            "localise": [False, []],
            "pointFixe": [False, []],
            "direction": [False, []],
            "origine": [False, []],
            "ordre": [False, []],
            "nombre": [False, []],
            "pointCardinal": [False, []],
            "spatial": [False, []],
            "mesure": [False, []],
            "utilisable": [False, []]
        }

        with open(cheminFichier, "r", encoding="utf-8") as fichier:
            lignes = fichier.readlines()
            # On inverse les lignes si dans le contre sens
            if not bonSens:
                lignes = list(reversed(lignes))

            for numeroEnigme, ligne in enumerate(lignes):
                mots = ligne.strip().split()
                if not mots:
                    continue
                titre = mots[0]
                ligneSansTags = []

                i = 0  # Compteur pour les index des mots réellement ajoutés
                motSource = mots[1:]
                # On inverse les mots si dans le contre sens
                if not bonSens:
                    motSource = reversed(motSource)

                for mot in motSource:
                    # On détecte les mots avec des [] ex: cherche[localise] cherche = Mot / localise= Tag
                    if "[" in mot and mot.endswith("]"):
                        motBrut, tag = mot.split("[")
                        tag = tag.rstrip("]")

                        motNorm = _normalizeWordLocal(motBrut)

                        # On ne rajoute pas les mots dont le tag est "exclure"
                        if tag != tagExclure:
                            ligneSansTags.append(motNorm)
                            if tag in self.indexCategories:
                                self.indexCategories[tag][1].append((motNorm, numeroEnigme, i))
                            i += 1
                    else:
                        motNorm = _normalizeWordLocal(mot)
                        ligneSansTags.append(motNorm)
                        i += 1

                self.dictionnaire.append((titre, ligneSansTags))

    def getCategories(self):
        return list(self.indexCategories.keys())

    def getListeCategorie(self, categorie):
        return self.indexCategories[categorie][1]

    def getIndexCategories(self):
        return self.indexCategories

    def getCategoriesSelectionnees(self):
        return [categorie for categorie, (actif, _) in self.indexCategories.items() if actif]

    def activerCategorie(self, nomCategorie, actif=True):
        if nomCategorie in self.indexCategories:
            self.indexCategories[nomCategorie][0] = actif

    def getMot(self, enigmeIndex, motIndex):
        if enigmeIndex >= len(self.dictionnaire):
            raise IndexError("Indice d'énigme hors limites")
        titre, ligne = self.dictionnaire[enigmeIndex]
        if not ligne:
            raise ValueError("Ligne vide")
        motIndex = motIndex % len(ligne)
        return ligne[motIndex]

    def __len__(self):
        return len(self.dictionnaire)

    def __getitem__(self, enigmeIndex):
        titre, ligne = self.dictionnaire[enigmeIndex]

        # Si le filtrage n'est pas actif, on renvoie la ligne complète
        if not self.filtrageGlobal:
            return LigneCirculaire(ligne)

        # Création d’un masque de visibilité
        masque = [False] * len(ligne)
        categories_actives = self.getCategoriesSelectionnees()
        for categorie in categories_actives:
            for (mot, idxEnigme, idxMot) in self.indexCategories[categorie][1]:
                if idxEnigme == enigmeIndex:
                    masque[idxMot] = True

        return LigneCirculaire(ligne, masque)

    def getTitres(self):
        return [ligne[0] if ligne else "" for ligne in self.dictionnaire]

    def getTitre(self, enigmeIndex):
        return self.dictionnaire[enigmeIndex][0]

    def nbMotMax(self):
        return max((len(ligne) for _, ligne in self.dictionnaire), default=0)

    def nbMotsLigne(self, ligneIdx):
        _, ligne = self.dictionnaire[ligneIdx]
        return len(ligne)

    # ============================
    # API Extended (nouvelle)
    # ============================

    def getNbEnigmes(self) -> int:
        return len(self.dictionnaire)

    def getPlageMax(self) -> int:
        return int(self.nbMotMax())

    def getMotExtended(self, rowExt: int, colExt: int) -> str:
        rowExt = int(rowExt)
        colExt = int(colExt)

        if colExt == 0:
            raise ValueError("colExt=0 interdit en référentiel Extended")

        plage = int(self.getPlageMax())
        if abs(colExt) > plage:
            raise IndexError(f"colExt hors plage: {colExt} (plageMax={plage})")

        enigmeIndex = rowExt - 1
        if enigmeIndex < 0 or enigmeIndex >= self.getNbEnigmes():
            raise IndexError(f"rowExt hors limites: {rowExt}")

        # mapping Extended -> index interne 0-based
        motIndex0 = (colExt - 1) if colExt > 0 else colExt  # -1 => dernier mot (via modulo)

        return self.getMot(enigmeIndex, motIndex0)

    def getExtendedColumns(self, plageMax=None) -> list[int]:
        p = self.getPlageMax() if plageMax is None else int(plageMax)
        cols = list(range(-p, 0)) + list(range(1, p+1))
        return cols

    def getRowLabelsAbs(self) -> list[int]:
        """
        Labels de lignes en ABS (Extended row) : 1..10, wrap modulo 10.
        r0 est l'origine en index de ligne TkSheet (0-based).
        """
        return [((i % 10) + 1) for i in range(self.getNbEnigmes())]

    def getRelativeColumns(self, colRefExt: int, plageMax=None, *, refMode: str = "origin") -> list[int]:
        """
        Retourne les labels de colonnes en référentiel RELATIF, dans l'ordre Extended:
            [-plageMax..-1] U [1..plageMax]

        - colRefExt : référence en Extended (≠0)
        - refMode   : "origin" (delta = col - ref) ou "target" (delta = -(col - ref))
        - delta=0 est autorisé (label relatif)
        """
        colRefExt = int(colRefExt)
        if colRefExt == 0:
            raise ValueError("colRefExt=0 interdit en référentiel Extended")

        if refMode not in ("origin", "target"):
            raise ValueError(f"refMode invalide: {refMode}")

        colsExt = self.getExtendedColumns(plageMax=plageMax)
        sign = -1 if refMode == "target" else 1
        return [sign * (int(c) - colRefExt) for c in colsExt]

    def getRowLabelsRel(self, r0: int, *, refMode: str = "origin") -> list[int]:
        """
        Labels REL (delta) pour toutes les énigmes du dico.
        """
        if refMode not in ("origin", "target"):
            raise ValueError(f"refMode invalide: {refMode}")
        sign = -1 if refMode == "target" else 1
        n = self.getNbEnigmes()
        return [sign * ((i - r0) % 10) for i in range(n)]

    def applyRelativeToExtended(self, colRefExt: int, deltaCol: int, *, refMode: str = "origin") -> int:
        """
        Convertit un déplacement RELATIF (deltaCol) appliqué à une référence Extended (colRefExt)
        en une colonne Extended.

        - refMode="origin": ext = ref + delta
        - refMode="target": ext = ref - delta  (car les deltas affichés sont inversés)

        Règle "pas de 0" (normative):
        - si le résultat vaut 0, on force à +1 si delta>=0, sinon -1.
        """
        colRefExt = int(colRefExt)
        deltaCol = int(deltaCol)
        if colRefExt == 0:
            raise ValueError("colRefExt=0 interdit en référentiel Extended")

        if refMode not in ("origin", "target"):
            raise ValueError(f"refMode invalide: {refMode}")

        ext = (colRefExt - deltaCol) if refMode == "target" else (colRefExt + deltaCol)

        if ext == 0:
            ext = 1 if deltaCol >= 0 else -1

        plage = int(self.getPlageMax())
        if abs(ext) > plage:
            raise IndexError(f"Résultat hors plage: {ext} (plageMax={plage})")

        return int(ext)


class SequenceCategorie:
    def __init__(self, dictionnaire):
        super().__init__()
        self.listeCategories = []           # liste des catégories de la séquence
        self.selections = []                # liste des mots sélectionnés pour chaque catégorie
        self.dictionnaire = dictionnaire    # instance de DictionnaireEnigmes
        self.modeIndexRelatif = False       # Par défaut, on travaille en Index absolu

    def setModeIndexRelatif(self, modeIndex):
        self.modeIndexRelatif = modeIndex

    def ajouterCategorie(self, categorie):
        if categorie in self.dictionnaire.getCategories():
            self.listeCategories.append(categorie)
            self.selections.append([])              # aucune sélection au départ
            return len(self.listeCategories)-1
        return None

    def retirerCategorie(self, index):
        if 0 <= index < len(self.listeCategories):
            del self.listeCategories[index]
            del self.selections[index]

    def mettreAJourCategorie(self, index, nouvelleCategorie):
        if nouvelleCategorie not in self.dictionnaire.getCategories():
            return False  # Catégorie inconnue

        if 0 <= index < len(self.listeCategories):
            self.listeCategories[index] = nouvelleCategorie
            self.selections[index] = []  # Réinitialiser les sélections
            return True
        elif index == len(self.listeCategories):
            self.ajouterCategorie(nouvelleCategorie)
            return True
        return False

    def ajouterMotSelectionne(self, indexCategorie, indexEnigme, indexMot):
        if 0 <= indexCategorie < len(self.selections):
            if (indexEnigme, indexMot) not in self.selections[indexCategorie]:
                self.selections[indexCategorie].append((indexEnigme, indexMot))

    def retirerMotSelectionne(self, indexCategorie, indexEnigme, indexMot):
        if 0 <= indexCategorie < len(self.selections):
            if (indexEnigme, indexMot) in self.selections[indexCategorie]:
                self.selections[indexCategorie].remove((indexEnigme, indexMot))

    def getMotsSelectionnes(self, indexCategorie):
        if 0 <= indexCategorie < len(self.selections):
            return self.selections[indexCategorie]
        return []

    def getCategories(self):
        return self.listeCategories

    def getCategorie(self, indexCategorie):
        if 0 <= indexCategorie < len(self.listeCategories):
            return self.listeCategories[indexCategorie]
        return None

    def afficheSequence(self):
        listeStr = ""
        for c in self.listeCategories:
            listeStr += f"{c} : "
        print(listeStr)

    def getComplexiteSequence(self):
        i = 0
        complexite = 1
        for index in range(len(self.listeCategories)):
            if len(self.selections[index]) > 0:
                i += 1
                complexite *= len(self.selections[index])
            else:
                break
        if i == 0:
            return 0, 0
        else:
            return i, complexite

    def listeToutesSequencesPossibles(self):
        if not self.selections or any(len(sel) == 0 for sel in self.selections):
            return []

        # Produit cartésien de toutes les sélections
        toutesCombinaisons = product(*self.selections)
        resultats = []

        for combinaison in toutesCombinaisons:
            if self.modeIndexRelatif:
                # On part de (0,0) et on calcule les deltas
                chemin = []
                prev = combinaison[0]
                chemin = [f"({prev[0]}, {prev[1]})"]
                for curr in combinaison[1:]:
                    dx = curr[0] - prev[0]
                    dy = curr[1] - prev[1]
                    chemin.append(f"(+{dx}, +{dy})" if dx >= 0 and dy >= 0 else f"({dx}, {dy})")
                    prev = curr
            else:
                # Coordonnées absolues
                chemin = [f"({e},{m})" for (e, m) in combinaison]

            resultats.append(" → ".join(chemin))

        return resultats

    def conversionIndexMot(self, indexCategorie, indexMot):
        """
        Retourne un couple (enigme, mot), soit :
        - en index absolu : (enigme, mot)
        - en index relatif : (delta_enigme, delta_mot) par rapport à la sélection précédente
        """
        if not (0 <= indexCategorie < len(self.selections)):
            return None  # Catégorie invalide

        mots = self.selections[indexCategorie]
        if not mots or not (0 <= indexMot < len(mots)):
            return None  # Mot invalide ou non sélectionné

        enigme, mot = mots[indexMot]

        if not self.modeIndexRelatif or indexCategorie == 0:
            return enigme, mot

        # Calcul du delta par rapport au précédent mot sélectionné
        motsPrecedents = self.selections[indexCategorie - 1]
        if not motsPrecedents:
            return enigme, mot  # Pas de point de référence précédent

        enigmePrev, motPrev = motsPrecedents[0]  # pour le moment, seul le premier mot est pris en compte
        deltaEnigme = enigme - enigmePrev
        deltaMot = mot - motPrev
        return deltaEnigme, deltaMot


if __name__ == "__main__":
    dico = DictionnaireEnigmes("data/livre.txt")
    print(dico[5][10])
    for categorie in dico.getCategories():
        print(categorie, ":", dico.getIndexCategories()[categorie])
