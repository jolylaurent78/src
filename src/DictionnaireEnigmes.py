from itertools import product
from enum import Enum
from typing import Iterator
import re
import unicodedata

PATTERN_MIN_TOKENS = 2
PATTERN_MAX_TOKENS = 5
LISTEPATTERNS_MAX = 8


class DicoScope(Enum):
    STRICT = "STRICT"
    MIRRORING = "MIRRORING"
    EXTENDED = "EXTENDED"


def _normalize_scope(scope) -> DicoScope:
    if isinstance(scope, DicoScope):
        return scope
    if isinstance(scope, str):
        key = scope.strip().upper()
        for s in DicoScope:
            if s.name == key or s.value == key:
                return s
    raise ValueError(f"scope invalide: {scope}")


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
    # API DicoScope / ABS-REL V1
    # ============================

    def getNbLignes(self) -> int:
        return len(self.dictionnaire)

    def normalizeRow(self, rowExt: int) -> int:
        n = int(self.getNbLignes())
        if n <= 0:
            raise ValueError("nbLignes=0")
        rowExt = int(rowExt)
        return ((rowExt - 1) % n) + 1

    def getRowSize(self, rowExt: int) -> int:
        rowReal = self.normalizeRow(rowExt)
        enigmeIndex = rowReal - 1
        _, ligne = self.dictionnaire[enigmeIndex]
        return len(ligne)

    # ============================
    # API Extended (nouvelle)
    # ============================

    def getNbEnigmes(self) -> int:
        return self.getNbLignes()

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

    def iterCoords(self, scope: DicoScope) -> Iterator[tuple[int, int]]:
        scope_norm = _normalize_scope(scope)
        n = int(self.getNbLignes())
        if n <= 0:
            return
            yield  # pragma: no cover (generator empty)

        for rowExt in range(1, n + 1):
            if scope_norm == DicoScope.EXTENDED:
                max_col = int(self.getPlageMax())
            else:
                max_col = int(self.getRowSize(rowExt))

            if scope_norm == DicoScope.STRICT:
                for colExt in range(1, max_col + 1):
                    yield (rowExt, colExt)
                continue

            # MIRRORING / EXTENDED: nÃ©gatifs puis positifs
            for colExt in range(-max_col, 0):
                yield (rowExt, colExt)
            for colExt in range(1, max_col + 1):
                yield (rowExt, colExt)

    def computeDeltaAbs(self, originAbs: tuple[int, int], targetAbs: tuple[int, int]) -> tuple[int, int]:
        rowO, colO = originAbs
        rowT, colT = targetAbs

        # On doit linéariser les colonnes pour retirer le "trou" au 0
        def _colExtToLin(colExt: int) -> int:
            return colExt - 1 if colExt > 0 else colExt

        deltaCol = _colExtToLin(colT) - _colExtToLin(colO)

        # On reste standard pour les colonnes
        n = int(self.getNbLignes())
        deltaRow = (rowT - rowO) % n

        return (deltaRow, deltaCol)

    def applyDeltaAbs(self, originAbs: tuple[int, int], deltaRel: tuple[int, int]) -> tuple[int, int]:
        rowO, colO = originAbs
        deltaRow, deltaCol = deltaRel
        rowO = int(rowO)
        colO = int(colO)
        deltaRow = int(deltaRow)
        deltaCol = int(deltaCol)

        n = int(self.getNbLignes())
        if n <= 0:
            raise ValueError("nbLignes=0")
        if not (1 <= rowO <= n):
            raise IndexError(f"rowO hors limites: {rowO}")
        if colO == 0:
            raise ValueError("colO=0 interdit en ABS")

        if deltaRow < 0 or deltaRow >= n:
            raise ValueError(f"deltaRow hors bornes: {deltaRow} (0..{n-1})")

        rowT = self.normalizeRow(rowO + deltaRow)
        colT = colO + deltaCol

        if colT == 0:
            raise ValueError("colT=0 interdit en ABS")

        return (int(rowT), int(colT))

    def recalageAbs(self, coordAbs: tuple[int, int]) -> tuple[int, int]:
        rowExt, colExt = coordAbs
        if colExt == 0:
            raise ValueError("colExt=0 interdit en ABS")

        rowReal = self.normalizeRow(rowExt)
        nrow = self.getRowSize(rowReal)
        if nrow <= 0:
            raise ValueError("Ligne vide")

        # mapping Extended -> index interne 0-based (mÃªme convention que getMotExtended)
        motIndex0 = (colExt - 1) if colExt > 0 else colExt
        motIndex0Real = motIndex0 % nrow
        colReal = motIndex0Real + 1
        return (rowReal, colReal)

    def getExtendedColumns(self, plageMax=None) -> list[int]:
        p = self.getPlageMax() if plageMax is None else int(plageMax)
        cols = list(range(-p, 0)) + list(range(1, p+1))
        return cols

    def getRowLabelsAbs(self) -> list[int]:
        """
        Labels de lignes en ABS (Extended row) : 1..nbLignes.
        r0 est l'origine en index de ligne TkSheet (0-based).
        """
        n = int(self.getNbLignes())
        if n <= 0:
            return []
        return [i + 1 for i in range(n)]

    # UI-only (refMode)
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

    # UI-only (refMode)
    def getRowLabelsRel(self, r0: int, *, refMode: str = "origin") -> list[int]:
        """
        Labels REL (delta) pour toutes les énigmes du dico.
        """
        if refMode not in ("origin", "target"):
            raise ValueError(f"refMode invalide: {refMode}")
        sign = -1 if refMode == "target" else 1
        n = int(self.getNbLignes())
        return [sign * ((i - r0) % n) for i in range(n)]

    # UI-only (refMode)
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


class Pattern:
    def __init__(self, dico: DictionnaireEnigmes):
        self._dico = dico
        self._tokens = []
        self._token_strings = []
        self._syntax = ""
        self._ready = False
        self._allWordsSet = None
        self._categoryWordsSets = None
        self._categoryMapNormToRaw = self._buildCategoryMap()

    def _buildCategoryMap(self) -> dict:
        categories = self._dico.getCategories()
        mapping = {}
        for cat in categories:
            mapping[_normalizeWordLocal(cat)] = cat
        return mapping

    def _collectAllWords(self) -> set:
        words = set()
        for _, ligne in self._dico.dictionnaire:
            for mot in ligne:
                words.add(mot)
        return words

    def isReady(self) -> bool:
        return bool(self._ready)

    def getTokenCount(self) -> int:
        return len(self._tokens)

    def getSyntax(self) -> str:
        return self._syntax

    # Dans le pattern, trouve t on un joker à la position index? Utilisé dans l'algo ABS pour skipper la recherche
    def isJokerAt(self, index: int) -> bool:
        if index < 0 or index >= len(self._tokens):
            return False
        kind, _ = self._tokens[index]
        return kind == "joker"

    def setSyntax(self, syntax: str, *, allow_short: bool = False) -> tuple[bool, str]:
        if syntax is None:
            return False, "Pattern vide"

        raw_tokens = syntax.split()
        n = len(raw_tokens)
        min_tokens = 1 if allow_short else PATTERN_MIN_TOKENS

        if n == 0:
            return False, "Pattern vide"
        if n < min_tokens:
            return False, f"Pattern trop court (min = {PATTERN_MIN_TOKENS})"
        if n > PATTERN_MAX_TOKENS:
            return False, f"Pattern trop long (max = {PATTERN_MAX_TOKENS})"

        token_strings = []
        tokens = []
        has_discriminant = False
        all_words = None

        for token in raw_tokens:
            if token == "*":
                return False, "Token '*' interdit, utiliser '[*]'"

            if token == "[*]":
                token_strings.append("[*]")
                tokens.append(("joker", None))
                continue

            if "[" in token or "]" in token:
                if re.match(r"^\[[^\[\]\s]+\]$", token):
                    inner = token[1:-1]
                    if inner == "*":
                        return False, "Token '*' interdit, utiliser '[*]'"
                    cat_norm = _normalizeWordLocal(inner)
                    if cat_norm not in self._categoryMapNormToRaw:
                        return False, f"Categorie inconnue: {cat_norm}"
                    token_strings.append(f"[{cat_norm}]")
                    tokens.append(("cat", cat_norm))
                    has_discriminant = True
                    continue
                return False, f"Token invalide: '{token}'"

            mot_norm = _normalizeWordLocal(token)
            if all_words is None:
                all_words = self._collectAllWords()
            if mot_norm not in all_words:
                return False, f"Mot inconnu du dictionnaire: {mot_norm}"
            token_strings.append(mot_norm)
            tokens.append(("word", mot_norm))
            has_discriminant = True

        if not has_discriminant:
            return False, "Pattern non discriminant (uniquement des jokers)"

        self._tokens = tokens
        self._token_strings = token_strings
        self._syntax = " ".join(token_strings)
        self._allWordsSet = None
        self._categoryWordsSets = None
        self._ready = False
        return True, ""

    def loadCache(self) -> None:
        if not self._tokens:
            raise RuntimeError("Pattern non initialise")

        self._allWordsSet = self._collectAllWords()
        self._categoryWordsSets = {}

        for kind, value in self._tokens:
            if kind != "cat":
                continue
            cat_norm = value
            if cat_norm in self._categoryWordsSets:
                continue

            raw_cat = self._categoryMapNormToRaw.get(cat_norm)
            if raw_cat is None:
                raise ValueError(f"Categorie absente: {cat_norm}")

            cat_list = self._dico.getListeCategorie(raw_cat)
            if not cat_list:
                raise ValueError(f"Categorie vide: {cat_norm}")

            word_set = {mot for (mot, _, _) in cat_list}
            if not word_set:
                raise ValueError(f"Categorie vide: {cat_norm}")

            self._categoryWordsSets[cat_norm] = word_set

        self._ready = True

    def validateSequence(self, sequenceWords: list[str], mode: str = "safe") -> str:
        if not self.isReady():
            raise RuntimeError("Pattern non pret")

        if mode not in ("safe", "last"):
            raise ValueError(f"mode invalide: {mode}")

        k = len(sequenceWords)
        n = len(self._tokens)
        if k < 1 or k > n:
            raise ValueError(f"Longueur sequence invalide: {k} (attendu 1..{n})")

        if mode == "safe":
            indices = range(k)
        else:
            indices = [k - 1]

        for i in indices:
            kind, value = self._tokens[i]
            w = sequenceWords[i]
            if kind == "joker":
                continue
            if kind == "cat":
                if w not in self._categoryWordsSets.get(value, set()):
                    return "NO"
                continue
            if w != value:
                return "NO"

        if k < n:
            return "MAYBE"
        return "YES"


class PatternStateSet:
    def __init__(self, patternCount: int, packed: int = 0):
        if patternCount < 0 or patternCount > LISTEPATTERNS_MAX:
            raise ValueError(f"patternCount invalide: {patternCount}")

        self._patternCount = int(patternCount)

        # Masque des bits utiles (2 bits par pattern)
        bitCount = 2 * self._patternCount
        mask = (1 << bitCount) - 1 if bitCount > 0 else 0

        # 1) Interdire les bits hors plage (BUG)
        if packed & ~mask:
            raise RuntimeError(f"packed hors plage: {packed}")

        self._packed = int(packed) & mask

        # 2) Interdire l'état réservé 0b11 (BUG)
        for i in range(self._patternCount):
            v = (self._packed >> (2 * i)) & 0b11
            if v == 0b11:
                raise RuntimeError(f"Etat reserve 0b11 detecte pour pattern {i}")

    def getPacked(self) -> int:
        return int(self._packed)

    def getPatternCount(self) -> int:
        return int(self._patternCount)

    def _checkIndex(self, index: int) -> None:
        if index < 0 or index >= self._patternCount:
            raise IndexError(f"index hors bornes: {index}")

    def getState(self, index: int) -> str:
        self._checkIndex(index)
        shift = 2 * index
        val = (self._packed >> shift) & 0b11
        if val == 0b00:
            return "NO"
        if val == 0b01:
            return "MAYBE"
        if val == 0b10:
            return "YES"
        raise RuntimeError(f"Etat invalide (11) pour index {index}")

    def setState(self, index: int, state: str) -> None:
        self._checkIndex(index)
        if state == "NO":
            val = 0b00
        elif state == "MAYBE":
            val = 0b01
        elif state == "YES":
            val = 0b10
        else:
            raise ValueError(f"state invalide: {state}")

        shift = 2 * index
        mask = 0b11 << shift
        self._packed = (self._packed & ~mask) | (val << shift)

    def hasMaybe(self) -> bool:
        for i in range(self._patternCount):
            shift = 2 * i
            val = (self._packed >> shift) & 0b11
            if val == 0b01:
                return True
            if val == 0b11:
                raise RuntimeError(f"Etat invalide (11) pour index {i}")
        return False

    def getYesIndexes(self) -> list[int]:
        res = []
        for i in range(self._patternCount):
            shift = 2 * i
            val = (self._packed >> shift) & 0b11
            if val == 0b10:
                res.append(i)
            elif val == 0b11:
                raise RuntimeError(f"Etat invalide (11) pour index {i}")
        return res


class ListePatterns:
    def __init__(self, dico: DictionnaireEnigmes, patternsSyntax: list[str]):
        n = len(patternsSyntax)
        if n < 1 or n > LISTEPATTERNS_MAX:
            raise ValueError(f"Nombre de patterns invalide: {n}")

        self._patterns = []
        for syntax in patternsSyntax:
            p = Pattern(dico)
            allow_short = len(str(syntax).split()) == 1
            ok, msg = p.setSyntax(syntax, allow_short=allow_short)
            if not ok:
                raise ValueError(msg)
            p.loadCache()
            self._patterns.append(p)

    def getPatternCount(self) -> int:
        return len(self._patterns)

    def maxLen(self) -> int:
        return max((p.getTokenCount() for p in self._patterns), default=0)
    
    def getPackedInitialState(self) -> int:
        ps = PatternStateSet(len(self._patterns))
        for i in range(len(self._patterns)):
            ps.setState(i, "MAYBE")
        return ps.getPacked()

    def hasJokerAt(self, depth: int, packedState: int) -> bool:
        ps = PatternStateSet(len(self._patterns), packedState)
        for i, p in enumerate(self._patterns):
            if ps.getState(i) == "NO":
                continue
            if p.isJokerAt(depth):
                return True
        return False

    def splitPackedByJokerAt(self, depth: int, packedState: int) -> tuple[int, int]:
        ps_in = PatternStateSet(len(self._patterns), packedState)
        ps_j = PatternStateSet(len(self._patterns))
        ps_n = PatternStateSet(len(self._patterns))

        for i, p in enumerate(self._patterns):
            st = ps_in.getState(i)
            if st == "NO":
                ps_j.setState(i, "NO")
                ps_n.setState(i, "NO")
                continue

            if p.isJokerAt(depth):
                ps_j.setState(i, st)
                ps_n.setState(i, "NO")
            else:
                ps_j.setState(i, "NO")
                ps_n.setState(i, st)

        return ps_j.getPacked(), ps_n.getPacked()

    def validateSequence(
        self,
        sequenceWords: list[str],
        mode: str,
        packedState: int | None = None,
    ) -> tuple[bool, list[int], int]:
        if len(sequenceWords) < 1:
            raise ValueError("sequenceWords vide")
        if mode not in ("safe", "last"):
            raise ValueError(f"mode invalide: {mode}")

        for p in self._patterns:
            if not p.isReady():
                raise RuntimeError("Pattern non pret")

        if packedState is None:
            ps = PatternStateSet(len(self._patterns))
        else:
            ps = PatternStateSet(len(self._patterns), packedState)

        for i, p in enumerate(self._patterns):
            if packedState is not None and ps.getState(i) == "NO":
                continue

            if len(sequenceWords) > p.getTokenCount():
                ri = "NO"
            else:
                ri = p.validateSequence(sequenceWords, mode)
            ps.setState(i, ri)

        continueExplore = ps.hasMaybe()
        yesIndexes = ps.getYesIndexes()
        packedState = ps.getPacked()

        return continueExplore, yesIndexes, packedState


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
