"""assembleur_decryptor_engine.py
Moteur de decryptage ABS (V1) sans dependance UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.DictionnaireEnigmes import DicoScope, DictionnaireEnigmes, ListePatterns
from src.assembleur_decryptor import DecryptorConfig

if TYPE_CHECKING:
    from src.assembleur_core import TopologyCheminTriplet, TopologyChemins


@dataclass(frozen=True)
class DecryptageCandidate:
    """Sequence partielle validee (MAYBE ou YES pour au moins un pattern)."""

    words: list[str]
    coordsAbs: list[tuple[int, int]]
    patternPacked: int
    scoreMax: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "words", list(self.words))
        object.__setattr__(self, "coordsAbs", [(int(r), int(c)) for (r, c) in self.coordsAbs])
        object.__setattr__(self, "patternPacked", int(self.patternPacked))
        score = float(self.scoreMax)
        if score < 0.0:
            score = 0.0
        object.__setattr__(self, "scoreMax", score)


@dataclass(frozen=True)
class SolutionDecryptage:
    """Solution autonome et figee."""

    patternIndex: int
    words: list[str]
    coordsAbs: list[tuple[int, int]]
    scoreMax: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "patternIndex", int(self.patternIndex))
        object.__setattr__(self, "words", list(self.words))
        object.__setattr__(self, "coordsAbs", [(int(r), int(c)) for (r, c) in self.coordsAbs])
        score = float(self.scoreMax)
        if score < 0.0:
            score = 0.0
        object.__setattr__(self, "scoreMax", score)


class DecryptorEngine:
    """Moteur ABS V1 (mono-thread, sans UX)."""

    def __init__(self, chemins: TopologyChemins, dico: DictionnaireEnigmes) -> None:
        self.chemins = chemins
        self.dico = dico
        self.triplets = chemins.triplets

    def _build_hits(
        self,
        triplet: TopologyCheminTriplet,
        scope: DicoScope,
        decryptorConfig: DecryptorConfig,
    ) -> list[tuple[int, int, str, float]]:
        hits: list[tuple[int, int, str, float]] = []
        for (rowExt, colExt) in self.dico.iterCoords(scope):
            word = self.dico.getMotExtended(rowExt, colExt)
            clock = decryptorConfig.decryptor.clockStateFromDicoCell(
                row=rowExt,
                col=colExt,
                word=word,
                mode="abs",
            )
            ok, hitScore = decryptorConfig.match(triplet, clock)
            if not ok:
                continue
            hits.append((rowExt, colExt, word, float(hitScore)))
        return hits

    def runAbs(
        self,
        scope: DicoScope,
        listePatterns: ListePatterns,
        decryptorConfig: DecryptorConfig,
        patternMode: str = "last",
    ) -> list[SolutionDecryptage]:
        if len(self.triplets) < 1:
            raise ValueError("triplets vides")
        if patternMode not in ("safe", "last"):
            raise ValueError(f"patternMode invalide: {patternMode}")

        pattern_count = int(listePatterns.getPatternCount())
        if pattern_count < 1 or pattern_count > 8:
            raise ValueError(f"patternCount invalide: {pattern_count}")

        results: list[SolutionDecryptage] = []
        frontier: list[DecryptageCandidate] = []

        JOKER_WORD_ABS = "*"
        JOKER_COORD_ABS = (0, 0)

        def emitSolutions(
            words: list[str],
            coordsAbs: list[tuple[int, int]],
            scoreSum: float,
            yesIndexes: list[int],
        ) -> None:
            n = len(words)
            avg = (float(scoreSum) / float(n)) if n > 0 else 0.0
            for pi in yesIndexes:
                results.append(
                    SolutionDecryptage(
                        patternIndex=pi,
                        words=words,
                        coordsAbs=coordsAbs,
                        scoreMax=avg,
                    )
                )



        # Passe 0: creation frontier
        hits0 = self._build_hits(self.triplets[0], scope, decryptorConfig)
        initPacked = listePatterns.getPackedInitialState()
        packedJoker0, packedNonJoker0 = listePatterns.splitPackedByJokerAt(0, initPacked)

        # 0a) Branche joker : EXCLUSIVE pour patterns joker
        if hits0 and packedJoker0 != 0:
            words0 = [JOKER_WORD_ABS]
            continueExplore, yesIndexes, packedOut = listePatterns.validateSequence(
                words0,
                patternMode,
                packedState=packedJoker0,
            )
            if continueExplore or yesIndexes:
                coords0 = [JOKER_COORD_ABS]
                frontier.append(
                    DecryptageCandidate(
                        words=words0,
                        coordsAbs=coords0,
                        patternPacked=packedOut,
                        scoreMax=0.0,
                    )
                )
                emitSolutions(words0, coords0, 0.0, yesIndexes)

        # 0b) Branches normales : uniquement patterns NON joker
        if packedNonJoker0 != 0:
            for (rowExt, colExt, word, hitScore) in hits0:
                words0 = [word]
                continueExplore, yesIndexes, packedOut = listePatterns.validateSequence(
                    words0,
                    patternMode,
                    packedState=packedNonJoker0,
                )
                if (not continueExplore) and (not yesIndexes):
                    continue

                coords0 = [(rowExt, colExt)]
                frontier.append(
                    DecryptageCandidate(
                        words=words0,
                        coordsAbs=coords0,
                        patternPacked=packedOut,
                        scoreMax=hitScore,
                    )
                )
                emitSolutions(words0, coords0, hitScore, yesIndexes)

        # -------------------------
        # Passes suivantes
        # -------------------------
        for depth in range(1, len(self.triplets)):
            hits = self._build_hits(self.triplets[depth], scope, decryptorConfig)
            next_frontier: list[DecryptageCandidate] = []

            for cand in frontier:
                packedJoker, packedNonJoker = listePatterns.splitPackedByJokerAt(depth, cand.patternPacked)

                # 1) Branche joker : EXCLUSIVE pour patterns joker
                if hits and packedJoker != 0:
                    new_words = cand.words + [JOKER_WORD_ABS]
                    continueExplore, yesIndexes, packedOut = listePatterns.validateSequence(
                        new_words,
                        patternMode,
                        packedState=packedJoker,
                    )
                    if continueExplore or yesIndexes:
                        new_coords = cand.coordsAbs + [JOKER_COORD_ABS]
                        next_frontier.append(
                            DecryptageCandidate(
                                words=new_words,
                                coordsAbs=new_coords,
                                patternPacked=packedOut,
                                scoreMax=cand.scoreMax,
                            )
                        )
                        emitSolutions(new_words, new_coords, cand.scoreMax, yesIndexes)

                # 2) Branches normales : uniquement patterns NON joker
                if packedNonJoker != 0:
                    for (rowExt, colExt, word, hitScore) in hits:
                        new_words = cand.words + [word]
                        continueExplore, yesIndexes, packedOut = listePatterns.validateSequence(
                            new_words,
                            patternMode,
                            packedState=packedNonJoker,
                        )
                        if (not continueExplore) and (not yesIndexes):
                            continue

                        new_coords = cand.coordsAbs + [(rowExt, colExt)]
                        newScoreSum = cand.scoreMax + hitScore
                        next_frontier.append(
                            DecryptageCandidate(
                                words=new_words,
                                coordsAbs=new_coords,
                                patternPacked=packedOut,
                                scoreMax=newScoreSum,
                            )
                        )
                        emitSolutions(new_words, new_coords, newScoreSum, yesIndexes)

            frontier = next_frontier
            if not frontier:
                break

        return results
