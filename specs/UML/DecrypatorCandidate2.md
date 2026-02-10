```mermaid
sequenceDiagram
    participant Engine as DecryptorEngine
    participant Dico as Dictionnaire
    participant Dec as Decryptor
    participant LP as ListePatterns
    participant Cand as DecryptageCandidate
    participant Sol as DecryptageSolution

    Note over Engine: Parcours DFS par niveaux. A chaque niveau, on parcourt tous les mots du dictionnaire.

    loop Parcours des mots du dictionnaire (niveau courant)
        Engine->>Dico: Proposer un mot candidat
        Dico-->>Engine: Mot + Coordonnées

        Engine->>Dec: Décrypter (coordonnées)
        Dec-->>Engine: ClockState

        alt Mesures KO
            Note over Engine: Rejeter le mot\nContinuer avec le mot suivant (même niveau)
        else Mesures OK
            Engine->>LP: Evaluer patterns (séquence courante + mot)
            LP-->>Engine: Résultat exploration\n(+ éventuelle nouvelle solution)

            alt Résultat = NouvelleSolution
                Engine->>Sol: Créer / émettre une solution figée
                Note over Engine: Continuer l'exploration\nMot suivant (même niveau)
            else Résultat = NePasExplorer
                Note over Engine: Rejeter ce mot\nMot suivant (même niveau)
            else Résultat = PeutEtre
                Engine->>Cand: Empiler le noeud candidat accepté
                Note over Engine: Descendre d'un niveau (DFS)
            end
        end
    end

    Note over Engine: Si aucun PeutEtre trouvé à ce niveau\nBacktrack: dépiler et revenir au niveau précédent
    Engine->>Cand: Depiler (retour niveau précédent)
