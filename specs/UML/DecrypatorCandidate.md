```mermaid
sequenceDiagram
    participant Engine as DecryptorEngine
    participant Candidate as DecryptageCandidate
    participant Patterns as PatternStateSet (int packé)

    Note over Engine: Nœud transmis à Candidate déjà validé (mesures OK, patterns évalués)

    %% Initialisation
    Engine->>Candidate: Démarrer un parcours (état vide)
    Candidate->>Patterns: Initialiser l'état (int)

    %% Descente DFS
    loop Descente (nœud validé)
        Engine->>Candidate: Ajouter un pas validé à la séquence
        Candidate->>Patterns: Mettre à jour l'état compact (int)
        Note over Candidate,Patterns: L'état précédent est empilé (snapshot int)\npour permettre le backtracking sans jeton
        Candidate-->>Engine: État courant mis à jour

        alt Solution détectée par l’Engine
            Engine->>Candidate: Demander un instantané de la séquence
            Candidate-->>Engine: Retourner une solution figée
        end
    end

    %% Backtrack
    loop Backtrack (remontée)
        Engine->>Candidate: Retirer le dernier pas
        Candidate->>Patterns: Restaurer l'état précédent via pile de snapshots (int)
        Candidate-->>Engine: Retour arrière effectué
    end
