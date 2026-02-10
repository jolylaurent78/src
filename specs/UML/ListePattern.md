```mermaid
sequenceDiagram
    participant Engine as DecryptorEngine
    participant LP as ListePatterns
    participant Cand as DecryptageCandidate
    participant PSS as PatternStateSet (int packé)

    Note over Engine: Le nœud candidat est déjà validé côté mesures

    %% Boucle principale : parcours du dictionnaire à ce niveau
    loop Pour chaque mot du dictionnaire (niveau courant)
        Engine->>LP: Évaluer les patterns sur la séquence courante + mot candidat
        LP-->>Engine: DécisionExploration (PeutÊtre / NePasExplorer)\n+ SolutionsDétectées (0..N)

        alt Solutions détectées (>=1)
            Engine->>Cand: Demander un snapshot de la séquence courante
            Cand-->>Engine: Snapshot de séquence
            Note over Engine: Construire et publier DecryptageSolution
        end

        alt DécisionExploration = NePasExplorer
            Note over Engine: Branche morte → passer au mot suivant
        else DécisionExploration = PeutÊtre
            Engine->>Cand: Empiler l’état courant (snapshot int)
            Engine->>Cand: Ajouter le nœud candidat accepté à la séquence
            Cand->>PSS: Mettre à jour l’état compact (int)

            %% Descente DFS
            Note over Engine: Descente d’un niveau (DFS)

            %% Backtracking
            Engine->>Cand: Retirer le dernier nœud (backtrack)
            Cand->>Cand: Restaurer l’état via pile de snapshots (int)
        end
    end
