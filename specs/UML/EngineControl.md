```mermaid
sequenceDiagram
participant Tk
participant Ctrl
participant Eng
participant Q

Tk->>Ctrl: create
Tk->>Q: create
Tk->>Eng: start
activate Eng

loop EngineLoop
    Eng->>Ctrl: checkStop
    Eng->>Ctrl: checkPause
    Eng->>Eng: compute
    Eng->>Q: putSolution
end

Eng->>Q: putDone
deactivate Eng

loop UiPolling
    Tk->>Q: getEvent
    Tk->>Tk: updateUi
end

Tk->>Ctrl: requestPause
Tk->>Ctrl: requestStop
