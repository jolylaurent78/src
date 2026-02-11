from src.assembleur_decryptor import ClockDicoDecryptor, DecryptorConfig


def makeDecryptorConfigAngle180Only(tol: float = 0.01) -> DecryptorConfig:
    decryptor = ClockDicoDecryptor()
    return DecryptorConfig(
        decryptor=decryptor,
        useAzA=False,
        useAzB=False,
        useAngle180=True,
        toleranceDeg=float(tol),
    )
