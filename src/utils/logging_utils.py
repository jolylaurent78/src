"""Loggers fichiers légers et idempotents pour AssembleurTriangles."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5
_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _get_logger(name: str, filename: str) -> logging.Logger:
    """Retourne un logger fichier unique, sans multiplier ses handlers."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = _LOG_DIR / filename
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    existing = None
    for handler in list(logger.handlers):
        if getattr(handler, "_assembleur_log_file", None) != str(path):
            continue
        existing = handler
        break

    if existing is None:
        handler = RotatingFileHandler(
            path,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        handler._assembleur_log_file = str(path)
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)
    return logger


def get_app_logger() -> logging.Logger:
    """Logger général de l'application, écrit dans ``logs/app.log``."""
    return _get_logger("APP", "app.log")


def get_mig_geo_logger() -> logging.Logger:
    """Logger dédié aux migrations et audits géométriques/topologiques."""
    return _get_logger("MIG-GEO", "mig_geo.log")
