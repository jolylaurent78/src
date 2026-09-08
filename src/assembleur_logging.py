"""Configuration centralisee du logging applicatif."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.assembleur_paths import ApplicationPaths


LOG_FILENAME = "assembleur.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
_APPLICATION_LOGGER_NAME = "src"
_FORMATTER = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _is_application_handler(handler: logging.Handler) -> bool:
    return bool(getattr(handler, "_assembleur_logging_handler", False))


def configure_logging(paths: ApplicationPaths) -> logging.Logger:
    """Configure les handlers applicatifs fichier et, en SYS, console.

    La configuration remplace uniquement les handlers installes par cette
    application afin de rester idempotente sans perturber les bibliotheques.
    """
    paths.ensure_user_data_directories()
    logger = logging.getLogger(_APPLICATION_LOGGER_NAME)
    logger.setLevel(logging.DEBUG if paths.catalogue_mode == "SYS" else logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if not _is_application_handler(handler):
            continue
        logger.removeHandler(handler)
        handler.close()

    file_handler = RotatingFileHandler(
        paths.logs_dir / LOG_FILENAME,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler._assembleur_logging_handler = True
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)

    if paths.catalogue_mode == "SYS":
        console_handler = logging.StreamHandler()
        console_handler._assembleur_logging_handler = True
        console_handler.setFormatter(_FORMATTER)
        logger.addHandler(console_handler)

    return logger
