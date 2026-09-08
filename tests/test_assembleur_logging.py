from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import pytest

from src.assembleur_logging import BACKUP_COUNT, MAX_BYTES, configure_logging
from src.assembleur_paths import ApplicationPaths


@pytest.fixture(autouse=True)
def _remove_application_handlers_after_test():
    yield
    logger = logging.getLogger("src")
    for handler in _application_handlers(logger):
        logger.removeHandler(handler)
        handler.close()


def _paths(tmp_path, mode: str) -> ApplicationPaths:
    paths = ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user-data",
        catalogue_mode=mode,
    )
    if mode == "USER":
        paths.default_scenarios_dir.mkdir(parents=True)
    return paths


def _application_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [
        handler
        for handler in logger.handlers
        if getattr(handler, "_assembleur_logging_handler", False)
    ]


def _file_handler(logger: logging.Logger) -> RotatingFileHandler:
    return next(
        handler
        for handler in _application_handlers(logger)
        if isinstance(handler, RotatingFileHandler)
    )


def _console_handlers(logger: logging.Logger) -> list[logging.StreamHandler]:
    return [
        handler
        for handler in _application_handlers(logger)
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]


def test_user_creates_the_central_log_file(tmp_path) -> None:
    paths = _paths(tmp_path, "USER")
    configure_logging(paths)

    logging.getLogger("src.test_logging").info("test-user")
    _file_handler(logging.getLogger("src")).flush()

    log_path = paths.logs_dir / "assembleur.log"
    assert log_path.is_file()
    assert "test-user" in log_path.read_text(encoding="utf-8")


def test_sys_configures_file_and_console_handlers(tmp_path) -> None:
    paths = _paths(tmp_path, "SYS")
    logger = configure_logging(paths)
    logging.getLogger("src.test_logging").debug("test-sys")
    _file_handler(logger).flush()

    assert isinstance(_file_handler(logger), RotatingFileHandler)
    assert len(_console_handlers(logger)) == 1
    assert "test-sys" in (paths.logs_dir / "assembleur.log").read_text(
        encoding="utf-8"
    )


def test_user_has_no_console_handler(tmp_path) -> None:
    logger = configure_logging(_paths(tmp_path, "USER"))

    assert len(_console_handlers(logger)) == 0


@pytest.mark.parametrize(("mode", "expected_level"), [("SYS", logging.DEBUG), ("USER", logging.INFO)])
def test_mode_sets_the_effective_application_level(tmp_path, mode, expected_level) -> None:
    logger = configure_logging(_paths(tmp_path, mode))

    assert logger.getEffectiveLevel() == expected_level


def test_reconfiguration_replaces_handlers_without_duplicating_messages(tmp_path) -> None:
    paths = _paths(tmp_path, "USER")
    configure_logging(paths)
    logger = configure_logging(paths)
    logging.getLogger("src.test_logging").info("one-message")
    _file_handler(logger).flush()

    assert len(_application_handlers(logger)) == 1
    assert (paths.logs_dir / "assembleur.log").read_text(encoding="utf-8").count(
        "one-message"
    ) == 1


def test_rotation_uses_utf8_and_the_configured_retention(tmp_path) -> None:
    logger = configure_logging(_paths(tmp_path, "USER"))
    handler = _file_handler(logger)

    assert handler.maxBytes == MAX_BYTES
    assert handler.backupCount == BACKUP_COUNT
    assert handler.encoding == "utf-8"
