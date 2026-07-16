from src.utils import logging_utils


def _handlers_for(logger, path):
    return [
        handler for handler in logger.handlers
        if getattr(handler, "_assembleur_log_file", None) == str(path)
    ]


def test_loggers_create_files_without_duplicate_handlers(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "_LOG_DIR", tmp_path)

    app_logger = logging_utils.get_app_logger()
    mig_logger = logging_utils.get_mig_geo_logger()
    logging_utils.get_app_logger()
    logging_utils.get_mig_geo_logger()

    assert (tmp_path / "app.log").is_file()
    assert (tmp_path / "mig_geo.log").is_file()
    assert len(_handlers_for(app_logger, tmp_path / "app.log")) == 1
    assert len(_handlers_for(mig_logger, tmp_path / "mig_geo.log")) == 1


def test_mig_geo_log_rotates_with_standard_rotating_handler(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "_LOG_DIR", tmp_path)
    logger = logging_utils.get_mig_geo_logger()
    handler = _handlers_for(logger, tmp_path / "mig_geo.log")[0]
    handler.maxBytes = 100
    handler.backupCount = 5

    for _ in range(10):
        logger.info("[MIG-GEO] ligne de test suffisamment longue pour déclencher une rotation")
    handler.flush()

    assert (tmp_path / "mig_geo.log").is_file()
    assert (tmp_path / "mig_geo.log.1").is_file()
