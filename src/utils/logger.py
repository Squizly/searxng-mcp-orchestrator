import logging
import sys

from config.settings import settings

_LEVEL_MAP = {
    "DEBUG": "ОТЛАДКА",
    "INFO": "ИНФО",
    "WARNING": "ВНИМАНИЕ",
    "ERROR": "ОШИБКА",
    "CRITICAL": "КРИТИЧНО",
}

class LevelFormatter(logging.Formatter):
    """Форматтер, который заменяет уровни логирования на русские."""

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        record.levelname = _LEVEL_MAP.get(record.levelname, record.levelname)
        try:
            return super().format(record)
        finally:
            record.levelname = original


def setup_logger(name: str = "searxng_agent") -> logging.Logger:
    """Настраивает логгер с выводом в консоль и файл."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        return logger

    console_formatter = LevelFormatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    file_formatter = LevelFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, settings.log_console_level.upper()))
    logger.addHandler(console_handler)

    log_file_path = settings.log_file
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    noisy = [
        "mcp",
        "fastmcp",
        "anyio",
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
    ]
    for noisy_name in noisy:
        logging.getLogger(noisy_name).setLevel(logging.WARNING)

    return logger
