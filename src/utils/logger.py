import logging
import sys

from config.settings import settings
from src.utils.runtime import configure_runtime_environment

try:
    from rich.console import Console
    from rich.logging import RichHandler
except Exception:  # pragma: no cover
    Console = None
    RichHandler = None

_LEVEL_MAP = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "fatal",
}


def _apply_russian_level_names() -> None:
    for level, level_name in _LEVEL_MAP.items():
        logging.addLevelName(level, level_name)


def setup_logger(name: str = "searxng_agent") -> logging.Logger:
    """Настраивает логгер с выводом в консоль и файл."""
    configure_runtime_environment()

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    _apply_russian_level_names()

    if logger.handlers:
        return logger

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if settings.show_logs:
        if RichHandler is not None and Console is not None:
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_level=False,
                show_path=False,
                markup=False,
                rich_tracebacks=True,
                log_time_format="%H:%M:%S",
            )
            console_handler.setFormatter(logging.Formatter("· %(message)s"))
        else:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s · %(levelname)s · %(message)s", datefmt="%H:%M:%S")
            )

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
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "filelock",
    ]
    for noisy_name in noisy:
        logging.getLogger(noisy_name).setLevel(logging.WARNING)

    # Trafilatura часто шумит warning'ами по битым ссылкам на страницах,
    # это не влияет на извлечение текста и засоряет вывод.
    very_noisy = [
        "trafilatura",
        "trafilatura.xml",
        "courlan",
        "htmldate",
    ]
    for noisy_name in very_noisy:
        noisy_logger = logging.getLogger(noisy_name)
        noisy_logger.setLevel(logging.ERROR)
        noisy_logger.propagate = False

    quiet_errors_only = [
        "huggingface_hub",
        "huggingface_hub.utils",
        "huggingface_hub.utils._http",
        "transformers",
        "sentence_transformers",
    ]
    for noisy_name in quiet_errors_only:
        noisy_logger = logging.getLogger(noisy_name)
        noisy_logger.setLevel(logging.ERROR)
        noisy_logger.propagate = False

    return logger
