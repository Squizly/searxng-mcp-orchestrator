from __future__ import annotations

import os
import warnings


_TORCH_DTYPE_WARNING = r".*`torch_dtype` is deprecated! Use `dtype` instead!.*"


def configure_runtime_environment() -> None:
    """Подавляет шумные предупреждения и настраивает тихий режим для зависимостей."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for category in (DeprecationWarning, FutureWarning, UserWarning, Warning):
        warnings.filterwarnings(
            "ignore",
            message=_TORCH_DTYPE_WARNING,
            category=category,
        )

    try:
        from huggingface_hub.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass
