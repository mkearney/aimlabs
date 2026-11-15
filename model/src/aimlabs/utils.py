import logging
import os
import random

import structlog

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure structlog for console output.

    Called automatically the first time `get_logger` is used; call it yourself
    beforehand to change the level.
    """
    global _configured
    try:  # structlog >= 25.1 renamed `pad_event` to `pad_event_to`
        renderer = structlog.dev.ConsoleRenderer(pad_event_to=7)
    except TypeError:
        renderer = structlog.dev.ConsoleRenderer(pad_event=7)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            # default event padding is 30 characters; keep the log lines compact
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str | None = None, **initial_values) -> structlog.BoundLogger:
    """Return a structlog logger, configuring structlog on first use."""
    if not _configured:
        configure_logging()
    if name:
        initial_values = {"component": name, **initial_values}
    return structlog.get_logger(**initial_values)


def seed_everything(seed: int) -> None:
    """Seed python, numpy (if installed), and torch RNGs for reproducibility."""
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch.manual_seed(seed)


def get_device(preferred: str | None = None):
    """Return the best available torch device (cuda > mps > cpu)."""
    import torch

    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
