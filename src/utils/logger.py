"""
src/utils/logger.py
~~~~~~~~~~~~~~~~~~~
Centralised logging setup using loguru.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("processing frame {frame}", frame=42)
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_level: str = "INFO", log_dir: str | Path | None = None) -> None:
    """
    Configure loguru sinks.

    Args:
        log_level: Minimum log level string (DEBUG/INFO/WARNING/ERROR).
        log_dir:   Optional directory for rotating file logs. When None, only
                   stdout is used.
    """
    # Remove the default loguru handler
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console sink
    logger.add(sys.stdout, level=log_level, format=fmt, colorize=True)

    # Optional file sink with rotation
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path / "pipeline_{time:YYYY-MM-DD}.log",
            level=log_level,
            format=fmt,
            rotation="10 MB",
            retention="14 days",
            compression="gz",
        )


def get_logger(name: str):
    """Return a child logger bound to *name* (typically ``__name__``)."""
    return logger.bind(name=name)
