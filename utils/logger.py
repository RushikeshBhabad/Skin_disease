"""
Structured logging for the Skin Disease Detection System.
Provides console and file logging with consistent formatting.
"""

import logging
import os
import sys
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Create a configured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module).
        log_file: Optional path to a log file. Defaults to logs/app.log.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (DEBUG level)
    if log_file is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "app.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
