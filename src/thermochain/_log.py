# coding: utf-8
"""Stdout logger setup (vendored from gvpy.misc.log)."""
import sys

from loguru import logger


def log():
    """Set up a colorized stdout logger using loguru."""
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<e>{time:YYYY-MM-DD HH:mm:ss}</e> | {level} | <level>{message}</level>",
    )
    return logger
