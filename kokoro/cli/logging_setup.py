"""Logging configuration for Kokoro TTS CLI."""

from loguru import logger


def setup_logging(debug: bool = False) -> None:
    if debug:
        logger.level("DEBUG")