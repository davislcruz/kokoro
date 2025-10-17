"""Configuration module for Kokoro TTS.

This module provides access to configuration constants and settings.
"""

from .defaults import *
from .languages import *

__all__ = [
    # Language related
    'ALIASES',
    'LANG_CODES',
    'SUPPORTED_LANGUAGES',

    # Audio constants
    'SAMPLE_RATE',
    'MAGIC_DIVISOR',
    'MAX_PHONEME_LENGTH',

    # Default settings
    'DEFAULT_VOICE',
    'DEFAULT_REPO_ID',
    'DEFAULT_CHUNK_SIZE',
    'SUPPORTED_DEVICES',
    'DEFAULT_SPEED',
]