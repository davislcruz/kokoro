"""Shared constants for Kokoro TTS.

This module centralizes all constants to avoid circular dependencies
and improve maintainability.
"""

# Language aliases for user convenience
ALIASES = {
    'en-us': 'a',
    'en-gb': 'b',
    'es': 'e',
    'fr-fr': 'f',
    'hi': 'h',
    'it': 'i',
    'pt-br': 'p',
    'ja': 'j',
    'zh': 'z',
}

# Language code mappings
LANG_CODES = {
    # pip install misaki[en]
    'a': 'American English',
    'b': 'British English',

    # espeak-ng
    'e': 'es',
    'f': 'fr-fr',
    'h': 'hi',
    'i': 'it',
    'p': 'pt-br',

    # pip install misaki[ja]
    'j': 'Japanese',

    # pip install misaki[zh]
    'z': 'Mandarin Chinese',
}

# Audio processing constants
SAMPLE_RATE = 24000
MAGIC_DIVISOR = 80  # For timestamp calculations
MAX_PHONEME_LENGTH = 510  # Maximum phoneme sequence length

# Default chunk size for non-English text processing
DEFAULT_CHUNK_SIZE = 400

# Default voice
DEFAULT_VOICE = "af_heart"

# Default repository
DEFAULT_REPO_ID = 'hexgrad/Kokoro-82M'

# Default output directory for generated audio files
DEFAULT_OUTPUT_DIR = "output"