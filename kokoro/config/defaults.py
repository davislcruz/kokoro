"""Default configuration constants for Kokoro TTS."""

import json
from pathlib import Path

# Load configuration from JSON files
_config_dir = Path(__file__).parent
_defaults = json.loads((_config_dir / 'defaults.json').read_text())
_languages = json.loads((_config_dir / 'languages.json').read_text())

# Audio processing constants
SAMPLE_RATE = _defaults['audio']['sample_rate']
MAGIC_DIVISOR = _defaults['audio']['magic_divisor']
MAX_PHONEME_LENGTH = _defaults['audio']['max_phoneme_length']

# Processing constants
DEFAULT_CHUNK_SIZE = _defaults['processing']['default_chunk_size']

# Model constants
DEFAULT_VOICE = _defaults['models']['default_voice']
DEFAULT_REPO_ID = _defaults['models']['default_repo_id']

# CLI constants
SUPPORTED_DEVICES = _defaults['cli']['supported_devices']
DEFAULT_SPEED = _defaults['cli']['default_speed']