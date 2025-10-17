"""Language configuration for Kokoro TTS."""

import json
from pathlib import Path

# Load language configuration
_config_dir = Path(__file__).parent
_languages = json.loads((_config_dir / 'languages.json').read_text())

# Language aliases for user convenience
ALIASES = _languages['aliases']

# Language code mappings
LANG_CODES = _languages['lang_codes']

# Supported languages list
SUPPORTED_LANGUAGES = _languages['supported_languages']