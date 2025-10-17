"""CLI module for Kokoro TTS.

This module handles command-line interface components.
"""

from .parser import create_parser
from .input_handler import load_text
from .output_handler import validate_and_prepare_output
from .audio import create_audio_stream, save_as_file
from .metrics import format_metrics
from .logging_setup import setup_logging

__all__ = [
    'create_parser',
    'load_text',
    'validate_and_prepare_output',
    'create_audio_stream',
    'save_as_file',
    'format_metrics',
    'setup_logging'
]