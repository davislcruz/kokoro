"""Input text handling for Kokoro TTS CLI."""

from pathlib import Path
import sys
from loguru import logger


def load_text(text_arg: str = None, input_file: Path = None) -> str:
    """
    Load input text from one of three sources:
    1. Command-line text argument
    2. Input file
    3. Standard input (stdin)
    """
    if text_arg and input_file:
        raise ValueError("Cannot specify both --text and --input-file")

    if text_arg:
        logger.debug(f"Using text from command line: {text_arg!r}")
        return text_arg

    if input_file:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        text = input_file.read_text(encoding="utf-8")
        logger.debug(f"Loaded text from file {input_file}: {text!r}")
        return text

    return _read_stdin()


def _read_stdin() -> str:
    print("Reading from stdin... Press Ctrl+D (Unix) or Ctrl+Z + Enter (Windows) to finish.", flush=True)
    text = "".join(sys.stdin)
    logger.debug(f"Read text from stdin: {text!r}")
    return text