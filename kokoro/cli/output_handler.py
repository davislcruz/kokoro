"""Output file handling for Kokoro TTS CLI."""

from pathlib import Path
from loguru import logger


def validate_and_prepare_output(output_file: Path) -> Path:
    if not output_file.suffix == ".wav":
        logger.warning("The output file name should end with .wav")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file