"""Output file handling for Kokoro TTS CLI."""

from pathlib import Path
from datetime import datetime
from loguru import logger
from ..config import DEFAULT_OUTPUT_DIR


def generate_default_filename() -> Path:
    """Generate a default filename with timestamp in the output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(DEFAULT_OUTPUT_DIR) / f"audio_{timestamp}.wav"


def validate_and_prepare_output(output_file: Path = None) -> Path:
    """Validate and prepare output file path.

    Args:
        output_file: Optional path to output file. If None, generates default.

    Returns:
        Path object for the output file.
    """
    if output_file is None:
        output_file = generate_default_filename()
        logger.info(f"No output file specified, using: {output_file}")

    if not output_file.suffix == ".wav":
        logger.warning("The output file name should end with .wav")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file