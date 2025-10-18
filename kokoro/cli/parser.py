"""Argument parser for Kokoro TTS CLI."""

import argparse
from pathlib import Path
from ..config import DEFAULT_VOICE, SUPPORTED_LANGUAGES


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kokoro",
        description="Kokoro TTS - Text-to-Speech synthesis",
        epilog=(
            "Examples:\n"
            "  python3 -m kokoro --text \"Hello World\" -o output.wav\n"
            "  echo \"Hello\" | python3 -m kokoro -o output.wav\n"
            "  python3 -m kokoro -i input.txt -o output.wav --voice af_heart"
        ),
    )

    # Voice and language
    parser.add_argument("-m", "--voice", default=DEFAULT_VOICE, help=f"Voice to use (default: {DEFAULT_VOICE})")
    parser.add_argument("-l", "--language", help="Language to use (defaults to the one corresponding to the voice)", choices=SUPPORTED_LANGUAGES)

    # Input / Output
    parser.add_argument("-o", "--output-file", "--output_file", type=Path, help="Path to output WAV file (default: output/audio_TIMESTAMP.wav)")
    parser.add_argument("-i", "--input-file", "--input_file", type=Path, help="Path to input text file (default: stdin)")
    parser.add_argument("-t", "--text", help="Text to use instead of reading from stdin")

    # Options
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed (default: 1.0)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Device to use for inference")

    # Flags
    parser.add_argument("--speak", action="store_true", help="Play the generated audio file after creation")
    parser.add_argument("--metrics", action="store_true", help="Show detailed performance metrics")
    parser.add_argument("--debug", action="store_true", help="Print DEBUG messages to console")

    return parser