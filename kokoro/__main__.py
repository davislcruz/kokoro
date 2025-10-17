"""Kokoro TTS CLI Entry Point.

Example usage:
    python3 -m kokoro --text "Hello World" -o output.wav --debug
    echo "Bom dia mundo" | python3 -m kokoro -o audio.wav -l p --voice pm_alex
    python3 -m kokoro -i input.txt -o output.wav --voice af_heart --speak

Common issues:
    pip not installed: `uv pip install pip`
    espeak not installed: `apt-get install espeak-ng`
"""

from loguru import logger

from .cli import (
    create_parser,
    load_text,
    validate_and_prepare_output,
    save_as_file,
    format_metrics,
    setup_logging
)
from .cli.playback_handler import play_audio_file


def main() -> None:
    """Main entry point for Kokoro TTS CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)
    logger.debug(f"CLI arguments: {args}")

    try:
        # Determine voice and language
        voice = args.voice
        lang = args.language or voice[0]

        # Validate speed
        if args.speed <= 0:
            raise ValueError("Speed must be greater than 0")

        # Load text
        text = load_text(args.text, args.input_file)

        # Prepare output
        output_file = validate_and_prepare_output(args.output_file)

        logger.debug(f"Using voice: {voice}, language: {lang}, speed: {args.speed}")
        logger.debug(f"Input text: {text!r}")
        logger.debug(f"Output file: {output_file}")

        # Generate and save audio
        metrics = save_as_file(
            output_file=output_file,
            text=text,
            language=lang,
            voice=voice,
            speed=args.speed,
            device=args.device,
            return_metadata=args.metrics,
        )

        # Display performance metrics
        if metrics:
            print(format_metrics(metrics, detailed=args.metrics))

        # Play audio if requested
        if args.speak:
            play_audio_file(output_file)

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            logger.exception("Detailed error information:")
        raise


if __name__ == "__main__":
    main()