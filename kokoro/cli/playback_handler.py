"""Audio playback handling for Kokoro TTS CLI."""

import subprocess
from pathlib import Path
from loguru import logger


def play_audio_file(audio_file: Path) -> None:
    try:
        logger.info(f"Playing audio file: {audio_file}")
        subprocess.run(["paplay", str(audio_file)], check=True)
        logger.info("Audio playback completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Could not play audio: {e}")
        logger.error("Please check your audio system.")
    except FileNotFoundError:
        logger.error("paplay not found. Please install PulseAudio or play the file manually.")
        logger.error("You can try: aplay, mpv, or vlc to play the file.")