"""Audio handling for Kokoro TTS CLI."""

import time
import wave
from pathlib import Path
from typing import Generator, Iterator, Any, TYPE_CHECKING

import numpy as np
from loguru import logger
from ..config import SAMPLE_RATE, DEFAULT_REPO_ID

if TYPE_CHECKING:
    from kokoro import KPipeline


def create_audio_stream(
    text: str,
    language: str,
    voice: str,
    speed: float = 1.0,
    device=None
) -> Generator["KPipeline.Result", None, None]:
    """Generate audio chunks from text using Kokoro pipeline."""
    from kokoro import KPipeline

    if not voice.startswith(language):
        logger.warning(f"Voice {voice} is not made for language {language}")

    # TODO: Consider adding --repo-id CLI flag to allow using alternative models
    # (e.g., hexgrad/Kokoro-82M-v1.1-zh for Chinese-optimized model)
    pipeline = KPipeline(lang_code=language, repo_id=DEFAULT_REPO_ID, device=device)
    yield from pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")


def _format_to_bytes(
    text: str,
    language: str,
    voice: str,
    speed: float,
    device: Any
) -> Iterator[bytes]:
    """Generate audio bytes lazily, independent of file I/O.

    This function separates audio generation/processing from file writing,
    following the Single Responsibility Principle.
    """
    for result in create_audio_stream(text, language, voice, speed, device):
        if result.audio is not None:
            yield (result.audio.numpy() * 32767).astype(np.int16).tobytes()


def save_as_file(
    output_file: Path,
    text: str,
    language: str,
    voice: str,
    speed: float = 1.0,
    device: Any = None,
    return_metadata: bool = False
) -> dict | None:
    """Save audio from text to a WAV file.

    Args:
        output_file: Path to save the audio.
        text: Input text to synthesize.
        language: Language code.
        voice: Voice to use.
        speed: Speech speed.
        device: Inference device.
        return_metadata: If True, returns performance metrics.

    Returns:
        dict with audio metadata if `return_metadata` is True, else None.
    """
    total_samples = 0
    chunk_count = 0
    start_time = time.perf_counter() if return_metadata else None

    with wave.open(str(output_file.resolve()), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

        for audio_bytes in _format_to_bytes(text, language, voice, speed, device):
            wav_file.writeframes(audio_bytes)
            if return_metadata:
                total_samples += len(audio_bytes) // 2  # 2 bytes per sample
                chunk_count += 1

    if not return_metadata:
        return None

    synthesis_time = time.perf_counter() - start_time
    audio_duration = total_samples / SAMPLE_RATE
    rtf = synthesis_time / audio_duration if audio_duration > 0 else float("inf")

    return {
        "synthesis_time": synthesis_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "total_samples": total_samples,
        "chunk_count": chunk_count,
        "text_length": len(text),
    }
