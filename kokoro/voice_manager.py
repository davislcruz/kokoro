"""Voice management system for handling voice loading and caching."""

from typing import Dict, Union
import torch
from loguru import logger
from .config import LANG_CODES
from .local_model_manager import LocalModelManager


class VoiceManager:
    """Handles voice loading and caching with device-specific optimization."""

    def __init__(self, repo_id: str, lang_code: str):
        """Initialize voice manager.

        Args:
            repo_id: HuggingFace repository ID for voice models
            lang_code: Language code for voice validation warnings
        """
        self.repo_id = repo_id
        self.lang_code = lang_code
        self.voices: Dict[str, torch.Tensor] = {}
        self._device_voice_cache: Dict[str, torch.Tensor] = {}
        self.local_manager = LocalModelManager(repo_id=repo_id)

    def load_single_voice(self, voice: str) -> torch.Tensor:
        """Load a single voice from file or cache.

        Args:
            voice: Voice name or file path

        Returns:
            Voice tensor loaded from file

        Raises:
            RuntimeError: If voice file cannot be loaded
        """
        if voice in self.voices:
            return self.voices[voice]

        if voice.endswith('.pt'):
            f = voice
        else:
            f = str(self.local_manager.get_voice(voice))
            if not voice.startswith(self.lang_code):
                v = LANG_CODES.get(voice, voice)
                p = LANG_CODES.get(self.lang_code, self.lang_code)
                logger.warning(f'Language mismatch, loading {v} voice into {p} pipeline.')

        pack = torch.load(f, weights_only=True)
        self.voices[voice] = pack
        return pack

    def load_voice(self, voice: Union[str, torch.FloatTensor], delimiter: str = ",") -> torch.FloatTensor:
        """Load voice(s) with support for voice blending.

        Args:
            voice: Voice name(s) or tensor. Multiple voices can be separated by delimiter
            delimiter: Delimiter for separating multiple voice names

        Returns:
            Voice tensor (averaged if multiple voices specified)
        """
        if isinstance(voice, torch.FloatTensor):
            return voice

        if voice in self.voices:
            return self.voices[voice]  # type: ignore[return-value]

        logger.debug(f"Loading voice: {voice}")
        packs = [self.load_single_voice(v) for v in voice.split(delimiter)]

        if len(packs) == 1:
            return packs[0]

        # Average multiple voices
        self.voices[voice] = torch.mean(torch.stack(packs), dim=0)
        return self.voices[voice]  # type: ignore[return-value]

    def load_voice_for_device(self, voice: Union[str, torch.FloatTensor], device: torch.device) -> torch.FloatTensor:
        """Load voice with device-specific caching to avoid redundant transfers.

        Args:
            voice: Voice name or tensor
            device: Target device for the voice tensor

        Returns:
            Voice tensor on the specified device
        """
        if isinstance(voice, torch.FloatTensor):
            if voice.device == device:
                return voice
            return voice.to(device)

        device_str = str(device)
        cache_key = f"{voice}_{device_str}"

        if cache_key not in self._device_voice_cache:
            voice_tensor = self.load_voice(voice)
            self._device_voice_cache[cache_key] = voice_tensor.to(device)

        return self._device_voice_cache[cache_key]

    def clear_cache(self):
        """Clear all voice caches to free memory."""
        self.voices.clear()
        self._device_voice_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about current cache usage.

        Returns:
            Dictionary with cache statistics
        """
        return {
            'voices_cached': len(self.voices),
            'device_voices_cached': len(self._device_voice_cache),
            'total_memory_tensors': len(self.voices) + len(self._device_voice_cache)
        }