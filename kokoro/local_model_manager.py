"""Local model manager for downloading and caching Kokoro models and assets."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
from loguru import logger
from huggingface_hub import hf_hub_download


class LocalModelManager:
    """Manages local storage and downloading of Kokoro models, configs, and voices."""

    def __init__(self, base_path: Optional[str] = None, repo_id: str = 'hexgrad/Kokoro-82M'):
        """Initialize the local model manager.

        Args:
            base_path: Base directory for storing models. Defaults to current directory.
            repo_id: Hugging Face repository ID for downloading models.
        """
        self.repo_id = repo_id
        if base_path:
            self.base_path = Path(base_path)
        else:
            # Default to kokoro package directory
            self.base_path = Path(__file__).parent

        # Define folder structure inside kokoro package
        self.config_dir = self.base_path / 'config'
        self.models_dir = self.base_path / 'models'
        self.voices_dir = self.base_path / 'voices'

        # Model filename mapping
        self.model_files = {
            'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
            'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
        }

        # Create directories
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.config_dir, self.models_dir, self.voices_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    def _download_if_missing(self, local_path: Path, repo_filename: str) -> Path:
        """Download file from HF Hub only if it doesn't exist locally.

        Args:
            local_path: Local file path where the file should be stored
            repo_filename: Filename in the HF repository

        Returns:
            Path to the local file
        """
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.info(f"✅ Using existing local file: {local_path}")
            return local_path

        logger.info(f"📥 File not found locally, downloading {repo_filename} from {self.repo_id}")
        try:
            # Download to temp location first, then move to avoid repo structure
            from huggingface_hub import hf_hub_download
            temp_file = hf_hub_download(
                repo_id=self.repo_id,
                filename=repo_filename,
                cache_dir=None  # Use default cache temporarily
            )

            # Copy to our local structure
            import shutil
            shutil.copy2(temp_file, local_path)
            logger.info(f"💾 Downloaded and copied to: {local_path}")

        except Exception as e:
            logger.error(f"❌ Failed to download {repo_filename}: {e}")
            raise

        return local_path

    def get_config(self) -> Dict[str, Any]:
        """Get config.json, downloading if necessary.

        Returns:
            Loaded configuration dictionary
        """
        config_path = self.config_dir / 'config.json'
        local_file = self._download_if_missing(config_path, 'config.json')

        with open(local_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        logger.debug(f"Loaded config from: {local_file}")
        return config

    def get_model(self, repo_id: Optional[str] = None) -> Path:
        """Get model file, downloading if necessary.

        Args:
            repo_id: Optional repository ID, defaults to instance repo_id

        Returns:
            Path to the local model file
        """
        repo_id = repo_id or self.repo_id

        if repo_id not in self.model_files:
            raise ValueError(f"Unknown repository: {repo_id}. Known repos: {list(self.model_files.keys())}")

        model_filename = self.model_files[repo_id]
        model_path = self.models_dir / model_filename

        return self._download_if_missing(model_path, model_filename)

    def get_voice(self, voice: str) -> Path:
        """Get voice file, downloading if necessary.

        Args:
            voice: Voice name (e.g., 'en-us', 'af', 'ar')

        Returns:
            Path to the local voice file
        """
        # Handle both 'voice' and 'voice.pt' formats
        if not voice.endswith('.pt'):
            voice_filename = f"{voice}.pt"
        else:
            voice_filename = voice
            voice = voice[:-3]  # Remove .pt for local storage

        voice_path = self.voices_dir / voice_filename
        repo_filename = f"voices/{voice_filename}"

        return self._download_if_missing(voice_path, repo_filename)

    def list_local_voices(self) -> list[str]:
        """List all locally available voice files.

        Returns:
            List of voice names (without .pt extension)
        """
        if not self.voices_dir.exists():
            return []

        voices = []
        for voice_file in self.voices_dir.glob('*.pt'):
            if voice_file.stat().st_size > 0:  # Only non-empty files
                voices.append(voice_file.stem)  # Remove .pt extension

        return sorted(voices)

    def has_local_model(self, repo_id: Optional[str] = None) -> bool:
        """Check if model is available locally without downloading.

        Args:
            repo_id: Optional repository ID, defaults to instance repo_id

        Returns:
            True if model exists locally and is non-empty
        """
        repo_id = repo_id or self.repo_id
        if repo_id not in self.model_files:
            return False

        model_filename = self.model_files[repo_id]
        model_path = self.models_dir / model_filename
        return model_path.exists() and model_path.stat().st_size > 0

    def has_local_voice(self, voice: str) -> bool:
        """Check if voice is available locally without downloading.

        Args:
            voice: Voice name (e.g., 'en-us', 'af', 'ar')

        Returns:
            True if voice exists locally and is non-empty
        """
        if not voice.endswith('.pt'):
            voice_filename = f"{voice}.pt"
        else:
            voice_filename = voice

        voice_path = self.voices_dir / voice_filename
        return voice_path.exists() and voice_path.stat().st_size > 0

    def has_local_config(self) -> bool:
        """Check if config is available locally without downloading.

        Returns:
            True if config exists locally and is non-empty
        """
        config_path = self.config_dir / 'config.json'
        return config_path.exists() and config_path.stat().st_size > 0

    def clear_cache(self, component: Optional[str] = None) -> None:
        """Clear local cache for specified component or all components.

        Args:
            component: 'config', 'models', 'voices', or None for all
        """
        if component == 'config' or component is None:
            for file in self.config_dir.glob('*'):
                file.unlink()
                logger.info(f"Removed: {file}")

        if component == 'models' or component is None:
            for file in self.models_dir.glob('*'):
                file.unlink()
                logger.info(f"Removed: {file}")

        if component == 'voices' or component is None:
            for file in self.voices_dir.glob('*'):
                file.unlink()
                logger.info(f"Removed: {file}")

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about local storage usage.

        Returns:
            Dictionary with storage statistics
        """
        def get_dir_size(path: Path) -> int:
            """Get total size of directory in bytes."""
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

        def count_files(path: Path) -> int:
            """Count files in directory."""
            return len([f for f in path.rglob('*') if f.is_file()])

        info = {
            'base_path': str(self.base_path),
            'repo_id': self.repo_id,
            'config': {
                'path': str(self.config_dir),
                'files': count_files(self.config_dir),
                'size_mb': get_dir_size(self.config_dir) / (1024 * 1024),
            },
            'models': {
                'path': str(self.models_dir),
                'files': count_files(self.models_dir),
                'size_mb': get_dir_size(self.models_dir) / (1024 * 1024),
            },
            'voices': {
                'path': str(self.voices_dir),
                'files': count_files(self.voices_dir),
                'size_mb': get_dir_size(self.voices_dir) / (1024 * 1024),
                'available_voices': self.list_local_voices(),
            }
        }

        return info


# Convenience functions for easy usage
def get_config(base_path: Optional[str] = None, repo_id: str = 'hexgrad/Kokoro-82M') -> Dict[str, Any]:
    """Convenience function to get config.json."""
    manager = LocalModelManager(base_path, repo_id)
    return manager.get_config()


def get_model(base_path: Optional[str] = None, repo_id: str = 'hexgrad/Kokoro-82M') -> Path:
    """Convenience function to get model file."""
    manager = LocalModelManager(base_path, repo_id)
    return manager.get_model()


def get_voice(voice: str, base_path: Optional[str] = None, repo_id: str = 'hexgrad/Kokoro-82M') -> Path:
    """Convenience function to get voice file."""
    manager = LocalModelManager(base_path, repo_id)
    return manager.get_voice(voice)


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = LocalModelManager()

    # Get files (will download if missing)
    config = manager.get_config()
    model_path = manager.get_model()
    voice_path = manager.get_voice('en-us')

    # Show storage info
    info = manager.get_storage_info()
    print("Storage info:", info)

    # List available voices
    voices = manager.list_local_voices()
    print("Available voices:", voices)