"""Language processor factory implementing the strategy pattern for G2P processing."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Tuple, Union, Optional, Callable
from loguru import logger
from misaki import en, espeak  # type: ignore[import-untyped]
from .config import LANG_CODES


class LanguageProcessor(ABC):
    """Abstract base class for language-specific G2P processors."""

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the language processor with specific configuration."""
        pass

    @abstractmethod
    def __call__(self, text: str) -> Union[str, Tuple[str, Any]]:
        """Process text into phonemes."""
        pass


class EnglishProcessor(LanguageProcessor):
    """Processor for American English text."""

    def __init__(self, trf: bool = False, **kwargs):
        try:
            fallback = espeak.EspeakFallback(british=False)
        except Exception as e:
            logger.warning("EspeakFallback not Enabled: OOD words will be skipped")
            logger.warning({str(e)})
            fallback = None
        self.g2p = en.G2P(trf=trf, british=False, fallback=fallback, unk='')

    def __call__(self, text: str) -> Tuple[str, Any]:
        return self.g2p(text)


class BritishEnglishProcessor(LanguageProcessor):
    """Processor for British English text."""

    def __init__(self, trf: bool = False, **kwargs):
        try:
            fallback = espeak.EspeakFallback(british=True)
        except Exception as e:
            logger.warning("EspeakFallback not Enabled: OOD words will be skipped")
            logger.warning({str(e)})
            fallback = None
        self.g2p = en.G2P(trf=trf, british=True, fallback=fallback, unk='')

    def __call__(self, text: str) -> Tuple[str, Any]:
        return self.g2p(text)


class JapaneseProcessor(LanguageProcessor):
    """Processor for Japanese text."""

    def __init__(self, **kwargs):
        try:
            from misaki import ja
            self.g2p = ja.JAG2P()
        except ImportError:
            logger.error("You need to `pip install misaki[ja]` to use lang_code='j'")
            raise

    def __call__(self, text: str) -> Union[str, Tuple[str, Any]]:
        return self.g2p(text)


class ChineseProcessor(LanguageProcessor):
    """Processor for Mandarin Chinese text."""

    def __init__(self, repo_id: str, en_callable: Optional[Callable[[str], str]] = None, **kwargs):
        try:
            from misaki import zh
            version = None if repo_id.endswith('/Kokoro-82M') else '1.1'
            self.g2p = zh.ZHG2P(version=version, en_callable=en_callable)
        except ImportError:
            logger.error("You need to `pip install misaki[zh]` to use lang_code='z'")
            raise

    def __call__(self, text: str) -> Union[str, Tuple[str, Any]]:
        return self.g2p(text)


class EspeakProcessor(LanguageProcessor):
    """Generic processor for languages supported by eSpeak-NG."""

    def __init__(self, language: str, **kwargs):
        logger.warning(f"Using EspeakG2P(language='{language}'). Chunking logic not yet implemented, so long texts may be truncated unless you split them with '\\n'.")
        self.g2p = espeak.EspeakG2P(language=language)

    def __call__(self, text: str) -> Union[str, Tuple[str, Any]]:
        return self.g2p(text)


class LanguageProcessorFactory:
    """Factory for creating language-specific processors using the strategy pattern."""

    _processors: Dict[str, Type[LanguageProcessor]] = {
        'a': EnglishProcessor,
        'b': BritishEnglishProcessor,
        'j': JapaneseProcessor,
        'z': ChineseProcessor,
    }

    _espeak_languages = {
        'e': 'es',
        'f': 'fr-fr',
        'h': 'hi',
        'i': 'it',
        'p': 'pt-br',
    }

    @classmethod
    def create_processor(cls, lang_code: str, **kwargs) -> LanguageProcessor:
        """Create a language processor for the given language code.

        Args:
            lang_code: Language code (e.g., 'a', 'b', 'j', 'z', 'e', etc.)
            **kwargs: Additional arguments passed to the processor constructor

        Returns:
            Configured language processor instance

        Raises:
            ValueError: If the language code is not supported
        """
        if lang_code in cls._processors:
            return cls._processors[lang_code](**kwargs)
        elif lang_code in cls._espeak_languages:
            language = cls._espeak_languages[lang_code]
            return EspeakProcessor(language=language, **kwargs)
        else:
            raise ValueError(f"Unsupported language code: {lang_code}")

    @classmethod
    def register_processor(cls, lang_code: str, processor_class: Type[LanguageProcessor]):
        """Register a new language processor.

        Args:
            lang_code: Language code to register
            processor_class: Processor class to use for this language
        """
        cls._processors[lang_code] = processor_class

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """Get a mapping of supported language codes to their descriptions."""
        return LANG_CODES.copy()