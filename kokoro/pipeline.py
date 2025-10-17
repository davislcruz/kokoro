from .model import KModel
from .language_processors import LanguageProcessorFactory
from .voice_manager import VoiceManager
from .text_processor import TextProcessor
from .config import ALIASES, LANG_CODES, DEFAULT_REPO_ID, SAMPLE_RATE, MAX_PHONEME_LENGTH, DEFAULT_CHUNK_SIZE
from dataclasses import dataclass
from loguru import logger
try:
    from misaki import en, espeak  # type: ignore[import-untyped]
    MISAKI_AVAILABLE = True
except ImportError:
    MISAKI_AVAILABLE = False
    en = None
    espeak = None
from typing import Callable, Generator, List, Optional, Tuple, Union, Dict, Any
import re
import torch
import os

class KPipeline:
    '''
    KPipeline is a language-aware support class with 2 main responsibilities:
    1. Perform language-specific G2P, mapping (and chunking) text -> phonemes
    2. Manage and store voices, lazily downloaded from HF if needed

    You are expected to have one KPipeline per language. If you have multiple
    KPipelines, you should reuse one KModel instance across all of them.

    KPipeline is designed to work with a KModel, but this is not required.
    There are 2 ways to pass an existing model into a pipeline:
    1. On init: us_pipeline = KPipeline(lang_code='a', model=model)
    2. On call: us_pipeline(text, voice, model=model)

    By default, KPipeline will automatically initialize its own KModel. To
    suppress this, construct a "quiet" KPipeline with model=False.

    A "quiet" KPipeline yields (graphemes, phonemes, None) without generating
    any audio. You can use this to phonemize and chunk your text in advance.

    A "loud" KPipeline _with_ a model yields (graphemes, phonemes, audio).
    '''
    def __init__(
        self,
        lang_code: str,
        repo_id: Optional[str] = None,
        model: Union[KModel, bool] = True,
        trf: bool = False,
        en_callable: Optional[Callable[[str], str]] = None,
        device: Optional[str] = None
    ):
        """Initialize a KPipeline.
        
        Args:
            lang_code: Language code for G2P processing
            model: KModel instance, True to create new model, False for no model
            trf: Whether to use transformer-based G2P
            device: Override default device selection ('cuda' or 'cpu', or None for auto)
                   If None, will auto-select cuda if available
                   If 'cuda' and not available, will explicitly raise an error
        """
        if repo_id is None:
            repo_id = DEFAULT_REPO_ID
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        lang_code = lang_code.lower()
        lang_code = ALIASES.get(lang_code, lang_code)
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)
        self.lang_code = lang_code
        self.model = None
        if isinstance(model, KModel):
            self.model = model
        elif model:
            device = self._select_device(device)
            try:
                self.model = KModel(repo_id=repo_id).to(device).eval()
            except RuntimeError as e:
                if device == 'cuda':
                    raise RuntimeError(f"""Failed to initialize model on CUDA: {e}.
                                       Try setting device='cpu' or check CUDA installation.""")
                raise

        # Initialize composed components
        self.voice_manager = VoiceManager(repo_id=repo_id, lang_code=lang_code)
        self.text_processor = TextProcessor(lang_code=lang_code)

        # Use the language processor factory
        self.g2p = LanguageProcessorFactory.create_processor(
            lang_code=lang_code,
            trf=trf,
            repo_id=repo_id,
            en_callable=en_callable
        )

    def _select_device(self, requested_device: Optional[str]) -> str:
        """Select appropriate device with clear validation.

        Args:
            requested_device: User-requested device ('cuda', 'mps', 'cpu', or None for auto)

        Returns:
            Validated device string

        Raises:
            RuntimeError: If requested device is not available
        """
        if requested_device == 'cuda':
            return self._validate_cuda_device()
        elif requested_device == 'mps':
            return self._validate_mps_device()
        elif requested_device is None:
            return self._auto_select_device()
        return requested_device

    def _validate_cuda_device(self) -> str:
        """Validate CUDA device availability."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return 'cuda'

    def _validate_mps_device(self) -> str:
        """Validate MPS device availability."""
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1':
            raise RuntimeError("MPS requested but fallback not enabled")
        return 'mps'

    def _auto_select_device(self) -> str:
        """Automatically select the best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1' and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def load_single_voice(self, voice: str):
        """Delegate to voice manager."""
        return self.voice_manager.load_single_voice(voice)

    def load_voice(self, voice: Union[str, torch.FloatTensor], delimiter: str = ",") -> torch.FloatTensor:
        """Delegate to voice manager."""
        return self.voice_manager.load_voice(voice, delimiter)

    def load_voice_for_device(self, voice: Union[str, torch.FloatTensor], device: torch.device) -> torch.FloatTensor:
        """Delegate to voice manager."""
        return self.voice_manager.load_voice_for_device(voice, device)

    @staticmethod
    def tokens_to_ps(tokens: List[en.MToken]) -> str:
        """Delegate to text processor."""
        return TextProcessor.tokens_to_ps(tokens)

    @staticmethod
    def waterfall_last(
        tokens: List[en.MToken],
        next_count: int,
        waterfall: List[str] = ['!.?…', ':;', ',—'],
        bumps: List[str] = [')', '"']
    ) -> int:
        """Delegate to text processor."""
        return TextProcessor.waterfall_last(tokens, next_count, waterfall, bumps)

    @staticmethod
    def tokens_to_text(tokens: List[en.MToken]) -> str:
        """Delegate to text processor."""
        return TextProcessor.tokens_to_text(tokens)

    def en_tokenize(
        self,
        tokens: List[en.MToken]
    ) -> Generator[Tuple[str, str, List[en.MToken]], None, None]:
        """Delegate to text processor."""
        yield from self.text_processor.en_tokenize(tokens)

    @staticmethod
    def infer(
        model: KModel,
        ps: str,
        pack: torch.FloatTensor,
        speed: Union[float, Callable[[int], float]] = 1
    ) -> KModel.Output:
        if callable(speed):
            speed = speed(len(ps))
        return model(ps, pack[len(ps)-1], speed, return_output=True)

    def _generate_audio_output(self, model: Optional[KModel], ps: str, pack: Optional[torch.FloatTensor], speed: Union[float, Callable[[int], float]]) -> Optional[KModel.Output]:
        """Generate audio output with consistent null checks.

        Args:
            model: KModel instance for inference
            ps: Phoneme string
            pack: Voice tensor pack
            speed: Speed modifier for synthesis

        Returns:
            KModel.Output if model and pack are provided, None otherwise
        """
        return KPipeline.infer(model, ps, pack, speed) if model and pack is not None else None

    def generate_from_tokens(
        self,
        tokens: Union[str, List[en.MToken]],
        voice: str,
        speed: float = 1,
        model: Optional[KModel] = None
    ) -> Generator['KPipeline.Result', None, None]:
        """Generate audio from either raw phonemes or pre-processed tokens.
        
        Args:
            tokens: Either a phoneme string or list of pre-processed MTokens
            voice: The voice to use for synthesis
            speed: Speech speed modifier (default: 1)
            model: Optional KModel instance (uses pipeline's model if not provided)
        
        Yields:
            KPipeline.Result containing the input tokens and generated audio
            
        Raises:
            ValueError: If no voice is provided or token sequence exceeds model limits
        """
        model = model or self.model
        if model and voice is None:
            raise ValueError('Specify a voice: pipeline.generate_from_tokens(..., voice="af_heart")')

        pack = self.load_voice_for_device(voice, model.device) if model and voice is not None else None

        # Handle raw phoneme string
        if isinstance(tokens, str):
            logger.debug("Processing phonemes from raw string")
            if len(tokens) > MAX_PHONEME_LENGTH:
                raise ValueError(f'Phoneme string too long: {len(tokens)} > {MAX_PHONEME_LENGTH}')
            output = self._generate_audio_output(model, tokens, pack, speed)  # type: ignore[arg-type]
            yield self.Result(graphemes='', phonemes=tokens, output=output)
            return
        
        logger.debug("Processing MTokens")
        # Handle pre-processed tokens
        for gs, ps, tks in self.en_tokenize(tokens):
            if not ps:
                continue
            elif len(ps) > MAX_PHONEME_LENGTH:
                logger.warning(f"Unexpected len(ps) == {len(ps)} > {MAX_PHONEME_LENGTH} and ps == '{ps}'")
                logger.warning(f"Truncating to {MAX_PHONEME_LENGTH} characters")
                ps = ps[:MAX_PHONEME_LENGTH]
            output = self._generate_audio_output(model, ps, pack, speed)  # type: ignore[arg-type]
            if output is not None and output.pred_dur is not None:
                KPipeline.join_timestamps(tks, output.pred_dur)
            yield self.Result(graphemes=gs, phonemes=ps, tokens=tks, output=output)

    @staticmethod
    def join_timestamps(tokens: List[en.MToken], pred_dur: torch.LongTensor):
        """Delegate to text processor."""
        TextProcessor.join_timestamps(tokens, pred_dur)

    @dataclass
    class Result:
        graphemes: str
        phonemes: str
        tokens: Optional[List[en.MToken]] = None
        output: Optional[KModel.Output] = None
        text_index: Optional[int] = None

        @property
        def audio(self) -> Optional[torch.FloatTensor]:
            return None if self.output is None else self.output.audio

        @property
        def pred_dur(self) -> Optional[torch.LongTensor]:
            return None if self.output is None else self.output.pred_dur

        ### MARK: BEGIN BACKWARD COMPAT ###
        def __iter__(self):
            yield self.graphemes
            yield self.phonemes
            yield self.audio

        def __getitem__(self, index):
            return [self.graphemes, self.phonemes, self.audio][index]

        def __len__(self):
            return 3
        #### MARK: END BACKWARD COMPAT ####

    def __call__(
        self,
        text: Union[str, List[str]],
        voice: Optional[str] = None,
        speed: Union[float, Callable[[int], float]] = 1,
        split_pattern: Optional[str] = r'\n+',
        model: Optional[KModel] = None
    ) -> Generator['KPipeline.Result', None, None]:
        """Process text through the TTS pipeline.

        Args:
            text: Text to process (string or list of strings)
            voice: Voice to use for synthesis
            speed: Speed modifier for synthesis
            split_pattern: Pattern to split text segments
            model: Optional KModel instance

        Yields:
            KPipeline.Result objects containing processed text and audio
        """
        model = model or self.model
        if model and voice is None:
            raise ValueError('Specify a voice: en_us_pipeline(text="Hello world!", voice="af_heart")')
        pack = self.load_voice_for_device(voice, model.device) if model and voice is not None else None

        # Convert input to list of segments
        text_segments = self.text_processor.prepare_text_segments(text, split_pattern)

        # Process each segment
        for graphemes_index, graphemes in enumerate(text_segments):
            if not graphemes.strip():  # Skip empty segments
                continue

            if self.lang_code in 'ab':
                yield from self._process_english_text(graphemes, model, pack, speed, graphemes_index)
            else:
                yield from self._process_non_english_text(graphemes, model, pack, speed, graphemes_index)


    def _process_english_text(
        self,
        graphemes: str,
        model: Optional[KModel],
        pack: Optional[torch.FloatTensor],
        speed: Union[float, Callable[[int], float]],
        graphemes_index: int
    ) -> Generator['KPipeline.Result', None, None]:
        """Process English text through G2P and tokenization.

        Args:
            graphemes: Text to process
            model: KModel instance
            pack: Voice tensor
            speed: Speed modifier
            graphemes_index: Index of current text segment

        Yields:
            KPipeline.Result objects for each chunk
        """
        logger.debug(f"Processing English text: {graphemes[:50]}{'...' if len(graphemes) > 50 else ''}")
        _, tokens = self.g2p(graphemes)
        for gs, ps, tks in self.en_tokenize(tokens):
            if not ps:
                continue
            elif len(ps) > MAX_PHONEME_LENGTH:
                logger.warning(f"Unexpected len(ps) == {len(ps)} > {MAX_PHONEME_LENGTH} and ps == '{ps}'")
                ps = ps[:MAX_PHONEME_LENGTH]
            output = self._generate_audio_output(model, ps, pack, speed)  # type: ignore[arg-type]
            if output is not None and output.pred_dur is not None:
                KPipeline.join_timestamps(tks, output.pred_dur)
            yield self.Result(graphemes=gs, phonemes=ps, tokens=tks, output=output, text_index=graphemes_index)

    def _process_non_english_text(
        self,
        graphemes: str,
        model: Optional[KModel],
        pack: Optional[torch.FloatTensor],
        speed: Union[float, Callable[[int], float]],
        graphemes_index: int
    ) -> Generator['KPipeline.Result', None, None]:
        """Process non-English text with chunking logic.

        Args:
            graphemes: Text to process
            model: KModel instance
            pack: Voice tensor
            speed: Speed modifier
            graphemes_index: Index of current text segment

        Yields:
            KPipeline.Result objects for each chunk
        """
        chunks = self.text_processor.chunk_text(graphemes)

        # Process each chunk
        for chunk in chunks:
            if not chunk.strip():
                continue

            # Handle different g2p return types
            g2p_result = self.g2p(chunk)
            if isinstance(g2p_result, tuple):
                ps, _ = g2p_result
            else:
                ps = g2p_result
            if not ps:
                continue
            elif len(ps) > MAX_PHONEME_LENGTH:
                logger.warning(f'Truncating len(ps) == {len(ps)} > {MAX_PHONEME_LENGTH}')
                ps = ps[:MAX_PHONEME_LENGTH]

            output = self._generate_audio_output(model, ps, pack, speed)  # type: ignore[arg-type]
            yield self.Result(graphemes=chunk, phonemes=ps, output=output, text_index=graphemes_index)

    # Convenience methods for accessing component functionality
    def clear_voice_cache(self):
        """Clear voice cache to free memory."""
        self.voice_manager.clear_cache()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get voice cache statistics."""
        return self.voice_manager.get_cache_stats()

    # Maintain backward compatibility with old voice attributes
    @property
    def voices(self) -> Dict[str, torch.Tensor]:
        """Access to voice cache for backward compatibility."""
        return self.voice_manager.voices

    @property
    def _device_voice_cache(self) -> Dict[str, torch.Tensor]:
        """Access to device voice cache for backward compatibility."""
        return self.voice_manager._device_voice_cache

