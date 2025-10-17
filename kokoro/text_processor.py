"""Text processing utilities for handling language-specific text processing and chunking."""

from typing import List, Union, Optional, Generator, Tuple, Any
import re
import torch
from loguru import logger
try:
    from misaki import en  # type: ignore[import-untyped]
    MISAKI_AVAILABLE = True
except ImportError:
    MISAKI_AVAILABLE = False
    en = None
from .config import DEFAULT_CHUNK_SIZE, MAGIC_DIVISOR, MAX_PHONEME_LENGTH, SAMPLE_RATE


class TextProcessor:
    """Handles language-specific text processing and chunking logic."""

    def __init__(self, lang_code: str):
        """Initialize text processor.

        Args:
            lang_code: Language code for processing logic
        """
        self.lang_code = lang_code

    def prepare_text_segments(self, text: Union[str, List[str]], split_pattern: Optional[str]) -> List[str]:
        """Prepare text segments for processing.

        Args:
            text: Input text (string or list)
            split_pattern: Pattern to split text segments

        Returns:
            List of text segments
        """
        if isinstance(text, str):
            return re.split(split_pattern, text.strip()) if split_pattern else [text]
        return text

    def chunk_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """Split long text into smaller chunks using sentence boundaries when possible.

        Args:
            text: Text to chunk
            chunk_size: Target size for each chunk

        Returns:
            List of text chunks
        """
        chunks = []

        # Try to split on sentence boundaries first
        sentences = re.split(r'([.!?]+)', text)
        current_chunk = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            # Add the punctuation back if it exists
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        # If no chunks were created (no sentence boundaries), fall back to character-based chunking
        if not chunks:
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        return chunks

    @staticmethod
    def tokens_to_ps(tokens: List[en.MToken]) -> str:
        """Convert tokens to phoneme string with optimized string operations.

        Args:
            tokens: List of MTokens to convert

        Returns:
            Concatenated phoneme string
        """
        parts = []
        for t in tokens:
            parts.append(t.phonemes)
            if t.whitespace:
                parts.append(' ')
        return ''.join(parts).strip()

    @staticmethod
    def tokens_to_text(tokens: List[en.MToken]) -> str:
        """Convert tokens to text string.

        Args:
            tokens: List of MTokens to convert

        Returns:
            Concatenated text string
        """
        return ''.join(t.text + t.whitespace for t in tokens).strip()

    @staticmethod
    def waterfall_last(
        tokens: List[en.MToken],
        next_count: int,
        waterfall: List[str] = ['!.?…', ':;', ',—'],
        bumps: List[str] = [')', '"']
    ) -> int:
        """Find optimal chunking point based on punctuation hierarchy.

        Uses a waterfall approach to find the best place to split text,
        prioritizing stronger punctuation marks over weaker ones.

        Args:
            tokens: List of MTokens to search for split points
            next_count: Target character count for next chunk
            waterfall: Punctuation priority list (highest to lowest priority)
                      Default: ['!.?…', ':;', ',—'] prioritizes sentence endings
            bumps: Additional characters to include after punctuation
                   Default: [')', '"'] for closing quotes and parentheses

        Returns:
            Index of optimal chunking point within tokens list
            Returns len(tokens) if no good split point is found
        """
        for w in waterfall:
            z = next((i for i, t in reversed(list(enumerate(tokens))) if t.phonemes in set(w)), None)
            if z is None:
                continue
            z += 1
            if z < len(tokens) and tokens[z].phonemes in bumps:
                z += 1
            if next_count - len(TextProcessor.tokens_to_ps(tokens[:z])) <= MAX_PHONEME_LENGTH:
                return z
        return len(tokens)

    def en_tokenize(
        self,
        tokens: List[en.MToken]
    ) -> Generator[Tuple[str, str, List[en.MToken]], None, None]:
        """Tokenize English text with intelligent chunking.

        Args:
            tokens: List of MTokens from G2P processing

        Yields:
            Tuples of (graphemes, phonemes, token_chunk)
        """
        tks: List[en.MToken] = []
        pcount = 0

        for t in tokens:
            # American English: ɾ => T (commented out for now)
            t.phonemes = '' if t.phonemes is None else t.phonemes
            next_ps = t.phonemes + (' ' if t.whitespace else '')
            next_pcount = pcount + len(next_ps.rstrip())

            if next_pcount > MAX_PHONEME_LENGTH:
                z = self.waterfall_last(tks, next_pcount)
                text = self.tokens_to_text(tks[:z])
                logger.debug(f"Chunking text at {z}: '{text[:30]}{'...' if len(text) > 30 else ''}'")
                ps = self.tokens_to_ps(tks[:z])
                yield text, ps, tks[:z]
                tks = tks[z:]
                pcount = len(self.tokens_to_ps(tks))
                if not tks:
                    next_ps = next_ps.lstrip()

            tks.append(t)
            pcount += len(next_ps)

        if tks:
            text = self.tokens_to_text(tks)
            ps = self.tokens_to_ps(tks)
            yield ''.join(text).strip(), ''.join(ps).strip(), tks

    @staticmethod
    def join_timestamps(tokens: List[en.MToken], pred_dur: torch.LongTensor):
        """Join predicted durations with tokens to calculate timing information.

        This method maps the model's predicted phoneme durations to token-level
        timestamps, enabling word-level timing alignment for the generated audio.

        Args:
            tokens: List of MTokens to annotate with timing information
            pred_dur: Predicted duration tensor from the model containing
                     phoneme-level timing information

        Note:
            - Modifies tokens in-place by setting start_ts and end_ts attributes
            - Uses MAGIC_DIVISOR of 80 to convert from model frames to seconds
            - Handles both phoneme tokens and whitespace separately
            - Accounts for beginning-of-sequence and end-of-sequence tokens
        """

        # Multiply by 600 to go from pred_dur frames to sample_rate (24000)
        # Equivalent to dividing pred_dur frames by 40 to get timestamp in seconds
        # We will count nice round half-frames, so the divisor is 80
        if not tokens or len(pred_dur) < 3:
            # We expect at least 3: <bos>, token, <eos>
            return
        # We track 2 counts, measured in half-frames: (left, right)
        # This way we can cut space characters in half
        # TODO: Is -3 an appropriate offset?
        left = right = 2 * max(0, pred_dur[0].item() - 3)
        # Updates:
        # left = right + (2 * token_dur) + space_dur
        # right = left + space_dur
        i = 1
        for t in tokens:
            if i >= len(pred_dur)-1:
                break
            if not t.phonemes:
                if t.whitespace:
                    i += 1
                    left = right + pred_dur[i].item()
                    right = left + pred_dur[i].item()
                    i += 1
                continue
            j = i + len(t.phonemes)
            if j >= len(pred_dur):
                break
            t.start_ts = left / MAGIC_DIVISOR
            token_dur = pred_dur[i: j].sum().item()
            space_dur = pred_dur[j].item() if t.whitespace else 0
            left = right + (2 * token_dur) + space_dur
            t.end_ts = left / MAGIC_DIVISOR
            right = left + space_dur
            i = j + (1 if t.whitespace else 0)