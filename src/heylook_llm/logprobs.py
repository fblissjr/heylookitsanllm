# src/heylook_llm/logprobs.py
"""
Logprobs processing for OpenAI-compatible API responses.

Converts mlx-lm's GenerationResponse logprobs (full vocabulary log-softmax)
to OpenAI's structured logprobs format.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

# Try to import mlx for array handling
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@dataclass
class TokenLogprob:
    """OpenAI-compatible token log probability."""
    token: str
    token_id: int  # Extension: include token ID for programmatic access
    logprob: float
    bytes: Optional[List[int]] = None  # UTF-8 bytes of token

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible dict."""
        result = {
            "token": self.token,
            "logprob": self.logprob,
        }
        if self.bytes is not None:
            result["bytes"] = self.bytes
        # Extension: include token_id
        result["token_id"] = self.token_id
        return result


@dataclass
class ContentLogprob:
    """Logprob entry for a single generated token with top alternatives."""
    token: str
    token_id: int
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: List[TokenLogprob] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible dict."""
        result = {
            "token": self.token,
            "logprob": self.logprob,
            "top_logprobs": [t.to_dict() for t in self.top_logprobs],
        }
        if self.bytes is not None:
            result["bytes"] = self.bytes
        # Extension: include token_id
        result["token_id"] = self.token_id
        return result


@dataclass
class LogprobsResult:
    """Complete logprobs result for a response."""
    content: List[ContentLogprob] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible dict."""
        return {
            "content": [c.to_dict() for c in self.content]
        }


class LogprobsCollector:
    """Collects logprobs during generation for later formatting."""

    def __init__(self, tokenizer, top_logprobs: int = 5):
        """
        Initialize logprobs collector.

        Args:
            tokenizer: The tokenizer for decoding tokens
            top_logprobs: Number of top alternatives to include (0-20)
        """
        self.tokenizer = tokenizer
        self.top_logprobs = min(max(top_logprobs, 0), 20)
        self.content: List[ContentLogprob] = []

    def add_token(self, token_id: int, logprobs_array) -> None:
        """
        Add a generated token with its logprobs.

        Args:
            token_id: The generated token ID
            logprobs_array: mlx array of log probabilities for full vocabulary
        """
        if logprobs_array is None:
            return

        try:
            # Convert mlx array to numpy for processing
            if HAS_MLX and isinstance(logprobs_array, mx.array):
                # Ensure we have a 1D array
                if len(logprobs_array.shape) > 1:
                    logprobs_array = logprobs_array.squeeze()
                logprobs_np = logprobs_array.tolist()
            else:
                logprobs_np = list(logprobs_array)

            # Get the logprob of the selected token
            selected_logprob = float(logprobs_np[token_id])

            # Decode the selected token
            token_text = self._decode_token(token_id)
            token_bytes = self._get_token_bytes(token_text)

            # Get top-k alternatives if requested
            top_alternatives = []
            if self.top_logprobs > 0:
                top_alternatives = self._get_top_logprobs(logprobs_np)

            # Create the content logprob entry
            entry = ContentLogprob(
                token=token_text,
                token_id=token_id,
                logprob=selected_logprob,
                bytes=token_bytes,
                top_logprobs=top_alternatives
            )
            self.content.append(entry)

        except (IndexError, ValueError, RuntimeError, TypeError) as e:
            logging.warning(f"Failed to process logprobs for token {token_id}: {e}", exc_info=True)

    def _decode_token(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        try:
            # Use the tokenizer to decode
            return self.tokenizer.decode([token_id])
        except Exception:
            return f"<token_{token_id}>"

    def _get_token_bytes(self, token_text: str) -> List[int]:
        """Get UTF-8 bytes for a token."""
        try:
            return list(token_text.encode('utf-8'))
        except Exception:
            return []

    def _get_top_logprobs(self, logprobs: List[float]) -> List[TokenLogprob]:
        """Get top-k tokens by log probability."""
        # Create (index, logprob) pairs and sort by logprob descending
        indexed = [(i, lp) for i, lp in enumerate(logprobs)]
        indexed.sort(key=lambda x: x[1], reverse=True)

        top_tokens = []
        for token_id, logprob in indexed[:self.top_logprobs]:
            token_text = self._decode_token(token_id)
            token_bytes = self._get_token_bytes(token_text)
            top_tokens.append(TokenLogprob(
                token=token_text,
                token_id=token_id,
                logprob=float(logprob),
                bytes=token_bytes
            ))

        return top_tokens

    def get_result(self) -> LogprobsResult:
        """Get the final logprobs result."""
        return LogprobsResult(content=self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Get the result as an OpenAI-compatible dict."""
        return self.get_result().to_dict()


class StreamingLogprobsCollector(LogprobsCollector):
    """
    Streaming variant that yields logprobs per-token for SSE responses.
    """

    def add_token_and_get_delta(self, token_id: int, logprobs_array) -> Optional[Dict[str, Any]]:
        """
        Add a token and immediately return its logprobs delta.

        Args:
            token_id: The generated token ID
            logprobs_array: mlx array of log probabilities

        Returns:
            Dict with the logprobs delta for this token, or None on error
        """
        initial_len = len(self.content)
        self.add_token(token_id, logprobs_array)

        if len(self.content) > initial_len:
            # Return the last added entry as a delta
            return {"content": [self.content[-1].to_dict()]}
        return None
