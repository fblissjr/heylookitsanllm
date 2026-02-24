# src/heylook_llm/providers/mlx_batch_text.py
"""
Batch text generation processor wrapping mlx-lm's BatchGenerator.

This module provides heylookitsanllm-specific batch processing for text generation,
enabling 2-4x throughput improvements for concurrent text-only requests.

Architecture:
- Wraps mlx-lm's proven BatchGenerator implementation
- Handles variable-length prompts via left-padding
- Provides both blocking and streaming batch interfaces
- Integrates with heylookitsanllm monitoring and error handling
"""

import logging
import time
from typing import List, Optional, Union
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, wired_limit

# Import generation stream from mlx_provider
from .mlx_provider import generation_stream


@dataclass
class BatchResult:
    """Result for a single sequence in a batch."""
    uid: int
    text: str
    tokens: List[int]
    finish_reason: str
    prompt_tokens: int
    generation_tokens: int


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    batch_size: int
    total_time: float
    prefill_time: float
    generation_time: float
    throughput_req_per_sec: float
    throughput_tok_per_sec: float
    memory_peak_mb: float


class TextBatchProcessor:
    """
    Wrapper around mlx-lm's BatchGenerator for batch text generation.

    Provides heylookitsanllm-specific features:
    - Integration with our monitoring system
    - Request tracking and error handling
    - Metrics collection
    - Streaming and blocking interfaces

    Usage:
        processor = TextBatchProcessor(model, tokenizer)
        results = processor.process_batch(prompts, max_tokens=[50, 100, 75])
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_tokens: int = 512,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048
    ):
        """
        Initialize batch processor.

        Args:
            model: MLX model for generation
            tokenizer: Tokenizer for encoding/decoding
            max_tokens: Default max tokens per sequence
            completion_batch_size: Max concurrent generations
            prefill_batch_size: Max prefill parallelism
            prefill_step_size: Chunk size for memory efficiency
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        # Create BatchGenerator from mlx-lm
        self.generator = BatchGenerator(
            model=model,
            max_tokens=max_tokens,
            stop_tokens=tokenizer.eos_token_ids if hasattr(tokenizer, 'eos_token_ids') else set(),
            completion_batch_size=completion_batch_size,
            prefill_batch_size=prefill_batch_size,
            prefill_step_size=prefill_step_size
        )

        logging.info(
            f"[BATCH TEXT] Initialized with completion_batch_size={completion_batch_size}, "
            f"prefill_batch_size={prefill_batch_size}, prefill_step_size={prefill_step_size}"
        )

    def process_batch(
        self,
        prompts: List[List[int]],
        max_tokens: Optional[Union[List[int], int]] = None
    ) -> List[BatchResult]:
        """
        Process a batch of prompts and return completions.

        This is the blocking interface - waits for all sequences to complete.

        Args:
            prompts: List of tokenized prompts
            max_tokens: Per-prompt token limits (list or single int)

        Returns:
            List of BatchResult objects in same order as prompts
        """
        start_time = time.time()
        prefill_start = time.time()

        # Wrap in wired_limit for optimal Metal memory management
        with wired_limit(self.model, [generation_stream]):
            # Insert prompts into batch generator
            uids = self.generator.insert(prompts, max_tokens)
            prefill_time = time.time() - prefill_start

            # Collect results
            results = {uid: [] for uid in uids}
            finish_reasons = {}
            generation_start = time.time()

            # Generate tokens
            while responses := self.generator.next():
                for response in responses:
                    if response.finish_reason != "stop":
                        results[response.uid].append(response.token)
                    else:
                        finish_reasons[response.uid] = response.finish_reason

            generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        # Build BatchResult objects
        batch_results = []
        for i, uid in enumerate(uids):
            tokens = results[uid]
            text = self.tokenizer.decode(tokens)
            finish_reason = finish_reasons.get(uid, "length")

            batch_results.append(BatchResult(
                uid=uid,
                text=text,
                tokens=tokens,
                finish_reason=finish_reason,
                prompt_tokens=len(prompts[i]),
                generation_tokens=len(tokens)
            ))

        # Log statistics
        total_tokens = sum(r.prompt_tokens + r.generation_tokens for r in batch_results)
        stats = BatchStats(
            batch_size=len(prompts),
            total_time=total_time,
            prefill_time=prefill_time,
            generation_time=generation_time,
            throughput_req_per_sec=len(prompts) / total_time,
            throughput_tok_per_sec=total_tokens / total_time,
            memory_peak_mb=mx.get_peak_memory() / 1e6
        )

        logging.info(
            f"[BATCH TEXT] Completed batch: {stats.batch_size} requests in {stats.total_time:.2f}s "
            f"({stats.throughput_req_per_sec:.1f} req/s, {stats.throughput_tok_per_sec:.1f} tok/s)"
        )

        return batch_results

def should_use_batching(num_requests: int, same_model: bool, no_streaming: bool) -> bool:
    """
    Determine if batch processing should be used.

    Args:
        num_requests: Number of requests in batch
        same_model: Whether all requests use the same model
        no_streaming: Whether streaming is disabled for all requests

    Returns:
        True if batching should be used

    Batching is beneficial when:
    - Same model across all requests
    - No streaming required (batch is inherently blocking)
    - Sufficient batch size (>= 3 requests)
    """
    if not same_model:
        logging.debug("[BATCH TEXT] Skipping batch: different models")
        return False

    if not no_streaming:
        logging.debug("[BATCH TEXT] Skipping batch: streaming requested")
        return False

    if num_requests < 3:
        logging.debug(f"[BATCH TEXT] Skipping batch: too small ({num_requests} < 3)")
        return False

    logging.info(f"[BATCH TEXT] Using batch processing for {num_requests} requests")
    return True
