# src/heylook_llm/schema/batch.py
#
# Batch processing models for the new API.
# Merges the two overlapping batch systems (BatchChatRequest and the
# /v1/batch/process endpoint) into a single, clean interface.

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from heylook_llm.schema.messages import MessageCreateRequest
from heylook_llm.schema.responses import MessageResponse, Usage


class BatchRequest(BaseModel):
    """Request body for POST /v1/batch.

    Submit multiple message requests for parallel processing.
    All requests use the same model but can have different prompts
    and parameters.
    """
    model: Optional[str] = Field(
        None, description="Model ID. Applied to all requests unless overridden per-request."
    )
    requests: List[MessageCreateRequest]

    # Batch processing tuning
    completion_batch_size: Optional[int] = Field(
        32, description="Max concurrent generations"
    )
    prefill_batch_size: Optional[int] = Field(
        8, description="Max prefill parallelism"
    )
    prefill_step_size: Optional[int] = Field(
        2048, description="Chunk size for prefill"
    )

    @field_validator("requests")
    @classmethod
    def validate_requests_not_empty(cls, v: List[MessageCreateRequest]) -> List[MessageCreateRequest]:
        if not v:
            raise ValueError("Requests list cannot be empty")
        if len(v) < 2:
            raise ValueError("Batch requests must contain at least 2 requests")
        return v


class BatchStats(BaseModel):
    """Aggregate statistics for a batch run."""
    total_requests: int
    completed: int = 0
    failed: int = 0
    elapsed_seconds: float
    throughput_req_per_sec: float
    throughput_tok_per_sec: float
    prefill_time: Optional[float] = None
    generation_time: Optional[float] = None
    memory_peak_mb: Optional[float] = None


class BatchResult(BaseModel):
    """Result for a single request within a batch."""
    index: int = Field(..., description="Index of the original request")
    status: Literal["completed", "failed"] = "completed"
    response: Optional[MessageResponse] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    """Response from POST /v1/batch."""
    id: str = Field(..., description="Batch job ID, prefixed 'batch_'")
    type: Literal["batch_result"] = "batch_result"
    model: str
    results: List[BatchResult]
    stats: BatchStats
    usage: Usage = Field(
        default_factory=Usage,
        description="Aggregate token usage across all requests",
    )
