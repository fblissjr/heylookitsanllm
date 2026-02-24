# src/heylook_llm/schema/system.py
#
# System-level models: capabilities discovery, performance metrics,
# and server health. These consolidate the current /v1/capabilities,
# /v1/system/metrics, and /v1/performance endpoints.

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProviderCapability(BaseModel):
    """What a specific provider can do."""
    name: str = Field(..., description="Provider name (mlx, mlx_stt)")
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_thinking: bool = False
    supports_logprobs: bool = False
    supports_hidden_states: bool = False
    supports_batch: bool = False
    supports_embeddings: bool = False


class SystemCapabilities(BaseModel):
    """Response for GET /v1/system/capabilities.

    Tells the frontend what features are available on this server.
    """
    server_version: str
    platform: str = Field(..., description="darwin, linux, or win32")
    providers: List[ProviderCapability] = Field(default_factory=list)
    features: List[str] = Field(
        default_factory=list,
        description="Server-level feature flags (e.g. 'analytics', 'batch', 'eval')",
    )
    max_loaded_models: int = 2


class ModelPerformanceSnapshot(BaseModel):
    """Performance snapshot for a single loaded model."""
    model_id: str
    provider: str
    context_used: int = 0
    context_capacity: int = 0
    context_percent: float = 0.0
    memory_mb: float = 0.0
    requests_active: int = 0
    avg_generation_tps: Optional[float] = None
    avg_prompt_tps: Optional[float] = None


class SystemResourceSnapshot(BaseModel):
    """System resource usage at a point in time."""
    ram_used_gb: float
    ram_available_gb: float
    ram_total_gb: float
    cpu_percent: float


class SystemPerformance(BaseModel):
    """Response for GET /v1/system/performance.

    Combines system resources and per-model performance into one snapshot.
    """
    timestamp: str = Field(..., description="ISO timestamp")
    system: SystemResourceSnapshot
    models: Dict[str, ModelPerformanceSnapshot] = Field(default_factory=dict)
