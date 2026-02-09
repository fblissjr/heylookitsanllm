# src/heylook_llm/config.py
from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional, Union, Dict

class ImageUrl(BaseModel):
    url: str

class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str

class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl

ContentPart = Union[TextContentPart, ImageContentPart]

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[ContentPart]]
    thinking: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=0.1, le=2.0)
    repetition_context_size: Optional[int] = Field(None, ge=1)
    max_tokens: Optional[int] = Field(None, gt=0)
    stream: bool = False
    include_performance: bool = False
    seed: Optional[int] = None
    
    # Batch processing extensions
    processing_mode: Optional[str] = Field(None, description="conversation|sequential|sequential_with_context")
    return_individual: Optional[bool] = Field(None, description="Return individual responses vs combined")
    include_timing: Optional[bool] = Field(None, description="Include timing information")
    
    # Image resizing parameters (from multipart endpoint)
    resize_max: Optional[int] = Field(None, description="Resize images to max dimension (e.g., 512, 768, 1024)")
    resize_width: Optional[int] = Field(None, description="Resize images to specific width")
    resize_height: Optional[int] = Field(None, description="Resize images to specific height")
    image_quality: Optional[int] = Field(None, ge=1, le=100, description="JPEG quality for resized images")
    preserve_alpha: Optional[bool] = Field(None, description="Preserve alpha channel (outputs PNG)")

    # Thinking mode control (Qwen3 models)
    enable_thinking: Optional[bool] = Field(None, description="Enable thinking mode for Qwen3 models")

    # Additional sampler parameters
    presence_penalty: Optional[float] = Field(None, ge=0.0, le=2.0, description="Reduce repetition (0-2, recommended 1.5 for Qwen3 thinking)")

    # Logprobs support (OpenAI-compatible)
    logprobs: Optional[bool] = Field(None, description="Return log probabilities for output tokens")
    top_logprobs: Optional[int] = Field(None, ge=0, le=20, description="Number of top tokens with log probabilities (0-20)")

    # Streaming options (OpenAI-compatible)
    stream_options: Optional[Dict] = Field(None, description="Options for streaming: {include_usage: true} to get usage stats")

    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v

class PerformanceMetrics(BaseModel):
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]
    usage: Dict
    performance: Optional[PerformanceMetrics] = None


class BatchChatRequest(BaseModel):
    """Request for batch chat completions."""
    requests: List[ChatRequest]
    processing_mode: str = "batch"

    # Batch-specific parameters
    completion_batch_size: Optional[int] = Field(32, description="Max concurrent generations")
    prefill_batch_size: Optional[int] = Field(8, description="Max prefill parallelism")
    prefill_step_size: Optional[int] = Field(2048, description="Chunk size for prefill")

    @validator('requests')
    def validate_requests(cls, v):
        if not v:
            raise ValueError("Requests list cannot be empty")
        if len(v) < 2:
            raise ValueError("Batch requests must contain at least 2 requests")
        return v


class BatchStats(BaseModel):
    """Statistics for batch processing."""
    total_requests: int
    elapsed_seconds: float
    throughput_req_per_sec: float
    throughput_tok_per_sec: float
    prefill_time: float
    generation_time: float
    memory_peak_mb: float


class BatchChatResponse(BaseModel):
    """Response for batch chat completions."""
    object: str = "list"
    data: List[ChatCompletionResponse]
    batch_stats: BatchStats

class MLXModelConfig(BaseModel):
    model_path: str
    draft_model_path: Optional[str] = None
    num_draft_tokens: Optional[int] = 6
    vision: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    cache_type: Literal["standard", "rotating", "quantized"] = "standard"
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = Field(None, ge=1, le=8)
    kv_group_size: int = 64
    quantized_kv_start: int = 2048
    # Thinking mode (Qwen3 models with <think> blocks)
    enable_thinking: bool = False
    # Hidden states defaults (for /v1/hidden_states endpoint)
    default_hidden_layer: int = -2  # Z-Image uses penultimate layer
    default_max_length: int = 512
    # Thinking support metadata (for model capabilities discovery)
    supports_thinking: bool = False
    thinking_token_ids: Optional[List[int]] = None  # e.g., [151667, 151668] for <think>, </think>

class LlamaCppModelConfig(BaseModel):
    model_path: str
    mmproj_path: Optional[str] = None
    chat_format: Optional[str] = None
    chat_format_template: Optional[str] = None
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    vision: bool = False

class MLXSTTModelConfig(BaseModel):
    """Configuration for MLX STT models (parakeet-mlx)."""
    model_path: str = "mlx-community/parakeet-tdt-0.6b-v3"  # HF repo or local path
    chunk_duration: int = 120  # Chunk duration in seconds (0 to disable)
    overlap_duration: int = 15  # Overlap duration in seconds
    use_local_attention: bool = False
    local_attention_context: int = 256
    fp32: bool = False  # Use fp32 instead of bf16
    cache_dir: Optional[str] = None  # HuggingFace cache directory

class ModelConfig(BaseModel):
    id: str
    provider: Literal["mlx", "llama_cpp", "gguf", "mlx_stt"]  # Support all providers
    config: Union[MLXModelConfig, LlamaCppModelConfig, MLXSTTModelConfig]
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True
    # Model capabilities for discovery (e.g., ["hidden_states", "chat", "thinking", "vision"])
    capabilities: List[str] = Field(default_factory=list)

    @validator('config', pre=True)
    def validate_config_type(cls, v, values):
        provider = values.get('provider')
        if provider == 'mlx':
            return MLXModelConfig(**v)
        elif provider in ['llama_cpp', 'gguf']:  # Support both names
            return LlamaCppModelConfig(**v)
        elif provider == 'mlx_stt':
            return MLXSTTModelConfig(**v)
        raise ValueError(f"Unknown provider '{provider}' for model config validation")

class AppConfig(BaseModel):
    models: List[ModelConfig]
    default_model: Optional[str] = None
    max_loaded_models: int = Field(2, ge=1)

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        return next((m for m in self.models if m.id == model_id and m.enabled), None)

    def get_enabled_models(self) -> List[ModelConfig]:
        return [m for m in self.models if m.enabled]


# =============================================================================
# System Metrics Models
# =============================================================================

class SystemResourceMetrics(BaseModel):
    """System-wide resource metrics (RAM, CPU)."""
    ram_used_gb: float = Field(..., description="RAM currently used in GB")
    ram_available_gb: float = Field(..., description="RAM available in GB")
    ram_total_gb: float = Field(..., description="Total system RAM in GB")
    cpu_percent: float = Field(..., description="CPU usage percentage")


class ModelMetrics(BaseModel):
    """Per-model metrics (context usage, memory)."""
    context_used: int = Field(..., description="Tokens currently in context")
    context_capacity: int = Field(..., description="Maximum context window size")
    context_percent: float = Field(..., description="Context usage percentage")
    memory_mb: float = Field(..., description="Model memory usage in MB")
    requests_active: int = Field(0, description="Active requests for this model")


class SystemMetricsResponse(BaseModel):
    """Response for GET /v1/system/metrics endpoint."""
    timestamp: str = Field(..., description="ISO timestamp of metrics collection")
    system: SystemResourceMetrics
    models: Dict[str, ModelMetrics] = Field(default_factory=dict, description="Metrics per loaded model")


# =============================================================================
# Cache Management Models
# =============================================================================

class CacheInfo(BaseModel):
    """Information about a saved prompt cache."""
    cache_id: str = Field(..., description="Unique cache identifier")
    model: str = Field(..., description="Model ID this cache belongs to")
    name: str = Field(..., description="User-friendly cache name")
    description: Optional[str] = Field(None, description="Optional description")
    tokens_cached: int = Field(..., description="Number of tokens in cache")
    size_mb: float = Field(..., description="Cache file size in MB")
    created_at: str = Field(..., description="ISO timestamp of creation")


# Placeholder models for future /v1/cache/save endpoint (not yet implemented)
class CacheSaveRequest(BaseModel):
    """Request to save current prompt cache."""
    model: str = Field(..., description="Model ID to save cache for")
    name: str = Field(..., description="User-friendly name for the cache")
    description: Optional[str] = Field(None, description="Optional description")


class CacheSaveResponse(BaseModel):
    """Response from cache save operation."""
    cache_id: str
    model: str
    name: str
    tokens_cached: int
    size_mb: float
    created_at: str


class CacheListResponse(BaseModel):
    """Response for listing saved caches."""
    caches: List[CacheInfo] = Field(default_factory=list)


class CacheClearRequest(BaseModel):
    """Request to clear caches."""
    model: Optional[str] = Field(None, description="Model ID to clear caches for (all if omitted)")


class CacheClearResponse(BaseModel):
    """Response from cache clear operation."""
    deleted_count: int


# =============================================================================
# Enhanced Streaming Metadata
# Schema documentation for stream_options.include_usage response fields.
# These models document the API contract; actual streaming uses dicts directly.
# =============================================================================

class EnhancedUsage(BaseModel):
    """Extended usage statistics including thinking tokens."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: Optional[int] = Field(None, description="Tokens used in thinking blocks")
    content_tokens: Optional[int] = Field(None, description="Tokens in actual content")
    total_tokens: int = 0


class GenerationTiming(BaseModel):
    """Timing breakdown for generation phases."""
    thinking_duration_ms: Optional[int] = Field(None, description="Time spent in thinking phase")
    content_duration_ms: Optional[int] = Field(None, description="Time spent generating content")
    total_duration_ms: int = Field(..., description="Total generation time")


class GenerationConfig(BaseModel):
    """Sampler configuration used for generation."""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    enable_thinking: Optional[bool] = None
    max_tokens: Optional[int] = None


# =============================================================================
# SSE Stream Chunk Models
# These document the Server-Sent Events payload for streaming responses.
# =============================================================================

class TopLogprobEntry(BaseModel):
    """A candidate token with its log probability (used in top_logprobs arrays)."""
    token: str = Field(..., description="Token text")
    token_id: int = Field(..., description="Token vocabulary ID")
    logprob: float = Field(..., description="Log probability of this token")
    bytes: List[int] = Field(default_factory=list, description="UTF-8 byte values")

class TokenLogprobInfo(BaseModel):
    """Token with its log probability and alternative candidates."""
    token: str = Field(..., description="Token text")
    token_id: int = Field(..., description="Token vocabulary ID")
    logprob: float = Field(..., description="Log probability of this token")
    bytes: List[int] = Field(default_factory=list, description="UTF-8 byte values")
    top_logprobs: Optional[List[TopLogprobEntry]] = Field(
        None, description="Alternative tokens with their logprobs"
    )

class StreamLogprobs(BaseModel):
    """Logprobs attached to a streaming chunk."""
    content: List[TokenLogprobInfo] = Field(
        default_factory=list, description="Token-level logprob data for this chunk"
    )

class StreamDelta(BaseModel):
    """Delta content in a streaming chunk."""
    role: Optional[str] = Field(None, description="Role (only in first chunk)")
    content: Optional[str] = Field(None, description="Text content delta")
    thinking: Optional[str] = Field(None, description="Thinking content delta")

class StreamChoice(BaseModel):
    """Single choice in a streaming chunk."""
    index: int = 0
    delta: StreamDelta = Field(default_factory=StreamDelta)
    logprobs: Optional[StreamLogprobs] = None
    finish_reason: Optional[str] = Field(
        None, description="'stop', 'length', or null while streaming"
    )

class StreamChunk(BaseModel):
    """SSE payload for a single streaming chunk (data: {...}).

    Sent as Server-Sent Events on the /v1/chat/completions endpoint
    when stream=true. The final chunk includes usage, timing, and
    generation_config when stream_options.include_usage=true.
    """
    id: str = Field(..., description="Response identifier (chatcmpl-...)")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model ID used for generation")
    choices: List[StreamChoice]
    usage: Optional[EnhancedUsage] = Field(
        None, description="Token usage (final chunk only, requires stream_options.include_usage)"
    )
    timing: Optional[GenerationTiming] = Field(
        None, description="Generation timing breakdown (final chunk only)"
    )
    generation_config: Optional[GenerationConfig] = Field(
        None, description="Sampler settings used (final chunk only)"
    )
    stop_reason: Optional[str] = Field(
        None, description="Why generation stopped (final chunk only)"
    )
