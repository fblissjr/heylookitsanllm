# src/edge_llm/config.py
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
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart]]

class ChatRequest(BaseModel):
    model: Optional[str] = None  # Allow None for default model selection
    messages: List[ChatMessage]

    # Generation parameters with sensible defaults
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(0, ge=0)
    min_p: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(1.1, ge=0.1, le=2.0)
    repetition_context_size: Optional[int] = Field(20, ge=1)

    max_tokens: Optional[int] = Field(512, gt=0, le=8192)
    stream: bool = False
    include_performance: bool = False

    # Advanced features
    draft_model: Optional[str] = None
    num_draft_tokens: Optional[int] = Field(5, ge=1, le=20)
    seed: Optional[int] = None

    # Sampling parameters
    xtc_probability: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    xtc_threshold: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    logit_bias: Optional[Dict[str, float]] = None

    # KV Cache parameters
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = Field(None, ge=1, le=8)
    quantized_kv_start: Optional[int] = Field(5000, ge=0)

    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v

    @validator('logit_bias')
    def validate_logit_bias(cls, v):
        if v is not None:
            # Convert string keys to integers if needed
            return {int(k) if isinstance(k, str) and k.isdigit() else k: float(val) for k, val in v.items()}
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

# --- Model Configuration ---
class MLXModelConfig(BaseModel):
    model_path: str
    draft_model_path: Optional[str] = None
    num_draft_tokens: int = Field(5, ge=1, le=20)
    vision: bool = False

    # Default generation parameters (can be overridden per request)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0, le=8192)
    repetition_penalty: Optional[float] = Field(None, ge=0.1, le=2.0)

    # Advanced cache settings
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = Field(None, ge=1, le=8)
    quantized_kv_start: Optional[int] = Field(5000, ge=0)
    prompt_cache_file: Optional[str] = None  # For persistent caching

class LlamaCppModelConfig(BaseModel):
    model_path: str
    mmproj_path: Optional[str] = None
    chat_format: Optional[str] = None
    chat_format_template: Optional[str] = None
    eos_token: Optional[str] = None
    bos_token: Optional[str] = None

    # Llama.cpp specific settings
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    n_batch: int = Field(512, ge=1)
    n_threads: Optional[int] = None

    # Model flags
    draft_model: bool = False
    vision: bool = False

    # Default generation parameters
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0, le=8192)
    repetition_penalty: Optional[float] = Field(None, ge=0.1, le=2.0)

class ModelConfig(BaseModel):
    id: str
    provider: Literal["mlx", "llama_cpp"]
    config: Union[MLXModelConfig, LlamaCppModelConfig]

    # Optional model metadata
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True

    @validator('config')
    def validate_config_type(cls, v, values):
        provider = values.get('provider')
        if provider == 'mlx' and not isinstance(v, MLXModelConfig):
            raise ValueError(f"MLX provider requires MLXModelConfig, got {type(v)}")
        elif provider == 'llama_cpp' and not isinstance(v, LlamaCppModelConfig):
            raise ValueError(f"Llama.cpp provider requires LlamaCppModelConfig, got {type(v)}")
        return v

class AppConfig(BaseModel):
    models: List[ModelConfig]

    # Global server settings
    default_model: Optional[str] = None
    max_concurrent_requests: int = Field(10, ge=1, le=100)
    request_timeout: int = Field(300, ge=30, le=3600)  # seconds

    # Logging configuration
    log_level: str = Field("INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @validator('models')
    def validate_models(cls, v):
        if not v:
            raise ValueError("At least one model must be configured")

        # Check for duplicate model IDs
        ids = [model.id for model in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Model IDs must be unique")

        return v

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID."""
        return next((m for m in self.models if m.id == model_id and m.enabled), None)

    def get_enabled_models(self) -> List[ModelConfig]:
        """Get all enabled models."""
        return [m for m in self.models if m.enabled]
