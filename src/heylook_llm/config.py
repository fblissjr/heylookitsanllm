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
    cache_type: Literal["standard", "rotating", "quantized"] = "standard"
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = Field(None, ge=1, le=8)
    kv_group_size: int = 64
    quantized_kv_start: int = 2048

class LlamaCppModelConfig(BaseModel):
    model_path: str
    mmproj_path: Optional[str] = None
    chat_format: Optional[str] = None
    chat_format_template: Optional[str] = None
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    vision: bool = False

class CoreMLSTTModelConfig(BaseModel):
    model_path: str
    compute_units: Literal["CPU_ONLY", "CPU_AND_NE", "ALL"] = "ALL"
    sample_rate: int = 16000
    max_audio_seconds: int = 15
    vocab_size: int = 128
    blank_id: int = 127
    num_layers: int = 12
    hidden_size: int = 320
    max_symbols_per_timestep: int = 10
    durations: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])

class ModelConfig(BaseModel):
    id: str
    provider: Literal["mlx", "llama_cpp", "gguf", "coreml_stt"]  # Support all providers
    config: Union[MLXModelConfig, LlamaCppModelConfig, CoreMLSTTModelConfig]
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True

    @validator('config', pre=True)
    def validate_config_type(cls, v, values):
        provider = values.get('provider')
        if provider == 'mlx':
            return MLXModelConfig(**v)
        elif provider in ['llama_cpp', 'gguf']:  # Support both names
            return LlamaCppModelConfig(**v)
        elif provider == 'coreml_stt':
            return CoreMLSTTModelConfig(**v)
        raise ValueError(f"Unknown provider '{provider}' for model config validation")

class AppConfig(BaseModel):
    models: List[ModelConfig]
    default_model: Optional[str] = None
    max_loaded_models: int = Field(2, ge=1)

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        return next((m for m in self.models if m.id == model_id and m.enabled), None)

    def get_enabled_models(self) -> List[ModelConfig]:
        return [m for m in self.models if m.enabled]
