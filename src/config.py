# src/config.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union, Dict

# --- API Request/Response Models ---
class ImageUrl(BaseModel): url: str
class TextContentPart(BaseModel): type: Literal["text"]; text: str
class ImageContentPart(BaseModel): type: Literal["image_url"]; image_url: ImageUrl
ContentPart = Union[TextContentPart, ImageContentPart]
class ChatMessage(BaseModel): role: Literal["system", "user", "assistant"]; content: Union[str, List[ContentPart]]

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=-1, description="Set to -1 to disable.")
    repetition_penalty: Optional[float] = Field(None, ge=0.0)
    repetition_context_size: Optional[int] = Field(20, ge=0)
    max_tokens: int = Field(512, gt=0)
    stream: bool = False
    # --- New Optional Fields ---
    include_performance: bool = Field(False, description="Include performance metrics in the response.")
    draft_model: Optional[str] = Field(None, description="Runtime override for a draft model path (MLX only).")
    num_draft_tokens: Optional[int] = Field(None, description="Runtime override for number of draft tokens (MLX only).")

class PerformanceMetrics(BaseModel): prompt_tps: float; generation_tps: float; peak_memory_gb: float

class ChatCompletionResponse(BaseModel):
    id: str; object: str; created: int; model: str
    choices: List[Dict]; usage: Dict
    performance: Optional[PerformanceMetrics] = None

# --- models.yaml Configuration Models ---
class MLXModelConfig(BaseModel):
    model_path: str
    draft_model_path: Optional[str] = None
    num_draft_tokens: int = 5

class LlamaCppModelConfig(BaseModel):
    model_path: str; mmproj_path: Optional[str] = None; chat_format: Optional[str] = None
    chat_format_template: Optional[str] = None; eos_token: Optional[str] = None; bos_token: Optional[str] = None
    n_gpu_layers: int = -1; n_ctx: int = 4096

class ModelConfig(BaseModel):
    id: str; provider: Literal["mlx", "llama_cpp"]; config: Union[MLXModelConfig, LlamaCppModelConfig]
class AppConfig(BaseModel): models: List[ModelConfig]
