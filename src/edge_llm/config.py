# src/edge_llm/config.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union, Dict

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
    repetition_penalty: Optional[float] = Field(1.1)
    max_tokens: int = Field(512, gt=0)
    stream: bool = False
    include_performance: bool = Field(False)
    draft_model: Optional[str] = None
    num_draft_tokens: Optional[int] = None

class PerformanceMetrics(BaseModel): prompt_tps: float; generation_tps: float; peak_memory_gb: float

class ChatCompletionResponse(BaseModel):
    id: str; object: str; created: int; model: str
    choices: List[Dict]; usage: Dict
    performance: Optional[PerformanceMetrics] = None

class MLXModelConfig(BaseModel):
    model_path: str
    draft_model_path: Optional[str] = None
    num_draft_tokens: int = 5

class LlamaCppModelConfig(BaseModel):
    model_path: str; mmproj_path: Optional[str] = None; chat_format: Optional[str] = None
    chat_format_template: Optional[str] = None; eos_token: Optional[str] = None; bos_token: Optional[str] = None
    n_gpu_layers: int = -1; n_ctx: int = 4096
    draft_model: bool = False

class ModelConfig(BaseModel):
    id: str; provider: Literal["mlx", "llama_cpp"]; config: Union[MLXModelConfig, LlamaCppModelConfig]

class AppConfig(BaseModel): models: List[ModelConfig]
