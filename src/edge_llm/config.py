# src/edge_llm/config.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union, Dict

# Why: Defines the strict data structures for API requests and YAML configuration,
# providing automatic validation and type hints throughout the application.

class ImageUrl(BaseModel):
    url: str

class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str

class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl

# Why: This was previously nested incorrectly. It must be at the top level.
ContentPart = Union[TextContentPart, ImageContentPart]

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart]]

class ChatRequest(BaseModel):
    # Why: This is the critical fix. Making the model optional allows requests
    # to be validated even if the client doesn't specify a model.
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(1.1)
    # Why: Removed the duplicate max_tokens field.
    max_tokens: Optional[int] = Field(512, gt=0)
    stream: bool = False
    include_performance: bool = Field(False)
    draft_model: Optional[str] = None
    num_draft_tokens: Optional[int] = None

class PerformanceMetrics(BaseModel):
    prompt_tps: float; generation_tps: float; peak_memory_gb: float

class ChatCompletionResponse(BaseModel):
    id: str; object: str; created: int; model: str
    choices: List[Dict]; usage: Dict
    performance: Optional[PerformanceMetrics] = None

# --- models.yaml Configuration Models ---
class MLXModelConfig(BaseModel):
    model_path: str; draft_model_path: Optional[str] = None; num_draft_tokens: int = 5
    vision: bool = False; temperature: Optional[float] = None; top_p: Optional[float] = None
    max_tokens: Optional[int] = None; repetition_penalty: Optional[float] = None

class LlamaCppModelConfig(BaseModel):
    model_path: str; mmproj_path: Optional[str] = None; chat_format: Optional[str] = None
    chat_format_template: Optional[str] = None; eos_token: Optional[str] = None; bos_token: Optional[str] = None
    n_gpu_layers: int = -1; n_ctx: int = 4096; draft_model: bool = False; vision: bool = False
    temperature: Optional[float] = None; top_p: Optional[float] = None
    max_tokens: Optional[int] = None; repetition_penalty: Optional[float] = None

class ModelConfig(BaseModel):
    id: str; provider: Literal["mlx", "llama_cpp"]; config: Union[MLXModelConfig, LlamaCppModelConfig]

class AppConfig(BaseModel):
    models: List[ModelConfig]
