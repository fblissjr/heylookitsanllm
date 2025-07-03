# src/heylook_llm/providers/base.py
from abc import ABC, abstractmethod
from typing import Generator, Dict

class BaseProvider(ABC):
    """Abstract base class for all model backends."""
    def __init__(self, model_id: str, config: Dict, verbose: bool):
        self.model_id = model_id
        # Why: Store the model-specific config to access its defaults later.
        self.config = config
        self.load_model(config, verbose)

    @abstractmethod
    def load_model(self, config: Dict, verbose: bool): raise NotImplementedError
    @abstractmethod
    def create_chat_completion(self, request: Dict) -> Generator: raise NotImplementedError
