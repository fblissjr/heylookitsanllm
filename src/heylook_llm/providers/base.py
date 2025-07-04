# src/heylook_llm/providers/base.py
from abc import ABC, abstractmethod
from typing import Generator, Dict
from ..config import ChatRequest

class BaseProvider(ABC):
    """Abstract base class for all model backends."""
    def __init__(self, model_id: str, config: Dict, verbose: bool):
        self.model_id = model_id
        self.config = config
        self.verbose = verbose

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def create_chat_completion(self, request: ChatRequest) -> Generator:
        raise NotImplementedError

    def unload(self):
        """Optional method to explicitly release resources."""
        pass

    def __del__(self):
        self.unload()
