# src/edge_llm/providers/base.py
from abc import ABC, abstractmethod
from typing import Generator, Dict

class BaseProvider(ABC):
    def __init__(self, model_id: str, config: Dict, verbose: bool):
        self.model_id = model_id
        self.load_model(config, verbose)

    @abstractmethod
    def load_model(self, config: Dict, verbose: bool): raise NotImplementedError
    @abstractmethod
    def create_chat_completion(self, request: Dict) -> Generator: raise NotImplementedError
