"""
Ollama to OpenAI API translation logic
Handles format conversion between Ollama and OpenAI API formats
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import base64
import logging

logger = logging.getLogger(__name__)

class OllamaTranslator:
    """Translates between Ollama and OpenAI API formats"""
    
    def __init__(self):
        pass
    
    def translate_ollama_chat_to_openai(self, ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Ollama /api/chat request to OpenAI /v1/chat/completions format"""
        
        # Start with basic structure
        openai_request = {
            "model": ollama_request.get("model", "default"),
            "messages": self._translate_messages(ollama_request.get("messages", [])),
            "stream": False  # We're skipping streaming for now
        }
        
        # Map Ollama parameters to OpenAI parameters
        param_mapping = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "max_tokens": "max_tokens",
            "seed": "seed"
        }
        
        # Check direct parameters
        for ollama_param, openai_param in param_mapping.items():
            if ollama_param in ollama_request:
                openai_request[openai_param] = ollama_request[ollama_param]
        
        # Check options dict (Ollama sometimes nests parameters here)
        options = ollama_request.get("options", {})
        for ollama_param, openai_param in param_mapping.items():
            if ollama_param in options:
                openai_request[openai_param] = options[ollama_param]
        
        return openai_request
    
    def translate_ollama_generate_to_openai(self, ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Ollama /api/generate request to OpenAI /v1/chat/completions format"""
        
        # Convert prompt to messages format
        messages = []
        
        # Add system message if present
        if "system" in ollama_request and ollama_request["system"]:
            messages.append({
                "role": "system",
                "content": ollama_request["system"]
            })
        
        # Add user message with prompt
        user_message = {
            "role": "user",
            "content": ollama_request.get("prompt", "")
        }
        
        # Handle images in generate request
        if "images" in ollama_request and ollama_request["images"]:
            user_message["content"] = self._create_content_with_images(
                ollama_request["prompt"], 
                ollama_request["images"]
            )
        
        messages.append(user_message)
        
        # Create OpenAI request
        openai_request = {
            "model": ollama_request.get("model", "default"),
            "messages": messages,
            "stream": False
        }
        
        # Map parameters (same as chat)
        param_mapping = {
            "temperature": "temperature",
            "top_p": "top_p", 
            "top_k": "top_k",
            "max_tokens": "max_tokens",
            "seed": "seed"
        }
        
        for ollama_param, openai_param in param_mapping.items():
            if ollama_param in ollama_request:
                openai_request[openai_param] = ollama_request[ollama_param]
        
        # Check options dict
        options = ollama_request.get("options", {})
        for ollama_param, openai_param in param_mapping.items():
            if ollama_param in options:
                openai_request[openai_param] = options[ollama_param]
        
        return openai_request
    
    def _translate_messages(self, ollama_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Ollama messages to OpenAI format"""
        
        openai_messages = []
        
        for msg in ollama_messages:
            openai_msg = {
                "role": msg.get("role", "user")
            }
            
            # Handle images in message
            if "images" in msg and msg["images"]:
                openai_msg["content"] = self._create_content_with_images(
                    msg.get("content", ""),
                    msg["images"]
                )
            else:
                openai_msg["content"] = msg.get("content", "")
            
            openai_messages.append(openai_msg)
        
        return openai_messages
    
    def _create_content_with_images(self, text: str, images: List[str]) -> List[Dict[str, Any]]:
        """Create OpenAI content format with text and images"""
        
        content = []
        
        # Add text content
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        # Add image content
        for image_b64 in images:
            # Ollama sends base64 encoded images
            # OpenAI expects data URLs
            if not image_b64.startswith("data:"):
                # Assume it's a raw base64 string, prepend data URL prefix
                image_url = f"data:image/jpeg;base64,{image_b64}"
            else:
                image_url = image_b64
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
        
        return content
    
    def translate_openai_to_ollama_chat(self, openai_response: Dict[str, Any], 
                                       model: str, request_start_time: float) -> Dict[str, Any]:
        """Convert OpenAI chat completion response to Ollama /api/chat format"""
        
        # Extract message content
        message_content = ""
        if "choices" in openai_response and openai_response["choices"]:
            choice = openai_response["choices"][0]
            if "message" in choice:
                message_content = choice["message"].get("content", "")
        
        # Calculate duration in nanoseconds (Ollama format)
        total_duration = int((time.time() - request_start_time) * 1_000_000_000)
        
        # Extract usage stats
        usage = openai_response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        ollama_response = {
            "model": model,
            "created_at": datetime.now().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": message_content
            },
            "done": True,
            "total_duration": total_duration,
            "load_duration": 0,  # We don't track this
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": 0,  # We don't track this separately
            "eval_count": completion_tokens,
            "eval_duration": total_duration  # Use total duration as approximation
        }
        
        return ollama_response
    
    def translate_openai_to_ollama_generate(self, openai_response: Dict[str, Any], 
                                           model: str, request_start_time: float) -> Dict[str, Any]:
        """Convert OpenAI response to Ollama /api/generate format"""
        
        # Extract message content
        response_text = ""
        if "choices" in openai_response and openai_response["choices"]:
            choice = openai_response["choices"][0]
            if "message" in choice:
                response_text = choice["message"].get("content", "")
        
        # Calculate duration in nanoseconds
        total_duration = int((time.time() - request_start_time) * 1_000_000_000)
        
        # Extract usage stats
        usage = openai_response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        ollama_response = {
            "model": model,
            "created_at": datetime.now().isoformat() + "Z",
            "response": response_text,
            "done": True,
            "context": [],  # We don't maintain context
            "total_duration": total_duration,
            "load_duration": 0,
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": 0,
            "eval_count": completion_tokens,
            "eval_duration": total_duration
        }
        
        return ollama_response
    
    def translate_openai_models_to_ollama(self, openai_models: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI /v1/models response to Ollama /api/tags format"""
        
        ollama_models = []
        
        if "data" in openai_models:
            for model in openai_models["data"]:
                ollama_model = {
                    "name": model.get("id", "unknown"),
                    "model": model.get("id", "unknown"),
                    "modified_at": datetime.now().isoformat() + "Z",
                    "size": 0,  # We don't track size
                    "digest": "unknown",
                    "details": {
                        "parent_model": "",
                        "format": "unknown",
                        "family": "unknown",
                        "families": [],
                        "parameter_size": "unknown",
                        "quantization_level": "unknown"
                    }
                }
                ollama_models.append(ollama_model)
        
        return {"models": ollama_models}
