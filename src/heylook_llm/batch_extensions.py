# src/heylook_llm/batch_extensions.py
"""
Batch processing extensions for the OpenAI-compatible API.
Handles multiple independent conversations in a single request.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from heylook_llm.config import ChatMessage


class ConversationDelimiter(str):
    """Special marker to separate conversations in batch mode"""
    CONVERSATION_BOUNDARY = "___CONVERSATION_BOUNDARY___"


class BatchChatRequestV2(BaseModel):
    """
    Extended chat request that supports multiple conversations.
    
    Uses a special system message with delimiter to separate conversations:
    {"role": "system", "content": "___CONVERSATION_BOUNDARY___"}
    """
    model: Optional[str] = None
    messages: List[ChatMessage]
    
    # Standard OpenAI parameters (apply to all conversations)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    
    # Batch processing options
    processing_mode: str = Field("conversation", description="conversation|sequential")
    return_individual: bool = Field(True, description="Return separate responses per conversation")
    
    @property
    def conversations(self) -> List[List[ChatMessage]]:
        """Split messages into separate conversations based on delimiter"""
        conversations = []
        current_conversation = []
        
        for msg in self.messages:
            # Check if this is a conversation boundary
            if (msg.role == "system" and 
                msg.content == ConversationDelimiter.CONVERSATION_BOUNDARY):
                if current_conversation:
                    conversations.append(current_conversation)
                    current_conversation = []
            else:
                current_conversation.append(msg)
        
        # Add the last conversation
        if current_conversation:
            conversations.append(current_conversation)
        
        return conversations


# Helper function to build batch requests
def create_batch_request(conversations: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """
    Helper to create a batch request from multiple conversations.
    
    Args:
        conversations: List of conversation dicts, each with 'messages' key
        **kwargs: Additional parameters (model, temperature, etc.)
    
    Returns:
        Dict ready to send as JSON to the API
    """
    all_messages = []
    
    for i, conv in enumerate(conversations):
        # Add boundary marker between conversations (except before first)
        if i > 0:
            all_messages.append({
                "role": "system",
                "content": ConversationDelimiter.CONVERSATION_BOUNDARY
            })
        
        # Add all messages from this conversation
        all_messages.extend(conv["messages"])
    
    return {
        "messages": all_messages,
        "processing_mode": "sequential",
        "return_individual": True,
        **kwargs
    }


# Example usage for your case
def create_shrug_batch_request(prompts_and_images: List[Dict]) -> Dict[str, Any]:
    """
    Create a batch request for multiple ShrugPrompter-style inferences.
    
    Args:
        prompts_and_images: List of dicts with 'system', 'user', and 'image_b64' keys
    
    Returns:
        Batch request ready to send
    """
    conversations = []
    
    for item in prompts_and_images:
        conversation = {
            "messages": [
                {"role": "system", "content": item["system"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item["user"]},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{item['image_b64']}"}}
                    ]
                }
            ]
        }
        conversations.append(conversation)
    
    return create_batch_request(
        conversations,
        model="qwen2.5-vl-72b-mlx",
        max_tokens=512,
        temperature=1.0
    )