# src/heylook_llm/batch_processor.py
"""
Batch processing extension for OpenAI-compatible API.

Supports two modes:
1. Conversation mode (default): Messages are treated as conversation history
2. Batch mode: Messages are processed sequentially as separate requests
"""

import asyncio
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from fastapi import HTTPException
from pydantic import BaseModel, Field

from heylook_llm.config import ChatRequest, ChatMessage, ChatCompletionResponse


class ProcessingMode(str, Enum):
    """Processing mode for multiple messages"""
    CONVERSATION = "conversation"  # Default OpenAI behavior - all messages are context
    SEQUENTIAL = "sequential"      # Each user message starts fresh (no context)
    SEQUENTIAL_WITH_CONTEXT = "sequential_with_context"  # Each message processed separately but with growing context
    PARALLEL = "parallel"          # Process messages in parallel
    PARALLEL_WITH_CONTEXT = "parallel_with_context"  # Process in parallel but maintain context order


class BatchChatRequest(BaseModel):
    """Extended chat request with batch processing options"""
    model: Optional[str] = None
    messages: List[ChatMessage]
    
    # Standard OpenAI parameters
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=0.1, le=2.0)
    repetition_context_size: Optional[int] = Field(None, ge=1)
    max_tokens: Optional[int] = Field(None, gt=0)
    stream: bool = False
    seed: Optional[int] = None
    
    # Batch processing extensions
    processing_mode: ProcessingMode = ProcessingMode.CONVERSATION
    batch_size: Optional[int] = Field(1, ge=1, le=10, description="Number of messages to process together")
    include_context: bool = Field(True, description="Include previous messages as context in sequential mode")
    max_context_messages: Optional[int] = Field(None, ge=0, description="Limit context to N previous messages")
    
    # Response options
    return_individual: bool = Field(False, description="Return individual responses vs combined")
    include_timing: bool = Field(False, description="Include timing information in response")


class BatchResponse(BaseModel):
    """Response for batch processing requests"""
    id: str
    object: str = "chat.completion.batch"
    created: int
    model: str
    processing_mode: ProcessingMode
    
    # For conversation mode or combined response
    choices: Optional[List[Dict]] = None
    usage: Optional[Dict] = None
    
    # For individual responses
    completions: Optional[List[ChatCompletionResponse]] = None
    
    # Timing information
    timing: Optional[Dict[str, float]] = None
    
    # Batch metadata
    batch_info: Optional[Dict[str, Any]] = None


class MessageGroup:
    """Group of messages to process together"""
    def __init__(self, messages: List[ChatMessage], include_context: bool = True):
        self.messages = messages
        self.include_context = include_context
        self.context: List[ChatMessage] = []
        self.response: Optional[Any] = None
        self.error: Optional[str] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None


class BatchProcessor:
    """
    Handles batch processing of chat requests.
    
    Supports:
    - Conversation mode (standard OpenAI behavior)
    - Sequential mode (process messages separately in order)
    - Context management for sequential processing
    """
    
    def __init__(self, router):
        self.router = router
        
    async def process_batch_request(self, batch_request: BatchChatRequest) -> BatchResponse:
        """
        Process a batch chat request based on the specified mode.
        
        Args:
            batch_request: Extended chat request with batch options
            
        Returns:
            BatchResponse with results based on processing mode
        """
        start_time = time.time()
        
        if batch_request.processing_mode == ProcessingMode.CONVERSATION:
            # Standard OpenAI behavior - all messages are conversation history
            return await self._process_conversation_mode(batch_request, start_time)
        
        elif batch_request.processing_mode == ProcessingMode.SEQUENTIAL:
            # Process each message separately in order
            return await self._process_sequential_mode(batch_request, start_time)
        
        elif batch_request.processing_mode == ProcessingMode.SEQUENTIAL_WITH_CONTEXT:
            # Process messages separately but maintain context
            batch_request.include_context = True
            return await self._process_sequential_mode(batch_request, start_time)
        
        elif batch_request.processing_mode == ProcessingMode.PARALLEL:
            # Process messages in parallel
            return await self._process_parallel_mode(batch_request, start_time, with_context=False)
        
        elif batch_request.processing_mode == ProcessingMode.PARALLEL_WITH_CONTEXT:
            # Process in parallel but maintain context
            return await self._process_parallel_mode(batch_request, start_time, with_context=True)
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown processing mode: {batch_request.processing_mode}"
            )
    
    async def _process_conversation_mode(
        self, 
        batch_request: BatchChatRequest, 
        start_time: float
    ) -> BatchResponse:
        """Process all messages as a single conversation (standard behavior)."""
        
        # Create standard ChatRequest
        chat_request = ChatRequest(
            model=batch_request.model,
            messages=batch_request.messages,
            temperature=batch_request.temperature,
            top_p=batch_request.top_p,
            top_k=batch_request.top_k,
            min_p=batch_request.min_p,
            repetition_penalty=batch_request.repetition_penalty,
            repetition_context_size=batch_request.repetition_context_size,
            max_tokens=batch_request.max_tokens,
            stream=False,  # Batching doesn't support streaming
            seed=batch_request.seed
        )
        
        # Get provider and generate response
        provider = self.router.get_provider(chat_request.model)
        generator = provider.create_chat_completion(chat_request)
        
        # Collect full response
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        
        for chunk in generator:
            full_text += chunk.text
            prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
            completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)
        
        # Build response
        response = BatchResponse(
            id=f"chatcmpl-batch-{uuid.uuid4()}",
            created=int(time.time()),
            model=batch_request.model,
            processing_mode=ProcessingMode.CONVERSATION,
            choices=[{"message": {"role": "assistant", "content": full_text}}],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
        
        if batch_request.include_timing:
            response.timing = {
                "total_time": time.time() - start_time,
                "processing_time": time.time() - start_time
            }
        
        return response
    
    async def _process_sequential_mode(
        self, 
        batch_request: BatchChatRequest, 
        start_time: float
    ) -> BatchResponse:
        """Process each message separately in order, optionally with context."""
        
        # Split messages into groups based on role
        message_groups = self._create_message_groups(
            batch_request.messages,
            batch_request.include_context,
            batch_request.max_context_messages
        )
        
        # Process each group sequentially
        completions = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        context_messages = []
        
        for group in message_groups:
            group_start = time.time()
            
            # Build messages for this request
            if batch_request.include_context:
                messages_for_request = context_messages + group.messages
            else:
                messages_for_request = group.messages
            
            # Create ChatRequest for this group
            chat_request = ChatRequest(
                model=batch_request.model,
                messages=messages_for_request,
                temperature=batch_request.temperature,
                top_p=batch_request.top_p,
                top_k=batch_request.top_k,
                min_p=batch_request.min_p,
                repetition_penalty=batch_request.repetition_penalty,
                repetition_context_size=batch_request.repetition_context_size,
                max_tokens=batch_request.max_tokens,
                stream=False,
                seed=batch_request.seed
            )
            
            try:
                # Get provider and generate response
                provider = self.router.get_provider(chat_request.model)
                generator = provider.create_chat_completion(chat_request)
                
                # Collect response
                full_text = ""
                prompt_tokens = 0
                completion_tokens = 0
                
                for chunk in generator:
                    full_text += chunk.text
                    prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
                    completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)
                
                # Track tokens
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                
                # Create individual response
                completion = ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=batch_request.model,
                    choices=[{"message": {"role": "assistant", "content": full_text}}],
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                )
                
                completions.append(completion)
                
                # Update context for next iteration
                if batch_request.include_context:
                    # Add user messages and assistant response to context
                    context_messages.extend(group.messages)
                    context_messages.append(ChatMessage(
                        role="assistant",
                        content=full_text
                    ))
                    
                    # Limit context size if specified
                    if batch_request.max_context_messages:
                        context_messages = context_messages[-batch_request.max_context_messages:]
                
                group.end_time = time.time()
                
            except Exception as e:
                logging.error(f"Error processing message group: {e}")
                group.error = str(e)
                # Continue processing other groups
        
        # Build response based on return_individual flag
        response = BatchResponse(
            id=f"chatcmpl-batch-{uuid.uuid4()}",
            created=int(time.time()),
            model=batch_request.model,
            processing_mode=ProcessingMode.SEQUENTIAL
        )
        
        if batch_request.return_individual:
            response.completions = completions
        else:
            # Combine all responses
            combined_content = "\n\n".join([
                comp.choices[0]["message"]["content"] 
                for comp in completions
            ])
            response.choices = [{"message": {"role": "assistant", "content": combined_content}}]
            response.usage = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        
        # Add batch info
        response.batch_info = {
            "groups_processed": len(message_groups),
            "successful": len(completions),
            "failed": len([g for g in message_groups if g.error])
        }
        
        if batch_request.include_timing:
            response.timing = {
                "total_time": time.time() - start_time,
                "average_per_group": (time.time() - start_time) / len(message_groups)
            }
        
        return response
    
    async def _process_parallel_mode(
        self,
        batch_request: BatchChatRequest,
        start_time: float,
        with_context: bool
    ) -> BatchResponse:
        """Process messages in parallel with optional context management."""
        
        # Split messages into groups
        message_groups = self._create_message_groups(
            batch_request.messages,
            with_context,
            batch_request.max_context_messages
        )
        
        # Limit concurrency to avoid overwhelming the system
        max_concurrent = min(len(message_groups), batch_request.batch_size or 4)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_group(group: MessageGroup, context_messages: List[ChatMessage]) -> tuple:
            """Process a single group with semaphore control"""
            async with semaphore:
                try:
                    group.start_time = time.time()
                    
                    # Build messages for this request
                    if with_context and context_messages:
                        messages_for_request = context_messages + group.messages
                    else:
                        messages_for_request = group.messages
                    
                    # Create ChatRequest
                    chat_request = ChatRequest(
                        model=batch_request.model,
                        messages=messages_for_request,
                        temperature=batch_request.temperature,
                        top_p=batch_request.top_p,
                        top_k=batch_request.top_k,
                        min_p=batch_request.min_p,
                        repetition_penalty=batch_request.repetition_penalty,
                        repetition_context_size=batch_request.repetition_context_size,
                        max_tokens=batch_request.max_tokens,
                        stream=False,
                        seed=batch_request.seed
                    )
                    
                    # Process in thread pool to avoid blocking async loop
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        self._process_single_request_sync,
                        chat_request
                    )
                    
                    group.response = response
                    group.end_time = time.time()
                    
                    return group, response
                    
                except Exception as e:
                    logging.error(f"Error processing group in parallel: {e}")
                    group.error = str(e)
                    group.end_time = time.time()
                    return group, None
        
        # Process groups
        if with_context:
            # For context mode, we need to maintain order somewhat
            # Process in batches to maintain some context coherence
            completions = []
            context_messages = []
            
            # Process in chunks
            for i in range(0, len(message_groups), max_concurrent):
                batch = message_groups[i:i + max_concurrent]
                
                # Process batch in parallel
                tasks = [
                    process_group(group, context_messages.copy())
                    for group in batch
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Update context with results (in order)
                for group, response in results:
                    if response:
                        completions.append(response)
                        if with_context:
                            context_messages.extend(group.messages)
                            context_messages.append(ChatMessage(
                                role="assistant",
                                content=response.choices[0]["message"]["content"]
                            ))
                            if batch_request.max_context_messages:
                                context_messages = context_messages[-batch_request.max_context_messages:]
        else:
            # No context needed - process all in parallel
            tasks = [
                process_group(group, [])
                for group in message_groups
            ]
            
            results = await asyncio.gather(*tasks)
            completions = [r[1] for r in results if r[1] is not None]
        
        # Calculate totals
        total_prompt_tokens = sum(c.usage.get("prompt_tokens", 0) for c in completions)
        total_completion_tokens = sum(c.usage.get("completion_tokens", 0) for c in completions)
        
        # Build response
        response = BatchResponse(
            id=f"chatcmpl-batch-{uuid.uuid4()}",
            created=int(time.time()),
            model=batch_request.model,
            processing_mode=ProcessingMode.PARALLEL if not with_context else ProcessingMode.PARALLEL_WITH_CONTEXT
        )
        
        if batch_request.return_individual:
            response.completions = completions
        else:
            # Combine all responses
            combined_content = "\n\n".join([
                comp.choices[0]["message"]["content"]
                for comp in completions
            ])
            response.choices = [{"message": {"role": "assistant", "content": combined_content}}]
            response.usage = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        
        # Add batch info
        response.batch_info = {
            "groups_processed": len(message_groups),
            "successful": len(completions),
            "failed": len([g for g in message_groups if g.error]),
            "max_concurrent": max_concurrent
        }
        
        if batch_request.include_timing:
            end_time = time.time()
            response.timing = {
                "total_time": end_time - start_time,
                "average_per_group": (end_time - start_time) / len(message_groups),
                "parallel_speedup": len(message_groups) / max_concurrent if max_concurrent > 0 else 1
            }
        
        return response
    
    def _process_single_request_sync(self, chat_request: ChatRequest) -> ChatCompletionResponse:
        """Synchronous wrapper for processing a single request"""
        provider = self.router.get_provider(chat_request.model)
        generator = provider.create_chat_completion(chat_request)
        
        # Collect response
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        
        for chunk in generator:
            full_text += chunk.text
            prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
            completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)
        
        # Create response
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=chat_request.model,
            choices=[{"message": {"role": "assistant", "content": full_text}}],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
    
    def _create_message_groups(
        self, 
        messages: List[ChatMessage],
        include_context: bool,
        max_context_messages: Optional[int]
    ) -> List[MessageGroup]:
        """
        Split messages into groups for sequential processing.
        
        Strategy:
        - Split on ___CONVERSATION_BOUNDARY___ markers
        - If no boundaries, each user message forms a group
        - Assistant messages are not processed as separate groups
        """
        groups = []
        
        # Debug logging
        logging.info(f"[BATCH PROCESSOR] Creating message groups from {len(messages)} messages")
        for idx, msg in enumerate(messages):
            if isinstance(msg.content, str):
                content_preview = str(msg.content)[:100]
            elif isinstance(msg.content, list):
                # For list content (like images), show more detail
                parts_info = []
                for part in msg.content:
                    if hasattr(part, 'type'):
                        if part.type == 'text':
                            parts_info.append(f"text: '{part.text[:50]}...'" if len(part.text) > 50 else f"text: '{part.text}'")
                        elif part.type == 'image_url':
                            parts_info.append("image_url")
                content_preview = f"[List with {len(msg.content)} parts: {', '.join(parts_info)}]"
            else:
                content_preview = f"[{type(msg.content).__name__}]"
            logging.info(f"[BATCH PROCESSOR] Message {idx}: role={msg.role}, content={content_preview}")
        
        # First, check if we have conversation boundaries in the content
        has_boundaries = False
        for message in messages:
            if isinstance(message.content, str) and "___CONVERSATION_BOUNDARY___" in message.content:
                has_boundaries = True
                break
        
        if has_boundaries:
            # Split based on conversation boundaries
            logging.info(f"Processing messages with conversation boundaries")
            current_conversation = []
            
            for message in messages:
                if isinstance(message.content, str) and "___CONVERSATION_BOUNDARY___" in message.content:
                    # If we have accumulated messages, create a group
                    if current_conversation:
                        groups.append(MessageGroup(current_conversation, include_context))
                        current_conversation = []
                    
                    # Check if this is JUST a boundary marker or has additional content
                    content_without_boundary = message.content.replace("___CONVERSATION_BOUNDARY___", "").strip()
                    if content_without_boundary:
                        # There's additional content beyond the boundary marker
                        # Split the content on boundaries and process each part
                        parts = message.content.split("___CONVERSATION_BOUNDARY___")
                        
                        # Filter out empty parts first
                        non_empty_parts = [p.strip() for p in parts if p.strip()]
                        
                        for i, part in enumerate(non_empty_parts):
                            # Add the part to current conversation
                            msg_copy = ChatMessage(
                                role=message.role,
                                content=part,
                                name=message.name,
                                tool_call_id=message.tool_call_id,
                                tool_calls=message.tool_calls
                            )
                            current_conversation.append(msg_copy)
                            
                            # If not the last part, finalize this conversation
                            if i < len(non_empty_parts) - 1:
                                groups.append(MessageGroup(current_conversation, include_context))
                                current_conversation = []
                    # If it's just a boundary marker, we already handled creating the group above
                else:
                    # Regular message without boundaries
                    current_conversation.append(message)
            
            # Add any remaining conversation
            if current_conversation:
                groups.append(MessageGroup(current_conversation, include_context))
            
            logging.info(f"Created {len(groups)} message groups from boundaries")
            for idx, group in enumerate(groups):
                logging.debug(f"Group {idx + 1}: {len(group.messages)} messages")
        
        else:
            # No boundaries - use original logic
            # For sequential processing, we want to group system/tool messages with the next user message
            system_messages = []
            
            for i, message in enumerate(messages):
                if message.role == "user":
                    # Create a group with any accumulated system messages + this user message
                    group_messages = system_messages + [message]
                    groups.append(MessageGroup(group_messages, include_context))
                    system_messages = []  # Reset for next group
                elif message.role in ["system", "tool"]:
                    # Accumulate system/tool messages for the next user message
                    system_messages.append(message)
                elif message.role == "assistant":
                    # Skip assistant messages - they'll be generated
                    continue
            
            # If there are leftover system messages without a user message, create a group
            # (This should be rare but handles edge cases)
            if system_messages:
                logging.warning(f"[BATCH PROCESSOR] Found {len(system_messages)} system/tool messages without a user message")
                groups.append(MessageGroup(system_messages, include_context))
        
        logging.info(f"[BATCH PROCESSOR] Total groups created: {len(groups)}")
        for idx, group in enumerate(groups):
            logging.info(f"[BATCH PROCESSOR] Group {idx}: {len(group.messages)} messages")
            for msg_idx, msg in enumerate(group.messages):
                if isinstance(msg.content, str):
                    content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                elif isinstance(msg.content, list):
                    content_preview = f"[{len(msg.content)} parts]"
                else:
                    content_preview = str(type(msg.content))
                logging.info(f"[BATCH PROCESSOR]   - Message {msg_idx}: {msg.role} - {content_preview}")
        return groups