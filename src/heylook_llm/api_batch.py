# src/heylook_llm/api_batch.py
"""
Batch API endpoint for processing multiple independent chat completions.
Follows OpenAI's batch API pattern but for synchronous processing.
"""

from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import time
import asyncio
import logging

from .schemas import ChatRequest, ChatCompletionResponse
from .api import create_chat_completion


class BatchChatRequest(BaseModel):
    """Single request within a batch"""
    custom_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    request: ChatRequest
    
    
class BatchChatRequests(BaseModel):
    """Container for multiple chat requests"""
    requests: List[BatchChatRequest]
    parallel: bool = Field(default=False, description="Process requests in parallel")
    max_parallel: int = Field(default=4, description="Maximum parallel requests")


class BatchChatResponse(BaseModel):
    """Response for a single request in the batch"""
    custom_id: str
    response: Optional[ChatCompletionResponse] = None
    error: Optional[Dict[str, Any]] = None
    

class BatchChatResponses(BaseModel):
    """Container for batch responses"""
    responses: List[BatchChatResponse]
    total_time: float
    

async def create_batch_chat_completions(
    batch_requests: BatchChatRequests,
    router
) -> BatchChatResponses:
    """
    Process multiple chat completion requests as a batch.
    
    This is similar to OpenAI's batch API but processes synchronously
    and returns results immediately.
    """
    start_time = time.time()
    
    if batch_requests.parallel:
        # Process in parallel with concurrency limit
        responses = await _process_parallel(batch_requests, router)
    else:
        # Process sequentially
        responses = await _process_sequential(batch_requests, router)
    
    total_time = time.time() - start_time
    
    return BatchChatResponses(
        responses=responses,
        total_time=total_time
    )


async def _process_sequential(batch_requests: BatchChatRequests, router) -> List[BatchChatResponse]:
    """Process requests sequentially"""
    responses = []
    
    for batch_req in batch_requests.requests:
        try:
            # Create a mock request object for compatibility
            class MockRequest:
                def __init__(self):
                    self.state = type('obj', (object,), {'router_instance': router})
            
            mock_request = MockRequest()
            
            # Process the request
            result = await create_chat_completion(batch_req.request.model_dump(), mock_request)
            
            responses.append(BatchChatResponse(
                custom_id=batch_req.custom_id,
                response=ChatCompletionResponse(**result)
            ))
            
        except Exception as e:
            logging.error(f"Batch request {batch_req.custom_id} failed: {e}")
            responses.append(BatchChatResponse(
                custom_id=batch_req.custom_id,
                error={
                    "message": str(e),
                    "type": type(e).__name__
                }
            ))
    
    return responses


async def _process_parallel(batch_requests: BatchChatRequests, router) -> List[BatchChatResponse]:
    """Process requests in parallel with concurrency control"""
    semaphore = asyncio.Semaphore(batch_requests.max_parallel)
    
    async def process_one(batch_req: BatchChatRequest) -> BatchChatResponse:
        async with semaphore:
            try:
                # Create a mock request object
                class MockRequest:
                    def __init__(self):
                        self.state = type('obj', (object,), {'router_instance': router})
                
                mock_request = MockRequest()
                
                # Process the request
                result = await create_chat_completion(batch_req.request.model_dump(), mock_request)
                
                return BatchChatResponse(
                    custom_id=batch_req.custom_id,
                    response=ChatCompletionResponse(**result)
                )
                
            except Exception as e:
                logging.error(f"Batch request {batch_req.custom_id} failed: {e}")
                return BatchChatResponse(
                    custom_id=batch_req.custom_id,
                    error={
                        "message": str(e),
                        "type": type(e).__name__
                    }
                )
    
    # Process all requests in parallel
    tasks = [process_one(req) for req in batch_requests.requests]
    responses = await asyncio.gather(*tasks)
    
    return responses