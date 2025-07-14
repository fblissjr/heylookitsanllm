# src/heylook_llm/queue_manager.py
"""Request queue and batch processing manager for parallel inference"""
import asyncio
import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading

from heylook_llm.config import ChatRequest, ChatCompletionResponse
from heylook_llm.providers.base import BaseProvider


class RequestStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueuedRequest:
    """Individual request in the queue"""
    id: str
    request: ChatRequest
    status: RequestStatus = RequestStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    future: Optional[asyncio.Future] = None


class BatchConfig:
    """Configuration for batch processing"""
    def __init__(
        self,
        max_batch_size: int = 1,
        max_concurrent_batches: int = 1,
        batch_timeout_ms: int = 100,
        max_queue_size: int = 100,
        enable_dynamic_batching: bool = True
    ):
        self.max_batch_size = max_batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.batch_timeout_ms = batch_timeout_ms
        self.max_queue_size = max_queue_size
        self.enable_dynamic_batching = enable_dynamic_batching


class QueueManager:
    """
    Manages request queuing and batch processing for parallel inference.
    
    Features:
    - Request queuing with configurable limits
    - Dynamic batching based on model capabilities
    - Parallel processing when capacity allows
    - Backpressure handling
    - Request status tracking
    """
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.queue: List[QueuedRequest] = []
        self.processing: Dict[str, QueuedRequest] = {}
        self.completed: Dict[str, QueuedRequest] = {}
        self._lock = threading.Lock()
        self._queue_event = threading.Event()
        self._shutdown = False
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_batches,
            thread_name_prefix="inference"
        )
        
        # Start background processor
        self._processor_thread = threading.Thread(
            target=self._process_queue_loop,
            daemon=True,
            name="queue-processor"
        )
        self._processor_thread.start()
        
        logging.info(f"Queue manager initialized with batch_size={self.config.max_batch_size}, "
                    f"concurrent_batches={self.config.max_concurrent_batches}")
    
    async def submit_request(self, request: ChatRequest) -> str:
        """
        Submit a request to the queue.
        
        Returns:
            request_id: Unique ID for tracking the request
        
        Raises:
            RuntimeError: If queue is full
        """
        request_id = f"req-{uuid.uuid4()}"
        
        with self._lock:
            if len(self.queue) >= self.config.max_queue_size:
                raise RuntimeError(f"Queue full: {len(self.queue)} requests waiting")
            
            # Create future for async response
            future = asyncio.Future()
            
            queued_request = QueuedRequest(
                id=request_id,
                request=request,
                future=future
            )
            
            self.queue.append(queued_request)
            logging.debug(f"Request {request_id} queued. Queue size: {len(self.queue)}")
        
        # Signal processor thread
        self._queue_event.set()
        
        return request_id
    
    async def submit_batch(self, requests: List[ChatRequest]) -> List[str]:
        """
        Submit multiple requests as a potential batch.
        
        Args:
            requests: List of chat requests
            
        Returns:
            List of request IDs
        """
        request_ids = []
        
        for request in requests:
            request_id = await self.submit_request(request)
            request_ids.append(request_id)
        
        return request_ids
    
    async def get_result(self, request_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a request (blocking until complete).
        
        Args:
            request_id: The request ID to get results for
            timeout: Optional timeout in seconds
            
        Returns:
            The inference result
            
        Raises:
            TimeoutError: If timeout is exceeded
            RuntimeError: If request failed
        """
        # Check if already completed
        with self._lock:
            if request_id in self.completed:
                req = self.completed[request_id]
                if req.error:
                    raise RuntimeError(f"Request failed: {req.error}")
                return req.result
            
            # Find in queue or processing
            req = None
            for r in self.queue:
                if r.id == request_id:
                    req = r
                    break
            
            if not req and request_id in self.processing:
                req = self.processing[request_id]
            
            if not req:
                raise ValueError(f"Request {request_id} not found")
            
            future = req.future
        
        # Wait for completion
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request {request_id} timed out after {timeout}s")
    
    def get_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a request."""
        with self._lock:
            # Check all locations
            for req in self.queue:
                if req.id == request_id:
                    return self._request_to_status(req)
            
            if request_id in self.processing:
                return self._request_to_status(self.processing[request_id])
            
            if request_id in self.completed:
                return self._request_to_status(self.completed[request_id])
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status."""
        with self._lock:
            return {
                "queued": len(self.queue),
                "processing": len(self.processing),
                "completed": len(self.completed),
                "total_capacity": self.config.max_queue_size,
                "available_capacity": self.config.max_queue_size - len(self.queue)
            }
    
    def _request_to_status(self, req: QueuedRequest) -> Dict[str, Any]:
        """Convert request to status dict."""
        status = {
            "id": req.id,
            "status": req.status.value,
            "created_at": req.created_at,
            "model": req.request.model
        }
        
        if req.started_at:
            status["started_at"] = req.started_at
            status["queue_time"] = req.started_at - req.created_at
        
        if req.completed_at:
            status["completed_at"] = req.completed_at
            status["processing_time"] = req.completed_at - req.started_at
            status["total_time"] = req.completed_at - req.created_at
        
        if req.error:
            status["error"] = req.error
        
        return status
    
    def _process_queue_loop(self):
        """Background thread that processes the queue."""
        while not self._shutdown:
            try:
                self._queue_event.wait(timeout=0.1)  # Check periodically
                self._queue_event.clear()
                
                if self._shutdown:
                    break
                
                # Get next batch
                batch = self._get_next_batch()
                if not batch:
                    continue
                
                # Check if we can process
                with self._lock:
                    active_tasks = len(self.processing)
                
                if active_tasks >= self.config.max_concurrent_batches:
                    # Put batch back
                    with self._lock:
                        self.queue = batch + self.queue
                    time.sleep(0.01)  # Small delay before retry
                    continue
                
                # Submit batch for processing
                self.executor.submit(self._process_batch, batch)
                
            except Exception as e:
                logging.error(f"Queue processor error: {e}", exc_info=True)
    
    def _get_next_batch(self) -> Optional[List[QueuedRequest]]:
        """Get the next batch of requests to process."""
        with self._lock:
            if not self.queue:
                return None
            
            # Simple batching: take up to max_batch_size requests with same model
            batch = []
            target_model = self.queue[0].request.model
            
            i = 0
            while i < len(self.queue) and len(batch) < self.config.max_batch_size:
                if self.queue[i].request.model == target_model:
                    batch.append(self.queue.pop(i))
                else:
                    i += 1
            
            if batch:
                # Move to processing
                for req in batch:
                    req.status = RequestStatus.PROCESSING
                    req.started_at = time.time()
                    self.processing[req.id] = req
                
                logging.info(f"Created batch of {len(batch)} requests for model {target_model}")
            
            return batch if batch else None
    
    def _process_batch(self, batch: List[QueuedRequest]):
        """Process a batch of requests."""
        try:
            # For now, process sequentially within batch
            # TODO: Implement true parallel processing when models support it
            for req in batch:
                try:
                    self._process_single_request(req)
                except Exception as e:
                    logging.error(f"Failed to process request {req.id}: {e}")
                    req.error = str(e)
                    req.status = RequestStatus.FAILED
                    req.completed_at = time.time()
                    
                    # Set future exception
                    if req.future and not req.future.done():
                        asyncio.run_coroutine_threadsafe(
                            self._set_future_exception(req.future, e),
                            asyncio.get_event_loop()
                        )
                
                finally:
                    # Move to completed
                    with self._lock:
                        if req.id in self.processing:
                            del self.processing[req.id]
                        self.completed[req.id] = req
        
        except Exception as e:
            logging.error(f"Batch processing error: {e}", exc_info=True)
    
    def _process_single_request(self, req: QueuedRequest):
        """Process a single request (placeholder - integrate with router)."""
        # This is where we'd integrate with the actual inference
        # For now, just simulate
        logging.info(f"Processing request {req.id}")
        time.sleep(0.1)  # Simulate processing
        
        # Mock response
        req.result = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.request.model,
            "choices": [{"message": {"role": "assistant", "content": "Mock response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        
        req.status = RequestStatus.COMPLETED
        req.completed_at = time.time()
        
        # Set future result
        if req.future and not req.future.done():
            asyncio.run_coroutine_threadsafe(
                self._set_future_result(req.future, req.result),
                asyncio.get_event_loop()
            )
    
    async def _set_future_result(self, future: asyncio.Future, result: Any):
        """Set future result in async context."""
        if not future.done():
            future.set_result(result)
    
    async def _set_future_exception(self, future: asyncio.Future, exception: Exception):
        """Set future exception in async context."""
        if not future.done():
            future.set_exception(exception)
    
    def shutdown(self):
        """Shutdown the queue manager."""
        logging.info("Shutting down queue manager...")
        self._shutdown = True
        self._queue_event.set()
        self._processor_thread.join(timeout=5)
        self.executor.shutdown(wait=True)