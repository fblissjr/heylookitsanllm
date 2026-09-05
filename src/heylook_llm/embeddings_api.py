# src/heylook_llm/embeddings_api.py
"""POST /v1/embeddings (mlx_embedding provider). The route only; the work is
in embeddings.py. Split out of api.py in v1.79.67."""
import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from heylook_llm.auth import require_api_key
from heylook_llm.providers.common.generation_gate import ModelBusyError

embeddings_router = APIRouter(tags=["Embeddings"])


@embeddings_router.post("/v1/embeddings",
    summary="Create Embeddings",
    description="""
Generate embeddings for text using the specified model.

**Key Features:**
- Extract actual model embeddings (not hallucinated numbers)
- Support for both text-only and vision models
- Multiple pooling strategies (mean, cls, last, max)
- Optional dimension truncation

**Use Cases:**
- Text similarity search
- Semantic clustering
- Cross-modal alignment
- Prompt interpolation
- Document retrieval

**Request Body:**
- `input` (string | array[string]): Text(s) to embed
- `model` (string): Model ID to use
- `dimensions` (integer, optional): Truncate to N dimensions
- `encoding_format` (string, optional): "float" or "base64"
- `user` (string, optional): User identifier
    """,
    response_description="Embeddings in the OpenAI list shape",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "examples": {
                        "single": {
                            "summary": "Single embedding",
                            "value": {
                                "object": "list",
                                "data": [{
                                    "object": "embedding",
                                    "embedding": [0.0234, -0.1567, 0.8901],
                                    "index": 0
                                }],
                                "model": "dolphin-mistral",
                                "usage": {"prompt_tokens": 10, "total_tokens": 10}
                            }
                        }
                    }
                }
            }
        }
    },
    dependencies=[Depends(require_api_key)],
)
async def create_embeddings_endpoint(
    request: Request,
    embedding_request: dict = Body(...)
):
    """Create embeddings for the given input text(s)."""
    from heylook_llm.embeddings import EmbeddingRequest, create_embeddings

    try:
        # Parse request
        req = EmbeddingRequest(**embedding_request)

        # Get router
        router = request.app.state.router_instance

        # Create embeddings
        response = await create_embeddings(req, router)

        return response.model_dump()

    except ModelBusyError:
        raise  # -> app-level 503; a 500 here would call backpressure a failure
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
