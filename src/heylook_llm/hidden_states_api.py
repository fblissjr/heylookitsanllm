# src/heylook_llm/hidden_states_api.py
"""POST /v1/hidden_states and /v1/hidden_states/structured. The routes only;
the extraction is in hidden_states.py. Split out of api.py in v1.79.67."""
import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from heylook_llm.auth import require_api_key
from heylook_llm.providers.common.generation_gate import ModelBusyError

hidden_states_router = APIRouter(tags=["Hidden States"])


@hidden_states_router.post("/v1/hidden_states",
    summary="Extract Hidden States",
    description="""
Extract raw hidden states from a specific layer of an LLM model.

**Key Differences from /v1/embeddings:**
- Returns full sequence [seq_len, hidden_dim], not pooled
- Extracts from specific layer (default: -2, second-to-last)
- Filters out padding tokens via attention mask
- Designed for use as text encoder backend for image generation

**Use Cases:**
- Text encoder for DiT-based image generation (Z-Image, etc.)
- Model interpretability and analysis
- Cross-modal alignment with per-token embeddings

**Request Body:**
- `input` (string | array[string]): Text(s) to encode (with chat template applied)
- `model` (string): Model ID to use
- `layer` (integer, optional): Layer to extract from (default: -2)
- `max_length` (integer, optional): Max sequence length (default: 512)
- `return_attention_mask` (boolean, optional): Include attention mask
- `encoding_format` (string, optional): "float" (default) or "base64"

**Note:** Only supported for MLX models.
    """,
    response_description="Hidden states with shape metadata",
    responses={
        200: {
            "description": "Hidden states extracted successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "float_format": {
                            "summary": "Float format response",
                            "value": {
                                "hidden_states": [[0.123, -0.456], [0.789, 0.012]],
                                "shape": [2, 2560],
                                "model": "Qwen3-4B-mxfp4-mlx",
                                "layer": -2,
                                "dtype": "bfloat16"
                            }
                        },
                        "base64_format": {
                            "summary": "Base64 format response",
                            "value": {
                                "hidden_states": "SGVsbG8gV29ybGQ=",
                                "shape": [21, 2560],
                                "model": "Qwen3-4B-mxfp4-mlx",
                                "layer": -2,
                                "dtype": "bfloat16",
                                "encoding_format": "base64"
                            }
                        }
                    }
                }
            }
        },
        422: {
            "description": "Model doesn't support hidden state extraction",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Hidden state extraction is not supported for this model."
                    }
                }
            }
        }
    },
    dependencies=[Depends(require_api_key)],
)
async def extract_hidden_states_endpoint(
    request: Request,
    hidden_states_request: dict = Body(...)
):
    """Extract hidden states from the specified layer of an LLM."""
    from heylook_llm.hidden_states import HiddenStatesRequest, create_hidden_states

    try:
        # Parse request
        req = HiddenStatesRequest(**hidden_states_request)

        # Get router
        router = request.app.state.router_instance

        # Extract hidden states
        response = await create_hidden_states(req, router)

        return response.model_dump(exclude_none=True)

    except NotImplementedError as e:
        # Model doesn't support hidden state extraction
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        # Invalid request parameters
        raise HTTPException(status_code=400, detail=str(e))
    except ModelBusyError:
        raise  # -> app-level 503; a 500 here would call backpressure a failure
    except Exception as e:
        logging.error(f"Error extracting hidden states: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@hidden_states_router.post("/v1/hidden_states/structured",
    summary="Extract Structured Hidden States",
    description="""
Extract hidden states with server-side chat template application and token boundary tracking.

**Key Differences from /v1/hidden_states:**
- Accepts chat components separately (user_prompt, system_prompt, etc.)
- Server applies Qwen3 chat template internally
- Returns token boundary information for each section
- Supports pre-filled thinking/assistant content

**Use Cases:**
- Z-Image embeddings with precise template control
- Token attribution research
- Ablation studies on prompt sections
- Debugging chat template formatting

**Request Body:**
- `model` (string): Model ID to use
- `user_prompt` (string): User message content (required)
- `system_prompt` (string, optional): System prompt content
- `thinking_content` (string, optional): Pre-filled thinking block
- `assistant_content` (string, optional): Pre-filled assistant response
- `enable_thinking` (boolean, optional): Control thinking mode (default: true)
- `layer` (integer, optional): Layer to extract from (default: -2)
- `max_length` (integer, optional): Max sequence length (default: 512)
- `encoding_format` (string, optional): "float" (default) or "base64"
- `return_token_boundaries` (boolean, optional): Return token indices per section
- `return_formatted_prompt` (boolean, optional): Return formatted prompt string

**Note:** Only supported for MLX models with Qwen3-style chat templates.
    """,
    response_description="Hidden states with token boundaries",
    responses={
        200: {
            "description": "Structured hidden states extracted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "hidden_states": "SGVsbG8gV29ybGQ=",
                        "shape": [120, 2560],
                        "model": "Qwen3-4B",
                        "layer": -2,
                        "dtype": "bfloat16",
                        "encoding_format": "base64",
                        "token_boundaries": {
                            "system": {"start": 0, "end": 35},
                            "user": {"start": 35, "end": 80}
                        },
                        "token_counts": {
                            "system": 35,
                            "user": 45,
                            "total": 120
                        }
                    }
                }
            }
        },
        422: {
            "description": "Model doesn't support structured hidden state extraction",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Structured hidden states only supported for MLX models."
                    }
                }
            }
        }
    },
    dependencies=[Depends(require_api_key)],
)
async def extract_structured_hidden_states(
    request: Request,
    structured_request: dict = Body(...)
):
    """Extract structured hidden states with server-side chat template and token boundaries."""
    from heylook_llm.hidden_states import (
        StructuredHiddenStatesRequest,
        create_structured_hidden_states,
    )

    try:
        req = StructuredHiddenStatesRequest(**structured_request)
        router = request.app.state.router_instance
        response = await create_structured_hidden_states(req, router)
        return response.model_dump(exclude_none=True)

    except NotImplementedError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelBusyError:
        raise  # -> app-level 503; a 500 here would call backpressure a failure
    except Exception as e:
        logging.error(f"Error extracting structured hidden states: {e}")
        raise HTTPException(status_code=500, detail=str(e))
