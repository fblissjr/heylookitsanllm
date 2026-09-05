# src/heylook_llm/openapi_doc.py
"""The generated OpenAPI document: the narrative description, servers,
external docs and request examples that FastAPI cannot derive. Split out of
api.py in v1.79.67; api.py installs build_openapi as app.openapi."""
from fastapi.openapi.utils import get_openapi

from heylook_llm import __version__
from heylook_llm.config import DEFAULT_PORT


def build_openapi(app):
    """Build (once) and cache the schema on ``app``, as FastAPI's own does."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=f"""
# HeylookLLM API

Local multimodal LLM inference on Apple Silicon. One inference wire, one
model registry, per-model engine choice.

Server version **{__version__}**. Default base URL `http://localhost:{DEFAULT_PORT}`.

> **Integrating an external app?** Read
> [docs/api_integration.md](https://github.com/fblissjr/heylookitsanllm/blob/main/docs/api_integration.md)
> first. This schema is generated from the code and is authoritative for
> field names, types and bounds; that document covers which endpoint to pick
> and what will bite you, which a schema cannot express.

## Providers

- **mlx** -- text and vision, via mlx-lm / mlx-vlm, Metal-accelerated.
- **gguf** -- one `llama-server` subprocess per loaded model. Adds audio
  input; MLX rejects audio (400), because audio towers are stripped at load.
- **mlx_embedding** -- embeddings.

## The inference wire

**`POST /v1/messages`** (Messages API) -- Anthropic Messages-conformant:
top-level `system`, typed content blocks, block-structured SSE ending at
`message_stop` with no `[DONE]`, plus documented heylook extensions
(`heylook_logprobs`, `heylook_progress`, `X-Request-ID` cancellation). This
is the wire the bundled `/v3` frontend speaks. Media blocks use Anthropic's
nested `source`
(`{{"type":"image","source":{{"type":"base64","media_type":...,"data":...}}}}`);
heylook's older flat spelling is still accepted. There is no server-side
resize -- clients downscale before sending. The OpenAI-compatible
`/v1/chat/completions` route was removed in v1.79.66: nothing the project
cares about spoke it, and one generation now has one grammar.

Every sampler knob is optional: **absent means the server-side sampler
cascade decides**. Sending a client-side default for `max_tokens` silently
overrides the model's configured floor.

## Quick start

```bash
curl http://localhost:{DEFAULT_PORT}/v1/models        # what is served right now
curl http://localhost:{DEFAULT_PORT}/v1/capabilities  # version, sampler roster
```

Model ids are **not stable across installs** -- the registry is
override-only, so any model under a scanned folder is served with derived
defaults. Resolve ids from `/v1/models` at runtime and gate features on each
row's `capabilities` (what this server will actually serve) rather than its
`modalities` (what the checkpoint author declared).

```bash
curl http://localhost:{DEFAULT_PORT}/v1/messages \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "<id from /v1/models>",
    "system": "You are concise.",
    "messages": [{{"role": "user", "content": "Hello!"}}],
    "max_tokens": 256
  }}'
```

Vision: an image block with Anthropic's nested `source` (downscale it
yourself first; the server does not resize):

```bash
curl http://localhost:{DEFAULT_PORT}/v1/messages \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "<a model whose capabilities include \\"vision\\">",
    "messages": [{{"role": "user", "content": [
      {{"type": "text", "text": "What is in this image?"}},
      {{"type": "image", "source": {{"type": "base64", "media_type": "image/jpeg", "data": "..."}}}}
    ]}}],
    "max_tokens": 512
  }}'
```

## Client libraries

The Anthropic SDKs reach this server with `base_url` set to its origin (the
SDK appends `/v1/messages` itself); the deliberate differences from
Anthropic's own service are listed in `docs/api_integration.md`.

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:{DEFAULT_PORT}", api_key="not-needed")
response = client.messages.create(
    model="<id from /v1/models>",
    max_tokens=256,
    messages=[{{"role": "user", "content": "Hello!"}}],
)
```

## Errors

- **400** -- unknown or disabled `model`, or none given with no server
  default. The reason and the available ids are in `detail`. Pick another
  model.
- **500** -- the model exists but failed to load. That model is broken.
- **422** -- the request body failed validation. The offending field and
  reason are in `detail`.
- **503** -- generation queue full: `{{"error":{{"code":"model_overloaded"}}}}`
  plus `Retry-After`. Back off and retry; the server serialises generation.
- **In-band SSE `error`** -- a failure after the response headers flushed,
  so the status is already 200. Treat `invalid_request_error` as a 400 and
  `api_error` as a 500, and never render its message as model output.

## Operational notes

- **Startup loads nothing**, and the load runs BEFORE the response begins --
  nothing is written to the connection while it does, on `/v1/messages` and
  the conversation generate route alike. A cold
  model therefore looks like a hang. `POST /v1/models/{{id}}/load` pays it up
  front in a call you can show progress against; it is the same
  `get_provider` the generate call makes, so it adds no work. Add
  `?warm=true` for a readiness call that also pays the first forward pass --
  it takes the generation gate, so keep it out of a per-request pre-flight.
- One model resident by default, LRU eviction; batch work by model.
- Auth is opt-in and off by default: `HEYLOOK_API_KEY`
  (`Authorization: Bearer`, loopback-exempt unless
  `HEYLOOK_API_KEY_ENFORCE_LOOPBACK=true`) gates inference,
  `HEYLOOK_ADMIN_TOKEN` (`X-Heylook-Admin-Token`) gates admin.
- Send `X-Request-ID`. `/v1/messages` and the generate route echo it back, it correlates the
  server-side logs, and it is the handle `DELETE /v1/requests/{{id}}` cancels
  by -- the only way to stop a NON-streaming run, which writes nothing until
  it finishes and so never notices an abandoned client.
- Models are configured in `models.toml`, but entries are overrides only --
  a new download needs no edit.
        """,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    # Add server information
    openapi_schema["servers"] = [
        {
            "url": f"http://localhost:{DEFAULT_PORT}",
            "description": "Default server"
        }
    ]

    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "GitHub Repository",
        "url": "https://github.com/fblissjr/heylookitsanllm"
    }

    # Enhanced component schemas
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}

    # (The OpenAI SSE chunk schemas that used to be appended here left with
    # that route in v1.79.66; the Messages events are declared in schema/.)

    # Add example schemas.
    #
    # Model ids here are PLACEHOLDERS on purpose. A real id is install-local
    # (the registry serves whatever is under the scanned folders), and the
    # concrete ids that used to sit here outlived the models by years --
    # a copy-pasteable example that 400s is worse than an obvious blank.
    _MODEL_PLACEHOLDER = "<id from GET /v1/models>"
    openapi_schema["components"]["examples"] = {
        "messages_text_request": {
            "summary": "Messages wire: top-level system, string content",
            "value": {
                "model": _MODEL_PLACEHOLDER,
                "system": "You are concise.",
                "messages": [{"role": "user", "content": "Tell me a story"}],
                "stream": True,
                "max_tokens": 1024
            }
        },
        "messages_vision_request": {
            "summary": "Messages wire: image block (Anthropic's nested source)",
            "value": {
                "model": "<a model whose capabilities include \"vision\">",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": "<raw base64, no data: prefix>"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 512
            }
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema
