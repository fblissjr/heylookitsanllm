"""Translate the common item grammar to llama-server's multimodal endpoint."""
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

from .base import InvalidGenerationRequest
from .common.embedding import check_cancel


def embed(provider, items, abort):
    vectors, tokens = [], 0
    for index, (prompt, parts) in enumerate(items):
        check_cancel(abort)
        content = [{"type": "text", "text": prompt}] if prompt else []
        for part in parts:
            kind = part["type"]
            if kind == "text":
                content.append(part)
            else:
                data = base64.b64encode(Path(part["path"]).read_bytes()).decode("ascii")
                uri = f"data:{part['media_type']};base64,{data}"
                if kind == "image":
                    content.append({"type": "image_url", "image_url": {"url": uri}})
                else:
                    key = "input_" + kind
                    content.append({"type": key, key: {"data": uri}})
        body = {"input": [{"content": content}], "encoding_format": "float"}
        req = urllib.request.Request(provider._base_url + "/v1/embeddings",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            # Hold the gate until the non-streaming forward completes. An abort
            # never assumes that closing the socket stops a GPU kernel.
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode(errors="replace")
            raise InvalidGenerationRequest(f"input[{index}]: llama-server {exc.code}: {detail}") from exc
        except (TimeoutError, OSError):
            # Unknown backend completion: terminate the subprocess before the
            # shared gate can admit another inference. Do not race its GPU work.
            provider.unload()
            raise
        data = result.get("data", [])
        if len(data) != 1 or "embedding" not in data[0]:
            raise RuntimeError(f"input[{index}]: malformed llama-server embedding response")
        vectors.append(data[0]["embedding"])
        usage = result.get("usage", {}).get("prompt_tokens")
        tokens = tokens + usage if tokens is not None and isinstance(usage, int) else None
        check_cancel(abort)
    return vectors, tokens
