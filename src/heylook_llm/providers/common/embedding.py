"""Bounded media ownership and vector validation shared by both engines."""
import base64
import binascii
import math
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from heylook_llm.schema.embeddings import (
    EmbeddingResult, EmbeddingVector, EmbeddingWindow, MEDIA_BYTES_BY_TYPE, MAX_REQUEST_MEDIA_BYTES,
)
from heylook_llm.providers.base import InvalidGenerationRequest
from .generation_gate import GenerationCancelled


def check_cancel(abort):
    if abort is not None and abort.is_set():
        raise GenerationCancelled("Embedding request cancelled")


@contextmanager
def resolved_items(request, info, abort):
    total = 0
    with tempfile.TemporaryDirectory(prefix="heylook-embedding-") as folder:
        items = []
        for index, item in enumerate(request.input):
            check_cancel(abort)
            prompt = ""
            if item.prompt_name is not None:
                if item.prompt_name not in info["prompts"]:
                    raise InvalidGenerationRequest(f"input[{index}]: unknown prompt_name {item.prompt_name!r}")
                prompt = info["prompts"][item.prompt_name]
            parts = []
            for block in item.content:
                if block.type not in info["modalities"]:
                    raise InvalidGenerationRequest(f"input[{index}]: model does not support {block.type}")
                if block.type == "text":
                    parts.append({"type": "text", "text": block.text})
                    continue
                source = block.source
                cap = MEDIA_BYTES_BY_TYPE[block.type]
                try:
                    if source.type == "base64":
                        data = base64.b64decode(source.data, validate=True)
                    else:
                        with urllib.request.urlopen(source.url, timeout=30) as response:
                            if response.url.split(":", 1)[0] not in ("http", "https"):
                                raise ValueError("media redirect must use HTTP(S)")
                            data = response.read(cap + 1)
                except (ValueError, OSError, binascii.Error) as exc:
                    raise InvalidGenerationRequest(f"input[{index}]: cannot read {block.type}: {exc}") from exc
                total += len(data)
                if not data or len(data) > cap or total > MAX_REQUEST_MEDIA_BYTES:
                    raise InvalidGenerationRequest(f"input[{index}]: empty media or media byte limit exceeded")
                # Known MIME suffixes help the upstream decoder choose a container;
                # never use a caller-controlled path or arbitrary extension.
                import mimetypes
                suffix = mimetypes.guess_extension(source.media_type) or ".bin"
                path = Path(folder) / f"{index}-{len(parts)}{suffix}"
                path.write_bytes(data)
                part = {"type": block.type, "path": str(path), "media_type": source.media_type}
                if getattr(block, "windows", None) is not None:
                    part["windows"] = block.windows.model_dump()
                parts.append(part)
            items.append((prompt, parts))
        yield items


def finish(request, info, vectors, tokens):
    dimensions = request.dimensions or info["native_dimensions"]
    if dimensions not in info["dimensions"]:
        raise InvalidGenerationRequest(f"Unsupported dimensions: {dimensions}; choose {info['dimensions']}")
    if len(vectors) != len(request.input):
        raise RuntimeError("Embedding engine returned the wrong number of vectors")
    def final(index, vector):
        if len(vector) != info["native_dimensions"] or not all(math.isfinite(v) for v in vector):
            raise RuntimeError(f"input[{index}]: engine returned an invalid embedding")
        vector = vector[:dimensions]
        norm = math.sqrt(math.fsum(v * v for v in vector))
        if not norm:
            raise RuntimeError(f"input[{index}]: engine returned a zero embedding")
        return [v / norm for v in vector]

    data = []
    for index, out in enumerate(vectors):
        # An item whose video asked for windows comes back as {"embedding", "windows"}.
        vector, windows = (out["embedding"], out["windows"]) if isinstance(out, dict) else (out, None)
        data.append(EmbeddingVector(
            index=index, embedding=final(index, vector), prompt_name=request.input[index].prompt_name,
            windows=None if windows is None else [
                EmbeddingWindow(start=w["start"], end=w["end"], frame_times=w["frame_times"],
                                embedding=final(index, w["embedding"])) for w in windows]))
    return EmbeddingResult(model=request.model, dimensions=dimensions, data=data, input_tokens=tokens)
