"""Native mlx-vlm embeddings, executed only on a leased pinned worker."""
from .base import InvalidGenerationRequest
from .common.embedding import check_cancel


def load(provider):
    # Explicit /load uses the default executor too. Always build MLX state on
    # a persistent worker, even when the caller is not the embedding route.
    from heylook_llm.streaming_utils import _executor_pool
    executor = _executor_pool.acquire()
    try:
        executor.submit(_load, provider).result()
    finally:
        _executor_pool.release(executor)


def _load(provider):
    import mlx.core as mx
    from mlx_vlm import load as load_model
    from mlx_vlm.generate.common import generation_stream
    with mx.stream(generation_stream):
        provider.model, provider.processor = load_model(provider.config["model_path"])
        mx.eval(provider.model.parameters())
    provider.context_length = provider.embedding["max_input_tokens"]


def embed(provider, items, abort):
    import mlx.core as mx
    from mlx_vlm.generate.common import generation_stream
    vectors, tokens = [], 0
    # Item-at-a-time bounds media/padding allocations and permits cancellation
    # between forwards. Upstream handles composition within each item.
    with mx.stream(generation_stream):
        for index, (prompt, parts) in enumerate(items):
            check_cancel(abort)
            content = [{"type": p["type"], **({"text": p["text"]} if p["type"] == "text"
                       else {"url": p["path"]})} for p in parts]
            messages = ([{"role": "system", "content": prompt}] if prompt else [])
            messages.append({"role": "user", "content": content})
            try:
                inputs = provider.processor.apply_chat_template(
                    messages, tokenize=True, return_dict=True, return_tensors="mlx",
                    truncation=False, load_audio_from_video=False,
                )
                length = int(inputs["input_ids"].shape[-1])
                if length > provider.context_length:
                    raise ValueError(f"expanded input has {length} tokens; limit is {provider.context_length}")
                output = provider.model(**inputs).text_embeds
                mx.eval(output)
                vectors.append(output[0].tolist())
                tokens += length
            except (ValueError, TypeError) as exc:
                raise InvalidGenerationRequest(f"input[{index}]: {exc}") from exc
            check_cancel(abort)
    return vectors, tokens
