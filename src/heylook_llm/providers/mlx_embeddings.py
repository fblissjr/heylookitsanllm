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
        materialize_state(provider.model)
    provider.context_length = provider.embedding["max_input_tokens"]


def materialize_state(model):
    """Detach every array from the loading worker's lazy stream graph."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    # parameters() excludes private buffers (e.g. rotary _frequencies).
    # Those are model state too, and inference may lease a different worker.
    mx.eval([value for _, value in tree_flatten(model.state) if isinstance(value, mx.array)])


def _messages(prompt, parts):
    content = [{"type": p["type"], **({"text": p["text"]} if p["type"] == "text"
               else {"url": p["path"]})} for p in parts]
    messages = ([{"role": "system", "content": prompt}] if prompt else [])
    messages.append({"role": "user", "content": content})
    return messages


def _render(provider, conversations):
    return provider.processor.apply_chat_template(
        conversations, tokenize=True, return_dict=True, return_tensors="mlx",
        padding=True, truncation=False, load_audio_from_video=False,
    )


def _too_long(provider, index, length):
    return InvalidGenerationRequest(
        f"input[{index}]: expanded input has {length} tokens; limit is {provider.context_length}")


VISUAL_PATCH_KEYS = (("pixel_values", "image_position_ids"),
                     ("pixel_values_videos", "video_position_ids"))


def trim_visual_padding(inputs):
    """Drop trailing patch slots that no image or frame uses.

    The processor pads every grid to the largest budget (position id -1); the
    vision tower masks those slots, so cutting them changes no valid patch and
    shortens attention. Rows keep their own padding up to the longest one.
    Measured: internal/claude/embeddinggemma2_probes/vision_candidates.txt (C4).
    """
    import mlx.core as mx
    for values, positions in VISUAL_PATCH_KEYS:
        ids = inputs.get(positions)
        if ids is None or ids.ndim != 3:
            continue
        n = int(mx.sum(mx.all(ids != -1, axis=-1), axis=-1).max().item())
        if 0 < n < ids.shape[1]:
            inputs[values] = inputs[values][:, :n]
            inputs[positions] = ids[:, :n]
    return inputs


def _embed_one(provider, index, item, vectors):
    import mlx.core as mx
    try:
        inputs = _render(provider, _messages(*item))
        length = int(inputs["input_ids"].shape[-1])
        if length > provider.context_length:
            raise _too_long(provider, index, length)
        output = provider.model(**trim_visual_padding(dict(inputs))).text_embeds
        mx.eval(output)
    except (ValueError, TypeError) as exc:
        raise InvalidGenerationRequest(f"input[{index}]: {exc}") from exc
    vectors[index] = output[0].tolist()
    return length


def _embed_text_rows(provider, rows, items, vectors, abort):
    """All text-only items in one render, then padded forwards in chunks.

    Each row's own length comes from its attention mask, so token counts and
    the context check equal the single-item path. A chunk holds consecutive
    rows while rows * longest row stays within one context of padded tokens.
    """
    import mlx.core as mx
    try:
        inputs = _render(provider, [_messages(*items[i]) for i in rows])
    except (ValueError, TypeError):
        return sum(_embed_one(provider, i, items[i], vectors) for i in rows)  # names the bad index
    lengths = [int(n) for n in inputs["attention_mask"].sum(axis=-1).tolist()]
    for i, n in zip(rows, lengths):
        if n > provider.context_length:
            raise _too_long(provider, i, n)
    left = getattr(provider.processor.tokenizer, "padding_side", "right") == "left"
    start = 0
    while start < len(rows):
        end, longest = start, 0
        while end < len(rows) and (end - start + 1) * max(longest, lengths[end]) <= provider.context_length:
            longest = max(longest, lengths[end]); end += 1
        end = max(end, start + 1)
        longest = max(lengths[start:end])
        cols = slice(-longest, None) if left else slice(0, longest)
        output = provider.model(input_ids=inputs["input_ids"][start:end, cols],
                                attention_mask=inputs["attention_mask"][start:end, cols]).text_embeds
        mx.eval(output)
        for i, vector in zip(rows[start:end], output.tolist()):
            vectors[i] = vector
        start = end
        check_cancel(abort)
    return sum(lengths)


def embed(provider, items, abort):
    import mlx.core as mx
    from mlx_vlm.generate.common import generation_stream
    vectors, tokens = [None] * len(items), 0
    text_rows = [i for i, (_, parts) in enumerate(items) if all(p["type"] == "text" for p in parts)]
    text_set = set(text_rows)
    # Text batches; media items stay one forward each, which bounds media
    # allocations and permits cancellation between them. Upstream handles
    # composition within each item.
    with mx.stream(generation_stream):
        check_cancel(abort)
        if text_rows:
            tokens += _embed_text_rows(provider, text_rows, items, vectors, abort)
        for index, item in enumerate(items):
            if index in text_set:
                continue
            check_cancel(abort)
            tokens += _embed_one(provider, index, item, vectors)
            check_cancel(abort)
    return vectors, tokens
