# MLX Embedding Provider

Last updated: 2026-03-13

## Overview

`MLXEmbeddingProvider` loads sentence-transformer style embedding models on Apple Silicon via MLX. It uses a custom `EmbeddingModel` class that wraps any mlx-lm-compatible transformer backbone with bidirectional attention (additive padding mask), mean pooling, dense projections, and L2 normalization. Supports quantized variants (4-bit, 8-bit).

Registered as provider type `"mlx_embedding"` in `config.py` and `router.py`.

## Key Files

| File | Role |
|---|---|
| `src/heylook_llm/models/embedding_model.py` | `EmbeddingModel`, `load_backbone()`, `mean_pooling`, `normalize_embeddings`, `_make_padding_mask` |
| `src/heylook_llm/providers/mlx_embedding_provider.py` | `MLXEmbeddingProvider`, task prefixes, quantized loading |
| `tests/unit/test_embedding_model.py` | Model-level tests (pooling, norm, sanitize, forward, padding mask) |
| `tests/unit/test_mlx_embedding_provider.py` | Provider-level tests (init, prefixes, embeddings, tokenizer, quantization) |

## Architecture

### Dynamic Backbone Loading

The backbone is loaded dynamically via `mlx_lm.utils._get_classes()`, which resolves the model architecture from `config.json`. This means `EmbeddingModel` works with any architecture that mlx-lm supports -- it is not tied to Gemma3.

```python
def load_backbone(model_config: dict):
    from mlx_lm.utils import _get_classes
    ModelClass, ArgsClass = _get_classes(model_config)
    # filter config dict to valid keys, instantiate
    return model, args
```

### Forward Pass

```
input_ids
  -> backbone embed_tokens -> (architecture-specific scaling)
  -> _make_padding_mask(attention_mask) -> (B, 1, 1, seq_len) additive mask
  -> N transformer layers (padding mask, not causal)
  -> final norm
  -> mean_pooling(hidden_states, attention_mask)
  -> Dense(hidden_size -> proj_size, no bias, identity activation)
  -> Dense(proj_size -> output_size, no bias, identity activation)
  -> L2 normalize
  -> unit-normalized embedding vector
```

The dense projection layer sizes are read from the model checkpoint (`2_Dense/config.json`, `3_Dense/config.json`).

### Bidirectional Attention

The critical difference from generative models: no causal mask. Every token attends to every other real token. This is required for embedding quality -- a causal model cannot build a representation of token N that incorporates tokens N+1 through end.

### Padding Mask

`_make_padding_mask()` creates a `(B, 1, 1, seq_len)` additive float mask:
- `0.0` for real tokens (attend normally)
- `-inf` (`finfo.min`) for padding tokens (softmax drives attention weight to zero)

The shape broadcasts across all attention heads and all query positions. When no padding is present, the mask is `None` for zero overhead.

**Why this matters**: Without a padding mask, padding tokens contaminate content token representations through self-attention. Identical content with different padding produces different embeddings. With the mask, `[1, 2, 3]` and `[1, 2, 3, PAD, PAD]` produce identical embeddings.

### Mean Pooling

Averages hidden states over the sequence dimension, weighted by the attention mask. Padding positions contribute zero to both numerator and denominator. Denominator is clamped to `1e-9` to avoid NaN on fully-masked inputs.

### Weight Loading

`sanitize()` handles two weight key layouts:
1. **Flat HF keys** (`embed_tokens.weight`) -- adds `model.` prefix to match `self.model` (the backbone)
2. **Already-prefixed keys** (`model.embed_tokens.weight`) -- passes through

Removes `lm_head.*` weights. Dense layer weights (`linear.weight`) come from separate safetensors files in subdirectories (`2_Dense/model.safetensors`, `3_Dense/model.safetensors`) and are mapped to `dense_layers.{i}.weight` during loading.

## Quantization

`load_model()` checks the model's `config.json` for a `"quantization"` key (present in mlx-community quantized variants). When found, applies `nn.quantize()` before loading weights:

```python
quantization = model_config.get("quantization")
if quantization is not None:
    nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"])
```

`nn.quantize` replaces `nn.Linear` with `nn.QuantizedLinear` throughout the model. Works because `EmbeddingModel` uses standard mlx-lm layer structure.

Tested with: `mlx-community/embeddinggemma-300m-4bit` (group_size=32, bits=4).

## Task Prefixes

EmbeddingGemma (and compatible models) achieve better retrieval quality with task-specific prefixes. The provider accepts an optional `task` parameter on `get_embeddings()`.

| task key | prefix |
|---|---|
| `query` | `task: search result \| query: ` |
| `document` | `title: none \| text: ` |
| `code_retrieval` | `task: code retrieval \| query: ` |
| `clustering` | `task: clustering \| query: ` |
| `classification` | `task: classification \| query: ` |
| `sentence_similarity` | `task: sentence similarity \| query: ` |
| `summarization` | `task: summarization \| query: ` |

When `task=None` or an unknown task key is passed, text is used as-is.

## Usage via API

```toml
[[models]]
id = "embeddinggemma-300m"
provider = "mlx_embedding"
enabled = true

  [models.config]
  model_path = "/path/to/google_embeddinggemma-300m"
  max_length = 2048
```

Then `POST /v1/embeddings` with `model="embeddinggemma-300m"`.

## Usage as Library

```python
from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

provider = MLXEmbeddingProvider(
    "embeddinggemma",
    {"model_path": "/path/to/google_embeddinggemma-300m"},
    verbose=False,
)
provider.load_model()
embeddings = provider.get_embeddings(["hello world"], task="query")
```

## Design Decisions

- Uses dynamic backbone loading (`load_backbone()` via `mlx_lm.utils._get_classes()`) instead of hardcoded Gemma3 imports. Supports any mlx-lm-compatible architecture.
- Does NOT use mlx-lm's `Model` wrapper (which adds an lm_head). Uses the backbone directly since we don't need the language modeling head.
- Dense layers are plain `nn.Linear` with no activation (HF config calls this "Identity activation").
- `float32` dtype recommended. Some embedding model activations lose precision in `float16`. `bfloat16` is acceptable.
- `create_chat_completion` raises `NotImplementedError` -- embedding-only provider.
- `embeddings.py` dispatches to `provider.get_embeddings()` when available, falls back to the legacy `MLXEmbeddingExtractor` for generative models.

## Performance (Apple Silicon)

- Model load: ~2-5s (local safetensors)
- Embedding throughput: ~18-50 chunks/s depending on sequence length
- Memory: ~600MB for full fp32 model (embeddinggemma-300m)
