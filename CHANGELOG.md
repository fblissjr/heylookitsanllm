# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.3.0

### Added

- **API schema module** (`src/heylook_llm/schema/`): New Anthropic Messages-inspired API schema with typed content blocks (TextBlock, ImageBlock, ThinkingBlock, LogprobsBlock, HiddenStatesBlock), structured streaming events, and bidirectional converters to/from the existing OpenAI-compatible format. Purely additive -- existing endpoints unchanged.
- **Pre-commit safety hook**: Rejects staged files in `internal/`, `coderef/`, `.claude/`, `.archive/`, `modelzoo/*`, `models.toml`, `.env`, and files containing personal filesystem paths.
- **Frontend platform documentation** (`internal/frontend/`): Applet catalog, platform architecture, API schema design, design system, and migration plan.

### Changed

- **Dependency update**: Removed `huggingface-hub<1.0` upper bound pin to allow `transformers>=5.0` (required by latest mlx-lm from git)

## 1.2.0

### Changed

- **Dependency modernization**: Added `[tool.uv.sources]` for mlx-lm, mlx-vlm, and parakeet-mlx to pull latest from git instead of PyPI
- **STT migration**: Replaced CoreML STT provider (coremltools + manual RNNT decoder) with MLX STT provider (parakeet-mlx high-level API)
- **Setup scripts**: Rewrote `setup.sh` and `update-heylook.sh` for uv-only workflow, removed all llama-cpp-python references (deprecated, replaced by llama-server subprocess)

### Removed

- `coreml_stt_provider.py` (465 lines) -- replaced by `mlx_stt_provider.py` (265 lines, parakeet-mlx)
- `CoreMLSTTModelConfig` and `coreml_stt` provider type from config
- `llama-cpp` from `[all]` extra (llama-server subprocess is the supported path)
- pip fallback in setup/update scripts (uv required for `[tool.uv.sources]` git resolution)

### Migration

- Users with `provider = "coreml_stt"` in models.toml must change to `provider = "mlx_stt"` and update model path to a HuggingFace repo ID (e.g., `mlx-community/parakeet-tdt-0.6b-v3`)
- `setup.sh` and `update-heylook.sh` now require uv (no pip fallback)

## 1.1.1

### Added

- **MLX Performance Optimizations**: Following mlx-lm reference patterns for better Metal utilization
  - Compiled presence penalty processor with `@mx.compile` decorator for Metal kernel optimization
  - Compiled vision preprocessing (normalize + transpose) for faster image encoding
  - Pre-computed ImageNet constants as module-level `mx.array` objects
  - Use `mx.broadcast_to` instead of list multiplication for memory-efficient grid creation
  - Replaced blocking `mx.eval` with `mx.async_eval` for better pipeline pipelining
  - Added buffer cache cleanup method with configurable retention
  - Release temporaries before sync points to reduce memory pressure

- **MLX Performance Test Suite**: Comprehensive tests for MLX optimizations
  - `tests/unit/mlx_perf/`: Compilation correctness, type consistency, sync boundary tests
  - `tests/integration/mlx_perf/`: Throughput benchmarks, TTFT tests, memory profiling
  - New pytest markers: `mlx_perf`, `slow` for selective test execution
  - Configurable test model path via `HEYLOOK_TEST_MODEL` env var

- **Structured Hidden States Endpoint**: New `/v1/hidden_states/structured` for server-side chat template
  - Accepts chat components separately (user_prompt, system_prompt, thinking_content, assistant_content)
  - Server applies Qwen3 chat template internally with `enable_thinking` support
  - Returns token boundaries for each section (system, user, think, assistant)
  - Returns token counts per section and total
  - Optional `formatted_prompt` field for debugging
  - Enables ablation studies and token attribution research for Z-Image

- **Model Capabilities Discovery**: Expose model capabilities in `/v1/models` response
  - New `capabilities` field in model config (e.g., `["hidden_states", "chat", "thinking", "vision"]`)
  - New `supports_thinking` and `thinking_token_ids` fields in MLXModelConfig
  - `/v1/models` now includes `provider` and `capabilities` when configured
  - Enables programmatic capability discovery for multi-model clients

- **Auto Model Selection**: Fallback to loaded model when no model specified in request
  - Uses most recently used model from LRU cache
  - Falls back to `default_model` from config if no models loaded
  - Provides clear error message with available models if no default

- **Token-Level Thinking Parser**: More efficient parsing of Qwen3 thinking blocks
  - New `TokenLevelThinkingParser` uses token IDs (151667/151668) for precise detection
  - New `HybridThinkingParser` auto-selects between token-level and text-based parsing
  - Eliminates regex buffering overhead when token IDs are available
  - Instant detection of `<think>`/`</think>` boundaries via special token IDs
  - Backwards compatible: falls back to text parsing for models without token IDs
  - Integrated into streaming response generator for automatic use

- **Hidden States Config Defaults**: Model-level defaults for hidden states extraction
  - New `default_hidden_layer` config option in MLXModelConfig (default: -2)
  - New `default_max_length` config option in MLXModelConfig (default: 512)
  - Hidden states endpoint now applies model config defaults when request uses defaults

- **Enhanced Streaming Metadata**: Detailed generation stats in final streaming chunk
  - New `stream_options: {include_usage: true}` parameter in ChatRequest
  - `usage` object with `prompt_tokens`, `completion_tokens`, `thinking_tokens`, `content_tokens`, `total_tokens`
  - `timing` object with `thinking_duration_ms`, `content_duration_ms`, `total_duration_ms`
  - `generation_config` object with sampler settings used (temperature, top_p, top_k, min_p, max_tokens, enable_thinking)
  - `stop_reason` field indicating why generation stopped: `eos_token`, `max_tokens`, `stop_sequence`, or `length`
  - Properly maps MLX `finish_reason` values to OpenAI-compatible stop reasons
  - Enables frontend display of thinking tokens, timing breakdown, and stop reason

- **OpenAI-Compatible Logprobs Support**: Return token log probabilities in chat completions
  - New `logprobs: bool` parameter to enable log probability output
  - New `top_logprobs: int` parameter to specify number of top alternatives (0-20)
  - Non-streaming responses include `choice.logprobs.content` array with per-token data
  - Streaming responses include `choice.logprobs` delta in each chunk
  - Each token entry includes: `token`, `token_id`, `logprob`, `bytes`, `top_logprobs[]`
  - Leverages mlx-lm's full vocabulary log-softmax for accurate probabilities
  - New `logprobs.py` module with `LogprobsCollector` and `StreamingLogprobsCollector`

### Fixed

- **TextOnlyStrategy model_config**: Fixed AttributeError when `enable_thinking` not in request
  - Added `model_config` parameter to `TextOnlyStrategy.__init__`
  - Changed `getattr()` to `.get()` for proper dict access
  - Fixes Qwen3 thinking mode fallback to model config defaults

- **Qwen3-VL-MOE Vision Support**: Fixed "Image features and image tokens do not match" error
  - Use `mlx_vlm.prompt_utils.apply_chat_template` for proper image token insertion
  - Removed manual image placeholder insertion that conflicted with library handling
  - Now properly passes `num_images` to prompt formatting

### Previously Added

- **Qwen3 Thinking Token Support**: Parse `<think>...</think>` blocks from Qwen3 model outputs
  - Non-streaming responses include `message.thinking` field with parsed reasoning content
  - Streaming responses emit `delta.thinking` during thinking, `delta.content` for response
  - Multiple thinking blocks are concatenated with `---` separators
  - New `thinking_parser.py` module for robust parsing
- **Model Configuration for Thinking Mode**: Add `enable_thinking` parameter to MLXModelConfig
  - When enabled, automatically applies Qwen3 optimal sampler defaults:
    - temperature=0.6 (greedy decoding causes repetition)
    - top_p=0.95
    - top_k=20
    - presence_penalty=1.5
  - Pass `enable_thinking` to chat template for Qwen3 tokenizers
- **Presence Penalty Support**: Add `presence_penalty` parameter to ChatRequest and MLXModelConfig
  - Discourages reuse of tokens that have already appeared
  - Recommended value 1.5 for Qwen3 thinking mode
  - Custom logits processor implementation for mlx-lm compatibility

### Changed

- Updated `models.toml.example` with thinking model configuration documentation
- Extended sampler builder to support presence_penalty processor
