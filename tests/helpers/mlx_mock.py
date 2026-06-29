# tests/helpers/mlx_mock.py
#
# Reusable MLX mocking utilities for tests that need to import provider code
# without having MLX installed.  Uses sys.modules patching for the full
# MLX module tree and all transitive dependencies.
#
# WARNING: ALWAYS apply these mocks with a scoped `with patch.dict(sys.modules, ...)`
# context (or the `mock_mlx` fixture in conftest.py). NEVER call
# `patch.dict(...).start()` at module level -- pytest imports every test module
# during collection, so a module-level start() replaces real `mlx`/`mlx_lm` with
# MagicMocks for the ENTIRE session, silently breaking every later real-MLX test.
# That leak caused ~50 spurious "Metal context" failures until it was scoped.

from unittest.mock import MagicMock


def create_mlx_module_mocks() -> dict:
    """Create a complete mock of the mlx module tree and all transitive deps.

    Returns a dict suitable for ``unittest.mock.patch.dict('sys.modules', ...)``.
    Covers:
    - mlx, mlx.core, mlx.nn
    - mlx_lm (utils, generate, sample_utils, models, models.cache)
    - mlx_vlm (utils, generate, prompt_utils)
    - PIL / PIL.Image
    - transformers (PreTrainedTokenizer)
    - numpy (partial -- enough for import to succeed)
    """
    mock_mx = MagicMock()
    mock_mx.core = MagicMock()
    mock_mx.nn = MagicMock()
    # mx.new_thread_local_stream and mx.default_device are called at module
    # level (the provider imports `mlx.core as mx`, i.e. mock_mx.core).
    mock_mx.core.new_thread_local_stream.return_value = MagicMock()
    mock_mx.core.default_device.return_value = MagicMock()

    # mlx_lm tree
    mock_mlx_lm = MagicMock()
    mock_mlx_lm.utils = MagicMock()
    mock_mlx_lm.generate = MagicMock()
    mock_mlx_lm.generate.stream_generate = MagicMock()
    mock_mlx_lm.generate.wired_limit = MagicMock()
    mock_mlx_lm.sample_utils = MagicMock()
    mock_mlx_lm.sample_utils.make_sampler = MagicMock()
    mock_mlx_lm.sample_utils.make_logits_processors = MagicMock(return_value=[])
    mock_mlx_lm.models = MagicMock()
    mock_mlx_lm.models.cache = MagicMock()
    mock_mlx_lm.models.cache.KVCache = MagicMock()
    mock_mlx_lm.models.cache.QuantizedKVCache = MagicMock()
    mock_mlx_lm.models.cache.RotatingKVCache = MagicMock()
    mock_mlx_lm.models.cache.trim_prompt_cache = MagicMock()
    mock_mlx_lm.models.cache.can_trim_prompt_cache = MagicMock(return_value=False)

    # mlx_vlm tree
    mock_mlx_vlm = MagicMock()
    mock_mlx_vlm.utils = MagicMock()
    mock_mlx_vlm.generate = MagicMock()
    mock_mlx_vlm.stream_generate = MagicMock()
    mock_mlx_vlm.prompt_utils = MagicMock()
    mock_mlx_vlm.prompt_utils.apply_chat_template = MagicMock(return_value="formatted prompt")

    mock_pil = MagicMock()
    mock_transformers = MagicMock()

    modules = {
        # MLX core
        "mlx": mock_mx,
        "mlx.core": mock_mx.core,
        "mlx.nn": mock_mx.nn,
        # mlx_lm
        "mlx_lm": mock_mlx_lm,
        "mlx_lm.utils": mock_mlx_lm.utils,
        "mlx_lm.generate": mock_mlx_lm.generate,
        "mlx_lm.sample_utils": mock_mlx_lm.sample_utils,
        "mlx_lm.models": mock_mlx_lm.models,
        "mlx_lm.models.cache": mock_mlx_lm.models.cache,
        # mlx_vlm
        "mlx_vlm": mock_mlx_vlm,
        "mlx_vlm.utils": mock_mlx_vlm.utils,
        "mlx_vlm.generate": mock_mlx_vlm.generate,
        "mlx_vlm.prompt_utils": mock_mlx_vlm.prompt_utils,
        # PIL
        "PIL": mock_pil,
        "PIL.Image": mock_pil.Image,
        # transformers
        "transformers": mock_transformers,
    }
    return modules


def create_mock_model():
    """Create a mock MLX model object with standard methods."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.max_position_embeddings = 32768
    model.parameters.return_value = []
    return model


def create_mock_vlm_model():
    """Mock VLM model with a `.language_model` sub-model.

    VLM code paths (wrapper construction, position reset, vision strategy) reach
    through `model.language_model`, which a bare create_mock_model() doesn't set.
    Use this for any test exercising VLM provider behavior so the sub-model isn't
    forgotten.
    """
    model = create_mock_model()
    model.language_model = create_mock_model()
    return model


def create_mock_tokenizer():
    """Create a mock tokenizer with apply_chat_template, encode, decode."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    tokenizer.encode.return_value = [1, 2, 3, 4]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.eos_token_id = 0
    return tokenizer


def create_mock_processor(with_tokenizer: bool = True):
    """Create a mock VLM processor mirroring mlx-vlm's processor shape.

    Real mlx-vlm processors expose `.tokenizer` and do NOT have a `_tokenizer`
    attribute -- that lives on the inner TokenizerWrapper. A bare MagicMock
    fabricates every attribute on access, so `processor._tokenizer` would be a
    phantom auto-mock. BaseProvider.get_tokenizer() checks `_tokenizer` first,
    so it would return that phantom (whose encode() isn't iterable) instead of
    the real `.tokenizer`, silently breaking any code that tokenizes -- e.g.
    provider.warmup(). Delete `_tokenizer` so get_tokenizer falls through to
    `.tokenizer`, matching real processor behavior.
    """
    processor = MagicMock()
    del processor._tokenizer  # match real processors: no private _tokenizer attr
    if with_tokenizer:
        processor.tokenizer = create_mock_tokenizer()
    else:
        # Model a processor with NO usable tokenizer. get_tokenizer() falls back
        # through _tokenizer -> tokenizer -> decode(); a bare MagicMock fabricates
        # all three, so delete every fallback or get_tokenizer returns the
        # processor itself instead of None.
        del processor.tokenizer
        del processor.decode
    return processor


class FakeChunk:
    """Fake generation chunk with .text and .token_id attributes."""

    def __init__(self, text: str, token_id: int = 0):
        self.text = text
        self.token_id = token_id
        self.token = token_id
