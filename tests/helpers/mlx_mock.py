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


def real_mlx_available() -> bool:
    """True when the genuine MLX stack imports (i.e. we're on Apple hardware).

    Callers that only need the mock so IMPORTS succeed (contract tests, which
    drive FakeProvider and never touch a real generation path) should skip the
    sys.modules patch entirely when this returns True: patching replaces a
    working package tree with MagicMocks, which both breaks submodule imports
    the mock tree doesn't enumerate and -- for a session/package-scoped patch --
    binds MagicMocks into any heylook module FIRST imported under it, for the
    rest of the process. Tests that assert on MagicMock behavior (the
    ``mock_mlx`` fixture's clients) must keep patching unconditionally.
    """
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        import mlx_vlm  # noqa: F401
    except Exception:
        return False
    return True


def create_mlx_module_mocks() -> dict:
    """Create a complete mock of the mlx module tree and all transitive deps.

    Returns a dict suitable for ``unittest.mock.patch.dict('sys.modules', ...)``.

    Coverage obligation is IMPORT-TIME ONLY: every DOTTED module path any
    heylook module reaches for needs its own key, because ``import a.b`` /
    ``from a.b import x`` looks up ``sys.modules['a.b']`` and a MagicMock ``a``
    is not a package. Plain attribute pulls off an already-mocked module
    (``from mlx_lm.generate import BatchGenerator``) are free. A path missing
    here surfaces as "No module named 'X'; 'Y' is not a package" -- which is
    how ``mlx_lm.tokenizer_utils`` went missing until 2026-08-18. When you add
    a module-level MLX import to src/, add its path here.

    But add ONLY paths a MODULE-LEVEL import needs. Speculatively covering
    function-level optional imports is not free: adding ``mlx.utils``,
    ``mlx_lm.models.base`` and ``mlx_vlm.generate.diffusion`` here turned three
    green unit tests red, because product code treats those as CAPABILITY
    PROBES -- ``MLXProvider._detect_diffusion`` returns False precisely when
    ``mlx_vlm.generate.diffusion`` won't import, and a test pins that. Making
    the mock tree "more complete" makes the absent-dependency branch untestable.

    Covers:
    - mlx, mlx.core, mlx.nn
    - mlx_lm (utils, generate, sample_utils, tokenizer_utils, models,
      models.cache)
    - mlx_vlm (utils, generate, prompt_utils)
    - PIL / PIL.Image
    - transformers (PreTrainedTokenizer)
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
    mock_mlx_lm.tokenizer_utils = MagicMock()

    # A real TYPE, not a MagicMock instance: consumers do
    # `isinstance(tok, TokenizerWrapper)` (generation_core.ensure_gen_tokenizer),
    # and isinstance against an instance raises TypeError -- which is exactly
    # how tests/unit/test_generation_core.py failed in ISOLATION for as long
    # as its first import happened under this mock (ordering review
    # 2026-08-18: the only file of 100 failing alone).
    class _FakeTokenizerWrapper:
        pass

    mock_mlx_lm.tokenizer_utils.TokenizerWrapper = _FakeTokenizerWrapper
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
        "mlx_lm.tokenizer_utils": mock_mlx_lm.tokenizer_utils,
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

    def __init__(self, text: str, token_id: int = 0, peak_memory: float = 0.0):
        self.text = text
        self.token_id = token_id
        self.token = token_id
        # ChunkTelemetry.absorb() reads `peak_memory` off each chunk and keeps
        # the max. Without it here no test could tell a response that carries
        # peak memory from one that silently drops it.
        self.peak_memory = peak_memory
