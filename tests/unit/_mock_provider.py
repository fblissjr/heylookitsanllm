"""Shared MockProvider for router + idle-unload tests.

Satisfies BaseProvider's abstract contract without touching MLX. Imported
via sibling-dir path injection (pytest adds the test file's parent to
sys.path), same pattern used by ``_fake_request.py`` for auth tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from heylook_llm.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Minimal BaseProvider stand-in. Per-instance ``unload`` is a
    MagicMock so tests can assert it was called during eviction or idle
    unload."""

    def __init__(self, model_id, model_config, is_debug):
        self.model_id = model_id
        self.model_config = model_config
        self.is_debug = is_debug
        self.unload = MagicMock()

    def load_model(self):  # pragma: no cover -- tests use pre-loaded state
        pass

    def create_chat_completion(self, request):  # pragma: no cover
        pass
