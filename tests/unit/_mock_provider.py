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
        # super() FIRST: it is what installs the base-class state every
        # provider is contractually assumed to have -- `_active_generations`
        # and its lock, which the router's teardown guard reads through
        # `active_generations`. Hand-setting three attributes instead meant a
        # new obligation on BaseProvider arrived here as an AttributeError in
        # whichever test first drove the real path, which is exactly the trap
        # tests/contract/conftest.py's FakeProvider documents avoiding.
        super().__init__(model_id, model_config, is_debug)
        # Kept as their own names: the router and these tests read
        # `model_config`/`is_debug`, while BaseProvider spells them
        # `config`/`verbose`.
        self.model_config = model_config
        self.is_debug = is_debug
        self.unload = MagicMock()

    def load_model(self):  # pragma: no cover -- tests use pre-loaded state
        pass

    def create_chat_completion(self, request, abort_event=None):  # pragma: no cover
        pass
