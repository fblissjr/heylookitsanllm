# tests/unit/test_abort.py
"""Tests for the cooperative abort mechanism."""


import pytest

from heylook_llm.providers.abort import AbortEvent


class TestAbortEvent:
    """AbortEvent basic behavior."""

    # ops applied in order to a fresh event -> is_set() afterwards; the repr
    # rows check the state word instead.
    @pytest.mark.parametrize("ops, expect_set, repr_word", [
        ((), False, None),
        (("set",), True, None),
        (("set", "clear"), False, None),
        (("set", "set"), True, None),
        (("clear", "clear"), False, None),
        ((), None, "clear"),
        (("set",), None, "set"),
    ], ids=["initial_state_is_clear", "set_makes_is_set_true", "clear_resets_after_set",
            "multiple_sets_are_idempotent", "multiple_clears_are_idempotent",
            "repr_clear", "repr_set"])
    def test_state(self, ops, expect_set, repr_word):
        abort = AbortEvent()
        for op in ops:
            getattr(abort, op)()
        if repr_word is not None:
            assert repr_word in repr(abort)
        else:
            assert abort.is_set() is expect_set
