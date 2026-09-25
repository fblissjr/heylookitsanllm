# tests/unit/test_abort.py
"""Tests for the cooperative abort mechanism."""


import pytest

from heylook_llm.providers.abort import AbortEvent


class TestAbortEvent:
    """AbortEvent basic behavior."""

    def test_initial_state_is_clear(self):
        abort = AbortEvent()
        assert not abort.is_set()

    def test_set_makes_is_set_true(self):
        abort = AbortEvent()
        abort.set()
        assert abort.is_set()

    def test_clear_resets_after_set(self):
        abort = AbortEvent()
        abort.set()
        abort.clear()
        assert not abort.is_set()

    def test_multiple_sets_are_idempotent(self):
        abort = AbortEvent()
        abort.set()
        abort.set()
        assert abort.is_set()

    def test_multiple_clears_are_idempotent(self):
        abort = AbortEvent()
        abort.clear()
        abort.clear()
        assert not abort.is_set()

    def test_repr_clear(self):
        abort = AbortEvent()
        assert "clear" in repr(abort)

    def test_repr_set(self):
        abort = AbortEvent()
        abort.set()
        assert "set" in repr(abort)


