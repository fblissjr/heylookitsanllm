"""Parity between the two PUBLIC request schemas.

heylook exposes one internal request model (`ChatRequest`, which is also the
OpenAI wire and what providers are driven with) and one Anthropic-shaped wire
(`MessageCreateRequest`, which converts into it). 19 of their fields overlap,
so every new knob is a decision made TWICE -- and getting it wrong is silent:
the field simply does not exist on the other surface, and the client that
speaks it loses the control with no error.

That has already happened in this repo in the sibling form CLAUDE.md records
(`_SAMPLER_KEYS` drifting from `REQUEST_SAMPLER_FIELDS` and dropping
`reasoning_effort` from the only surface that generates server-side). These
tests are the guard for the schema-level version of it. They do NOT demand the
two wires be identical -- they demand every asymmetry be DECLARED, so adding a
knob to one wire forces a decision about the other rather than a silence.
"""

import pytest

from heylook_llm.config import ChatRequest
from heylook_llm.samplers import REQUEST_SAMPLER_FIELDS
from heylook_llm.schema.messages import MessageCreateRequest


# Knobs that exist on BOTH wires under different names. The Messages wire
# renames deliberately (its own docstring: "thinking is a top-level bool
# instead of enable_thinking"); v3 derives the rename in messagesParams()
# rather than keeping a second bag.
WIRE_ALIASES = {"enable_thinking": "thinking"}

# Fields that live on ONE wire ON PURPOSE, each with the reason. A field that
# shows up here without being added deliberately is the bug this file exists to
# catch -- the fix is to mirror it, or to add it here with why not.
OPENAI_ONLY = {
    "enable_thinking": "spelled `thinking` on Messages (see WIRE_ALIASES)",
    "continue_final_message": (
        "explicit override of the continuation CONVENTION. Messages can still "
        "continue -- ChatRequest.is_continuation() falls back to 'final message "
        "is an assistant turn' -- it just cannot force the flag against that "
        "convention. v3's chat does not need to: /generate owns continue mode."
    ),
    "processing_mode": "batch-only; moves to BatchRequest on the Messages side",
    "return_individual": "batch-only; moves to BatchRequest on the Messages side",
    "resize_max": "image preprocessing; Messages handles it at /v1/messages/multipart",
    "resize_width": "image preprocessing; Messages handles it at /v1/messages/multipart",
    "resize_height": "image preprocessing; Messages handles it at /v1/messages/multipart",
    "image_quality": "image preprocessing; Messages handles it at /v1/messages/multipart",
    "preserve_alpha": "image preprocessing; Messages handles it at /v1/messages/multipart",
    "include_timing": "spelled `include_performance` on Messages",
}

MESSAGES_ONLY = {
    "system": "top-level on Messages; a system ROLE in the array on the OpenAI wire",
    "thinking": "the Messages spelling of enable_thinking (see WIRE_ALIASES)",
    "metadata": "Anthropic passthrough; no OpenAI equivalent",
    "show_special_tokens": (
        "v1.79.6 display pref, deliberately NOT on the OpenAI wire: it is v3's "
        "toggle, and /v1/chat/completions is the interop surface where the "
        "declared-specials strip stays unconditional (DESIGN.md §6)."
    ),
}

# Shared fields whose SHAPE legitimately differs -- the wire's own message and
# option types. Everything else shared must agree exactly.
STRUCTURAL_FIELDS = {"messages", "stream_options"}


@pytest.mark.unit
class TestRequestSchemaParity:
    def test_every_sampler_knob_reaches_both_wires(self):
        """The sampler roster is the registry's (REQUEST_SAMPLER_FIELDS), never
        a hand-list here -- a knob added there must appear on both wires or a
        Messages client silently loses it."""
        missing = []
        for field in REQUEST_SAMPLER_FIELDS:
            if field not in ChatRequest.model_fields:
                missing.append(f"ChatRequest is missing {field!r}")
            messages_name = WIRE_ALIASES.get(field, field)
            if messages_name not in MessageCreateRequest.model_fields:
                missing.append(
                    f"MessageCreateRequest is missing {messages_name!r} "
                    f"(sampler knob {field!r})")
        assert not missing, "\n".join(missing)

    def test_asymmetries_are_declared(self):
        """Every field on one wire and not the other must be in the tables
        above WITH a reason. Failing here is not "fix the test" -- it means a
        knob landed on one surface and the other was never considered."""
        openai = set(ChatRequest.model_fields)
        messages = set(MessageCreateRequest.model_fields)

        undeclared_openai = (openai - messages) - set(OPENAI_ONLY)
        undeclared_messages = (messages - openai) - set(MESSAGES_ONLY)
        assert not undeclared_openai, (
            f"on /v1/chat/completions only, undeclared: {sorted(undeclared_openai)} "
            "-- mirror onto MessageCreateRequest, or add to OPENAI_ONLY with why not")
        assert not undeclared_messages, (
            f"on /v1/messages only, undeclared: {sorted(undeclared_messages)} "
            "-- mirror onto ChatRequest, or add to MESSAGES_ONLY with why not")

        # And the tables must not rot the other way: an entry for a field that
        # is now on both wires (or on neither) is a stale claim.
        stale_openai = set(OPENAI_ONLY) - (openai - messages)
        stale_messages = set(MESSAGES_ONLY) - (messages - openai)
        assert not stale_openai, f"OPENAI_ONLY names non-exclusive fields: {sorted(stale_openai)}"
        assert not stale_messages, f"MESSAGES_ONLY names non-exclusive fields: {sorted(stale_messages)}"

    def test_declared_asymmetries_carry_a_reason(self):
        for table, name in ((OPENAI_ONLY, "OPENAI_ONLY"), (MESSAGES_ONLY, "MESSAGES_ONLY")):
            for field, reason in table.items():
                assert reason.strip(), f"{name}[{field!r}] has no reason"

    def test_shared_knobs_agree_on_type_and_default(self):
        """A knob present on both wires with different bounds is the same
        defect one step later: a value one API accepts and the other 422s.
        (The reasoning_effort Literal is shared for exactly this reason.)"""
        mismatches = []
        for field in sorted(set(ChatRequest.model_fields) & set(MessageCreateRequest.model_fields)):
            if field in STRUCTURAL_FIELDS:
                continue
            a = ChatRequest.model_fields[field]
            b = MessageCreateRequest.model_fields[field]
            if str(a.annotation) != str(b.annotation):
                mismatches.append(f"{field}: {a.annotation} vs {b.annotation}")
            elif a.default != b.default:
                mismatches.append(f"{field}: default {a.default!r} vs {b.default!r}")
        assert not mismatches, "shared fields disagree:\n" + "\n".join(mismatches)
