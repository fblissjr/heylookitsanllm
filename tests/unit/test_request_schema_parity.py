"""Parity between the public wire and the internal request model.

heylook has one internal request model (`ChatRequest`, what providers are
driven with; until v1.79.66 it was also the OpenAI wire) and one public
Anthropic-shaped wire (`MessageCreateRequest`, which converts into it). Most
of their fields overlap, so every new knob is a decision made TWICE -- and
getting it wrong is silent: a knob added to ChatRequest that the wire does not
carry is unreachable by every client, and a wire field ChatRequest does not
carry never reaches the provider, with no error either way.

That has already happened in this repo in the sibling form CLAUDE.md records
(`_SAMPLER_KEYS` drifting from `REQUEST_SAMPLER_FIELDS` and dropping
`reasoning_effort` from the only surface that generates server-side). These
tests are the guard for the schema-level version of it. They do NOT demand the
two models be identical -- they demand every asymmetry be DECLARED, so adding
a knob to one forces a decision about the other rather than a silence.
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

# Fields that live on ONE side ON PURPOSE, each with the reason. A field that
# shows up here without being added deliberately is the bug this file exists to
# catch -- the fix is to mirror it, or to add it here with why not.
#
# History worth keeping: this table once carried the OpenAI route's own knobs
# (processing_mode, return_individual, include_timing, include_performance and
# the five server-side resize params). Two of them had been exempted on
# rationales naming mechanisms that did not exist (an endpoint that was never
# built; a rename that never happened), which is the rot this file exists to
# catch. All of them left with the route in v1.79.66.
INTERNAL_ONLY = {
    "enable_thinking": "spelled `thinking` on Messages (see WIRE_ALIASES)",
    "continue_final_message": (
        "explicit override of the continuation CONVENTION. Messages can still "
        "continue -- ChatRequest.is_continuation() falls back to 'final message "
        "is an assistant turn' -- it just cannot force the flag against that "
        "convention. It is reachable: /v1/conversations/{id}/generate's "
        "continue mode sets it server-side, which is how v3's Save & Continue "
        "works."
    ),
}

MESSAGES_ONLY = {
    "system": "top-level on Messages; a system ROLE in the message array internally",
    "thinking": "the Messages spelling of enable_thinking (see WIRE_ALIASES)",
    "metadata": "Anthropic passthrough; the provider request has no use for it",
    "show_special_tokens": (
        "v1.79.6 display pref: it steers the route's parser (strip_specials) "
        "and is deliberately never forwarded to the provider request "
        "(DESIGN.md §6)."
    ),
}

# Shared fields whose SHAPE legitimately differs -- the wire's own message and
# option types. Everything else shared must agree exactly.
STRUCTURAL_FIELDS = {"messages", "stream_options"}


@pytest.mark.unit
class TestRequestSchemaParity:
    def test_every_sampler_knob_reaches_both_wires(self):
        """The sampler roster is the registry's (REQUEST_SAMPLER_FIELDS), never
        a hand-list here -- a knob added there must appear on both sides or a
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
        openai = set(ChatRequest.model_fields)  # the internal model
        messages = set(MessageCreateRequest.model_fields)

        undeclared_internal = (openai - messages) - set(INTERNAL_ONLY)
        undeclared_messages = (messages - openai) - set(MESSAGES_ONLY)
        assert not undeclared_internal, (
            f"on ChatRequest only, undeclared: {sorted(undeclared_internal)} "
            "-- no client can reach it; mirror onto MessageCreateRequest, or add "
            "to INTERNAL_ONLY with why not")
        assert not undeclared_messages, (
            f"on /v1/messages only, undeclared: {sorted(undeclared_messages)} "
            "-- mirror onto ChatRequest, or add to MESSAGES_ONLY with why not")

        # And the tables must not rot the other way: an entry for a field that
        # is now on both sides (or on neither) is a stale claim.
        stale_internal = set(INTERNAL_ONLY) - (openai - messages)
        stale_messages = set(MESSAGES_ONLY) - (messages - openai)
        assert not stale_internal, f"INTERNAL_ONLY names non-exclusive fields: {sorted(stale_internal)}"
        assert not stale_messages, f"MESSAGES_ONLY names non-exclusive fields: {sorted(stale_messages)}"

    def test_declared_asymmetries_carry_a_reason(self):
        for table, name in ((INTERNAL_ONLY, "INTERNAL_ONLY"), (MESSAGES_ONLY, "MESSAGES_ONLY")):
            for field, reason in table.items():
                assert reason.strip(), f"{name}[{field!r}] has no reason"

    def test_shared_knobs_agree_on_type_and_default(self):
        """A knob present on both sides with different bounds is the same
        defect one step later: a value the wire accepts and the converter
        cannot carry.
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
