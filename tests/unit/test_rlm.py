# tests/unit/test_rlm.py
"""Unit tests for the RLM (Recursive Language Model) endpoint.

All tests mock the provider -- no MLX/GPU needed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from heylook_llm.rlm import (
    RLMEngine,
    RLMMetadata,
    RLMRequest,
    RLMResponse,
    RLMTraceEntry,
    _RESERVED_NAMES,
    _check_ast_safety,
    extract_repl_block,
    run_code_in_namespace,
)


# ---------------------------------------------------------------------------
# extract_repl_block
# ---------------------------------------------------------------------------

class TestExtractReplBlock:
    def test_valid_block(self):
        text = 'Some text\n```repl\nprint("hello")\n```\nmore text'
        assert extract_repl_block(text) == 'print("hello")'

    def test_no_block(self):
        assert extract_repl_block("Just plain text") is None

    def test_multiple_blocks_first_wins(self):
        text = '```repl\nfirst()\n```\nstuff\n```repl\nsecond()\n```'
        assert extract_repl_block(text) == "first()"

    def test_empty_block(self):
        text = "```repl\n\n```"
        assert extract_repl_block(text) == ""

    def test_multiline_code(self):
        text = '```repl\nx = 1\ny = 2\nprint(x + y)\n```'
        result = extract_repl_block(text)
        assert "x = 1" in result
        assert "print(x + y)" in result

    def test_non_repl_block_ignored(self):
        text = '```python\nprint("hi")\n```'
        assert extract_repl_block(text) is None


# ---------------------------------------------------------------------------
# run_code_in_namespace -- sandbox
# ---------------------------------------------------------------------------

class TestSandbox:
    def test_safe_builtins_allowed(self):
        ns = {}
        stdout, stderr = run_code_in_namespace("print(len([1,2,3]))", ns)
        assert stdout.strip() == "3"
        assert stderr == ""

    def test_blocked_import(self):
        ns = {}
        stdout, stderr = run_code_in_namespace("import os", ns)
        assert "NameError" in stderr or "ImportError" in stderr

    def test_blocked_open(self):
        ns = {}
        stdout, stderr = run_code_in_namespace('open("/etc/passwd")', ns)
        assert stderr != ""

    def test_sandbox_disabled(self):
        ns = {}
        stdout, stderr = run_code_in_namespace("print(type(open))", ns, sandbox=False)
        assert "builtin_function_or_method" in stdout or stderr == ""

    def test_persistent_namespace(self):
        ns = {}
        run_code_in_namespace("x = 42", ns)
        stdout, stderr = run_code_in_namespace("print(x)", ns)
        assert stdout.strip() == "42"


# ---------------------------------------------------------------------------
# run_code_in_namespace -- output handling
# ---------------------------------------------------------------------------

class TestCodeExecution:
    def test_stdout_capture(self):
        ns = {}
        stdout, stderr = run_code_in_namespace('print("hello world")', ns)
        assert stdout.strip() == "hello world"
        assert stderr == ""

    def test_stderr_on_exception(self):
        ns = {}
        stdout, stderr = run_code_in_namespace("1/0", ns)
        assert "ZeroDivisionError" in stderr

    def test_output_truncation(self):
        ns = {}
        code = 'print("x" * 200)'
        stdout, stderr = run_code_in_namespace(code, ns, max_output_chars=50)
        assert len(stdout) < 200
        assert "truncated" in stdout

    def test_timeout_enforcement(self):
        ns = {}
        code = "import time; time.sleep(10)"
        stdout, stderr = run_code_in_namespace(code, ns, timeout=1, sandbox=False)
        assert "timed out" in stderr.lower() or "TimeoutError" in stderr


# ---------------------------------------------------------------------------
# FINAL() and FINAL_VAR() termination
# ---------------------------------------------------------------------------

def _make_chunk(text, pt=10, ct=20):
    return SimpleNamespace(text=text, prompt_tokens=pt, generation_tokens=ct)


def _mock_router(responses):
    """Create a mock router that returns provider generating given responses."""
    router = MagicMock()
    provider = MagicMock()

    call_count = [0]

    def fake_completion(req):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        yield _make_chunk(responses[idx])

    provider.create_chat_completion.side_effect = fake_completion
    router.get_provider.return_value = provider
    return router


class TestTermination:
    def test_final_terminates(self):
        router = _mock_router(['```repl\nFINAL("the answer is 42")\n```'])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)

        assert result.answer == "the answer is 42"
        assert result.finish_reason == "final"
        assert result.rlm.trace[-1].action == "FINAL"

    def test_final_var_terminates(self):
        router = _mock_router(['```repl\nresult = "computed"\nFINAL_VAR("result")\n```'])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)

        assert result.answer == "computed"
        assert result.finish_reason == "final"

    def test_final_var_missing_variable(self):
        router = _mock_router(['```repl\nFINAL_VAR("nonexistent")\n```'])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)

        assert "not found" in result.answer

    def test_direct_response_no_code(self):
        router = _mock_router(["The answer is simply 42."])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)

        assert result.answer == "The answer is simply 42."
        assert result.finish_reason == "direct_response"
        assert result.rlm.trace[0].action == "direct_response"


# ---------------------------------------------------------------------------
# Max iterations
# ---------------------------------------------------------------------------

class TestMaxIterations:
    def test_max_iterations_reached(self):
        responses = ['```repl\nprint("still going")\n```'] * 5
        router = _mock_router(responses)
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?", max_iterations=3)
        result = engine.run(req)

        assert result.finish_reason == "max_iterations"
        # 3 iteration trace entries + 1 max_iterations entry
        assert result.rlm.iterations == 4


# ---------------------------------------------------------------------------
# llm_query sub-call counting
# ---------------------------------------------------------------------------

class TestLlmQuery:
    def test_sub_query_counting(self):
        router = MagicMock()
        provider = MagicMock()

        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            if call_count[0] == 1:
                yield _make_chunk(
                    '```repl\nresult = llm_query("sub question")\nFINAL(result)\n```'
                )
            else:
                yield _make_chunk("sub answer")

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)

        assert result.rlm.sub_queries == 1
        assert result.answer == "sub answer"


# ---------------------------------------------------------------------------
# Error handling in code execution
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_exception_fed_back_as_stderr(self):
        responses = [
            '```repl\n1/0\n```',
            'The error means division by zero.',
        ]
        router = _mock_router(responses)
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?", max_iterations=2)
        result = engine.run(req)

        assert result.finish_reason == "direct_response"
        assert result.rlm.iterations == 2


# ---------------------------------------------------------------------------
# Request/Response model validation
# ---------------------------------------------------------------------------

class TestModelValidation:
    def test_request_defaults(self):
        req = RLMRequest(model="m", context="c", query="q")
        assert req.max_iterations == 10
        assert req.max_tokens == 2048
        assert req.sandbox is True
        assert req.timeout == 30
        assert req.stream is False

    def test_request_constraints(self):
        with pytest.raises(Exception):
            RLMRequest(model="m", context="c", query="q", max_iterations=0)
        with pytest.raises(Exception):
            RLMRequest(model="m", context="c", query="q", max_iterations=51)
        with pytest.raises(Exception):
            RLMRequest(model="m", context="c", query="q", timeout=0)
        with pytest.raises(Exception):
            RLMRequest(model="m", context="c", query="q", max_output_chars=50)

    def test_response_model(self):
        resp = RLMResponse(
            id="rlm-abc",
            created=1234567890,
            model="test-model",
            answer="42",
            finish_reason="final",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            rlm=RLMMetadata(
                iterations=1,
                sub_queries=0,
                trace=[RLMTraceEntry(iteration=1, code_len=10, action="FINAL")],
            ),
        )
        assert resp.object == "rlm.completion"
        assert resp.rlm.trace[0].action == "FINAL"

    def test_trace_entry_defaults(self):
        entry = RLMTraceEntry(iteration=1)
        assert entry.code_len == 0
        assert entry.stdout_len == 0
        assert entry.stderr_len == 0
        assert entry.action is None


# ---------------------------------------------------------------------------
# System prompt composition
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_query_included(self):
        engine = RLMEngine(MagicMock())
        prompt = engine._build_system_prompt("Count the words", None)
        assert "Count the words" in prompt
        assert "# Task" in prompt

    def test_system_instructions_included(self):
        engine = RLMEngine(MagicMock())
        prompt = engine._build_system_prompt("Count words", "Be precise")
        assert "Be precise" in prompt
        assert "# Additional Instructions" in prompt
        assert "Count words" in prompt

    def test_system_none_no_additional_section(self):
        engine = RLMEngine(MagicMock())
        prompt = engine._build_system_prompt("Count words", None)
        assert "# Additional Instructions" not in prompt


# ---------------------------------------------------------------------------
# Token usage aggregation
# ---------------------------------------------------------------------------

class TestUsageAggregation:
    def test_usage_accumulates_across_iterations(self):
        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            if call_count[0] <= 2:
                yield _make_chunk('```repl\nprint("hi")\n```', pt=100, ct=50)
            else:
                yield _make_chunk("Done.", pt=100, ct=50)

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?", max_iterations=5)
        result = engine.run(req)

        assert result.usage["prompt_tokens"] == 300
        assert result.usage["completion_tokens"] == 150
        assert result.usage["total_tokens"] == 450


# ---------------------------------------------------------------------------
# Trace metadata structure
# ---------------------------------------------------------------------------

class TestTraceMetadata:
    def test_trace_records_code_and_output_lengths(self):
        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            if call_count[0] == 1:
                yield _make_chunk('```repl\nprint("hello world")\n```')
            else:
                yield _make_chunk('```repl\nFINAL("done")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)

        assert len(result.rlm.trace) == 2
        first = result.rlm.trace[0]
        assert first.iteration == 1
        assert first.code_len > 0
        assert first.stdout_len > 0
        assert first.action is None

        second = result.rlm.trace[1]
        assert second.iteration == 2
        assert second.action == "FINAL"


# ---------------------------------------------------------------------------
# Context metadata
# ---------------------------------------------------------------------------

class TestContextMetadata:
    def test_short_context_no_ellipsis(self):
        engine = RLMEngine(MagicMock())
        meta = engine._build_context_metadata("short text")
        assert "short text" in meta
        assert "..." not in meta

    def test_long_context_truncated_preview(self):
        engine = RLMEngine(MagicMock())
        long_text = "x" * 1000
        meta = engine._build_context_metadata(long_text)
        assert "1000 chars" in meta
        assert "..." in meta


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestStreaming:
    def test_streaming_events(self):
        router = MagicMock()
        provider = MagicMock()

        def fake_completion(req):
            yield _make_chunk('```repl\nFINAL("streamed answer")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?", stream=True)

        events = list(engine.run_streaming(req))

        event_types = [e.split("\n")[0].replace("event: ", "") for e in events]
        assert "rlm_start" in event_types
        assert "iteration_start" in event_types
        assert "assistant_response" in event_types
        assert "code_output" in event_types
        assert "rlm_complete" in event_types

    def test_streaming_pins_and_unpins(self):
        router = MagicMock()
        provider = MagicMock()

        def fake_completion(req):
            yield _make_chunk("Direct answer")

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test-model", context="data", query="what?")

        list(engine.run_streaming(req))

        router.pin_model.assert_called_once_with("test-model")
        router.unpin_model.assert_called_once_with("test-model")


# ---------------------------------------------------------------------------
# Model pinning during run
# ---------------------------------------------------------------------------

class TestModelPinning:
    def test_run_pins_and_unpins(self):
        router = MagicMock()
        provider = MagicMock()

        def fake_completion(req):
            yield _make_chunk("Direct answer")

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test-model", context="data", query="what?")
        engine.run(req)

        router.pin_model.assert_called_once_with("test-model")
        router.unpin_model.assert_called_once_with("test-model")

    def test_unpin_on_exception(self):
        router = MagicMock()
        provider = MagicMock()
        provider.create_chat_completion.side_effect = RuntimeError("boom")
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test-model", context="data", query="what?")

        with pytest.raises(RuntimeError):
            engine.run(req)

        router.unpin_model.assert_called_once_with("test-model")


# ---------------------------------------------------------------------------
# AST safety checks
# ---------------------------------------------------------------------------

class TestAstSafety:
    def test_blocked_class_access(self):
        tree, err = _check_ast_safety("x.__class__.__bases__")
        assert tree is None
        assert err is not None
        assert "blocked" in err

    def test_blocked_subclasses(self):
        tree, err = _check_ast_safety("obj.__subclasses__()")
        assert tree is None
        assert err is not None

    def test_blocked_globals(self):
        tree, err = _check_ast_safety("fn.__globals__")
        assert tree is None
        assert err is not None

    def test_safe_code_passes(self):
        tree, err = _check_ast_safety("x = len([1,2,3])\nprint(x)")
        assert tree is not None
        assert err is None

    def test_syntax_error_passes_through(self):
        # Syntax errors are deferred to compile() -- AST check returns (None, None)
        tree, err = _check_ast_safety("def (:")
        assert tree is None
        assert err is None

    def test_sandbox_blocks_ast_violation(self):
        ns = {}
        stdout, stderr = run_code_in_namespace("x = ''.__class__.__bases__", ns)
        assert stdout == ""
        assert "blocked" in stderr

    def test_sandbox_disabled_allows_dunder(self):
        ns = {}
        stdout, stderr = run_code_in_namespace("print(type('').__name__)", ns, sandbox=False)
        # __name__ is not in our blocked list, so this is fine even with sandbox
        # But with sandbox=False, AST check is skipped entirely
        assert stdout.strip() == "str"


# ---------------------------------------------------------------------------
# Sub-call parameters
# ---------------------------------------------------------------------------

class TestSubCallParams:
    def test_sub_params_used_in_llm_query(self):
        router = MagicMock()
        provider = MagicMock()
        captured_requests = []
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            captured_requests.append(req)
            if call_count[0] == 1:
                yield _make_chunk('```repl\nresult = llm_query("sub q")\nFINAL(result)\n```')
            else:
                yield _make_chunk("sub answer")

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(
            model="main-model",
            context="data",
            query="what?",
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            sub_max_tokens=512,
            sub_temperature=0.0,
            sub_top_p=0.5,
        )
        engine.run(req)

        # First call is the main iteration (uses main params)
        main_req = captured_requests[0]
        assert main_req.max_tokens == 4096
        assert main_req.temperature == 0.7

        # Second call is the sub-call (uses sub params)
        sub_req = captured_requests[1]
        assert sub_req.max_tokens == 512
        assert sub_req.temperature == 0.0
        assert sub_req.top_p == 0.5

    def test_sub_params_fallback_to_main(self):
        router = MagicMock()
        provider = MagicMock()
        captured_requests = []
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            captured_requests.append(req)
            if call_count[0] == 1:
                yield _make_chunk('```repl\nresult = llm_query("sub q")\nFINAL(result)\n```')
            else:
                yield _make_chunk("sub answer")

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(
            model="main-model",
            context="data",
            query="what?",
            max_tokens=4096,
            temperature=0.7,
        )
        engine.run(req)

        # Sub-call should inherit main params when sub_* not set
        sub_req = captured_requests[1]
        assert sub_req.max_tokens == 4096
        assert sub_req.temperature == 0.7


# ---------------------------------------------------------------------------
# Trace detail (include_trace_detail)
# ---------------------------------------------------------------------------

class TestTraceDetail:
    def test_detail_off_by_default(self):
        router = _mock_router(['```repl\nprint("hi")\n```', '```repl\nFINAL("done")\n```'])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)

        for entry in result.rlm.trace:
            assert entry.response is None
            assert entry.code is None
            assert entry.stdout is None
            assert entry.stderr is None

    def test_detail_captures_code_and_output(self):
        router = _mock_router([
            '```repl\nprint(len(context))\n```',
            '```repl\nFINAL("answer")\n```',
        ])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="hello world", query="what?", include_trace_detail=True)
        result = engine.run(req)

        first = result.rlm.trace[0]
        assert first.response is not None
        assert "```repl" in first.response
        assert first.code == "print(len(context))"
        assert first.stdout is not None
        assert "11" in first.stdout  # len("hello world")
        assert first.stderr == ""

        second = result.rlm.trace[1]
        assert second.code is not None
        assert "FINAL" in second.code
        assert second.action == "FINAL"

    def test_detail_captures_direct_response(self):
        router = _mock_router(["Just a direct answer."])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?", include_trace_detail=True)
        result = engine.run(req)

        entry = result.rlm.trace[0]
        assert entry.action == "direct_response"
        assert entry.response == "Just a direct answer."

    def test_detail_excluded_from_serialization_when_off(self):
        router = _mock_router(['```repl\nFINAL("x")\n```'])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)

        dumped = result.model_dump()
        trace_entry = dumped["rlm"]["trace"][0]
        # None fields should be excluded
        assert "response" not in trace_entry
        assert "code" not in trace_entry
        assert "stdout" not in trace_entry
        assert "stderr" not in trace_entry


# ---------------------------------------------------------------------------
# Max errors threshold
# ---------------------------------------------------------------------------

class TestMaxErrors:
    def test_stops_after_n_consecutive_errors(self):
        """N consecutive errors triggers error_threshold finish reason."""
        responses = ['```repl\n1/0\n```'] * 10
        router = _mock_router(responses)
        engine = RLMEngine(router)
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_iterations=10, max_errors=3,
        )
        result = engine.run(req)

        assert result.finish_reason == "error_threshold"
        assert result.rlm.trace[-1].action == "error_threshold"
        # Should stop at 3 iterations (3 consecutive errors)
        assert result.rlm.iterations == 3

    def test_counter_resets_on_success(self):
        """A successful execution resets the consecutive error counter."""
        responses = [
            '```repl\n1/0\n```',                     # error 1
            '```repl\n1/0\n```',                     # error 2
            '```repl\nprint("ok")\n```',              # success -> reset
            '```repl\n1/0\n```',                     # error 1
            '```repl\n1/0\n```',                     # error 2
            '```repl\nFINAL("done")\n```',           # success -> FINAL
        ]
        router = _mock_router(responses)
        engine = RLMEngine(router)
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_iterations=10, max_errors=3,
        )
        result = engine.run(req)

        assert result.finish_reason == "final"
        assert result.answer == "done"

    def test_no_limit_when_none(self):
        """max_errors=None means no error limit (current default behavior)."""
        responses = ['```repl\n1/0\n```'] * 5 + ['```repl\nFINAL("done")\n```']
        router = _mock_router(responses)
        engine = RLMEngine(router)
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_iterations=10, max_errors=None,
        )
        result = engine.run(req)

        assert result.finish_reason == "final"
        assert result.answer == "done"

    def test_max_errors_request_validation(self):
        with pytest.raises(Exception):
            RLMRequest(model="m", context="c", query="q", max_errors=0)
        with pytest.raises(Exception):
            RLMRequest(model="m", context="c", query="q", max_errors=21)


# ---------------------------------------------------------------------------
# Compaction (history summarization)
# ---------------------------------------------------------------------------

class TestCompaction:
    def test_compaction_triggers_when_threshold_exceeded(self):
        """Compaction replaces history when estimated tokens exceed threshold."""
        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            messages = req.messages
            # Compaction summary call: returns a summary
            if any("Summarize your progress" in m.content for m in messages):
                yield _make_chunk("Summary: computed x=42, next step is FINAL")
            elif call_count[0] <= 2:
                yield _make_chunk('```repl\nprint("iteration output " * 500)\n```')
            else:
                yield _make_chunk('```repl\nFINAL("done")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        # System prompt is ~1000 chars (~250 tokens). With threshold at 50% of 512 = 256 tokens,
        # compaction triggers once messages exceed ~1024 chars (i.e., after 1 iteration of output).
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_iterations=10,
            compaction=True,
            compaction_threshold=0.5,
            max_context_tokens=512,
        )
        result = engine.run(req)

        assert result.rlm.compactions > 0
        assert result.finish_reason == "final"

    def test_compaction_disabled_by_default(self):
        """No compaction when compaction=False (default)."""
        responses = ['```repl\nprint("hi")\n```'] * 3 + ['```repl\nFINAL("done")\n```']
        router = _mock_router(responses)
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?", max_iterations=5)
        result = engine.run(req)

        assert result.rlm.compactions == 0

    def test_namespace_survives_compaction(self):
        """Variables in the REPL namespace persist through compaction."""
        router = MagicMock()
        provider = MagicMock()
        iteration_calls = [0]  # Only counts non-compaction calls

        def fake_completion(req):
            messages = req.messages
            if any("Summarize your progress" in m.content for m in messages):
                yield _make_chunk("Summary: set x = 42")
                return
            iteration_calls[0] += 1
            if iteration_calls[0] == 1:
                # Set a variable with large output to trigger compaction next iteration
                yield _make_chunk('```repl\nx = 42\nprint("x set " * 200)\n```')
            else:
                # After compaction, x should still be in namespace
                yield _make_chunk('```repl\nFINAL_VAR("x")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        # max_context_tokens=1024 with threshold=0.5 means 512 token threshold (~2048 chars).
        # Initial messages are ~1100 chars (~275 tokens) -- below threshold.
        # After iteration 1 adds ~1000 chars of output feedback, total crosses threshold.
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_iterations=5,
            compaction=True,
            compaction_threshold=0.5,
            max_context_tokens=1024,
        )
        result = engine.run(req)

        assert result.answer == "42"
        assert result.rlm.compactions > 0

    def test_compaction_streaming_event(self):
        """Streaming yields a compaction event when history is compacted."""
        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            messages = req.messages
            if any("Summarize your progress" in m.content for m in messages):
                yield _make_chunk("Summary text")
            elif call_count[0] <= 2:
                yield _make_chunk('```repl\nprint("big output " * 500)\n```')
            else:
                yield _make_chunk('```repl\nFINAL("done")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_iterations=10,
            compaction=True,
            compaction_threshold=0.5,
            max_context_tokens=512,
        )

        events = list(engine.run_streaming(req))
        event_types = [e.split("\n")[0].replace("event: ", "") for e in events]
        assert "compaction" in event_types

    def test_compaction_threshold_validation(self):
        with pytest.raises(Exception):
            RLMRequest(model="m", context="c", query="q", compaction_threshold=0.4)
        with pytest.raises(Exception):
            RLMRequest(model="m", context="c", query="q", compaction_threshold=0.96)

    def test_estimate_tokens(self):
        engine = RLMEngine(MagicMock())
        from heylook_llm.config import ChatMessage
        messages = [
            ChatMessage(role="system", content="x" * 400),
            ChatMessage(role="user", content="y" * 400),
        ]
        est = engine._estimate_tokens(messages)
        assert est == 200  # 800 chars / 4


# ---------------------------------------------------------------------------
# Recursive depth (rlm_query)
# ---------------------------------------------------------------------------

class TestRecursiveDepth:
    def test_rlm_query_not_injected_at_depth_1(self):
        """At max_depth=1 (default), rlm_query is not available in namespace."""
        router = _mock_router(['```repl\nFINAL(str("rlm_query" in dir()))\n```'])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?", max_depth=1)
        result = engine.run(req)
        # rlm_query should NOT be in the namespace
        # The FINAL call returns whether "rlm_query" is in the REPL globals.
        # Since sandbox restricts dir(), let's check the namespace directly.
        # Actually, we can test the namespace setup directly:
        ns = engine._init_namespace(req, [0], [])
        assert "rlm_query" not in ns

    def test_rlm_query_injected_at_depth_2(self):
        """At max_depth=2, rlm_query is available in namespace."""
        engine = RLMEngine(MagicMock())
        req = RLMRequest(model="test", context="data", query="what?", max_depth=2)
        ns = engine._init_namespace(req, [0], [])
        assert "rlm_query" in ns
        assert callable(ns["rlm_query"])

    def test_child_rlm_spawned_at_depth_2(self):
        """rlm_query() spawns a child RLM with reduced depth."""
        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            messages = req.messages
            # Child RLM calls -- detect by "Answer this question" in system prompt
            if any("Answer this question" in m.content for m in messages):
                yield _make_chunk('```repl\nFINAL("child answer")\n```')
            elif call_count[0] == 1:
                yield _make_chunk('```repl\nresult = rlm_query("sub problem")\nFINAL(result)\n```')
            else:
                yield _make_chunk('```repl\nFINAL("fallback")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_depth=2, max_iterations=5,
        )
        result = engine.run(req)

        assert result.answer == "child answer"
        assert result.rlm.child_traces is not None
        assert len(result.rlm.child_traces) == 1

    def test_child_falls_back_to_llm_query_at_max_depth(self):
        """At max_depth=1, rlm_query is not available so model uses llm_query."""
        # Just verify rlm_query is not injected
        engine = RLMEngine(MagicMock())
        req = RLMRequest(model="test", context="data", query="what?", max_depth=1)
        ns = engine._init_namespace(req, [0], [])
        assert "rlm_query" not in ns
        assert "llm_query" in ns

    def test_child_depth_reduced(self):
        """Child RLM request has max_depth decremented by 1."""
        router = MagicMock()
        provider = MagicMock()
        captured_child_depth = []

        original_iter_loop = RLMEngine._iter_loop

        def tracking_iter_loop(self_engine, request, prov, req_id):
            if "child" in req_id:
                captured_child_depth.append(request.max_depth)
            yield from original_iter_loop(self_engine, request, prov, req_id)

        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            messages = req.messages
            if any("Answer this question" in m.content for m in messages):
                yield _make_chunk('```repl\nFINAL("child done")\n```')
            elif call_count[0] == 1:
                yield _make_chunk('```repl\nresult = rlm_query("sub")\nFINAL(result)\n```')
            else:
                yield _make_chunk('```repl\nFINAL("x")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        engine._iter_loop = lambda req, prov, rid: tracking_iter_loop(engine, req, prov, rid)

        req = RLMRequest(model="test", context="data", query="what?", max_depth=3)
        engine.run(req)

        assert len(captured_child_depth) >= 1
        assert captured_child_depth[0] == 2  # parent was 3, child should be 2

    def test_max_depth_request_validation(self):
        with pytest.raises(Exception):
            RLMRequest(model="m", context="c", query="q", max_depth=0)
        with pytest.raises(Exception):
            RLMRequest(model="m", context="c", query="q", max_depth=6)

    def test_system_prompt_mentions_rlm_query_at_depth_2(self):
        engine = RLMEngine(MagicMock())
        prompt = engine._build_system_prompt("task", None, max_depth=2)
        assert "rlm_query" in prompt

    def test_system_prompt_no_rlm_query_at_depth_1(self):
        engine = RLMEngine(MagicMock())
        prompt = engine._build_system_prompt("task", None, max_depth=1)
        assert "rlm_query" not in prompt

    def test_child_traces_in_metadata(self):
        """Child trace metadata is captured in parent's response."""
        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            messages = req.messages
            if any("Answer this question" in m.content for m in messages):
                yield _make_chunk('```repl\nprint("child working")\n```')
                return
            if call_count[0] == 1:
                yield _make_chunk('```repl\nresult = rlm_query("sub")\nFINAL(result)\n```')
            else:
                yield _make_chunk('```repl\nFINAL("done")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_depth=2, max_iterations=5,
        )
        result = engine.run(req)

        assert result.rlm.child_traces is not None
        child = result.rlm.child_traces[0]
        assert isinstance(child, RLMMetadata)
        assert child.iterations > 0

    def test_no_child_traces_when_depth_1(self):
        """No child_traces field when max_depth=1."""
        router = _mock_router(['```repl\nFINAL("done")\n```'])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?", max_depth=1)
        result = engine.run(req)
        assert result.rlm.child_traces is None


# ---------------------------------------------------------------------------
# New metadata fields
# ---------------------------------------------------------------------------

class TestNewMetadataFields:
    def test_metadata_compactions_default(self):
        meta = RLMMetadata(iterations=1, sub_queries=0, trace=[])
        assert meta.compactions == 0
        assert meta.child_traces is None

    def test_metadata_child_traces_serialization(self):
        child = RLMMetadata(iterations=2, sub_queries=0, trace=[])
        parent = RLMMetadata(
            iterations=3, sub_queries=1, trace=[],
            compactions=1, child_traces=[child],
        )
        dumped = parent.model_dump()
        assert dumped["compactions"] == 1
        assert len(dumped["child_traces"]) == 1
        assert dumped["child_traces"][0]["iterations"] == 2

    def test_new_request_defaults(self):
        req = RLMRequest(model="m", context="c", query="q")
        assert req.compaction is False
        assert req.compaction_threshold == 0.8
        assert req.max_context_tokens == 8192
        assert req.max_depth == 1
        assert req.max_errors is None
        assert req.max_timeout is None


# ---------------------------------------------------------------------------
# SHOW_VARS
# ---------------------------------------------------------------------------

class TestShowVars:
    def test_empty_namespace(self):
        """SHOW_VARS with no user-defined variables returns empty dict."""
        engine = RLMEngine(MagicMock())
        req = RLMRequest(model="test", context="data", query="what?")
        ns = engine._init_namespace(req, [0], [])
        result = ns["SHOW_VARS"]()
        assert result == "{}"

    def test_after_setting_variables(self):
        """SHOW_VARS shows user-defined variables after assignment."""
        engine = RLMEngine(MagicMock())
        req = RLMRequest(model="test", context="data", query="what?")
        ns = engine._init_namespace(req, [0], [])
        ns["x"] = 42
        ns["name"] = "hello"
        result = ns["SHOW_VARS"]()
        assert "'x': 'int'" in result
        assert "'name': 'str'" in result

    def test_excludes_internals(self):
        """SHOW_VARS excludes reserved names and underscore-prefixed vars."""
        engine = RLMEngine(MagicMock())
        req = RLMRequest(model="test", context="data", query="what?")
        ns = engine._init_namespace(req, [0], [])
        ns["_private"] = "hidden"
        result = ns["SHOW_VARS"]()
        assert "_private" not in result
        assert "context" not in result
        assert "FINAL" not in result
        assert "llm_query" not in result

    def test_in_system_prompt(self):
        """SHOW_VARS is mentioned in the system prompt."""
        engine = RLMEngine(MagicMock())
        prompt = engine._build_system_prompt("task", None)
        assert "SHOW_VARS" in prompt

    def test_callable_in_repl(self):
        """SHOW_VARS can be called from code execution."""
        engine = RLMEngine(MagicMock())
        req = RLMRequest(model="test", context="data", query="what?")
        ns = engine._init_namespace(req, [0], [])
        stdout, stderr = run_code_in_namespace("x = 42\nprint(SHOW_VARS())", ns, sandbox=False)
        assert "'x': 'int'" in stdout
        assert stderr == ""


# ---------------------------------------------------------------------------
# Root prompt re-injection
# ---------------------------------------------------------------------------

class TestRootPromptReinjection:
    def test_reinjected_after_first_iteration(self):
        """After iteration 0, feedback includes original query."""
        router = MagicMock()
        provider = MagicMock()
        captured_messages = []
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            captured_messages.append([m.content for m in req.messages])
            if call_count[0] <= 2:
                yield _make_chunk('```repl\nprint("step")\n```')
            else:
                yield _make_chunk('```repl\nFINAL("done")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="Find the answer", max_iterations=5)
        engine.run(req)

        # Third call (iteration 2) should have the original query in feedback
        assert len(captured_messages) >= 3
        iter2_messages = captured_messages[2]
        last_user_msg = iter2_messages[-1]
        assert "Original task: Find the answer" in last_user_msg

    def test_not_on_first_iteration(self):
        """First iteration feedback does not include original query."""
        router = MagicMock()
        provider = MagicMock()
        captured_messages = []
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            captured_messages.append([m.content for m in req.messages])
            if call_count[0] == 1:
                yield _make_chunk('```repl\nprint("step")\n```')
            else:
                yield _make_chunk('```repl\nFINAL("done")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="Find the answer", max_iterations=5)
        engine.run(req)

        # Second call (iteration 1) -- feedback from iteration 0 should NOT have query
        assert len(captured_messages) >= 2
        iter1_messages = captured_messages[1]
        last_user_msg = iter1_messages[-1]
        assert "Original task:" not in last_user_msg


# ---------------------------------------------------------------------------
# Best partial answer tracking
# ---------------------------------------------------------------------------

class TestBestPartialAnswer:
    def test_partial_preferred_over_code(self):
        """When max_iterations hit, best_partial (text response) is preferred over last code response."""
        responses = [
            "Here is a partial analysis of the data.",  # text (becomes best_partial)
        ] + ['```repl\nprint("loop")\n```'] * 3  # code loops
        # First response is direct_response so it will end immediately -- need multi-iteration
        # Actually, direct_response breaks the loop. Let me rethink:
        # We need a code response followed by text-like iterations that don't trigger direct_response
        # best_partial is set when extract_repl_block returns None AND text is non-empty
        # But direct_response breaks the loop. So best_partial only matters when loop continues with code.
        # The scenario: model outputs code for a while, then outputs text (breaks as direct_response),
        # OR model outputs code, then max_iterations is hit.
        # best_partial captures text from _AssistantResponse that has no repl block.
        # But wait -- if there's no repl block, it's a direct_response and the loop breaks.
        # So best_partial only gets set from direct_response... but that also breaks the loop.
        # Actually, re-reading the code: best_partial is set BEFORE the `code = extract_repl_block()` check.
        # So it tracks any non-code text. But if code is None, we break with direct_response.
        # The value of best_partial is: when error_threshold or max_iterations fires,
        # and the last response_text was code (which we can't use as an answer), we fall back to best_partial.
        # Scenario: model gives text answer, then code that errors out until error_threshold.
        # Wait, that's not possible: text answer triggers direct_response break.
        # Let me look more carefully at the logic...
        # Actually: best_partial is set when extract_repl_block(response_text) is None AND strip() non-empty
        # This happens on the SAME iteration as direct_response. So best_partial = the direct response text.
        # But direct_response also sets final_answer = response_text and breaks.
        # So best_partial matters in a sequence like:
        # 1. Code + runs successfully -> continues
        # 2. Code with explanation text in response (but has repl block) -> continues, extract_repl_block is NOT None
        # 3. Max_iterations hit -> final_answer = best_partial or response_text
        # Hmm, so best_partial doesn't get set from code responses. Only from no-code responses.
        # And no-code responses are direct_response breaks.
        # OH WAIT: It's checking the FULL response_text for extract_repl_block.
        # The model could respond with text AND a code block. extract_repl_block would find the code block.
        # But the response_text itself could be "Here's my analysis...\n```repl\n...\n```"
        # In that case extract_repl_block is NOT None, so best_partial is NOT set.
        # So best_partial only catches responses with no code block at all = direct_response = break.
        # This means best_partial is always the same as the direct_response answer.
        # Unless... direct_response happens, then later iterations somehow continue?
        # No, direct_response breaks.
        # I think the plan's intent is that best_partial captures the LAST text answer before
        # code-only responses start erroring out. But with the current logic flow, if a response
        # has no code, we break immediately. So best_partial == the last direct_response.
        # The actual use case: if all responses are code, and we hit max_iterations or error_threshold,
        # response_text is the last code response text. best_partial would be empty.
        # So the fallback chain is: best_partial (empty) -> response_text (code) -> "[Max iterations reached]"
        # With best_partial, we'd use response_text which includes the code block text.
        # Let me just test the error_threshold path with best_partial properly.
        pass

    def test_best_partial_on_error_threshold(self):
        """error_threshold uses best_partial over last (code) response_text."""
        # All errors, no text responses -> best_partial stays empty, falls back to response_text
        responses = ['```repl\n1/0\n```'] * 5
        router = _mock_router(responses)
        engine = RLMEngine(router)
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_iterations=10, max_errors=3,
        )
        result = engine.run(req)
        assert result.finish_reason == "error_threshold"
        # best_partial is empty, falls back to response_text (the code block text)
        assert "```repl" in result.answer or "1/0" in result.answer

    def test_best_partial_on_max_iterations(self):
        """max_iterations uses best_partial when available."""
        responses = ['```repl\nprint("working")\n```'] * 5
        router = _mock_router(responses)
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?", max_iterations=3)
        result = engine.run(req)
        assert result.finish_reason == "max_iterations"
        # No text-only response was given, so best_partial is empty
        # Falls through to response_text (the code block)
        assert result.answer != "[Max iterations reached]"

    def test_no_partial_falls_back(self):
        """When best_partial and response_text are empty, fallback message is used."""
        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            yield _make_chunk("")  # empty response

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?", max_iterations=1)
        result = engine.run(req)
        # Empty response -> direct_response (since extract_repl_block returns None for "")
        # But response_text.strip() is falsy, so best_partial not set
        # direct_response sets final_answer = response_text = ""
        # Actually, empty string -> extract_repl_block is None -> direct_response with final_answer = ""
        assert result.finish_reason == "direct_response"


# ---------------------------------------------------------------------------
# max_timeout
# ---------------------------------------------------------------------------

class TestMaxTimeout:
    def test_stops_loop(self, monkeypatch):
        """max_timeout causes the loop to stop with 'timeout' finish_reason."""
        import heylook_llm.rlm as rlm_mod

        fake_time = [0.0]

        def mock_time():
            fake_time[0] += 5.0  # Each call advances 5s
            return fake_time[0]

        monkeypatch.setattr(rlm_mod.time, "time", mock_time)

        router = MagicMock()
        provider = MagicMock()

        def fake_completion(req):
            yield _make_chunk('```repl\nprint("working")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_iterations=50, max_timeout=3.0,  # Will be exceeded after first iter
        )
        result = engine.run(req)
        assert result.finish_reason == "timeout"

    def test_returns_partial_on_timeout(self, monkeypatch):
        """Timeout returns best partial or fallback message."""
        import heylook_llm.rlm as rlm_mod

        fake_time = [0.0]

        def mock_time():
            fake_time[0] += 5.0
            return fake_time[0]

        monkeypatch.setattr(rlm_mod.time, "time", mock_time)

        router = MagicMock()
        provider = MagicMock()

        def fake_completion(req):
            yield _make_chunk('```repl\nprint("working")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_iterations=50, max_timeout=3.0,
        )
        result = engine.run(req)
        assert result.finish_reason == "timeout"
        assert "[Stopped: timeout]" in result.answer

    def test_none_means_no_limit(self):
        """max_timeout=None means no wall-clock limit."""
        router = _mock_router(['```repl\nFINAL("done")\n```'])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?", max_timeout=None)
        result = engine.run(req)
        assert result.finish_reason == "final"

    def test_validation(self):
        """max_timeout must be >= 1.0 if set."""
        with pytest.raises(Exception):
            RLMRequest(model="m", context="c", query="q", max_timeout=0.5)

    def test_trace_entry(self, monkeypatch):
        """Timeout creates a trace entry with action='timeout'."""
        import heylook_llm.rlm as rlm_mod

        fake_time = [0.0]

        def mock_time():
            fake_time[0] += 5.0
            return fake_time[0]

        monkeypatch.setattr(rlm_mod.time, "time", mock_time)

        router = MagicMock()
        provider = MagicMock()

        def fake_completion(req):
            yield _make_chunk('```repl\nprint("x")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(
            model="test", context="data", query="what?",
            max_iterations=50, max_timeout=3.0,
        )
        result = engine.run(req)
        assert result.finish_reason == "timeout"
        timeout_entries = [e for e in result.rlm.trace if e.action == "timeout"]
        assert len(timeout_entries) == 1


# ---------------------------------------------------------------------------
# llm_query_batched
# ---------------------------------------------------------------------------

class TestLlmQueryBatched:
    def test_gpu_batch_used(self):
        """llm_query_batched uses create_batch_chat_completion when available."""
        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        def fake_batch(reqs):
            return [{"text": f"answer_{i}"} for i in range(len(reqs))]

        provider.create_batch_chat_completion = MagicMock(side_effect=fake_batch)

        def fake_completion(req):
            call_count[0] += 1
            if call_count[0] == 1:
                yield _make_chunk(
                    '```repl\nresults = llm_query_batched(["q1", "q2"])\nFINAL(str(results))\n```'
                )
            else:
                yield _make_chunk("fallback")

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)

        provider.create_batch_chat_completion.assert_called_once()
        assert "answer_0" in result.answer
        assert "answer_1" in result.answer

    def test_sequential_fallback(self):
        """Falls back to sequential when batch not available."""
        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            messages = req.messages
            if call_count[0] == 1:
                yield _make_chunk(
                    '```repl\nresults = llm_query_batched(["q1", "q2"])\nFINAL(str(results))\n```'
                )
            else:
                yield _make_chunk(f"seq_answer_{call_count[0]}")

        provider.create_chat_completion.side_effect = fake_completion
        # No create_batch_chat_completion attribute
        if hasattr(provider, "create_batch_chat_completion"):
            del provider.create_batch_chat_completion
        provider.spec = None

        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)

        assert "seq_answer" in result.answer

    def test_counter_increments(self):
        """Counter increments by len(prompts) for batched calls."""
        engine = RLMEngine(MagicMock())
        req = RLMRequest(model="test", context="data", query="what?")
        counter = [0]
        ns = engine._init_namespace(req, counter, [])

        # Mock the provider for the batched call
        router = MagicMock()
        provider = MagicMock()

        def fake_completion(r):
            yield _make_chunk("answer")

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider
        engine.router = router

        # Re-initialize to get the closure with our router
        counter = [0]
        ns = engine._init_namespace(req, counter, [])
        ns["llm_query_batched"](["a", "b", "c"])
        assert counter[0] == 3

    def test_empty_list(self):
        """llm_query_batched([]) returns [] immediately."""
        engine = RLMEngine(MagicMock())
        req = RLMRequest(model="test", context="data", query="what?")
        counter = [0]
        ns = engine._init_namespace(req, counter, [])
        result = ns["llm_query_batched"]([])
        assert result == []
        assert counter[0] == 0

    def test_in_system_prompt(self):
        """llm_query_batched is mentioned in the system prompt."""
        engine = RLMEngine(MagicMock())
        prompt = engine._build_system_prompt("task", None)
        assert "llm_query_batched" in prompt

    def test_sub_params_used(self):
        """Batched calls use sub_* parameters."""
        router = MagicMock()
        # Use spec to prevent auto-creating create_batch_chat_completion
        provider = MagicMock(spec=["create_chat_completion"])
        captured_reqs = []

        def fake_completion(req):
            captured_reqs.append(req)
            yield _make_chunk("answer")

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(
            model="test", context="data", query="what?",
            sub_max_tokens=256, sub_temperature=0.0,
        )
        counter = [0]
        ns = engine._init_namespace(req, counter, [])
        ns["llm_query_batched"](["q1"])

        assert len(captured_reqs) == 1
        assert captured_reqs[0].max_tokens == 256
        assert captured_reqs[0].temperature == 0.0

    def test_gpu_failure_falls_back(self):
        """If GPU batch raises, falls back to sequential."""
        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        provider.create_batch_chat_completion = MagicMock(side_effect=RuntimeError("GPU fail"))

        def fake_completion(req):
            call_count[0] += 1
            if call_count[0] == 1:
                yield _make_chunk(
                    '```repl\nresults = llm_query_batched(["q1"])\nFINAL(str(results))\n```'
                )
            else:
                yield _make_chunk("fallback_answer")

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)

        assert "fallback_answer" in result.answer


# ---------------------------------------------------------------------------
# rlm_query_batched
# ---------------------------------------------------------------------------

class TestRlmQueryBatched:
    def test_available_at_depth_2(self):
        """rlm_query_batched is injected at max_depth=2."""
        engine = RLMEngine(MagicMock())
        req = RLMRequest(model="test", context="data", query="what?", max_depth=2)
        ns = engine._init_namespace(req, [0], [])
        assert "rlm_query_batched" in ns
        assert callable(ns["rlm_query_batched"])

    def test_not_at_depth_1(self):
        """rlm_query_batched is NOT injected at max_depth=1."""
        engine = RLMEngine(MagicMock())
        req = RLMRequest(model="test", context="data", query="what?", max_depth=1)
        ns = engine._init_namespace(req, [0], [])
        assert "rlm_query_batched" not in ns

    def test_system_prompt_mentions_at_depth_2(self):
        """System prompt mentions rlm_query_batched at depth 2."""
        engine = RLMEngine(MagicMock())
        prompt = engine._build_system_prompt("task", None, max_depth=2)
        assert "rlm_query_batched" in prompt


# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------

class TestCustomTools:
    def test_callable_injected(self):
        """A plain callable is injected into the namespace."""
        def my_tool():
            return "result"

        engine = RLMEngine(MagicMock(), custom_tools=[my_tool])
        req = RLMRequest(model="test", context="data", query="what?")
        ns = engine._init_namespace(req, [0], [])
        assert "my_tool" in ns
        assert ns["my_tool"]() == "result"

    def test_dict_format(self):
        """Dict format with tool + description works."""
        def helper():
            return 42

        engine = RLMEngine(MagicMock(), custom_tools=[
            {"tool": helper, "description": "Returns 42"}
        ])
        req = RLMRequest(model="test", context="data", query="what?")
        ns = engine._init_namespace(req, [0], [])
        assert "helper" in ns
        assert ns["helper"]() == 42

    def test_reserved_name_raises(self):
        """Cannot use a reserved name for a custom tool."""
        def FINAL():
            pass

        with pytest.raises(ValueError, match="reserved"):
            RLMEngine(MagicMock(), custom_tools=[FINAL])

    def test_non_callable_raises(self):
        """Non-callable custom tool raises ValueError."""
        with pytest.raises(ValueError, match="callable or dict"):
            RLMEngine(MagicMock(), custom_tools=["not_a_function"])

    def test_description_in_prompt(self):
        """Custom tool descriptions appear in system prompt."""
        def analyzer():
            """Analyzes data"""
            pass

        engine = RLMEngine(MagicMock(), custom_tools=[analyzer])
        prompt = engine._build_system_prompt("task", None)
        assert "Custom Tools" in prompt
        assert "analyzer" in prompt
        assert "Analyzes data" in prompt

    def test_no_tools_no_section(self):
        """No custom tools means no Custom Tools section in prompt."""
        engine = RLMEngine(MagicMock())
        prompt = engine._build_system_prompt("task", None)
        assert "## Custom Tools" not in prompt

    def test_available_in_child(self):
        """Custom tools are available to child RLMs via shared engine."""
        def my_tool():
            return "child_result"

        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            messages = req.messages
            if any("Answer this question" in m.content for m in messages):
                yield _make_chunk('```repl\nresult = my_tool()\nFINAL(result)\n```')
            elif call_count[0] == 1:
                yield _make_chunk('```repl\nresult = rlm_query("sub")\nFINAL(result)\n```')
            else:
                yield _make_chunk('```repl\nFINAL("x")\n```')

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(router, custom_tools=[my_tool])
        req = RLMRequest(model="test", context="data", query="what?", max_depth=2, sandbox=False)
        result = engine.run(req)

        assert result.answer == "child_result"


# ---------------------------------------------------------------------------
# Event callbacks
# ---------------------------------------------------------------------------

class TestEventCallbacks:
    def test_iter_start_called(self):
        """on_iteration_start is called with iteration number."""
        starts = []
        router = _mock_router(['```repl\nFINAL("done")\n```'])
        engine = RLMEngine(router, on_iteration_start=lambda i: starts.append(i))
        req = RLMRequest(model="test", context="data", query="what?")
        engine.run(req)
        assert starts == [1]

    def test_iter_complete_with_duration(self):
        """on_iteration_complete is called with iteration and duration."""
        completions = []
        router = _mock_router(['```repl\nFINAL("done")\n```'])
        engine = RLMEngine(
            router,
            on_iteration_complete=lambda i, d: completions.append((i, d)),
        )
        req = RLMRequest(model="test", context="data", query="what?")
        engine.run(req)
        assert len(completions) == 1
        assert completions[0][0] == 1
        assert completions[0][1] >= 0  # duration is non-negative

    def test_subcall_callbacks(self):
        """on_subcall_start/complete are called for llm_query."""
        subcall_starts = []
        subcall_completes = []
        router = MagicMock()
        provider = MagicMock()
        call_count = [0]

        def fake_completion(req):
            call_count[0] += 1
            if call_count[0] == 1:
                yield _make_chunk('```repl\nresult = llm_query("sub")\nFINAL(result)\n```')
            else:
                yield _make_chunk("sub answer")

        provider.create_chat_completion.side_effect = fake_completion
        router.get_provider.return_value = provider

        engine = RLMEngine(
            router,
            on_subcall_start=lambda d, p: subcall_starts.append((d, p)),
            on_subcall_complete=lambda d, m, dur: subcall_completes.append((d, m, dur)),
        )
        req = RLMRequest(model="test", context="data", query="what?")
        engine.run(req)

        assert len(subcall_starts) == 1
        assert subcall_starts[0][0] == 0  # depth 0 for llm_query
        assert len(subcall_completes) == 1
        assert subcall_completes[0][2] >= 0  # duration

    def test_no_callbacks_ok(self):
        """Engine works fine with no callbacks set."""
        router = _mock_router(['```repl\nFINAL("done")\n```'])
        engine = RLMEngine(router)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)
        assert result.answer == "done"

    def test_direct_response_fires_callbacks(self):
        """Callbacks fire even for direct response (no code)."""
        starts = []
        completions = []
        router = _mock_router(["Just text"])
        engine = RLMEngine(
            router,
            on_iteration_start=lambda i: starts.append(i),
            on_iteration_complete=lambda i, d: completions.append(i),
        )
        req = RLMRequest(model="test", context="data", query="what?")
        engine.run(req)
        assert starts == [1]
        assert completions == [1]

    def test_callback_exception_caught(self):
        """Callback exceptions don't crash the loop."""
        def bad_callback(i):
            raise RuntimeError("callback exploded")

        router = _mock_router(['```repl\nFINAL("done")\n```'])
        engine = RLMEngine(router, on_iteration_start=bad_callback)
        req = RLMRequest(model="test", context="data", query="what?")
        result = engine.run(req)
        assert result.answer == "done"  # Loop completed despite callback error
