# tests/unit/test_rlm.py
"""Unit tests for the RLM (Recursive Language Model) endpoint.

All tests mock the provider -- no MLX/GPU needed.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from heylook_llm.rlm import (
    RLMEngine,
    RLMMetadata,
    RLMRequest,
    RLMResponse,
    RLMTraceEntry,
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

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGALRM not available on Windows")
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
