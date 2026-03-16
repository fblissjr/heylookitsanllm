# src/heylook_llm/rlm.py
"""Recursive Language Model (RLM) endpoint.

Implements the RLM inference scaffold (Zhang, Kraska, Khattab -- arxiv 2512.24601v2)
as a server endpoint. The user's prompt is loaded into a Python REPL variable and the
model writes code to iteratively explore, slice, and transform it.

Each iteration calls the provider directly (no HTTP round-trip). The model stays loaded
across all iterations via pin_model().
"""

import ast
import asyncio
import builtins as _builtins_module
import ctypes
import io
import logging
import re
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Generator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from heylook_llm.config import ChatMessage, ChatRequest
from heylook_llm.optimizations import fast_json as json
from heylook_llm.router import ModelRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RLMRequest(BaseModel):
    model: str
    context: str
    query: str
    system: str | None = None
    max_iterations: int = Field(10, ge=1, le=50)
    max_tokens: int = Field(2048, gt=0)
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    sandbox: bool = True
    timeout: int = Field(30, ge=1, le=120)
    max_output_chars: int = Field(10_000, ge=100)
    sub_model: str | None = None
    sub_max_tokens: int | None = None
    sub_temperature: float | None = None
    sub_top_p: float | None = None
    enable_thinking: bool | None = None
    include_trace_detail: bool = False


class RLMTraceEntry(BaseModel):
    iteration: int
    code_len: int = 0
    stdout_len: int = 0
    stderr_len: int = 0
    action: str | None = None
    # Populated when include_trace_detail=True
    response: str | None = None
    code: str | None = None
    stdout: str | None = None
    stderr: str | None = None


class RLMMetadata(BaseModel):
    iterations: int
    sub_queries: int
    trace: list[RLMTraceEntry]


class RLMResponse(BaseModel):
    id: str
    object: str = "rlm.completion"
    created: int
    model: str
    answer: str
    finish_reason: str
    usage: dict
    rlm: RLMMetadata

    def model_dump(self, **kwargs):
        """Exclude None trace detail fields by default for clean output."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


# ---------------------------------------------------------------------------
# Internal loop event types (used to deduplicate sync/streaming paths)
# ---------------------------------------------------------------------------

@dataclass
class _IterStart:
    iteration: int

@dataclass
class _AssistantResponse:
    iteration: int
    text: str

@dataclass
class _CodeOutput:
    iteration: int
    stdout: str
    stderr: str
    code_len: int

@dataclass
class _LoopResult:
    answer: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    trace: list[RLMTraceEntry] = field(default_factory=list)
    sub_queries: int = 0


# ---------------------------------------------------------------------------
# RLM system prompt (adapted from paper Appendix C)
# ---------------------------------------------------------------------------

RLM_SYSTEM_PROMPT = """\
You have access to a Python REPL environment with a variable called `context` \
that contains the input data you need to process.

## Available Tools

- `context` (str): The full input text/data to process
- `llm_query(query: str) -> str`: Call a language model on a sub-question. \
Use this for tasks requiring reasoning over slices of the context.
- `print()`: Display intermediate results (output is fed back to you)
- Standard Python builtins for string manipulation, data processing, etc.

## How to Use the REPL

Write Python code in ```repl blocks:

```repl
# Your code here
result = context[:1000]
print(f"First 1000 chars: {result}")
```

The code executes in a persistent namespace -- variables persist across iterations.
You can write multiple iterations of code, building on previous results.

## Termination

When you have your final answer, call one of these functions in a code block:

- `FINAL("your answer text")` -- return a string answer
- `FINAL_VAR("variable_name")` -- return the value of a namespace variable

If your response contains no ```repl block, the response text itself is \
treated as the final answer.

## Strategy

1. Start by exploring the context: check its length, structure, first/last segments
2. Use slicing and search to find relevant sections
3. Use `llm_query()` for reasoning tasks on manageable chunks
4. Aggregate and format your final answer
5. Call FINAL() or FINAL_VAR() when done
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_REPL_BLOCK_RE = re.compile(r"```repl\n(.*?)```", re.DOTALL)


def extract_repl_block(text: str) -> str | None:
    """Extract the first ```repl code block from model output."""
    match = _REPL_BLOCK_RE.search(text)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# SSE formatting
# ---------------------------------------------------------------------------

def format_sse(event: str, data: dict) -> str:
    """Format an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Sandboxed code execution
# ---------------------------------------------------------------------------

_SAFE_BUILTIN_NAMES = {
    "abs", "all", "any", "bin", "bool", "bytes", "chr", "dict",
    "enumerate", "filter", "float", "format", "frozenset", "getattr",
    "hasattr", "hash", "hex", "id", "int", "isinstance", "iter", "len",
    "list", "map", "max", "min", "next", "object", "oct", "ord", "pow",
    "print", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "type", "zip",
}

_SANDBOX_BUILTINS: dict = {
    name: getattr(_builtins_module, name)
    for name in _SAFE_BUILTIN_NAMES
    if hasattr(_builtins_module, name)
}

# AST-level attribute access blocklist. Prevents obj.__class__.__bases__[0].__subclasses__()
# style escapes that builtins-only restriction doesn't catch.
_BLOCKED_ATTRS = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__code__", "__func__", "__self__",
    "__dict__", "__init_subclass__", "__set_name__",
    "__del__", "__delattr__", "__reduce__", "__reduce_ex__",
    "__getattribute__", "__setattr__",
    "__import__", "__loader__", "__spec__",
})


def _check_ast_safety(code: str) -> tuple[ast.Module, None] | tuple[None, str]:
    """Parse code and check AST for blocked attribute access.

    Returns (ast_tree, None) on success, or (None, error_message) on failure.
    Syntax errors return (None, None) -- deferred to compile().
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None  # type: ignore[return-value]

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_ATTRS:
            return None, f"Access to '{node.attr}' is blocked in sandbox mode"
    return tree, None


def _async_raise(tid: int, exc_type: type) -> None:
    """Raise an exception in another thread via CPython's PyThreadState_SetAsyncExc."""
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(exc_type)
    )
    if res == 0:
        raise ValueError(f"Invalid thread id: {tid}")
    if res > 1:
        # Revert if multiple threads affected (shouldn't happen)
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def run_code_in_namespace(
    code: str,
    namespace: dict,
    *,
    timeout: int = 30,
    max_output_chars: int = 10_000,
    sandbox: bool = True,
) -> tuple[str, str]:
    """Execute code in a persistent namespace, capturing stdout/stderr.

    Returns (stdout, stderr) as strings, truncated to max_output_chars.
    Sandboxed exec() is intentional here -- the RLM paper's core mechanism
    requires executing model-generated code in a controlled REPL.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # AST-level safety check + single parse
    compiled_code = None
    if sandbox:
        tree, ast_error = _check_ast_safety(code)
        if ast_error:
            return "", ast_error
        if tree is not None:
            compiled_code = compile(tree, "<rlm-repl>", "exec")

    if sandbox and (
        "__builtins__" not in namespace
        or not isinstance(namespace.get("__builtins__"), dict)
    ):
        namespace["__builtins__"] = _SANDBOX_BUILTINS.copy()

    try:
        if compiled_code is None:
            compiled_code = compile(code, "<rlm-repl>", "exec")

        if timeout > 0:
            # Thread-safe timeout: run exec in a worker thread, kill it if it hangs.
            # Works from any thread (unlike signal.SIGALRM which requires main thread).
            exec_error: list[Exception] = []

            def _run():
                try:
                    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                        exec(compiled_code, namespace)  # noqa: S102 -- sandboxed REPL
                except Exception as e:
                    exec_error.append(e)

            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            worker.join(timeout=timeout)

            if worker.is_alive():
                # Thread is stuck -- raise TimeoutError in it via CPython API
                if worker.ident is not None:
                    _async_raise(worker.ident, TimeoutError)
                worker.join(timeout=2)  # Give it a moment to unwind
                raise TimeoutError(f"Code execution timed out after {timeout}s")

            if exec_error:
                raise exec_error[0]
        else:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compiled_code, namespace)  # noqa: S102 -- sandboxed REPL

    except TimeoutError as e:
        stderr_buf.write(str(e))
    except Exception as e:
        stderr_buf.write(f"{type(e).__name__}: {e}")

    stdout = stdout_buf.getvalue()
    stderr = stderr_buf.getvalue()

    if len(stdout) > max_output_chars:
        stdout = (
            stdout[:max_output_chars]
            + f"\n... (truncated, {len(stdout)} total chars)"
        )
    if len(stderr) > max_output_chars:
        stderr = (
            stderr[:max_output_chars]
            + f"\n... (truncated, {len(stderr)} total chars)"
        )

    return stdout, stderr


# ---------------------------------------------------------------------------
# Feedback formatting
# ---------------------------------------------------------------------------

def _build_feedback(stdout: str, stderr: str) -> str:
    parts = []
    if stdout:
        parts.append(f"Output:\n{stdout}")
    if stderr:
        parts.append(f"Error:\n{stderr}")
    return "\n".join(parts) if parts else "(no output)"


# ---------------------------------------------------------------------------
# RLM Engine
# ---------------------------------------------------------------------------

class RLMEngine:
    """Runs the RLM iteration loop using provider.create_chat_completion() directly."""

    def __init__(self, router: ModelRouter):
        self.router = router

    def _build_system_prompt(self, query: str, system: str | None) -> str:
        prompt = RLM_SYSTEM_PROMPT
        if system:
            prompt += f"\n# Additional Instructions\n\n{system}\n"
        prompt += f"\n# Task\n\n{query}\n"
        return prompt

    def _build_context_metadata(self, context: str) -> str:
        ctx_len = len(context)
        preview = context[:500] + ("..." if ctx_len > 500 else "")
        return (
            f"Context loaded ({ctx_len} chars, type: str).\n"
            f"Preview:\n{preview}\n\n"
            f"Process this context according to your task instructions."
        )

    def _consume_generator(self, gen: Generator) -> tuple[str, int, int]:
        """Consume a provider generator, returning (full_text, prompt_tokens, generation_tokens)."""
        text_parts = []
        prompt_tokens = 0
        generation_tokens = 0

        for chunk in gen:
            if hasattr(chunk, "text"):
                text_parts.append(chunk.text)
            if hasattr(chunk, "prompt_tokens") and chunk.prompt_tokens:
                prompt_tokens = chunk.prompt_tokens
            if hasattr(chunk, "generation_tokens") and chunk.generation_tokens:
                generation_tokens = chunk.generation_tokens

        return "".join(text_parts), prompt_tokens, generation_tokens

    def _make_llm_query(self, request: RLMRequest, counter: list[int]):
        """Create the llm_query() closure injected into the REPL namespace."""
        def llm_query(query: str) -> str:
            counter[0] += 1
            model_id = request.sub_model or request.model
            provider = self.router.get_provider(model_id)

            chat_req = ChatRequest(
                model=model_id,
                messages=[ChatMessage(role="user", content=query)],
                max_tokens=request.sub_max_tokens or request.max_tokens,
                temperature=request.sub_temperature if request.sub_temperature is not None else request.temperature,
                top_p=request.sub_top_p if request.sub_top_p is not None else request.top_p,
            )
            gen = provider.create_chat_completion(chat_req)
            text, _, _ = self._consume_generator(gen)
            return text
        return llm_query

    def _init_namespace(self, request: RLMRequest, sub_query_counter: list[int]) -> dict:
        """Build the REPL namespace with context, FINAL/FINAL_VAR, and llm_query."""
        namespace: dict = {"context": request.context, "_rlm_final": None}

        def final_fn(value=""):
            namespace["_rlm_final"] = str(value)

        def final_var_fn(name):
            if name in namespace:
                namespace["_rlm_final"] = str(namespace[name])
            else:
                namespace["_rlm_final"] = f"[Variable {name!r} not found]"

        namespace["FINAL"] = final_fn
        namespace["FINAL_VAR"] = final_var_fn
        namespace["llm_query"] = self._make_llm_query(request, sub_query_counter)
        return namespace

    def _iter_loop(
        self,
        request: RLMRequest,
        provider,
        request_id: str,
    ) -> Generator[_IterStart | _AssistantResponse | _CodeOutput | _LoopResult, None, None]:
        """Core iteration loop. Yields typed events consumed by both sync and streaming paths."""
        sub_query_counter: list[int] = [0]
        namespace = self._init_namespace(request, sub_query_counter)

        system_prompt = self._build_system_prompt(request.query, request.system)
        context_msg = self._build_context_metadata(request.context)

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=context_msg),
        ]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        trace: list[RLMTraceEntry] = []
        final_answer = ""
        finish_reason = "max_iterations"
        response_text = ""

        for iteration in range(request.max_iterations):
            yield _IterStart(iteration=iteration + 1)

            chat_req = ChatRequest(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                enable_thinking=request.enable_thinking,
            )
            gen = provider.create_chat_completion(chat_req)
            response_text, pt, ct = self._consume_generator(gen)
            total_prompt_tokens += pt
            total_completion_tokens += ct

            logger.info(f"[RLM {request_id}] iteration={iteration + 1} response_len={len(response_text)}")

            yield _AssistantResponse(iteration=iteration + 1, text=response_text)

            detail = request.include_trace_detail
            code = extract_repl_block(response_text)
            if code is None:
                final_answer = response_text
                trace.append(RLMTraceEntry(
                    iteration=iteration + 1,
                    action="direct_response",
                    response=response_text if detail else None,
                ))
                finish_reason = "direct_response"
                break

            stdout, stderr = run_code_in_namespace(
                code, namespace,
                timeout=request.timeout,
                max_output_chars=request.max_output_chars,
                sandbox=request.sandbox,
            )

            yield _CodeOutput(
                iteration=iteration + 1,
                stdout=stdout, stderr=stderr, code_len=len(code),
            )

            entry = RLMTraceEntry(
                iteration=iteration + 1,
                code_len=len(code),
                stdout_len=len(stdout),
                stderr_len=len(stderr),
                response=response_text if detail else None,
                code=code if detail else None,
                stdout=stdout if detail else None,
                stderr=stderr if detail else None,
            )

            if namespace["_rlm_final"] is not None:
                final_answer = namespace["_rlm_final"]
                entry.action = "FINAL"
                trace.append(entry)
                finish_reason = "final"
                break

            trace.append(entry)
            messages.append(ChatMessage(role="assistant", content=response_text))
            messages.append(ChatMessage(role="user", content=_build_feedback(stdout, stderr)))
        else:
            if namespace["_rlm_final"] is not None:
                final_answer = namespace["_rlm_final"]
                finish_reason = "final"
            else:
                final_answer = response_text or "[Max iterations reached]"
                trace.append(RLMTraceEntry(iteration=request.max_iterations, action="max_iterations"))

        yield _LoopResult(
            answer=final_answer,
            finish_reason=finish_reason,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            trace=trace,
            sub_queries=sub_query_counter[0],
        )

    def run(self, request: RLMRequest) -> RLMResponse:
        """Run the RLM loop synchronously. Call from asyncio.to_thread()."""
        request_id = f"rlm-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        provider = self.router.get_provider(request.model)
        self.router.pin_model(request.model)

        try:
            result: _LoopResult | None = None
            for event in self._iter_loop(request, provider, request_id):
                if isinstance(event, _LoopResult):
                    result = event

            assert result is not None
            return RLMResponse(
                id=request_id,
                created=created,
                model=request.model,
                answer=result.answer,
                finish_reason=result.finish_reason,
                usage={
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
                rlm=RLMMetadata(
                    iterations=len(result.trace),
                    sub_queries=result.sub_queries,
                    trace=result.trace,
                ),
            )
        finally:
            self.router.unpin_model(request.model)

    def run_streaming(self, request: RLMRequest) -> Generator[str, None, None]:
        """Run the RLM loop, yielding SSE events."""
        request_id = f"rlm-{uuid.uuid4().hex[:12]}"

        provider = self.router.get_provider(request.model)
        self.router.pin_model(request.model)

        try:
            yield format_sse("rlm_start", {
                "id": request_id,
                "model": request.model,
                "context_length": len(request.context),
            })

            for event in self._iter_loop(request, provider, request_id):
                if isinstance(event, _IterStart):
                    yield format_sse("iteration_start", {"iteration": event.iteration})
                elif isinstance(event, _AssistantResponse):
                    yield format_sse("assistant_response", {
                        "iteration": event.iteration,
                        "text": event.text,
                    })
                elif isinstance(event, _CodeOutput):
                    yield format_sse("code_output", {
                        "iteration": event.iteration,
                        "stdout": event.stdout,
                        "stderr": event.stderr,
                        "code_len": event.code_len,
                    })
                elif isinstance(event, _LoopResult):
                    usage = {
                        "prompt_tokens": event.prompt_tokens,
                        "completion_tokens": event.completion_tokens,
                        "total_tokens": event.prompt_tokens + event.completion_tokens,
                    }
                    metadata = RLMMetadata(
                        iterations=len(event.trace),
                        sub_queries=event.sub_queries,
                        trace=event.trace,
                    )
                    yield format_sse("rlm_complete", {
                        "answer": event.answer,
                        "finish_reason": event.finish_reason,
                        "usage": usage,
                        "rlm": metadata.model_dump(),
                    })
        finally:
            self.router.unpin_model(request.model)


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

rlm_router = APIRouter(tags=["RLM"])


@rlm_router.post(
    "/v1/rlm/completions",
    summary="RLM Completion",
    description=(
        "Recursive Language Model inference. The model writes Python code to "
        "iteratively explore and transform the provided context, calling itself "
        "recursively via llm_query() as needed."
    ),
    response_model=RLMResponse,
)
async def rlm_completions(request: RLMRequest, http_request: Request):
    router: ModelRouter = http_request.app.state.router_instance
    engine = RLMEngine(router)

    if request.stream:
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        loop = asyncio.get_running_loop()

        def _produce():
            try:
                for event in engine.run_streaming(request):
                    asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
            except Exception as exc:
                logger.error(f"RLM streaming error: {exc}", exc_info=True)
                error_event = format_sse("rlm_error", {"error": str(exc)})
                asyncio.run_coroutine_threadsafe(queue.put(error_event), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        async def _async_stream():
            task = loop.run_in_executor(None, _produce)
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
            finally:
                await task

        return StreamingResponse(
            _async_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        response = await asyncio.to_thread(engine.run, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return response
