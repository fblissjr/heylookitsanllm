"""MODEL_BUSY must reach the 503, and nothing may quietly eat it.

WHY THIS IS ANCHORED ON THE TRIGGER, NOT ON THE HELPER
------------------------------------------------------
``busy_response.py`` has been the one speller for the busy 503 since
v1.79.53, and a helper cannot make anyone call it. Its docstring used to open
with a COUNT of the endpoints that did -- corrected from three to four in
.53 -- and a sweep the same day found SIX MORE routes answering backpressure
with a 500, a 400, and in one case a 200 carrying the busy sentence inside a
per-group error field. The count was accurate and useless: an enumerated
caller list is a record of who remembered, not a description of the mechanism.

A module whose purpose is being the one speller is structurally unable to
tell you who is NOT using it. So the question "who should have called this
and did not" is only unanswerable while you are anchored on the CALLEE. The
obligation is created by ``router.get_provider(...)``, whose call sites are
enumerable from source -- and the remainder, after subtracting the compliant
ones, is the answer.

WHAT EACH CHECK HERE COVERS, AND WHAT IT DOES NOT
-------------------------------------------------
Say this plainly, because a check whose reach is unstated gets read as
covering more than it does -- which is the failure this whole file is about.

* ``test_the_app_registers_a_model_busy_handler`` pins the MECHANISM. With the
  handler registered, a route that does nothing answers 503 for free; the
  failure mode is inverted, so a new route must actively swallow to get this
  wrong instead of actively remembering to get it right. If the handler is
  ever removed, every route regresses at once and silently -- this is the
  cheapest, highest-value assertion in the file.

* ``test_no_direct_get_provider_site_swallows_model_busy`` is a LOCAL,
  PRECISE property: a ``try`` whose own body calls ``get_provider`` must let
  ``ModelBusyError`` out. No reachability analysis, so no false positives and
  no allowlist to rot. It goes red when someone adds a new swallowing handler
  around a ``get_provider`` call. Verified it can: against the tree at
  856bb27 (immediately before v1.79.57) it reports ``jspace_api.py`` and
  ``batch_processor.py``, the two direct-site defects that release fixed
  (the latter module was deleted outright in v1.79.66 with the OpenAI route).

  A whole-package call-graph version of this was written and DISCARDED.
  Simple-name fixpoint matching over-approximated badly -- it marked most of
  the package reachable, including ``service_manager.uninstall_service_linux``
  and ``memory.tick``, and flagged a long list of handlers that were nearly
  all fine. That check would have needed an exemption list -- which is the
  census again, wearing the clothes of a test.

* WHAT NEITHER COVERS: a route that wraps a get_provider-calling HELPER in its
  own broad handler. That is where four of the six 2026-08-31 defects actually
  lived (``api.py``'s embeddings and hidden_states wrappers): the inner
  handler re-raised correctly and the outer one converted to a 500. It is not
  a local property and it is not cheaply static. ``tests/contract/`` pins
  those routes behaviourally instead. Both kinds are needed; neither
  substitutes for the other.
"""

import ast
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "heylook_llm"

_BROAD = {"Exception", "BaseException", "RuntimeError"}

# The one site that may swallow, with a rationale that INVALIDATES ITSELF if it
# stops being true. `ModelRouter.__init__` pre-warms `--model-id` at startup:
# nothing else can be generating while the router is being constructed, so
# MODEL_BUSY is unreachable there, and a failed pre-warm must not take the
# process down. The exemption is keyed on the enclosing function being
# `__init__` -- move that call into anything that serves a request and the
# exemption stops matching rather than silently covering it.
_STARTUP_ONLY = ("router.py", "__init__")


def _lets_model_busy_out(node: ast.Try) -> bool:
    for handler in node.handlers:
        spelled = ast.unparse(handler.type) if handler.type else "BaseException"
        caught = {x.strip() for x in spelled.strip("()").split(",")}
        if not (caught & _BROAD):
            continue
        # A typed handler on the same try takes it first.
        if any(other is not handler and other.type is not None
               and "ModelBusyError" in ast.unparse(other.type)
               for other in node.handlers):
            return True
        # A bare `raise` re-raises what was caught.
        if any(isinstance(n, ast.Raise) and n.exc is None for n in ast.walk(handler)):
            return True
        return False
    return True


def _direct_get_provider_sites():
    """Every ``try`` whose own body calls ``get_provider``, with its function."""
    for path in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - not our source
            continue
        enclosing = {}
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(fn):
                    enclosing[id(sub)] = fn.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            calls_it = any(
                isinstance(c, ast.Call) and ast.unparse(c.func).endswith(".get_provider")
                for stmt in node.body for c in ast.walk(stmt)
            )
            if calls_it:
                yield path.name, enclosing.get(id(node), "<module>"), node


def test_the_app_registers_a_model_busy_handler():
    """Without this, every route regresses at once and silently."""
    from heylook_llm.api import app
    from heylook_llm.providers.common.generation_gate import ModelBusyError

    assert ModelBusyError in app.exception_handlers, (
        "api.py no longer registers an exception_handler for ModelBusyError. "
        "That handler is what makes the correct answer the DEFAULT -- without "
        "it every route reverts to needing to remember busy_response.py, "
        "which is the arrangement that shipped six wrong statuses."
    )


def test_no_direct_get_provider_site_swallows_model_busy():
    sites = list(_direct_get_provider_sites())
    assert sites, "no get_provider call sites found -- this check has rotted"

    swallowing = [
        f"{fname}:{node.lineno} in {func}()"
        for fname, func, node in sites
        if not _lets_model_busy_out(node) and (fname, func) != _STARTUP_ONLY
    ]
    assert not swallowing, (
        "these handlers catch ModelBusyError and answer something other than "
        "503:\n  " + "\n  ".join(swallowing) + "\n"
        "Backpressure is transient and self-clearing. Reporting it as a 500 "
        "(broken model), a 400 (malformed request) or a 200 (success) all send "
        "a client down a branch that cannot recover. Add "
        "`except ModelBusyError: raise` above the broad handler so it reaches "
        "the app-level handler in api.py."
    )


def test_the_startup_exemption_still_describes_startup():
    """The exemption's reason, asserted rather than written down.

    A rationale in a comment cannot go red when it stops being true -- which
    is how the schema-parity test came to exempt a field for a rename that
    never happened. This one is executable: the exempted site is only exempt
    while it really is the router's constructor.
    """
    fname, func = _STARTUP_ONLY
    matches = [(f, fn) for f, fn, _ in _direct_get_provider_sites()
               if (f, fn) == (fname, func)]
    if not matches:
        pytest.fail(
            f"the {fname}:{func}() exemption no longer matches any get_provider "
            "site -- delete it from _STARTUP_ONLY rather than leaving an "
            "exemption that covers nothing (or, if the call moved into a "
            "request path, it is no longer exempt at all)"
        )
