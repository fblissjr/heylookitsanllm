"""Every route that can hit backpressure answers 503, not something else.

The behavioural half of the pair described in
``tests/unit/test_model_busy_reaches_the_handler.py``. That one is a local
static property and cannot see the shape four of these six defects actually
had: a route wrapping a get_provider-calling HELPER in its own broad handler,
where the inner handler re-raised correctly and the OUTER one converted to a
500. Only calling the route can see that.

What was wrong before v1.79.57, all measured by AST enumeration of
``get_provider`` call sites on 2026-08-31 rather than by reading:

    /v1/embeddings                    500
    /v1/hidden_states                 500
    /v1/hidden_states/structured      500
    /v1/jspace/analyze                400   <- worst: "your request is
                                              malformed" for a condition that
                                              clears on its own
    /v1/chat/completions (batch)      500
    /v1/chat/completions (sequential) 200 with the busy sentence stringified
                                          into a per-group error field

The routes that were already correct (`/v1/messages`,
`/v1/conversations/{id}/generate`, `/v1/models/{id}/load`) are pinned in their
own files; this one exists for the ones that were not. The two batch shapes
above were removed outright in v1.79.66 with the OpenAI route, so their pins
are gone too: nothing serves them to answer wrongly.
"""

import pytest

from heylook_llm.providers.common.generation_gate import ModelBusyError

_BUSY = ("MODEL_BUSY: cannot make room -- ['other-model'] is generating. "
         "Stop the generation or wait for it to finish.")


@pytest.fixture
def busy_router(client, monkeypatch):
    """Make the router refuse to hand out a provider, as it does under load."""
    router = client.app.state.router_instance

    def boom(*_args, **_kwargs):
        raise ModelBusyError(_BUSY)

    monkeypatch.setattr(router, "get_provider", boom)
    return client


def _assert_busy(res):
    assert res.status_code == 503, (
        f"backpressure answered {res.status_code}: {res.text[:300]}"
    )
    assert res.headers.get("Retry-After") == "1"
    body = res.json()
    assert body["error"]["code"] == "model_overloaded"
    # The router's own sentence, not a fixed one about a queue: the
    # eviction-blocked case names which models are generating and what to do.
    assert "other-model" in body["error"]["message"]


class TestBusyIsNeverAServerError:
    def test_embeddings(self, busy_router):
        _assert_busy(busy_router.post(
            "/v1/embeddings",
            json={"model": "test-mlx-model", "input": "hello"},
        ))

    def test_hidden_states(self, busy_router):
        _assert_busy(busy_router.post(
            "/v1/hidden_states",
            json={"model": "test-mlx-model", "input": "hello"},
        ))

