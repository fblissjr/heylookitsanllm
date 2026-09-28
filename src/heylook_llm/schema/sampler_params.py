# src/heylook_llm/schema/sampler_params.py
"""The stored sampler bag: a document's or a preset's `params`.

Validated where it is written (conversation, notebook and preset routes), so
the store holds only what a generation can send. Until v2.0.184 any JSON
object was stored: an unknown key sat there doing nothing, and a value of the
wrong type (`top_p: "high"`) was accepted and then failed every generation on
that document.

The keys are the request's own sampler fields (`samplers.REQUEST_SAMPLER_FIELDS`)
and each one's type and range are `ChatRequest`'s field, so a field added or
re-ranged there is accepted here with no second copy. Whether a MODEL can use a
key (a thinking control, a depth its template offers) is not judged here: the
store is model-agnostic and a document can change model. The page removes what
the selected model cannot use and says so; the generate route still drops it
at send for any other client.
"""

from typing import Annotated, Any

from pydantic import AfterValidator, ConfigDict, ValidationError, WithJsonSchema, create_model

from heylook_llm.config import ChatRequest
from heylook_llm.samplers import REQUEST_SAMPLER_FIELDS

_FIELDS: dict[str, Any] = {
    k: (ChatRequest.model_fields[k].annotation, ChatRequest.model_fields[k])
    for k in REQUEST_SAMPLER_FIELDS
}
_Bag = create_model("SamplerParams", __config__=ConfigDict(extra="forbid"), **_FIELDS)


def _validate(value: dict[str, Any]) -> dict[str, Any]:
    try:
        bag = _Bag.model_validate(value)
    except ValidationError as e:
        # One line per problem, naming the key: the message reaches a person
        # (the page's status line) as well as a program.
        problems = []
        for err in e.errors():
            key = ".".join(str(p) for p in err["loc"])
            problems.append(f"unknown key {key!r}" if err["type"] == "extra_forbidden"
                            else f"{key}: {err['msg']}")
        raise ValueError("; ".join(problems)) from None
    # A null is the same as absent (the model's own default), so it is not
    # stored; the coerced values are (an int temperature comes back a float).
    return bag.model_dump(exclude_none=True)


SamplerParams = Annotated[
    dict[str, Any],
    AfterValidator(_validate),
    WithJsonSchema(_Bag.model_json_schema()),
]
