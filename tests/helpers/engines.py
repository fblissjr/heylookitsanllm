"""Which ENGINE is each served model? One answer, shared by every live harness.

`tests/smoke/run.py` and `tests/eval/run.py` both had to answer this, and both
answered it from the same two endpoints with their own code. That is this
repo's own named defect class -- a hand-copied second copy that drifts -- and
it had already started: smoke split `mlx` on `effective_loader` while eval only
ever read `capabilities`, so the two disagreed about what a vision model was.

The taxonomy lives HERE, beside the harnesses, and not in `src/heylook_llm/`:
the server has no reason to carry a test taxonomy. What the server owes is the
FACTS -- `provider` and `effective_loader` on the admin row -- and it now does.

Why the arms are engines and not providers
------------------------------------------
    provider "mlx"  -> mlx-lm   (text)     ) two SEPARATE upstream repos, on
                    -> mlx-vlm  (vision)   ) separate release trains
    provider "gguf" -> llama-server subprocess (one engine, one local binary)

So "we covered mlx" is a claim about a config value, not about code. Which of
the two MLX libraries decodes is `effective_loader`, which the admin listing
serves for UNLOADED models too (v1.79.31) precisely so a harness can choose
its arms without loading anything.

`mlx_embedding` is deliberately out of scope (owner call, 2026-08-28): it
generates nothing, so no lifecycle arm applies to it. It classifies as
EXCLUDED, which is a named outcome -- distinct from unclassifiable, which is a
coverage hole with no name.

Stdlib only, same rule as the harnesses that import it (they run as scripts,
not under pytest, and must not need a venv beyond the server's own).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# The engines a live run can have an arm for. Order is display order.
ARMS = ("mlx-lm", "mlx-vlm", "gguf")

# Providers that are real engines but carry no arm, and why. Reported as
# excluded rather than unclassified: "we chose not to" and "we could not tell"
# are different answers and must not print the same.
EXCLUDED_PROVIDERS = {
    "mlx_embedding": "embeddings generate nothing -- no lifecycle arm applies "
                     "(out of scope, owner call 2026-08-28)",
}


@dataclass
class Coverage:
    """What a server offers, in engine terms."""

    by_engine: dict[str, str] = field(default_factory=dict)      # model_id -> arm
    capabilities: dict[str, set[str]] = field(default_factory=dict)  # model_id -> caps
    resident: set[str] = field(default_factory=set)              # already loaded
    excluded: dict[str, str] = field(default_factory=dict)       # model_id -> why
    unclassified: dict[str, str] = field(default_factory=dict)   # model_id -> why
    unconfirmable: dict[str, str] = field(default_factory=dict)  # named, not confirmed

    def models_for(self, arm: str) -> list[str]:
        return [m for m, a in self.by_engine.items() if a == arm]

    def arms_present(self) -> list[str]:
        return [a for a in ARMS if self.models_for(a)]

    def arms_absent(self) -> list[str]:
        return [a for a in ARMS if not self.models_for(a)]

    def engines_of(self, model_ids) -> list[str]:
        """The arms a chosen model list actually spans -- the question a
        coverage summary answers, and the one neither harness could answer."""
        spanned = {self.by_engine.get(m) for m in model_ids}
        return [a for a in ARMS if a in spanned]


def _get(server: str, path: str, timeout: int = 30):
    """(status, parsed-body-or-None). Never raises for a non-2xx: an admin
    endpoint behind a token answers 401, and that is a fact to report, not a
    traceback."""
    try:
        req = urllib.request.Request(f"{server.rstrip('/')}{path}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            return resp.status, (json.loads(payload) if payload else None)
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return None, None


def classify(server: str, *, get_json=None) -> Coverage:
    """Classify every model a server serves.

    ``get_json``: injectable ``(path) -> (status, body)`` for tests and for a
    harness that already carries an authenticated fetcher. Defaults to plain
    stdlib GETs.

    ENGINE IDENTITY IS ``effective_loader``, not the vision capability, and
    that distinction is the whole reason this module exists:

    - An explicit ``loader = "mlx-lm"`` on a dual-capable VLM still reports the
      vision capability. Splitting on the capability alone puts it in the
      mlx-vlm arm and goes green having run mlx-lm twice.
    - ``loader = "auto"`` degrades vision -> mlx-lm whenever mlx-vlm does not
      register the ``model_type``, which no amount of client-side reasoning can
      see.

    Both are answered by the server now. Where the field is ABSENT (an older
    server, or a row that predates it) the old capability inference runs as a
    fallback and the model is recorded as UNCONFIRMABLE -- named, but not
    claimed. A harness that names engines must be able to say which ones it
    could not confirm.
    """
    fetch = get_json or (lambda path: _get(server, path))

    st, models = fetch("/v1/models")
    if st != 200 or not models:
        raise RuntimeError(f"GET /v1/models failed: {st}")

    st, admin = fetch("/v1/admin/models")
    admin_by_id = {}
    if st == 200 and admin:
        admin_by_id = {m.get("id"): m for m in (admin.get("models") or [])}

    cov = Coverage()
    for entry in (models.get("data") or []):
        mid = entry["id"]
        caps = set(entry.get("capabilities") or [])
        row = admin_by_id.get(mid) or {}
        cov.capabilities[mid] = caps
        if row.get("loaded"):
            cov.resident.add(mid)

        provider = row.get("provider") or entry.get("provider")
        if provider in EXCLUDED_PROVIDERS:
            cov.excluded[mid] = EXCLUDED_PROVIDERS[provider]
            continue
        if provider == "gguf":
            cov.by_engine[mid] = "gguf"
            continue
        if provider != "mlx":
            cov.unclassified[mid] = (
                f"provider {provider!r} names no engine this harness knows"
                if provider else
                "no provider on either /v1/models or /v1/admin/models "
                "(is the admin endpoint behind a token?)")
            continue

        loader = row.get("effective_loader")
        if loader in ("mlx-lm", "mlx-vlm"):
            cov.by_engine[mid] = loader          # the server's own answer
            continue
        # Fallback: the pre-v1.79.31 inference. Kept because a harness pointed
        # at an older server should still run, and degraded honestly because
        # this is precisely the guess the field was added to replace.
        cov.by_engine[mid] = "mlx-vlm" if "vision" in caps else "mlx-lm"
        cov.unconfirmable[mid] = (
            "no `effective_loader` on the admin row -- engine inferred from the "
            "vision capability, which an explicit `loader` or an mlx-vlm "
            "degradation would contradict")

    return cov


def format_coverage(cov: Coverage, *, spanned: list[str] | None = None,
                    narrowed: bool = False) -> str:
    """The coverage paragraph both harnesses print.

    ``spanned``: the arms THIS RUN exercised, when that is narrower than what
    the server offers (an explicit --models / --arm). Absent = the run covered
    everything it could.

    The wording is the point. An engine with no model is UNCOVERED, and an
    uncovered engine is never reported as green -- that sentence is the whole
    invariant this plan exists to establish, so it is printed, not implied.
    """
    lines = []
    ran = spanned if spanned is not None else cov.arms_present()
    for arm in ARMS:
        models = cov.models_for(arm)
        if arm in ran:
            lines.append(f"  {arm:<8} covered ({len(models)} model(s) served)")
        elif models:
            lines.append(f"  {arm:<8} UNCOVERED -- {len(models)} model(s) served, none run")
        else:
            lines.append(f"  {arm:<8} UNCOVERED -- no model served for this engine")
    if cov.unconfirmable:
        lines.append("  engine identity NOT confirmed for: "
                     + ", ".join(sorted(cov.unconfirmable)))
    if cov.unclassified:
        lines.append("  unclassified (a coverage hole with no name): "
                     + ", ".join(sorted(cov.unclassified)))
    if cov.excluded:
        lines.append(f"  excluded by design: {', '.join(sorted(cov.excluded))}")
    if narrowed:
        lines.append("  run was NARROWED explicitly; an uncovered arm below is a "
                     "choice, not a gap.")
    return "\n".join(lines)
