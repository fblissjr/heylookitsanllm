"""Which ENGINE is each served model? One answer, shared by every live harness.

`tests/smoke/run.py` and `tests/eval/run.py` both had to answer this, and both
answered it from the same two endpoints with their own code. That is this
repo's own named defect class -- a hand-copied second copy that drifts -- and
it had already started: smoke split `mlx` on `effective_loader` while eval only
ever read `capabilities`, so the two disagreed about what a vision model was.

The taxonomy lives HERE, beside the harnesses, and not in `src/heylook_llm/`:
the server has no reason to carry a test taxonomy. What the server owes is the
FACTS -- `provider` and the engine contract's `engine.runtime` on the admin
row -- and it does.

Why the arms are not providers
------------------------------
    provider "mlx"  -> mlx-vlm, text-only model   (mlx-text)
                    -> mlx-vlm, vision model      (mlx-vision)
    provider "gguf" -> llama-server subprocess    (gguf)

Every MLX model runs on mlx-vlm's engine since plan W10 (mlx-lm is gone), but
a text model and a vision model still take different paths through heylook:
the template path (`is_vlm`), media handling, the vision prefill. So "we
covered mlx" is still a claim about a config value, not about code. The
library is `engine.runtime` (mlx-vlm | llama.cpp); the MLX split is the
served `vision` capability, which derives from the same resolver as `is_vlm`
(the retired `loader` field was the only way the two could disagree). Both
are on the admin listing for UNLOADED models too, precisely so a harness can
choose its arms without loading anything.

Stdlib only, same rule as the harnesses that import it (they run as scripts,
not under pytest, and must not need a venv beyond the server's own).
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

# The arms a live run can have. Order is display order.
ARMS = ("mlx-text", "mlx-vision", "gguf")

# The engine (`engine.runtime`, and config's `engines` vocabulary) each arm
# runs on. Pinned against config.ENGINES by test_config_effects.
ARM_ENGINE = {"mlx-text": "mlx-vlm", "mlx-vision": "mlx-vlm", "gguf": "gguf"}


@dataclass
class Coverage:
    """What a server offers, in engine terms."""

    by_engine: dict[str, str] = field(default_factory=dict)      # model_id -> arm
    capabilities: dict[str, set[str]] = field(default_factory=dict)  # model_id -> caps
    resident: set[str] = field(default_factory=set)              # already loaded
    unclassified: dict[str, str] = field(default_factory=dict)   # model_id -> why

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

    The library is ``engine.runtime``; an MLX model's arm is its served
    ``vision`` capability. A row with no runtime (a server older than the
    engine contract, or an admin endpoint behind a token) is UNCLASSIFIED,
    with the reason: named as a coverage hole, never guessed into an arm.
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

        runtime = ((row.get("engine") or {}).get("runtime") or {}).get("value")
        if runtime == "llama.cpp":
            cov.by_engine[mid] = "gguf"
        elif runtime == "mlx-vlm":
            cov.by_engine[mid] = "mlx-vision" if "vision" in caps else "mlx-text"
        elif runtime:
            cov.unclassified[mid] = f"engine.runtime {runtime!r} is no arm this harness knows"
        else:
            cov.unclassified[mid] = (
                "no engine.runtime on the admin row (a server older than the "
                "engine contract, or /v1/admin/models behind a token?)")

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
            lines.append(f"  {arm:<10} covered ({len(models)} model(s) served)")
        elif models:
            lines.append(f"  {arm:<10} UNCOVERED -- {len(models)} model(s) served, none run")
        else:
            lines.append(f"  {arm:<10} UNCOVERED -- no model served for this arm")
    if cov.unclassified:
        lines.append("  unclassified (a coverage hole with no name): "
                     + ", ".join(sorted(cov.unclassified)))
    if narrowed:
        lines.append("  run was NARROWED explicitly; an uncovered arm below is a "
                     "choice, not a gap.")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# model choice per arm
# ---------------------------------------------------------------------------
#
# Moved here from tests/smoke/run.py in v1.79.78 so the JS e2e harness can use
# the SAME answer rather than re-deriving "which model is this arm" in another
# language. Two implementations of that question is the hand-copied-list defect
# this repo keeps paying for; there is one now, and the JS side shells out.


def _post(server: str, path: str, body: dict, timeout: int = 60):
    payload = json.dumps(body).encode()
    try:
        req = urllib.request.Request(
            f"{server.rstrip('/')}{path}", data=payload, method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return None, None


def model_size_gb(server: str, model_id: str):
    """Measured weight size in GB, or None. NO LOAD.

    ``POST /v1/admin/models/{id}/fit`` sizes a model from FILE STATS (plus a
    vm_stat/sysctl read for the verdict, which is ignored here). The number it
    already computes is the only real cost signal any endpoint serves, and it
    is exactly what choosing an arm's model needs.
    """
    st, body = _post(server, f"/v1/admin/models/{urllib.parse.quote(model_id)}/fit",
                     {"headroom_gb": 0})
    if st != 200 or not isinstance(body, dict):
        return None
    gb = body.get("weights_gb")
    return gb if isinstance(gb, (int, float)) else None


def pick_models(server: str, by_arm: dict, overrides: dict, wanted, resident=frozenset()) -> dict:
    """One model per requested arm: resident if there is one, else the SMALLEST.

    Two heuristics, in that order, and the order is the point.

    RESIDENT first because it costs no load at all, and on a single-slot router
    loading something else evicts whatever the owner had running.

    Otherwise the smallest by MEASURED weight size. This used to sort by
    ``len(id)``, which is not a cost signal and never was: on this machine an
    unnarrowed run picked gpt-oss-120b, a 27B and a 27B for the three arms.
    That makes the release standard -- green on all three arms -- something
    nobody would run, which is the same failure the plan is about, one level up.

    Sizing is one cheap POST per candidate and only for the arms actually
    wanted, skipped entirely for an arm with an override. Where the endpoint
    cannot answer (older server, unreadable path) the model sorts LAST rather
    than first: an unknown size must not win a contest about smallness.
    ``--model ARM=ID`` remains the way to be sure.
    """
    chosen = {}
    for arm in wanted:
        if arm in overrides:
            chosen[arm] = overrides[arm]
            continue
        candidates = [m for m, a in by_arm.items() if a == arm]
        if not candidates:
            continue
        already = [m for m in candidates if m in resident]
        if already:
            chosen[arm] = sorted(already)[0]
            continue
        sizes = {m: model_size_gb(server, m) for m in candidates}
        chosen[arm] = sorted(
            candidates,
            key=lambda m: (sizes[m] is None, sizes[m] if sizes[m] is not None else 0.0, m),
        )[0]
    return chosen


def resolve_arms(server: str, wanted=None, overrides=None, *, caps_required=None) -> dict:
    """The whole answer a harness needs, in one call.

    Returns ``{arm: {"model": id, "capabilities": [...]}}`` for every arm that
    has a model, plus the arms that have none. An arm with no model is REPORTED
    as absent rather than omitted silently -- "served but not run" and "no model
    of this engine exists" are different facts and must not print the same.

    ``caps_required`` narrows candidates to models advertising every named
    capability (a gguf model with vision, say).
    """
    cov = classify(server)
    wanted = list(wanted or ARMS)
    overrides = dict(overrides or {})
    by_arm = dict(cov.by_engine)
    if caps_required:
        need = set(caps_required)
        by_arm = {m: a for m, a in by_arm.items()
                  if need <= set(cov.capabilities.get(m, ()))}
    chosen = pick_models(server, by_arm, overrides, wanted, cov.resident)
    return {
        "arms": {
            arm: {
                "model": chosen[arm],
                "capabilities": sorted(cov.capabilities.get(chosen[arm], ())),
                "resident": chosen[arm] in cov.resident,
            }
            for arm in wanted if arm in chosen
        },
        "absent": [a for a in wanted if a not in chosen],
    }


def _main(argv=None) -> int:
    """CLI so a non-Python harness can consume the taxonomy.

    `python -m helpers.engines --server URL --json` is what tests/e2e shells
    out to; it exists so the JS side never re-implements "which arm is this
    model", which the server answers via engine.runtime and capabilities.
    """
    import argparse
    ap = argparse.ArgumentParser(description="Resolve e2e/smoke arms to models.")
    ap.add_argument("--server", required=True)
    ap.add_argument("--arm", action="append", choices=ARMS,
                    help="repeatable; default is every arm")
    ap.add_argument("--model", action="append", default=[], metavar="ARM=ID",
                    help="pin one arm's model, repeatable")
    ap.add_argument("--cap", action="append", default=[], metavar="CAP",
                    help="only consider models advertising this capability")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    overrides = {}
    for spec in args.model:
        arm, _, mid = spec.partition("=")
        if arm not in ARMS or not mid:
            print(f"bad --model {spec!r}; expected ARM=ID", file=sys.stderr)
            return 2
        overrides[arm] = mid

    try:
        out = resolve_arms(args.server, args.arm, overrides, caps_required=args.cap)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for arm, info in out["arms"].items():
            resident = " (resident)" if info["resident"] else ""
            print(f"  {arm:<10} {info['model']}{resident}")
        for arm in out["absent"]:
            print(f"  {arm:<10} NO MODEL -- arm is uncovered, not green")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
