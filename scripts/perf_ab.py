"""A/B one model's speed and memory across code versions or configs.

Opt-in instrument, never part of a suite: it loads real models. The question
it answers is "what did this change cost or buy, and is it more than noise?"
-- with the conditions written into the record, so a number in a changelog
can point at a file instead of standing alone.

    # two code versions (git revs, or `.` for the working tree)
    uv run python scripts/perf_ab.py run --model Qwen3.5-0.8B-MLX-8bit \\
        --arm base=HEAD~1 --arm new=.

    # one version, two configs (JSON merged into the model's config)
    uv run python scripts/perf_ab.py run --model <gguf-id> \\
        --arm f16=. --arm 'q8=.:{"cache_type_k":"q8_0","cache_type_v":"q8_0"}'

    uv run python scripts/perf_ab.py report internal/claude/perf/ab_<...>.json

HOW IT KEEPS AN A/B HONEST
- One arm per FRESH process (a failed Metal command buffer or a warm
  allocator in one arm cannot leak into the other), rounds alternating arm
  order (A B, then B A) so drift in the machine lands on both.
- A git rev runs from a detached worktree through PYTHONPATH, and the worker
  PRINTS the heylook_llm it imported, which is checked against the arm: an
  editable install resolves to this checkout from anywhere, so "I ran the old
  code" is verified, never assumed (AGENTS.md, git-archive rule). All arms
  share this checkout's venv: a dependency change is not what this compares.
- Every request is cold (a nonce leads the prompt) unless the workload says
  follow-up, which measures a reused prefix on purpose.
- The first request in each process is warmup and is dropped.
- Before each arm process it waits for the machine to be quiet (no other
  inference process, GPU idle), and before round 0 it reads the weights once
  so no arm pays a cold disk read the others skip.
- Timing is read the same way on every engine, from the provider's own
  stream: time to the first chunk carrying output, and decode rate as
  (tokens - 1) over the time from first to last such chunk.
- Machine load is RECORDED, not assumed: other inference processes at start
  and end, and GPU utilization before the model loads. `report` refuses to
  call a verdict on a contaminated pair.

SAMPLED DURING EACH RUN (a side thread, every --interval seconds)
process tree physical footprint (what Activity Monitor calls Memory; the
llama-server child included), MLX active/cache/peak (MLX engine only),
system available and wired memory, swap used, the kernel's memory-pressure
level, system CPU, GPU utilization and GPU memory in use (ioreg; no root).

The verdict per metric is deliberately crude: two arms whose rep ranges
overlap are NOISE. It errs toward "no change", which is the safe direction
for a claim.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILLER = ("The archive held letters from a lighthouse keeper who wrote every "
          "evening about the weather, the ships, the gulls and the long dark. ")

# Built-in workloads. `words` is prompt length in words; `gen` the generation
# cap; `followup` adds a second turn to the SAME conversation (reuse on
# purpose); `image` attaches a synthetic picture (vision models only):
# "fresh" draws a new one per request (a cold vision tower), "repeat" the same
# one every time (the vision feature cache hits after warmup).
WORKLOADS = {
    "short": {"words": 150, "gen": 128},
    "long": {"words": 6000, "gen": 128},
    "followup": {"words": 1500, "gen": 64, "followup": 60},
    "vision": {"words": 40, "gen": 64, "image": "fresh"},
    "vision_repeat": {"words": 40, "gen": 64, "image": "repeat"},
    "vision_followup": {"words": 40, "gen": 64, "image": "fresh", "followup": 60},
}

# metric -> True when higher is better
METRICS = {
    "ttft_s": False, "decode_tps": True, "followup_ttft_s": False,
    "peak_footprint_gb": False, "peak_mlx_gb": False, "load_s": False,
}


# ---------------------------------------------------------------------------
# machine counters
# ---------------------------------------------------------------------------

class _RusageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten")]


def _libproc():
    try:
        return ctypes.CDLL("/usr/lib/libproc.dylib")
    except OSError:
        return None


_LIBPROC = _libproc()


def phys_footprint(pid: int) -> int | None:
    """The process's physical footprint in bytes (unified memory, Metal
    allocations included), from proc_pid_rusage. None where unavailable."""
    if _LIBPROC is None:
        return None
    info = _RusageInfoV2()
    if _LIBPROC.proc_pid_rusage(int(pid), 2, ctypes.byref(info)) != 0:
        return None
    return int(info.ri_phys_footprint)


def pressure_level() -> int | None:
    """kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical."""
    libc = ctypes.CDLL(None)
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    rc = libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level",
                           ctypes.byref(value), ctypes.byref(size), None, 0)
    return int(value.value) if rc == 0 else None


def gpu_counters() -> dict:
    """GPU utilization % and GPU memory in use, from ioreg (no root)."""
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return {}
    util = re.search(r'"Device Utilization %"=(\d+)', out)
    used = re.search(r'"In use system memory"=(\d+)', out)
    return {"gpu_util": int(util.group(1)) if util else None,
            "gpu_mem_gb": round(int(used.group(1)) / 1e9, 3) if used else None}


def foreign_inference(own: set[int]) -> list[str]:
    """Other inference processes on the machine (not this run's tree)."""
    import psutil

    found = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] in own:
            continue
        cmd = " ".join(p.info.get("cmdline") or [])
        if re.search(r"heylookllm|llama-server|perf_ab\.py _worker|mlx_lm|mlx_vlm", cmd):
            found.append(f"{p.info['pid']}: {cmd[:120]}")
    return found


class Sampler(threading.Thread):
    """Samples the counters until stopped. `phase` is set by the worker so
    each sample says what the model was doing."""

    def __init__(self, interval: float, mlx: bool):
        super().__init__(daemon=True)
        self.interval, self.mlx = interval, mlx
        self.samples: list[dict] = []
        self.phase = "start"
        self._halt = threading.Event()

    def run(self):
        import psutil

        me = psutil.Process()
        psutil.cpu_percent(interval=None)
        t0 = time.perf_counter()
        while not self._halt.is_set():
            tree = [me, *me.children(recursive=True)]
            row = {"t": round(time.perf_counter() - t0, 3), "phase": self.phase}
            prints = [phys_footprint(p.pid) for p in tree]
            row["footprint_gb"] = round(sum(x for x in prints if x) / 1e9, 3)
            vm, sw = psutil.virtual_memory(), psutil.swap_memory()
            row.update(available_gb=round(vm.available / 1e9, 3),
                       wired_gb=round(getattr(vm, "wired", 0) / 1e9, 3),
                       swap_gb=round(sw.used / 1e9, 3), pressure=pressure_level(),
                       cpu=psutil.cpu_percent(interval=None), **gpu_counters())
            if self.mlx:
                try:
                    import mlx.core as mx
                    row.update(mlx_active_gb=round(mx.get_active_memory() / 1e9, 3),
                               mlx_cache_gb=round(mx.get_cache_memory() / 1e9, 3),
                               mlx_peak_gb=round(mx.get_peak_memory() / 1e9, 3))
                except Exception:  # noqa: BLE001
                    pass
            self.samples.append(row)
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)


# ---------------------------------------------------------------------------
# worker: one arm, one fresh process
# ---------------------------------------------------------------------------

def _resolve_config(model_id: str, overrides: dict) -> tuple[str, dict]:
    import tomllib

    from heylook_llm.config import ModelConfig
    from heylook_llm.model_registry import discover, merge_discovered

    from heylook_llm.router import CONFIG_FILENAME

    data = tomllib.loads((REPO / CONFIG_FILENAME).read_text())
    for m in merge_discovered(data, discover(data))["models"]:
        if m["id"] == model_id:
            mc = ModelConfig.model_validate(m)
            cfg = mc.config.model_dump()
            cfg.update(overrides)
            return mc.provider, cfg
    sys.exit(f"{model_id}: not served (heylook.toml + discovery)")


def _image_url(variant: int = 0) -> str:
    """A synthetic picture; each ``variant`` differs in its pixels, so a
    content-keyed vision feature cache never matches two variants."""
    import base64
    import io

    from PIL import Image, ImageDraw

    bg = (30, 120, 200) if variant == 0 else (variant % 251, variant // 251 % 251, 77)
    im = Image.new("RGB", (768, 768), bg)
    ImageDraw.Draw(im).ellipse((200, 200, 560, 560), fill=(240, 60, 40))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _prompt(nonce: int, words: int) -> str:
    body = (FILLER * (words // len(FILLER.split()) + 1)).split()[:words]
    return f"[{nonce * 7919}] Read this, then continue it.\n\n" + " ".join(body)


def _request(model_id, messages, gen, seed):
    from heylook_llm.config import ChatRequest

    return ChatRequest.model_validate({
        "model": model_id, "messages": messages, "max_tokens": gen,
        "temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "enable_thinking": False, "seed": seed})


def _timed(provider, request) -> dict:
    t0 = time.perf_counter()
    first = last = None
    n, prompt_tokens, cached, text = 0, None, None, ""
    for chunk in provider.create_chat_completion(request):
        now = time.perf_counter()
        text += chunk.text or ""
        # The clock runs off the first chunk that CARRIES generated output
        # (text or pre-split thinking), not off `generation_tokens`: gguf
        # reports the count only on its final usage chunk, so keying on it
        # made every gguf TTFT the whole wall time and its decode rate null
        # (found 2026-09-26 on the flash-attention A/B). The count still
        # comes from the provider, at whatever chunk it arrives.
        if chunk.text or getattr(chunk, "thinking", None):
            if first is None:
                first = now
            last = now
        if chunk.generation_tokens:
            n = chunk.generation_tokens
        prompt_tokens = chunk.prompt_tokens or prompt_tokens
        if getattr(chunk, "cache", None) is not None:
            cached = chunk.cache.cached_tokens
    return {"ttft_s": round(first - t0, 4) if first else None,
            "decode_tps": round((n - 1) / (last - first), 2) if n > 1 and last > first else None,
            "gen_tokens": n, "prompt_tokens": prompt_tokens, "cached_tokens": cached,
            "wall_s": round(time.perf_counter() - t0, 4), "reply": text}


def worker(args) -> None:
    import heylook_llm

    print(f"HEYLOOK_FILE {heylook_llm.__file__}", flush=True)
    overrides = json.loads(args.overrides or "{}")
    provider_name, cfg = _resolve_config(args.model, overrides)
    own = {os.getpid()}
    foreign_start = foreign_inference(own)
    idle_gpu = gpu_counters().get("gpu_util")
    sampler = Sampler(args.interval, mlx=provider_name == "mlx")
    sampler.start()

    if provider_name == "mlx":
        from concurrent.futures import ThreadPoolExecutor

        import mlx.core as mx
        from mlx_vlm.generate.common import generation_stream

        from heylook_llm.providers.mlx_provider import MLXProvider as Provider
        pool = ThreadPoolExecutor(1)

        def on_worker(fn):
            def run():
                with mx.stream(generation_stream):
                    return fn()
            return pool.submit(run).result()
    else:
        from heylook_llm.providers.llama_server_provider import LlamaServerProvider as Provider

        def on_worker(fn):
            return fn()

    provider = Provider(args.model, cfg, False)
    sampler.phase = "load"
    t = time.perf_counter()
    on_worker(provider.load_model)
    load_s = round(time.perf_counter() - t, 3)
    vision = bool(getattr(provider, "is_vlm", False) or cfg.get("mmproj_path"))

    requests = []
    nonce = args.nonce_base
    for name in args.workloads.split(","):
        spec = WORKLOADS[name]
        if spec.get("image") and not vision:
            continue
        for rep in range(args.reps + 1):  # rep 0 is warmup
            nonce += 1
            content = _prompt(nonce, spec["words"])
            if spec.get("image"):
                variant = nonce if spec["image"] == "fresh" else 0
                content = [{"type": "image_url", "image_url": {"url": _image_url(variant)}},
                           {"type": "text", "text": content}]
            messages = [{"role": "user", "content": content}]
            sampler.phase = f"{name}:{rep}"
            row = on_worker(lambda: _timed(provider, _request(args.model, messages, spec["gen"], nonce)))
            if spec.get("followup"):
                messages += [{"role": "assistant", "content": row["reply"]},
                             {"role": "user", "content": _prompt(0, spec["followup"])}]
                sampler.phase = f"{name}:{rep}:followup"
                fu = on_worker(lambda: _timed(provider, _request(args.model, messages, spec["gen"], nonce)))
                row.update(followup_ttft_s=fu["ttft_s"], followup_cached=fu["cached_tokens"],
                           followup_prompt_tokens=fu["prompt_tokens"])
            row.pop("reply", None)
            row.update(workload=name, rep=rep, warmup=rep == 0)
            requests.append(row)
    sampler.phase = "done"
    time.sleep(args.interval * 2)
    sampler.stop()
    try:
        on_worker(provider.unload)
    except Exception:  # noqa: BLE001
        pass
    Path(args.out).write_text(json.dumps({
        "provider": provider_name, "config_overrides": overrides, "load_s": load_s,
        "foreign_start": foreign_start, "foreign_end": foreign_inference(own),
        "idle_gpu_util": idle_gpu, "requests": requests, "samples": sampler.samples,
    }))


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def _git(*args, cwd=REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


def _arm_source(spec: str, scratch: Path, created: list[Path]) -> tuple[str, str, bool, dict]:
    """(src dir, commit, dirty, overrides) for `rev[:json]`. A worktree this
    call adds is appended to `created`, so the run can remove it at the end."""
    rev, _, extra = spec.partition(":")
    overrides = json.loads(extra) if extra else {}
    if rev in (".", ""):
        return (str(REPO / "src"), _git("rev-parse", "HEAD"),
                bool(_git("status", "--porcelain", "--", "src")), overrides)
    sha = _git("rev-parse", rev)
    tree = scratch / f"tree-{sha[:12]}"
    if not tree.exists():
        _git("worktree", "add", "--detach", str(tree), sha)
        created.append(tree)
    return str(tree / "src"), sha, False, overrides


def _conditions() -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    versions = {}
    from importlib.metadata import distribution
    for dist in ("mlx", "mlx-vlm", "transformers"):
        try:
            d = distribution(dist)
            versions[dist] = d.version
            # A git pin's version string does not name the commit; the
            # install record does.
            direct = json.loads(d.read_text("direct_url.json") or "{}")
            if direct.get("vcs_info", {}).get("commit_id"):
                versions[dist] += f"@{direct['vcs_info']['commit_id'][:12]}"
        except Exception:  # noqa: BLE001
            versions[dist] = None
    build = None
    try:
        from heylook_llm.providers.llama_server_provider import LlamaServerProvider
        manifest = LlamaServerProvider.DEFAULT_BUILD.parents[1] / "heylook-build.json"
        if manifest.exists():
            build = json.loads(manifest.read_text())
    except Exception:  # noqa: BLE001
        pass
    return {"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "machine": sh("sysctl", "-n", "hw.model"),
            "macos": platform.mac_ver()[0], "python": platform.python_version(),
            "versions": versions, "llama_build": build,
            "thermal": sh("pmset", "-g", "therm")}


def wait_quiet(timeout: float, gpu_max: int = 5) -> dict:
    """Block until no other inference process runs and the GPU is idle, up
    to `timeout` seconds. Another session's run in the middle of an A/B is
    the likeliest confound on a shared machine; waiting is cheaper than a
    discarded pair. Returns what was waited for, for the record."""
    t0 = time.monotonic()
    while True:
        foreign = foreign_inference({os.getpid()})
        gpu = gpu_counters().get("gpu_util") or 0
        if not foreign and gpu <= gpu_max:
            return {"waited_s": round(time.monotonic() - t0, 1), "quiet": True}
        if time.monotonic() - t0 > timeout:
            return {"waited_s": round(time.monotonic() - t0, 1), "quiet": False,
                    "foreign": foreign, "gpu_util": gpu}
        time.sleep(5)


def prime_page_cache(model_path: str) -> dict:
    """Read the weights once so the first arm does not pay a cold disk read
    the others skip (it showed as a load-time 'win' for every later arm).
    Skipped when the files would not stay cached anyway."""
    import psutil

    root = Path(model_path).expanduser()
    files = ([p for p in root.iterdir() if p.suffix in (".safetensors", ".gguf")]
             if root.is_dir() else [p for p in root.parent.iterdir() if p.suffix == ".gguf"])
    total = sum(p.stat().st_size for p in files)
    if total > psutil.virtual_memory().total * 0.5:
        return {"primed": False, "bytes": total, "why": "larger than half of RAM"}
    t0 = time.monotonic()
    for p in files:
        with open(p, "rb") as f:
            while f.read(64 << 20):
                pass
    return {"primed": True, "bytes": total, "s": round(time.monotonic() - t0, 1)}


def run(args) -> None:
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "perf_ab"
    scratch.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        _run(args, scratch, created)
    finally:
        # Registered worktrees outlive the temp dir otherwise, one per rev ever
        # measured. SystemExit from a failed arm lands here too.
        for tree in created:
            subprocess.run(["git", "worktree", "remove", "--force", str(tree)],
                           cwd=REPO, capture_output=True)


def _run(args, scratch: Path, created: list[Path]) -> None:
    arms = {}
    for spec in args.arm:
        name, _, source = spec.partition("=")
        src, sha, dirty, overrides = _arm_source(source, scratch, created)
        arms[name] = {"spec": source, "src": src, "commit": sha, "dirty": dirty,
                      "overrides": overrides, "runs": []}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or REPO / "internal" / "claude" / "perf" /
               f"ab_{args.model}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"model": args.model, "workloads": args.workloads, "reps": args.reps,
              "rounds": args.rounds, "interval": args.interval,
              "conditions": _conditions(), "arms": arms}
    sys.path.insert(0, str(REPO / "src"))
    _, cfg = _resolve_config(args.model, {})
    record["page_cache"] = prime_page_cache(str(cfg.get("model_path") or ""))
    names = list(arms)
    nonce = 0
    for rnd in range(args.rounds):
        for name in (names if rnd % 2 == 0 else list(reversed(names))):
            arm = arms[name]
            part = scratch / f"{stamp}-{name}-{rnd}.json"
            nonce += 100_000
            env = {**os.environ, "PYTHONPATH": arm["src"]}
            cmd = [sys.executable, "-u", __file__, "_worker", "--model", args.model,
                   "--workloads", args.workloads, "--reps", str(args.reps),
                   "--interval", str(args.interval), "--nonce-base", str(nonce),
                   "--overrides", json.dumps(arm["overrides"]), "--out", str(part)]
            quiet = wait_quiet(args.quiet_timeout)
            print(f"round {rnd} arm {name} ({arm['spec']}) waited {quiet['waited_s']}s"
                  f"{'' if quiet['quiet'] else ' -- NOT quiet'}", flush=True)
            proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
            imported = re.search(r"HEYLOOK_FILE (.+)", proc.stdout)
            expect = str(Path(arm["src"]) / "heylook_llm")
            if not imported or not imported.group(1).startswith(expect):
                sys.exit(f"arm {name}: imported {imported.group(1) if imported else '?'}, "
                         f"expected under {expect}\n{proc.stderr[-2000:]}")
            if proc.returncode != 0 or not part.exists():
                sys.exit(f"arm {name} failed:\n{proc.stderr[-3000:]}")
            arm["runs"].append({"round": rnd, "quiet_wait": quiet, **json.loads(part.read_text())})
            out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out}")
    report_text(record)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def arm_values(arm: dict) -> dict:
    """{(workload, metric): [per-rep values]} for one arm, warmups dropped;
    the per-run metrics (load, peaks) are keyed under workload '*'."""
    out: dict = {}
    for run in arm["runs"]:
        for r in run["requests"]:
            if r["warmup"]:
                continue
            for m in ("ttft_s", "decode_tps", "followup_ttft_s"):
                if r.get(m) is not None:
                    out.setdefault((r["workload"], m), []).append(r[m])
        out.setdefault(("*", "load_s"), []).append(run["load_s"])
        foot = [s["footprint_gb"] for s in run["samples"] if s.get("footprint_gb")]
        if foot:
            out.setdefault(("*", "peak_footprint_gb"), []).append(max(foot))
        mlx = [s["mlx_peak_gb"] for s in run["samples"] if s.get("mlx_peak_gb")]
        if mlx:
            out.setdefault(("*", "peak_mlx_gb"), []).append(max(mlx))
    return out


# A verdict needs this many values per arm; per-run metrics (load time,
# peaks) get one per process, so they need at least this many rounds.
MIN_VALUES = 3
# Below this relative difference in medians a change is not worth a claim,
# however cleanly the ranges separate (reasoned, not measured: it sits under
# the owner's loop tolerances, which flag 3% slower and 10% more memory).
MIN_EFFECT = 0.02


def verdict(a: list, b: list, higher_is_better: bool) -> str:
    """For arm b against arm a: 'better' / 'worse' only when both have
    MIN_VALUES, the rep ranges do not overlap, and the medians differ by at
    least MIN_EFFECT; otherwise 'noise' (or 'too few runs')."""
    if not a or not b:
        return "n/a"
    if len(a) < MIN_VALUES or len(b) < MIN_VALUES:
        return "too few runs"
    ma, mb = statistics.median(a), statistics.median(b)
    if min(b) <= max(a) and min(a) <= max(b):
        return "noise"
    if ma and abs(mb - ma) / abs(ma) < MIN_EFFECT:
        return "noise"
    return "better" if (mb > ma) == higher_is_better else "worse"


def overall(verdicts: list[str]) -> str:
    better, worse = "better" in verdicts, "worse" in verdicts
    if better and worse:
        return "tradeoff"
    if better:
        return "free lunch"
    if worse:
        return "regression"
    return "no change beyond noise"


def contamination(record: dict) -> list[str]:
    notes = []
    for name, arm in record["arms"].items():
        for run in arm["runs"]:
            if run.get("foreign_start") or run.get("foreign_end"):
                notes.append(f"{name} round {run['round']}: other inference processes ran "
                             f"({len(run.get('foreign_start') or [])} at start, "
                             f"{len(run.get('foreign_end') or [])} at end)")
            if (run.get("idle_gpu_util") or 0) > 5:
                notes.append(f"{name} round {run['round']}: GPU {run['idle_gpu_util']}% busy before load")
    return notes


def report_text(record: dict) -> str:
    names = list(record["arms"])
    lines = [f"model {record['model']}  arms {', '.join(names)}  "
             f"rounds {record['rounds']} x reps {record['reps']}"]
    values = {n: arm_values(record["arms"][n]) for n in names}
    keys = sorted({k for v in values.values() for k in v})
    base = names[0]
    verdicts = []
    for key in keys:
        wl, metric = key
        cells = []
        for n in names:
            vs = values[n].get(key, [])
            cells.append(f"{n}: {statistics.median(vs):.4g} [{min(vs):.4g}..{max(vs):.4g}]"
                         if vs else f"{n}: -")
        tail = ""
        for n in names[1:]:
            v = verdict(values[base].get(key, []), values[n].get(key, []), METRICS.get(metric, False))
            verdicts.append(v)
            tail += f"  {n} vs {base}: {v}"
        lines.append(f"  {wl:9s} {metric:18s} " + "  ".join(cells) + tail)
    dirty = [n for n in names if record["arms"][n]["dirty"]]
    if dirty:
        lines.append(f"  note: arm(s) {', '.join(dirty)} ran an uncommitted src/ tree")
    notes = contamination(record)
    if notes:
        lines.append("  CONTAMINATED -- no verdict:")
        lines += [f"    {n}" for n in notes]
    else:
        lines.append(f"  overall: {overall(verdicts)}")
    text = "\n".join(lines)
    print(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--arm", action="append", required=True,
                   help="name=rev[:json-overrides]; rev '.' is the working tree")
    r.add_argument("--workloads", default="short,long,followup,vision")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--rounds", type=int, default=2)
    r.add_argument("--interval", type=float, default=0.5)
    r.add_argument("--quiet-timeout", type=float, default=900,
                   help="seconds to wait for other inference to stop before each arm")
    r.add_argument("--out")
    p = sub.add_parser("report")
    p.add_argument("path")
    w = sub.add_parser("_worker")
    for flag in ("--model", "--workloads", "--overrides", "--out"):
        w.add_argument(flag)
    w.add_argument("--reps", type=int)
    w.add_argument("--interval", type=float)
    w.add_argument("--nonce-base", type=int)
    args = ap.parse_args()
    if args.cmd == "_worker":
        worker(args)
    elif args.cmd == "run":
        run(args)
    else:
        report_text(json.loads(Path(args.path).read_text()))


if __name__ == "__main__":
    main()
