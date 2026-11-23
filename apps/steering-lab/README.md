# Steering Lab (`apps/steering-lab`)

Standalone **MLX activation steering & mechanistic exploration workbench** built for experimenting with non-1D steering on Apple Silicon (`Swift-1.5-Qwen3.8-27B-MLX-8bit` first, extensible to any `mlx-vlm` checkpoint).

See [PLAN.md](./PLAN.md) for the full math, operator specs, six lab instruments, runtime invariants, and Qwen 3.8 experiment playbook.

## What Is Inside

| Path | Role |
| :--- | :--- |
| `PLAN.md` | Design blueprint, 4 operators (Regime A/B, `IF -> THEN`, `S3`), invariants, and experiment playbook |
| `src/steering_lab/operators.py` | Pure NumPy/MLX operators: Regime A (`peer_ros`, clamped `[0, 1]` union SVD, capability `holds`, `tanh` RMS knee), Regime B (local-centered deadzone ramp + paired-shift `vr_diff` + rollback-safe running mean), `IF X -> THEN Y` ReLU neurons, `S3` random control, FP64 SVD builders & geometry diagnostics |
| `src/steering_lab/bank.py` | Multi-layer raw Span Bank I/O (`image_pad` / `prompt_text` / `think` / `answer`, `FA`/`GDN` layer kinds), model fingerprint validation, vector pack serialization (`.safetensors` + `meta.json`), and tangent-RMSNorm `lm_head` Adds/Erases X-ray |
| `src/steering_lab/chain_lens.py` | **The Chain Lens**: residual Ridge J-Lens transport (`L_mid -> L63`), corpus-mean contrastive multi-token decoding over Qwen 3.8's native MTP head, and round-trip cosine fidelity check |
| `src/steering_lab/corpora.py` | Starter contrastive text + vision corpora and capability hold prompts (math/code/JSON) |
| `src/steering_lab/runtime.py` | White-box MLX steering runtime: dynamic `FA`/`GDN` layer mapping, unrolled verify bypass guard + per-position `</think>` phase masking, `KL` + 4-gram loop telemetry, seed-locked multi-arm sweeps, and teacher-forced span capture |
| `src/steering_lab/cli.py` | CLI for `capture`, `build`, `inspect-bank`, `xray`, `chain-lens`, `demo-bank`, and `serve` |
| `src/steering_lab/server.py` | Local FastAPI + SSE lab server (default port `8111`, never collides with `heylook` on `:8000`/`:42193`) |
| `ui/` | Zero-build vanilla HTML/CSS/JS workbench (multi-slider mixer, geometry/X-ray bench, token heatmap microscope, side-by-side sweep strip) |
| `tests/` | CPU-only test suite covering all math invariants, bank/X-ray I/O, synthetic hybrid GDN+FA runtime + verify parity, and FastAPI endpoints (zero 28 GB GPU load) |

## Quickstart (Using Root `heylookitsanllm` Environment)

No second venv is needed — run directly from the repo root with `PYTHONPATH=apps/steering-lab/src`:

```bash
# 1. Run CPU-only test suite (safe while heylook / H3 is active on Metal)
PYTHONPATH=apps/steering-lab/src uv run --no-sync pytest apps/steering-lab/tests/ -q

# 2. Generate a tiny synthetic demo bank on CPU and launch the workbench UI (zero GPU load)
PYTHONPATH=apps/steering-lab/src uv run --no-sync python -m steering_lab.cli demo-bank --out apps/steering-lab/data/banks/demo_synth
PYTHONPATH=apps/steering-lab/src uv run --no-sync python -m steering_lab.cli serve --port 8111 --bank apps/steering-lab/data/banks/demo_synth
```

Open `http://127.0.0.1:8111/` in your browser.

### Live Qwen 3.8 MLX Workflow (When GPU Is Idle)

Wait until heavy Metal jobs on `:42193` are idle before loading a 28 GB checkpoint:

```bash
# Capture multi-layer raw spans once (saves under gitignored apps/steering-lab/data/banks/)
PYTHONPATH=apps/steering-lab/src uv run --no-sync python -m steering_lab.cli capture \
  --model Swift-1.5-Qwen3.8-27B-MLX-8bit \
  --corpus all \
  --out apps/steering-lab/data/banks/qwen38_swift15

# Build Regime A / B / Conditional vector packs from cached spans (CPU, milliseconds)
PYTHONPATH=apps/steering-lab/src uv run --no-sync python -m steering_lab.cli build \
  --bank apps/steering-lab/data/banks/qwen38_swift15 \
  --out apps/steering-lab/data/vectors/qwen38_swift15

# Inspect cross-layer rank collapse (FA band L39..L51 vs L63) and cosine matrix (CPU)
PYTHONPATH=apps/steering-lab/src uv run --no-sync python -m steering_lab.cli inspect-bank \
  --bank apps/steering-lab/data/banks/qwen38_swift15

# Launch the interactive workbench UI (model loads lazily when you click Load or run a sweep)
PYTHONPATH=apps/steering-lab/src uv run --no-sync python -m steering_lab.cli serve \
  --port 8111 \
  --model Swift-1.5-Qwen3.8-27B-MLX-8bit \
  --bank apps/steering-lab/data/banks/qwen38_swift15 \
  --vectors apps/steering-lab/data/vectors/qwen38_swift15
```

Generated banks and vectors under `apps/steering-lab/data/` are automatically gitignored via `apps/*/data/*`.
