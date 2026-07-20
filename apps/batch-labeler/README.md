# batch-labeler

Last updated: 2026-07-20

Standalone CLI for batch VLM image labeling against heylookitsanllm (or any
OpenAI-compatible `/v1/chat/completions` server). Ships rich built-in task
templates (structured labels, captions, tags, OCR), supports the server's
thinking mode, visual token budget, server-side resizing, and sampler presets,
and stores resumable results in JSONL.

## Why a rebuild (v0.2.0)

The v0.1 tool was a bare client: one hardcoded user prompt, no thinking
control, no vision budget, no retries, and it required hand-writing a system
prompt every run. v0.2 bundles curated prompts as tasks and exposes the
server capabilities that landed after v0.1 (enable_thinking, vision_tokens,
presets, performance telemetry).

## Install

```bash
cd apps/batch-labeler
uv sync
```

## Quick start

```bash
# 1. Start the server (repo root, separate terminal)
uv run heylookllm --port 8080

# 2. See what's available
uv run batch-labeler models          # vision-capable models highlighted
uv run batch-labeler tasks           # built-in task templates
uv run batch-labeler tasks label     # show a task's full prompts

# 3. Test-drive one image before committing to a batch
uv run batch-labeler try photo.jpg -m gemma-4-26b-a4b-it-8bit-mlx

# 4. Run the batch
uv run batch-labeler run path/to/dataset -m gemma-4-26b-a4b-it-8bit-mlx -o results.jsonl
```

If exactly one vision model is loaded on the server, `--model` can be omitted.

## Built-in tasks

| Task | Output | Description |
|------|--------|-------------|
| `label` (default) | JSON | Taxonomy labels: category, objects, colors, style, setting, lighting, mood, quality issues, confidence |
| `caption` | text | Dense single-paragraph caption, training-data style |
| `tags` | JSON | 5-20 flat keyword tags for search/filtering |
| `ocr` | JSON | Verbatim text extraction with language + legibility |

Each task carries its own system prompt, per-image user prompt, sampler preset
(`vlm-extract` / `vlm-describe` from the server's preset registry), max_tokens,
and -- for JSON tasks -- required keys that are validated per record.

### Custom tasks

Write a TOML file and pass `--task-file`:

```toml
[task]
name = "bird-id"
description = "Backyard bird photo identification"
system_prompt = """
You identify birds in photos. Respond with EXACTLY one JSON object:
{"species": string, "common_name": string, "count": integer, "behavior": string, "id_confidence": "low|medium|high"}
"""
user_prompt = "Identify the birds in this photo."
expects_json = true
required_keys = ["common_name", "id_confidence"]
preset = "vlm-extract"
max_tokens = 512
```

Unknown keys are rejected (catches typos). `--system-prompt`,
`--system-prompt-file`, and `--user-prompt` override any task's prompts.

## Server-feature flags

| Flag | Maps to | Notes |
|------|---------|-------|
| `--think` / `--no-think` | `enable_thinking` | Thinking-capable models (see `models` output); thinking text is stored in its own `thinking` field, never polluting the label |
| `--vision-tokens N` | `vision_tokens` | Visual token budget per image (16-16384), snapped to the model's processor grid |
| `--resize-max N` | `resize_max` | Server-side downscale before encoding; big win on phone-camera originals |
| `--image-quality Q` | `image_quality` | JPEG quality for the resize path |
| `--preset NAME` | `preset` | Server sampler preset; overrides the task's default |
| `--temperature/--top-p/--seed/--max-tokens` | same | Explicit values beat preset and task defaults (server-side cascade) |

## Run options

```
batch-labeler run IMAGE_DIR [-o results.jsonl] [--limit N] [--no-recursive]
                 [--dry-run] [--retries 2] [--timeout 300] [--server URL]
```

- `--dry-run` scans, reports counts, and prints the fully-resolved settings.
- `--limit N` processes only the first N pending images -- sample a batch,
  inspect, then run the rest.
- Server URL default is `http://localhost:8080`, overridable via the
  `BATCH_LABELER_SERVER` env var.
- Transient failures (timeout, connection, 5xx) retry with backoff; 4xx fail
  the image immediately. Failed images are NOT written, so a re-run retries
  exactly those.

## Output format

One JSON object per line:

```json
{
  "file_path": "path/to/dataset/photo1.jpg",
  "file_hash": "abc123...",
  "file_name": "photo1.jpg",
  "model_id": "gemma-4-26b-a4b-it-8bit-mlx",
  "task": "label",
  "label": {"category": "portrait", "...": "..."},
  "parse_ok": true,
  "raw_output": "...",
  "thinking": "only present when the model produced thinking",
  "usage": {"prompt_tokens": 1, "completion_tokens": 236, "total_tokens": 237},
  "performance": {"prompt_tps": 21.9, "generation_tps": 56.3, "peak_memory_gb": 28.5},
  "generation_time_ms": 13323,
  "timestamp": "2026-07-20T12:00:00",
  "settings": {"model": "...", "task": "label", "preset": "vlm-extract", "max_tokens": 1024}
}
```

- JSON tasks: `label` is the parsed object (`parse_ok: false` + `label: null`
  when the model's output wasn't valid JSON; `missing_keys` lists absent
  required keys). Text tasks: `label` is the raw string.
- `settings` echoes every non-default knob for reproducibility.

## Resume

Re-running the same command skips images whose `file_hash` already appears in
the output file. Safe to ctrl-c anytime; partial results are flushed per image.

## Tests

```bash
cd apps/batch-labeler
uv sync --dev
uv run pytest tests/ -v
```
