# batch-labeler

Last updated: 2026-09-25

Standalone CLI for batch VLM image labeling against heylookitsanllm's Messages
API (`POST /v1/messages`). Ships rich built-in task templates (structured
labels, captions, tags, OCR), supports the server's thinking switch, resizes
images client-side before sending, and stores resumable results in JSONL.

## v0.3.0: the Messages port

v0.2 spoke the OpenAI `/v1/chat/completions` route, which the server removed
in v1.79.66. v0.3 speaks `/v1/messages`: the system prompt is top-level, the
image is a base64 `image` block, and `--think`/`--no-think` send `thinking`.
The server-side knobs v0.2 exposed are gone because the server no longer has
them: named samplers (`--sampler`/`--preset`, removed v2.0.30),
`--vision-tokens` (v2.0.64), and `--resize-max`/`--image-quality` (with the
OpenAI route). Resizing is now client-side (`--max-edge`, below). A custom
task TOML that still sets `sampler` is an unknown-key error.

## Install

```bash
cd apps/batch-labeler
uv sync
```

## Quick start

```bash
# 1. Start the server (repo root, separate terminal)
uv run heylookllm --port 8000

# 2. See what's available
uv run batch-labeler models          # vision-capable models highlighted
uv run batch-labeler tasks           # built-in task templates
uv run batch-labeler tasks label     # show a task's full prompts

# 3. Test-drive one image before committing to a batch
uv run batch-labeler try photo.jpg -m Qwen3.5-0.8B-MLX-8bit

# 4. Run the batch
uv run batch-labeler run path/to/dataset -m Qwen3.5-0.8B-MLX-8bit -o results.jsonl
```

If exactly one vision model is loaded on the server, `--model` can be omitted.

## Built-in tasks

| Task | Output | Description |
|------|--------|-------------|
| `label` (default) | JSON | Taxonomy labels: category, objects, colors, style, setting, lighting, mood, quality issues, confidence |
| `caption` | text | Dense single-paragraph caption, training-data style |
| `tags` | JSON | 5-20 flat keyword tags for search/filtering |
| `ocr` | JSON | Verbatim text extraction with language + legibility |

Each task carries its own system prompt, per-image user prompt, max_tokens,
and -- for JSON tasks -- required keys that are validated per record. Sampling
otherwise follows the model's own defaults on the server; pass
`--temperature`/`--top-p` to override.

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
max_tokens = 512
```

Unknown keys are rejected (catches typos). `--system-prompt`,
`--system-prompt-file`, and `--user-prompt` override any task's prompts.

## Request flags

| Flag | Maps to | Notes |
|------|---------|-------|
| `--think` / `--no-think` | `thinking` | Thinking-capable models (see `models` output); thinking text is stored in its own `thinking` field, never polluting the label |
| `--max-edge N` | client-side resize | Downscale so the longest edge fits N before sending (default 2048, the chat page's cap; 0 sends images as they are). A vision tower's cost grows much faster than the pixel count, so an uncapped camera photo is the slow case. Recorded in each record's `settings` |
| `--temperature/--top-p/--seed/--max-tokens` | same | Explicit values beat task and model defaults (server-side cascade) |

## Run options

```
batch-labeler run IMAGE_DIR [-o results.jsonl] [--limit N] [--no-recursive]
                 [--dry-run] [--retries 2] [--timeout 300] [--server URL]
```

- `--dry-run` scans, reports counts, and prints the fully-resolved settings.
- `--limit N` processes only the first N pending images -- sample a batch,
  inspect, then run the rest.
- Server URL default is `http://localhost:8000`, overridable via the
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
  "model_id": "Qwen3.5-0.8B-MLX-8bit",
  "task": "label",
  "label": {"category": "portrait", "...": "..."},
  "parse_ok": true,
  "raw_output": "...",
  "thinking": "only present when the model produced thinking",
  "usage": {"input_tokens": 1, "output_tokens": 236},
  "performance": {"prompt_tps": 21.9, "generation_tps": 56.3, "peak_memory_gb": 28.5},
  "generation_time_ms": 13323,
  "timestamp": "2026-07-20T12:00:00",
  "settings": {"model": "...", "task": "label", "max_edge": 2048, "max_tokens": 1024}
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
