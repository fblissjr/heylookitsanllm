# batch-labeler

Standalone client for batch VLM image labeling. Sends images to any OpenAI-compatible `/v1/chat/completions` endpoint and stores structured results in JSONL.

## Install

```bash
cd apps/batch-labeler
uv pip install -e .
```

## Usage

```bash
batch-labeler \
  --image-dir /path/to/images \
  --model "mlx-community/Qwen2.5-VL-7B-Instruct-8bit" \
  --system-prompt-file instructions.txt \
  --output results.jsonl \
  --server http://localhost:8000
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--image-dir` | required | Directory containing images |
| `--model` | required | VLM model ID |
| `--system-prompt` | -- | Inline system prompt |
| `--system-prompt-file` | -- | Read prompt from file |
| `--output` | `results.jsonl` | Output JSONL path |
| `--server` | `http://localhost:8000` | Server base URL |
| `--max-tokens` | 1024 | Max tokens per response |
| `--temperature` | 0.1 | Sampling temperature |
| `--recursive` / `--no-recursive` | recursive | Scan subdirectories |
| `--dry-run` | false | Scan only, don't process |
| `--timeout` | 120 | Per-request timeout (seconds) |

### Resume

Re-running the same command skips already-processed images (matched by file hash in the output JSONL).

### Output Format

Each line is a JSON object:

```json
{"file_path": "/path/to/image.jpg", "file_hash": "abc123...", "file_name": "image.jpg", "model_id": "model-id", "label": {...}, "raw_output": "...", "generation_time_ms": 1234, "timestamp": "2026-03-10T12:00:00"}
```

## Tests

```bash
cd apps/batch-labeler
uv run pytest tests/ -v
```
