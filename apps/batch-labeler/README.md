# batch-labeler

Standalone client for batch VLM image labeling. Sends images to any OpenAI-compatible `/v1/chat/completions` endpoint and stores structured results in JSONL.

## Prerequisites

- Python 3.12+ managed by uv
- A running server with a VLM model loaded (heylookitsanllm backend, or any OpenAI-compatible server)
- A directory of images to label

## Install

```bash
cd apps/batch-labeler
uv sync
```

## End-to-End Tutorial

### 1. Start the backend with a VLM model

In a separate terminal, start heylookitsanllm and load a VLM:

```bash
uv run uvicorn heylook_llm.api:app --host 0.0.0.0 --port 8000
```

Then load a model via the API or frontend. Any VLM works -- Qwen2.5-VL, Gemma-3, etc.

### 2. Prepare your images

Put images in a directory. The labeler scans recursively by default and handles jpg, png, webp, and other PIL-supported formats.

```
my-images/
  photo1.jpg
  photo2.png
  subdir/
    photo3.webp
```

### 3. Write a system prompt

Create a text file with labeling instructions:

```bash
cat > instructions.txt << 'EOF'
You are an image classifier. For each image, respond with a JSON object:
{"category": "...", "description": "...", "confidence": 0.0-1.0}

Categories: landscape, portrait, object, text, diagram, other
EOF
```

Or pass the prompt inline with `--system-prompt "..."`.

### 4. Run the labeler

```bash
cd apps/batch-labeler
uv run batch-labeler \
  --image-dir /path/to/my-images \
  --model "mlx-community/Qwen2.5-VL-7B-Instruct-8bit" \
  --system-prompt-file instructions.txt \
  --output results.jsonl \
  --server http://localhost:8000
```

Progress is printed to stderr. Each completed image appends a line to the output JSONL.

### 5. Review results

Each line in `results.jsonl` is a JSON object:

```json
{
  "file_path": "/path/to/my-images/photo1.jpg",
  "file_hash": "abc123...",
  "file_name": "photo1.jpg",
  "model_id": "mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
  "label": {"category": "landscape", "description": "Mountain scene", "confidence": 0.95},
  "raw_output": "...",
  "generation_time_ms": 1234,
  "timestamp": "2026-03-10T12:00:00"
}
```

Parse with standard tools:

```bash
# Count by category
cat results.jsonl | uv run python -c "
import sys, orjson
for line in sys.stdin:
    obj = orjson.loads(line)
    print(obj.get('label', {}).get('category', 'unknown'))
" | sort | uniq -c | sort -rn
```

### 6. Resume interrupted runs

Re-running the same command skips already-processed images (matched by `file_hash` in the output JSONL). Safe to ctrl-c and resume.

## Options

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

## Tests

```bash
cd apps/batch-labeler
uv sync --dev
uv run pytest tests/ -v
```
