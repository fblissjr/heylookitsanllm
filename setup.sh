#!/bin/bash
# Automated setup for heylookitsanllm.
# One install path: `uv sync` pulls the full runtime + dev tooling (no extras).
set -e

echo "Hey Look It's an LLM - Setup"
echo "============================"
echo ""

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

case "$OSTYPE" in
    darwin*) echo "Detected macOS (Apple Silicon MLX backend supported)." ;;
    *)       echo "WARNING: non-macOS host -- base install resolves, but MLX inference is Apple-Silicon-only." ;;
esac
echo ""

echo "Installing (full runtime + dev tooling, no extras)..."
uv sync
echo ""
echo "Install complete."
echo "  (To bump the pinned upstreams later: uv run scripts/update_deps.py)"
echo "  (GGUF models also need a llama-server: uv run scripts/update_deps.py llama.cpp)"
echo ""
echo "Next steps:"
echo "  1. Add models:   heylookllm import --hf-cache      (or: --folder <dir>)"
echo "  2. Start server: heylookllm"
echo "  3. Open the UI:  http://localhost:1263/v3"
echo ""
echo "More options: heylookllm --help"
