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
echo "  (To update dependencies later: uv lock --upgrade && uv sync)"
echo "  (GGUF models also need a llama-server binary:"
echo "   uv run scripts/build_llama.py -- or brew install llama.cpp and point"
echo "   \$HEYLOOK_LLAMA_SERVER / models.toml server_binary at it)"
echo ""
echo "Next steps:"
echo "  1. Add models:   heylookllm import --hf-cache      (or: --folder <dir>)"
echo "  2. Start server: heylookllm"
echo "  3. Open the UI:  http://localhost:1263/v3"
echo ""
echo "More options: heylookllm --help"
