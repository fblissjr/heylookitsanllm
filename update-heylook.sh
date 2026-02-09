#!/bin/bash
# update-heylook.sh - Update heylookitsanllm and dependencies
set -e

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Updating heylookitsanllm and dependencies..."

# Pull latest from git sources
echo "Upgrading mlx-lm, mlx-vlm, and parakeet-mlx to latest from git..."
uv lock --upgrade-package mlx-lm --upgrade-package mlx-vlm --upgrade-package parakeet-mlx

# Detect installed extras
EXTRAS=""

if uv pip list 2>/dev/null | grep -q "^mlx-vlm "; then
    echo "Found MLX backend..."
    EXTRAS="$EXTRAS --extra mlx"
fi

if uv pip list 2>/dev/null | grep -q "^parakeet-mlx "; then
    echo "Found STT packages..."
    EXTRAS="$EXTRAS --extra stt"
fi

if uv pip list 2>/dev/null | grep -q "orjson"; then
    echo "Found performance packages..."
    EXTRAS="$EXTRAS --extra performance"
fi

if uv pip list 2>/dev/null | grep -q "duckdb"; then
    echo "Found analytics packages..."
    EXTRAS="$EXTRAS --extra analytics"
fi

if uv pip list 2>/dev/null | grep -q "py-spy"; then
    echo "Found profiling packages..."
    EXTRAS="$EXTRAS --extra profile"
fi

# Sync
if [ -n "$EXTRAS" ]; then
    echo "Syncing with extras:$EXTRAS"
    uv sync $EXTRAS
else
    echo "Syncing base packages..."
    uv sync
fi

echo "Done! Verify with: heylookllm --version"
