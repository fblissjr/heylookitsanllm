#!/usr/bin/env bash
set -euo pipefail

echo "Refreshing mlx, mlx-lm, and mlx-vlm..."
uv lock --refresh-package mlx --refresh-package mlx-lm --refresh-package mlx-vlm
uv sync
echo "Done."
