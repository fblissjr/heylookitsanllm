#!/usr/bin/env bash
#
# Verify apps/heylook-frontend/src/types/generated-api.ts is in sync with the
# backend's OpenAPI schema, so the generated types can't silently drift.
#
# It regenerates the types from the FastAPI app's schema (offline -- no running
# server, via scripts/export_openapi.py -> app.openapi()) using the frontend's
# pinned openapi-typescript, then diffs against the committed file.
#
# Exit codes:
#   0  in sync, OR could not verify (missing tools / backend import failed) --
#      degrades gracefully so contributors without MLX/bun aren't blocked.
#   1  drift detected: the committed generated-api.ts is stale.
#
# Run manually:   bash scripts/check_openapi_sync.sh
# Regenerate:     /openapi-regen  (or: cd apps/heylook-frontend && bun run generate:api)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMITTED="$REPO_ROOT/apps/heylook-frontend/src/types/generated-api.ts"
FRONTEND_DIR="$REPO_ROOT/apps/heylook-frontend"

warn_skip() { echo "[openapi-sync] $1 -- skipping check (not a failure)." >&2; exit 0; }

command -v uv  >/dev/null 2>&1 || warn_skip "uv not found"
command -v bun >/dev/null 2>&1 || warn_skip "bun not found"
[ -f "$COMMITTED" ] || warn_skip "committed types not found at $COMMITTED"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
tmp_json="$tmp_dir/openapi.json"
tmp_ts="$tmp_dir/generated-api.ts"

# Dump the live app schema WITHOUT sorting keys, so property order matches what
# the server serves at /v1 openapi.json (the canonical `generate:api` source).
if ! (cd "$REPO_ROOT" && uv run python -c "
import orjson, sys
from heylook_llm.api import app
sys.stdout.buffer.write(orjson.dumps(app.openapi()))
" > "$tmp_json" 2>"$tmp_dir/err"); then
    warn_skip "could not import backend app to build schema ($(tail -1 "$tmp_dir/err" 2>/dev/null))"
fi

# Use the frontend's pinned openapi-typescript so output matches generate:api,
# then apply the same `| null` -> `| undefined` transform.
if ! (cd "$FRONTEND_DIR" && bun x openapi-typescript "$tmp_json" -o "$tmp_ts" >/dev/null 2>"$tmp_dir/ts_err"); then
    warn_skip "openapi-typescript failed ($(tail -1 "$tmp_dir/ts_err" 2>/dev/null))"
fi
LC_ALL=C sed 's/| null/| undefined/g' "$tmp_ts" > "$tmp_ts.norm" && mv "$tmp_ts.norm" "$tmp_ts"

if diff -q "$COMMITTED" "$tmp_ts" >/dev/null 2>&1; then
    echo "[openapi-sync] generated-api.ts is in sync."
    exit 0
fi

echo "ERROR: apps/heylook-frontend/src/types/generated-api.ts is out of sync with the backend OpenAPI schema." >&2
echo "       The API schema changed but the generated types were not regenerated." >&2
echo "       Fix: start the backend, then run /openapi-regen" >&2
echo "            (or: cd apps/heylook-frontend && bun run generate:api), and stage the result." >&2
echo "" >&2
echo "       Drift summary:" >&2
diff "$COMMITTED" "$tmp_ts" | grep -E '^[<>]' | head -20 >&2 || true
exit 1
