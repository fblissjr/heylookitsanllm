# src/heylook_llm/frontend_static.py
"""Static file serving for the frontend, mounted at `/`.

Extracted from api.py in v1.79.77. It was sanctioned there -- CLAUDE.md names
the static server as legitimate app-assembly content -- but it was ~100 of that
file's ~425 lines and the only substantial logic in a module whose job is
wiring. It also carries invariants bought with real bugs, which deserve
somewhere to be read.

NO SPA FALLBACK, and that is load-bearing rather than an omission. The app
routes on the HASH (`#/chat`), so the server only ever sees `/` and real asset
paths -- there are no deep server-side routes to fall back FOR. Serving
index.html for anything unmatched would instead destroy 404 for the whole API
behind it: a typo'd `/v1/mesages` would answer 200 with a web page. So an
unknown path 404s, which also gives `/v3` and `/v2` their "this mount is gone"
answer for free, with no special-case route. If the frontend ever moves to the
History API, this decision has to be revisited along with it.

The other invariants, in one place:

* every asset REVALIDATES (`no-cache`). With no build step and no content
  hashes in the URLs, a cached module can only be invalidated by revalidation;
  without the header a browser applies HEURISTIC freshness (~10% of the file's
  age) and skips the request, silently mixing module versions -- frequently
  edited files refetch while rarely edited ones serve stale for hours, and the
  symptom is "X is not a function".
* revalidating costs a header round trip, not a body. Starlette's FileResponse
  sets an etag but has no conditional branch (only StaticFiles does), so every
  revalidation used to be answered with the whole file -- 427KB across 22
  assets per load. Free on localhost, right up until the client is a phone:
  iOS discards backgrounded tabs and reloads the document, making it a
  half-megabyte transfer per wake-from-eviction.
* the etag differs per content-coding and `vary` is set on EVERY exit,
  including the 304. One etag across both representations lets a shared cache
  answer an identity request with a gzip body.
* the gzip cache evicts per PATH, not wholesale. Clearing the dict on each miss
  meant a ~20-asset page load evicted its own entries, the cache held exactly
  one, and level-6 compression ran on every load -- on the same event loop that
  delivers SSE tokens.
"""

from __future__ import annotations

import gzip as _gzip
import hashlib as _hashlib
import logging
import pathlib as _pathlib
from email.utils import formatdate as _formatdate
from mimetypes import guess_type

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import FileResponse, Response

FRONTEND_DIR = (_pathlib.Path(__file__).resolve().parent.parent.parent / "frontend").resolve()

_NO_CACHE = {"Cache-Control": "no-cache"}
# Text assets only, and only above the size where a round trip dominates.
_GZIP_TYPES = (".js", ".mjs", ".css", ".html", ".json", ".svg")
_GZIP_MIN = 1024

# Compressed bytes keyed on the same (mtime, size) the etag derives from. The
# tree is static between edits, so this is a hit after the first request.
# Bounded by the number of gzip-eligible files (one entry each, older
# generations of the same path evicted on write), not by traffic.
_gzip_cache: dict = {}


def _asset_etag(stat_result) -> str:
    """Byte-identical to starlette's FileResponse etag (responses.py)."""
    base = f"{stat_result.st_mtime}-{stat_result.st_size}"
    return f'"{_hashlib.md5(base.encode(), usedforsecurity=False).hexdigest()}"'


def _file_response(path, request: Request):
    stat_result = path.stat()
    base_etag = _asset_etag(stat_result)
    wants_gzip = (path.suffix in _GZIP_TYPES
                  and stat_result.st_size >= _GZIP_MIN
                  and "gzip" in request.headers.get("accept-encoding", ""))
    # RFC 9110: distinct entity-tags per content-coding.
    etag = f'{base_etag[:-1]}-gzip"' if wants_gzip else base_etag
    # Vary on EVERY exit, not just the compressed one -- a 304 or an identity
    # 200 stored without it is the same cache-poisoning bug.
    headers = {**_NO_CACHE, "etag": etag, "vary": "accept-encoding"}

    if etag in [t.strip().removeprefix("W/")
                for t in request.headers.get("if-none-match", "").split(",")]:
        return Response(status_code=304, headers=headers)

    if wants_gzip:
        key = (str(path), stat_result.st_mtime, stat_result.st_size)
        body = _gzip_cache.get(key)
        if body is None:
            body = _gzip.compress(path.read_bytes(), compresslevel=6)
            # Evict older generations of THIS FILE, not the whole tree -- see
            # the module docstring for what clearing it wholesale cost.
            for stale in [k for k in _gzip_cache if k[0] == key[0]]:
                del _gzip_cache[stale]
            _gzip_cache[key] = body
        return Response(
            content=body,
            media_type=guess_type(path.name)[0] or "application/octet-stream",
            headers={
                **headers,
                "last-modified": _formatdate(stat_result.st_mtime, usegmt=True),
                "content-encoding": "gzip",
            },
        )
    return FileResponse(path, stat_result=stat_result, headers=headers)


def mount_frontend(app: FastAPI) -> None:
    """Register `/` and the asset catch-all.

    MUST be called after every router is included: the asset route is a
    catch-all, and registration order is the only thing keeping `/v1`, `/docs`
    and `/openapi.json` reachable.
    """
    if not FRONTEND_DIR.is_dir():
        # Loud, because the alternative is silence: the guard used to skip the
        # whole block, so a wrong path meant no frontend routes registered, no
        # error, and a server that starts clean while the UI 404s.
        logging.warning(
            "Frontend directory not found at %s -- no UI will be served. Every "
            "frontend route is skipped and / answers 404.", FRONTEND_DIR,
        )
        return

    # Plain `def`, not `async def`: these stat, read and gzip files, all of
    # which block. Starlette runs a sync handler in a threadpool, which is the
    # same reason the two admin routes that read model directories are sync --
    # the event loop here is the one delivering SSE tokens.
    @app.get("/", include_in_schema=False)
    def serve_frontend_index(request: Request):
        return _file_response(FRONTEND_DIR / "index.html", request)

    @app.get("/{rest:path}", include_in_schema=False)
    def serve_frontend_asset(request: Request, rest: str):
        """A real file, or 404."""
        resolved = (FRONTEND_DIR / rest).resolve()
        # The guard is load-bearing NOW in a way it was not under apps/:
        # `../pyproject.toml` used to resolve to a path that did not exist, so
        # is_file() was what actually refused it. From frontend/ at the repo
        # root it resolves to the real file.
        if resolved.is_relative_to(FRONTEND_DIR) and resolved.is_file():
            return _file_response(resolved, request)
        raise HTTPException(status_code=404, detail="Not found")
