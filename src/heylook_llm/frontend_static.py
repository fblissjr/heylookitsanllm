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
            #
            # Snapshot the keys and pop defensively: these handlers are sync,
            # so they run in anyio's THREADPOOL, and two threads missing on a
            # cold cache raced here. Iterating the live dict raised
            # "dictionary changed size during iteration" (reproduced under real
            # uvicorn with 8 concurrent clients on a cold cache -> 500 on a
            # static asset), and two threads evicting the same generation would
            # double-delete. Neither could happen while this was `async def` on
            # one loop; the sync change introduced both.
            for stale in list(_gzip_cache):
                if stale[0] == key[0]:
                    _gzip_cache.pop(stale, None)
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


def _serve(path, request: Request):
    """One asset, or 404. Never raises."""
    try:
        resolved = path.resolve()
        # The guard is load-bearing NOW in a way it was not under apps/:
        # `../pyproject.toml` used to resolve to a path that did not exist, so
        # is_file() was what actually refused it. From frontend/ at the repo
        # root it resolves to the real file. `.resolve()` follows symlinks
        # BEFORE the check, which is the order that makes a symlink escape fail.
        if resolved.is_relative_to(FRONTEND_DIR) and resolved.is_file():
            return _file_response(resolved, request)
    except (ValueError, OSError):
        # A NUL byte in the path makes resolve() raise ValueError before
        # is_file() (which swallows it) ever runs -- `GET /%00` was answering
        # 500 with a traceback per request. A bad path is a 404 like any other.
        pass
    raise HTTPException(status_code=404, detail="Not found")


def mount_frontend(app: FastAPI) -> None:
    """Register the frontend's routes at `/`.

    NO CATCH-ALL, and that is deliberate. `@app.get("/{rest:path}")` matches
    every path, which means starlette ALWAYS finds a partial match and so:
    `redirect_slashes` never fires (`POST /v1/messages/` went from 307->200 to
    405 Method Not Allowed, a real break for any client that concatenates URLs)
    and unknown paths answer 405 instead of 404 on every non-GET method. Both
    were measured, not theorised. Serving the tree's actual shape -- `/` plus
    the two asset directories -- costs nothing here, because the app routes on
    the HASH so the server only ever sees those, and it keeps the whole API's
    404 and redirect behaviour untouched.

    It also means `frontend/DESIGN.md` is no longer served at the web root,
    which a catch-all exposed.

    A new TOP-LEVEL asset (an image, a manifest) needs a route added here. That
    is the price of not having a catch-all, and it is worth it.
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
    #
    # HEAD as well as GET: `/` is the user-facing entry point now, and uptime
    # monitors probe with HEAD. FastAPI's APIRoute does not add it implicitly
    # the way starlette's Route does, so it answered 405.
    # `/index.html` as well as `/`: the catch-all served both, and dropping a
    # URL that used to work is a change nobody asked for.
    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    @app.api_route("/index.html", methods=["GET", "HEAD"], include_in_schema=False)
    def serve_frontend_index(request: Request):
        return _serve(FRONTEND_DIR / "index.html", request)

    @app.api_route("/js/{rest:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def serve_frontend_js(request: Request, rest: str):
        return _serve(FRONTEND_DIR / "js" / rest, request)

    @app.api_route("/css/{rest:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def serve_frontend_css(request: Request, rest: str):
        return _serve(FRONTEND_DIR / "css" / rest, request)
