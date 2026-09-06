# tests/contract/test_frontend_mounts.py
#
# The frontend is served at / (v1.79.76). It is the only frontend, so it no
# longer carries a version in its path -- /v3 went the way /v2 did.
#
# The contract that replaced the SPA fallback: an unknown path 404s. The app
# routes on the hash, so the server never sees a deep route and has nothing to
# fall back FOR; serving index.html for anything unmatched would have destroyed
# 404 for the whole API surface behind it.


def test_v2_mount_is_gone(client):
    # The deletion is the contract: a resurrected /v2 would mean the retired
    # app grew back a serving path. Still expressible as a status code only
    # because there is no catch-all fallback -- see test_unknown_paths_404.
    assert client.get("/v2").status_code == 404


def test_v3_mount_is_gone(client):
    # Same contract, one generation later. The frontend answers at / now.
    assert client.get("/v3").status_code == 404
    assert client.get("/v3/js/app.js").status_code == 404


def test_index_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert '<base href="/">' in r.text


def test_asset_served(client):
    r = client.get("/js/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_unknown_paths_404(client):
    """No SPA fallback: this is what keeps 404 meaningful for the API.

    With a fallback, a typo'd API path answers 200 with a web page, which is
    the silent failure the whole design avoids. `/some/deep/route` stands in
    for a hash-routed URL the server never actually sees.
    """
    for path in ("/some/deep/route", "/v1/mesages", "/nope.js"):
        assert client.get(path).status_code == 404, path


def test_api_and_docs_still_win_over_the_catch_all(client):
    # The catch-all is registered after every router, so ordering alone
    # protects these. Pinned because the protection is positional, and a
    # route added below it would be shadowed with no error.
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/v1/models").status_code == 200


def test_assets_are_revalidated(client):
    # No build step and unhashed URLs, so a cached module is only ever
    # invalidated by revalidation. Drop this header and browsers fall back to
    # HEURISTIC freshness (~10% of the file's age), which serves rarely-edited
    # modules stale for hours while their frequently-edited callers refetch --
    # a mixed-version frontend whose symptom is "X is not a function".
    for path in ("/", "/index.html", "/js/preset-bar.js"):
        r = client.get(path)
        assert r.headers.get("cache-control") == "no-cache", path


def test_path_traversal_is_refused(client):
    """%2e%2e decodes to ".." -- and this now matters more than it used to.

    Under apps/heylook-frontend-v3/, `../pyproject.toml` resolved to
    apps/pyproject.toml, which does not exist, so is_file() refused it and the
    is_relative_to guard was never the thing doing the work. From frontend/ at
    the repo root it resolves to the REAL pyproject.toml, so the guard is now
    the only thing between this request and that file.
    """
    r = client.get("/%2e%2e/pyproject.toml")
    assert r.status_code == 404
    assert "[project]" not in r.text


def test_revalidation_costs_no_body(client):
    # The other half of no-cache. Revalidating is only cheap if an unchanged
    # asset answers 304; starlette's FileResponse sets an etag but has no
    # conditional branch of its own (only StaticFiles does), so every
    # revalidation used to be answered with the whole file -- 427KB per load,
    # which is a half-megabyte transfer every time a phone reloads a tab iOS
    # evicted. Together with the test above: always revalidate, never resend.
    first = client.get("/js/app.js")
    etag = first.headers.get("etag")
    assert etag, "no etag to revalidate against"
    again = client.get("/js/app.js", headers={"if-none-match": etag})
    assert again.status_code == 304
    assert not again.content
    # A stale validator must still get the body.
    stale = client.get("/js/app.js", headers={"if-none-match": '"not-the-etag"'})
    assert stale.status_code == 200
    assert stale.content


def test_text_assets_are_compressed_and_sse_is_not_touched(client):
    # Compression lives in the frontend handler, NOT in GZipMiddleware: that
    # middleware wraps every response including the generate endpoint's SSE,
    # where buffering to a minimum size would sit on the first token. So the
    # win is scoped to static assets and the streaming path cannot regress.
    gz = client.get("/js/app.js", headers={"accept-encoding": "gzip"})
    assert gz.headers.get("content-encoding") == "gzip"
    assert "accept-encoding" in gz.headers.get("vary", "").lower()
    plain = client.get("/js/app.js", headers={"accept-encoding": "identity"})
    assert plain.headers.get("content-encoding") is None
    # Both spellings must deliver the same bytes.
    assert gz.content == plain.content
    from starlette.middleware.gzip import GZipMiddleware
    import heylook_llm.api as api
    assert not any(m.cls is GZipMiddleware for m in api.app.user_middleware), \
        "GZipMiddleware would wrap the SSE generate endpoint"


def test_the_gzip_cache_survives_a_multi_asset_page_load(client):
    """A page load fetches ~20 assets; the cache must still hold the first.

    It used to `clear()` on every miss, so each asset evicted the one before
    it and the cache held exactly ONE entry -- nothing was ever a hit, and the
    level-6 compression it exists to keep off the event loop ran on every
    load, on the same loop delivering SSE tokens. Eviction is per-PATH now
    (older generations of the same file), which is what "one generation of the
    tree at a time" was reaching for.

    Counted at `gzip.compress`, because a hit and a miss are byte-identical on
    the wire -- there is nothing in the response to assert on.
    """
    import gzip as _real_gzip
    from unittest.mock import patch
    import heylook_llm.api as api

    hdrs = {"accept-encoding": "gzip"}
    assets = ["/js/app.js", "/js/api.js", "/js/utils.js"]
    for a in assets:                       # warm, whatever the cache held before
        assert client.get(a, headers=hdrs).status_code == 200

    calls = []

    def counting_compress(data, compresslevel=6):
        calls.append(compresslevel)
        return _real_gzip.compress(data, compresslevel=compresslevel)

    with patch.object(api._gzip, "compress", counting_compress):
        for a in assets:
            r = client.get(a, headers=hdrs)
            assert r.headers.get("content-encoding") == "gzip"
    assert calls == [], (
        f"re-compressed {len(calls)} already-cached asset(s) -- a page load "
        "evicts its own entries")
