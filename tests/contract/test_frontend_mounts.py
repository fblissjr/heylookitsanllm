# tests/contract/test_frontend_mounts.py
#
# Static frontend mount: /v3 (the only frontend since v1.77.0 -- v2 deleted
# at cutover). Serves index.html for SPA routes and real files for assets,
# with path-traversal protection falling back to index.html.


def test_v2_mount_is_gone(client):
    # The deletion is the contract now: a resurrected /v2 mount would mean
    # the retired app grew back a serving path.
    assert client.get("/v2").status_code == 404


def test_v3_index_served(client):
    r = client.get("/v3")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert '<base href="/v3/">' in r.text


def test_v3_asset_served(client):
    r = client.get("/v3/js/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_v3_spa_fallback(client):
    r = client.get("/v3/some/deep/route")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_v3_assets_are_revalidated(client):
    # v3 is no-build with unhashed URLs, so a cached module is only ever
    # invalidated by revalidation. Drop this header and browsers fall back to
    # HEURISTIC freshness (~10% of the file's age), which serves rarely-edited
    # modules stale for hours while their frequently-edited callers refetch --
    # a mixed-version frontend whose symptom is "X is not a function".
    for path in ("/v3", "/v3/index.html", "/v3/js/preset-bar.js", "/v3/some/deep/route"):
        r = client.get(path)
        assert r.headers.get("cache-control") == "no-cache", path


def test_v3_path_traversal_falls_back_to_index(client):
    # %2e%2e decodes to ".." in the path param; resolution lands outside the
    # frontend dir, so the handler must serve index.html, not the target file.
    r = client.get("/v3/%2e%2e/pyproject.toml")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "[project]" not in r.text


def test_v3_revalidation_costs_no_body(client):
    # The other half of no-cache. Revalidating is only cheap if an unchanged
    # asset answers 304; starlette's FileResponse sets an etag but has no
    # conditional branch of its own (only StaticFiles does), so every
    # revalidation used to be answered with the whole file -- 427KB per load,
    # which is a half-megabyte transfer every time a phone reloads a tab iOS
    # evicted. Together with the test above: always revalidate, never resend.
    first = client.get("/v3/js/app.js")
    etag = first.headers.get("etag")
    assert etag, "no etag to revalidate against"
    again = client.get("/v3/js/app.js", headers={"if-none-match": etag})
    assert again.status_code == 304
    assert not again.content
    # A stale validator must still get the body.
    stale = client.get("/v3/js/app.js", headers={"if-none-match": '"not-the-etag"'})
    assert stale.status_code == 200
    assert stale.content


def test_v3_text_assets_are_compressed_and_sse_is_not_touched(client):
    # Compression lives in the v3 handler, NOT in GZipMiddleware: that
    # middleware wraps every response including the generate endpoint's SSE,
    # where buffering to a minimum size would sit on the first token. So the
    # win is scoped to static assets and the streaming path cannot regress.
    gz = client.get("/v3/js/app.js", headers={"accept-encoding": "gzip"})
    assert gz.headers.get("content-encoding") == "gzip"
    assert "accept-encoding" in gz.headers.get("vary", "").lower()
    plain = client.get("/v3/js/app.js", headers={"accept-encoding": "identity"})
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
    assets = ["/v3/js/app.js", "/v3/js/api.js", "/v3/js/utils.js"]
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
