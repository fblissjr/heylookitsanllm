# tests/contract/test_frontend_mounts.py
#
# Static frontend mounts: /v2 (legacy vanilla-JS frontend) and /v3 (rewrite).
# Both serve index.html for SPA routes and real files for assets, with
# path-traversal protection falling back to index.html.


def test_v2_index_served(client):
    r = client.get("/v2")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


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


def test_v3_path_traversal_falls_back_to_index(client):
    # %2e%2e decodes to ".." in the path param; resolution lands outside the
    # frontend dir, so the handler must serve index.html, not the target file.
    r = client.get("/v3/%2e%2e/pyproject.toml")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "[project]" not in r.text
