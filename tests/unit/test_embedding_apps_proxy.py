"""apps/embedding-apps/serve.py forwards a short allowlist and nothing else."""
import http.client
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

SERVE = Path(__file__).resolve().parents[2] / "apps" / "embedding-apps" / "serve.py"


def _load():
    spec = importlib.util.spec_from_file_location("embedding_apps_serve", SERVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Upstream(BaseHTTPRequestHandler):
    seen = []

    def _reply(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        _Upstream.seen.append((self.command, self.path, self.headers.get("X-Request-ID"), body))
        payload = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_DELETE = _reply

    def log_message(self, *args):
        pass


@pytest.fixture
def proxy():
    serve = _load()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    serve.Handler.upstream = urlsplit(f"http://127.0.0.1:{upstream.server_address[1]}")
    front = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=front.serve_forever, daemon=True).start()
    _Upstream.seen.clear()
    yield front.server_address[1]
    front.shutdown()
    upstream.shutdown()


def _call(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def test_allowlisted_routes_are_forwarded_with_request_id(proxy):
    assert _call(proxy, "GET", "/v1/models")[0] == 200
    status, _ = _call(proxy, "POST", "/v1/embeddings", b'{"a":1}',
                      {"Content-Type": "application/json", "X-Request-ID": "kit-1"})
    assert status == 200
    assert _call(proxy, "DELETE", "/v1/requests/kit-1")[0] == 200
    assert _Upstream.seen == [
        ("GET", "/v1/models", None, b""),
        ("POST", "/v1/embeddings", "kit-1", b'{"a":1}'),
        ("DELETE", "/v1/requests/kit-1", None, b""),
    ]


@pytest.mark.parametrize("method,path", [
    ("GET", "/v1/admin/config"),
    ("POST", "/v1/messages"),
    ("DELETE", "/v1/conversations/abc"),
    ("POST", "/v1/models"),
    ("DELETE", "/v1/requests/../admin"),
])
def test_everything_else_is_refused_and_never_reaches_upstream(proxy, method, path):
    body = b"{}" if method == "POST" else None
    assert _call(proxy, method, path, body)[0] == 404
    assert _Upstream.seen == []


def test_static_files_are_served(proxy):
    status, data = _call(proxy, "GET", "/media-finder/index.html")
    assert status == 200 and b"Media Finder" in data
