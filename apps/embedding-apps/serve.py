#!/usr/bin/env python3
"""Serve the embedding apps and proxy a short allowlist of /v1 routes to heylook.

The server sends no CORS headers on purpose, so a page on another origin cannot
call it. This launcher serves the static files and forwards the few routes the
apps need, so the browser sees one origin. Stdlib only; the server is untouched.

    python3 serve.py [--server http://127.0.0.1:8000] [--port 8765]
"""
import argparse
import http.client
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
# method -> allowed path patterns. Nothing else is forwarded.
ALLOWED = {
    "GET": [re.compile(r"^/v1/models$")],
    "POST": [re.compile(r"^/v1/embeddings$")],
    "DELETE": [re.compile(r"^/v1/requests/[A-Za-z0-9_.:-]{1,128}$")],
}
FORWARD_HEADERS = ("Content-Type", "X-Request-ID")
MAX_BODY = 96 * 1024 * 1024  # 64 MB of media plus base64 overhead


class Handler(SimpleHTTPRequestHandler):
    upstream = None  # urlsplit result, set in main()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def _allowed(self, method):
        path = urlsplit(self.path).path
        return any(p.match(path) for p in ALLOWED.get(method, []))

    def _proxy(self, method):
        if not self._allowed(method):
            self.send_error(404, "Not proxied")
            return
        body = None
        if method == "POST":
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                self.send_error(413, "Request body too large")
                return
            body = self.rfile.read(length)
        headers = {h: self.headers[h] for h in FORWARD_HEADERS if self.headers.get(h)}
        conn = http.client.HTTPConnection(self.upstream.hostname, self.upstream.port or 80, timeout=3600)
        try:
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
        except OSError as exc:
            self.send_error(502, f"heylook server unreachable: {exc}")
            return
        finally:
            conn.close()
        self.send_response(resp.status)
        for name in ("Content-Type", "X-Request-ID"):
            if resp.getheader(name):
                self.send_header(name, resp.getheader(name))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/v1/"):
            self._proxy("GET")
        else:
            super().do_GET()

    def do_POST(self):
        self._proxy("POST")

    def do_DELETE(self):
        self._proxy("DELETE")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default="http://127.0.0.1:8000", help="heylook base URL")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    Handler.upstream = urlsplit(args.server)
    if Handler.upstream.scheme != "http" or not Handler.upstream.hostname:
        ap.error("--server must be an http:// URL")
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Embedding apps: http://127.0.0.1:{args.port}/  ->  {args.server}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
