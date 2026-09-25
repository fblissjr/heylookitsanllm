# src/heylook_llm/host_check.py
"""Refuse a request whose Host header names a site this server is not (DNS
rebinding).

A web page on another site can point ITS OWN hostname at this machine's LAN
address. The browser then talks to heylook while believing it is still that
site, so the same-origin policy lets the page read every answer -- from a
browser already inside the LAN, which "LAN-only" does not cover. What gives it
away is the Host header: it carries the other site's name.

So a request is answered only when its Host is:
- an IP address (rebinding always arrives with a name, never an address),
- ``localhost``,
- one of this machine's own names (its hostname, short name and ``.local``
  name), or
- a name listed in heylook.toml's top-level ``allowed_hosts`` (a LAN DNS name
  or a VPN name the machine is reached by). ``["*"]`` turns the check off.

The list lives in heylook.toml only: no route writes it, and it is re-read
with the rest of the file on a config reload. A request with no Host header is
not a browser's, so it passes.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket

logger = logging.getLogger(__name__)


def host_only(value: str) -> str:
    """The name or address in a Host header: port, brackets and a trailing
    dot removed, lowercased."""
    v = value.strip().lower()
    if v.startswith("["):
        end = v.find("]")
        return v[1:end] if end > 0 else v
    if v.count(":") == 1:
        v = v.rsplit(":", 1)[0]
    return v.rstrip(".")


def own_names() -> frozenset[str]:
    """This machine's names, as a browser on the LAN might spell them."""
    names = {"localhost"}
    for raw in (socket.gethostname(), socket.getfqdn()):
        name = (raw or "").lower().rstrip(".")
        if not name:
            continue
        short = name.split(".")[0]
        names.update({name, short, f"{short}.local"})
    return frozenset(names)


def is_allowed(host_header: str, own: frozenset[str], extra: frozenset[str]) -> bool:
    if "*" in extra:
        return True
    host = host_only(host_header)
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    return host in own or host in extra


def _configured(scope) -> frozenset[str]:
    """heylook.toml's allowed_hosts, read through the router's config (so a
    config reload takes effect); empty until the router exists."""
    router = getattr(getattr(scope.get("app"), "state", None), "router_instance", None)
    hosts = getattr(getattr(router, "app_config", None), "allowed_hosts", None) or ()
    return frozenset(host_only(h) if h != "*" else h for h in hosts)


class HostCheckMiddleware:
    """Pure ASGI, so it also covers the static frontend and streaming routes."""

    def __init__(self, app):
        self.app = app
        self.own = own_names()
        self._reported: set[str] = set()

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        host = next((v.decode("latin-1") for k, v in scope.get("headers") or ()
                     if k == b"host"), None)
        if host is None or is_allowed(host, self.own, _configured(scope)):
            return await self.app(scope, receive, send)

        name = host_only(host)
        if name not in self._reported:
            self._reported.add(name)
            logger.warning(
                f"Refused a request for Host '{name}': not an IP address, not one of "
                f"this machine's names, and not in heylook.toml's allowed_hosts. If "
                f"you reach this server by that name, add it there.")
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        body = json.dumps({"detail": (
            f"This server does not answer to the host name '{name}'. If you reach "
            f"it by that name, add it to allowed_hosts at the top of heylook.toml "
            f"(DNS-rebinding guard)."
        )}).encode()
        await send({"type": "http.response.start", "status": 403,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
