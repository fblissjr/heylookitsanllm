# LAN setup: HTTPS via reverse proxy + optional admin token

Last updated: 2026-04-18

This guide covers running `heylookitsanllm` on a home LAN where more than
one host needs to reach it (e.g. a Ubuntu+GPU box calling from ComfyUI, a
phone on the same network hitting the web frontend). It favors a small,
boring HTTPS terminator (Caddy) in front of a loopback-bound inference
server over uvicorn-native TLS, and adds an opt-in header-based admin
token for defense-in-depth on mutating endpoints.

The defaults are chosen for a **single-user, trusted home LAN**. Multi-
tenant authentication is explicitly out of scope.

## Threat model recap

- The server runs on one machine (e.g. M2 Ultra) and is called by other
  hosts on the same LAN (e.g. an Ubuntu box running ComfyUI).
- The trust boundary is the home router. Anything inside the LAN is
  assumed friendly.
- We still want: (1) HTTPS so secrets in headers / bodies don't travel as
  plaintext, and (2) a cheap gate on admin endpoints so a mis-scoped
  guest WiFi or a roommate's laptop can't `POST /v1/data/clear`.

## Recommended topology

```
[client on LAN] --HTTPS--> [Caddy on :443] --HTTP--> [heylookllm on 127.0.0.1:8080]
```

- `heylookllm` binds to loopback only (`--host 127.0.0.1`, the default).
- Caddy terminates TLS on `:443` using its own internal CA.
- Admin endpoints optionally require `X-Heylook-Admin-Token` when the
  env var `HEYLOOK_ADMIN_TOKEN` is set.

## Caddy: quick start

### 1. Install

```sh
brew install caddy   # macOS
# or: sudo apt install caddy   # Ubuntu/Debian
```

### 2. Caddyfile

Put this at `/etc/caddy/Caddyfile` (Linux) or `~/Caddyfile` (macOS):

```Caddyfile
heylook.local {
    tls internal
    reverse_proxy 127.0.0.1:8080
}
```

`tls internal` tells Caddy to generate and manage a local CA. No
`mkcert`, no Let's Encrypt, no external DNS.

### 3. Trust the local CA

On the server machine:

```sh
caddy run --config ~/Caddyfile   # or `sudo systemctl enable --now caddy`
caddy trust
```

Copy the root CA to any other LAN host that will call the server. On
macOS it lives under
`~/Library/Application Support/Caddy/pki/authorities/local/root.crt`.
On Linux, `/var/lib/caddy/.local/share/caddy/pki/authorities/local/root.crt`.

On the Ubuntu client:

```sh
sudo cp root.crt /usr/local/share/ca-certificates/caddy-local.crt
sudo update-ca-certificates
```

### 4. Hosts entry

On each client:

```
<server-LAN-IP>   heylook.local
```

On macOS / Linux edit `/etc/hosts`. On Windows edit
`C:\Windows\System32\drivers\etc\hosts`. Alternatively, publish the name
via Bonjour / Avahi / router-level DNS; any of those works.

### 5. Verify

From the server:

```sh
curl https://heylook.local/v1/models
```

From a LAN client (after trusting the CA):

```sh
curl https://heylook.local/v1/models
```

Both should return a JSON body with no certificate warnings.

## Alternative: nginx

Works the same way, with a slightly longer config and external cert
management (mkcert is the simplest source of a LAN cert). If you're
already running nginx for something else, a minimal site file looks
like:

```nginx
server {
    listen 443 ssl http2;
    server_name heylook.local;

    ssl_certificate     /etc/nginx/certs/heylook.local.pem;
    ssl_certificate_key /etc/nginx/certs/heylook.local-key.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        # Don't buffer SSE streams.
        proxy_buffering off;
    }
}
```

Generate the certs with `mkcert -install && mkcert heylook.local`.

## Admin-token opt-in

Admin endpoints (`/v1/admin/*`, `/v1/data/clear`, `/v1/cache/clear`)
support an opt-in token check via the `HEYLOOK_ADMIN_TOKEN` env var.

### Enable

```sh
export HEYLOOK_ADMIN_TOKEN='choose-any-high-entropy-value'
heylookllm --host 127.0.0.1 --port 8080
```

### Use

Every admin request must carry the matching header:

```sh
curl -X POST https://heylook.local/v1/data/clear \
  -H "X-Heylook-Admin-Token: choose-any-high-entropy-value"
```

Requests without the header (or with a mismatched value) return
`401 Unauthorized`. The check is case-insensitive on the header name
(per RFC 7230); the value itself is compared exactly.

### What's NOT gated

Inference endpoints are deliberately not gated:

- `/v1/chat/completions`
- `/v1/messages/*`
- `/v1/embeddings`
- `/v1/rlm/*`

These are the normal-traffic paths every client app depends on. Gating
them would force every client to learn about a shared secret, which
defeats the reason we're running a single-user server in the first
place. The admin token adds a gate only on mutating / destructive
admin endpoints.

### Empty value = no-op

Setting `HEYLOOK_ADMIN_TOKEN=''` is treated the same as not setting it.
This catches the common footgun of `export HEYLOOK_ADMIN_TOKEN` with no
value attached.

## service templates

The launchd and systemd templates shipped under
`src/heylook_llm/data/services/` default `--host 127.0.0.1`. If you
install via `heylookllm service install`, the generated unit file
binds to loopback. Override with `--host 0.0.0.0` only if you're
skipping the reverse proxy and want direct LAN exposure on HTTP.

## Troubleshooting

- **Caddy says "permission denied" on :443**: give the binary the
  capability (`sudo setcap cap_net_bind_service=+ep $(which caddy)`) or
  run under a service manager that handles privileged ports.
- **`net::ERR_CERT_AUTHORITY_INVALID` in browser**: the local CA isn't
  trusted on that client. Re-run the copy + `update-ca-certificates`
  step (or equivalent on Windows / macOS keychain).
- **401 from admin endpoint you weren't expecting**: check whether
  `HEYLOOK_ADMIN_TOKEN` is set in the shell / systemd unit / launchd
  plist. A value of `''` is treated as unset; anything else is
  enforced.
- **Stream cut off at ~60s through nginx**: add `proxy_read_timeout
  10m;` inside the `location` block. Caddy has sensible defaults for
  SSE out of the box.
