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

## Two optional auth gates

The server ships with two independent env-var-driven auth checks. Both
are opt-in and default to off.

| Env var | Header | Gates | Default |
|---|---|---|---|
| `HEYLOOK_ADMIN_TOKEN` | `X-Heylook-Admin-Token: <value>` | Admin endpoints (`/v1/admin/*`, `/v1/data/clear`, `/v1/cache/clear`) | unset = open |
| `HEYLOOK_API_KEY` | `Authorization: Bearer <value>` | Inference endpoints (chat completions, batch chat completions, messages, embeddings, hidden states, RLM) | unset = open |

Empty value (`HEYLOOK_API_KEY=''`) is treated as unset -- catches the
common `export HEYLOOK_API_KEY` with no value attached. Both checks use
`hmac.compare_digest` so a wrong-length guess and a close-match guess
take the same time.

### Admin token

```sh
export HEYLOOK_ADMIN_TOKEN='choose-any-high-entropy-value'
heylookllm --host 127.0.0.1 --port 8080
```

```sh
curl -X POST https://heylook.local/v1/data/clear \
  -H "X-Heylook-Admin-Token: choose-any-high-entropy-value"
```

Requests without the header (or with a mismatched value) return `401`.

### Inference API key

Opt-in bearer auth for the inference routes. By default, **loopback
traffic is exempt** -- dev tools on the same machine that hit
`http://127.0.0.1:8080` don't need to carry the key. LAN and remote
clients do.

```sh
export HEYLOOK_API_KEY='choose-any-high-entropy-value'
heylookllm --host 127.0.0.1 --port 8080
```

On LAN clients (Ubuntu box, ComfyUI node, phone) send:

```sh
curl -X POST https://heylook.local/v1/chat/completions \
  -H "Authorization: Bearer choose-any-high-entropy-value" \
  -H "Content-Type: application/json" \
  -d '{"model": "...", "messages": [{"role": "user", "content": "hi"}]}'
```

OpenAI SDK clients (Python, JS, shrug-prompter, etc.) already send
`Authorization: Bearer <api_key>` as the standard field -- set it on
your client's `api_key` option.

#### Close the loopback carve-out

For paranoid setups (public-exposed server, shared machine) set:

```sh
export HEYLOOK_API_KEY_ENFORCE_LOOPBACK=true
```

Now `127.0.0.1` and `::1` also require the bearer header. Truthy values:
`true`, `1`, `yes`, `on` (case-insensitive). Anything else -- including
`false`, `no`, `0`, and empty -- keeps the default exempt behavior.

### What's explicitly NOT gated

- **Static v2 frontend** (`/v2/*`) -- it runs in the browser and would
  need its own credential-pass mechanism. Open either way on LAN.
- **`/v1/models`**, **`/v1/capabilities`**, **`/v1/system/metrics`**,
  **`/v1/performance/*`** -- read-only metadata that every frontend
  polls. Not behind the API-key gate today; revisit if you ever expose
  the server publicly.
- **Conversations / notebooks CRUD** (`/v1/conversations/*`,
  `/v1/notebooks/*`) -- user-data surfaces the web UI reads and writes.
  Not gated; gate via the admin token + network perimeter if needed.

The admin token + API key combine cleanly: a client hitting admin
endpoints needs both headers if both are set.

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
