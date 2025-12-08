# Running heylookllm as a Background Service

This guide covers running heylookllm as a persistent background service on macOS and Ubuntu (headless servers). The service will persist after SSH disconnection and auto-restart on failure.

## Security Model

heylookllm is designed for **personal/hobbyist use behind a VPN**. The default configuration is secure:

- **Binds to localhost (127.0.0.1)** - Only accessible from the local machine
- **No authentication** - Relies on network-level security (VPN, firewall)
- **No TLS** - Use a reverse proxy if you need HTTPS

For LAN access (e.g., from other machines on your VPN), you must explicitly bind to `0.0.0.0` and configure firewall rules.

## Quick Start

### Install as Service

```bash
# Local-only (most secure, default)
heylookllm service install

# LAN access (behind VPN)
heylookllm service install --host 0.0.0.0

# Custom port
heylookllm service install --host 0.0.0.0 --port 8080
```

### Manage Service

```bash
# Check status
heylookllm service status

# Start/stop/restart
heylookllm service start
heylookllm service stop
heylookllm service restart

# Uninstall
heylookllm service uninstall
```

## Platform-Specific Setup

### Ubuntu/Linux (systemd)

#### User-Level Service (Recommended)

User-level services run without root privileges and persist after SSH logout:

```bash
# Install the service
heylookllm service install --host 0.0.0.0

# The installer automatically enables lingering for persistence
# You can verify with:
loginctl show-user $USER | grep Linger
```

#### System-Wide Service (Optional)

For shared servers where multiple users need access:

```bash
# Requires sudo
sudo heylookllm service install --host 0.0.0.0 --system-wide
```

#### Viewing Logs

```bash
# User-level service
journalctl --user -u heylookllm -f

# System-wide service
journalctl -u heylookllm -f

# Application logs (if file logging enabled)
tail -f logs/heylookllm_*.log
```

#### Firewall Configuration (UFW)

If using UFW on Ubuntu, allow access only from your local network:

```bash
# Allow from specific subnet (your VPN/LAN range)
sudo ufw allow from 192.168.1.0/24 to any port 8080

# Or allow from specific IP
sudo ufw allow from 192.168.1.100 to any port 8080

# Verify rules
sudo ufw status
```

### macOS (launchd)

#### Installing the Service

```bash
# Install (runs as user-level LaunchAgent)
heylookllm service install --host 0.0.0.0

# The service starts automatically after installation
```

#### Viewing Logs

```bash
# launchd captures stdout/stderr
tail -f logs/heylookllm.stdout.log
tail -f logs/heylookllm.stderr.log

# Application logs
tail -f logs/heylookllm_*.log
```

#### Firewall Configuration (pf)

macOS uses the pf firewall. Create `/etc/pf.anchors/heylookllm`:

```
# Allow heylookllm from local network only
pass in on en0 proto tcp from 192.168.1.0/24 to any port 8080
block in on en0 proto tcp from any to any port 8080
```

Then add to `/etc/pf.conf`:

```
anchor "heylookllm"
load anchor "heylookllm" from "/etc/pf.anchors/heylookllm"
```

Apply with: `sudo pfctl -f /etc/pf.conf`

Alternatively, use the built-in Firewall in System Settings > Network > Firewall.

## Configuration Options

### Service Installation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` for LAN access |
| `--port` | `8080` | Server port |
| `--log-level` | `INFO` | Console log level (DEBUG, INFO, WARNING, ERROR) |
| `--log-dir` | `./logs` | Directory for log files |
| `--system-wide` | false | Linux: Install as system service (requires sudo) |

### Example Configurations

#### Minimal (Local Only)
```bash
heylookllm service install
```
Access only from the same machine at `http://127.0.0.1:8080`.

#### Home Lab (LAN via VPN)
```bash
heylookllm service install --host 0.0.0.0 --port 8080
```
Access from any machine on your local network at `http://<server-ip>:8080`.

#### Debug Mode
```bash
heylookllm service install --host 0.0.0.0 --log-level DEBUG
```
Verbose logging for troubleshooting.

## Security Recommendations

### Network Security

1. **Use a VPN**: The simplest security model. Run heylookllm on a VPN-only network.

2. **Firewall Rules**: Restrict access by IP/subnet:
   - Only allow your VPN subnet
   - Block all other incoming connections to the port

3. **Don't expose to the internet**: heylookllm has no authentication. Never expose it directly to the public internet.

### If You Need Remote Access

For accessing your server from outside your home network:

1. **VPN (Recommended)**: Connect to your home VPN, then access heylookllm on its LAN IP
   - Tailscale, WireGuard, or OpenVPN work well

2. **SSH Tunnel**: Forward the port over SSH
   ```bash
   ssh -L 8080:localhost:8080 user@your-server
   # Then access http://localhost:8080 locally
   ```

3. **Reverse Proxy with Auth** (Advanced): Put nginx/Caddy in front with HTTP Basic Auth or client certificates

### Resource Limits

For long-running services on shared systems, consider adding resource limits:

**Linux (systemd)**: Edit the service file to add:
```ini
[Service]
MemoryMax=80%
CPUQuota=90%
```

**macOS (launchd)**: Add to the plist:
```xml
<key>HardResourceLimits</key>
<dict>
    <key>NumberOfFiles</key>
    <integer>65536</integer>
</dict>
```

## Troubleshooting

### Service Won't Start

1. Check logs for errors:
   ```bash
   # Linux
   journalctl --user -u heylookllm -n 50

   # macOS
   cat logs/heylookllm.stderr.log
   ```

2. Verify the virtual environment:
   ```bash
   which heylookllm
   heylookllm --help
   ```

3. Check if port is in use:
   ```bash
   lsof -i :8080
   ```

### Service Stops After SSH Disconnect (Linux)

Enable lingering for your user:
```bash
loginctl enable-linger $USER
```

The service installer does this automatically, but verify with:
```bash
loginctl show-user $USER | grep Linger
```

### Permission Errors

- Ensure the working directory and log directory are writable
- For system-wide services, check user/group in the service file

### Can't Connect from Other Machines

1. Verify the server is listening on 0.0.0.0:
   ```bash
   netstat -tlnp | grep 8080
   # Should show 0.0.0.0:8080, not 127.0.0.1:8080
   ```

2. Check firewall rules allow the connection

3. Verify you're on the same network/VPN

## Uninstalling

```bash
# Stop and remove the service
heylookllm service uninstall

# The command removes:
# - Linux: ~/.config/systemd/user/heylookllm.service
# - macOS: ~/Library/LaunchAgents/com.heylookllm.server.plist
```

Log files in the `logs/` directory are preserved.
