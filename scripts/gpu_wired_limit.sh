#!/usr/bin/env bash
# scripts/gpu_wired_limit.sh -- read, raise, or persist macOS's GPU wired limit.
#
# `iogpu.wired_limit_mb` is the SYSTEM ceiling on GPU-wired memory. At its OS
# default (0) Metal reports a recommended working set of roughly 84% of RAM on
# a 192 GiB M2 Ultra (~161 GiB), and that number is what both engines size
# against: MLX refuses above it (heylook's mx.set_wired_limit is set TO it)
# and llama-server's --fit shrinks unset arguments (context first) to stay
# under it. A 145 GiB model at a 1M-token context lives inside the last
# 16 GiB of that ceiling, which is where a decode-time Metal OOM comes from.
#
# Raising the sysctl is the ONLY lever that enlarges the working set. It
# needs root, so heylook cannot do it for you; the admin fit panel and the
# generation error message print the value, this script applies it.
#
#   scripts/gpu_wired_limit.sh status            # current sysctl + what Metal reports
#   sudo scripts/gpu_wired_limit.sh set [MB]     # raise now (lost at reboot)
#   sudo scripts/gpu_wired_limit.sh install [MB] # also a LaunchDaemon that re-applies it at boot
#   sudo scripts/gpu_wired_limit.sh uninstall    # remove the daemon; sysctl back to 0 at next boot
#
# MB defaults to total RAM minus 12 GiB (the OS + everything that is not the
# model). Set it lower if you run other GPU-heavy work beside heylook.
set -euo pipefail

LABEL="com.heylook.gpu-wired-limit"
PLIST="/Library/LaunchDaemons/${LABEL}.plist"
RESERVE_MB=$((12 * 1024))

total_mb() { echo $(( $(sysctl -n hw.memsize) / 1024 / 1024 )); }
default_mb() { echo $(( $(total_mb) - RESERVE_MB )); }

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "error: $1 needs root -- run: sudo $0 $1 ${2:-}" >&2; exit 1
  fi
}

validate_mb() {
  local mb="$1" total; total=$(total_mb)
  case "$mb" in ''|*[!0-9]*) echo "error: MB must be an integer, got '$mb'" >&2; exit 2 ;; esac
  if [ "$mb" -gt $((total - 4096)) ]; then
    echo "error: $mb MB leaves under 4 GiB of $total MB for the OS -- refusing" >&2; exit 2
  fi
}

case "${1:-status}" in
  status)
    cur=$(sysctl -n iogpu.wired_limit_mb)
    total=$(total_mb)
    echo "total RAM             ${total} MB"
    if [ "$cur" = "0" ]; then
      echo "iogpu.wired_limit_mb  0  (OS default; Metal recommends ~84% of RAM)"
    else
      echo "iogpu.wired_limit_mb  ${cur}  (set; $((cur * 100 / total))% of RAM)"
    fi
    [ -f "$PLIST" ] && echo "boot daemon           installed ($PLIST)" || echo "boot daemon           not installed (a set value is lost at reboot)"
    echo "suggested             $(default_mb) MB  (total - 12 GiB)"
    ;;
  set)
    need_root set "${2:-}"; mb="${2:-$(default_mb)}"; validate_mb "$mb"
    sysctl -w iogpu.wired_limit_mb="$mb"
    echo "applied for this boot. Metal now reports the new working set to any NEW process;"
    echo "restart heylookllm (and reload any resident gguf model) to size against it."
    ;;
  install)
    need_root install "${2:-}"; mb="${2:-$(default_mb)}"; validate_mb "$mb"
    cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key><array>
    <string>/usr/sbin/sysctl</string><string>-w</string><string>iogpu.wired_limit_mb=${mb}</string>
  </array>
  <key>RunAtLoad</key><true/>
</dict></plist>
PL
    chown root:wheel "$PLIST"; chmod 644 "$PLIST"
    launchctl bootout system "$PLIST" 2>/dev/null || true
    launchctl bootstrap system "$PLIST"
    # bootstrap returns before the job has run, so a readback here races it
    # (it printed 0 on 2026-09-07 while the value applied a moment later).
    # Apply directly for THIS boot; the daemon covers the next ones.
    sysctl -w iogpu.wired_limit_mb="$mb" >/dev/null
    echo "installed $PLIST (re-applies iogpu.wired_limit_mb=${mb} at every boot); now:"
    sysctl iogpu.wired_limit_mb
    echo "restart heylookllm (and reload any resident gguf model) to size against the new ceiling."
    ;;
  uninstall)
    need_root uninstall
    launchctl bootout system "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "removed $PLIST; the running value stays until reboot (sysctl -w iogpu.wired_limit_mb=0 to drop it now)"
    ;;
  *) echo "usage: $0 status | set [MB] | install [MB] | uninstall" >&2; exit 2 ;;
esac
