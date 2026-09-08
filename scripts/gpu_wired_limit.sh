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
#
# `install` does TWO INDEPENDENT THINGS and this script keeps them independent:
# it raises the value for THIS boot (a plain sysctl, nothing to do with
# launchd), and it registers a LaunchDaemon for every boot after. Doing the
# sysctl first means a launchd failure can no longer cost you the half that
# was going to work anyway -- under `set -e` a failing `bootstrap` used to
# abort the run with the old job booted out, the new plist written, the value
# never applied, and not even an error line, which is the worst of the three
# states and the quietest.
#
# The daemon is then verified by asking LAUNCHD (`launchctl print`, which
# needs no root), never by reading the sysctl back: after the direct apply, a
# sysctl readback returns this script's own write and would say "installed"
# just as happily for a job that never ran. launchd reports `runs` and `last
# exit code`, which is the difference between "the plist is on disk" and "the
# job works" -- and `bootstrap` returns BEFORE the job has run, so the check
# waits for it rather than racing it, the same race a readback lost on
# 2026-09-07.
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

# The for-this-boot half. `set` IS this, and `install` starts with it -- one
# copy, because two copies of the same instruction diverge (they already had).
apply_mb() {
  sysctl -w iogpu.wired_limit_mb="$1" >/dev/null
  echo "applied for this boot: iogpu.wired_limit_mb=$1"
  echo "Metal reports the new working set to any NEW process, so restart"
  echo "heylookllm (and reload any resident gguf model) to size against it."
}

# launchd's own record of the job. Empty when nothing is registered. No root:
# `launchctl print` reads the system domain for any user.
daemon_print() { launchctl print "system/${LABEL}" 2>/dev/null || true; }

# Parse from a VARIABLE, never `... | grep -q`: a pipe under `set -o pipefail`
# can lose to SIGPIPE and read as a clean false.
field() {  # field <text> <regex-with-one-group>
  [[ "$1" =~ $2 ]] && echo "${BASH_REMATCH[1]}" || true
}

plist_mb() {
  [ -f "$PLIST" ] || return 0
  field "$(cat "$PLIST")" 'iogpu\.wired_limit_mb=([0-9]+)'
}

# ONE interpreter of launchd's record, because `status` and `install` asking
# the same question two different ways is how the first version of this got it
# wrong in both places at once. Echoes exactly one of:
#
#   absent        no plist, no job
#   unloaded      plist on disk, launchd has no such job (the silent one)
#   pending       loaded, no exit code recorded YET -- it may still be running
#   ok            loaded, last run exited 0
#   failed <code> loaded, last run exited non-zero
#
# Keyed on `last exit code`, which launchd writes when the job EXITS. `runs`
# is written when it is SPAWNED, and the window between the two is real: the
# first version polled `runs`, sampled inside that window, read an empty exit
# code, and reported a healthy install as "FAILED (exit ?)" -- the same race
# the sysctl readback lost, moved one field over. An unknown is `pending`, and
# `pending` is NEVER rendered as a failure.
daemon_state() {
  local out code
  out=$(daemon_print)
  if [ -z "$out" ]; then
    [ -f "$PLIST" ] && echo "unloaded" || echo "absent"
    return
  fi
  code=$(field "$out" 'last exit code = ([0-9]+)')
  if [ -z "$code" ]; then
    echo "pending"
  elif [ "$code" = "0" ]; then
    echo "ok"
  else
    echo "failed $code"
  fi
}

# `bootstrap` returns before the job has run, so settle rather than sample.
# A timeout stays `pending` -- "could not confirm" is its own answer and must
# not be dressed up as either success or failure.
await_settled() {
  local i state
  for i in $(seq 1 30); do
    state=$(daemon_state)
    [ "$state" = "pending" ] || { echo "$state"; return 0; }
    sleep 0.1
  done
  echo "pending"
}

# The states worth telling apart, rendered once. File existence alone cannot:
# a plist launchd rejected looks identical on disk to a working one, and that
# is exactly what this reports.
daemon_report() {
  local state loaded_mb disk_mb out
  state=$(daemon_state); out=$(daemon_print); disk_mb=$(plist_mb)
  loaded_mb=$(field "$out" 'iogpu\.wired_limit_mb=([0-9]+)')
  case "$state" in
    absent)
      echo "boot daemon           not installed (a set value is lost at reboot)"
      return ;;
    unloaded)
      echo "boot daemon           plist on disk but NOT loaded -- launchd has no such job."
      echo "                      It will not survive a reboot. Re-run: sudo $0 install ${disk_mb:-}"
      return ;;
    pending)
      echo "boot daemon           loaded, run not finished yet (applies ${loaded_mb:-?} MB at boot)" ;;
    ok)
      echo "boot daemon           loaded, ran ok (applies ${loaded_mb:-?} MB at boot)" ;;
    failed*)
      echo "boot daemon           loaded but its last run FAILED (exit ${state#failed }); the value"
      echo "                      will not come back at reboot. Check: launchctl print system/${LABEL}" ;;
  esac
  if [ -n "${disk_mb:-}" ] && [ -n "${loaded_mb:-}" ] && [ "$disk_mb" != "$loaded_mb" ]; then
    echo "                      NOTE: plist on disk says ${disk_mb} MB -- edited since it was"
    echo "                      loaded. Re-run: sudo $0 install ${disk_mb}"
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
    daemon_report
    echo "suggested             $(default_mb) MB  (total - 12 GiB)"
    ;;
  set)
    need_root set "${2:-}"; mb="${2:-$(default_mb)}"; validate_mb "$mb"
    apply_mb "$mb"
    ;;
  install)
    need_root install "${2:-}"; mb="${2:-$(default_mb)}"; validate_mb "$mb"
    # This boot FIRST: it cannot fail for a launchd reason, so nothing below
    # can cost you it.
    apply_mb "$mb"
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
    # Explicit, because `set -e` would otherwise abort here with a raw
    # launchctl error and no word about which half survived.
    if ! launchctl bootstrap system "$PLIST"; then
      echo "" >&2
      echo "error: launchctl could not load $PLIST." >&2
      echo "  This boot IS set (${mb} MB, applied above) -- but it will be LOST at" >&2
      echo "  reboot. The plist is on disk; 'sudo $0 uninstall' removes it." >&2
      exit 1
    fi
    state=$(await_settled)
    case "$state" in
      ok)
        echo ""
        echo "installed $PLIST -- verified: the job ran and exited 0, so ${mb} MB"
        echo "comes back at every boot. 'sudo $0 uninstall' removes it."
        ;;
      pending)
        # Not a failure and not a success. Say so.
        echo ""
        echo "installed $PLIST, but its first run had not finished after 3s, so this"
        echo "cannot confirm it works. This boot IS set (${mb} MB). Check before"
        echo "relying on the next one: $0 status"
        exit 1
        ;;
      *)
        echo "" >&2
        echo "error: the boot job ran and FAILED (exit ${state#failed }). This boot is" >&2
        echo "  set (${mb} MB), but the value will not come back at reboot." >&2
        echo "  Detail: launchctl print system/${LABEL}" >&2
        exit 1
        ;;
    esac
    ;;
  uninstall)
    need_root uninstall
    launchctl bootout system "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    if [ -n "$(daemon_print)" ]; then
      echo "warning: launchd still reports system/${LABEL} after bootout." >&2
      echo "  The plist is gone, so it will not return at reboot." >&2
    fi
    echo "removed $PLIST; the running value stays until reboot (sysctl -w iogpu.wired_limit_mb=0 to drop it now)"
    ;;
  *) echo "usage: $0 status | set [MB] | install [MB] | uninstall" >&2; exit 2 ;;
esac
