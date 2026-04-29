#!/data/data/com.termux/files/usr/bin/sh

# Continuous ping monitor for Android/Termux.
# Prints raw ping-related metrics only (no mode classification).
#
# Usage:
#   sh net_state_monitor.sh [target_ip] [interval_sec] [count]
# Example:
#   sh net_state_monitor.sh 1.1.1.1 1 0
# count=0 means run forever.

set -u
export PATH=/data/data/com.termux/files/usr/bin:/system/bin:/system/xbin:$PATH

TARGET="${1:-1.1.1.1}"
INTERVAL="${2:-1}"
COUNT="${3:-0}"
PING_COUNT="${PING_COUNT:-5}"
PING_TIMEOUT="${PING_TIMEOUT:-1}"

is_number() {
  case "$1" in
    ''|*[!0-9]*)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

if ! is_number "$INTERVAL" || ! is_number "$COUNT"; then
  echo "Error: interval and count must be non-negative integers."
  exit 1
fi

if ! command -v ip >/dev/null 2>&1; then
  echo "Error: 'ip' command not found."
  exit 1
fi

if ! command -v ping >/dev/null 2>&1; then
  echo "Error: 'ping' command not found. Install with: pkg install iputils"
  exit 1
fi

round=0
while :; do
  round=$((round + 1))

  ping_out="$(ping -c "$PING_COUNT" -W "$PING_TIMEOUT" "$TARGET" 2>/dev/null || true)"
  stats_line="$(echo "$ping_out" | grep -E '[0-9]+ packets transmitted' | tail -n 1 || true)"
  rtt_line="$(echo "$ping_out" | grep -E 'min/avg/max|round-trip min/avg/max' | tail -n 1 || true)"
  loss_field="$(echo "$ping_out" | grep -o '[0-9]\+% packet loss' | tail -n 1 || true)"
  tx="$(echo "$stats_line" | sed -n 's/^\([0-9]\+\) packets transmitted.*/\1/p')"
  rx="$(echo "$stats_line" | sed -n 's/^[0-9]\+ packets transmitted, \([0-9]\+\) [a-zA-Z]\+.*/\1/p')"
  loss="${loss_field%%%*}"

  [ -z "$tx" ] && tx=0
  [ -z "$rx" ] && rx=0
  [ -z "$loss" ] && loss=100

  ts="$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date)"
  echo "[$ts] round=$round target=$TARGET tx=$tx rx=$rx loss=${loss}% timeout=${PING_TIMEOUT}s"
  [ -n "$rtt_line" ] && echo "[$ts] rtt $rtt_line"

  if [ "$COUNT" -gt 0 ] && [ "$round" -ge "$COUNT" ]; then
    break
  fi

  sleep "$INTERVAL"
done
