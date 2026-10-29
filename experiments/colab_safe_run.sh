#!/usr/bin/env bash
# Run one self-contained Python script on a fresh Colab L4 VM with termination guarantees.
#
#   usage (inside WSL):  bash colab_safe_run.sh SCRIPT MAX_MINUTES OUTFILE [script args...]
#
# Guarantees, from innermost to outermost:
#   1. the script receives its budget as env BUDGET_SECONDS and must finish before it;
#   2. `colab run --timeout` bounds execution on the VM;
#   3. `timeout --kill-after` kills the local client if the VM call hangs;
#   4. an EXIT/INT/TERM trap always runs `colab stop -s NAME`, then audits the server:
#      it lists every active assignment and fails loudly if any session remains.
# Only --gpu L4 is ever requested (no fallback to other accelerators).
set -u
SCRIPT="$1"; MAXMIN="$2"; OUT="$3"; shift 3
NAME="pa-$(date +%m%d-%H%M%S)-$RANDOM"
LOG="${OUT%.*}.colab.log"
LOCKDIR=/tmp/pa_sessions; mkdir -p "$LOCKDIR"; echo $$ > "$LOCKDIR/$NAME"   # claim: this session is live
SECS=$((MAXMIN * 60))
: > "$LOG"

cleanup() {
  code=$?
  echo "[safe_run] cleanup: stopping $NAME" >> "$LOG"
  colab stop -s "$NAME" >> "$LOG" 2>&1 || true
  rm -f "$LOCKDIR/$NAME"
  AUDIT=$(colab sessions 2>&1 | grep -v -i "new version\|colab update\|update_check")
  echo "[safe_run] server audit:" >> "$LOG"; echo "$AUDIT" >> "$LOG"
  if echo "$AUDIT" | grep -q "No active sessions found on server"; then
    echo "[safe_run] exit=$code session=$NAME audit=CLEAN (no active sessions)"
  else
    echo "[safe_run] exit=$code session=$NAME audit=NOT CLEAN -- active sessions remain:"
    echo "$AUDIT"
    # stop only GHOSTS: sessions whose owning safe_run is gone (no lock, or lock's PID not alive).
    # Sessions of other live safe_runs (parallel jobs) are left alone.
    for s in $(echo "$AUDIT" | grep -o "pa-[0-9-]*"); do
      if [ -f "$LOCKDIR/$s" ] && kill -0 "$(cat "$LOCKDIR/$s")" 2>/dev/null; then
        echo "[safe_run] leaving $s (owned by live safe_run pid $(cat "$LOCKDIR/$s"))"
      else
        echo "[safe_run] stopping ghost $s"; colab stop -s "$s" >> "$LOG" 2>&1; rm -f "$LOCKDIR/$s"
      fi
    done
    colab sessions 2>&1 | grep -v -i "new version\|colab update\|update_check" | tail -3
  fi
}
trap cleanup EXIT INT TERM

timeout --kill-after=60 $((SECS + 300)) \
  colab run --gpu L4 -s "$NAME" --timeout "$SECS" --env "BUDGET_SECONDS=$SECS" \
  "$SCRIPT" "$@" > "$OUT" 2>> "$LOG"
