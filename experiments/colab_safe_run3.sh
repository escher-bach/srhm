#!/usr/bin/env bash
# colab_safe_run2.sh with a choice of runtime and an optional data file.  Since 2026-10-11 every run goes through Colab,
# CPU-only ones included (the laptop is short of memory).
#   usage (inside WSL):  bash colab_safe_run3.sh RUNTIME SCRIPT MAX_MINUTES OUTFILE DATA|- [KEY=VALUE ...]
#   RUNTIME: cpu | cpu-high (high-RAM CPU machine) | L4
# new session -> upload DATA (unless "-") -> exec SCRIPT -> on ANY exit: download the progress file, stop the session,
# audit the server.  The script must append its log lines to /content/progress.jsonl (env PROGRESS_FILE).
set -u
RUNTIME="$1"; SCRIPT="$2"; MAXMIN="$3"; OUT="$4"; DATA="$5"; shift 5
NAME="pa-$(date +%m%d-%H%M%S)-$RANDOM"
LOG="${OUT%.*}.colab.log"; PROG="${OUT%.*}.progress.jsonl"
LOCKDIR=/tmp/pa_sessions; mkdir -p "$LOCKDIR"; echo $$ > "$LOCKDIR/$NAME"
SECS=$((MAXMIN * 60))
: > "$LOG"; echo "$NAME" > "$OUT.session"
ENVS=(--env "BUDGET_SECONDS=$SECS" --env "PROGRESS_FILE=/content/progress.jsonl")
for kv in "$@"; do ENVS+=(--env "$kv"); done
case "$RUNTIME" in
  cpu) NEW=() ;;
  cpu-high) NEW=(--high-mem) ;;
  L4) NEW=(--gpu L4) ;;
  *) echo "unknown runtime $RUNTIME"; exit 2 ;;
esac

cleanup() {
  code=$?
  echo "[safe_run3] cleanup: fetching progress, stopping $NAME" >> "$LOG"
  timeout 120 colab download -s "$NAME" /content/progress.jsonl "$PROG" >> "$LOG" 2>&1 || true
  colab stop -s "$NAME" >> "$LOG" 2>&1 || true
  rm -f "$LOCKDIR/$NAME"
  AUDIT=$(colab sessions 2>&1 | grep -v -i "new version\|colab update\|update_check")
  echo "[safe_run3] server audit:" >> "$LOG"; echo "$AUDIT" >> "$LOG"
  if echo "$AUDIT" | grep -q "No active sessions found on server"; then
    echo "[safe_run3] exit=$code session=$NAME audit=CLEAN (no active sessions)"
  else
    echo "[safe_run3] exit=$code session=$NAME audit=NOT CLEAN -- active sessions remain:"
    echo "$AUDIT"
    for s in $(echo "$AUDIT" | grep -o "pa-[0-9-]*"); do
      if [ -f "$LOCKDIR/$s" ] && kill -0 "$(cat "$LOCKDIR/$s")" 2>/dev/null; then
        echo "[safe_run3] leaving $s (owned by live safe_run pid $(cat "$LOCKDIR/$s"))"
      else
        echo "[safe_run3] stopping ghost $s"; colab stop -s "$s" >> "$LOG" 2>&1; rm -f "$LOCKDIR/$s"
      fi
    done
    colab sessions 2>&1 | grep -v -i "new version\|colab update\|update_check" | tail -3
  fi
}
trap cleanup EXIT INT TERM

timeout 600 colab new "${NEW[@]}" -s "$NAME" >> "$LOG" 2>&1 || { echo "[safe_run3] could not create session" >> "$LOG"; exit 3; }
if [ "$DATA" != "-" ]; then
  timeout 900 colab upload -s "$NAME" "$DATA" "/content/$(basename "$DATA")" >> "$LOG" 2>&1 || { echo "[safe_run3] upload failed" >> "$LOG"; exit 4; }
fi
# exec with a watchdog: a lost connection leaves the client hanging and the session idle.  The script writes its first
# progress line within seconds; if none exists after 120 s, kill the client and retry.
for attempt in 1 2 3; do
  timeout --kill-after=60 $((SECS + 300)) \
    colab exec -s "$NAME" -f "$SCRIPT" --timeout "$SECS" "${ENVS[@]}" > "$OUT" 2>> "$LOG" &
  EXEC=$!
  sleep 120
  if timeout 120 colab download -s "$NAME" /content/progress.jsonl "$PROG.check" >> "$LOG" 2>&1; then
    rm -f "$PROG.check"; echo "[safe_run3] attempt $attempt: script is running" >> "$LOG"
    wait $EXEC; exit $?
  fi
  echo "[safe_run3] attempt $attempt: no progress after 120 s, restarting exec" >> "$LOG"
  pkill -TERM -P $EXEC 2>/dev/null; kill -TERM $EXEC 2>/dev/null; wait $EXEC 2>/dev/null
  timeout 120 colab restart-kernel -s "$NAME" >> "$LOG" 2>&1 || true
done
echo "[safe_run3] script never started" >> "$LOG"; exit 5
