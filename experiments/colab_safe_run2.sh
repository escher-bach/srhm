#!/usr/bin/env bash
# Like colab_safe_run.sh, for scripts that need a data file: `colab run` ships the script as ONE kernel message, and a
# 21 MB bundle (data embedded as base64) never started executing (2026-10-01: two runs lost, no output).  Here:
#   new session (L4) -> upload DATA to /content -> exec SCRIPT -> on ANY exit: download the script's progress file,
#   stop the session, audit the server.
#   usage (inside WSL):  bash colab_safe_run2.sh SCRIPT MAX_MINUTES OUTFILE DATA [KEY=VALUE ...]
# The script must append its log lines to /content/progress.jsonl (env PROGRESS_FILE) as well as stdout, so that
# progress survives a timeout and can be fetched mid-run:  colab download -s NAME /content/progress.jsonl local.jsonl
# The session name is written to OUTFILE.session.
set -u
SCRIPT="$1"; MAXMIN="$2"; OUT="$3"; DATA="$4"; shift 4
NAME="pa-$(date +%m%d-%H%M%S)-$RANDOM"
LOG="${OUT%.*}.colab.log"; PROG="${OUT%.*}.progress.jsonl"
LOCKDIR=/tmp/pa_sessions; mkdir -p "$LOCKDIR"; echo $$ > "$LOCKDIR/$NAME"
SECS=$((MAXMIN * 60))
: > "$LOG"; echo "$NAME" > "$OUT.session"
ENVS=(--env "BUDGET_SECONDS=$SECS" --env "PROGRESS_FILE=/content/progress.jsonl")
for kv in "$@"; do ENVS+=(--env "$kv"); done

cleanup() {
  code=$?
  echo "[safe_run2] cleanup: fetching progress, stopping $NAME" >> "$LOG"
  timeout 120 colab download -s "$NAME" /content/progress.jsonl "$PROG" >> "$LOG" 2>&1 || true
  colab stop -s "$NAME" >> "$LOG" 2>&1 || true
  rm -f "$LOCKDIR/$NAME"
  AUDIT=$(colab sessions 2>&1 | grep -v -i "new version\|colab update\|update_check")
  echo "[safe_run2] server audit:" >> "$LOG"; echo "$AUDIT" >> "$LOG"
  if echo "$AUDIT" | grep -q "No active sessions found on server"; then
    echo "[safe_run2] exit=$code session=$NAME audit=CLEAN (no active sessions)"
  else
    echo "[safe_run2] exit=$code session=$NAME audit=NOT CLEAN -- active sessions remain:"
    echo "$AUDIT"
    for s in $(echo "$AUDIT" | grep -o "pa-[0-9-]*"); do
      if [ -f "$LOCKDIR/$s" ] && kill -0 "$(cat "$LOCKDIR/$s")" 2>/dev/null; then
        echo "[safe_run2] leaving $s (owned by live safe_run pid $(cat "$LOCKDIR/$s"))"
      else
        echo "[safe_run2] stopping ghost $s"; colab stop -s "$s" >> "$LOG" 2>&1; rm -f "$LOCKDIR/$s"
      fi
    done
    colab sessions 2>&1 | grep -v -i "new version\|colab update\|update_check" | tail -3
  fi
}
trap cleanup EXIT INT TERM

timeout 600 colab new --gpu L4 -s "$NAME" >> "$LOG" 2>&1 || { echo "[safe_run2] could not create session" >> "$LOG"; exit 3; }
timeout 900 colab upload -s "$NAME" "$DATA" "/content/$(basename "$DATA")" >> "$LOG" 2>&1 || { echo "[safe_run2] upload failed" >> "$LOG"; exit 4; }
# exec with a watchdog: a lost connection leaves the client hanging and the session idle (2026-10-01: 80 idle minutes).
# The script writes its first progress line within seconds; if none exists after 180 s, kill the client and retry.
for attempt in 1 2 3; do
  timeout --kill-after=60 $((SECS + 300)) \
    colab exec -s "$NAME" -f "$SCRIPT" --timeout "$SECS" "${ENVS[@]}" > "$OUT" 2>> "$LOG" &
  EXEC=$!
  sleep 180
  if timeout 120 colab download -s "$NAME" /content/progress.jsonl "$PROG.check" >> "$LOG" 2>&1; then
    rm -f "$PROG.check"; echo "[safe_run2] attempt $attempt: script is running" >> "$LOG"
    wait $EXEC; exit $?
  fi
  echo "[safe_run2] attempt $attempt: no progress after 180 s, restarting exec" >> "$LOG"
  pkill -TERM -P $EXEC 2>/dev/null; kill -TERM $EXEC 2>/dev/null; wait $EXEC 2>/dev/null
  timeout 120 colab restart-kernel -s "$NAME" >> "$LOG" 2>&1 || true
done
echo "[safe_run2] script never started" >> "$LOG"; exit 5
