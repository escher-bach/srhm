#!/usr/bin/env bash
# fetch the progress file of running sessions:  bash colab_fetch.sh OUT [OUT ...]   (OUT as given to colab_safe_run3.sh)
for out in "$@"; do
  s=$(cat "$out.session"); prog="${out%.*}.progress.jsonl"
  timeout 100 colab download -s "$s" /content/progress.jsonl "$prog" > /dev/null 2>&1 && echo "fetched $prog ($(wc -l < "$prog") lines)" || echo "could not fetch $prog"
done
