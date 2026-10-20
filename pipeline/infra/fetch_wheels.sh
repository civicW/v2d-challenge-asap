#!/bin/bash
# Download every flash_chord lock wheel with curl retry/resume + sha256 verify.
# Files that already match their hash are skipped, so the script is resumable.
set -u
LIST=$1
OUT=$2
mkdir -p "$OUT"
ok=0
fail=0
n=0
total=$(grep -c . "$LIST")
while IFS=$'\t' read -r url sha; do
  [ -z "$url" ] && continue
  n=$((n + 1))
  fn="$OUT/$(basename "$url")"
  good=0
  for attempt in 1 2 3 4 5 6 7 8; do
    if [ -f "$fn" ] && [ "$(sha256sum "$fn" | cut -d' ' -f1)" = "$sha" ]; then
      good=1
      break
    fi
    curl -sL --retry 3 --retry-all-errors -C - --max-time 900 -o "$fn" "$url" || true
  done
  if [ "$good" = "1" ]; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    echo "FAILED: $(basename "$url")"
  fi
  if [ $((n % 10)) = "0" ]; then
    echo "progress: $n/$total ok=$ok fail=$fail $(date +%H:%M:%S)"
  fi
done < "$LIST"
echo "DONE ok=$ok fail=$fail total=$total $(date +%H:%M:%S)"
