#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

logs=(./profile_reports/*full_report.log)
if (( ${#logs[@]} == 0 )); then
  echo "No profile reports found." >&2
  exit 1
fi

# Match your specific printf patterns from each .cu file
pattern='(Slow|Average|Fast|Fastest) total execution time'
results=()

for log in "${logs[@]}"; do
  base=${log##*/}
  name=${base%_full_report.log}

  # Get the LAST match (your program output, not nsys overhead)
  match=$(grep -iE "$pattern" "$log" | tail -n1 || true)
  if [[ -z "$match" ]]; then
    echo "Warning: no runtime line found in $log" >&2
    continue
  fi

  # Extract the number before "ms"
  ms=$(sed -n 's/.*[^0-9]\([0-9][0-9]*\(\.[0-9]\+\)\?\)[[:space:]]*ms.*/\1/p' <<< "$match" | head -n1)
  if [[ -z "$ms" ]]; then
    echo "Warning: could not parse time from $log: $match" >&2
    continue
  fi

  results+=("$name $ms")
done

if (( ${#results[@]} == 0 )); then
  echo "No runtimes parsed." >&2
  exit 1
fi

printf "%-18s %12s\n" "profile" "time_ms"
printf "%-18s %12s\n" "-------" "-------"

printf "%s\n" "${results[@]}" | sort -k2,2n | while read -r name ms; do
  printf "%-18s %12s\n" "$name" "$ms"
done
