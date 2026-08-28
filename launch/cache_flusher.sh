#!/usr/bin/env bash
# Unconditional drop_caches every 60s for the whole boot window (Troubleshooting #6).
end=$((SECONDS+3600))
while [ $SECONDS -lt $end ]; do
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  sleep 60
done
