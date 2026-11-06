#!/bin/bash
# Kills the training container if host MemAvailable drops below a floor (unified-memory hosts freeze instead of OOM-killing).
# usage: WORKDIR=... watchdog.sh   (run alongside run_ddp.sh on each node)
: "${WORKDIR:?}"; FLOOR_MB=${FLOOR_MB:-8000}
while ! grep -q "^EXIT" "$WORKDIR/logs/ddp.log" 2>/dev/null; do
  avail=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
  echo "$(date +%T) avail_MB=$avail" >> "$WORKDIR/logs/ddp.mem"
  if [ "$avail" -lt "$FLOOR_MB" ]; then echo "WATCHDOG: avail ${avail}MB" >> "$WORKDIR/logs/ddp.mem"; docker kill uvx-ddp; fi
  sleep 3
done
