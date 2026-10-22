#!/usr/bin/env bash
set -euo pipefail
Xvfb :99 -screen 0 640x560x24 -nolisten tcp &
xpid=$!
trap 'kill ${tpid:-} ${wpid:-} "$xpid" 2>/dev/null || true' EXIT
for i in {1..50}; do xdpyinfo -display :99 >/dev/null 2>&1 && break; sleep 0.1; done
ros2 run turtlesim turtlesim_node > /app/output/logs/turtlesim.txt 2>&1 &
tpid=$!
python -m inspection.watchdog > /app/output/logs/watchdog.txt 2>&1 &
wpid=$!
python -m inspection.robot
