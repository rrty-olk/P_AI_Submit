#!/usr/bin/env bash
set -euo pipefail
export DISPLAY=:98
Xvfb :98 -screen 0 640x560x24 -nolisten tcp &
xpid=$!
trap 'kill ${tpid:-} ${wpid:-} "$xpid" 2>/dev/null || true' EXIT
for i in {1..50}; do xdpyinfo -display :98 >/dev/null 2>&1 && break; sleep .1; done
ros2 run turtlesim turtlesim_node > /app/output/logs/safety_turtlesim.txt 2>&1 &
tpid=$!
python -m inspection.watchdog > /app/output/logs/safety_watchdog.txt 2>&1 &
wpid=$!
python tests/ros_safety.py
