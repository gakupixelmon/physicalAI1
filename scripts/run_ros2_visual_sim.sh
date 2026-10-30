#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"

if [[ ! -f "$ROS_SETUP" ]]; then
  printf 'ROS 2 setup file not found: %s\n' "$ROS_SETUP" >&2
  exit 1
fi

source "$ROS_SETUP"
set -u
cd "$PROJECT_ROOT"
mkdir -p results

MISSION_LOG="$PROJECT_ROOT/results/ros2_simulation_log.json"
SIM_PYTHON=python3
if [[ -x "$PROJECT_ROOT/.venv-qiskit/bin/python" ]]; then
  SIM_PYTHON="$PROJECT_ROOT/.venv-qiskit/bin/python"
fi
if [[ "${SKIP_MISSION:-0}" != "1" ]]; then
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$SIM_PYTHON" src/main.py --mock --log-path "$MISSION_LOG"
elif [[ ! -s "$MISSION_LOG" ]]; then
  printf 'Mission log not found: %s\n' "$MISSION_LOG" >&2
  exit 1
fi

python3 ros2/mission_visualizer.py --log "$MISSION_LOG" --playback-speed 2.5 --loop &
VISUALIZER_PID=$!
cleanup() {
  kill "$VISUALIZER_PID" 2>/dev/null || true
  wait "$VISUALIZER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

rviz2 -d "$PROJECT_ROOT/ros2/mission_visualizer.rviz"
