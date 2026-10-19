#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bundle="$task_root/deployment_bundles/nandu1_v1_4_model_24000"
python_bin="$task_root/tron2_mujoco_sim/scripts/nandu1_python.sh"
noetic="$task_root/tron2_rl_deploy_ros/scripts/nandu1_noetic.sh"
if [[ ! -f "$task_root/.nandu1_noetic_ws/devel/setup.bash" ]]; then
  echo "Build ROS first: $noetic build" >&2;exit 2
fi
mkdir -p "$task_root/tron2_mujoco_sim/outputs"
"$noetic" run roslaunch tron2_hw nandu1_mujoco_policy.launch \
  task:=Isaac-NANDU1-Blind-Flat-v1-4 "model_bundle:=$bundle" \
  "telemetry_output:=$task_root/tron2_mujoco_sim/outputs/nandu1_v1_4_tracking.csv" &
ros_pid=$!
trap 'kill "$ros_pid" 2>/dev/null || true; wait "$ros_pid" 2>/dev/null || true' EXIT INT TERM
"$python_bin" "$task_root/tron2_mujoco_sim/simulator.py" \
  --task Isaac-NANDU1-Blind-Flat-v1-4 --model-bundle "$bundle" \
  --ip 127.0.0.1 --command-timeout .1 "$@"
