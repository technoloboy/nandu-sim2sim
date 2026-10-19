#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:-build}"
if ! command -v docker >/dev/null; then
  echo "Docker is not installed. Prepare a working Docker runtime, then rerun this script. No system packages were changed." >&2
  exit 2
fi
if [[ "$mode" == build ]]; then
  build_args=()
  if [[ -n ${NANDU1_BUILD_PROXY:-} ]]; then
    build_args=(--network host --build-arg "http_proxy=$NANDU1_BUILD_PROXY" --build-arg "https_proxy=$NANDU1_BUILD_PROXY" --build-arg "no_proxy=localhost,127.0.0.1,::1")
  fi
  docker build "${build_args[@]}" -t nandu1-noetic -f "$task_root/tron2_rl_deploy_ros/docker/nandu1-noetic.Dockerfile" "$task_root/tron2_rl_deploy_ros/docker"
fi
mkdir -p "$task_root/.nandu1_noetic_ws"
tty_args=()
if [[ "$mode" != build && -t 0 ]]; then tty_args=(-it); fi
container_args=()
if [[ -n ${NANDU1_CONTAINER_NAME:-} ]]; then
  if [[ ! "$NANDU1_CONTAINER_NAME" =~ ^nandu1-[a-zA-Z0-9_-]+$ ]]; then
    echo "Invalid owned NANDU1 container name" >&2
    exit 2
  fi
  container_args=(--name "$NANDU1_CONTAINER_NAME")
fi
exec docker run "${tty_args[@]}" "${container_args[@]}" --rm --network host --ipc host \
  -v "$task_root:$task_root" -v "$task_root/.nandu1_noetic_ws:/ws" \
  -e "NANDU1_ROOT=$task_root" \
  nandu1-noetic bash -lc '
    set -e
    source /opt/ros/noetic/setup.bash
    mkdir -p /ws/src
    ln -sfn "$NANDU1_ROOT/tron2_rl_deploy_ros" /ws/src/deploy
    ln -sfn "$NANDU1_ROOT/tron2_mujoco_sim/limxsdk-lowlevel" /ws/src/limxsdk_lowlevel
    ln -sfn "$NANDU1_ROOT/tron2_mujoco_sim/robot-description" /ws/src/robot_description
    export PYTHONPATH="$NANDU1_ROOT/tron2_mujoco_sim:${PYTHONPATH:-}"
    export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=127.0.0.1
    if [[ "$1" == build ]]; then
      catkin_make -DCATKIN_ENABLE_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
      catkin_make run_tests
      catkin_test_results
      (cd /ws/build && ctest --output-on-failure)
      /ws/devel/lib/tron2_controllers/nandu1_onnx_conformance "$NANDU1_ROOT/deployment_bundles/nandu1_v1_2_model_40000"
      if [[ -f "$NANDU1_ROOT/deployment_bundles/nandu1_v1_4_model_24000/manifest.yaml" ]]; then
        /ws/devel/lib/tron2_controllers/nandu1_onnx_conformance "$NANDU1_ROOT/deployment_bundles/nandu1_v1_4_model_24000"
        /ws/devel/lib/tron2_controllers/nandu1_v14_conformance "$NANDU1_ROOT/deployment_bundles/nandu1_v1_4_model_24000"
      fi
      exit
    fi
    source /ws/devel/setup.bash
    shift
    exec "$@"
  ' _ "$mode" "${@:2}"
