#!/usr/bin/env bash
# Single-localhost sim2sim launcher; paths are CLI arguments, not environment variables.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$task_root/tron2_mujoco_sim/scripts/nandu1_python.sh"
task=Isaac-NANDU1-Blind-Flat-v1-2
bundle=""
telemetry=""
command_source=external
sim_args=()
while (($#)); do
  case "$1" in
    --task|--model-bundle|--telemetry-output|--command-source)
      if (($#<2)); then echo "Missing value for $1" >&2;exit 2;fi
      case "$1" in --task) task="$2";; --model-bundle) bundle="$2";; --telemetry-output) telemetry="$2";; --command-source) command_source="$2";; esac
      shift 2;;
    --help|-h)
      echo "Usage: run_nandu1.sh --task TASK [--model-bundle PATH] [--telemetry-output PATH] [--command-source external|mouse] [simulator options]"
      echo "Seven legacy NANDU1 tasks plus original network-E/adaptive A-r2/E-r2 and Play aliases; localhost MuJoCo + ROS only. Do not run concurrent instances."
      exit 0;;
    *) sim_args+=("$1");shift;;
  esac
done
case "$command_source" in external|mouse) ;; *) echo "Invalid --command-source: $command_source" >&2;exit 2;; esac
if [[ "$command_source" == mouse ]];then
  for arg in "${sim_args[@]}";do
    if [[ "${arg%%=*}" == --headless ]];then echo "mouse cannot be combined with --headless; use external for collection" >&2;exit 2;fi
  done
fi
task="${task/Flat-Play-v/Flat-v}"
case "$task" in
  Isaac-NANDU1-Blind-Flat-v1-1) name=nandu1_v1_1_model_30000;;
  Isaac-NANDU1-Blind-Flat-v1-2) name=nandu1_v1_2_model_40000;;
  Isaac-NANDU1-Blind-Flat-v1-3) name=nandu1_v1_3_model_30000;;
  Isaac-NANDU1-Blind-Flat-v1-2-1) name=nandu1_v1_2_1_model_29000;;
  Isaac-NANDU1-Blind-Flat-v1-3-1) name=nandu1_v1_3_1_model_30000;;
  Isaac-NANDU1-Blind-Flat-v1-3-1-2) name=nandu1_v1_3_1_2_model_27500;;
  Isaac-NANDU1-Blind-Flat-v1-3-1-2-2) name=nandu1_v1_3_1_2_2_model_29500;;
  Isaac-NANDU1-Blind-Flat-v1-3-1-2-3) name=nandu1_v1_3_1_2_3_model_11000;;
  Isaac-NANDU1-Blind-Flat-v1-3-1-2-5) name=nandu1_v1_3_1_2_5_model_10500;;
  Isaac-NANDU1-Blind-Flat-v1-3-1-2-7) name=nandu1_v1_3_1_2_7_model_10000;;
  Isaac-NANDU1-Blind-Flat-v1-4) name=nandu1_v1_4_model_24000;;
  Isaac-NANDU1-Blind-Flat-v1-4-1) name=nandu1_v1_4_1_model_14000;;
  Isaac-NANDU1-Blind-Flat-v1-3-1-2-6) name=nandu1_v1_3_1_2_6_model_23500;;
  *) echo "Unsupported NANDU1 task: $task" >&2;exit 2;;
esac
if [[ -z "$bundle" ]];then bundle="$task_root/deployment_bundles/$name";fi
if [[ -z "$telemetry" ]];then telemetry="$task_root/tron2_mujoco_sim/outputs/${name%_model_*}_tracking.csv";fi
"$python_bin" "$task_root/tron2_rl_deploy_ros/tron2_controllers/scripts/validate_nandu1_bundle.py" --task "$task" --bundle "$bundle"
if [[ ! -f "$task_root/.nandu1_noetic_ws/devel/setup.bash" ]];then
  echo "Build ROS first: $task_root/tron2_rl_deploy_ros/scripts/nandu1_noetic.sh build" >&2;exit 2
fi
"$python_bin" -c 'import socket; s=socket.socket(); s.settimeout(.2); occupied=s.connect_ex(("127.0.0.1",11311))==0; s.close(); assert not occupied,"ROS master already running; stop the existing sim2sim first"'
exec "$python_bin" "$task_root/tron2_mujoco_sim/nandu1_process.py" --task "$task" --bundle "$bundle" --telemetry "$telemetry" --command-source "$command_source" -- "${sim_args[@]}"
