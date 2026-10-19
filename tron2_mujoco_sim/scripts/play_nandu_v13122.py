#!/usr/bin/env python3
"""Run a new-schema v13122 bundle in local MuJoCo at50Hz without ROS."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model-xml", type=Path, required=True, help="Existing audited NANDU1 robot.xml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=1000, help="Bounded number of50Hz control steps")
    parser.add_argument("--stage", type=int, choices=range(5))
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--vx", type=float, default=0.)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--up3-total-mass", type=float, default=.67148)
    parser.add_argument("--rear-mass", type=float, default=0.)
    parser.add_argument("--rear-x", type=float, default=-.275)
    parser.add_argument("--output", type=Path, help="Write tracking.csv/report.json to a new directory")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    import numpy as np
    import torch
    from nandu1_v13122 import AdaptiveRuntime, MujocoPhysics, CONTROL_DT, PHYSICS_DT, TASK
    torch.set_num_threads(1)
    physics = MujocoPhysics(args.model_xml, up3_total_mass=args.up3_total_mass,
                            rear_mass=args.rear_mass, rear_x=args.rear_x)
    initial_height = float(physics.truth()["tcp_position_world"][2])
    runtime = AdaptiveRuntime(args.bundle, initial_height=initial_height, stage=args.stage)
    desired = np.array([args.yaw, args.vx, initial_height if args.height is None else args.height])
    viewer = None
    if not args.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(physics.model, physics.data)
        viewer.cam.distance = 3.
        viewer.cam.elevation = -15.
    stream = None
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        stream = (args.output / "tracking.csv").open("w", newline="")
        writer = csv.writer(stream)
        writer.writerow(["sim_time"] + [f"{name}_{i}" for name, width in
            (("command", 3), ("actual", 3), ("encoder", 7), ("label", 7), ("tcp_position", 3),
             ("tcp_velocity_api", 3), ("tcp_velocity_trajectory", 3), ("memory", 2), ("eta", 8)) for i in range(width)])
    records, base_heights, previous_tcp, fault = [], [], None, ""
    completed = 0
    started = time.monotonic()
    try:
        for step in range(args.steps):
            if viewer is not None and not viewer.is_running():
                break
            observation = physics.observe(runtime.previous_issued_action.cpu().numpy()[0])
            result = runtime.step(observation, desired, step_id=step)
            truth = physics.truth()
            base_heights.append(float(truth["base_origin_world"][2]))
            tcp = truth["tcp_position_world"]
            derivative = np.full(3, np.nan) if previous_tcp is None else (tcp - previous_tcp) / CONTROL_DT
            previous_tcp = tcp.copy()
            row = np.r_[physics.data.time, result["commands"], truth["command_truth"], result["encoded"],
                        truth["encoder_label"], tcp, truth["tcp_velocity_world"], derivative,
                        result["memory"], result["eta"]]
            records.append(row)
            if stream is not None:
                writer.writerow(row)
            target_q, target_dq = physics.decode(result["action"])
            for _ in range(round(CONTROL_DT / PHYSICS_DT)):
                physics.apply(target_q, target_dq)
                physics.step()
            completed += 1
            if not np.isfinite(physics.data.qpos).all() or not np.isfinite(physics.data.qvel).all():
                fault = "nonfinite_physics"
                break
            if viewer is not None:
                viewer.cam.lookat[:] = physics.data.xpos[physics.bids["base_Link"]]
                viewer.sync()
                time.sleep(max(0., started + completed * CONTROL_DT - time.monotonic()))
    except (ValueError, RuntimeError) as error:
        fault = str(error)
    finally:
        if stream is not None:
            stream.close()
        if viewer is not None:
            viewer.close()
    data = np.asarray(records)
    report = dict(task=TASK, checkpoint_schema=runtime.metadata["checkpoint_schema"],
                  evaluation="local untrained-or-trained bundle physics probe; no training-result inference",
                  steps=completed, sim_duration=float(physics.data.time), policy_dt=CONTROL_DT, physics_dt=PHYSICS_DT,
                  fault=fault, payload=physics.payload.parameters,
                  torque_saturation_events=physics.torque_saturations,
                  velocity_projection_events=physics.velocity_projections,
                  target_projection_events=physics.target_projections,
                  initial_height=initial_height, requested_command=desired.tolist())
    if len(data):
        report["tracking_rmse"] = np.sqrt(np.mean((data[:, 1:4] - data[:, 4:7]) ** 2, axis=0)).tolist()
        report["encoder_rmse"] = np.sqrt(np.mean((data[:, 7:14] - data[:, 14:21]) ** 2, axis=0)).tolist()
        report["minimum_base_height"] = min(base_heights)
        report["final_base_height"] = float(physics.data.xpos[physics.bids["base_Link"], 2])
        if len(data) > 1:
            # Independent position differences versus MuJoCo rigid transport.
            report["trajectory_api_velocity_rmse"] = np.sqrt(np.mean((data[1:, 27:30] - data[1:, 24:27]) ** 2, axis=0)).tolist()
    if args.output is not None:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if fault else 0


if __name__ == "__main__":
    raise SystemExit(main())
