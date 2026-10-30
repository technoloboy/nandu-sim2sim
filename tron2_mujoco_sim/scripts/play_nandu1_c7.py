"""Run frozen C7 R0/R1/R2 model_30000 in MuJoCo, optionally with a mouse window."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from nandu1_c7_sim2sim import C7Simulation, VERSIONS, bundle_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", choices=VERSIONS, required=True)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--stage", "--evaluation-stage", type=int, choices=range(5))
    p.add_argument("--command", type=float, nargs=3, metavar=("YAW", "VX", "HEIGHT"))
    p.add_argument("--schedule", type=Path, help="JSON [{time:seconds, command:[yaw,vx,height]}], starts at 0")
    p.add_argument("--joystick", action="store_true", help="Independent mouse window; one causal filter in frozen processor")
    p.add_argument("--outer-loop", choices=("off", "truth"), default="off",
                   help="R0 only: simulation-truth TCP vx/height correction; not hardware-equivalent")
    p.add_argument("--outer-config", type=Path, help="Validated JSON gains; requires --outer-loop truth")
    p.add_argument("--headless", action="store_true", help="No MuJoCo window; requires a positive duration")
    p.add_argument("--original-scene", action="store_true", help="Show the frozen asset's original plain floor/background")
    p.add_argument("--duration", type=float, default=0., help="Simulation seconds; 0=until GUI closes")
    p.add_argument("--output", type=Path, help="New directory; default outputs/c7_r012/<version>-<time>")
    p.add_argument("--up3-mass", type=float, default=.67148)
    p.add_argument("--rear-mass", type=float, default=0.)
    p.add_argument("--rear-x", type=float, default=-.275)
    args = p.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0 or (args.headless and args.duration == 0):
        p.error("Use --duration > 0 with --headless")
    if args.joystick and (args.headless or args.schedule or args.command):
        p.error("--joystick requires a visible window and cannot be combined with --schedule/--command")
    if args.schedule and args.command:
        p.error("Choose --schedule or --command")
    if args.outer_loop == 'truth' and args.version != 'R0':
        p.error('Truth outer loop is available only for R0')
    if args.outer_config is not None and args.outer_loop != 'truth':
        p.error('--outer-config requires --outer-loop truth')
    output = args.output or ROOT / "outputs/c7_r012" / f"{args.version}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    torch.set_num_threads(1)
    sim = C7Simulation(args.bundle or bundle_for(args.version), args.version, args.stage,
                       up3_mass=args.up3_mass, rear_mass=args.rear_mass, rear_x=args.rear_x,
                       standard_scene=not args.headless and not args.original_scene,
                       outer_loop=args.outer_loop, outer_config=args.outer_config)
    schedule = json.loads(args.schedule.read_text()) if args.schedule else [
        dict(time=0., command=args.command or [0., 0., sim.initial_height])]
    if (not schedule or schedule[0]["time"] != 0
            or any(not math.isfinite(x["time"]) or x["time"] < 0 for x in schedule)
            or any(b["time"] <= a["time"] for a, b in zip(schedule, schedule[1:]))
            or any(np.asarray(x["command"]).shape != (3,) or not np.isfinite(x["command"]).all() for x in schedule)):
        raise ValueError("Schedule must start at 0, increase strictly, and contain finite 3D commands")
    output.mkdir(parents=True)
    viewer = panel = None
    rows, episodes = [], []
    outcome, error = "duration_complete", None
    try:
        if not args.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(sim.physics.model, sim.physics.data)
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135
            viewer.cam.lookat[:] = [0., 0., .7]
        if args.joystick:
            from nandu1_c7_mouse import MousePanel
            panel = MousePanel(args.version, sim.stage, sim.initial_height, outer_mode=args.outer_loop)
        print(f"{sim.metadata['task_name']} | iteration 30000 | stage {sim.stage}", flush=True)
        print(f"Bundle: {sim.directory}\nTrajectory: {output}", flush=True)
        print("Policy49/history490/estimate7 -> Actor86 -> action13; PD200Hz / policy50Hz", flush=True)
        print(f"Outer loop: {args.outer_loop}; original joystick intent preserved; no world position anchor", flush=True)
        tick, segment, episode = 0, 0, 0
        start = time.monotonic()
        while args.duration == 0 or tick*.02 < args.duration-1e-9:
            if viewer and not viewer.is_running():
                outcome = "viewer_closed"
                break
            request = panel.poll() if panel else schedule[segment]["command"]
            if request is None:
                outcome = "mouse_window_closed"
                break
            if panel and panel.reset_requested:
                episodes.append(dict(episode=episode, event="manual_reset", global_time=tick*.02,
                    velocity_projection_events=sim.physics.velocity_projection_events,
                    torque_saturation_events=sim.physics.torque_saturation_events))
                sim.reset()
                panel.reset(sim.initial_height)
                episode += 1
                request = [0., 0., sim.initial_height]
            if not panel:
                while segment+1 < len(schedule) and tick*.02 >= schedule[segment+1]["time"]-1e-9:
                    segment += 1
                request = schedule[segment]["command"]
            row = sim.step(request)
            row.update(global_time=(tick+1)*.02, episode=episode)
            rows.append(row)
            reason = sim.terminal_reason()
            if panel:
                q = row["up3_quaternion"]
                tilt = math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2), -1, 1)))
                panel.update(row, tilt)
            if viewer:
                viewer.cam.lookat[:2] = row["base_origin"][:2]
                viewer.sync()
                delay = start+(tick+1)*.02-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            tick += 1
            if reason:
                outcome = reason
                print(f"Stopped at {tick*.02:.2f}s: {reason}; failed samples retained.", flush=True)
                break
    except KeyboardInterrupt:
        outcome = "keyboard_interrupt"
    except Exception as exc:
        outcome, error = "runtime_error", repr(exc)
        raise
    finally:
        sim.physics.data.xfrc_applied[:] = 0
        if panel:
            panel.close()
        if viewer:
            viewer.close()
        if rows:
            np.savez_compressed(output / "trajectory.npz", **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
        metadata = dict(task=sim.metadata["task_name"], iteration=30000, stage=sim.stage,
            checkpoint_sha256=sim.metadata["checkpoint_sha256"], bundle=str(sim.directory),
            bundle_manifest_sha256=hashlib.sha256((sim.directory/"manifest.json").read_bytes()).hexdigest(),
            steps=len(rows), duration_s=len(rows)*.02, outcome=outcome, error=error,
            clocks_s=dict(policy=.02, pd=.005, physics=float(sim.physics.model.opt.timestep)),
            scene_style=sim.scene_style,
            outer_loop=sim.outer_metadata(),
            source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('nandu1_c7_sim2sim.py', 'nandu1_c7_outer_loop.py', 'scripts/play_nandu1_c7.py', 'nandu1_c7_mouse.py')},
            fixed_payload=sim.physics.payload.parameters, schedule=schedule if not panel else "mouse_requests_in_npz",
            manual_resets=episodes, velocity_projection_events=sim.physics.velocity_projection_events,
            torque_saturation_events=sim.physics.torque_saturation_events,
            counter_scope="Current episode; completed episode counts retained in manual_resets",
            friction_approximation=sim.physics.friction_approximation,
            measurement_timing="estimate/tcp/velocity/attitude _at_command fields at command_time; remaining truth after physics at time",
            note="Nominal clean MuJoCo evaluation. No automatic reset or deletion of failed samples. Not hardware acceptance.")
        (output/"manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
        print(f"{outcome}: {len(rows)} steps, saved {output}", flush=True)


if __name__ == "__main__":
    main()
