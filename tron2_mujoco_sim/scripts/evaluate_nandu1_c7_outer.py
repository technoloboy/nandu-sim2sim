"""Finite, fresh-initial-state outer-off/on paired MuJoCo experiments."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from nandu1_c7_physics import C7Simulation, PHYSICS_PROFILES, bundle_for
from nandu1_c7_outer_loop import OuterLoopConfig


def protocol_for(name, height):
    """Human requests, not pre-compensated targets; zero periods hold height."""
    if not np.isfinite(height):raise ValueError('Finite protocol height required')
    schedule = [(0., [0., 0., height])]
    duration = 60.
    if name in ('vx', 'yaw'):
        axis = 1 if name == 'vx' else 0
        for t, value in zip([8., 20., 26., 38., 44., 56., 62., 74.],
                            [.1, 0., -.1, 0., .3, 0., -.3, 0.] if axis == 1 else
                            [.2, 0., -.2, 0., .6, 0., -.6, 0.]):
            c = [0., 0., height]; c[axis] = value; schedule.append((t, c))
        duration = 84.
    elif name == 'height':
        schedule += [(8., [0., 0., height-.1]), (28., [0., 0., height+.1]), (48., [0., 0., height])]
        duration = 68.
    elif name == 'mixed':
        schedule += [(8., [.2, .1, height+.05]), (24., [-.2, -.1, height-.05]), (40., [0., 0., height])]
    elif name == 'recovery':
        # Deliberately outside request range; frozen reference clipping remains.
        schedule += [(8., [0., 2., 1.8]), (12., [0., 0., height])]
        duration = 32.
    elif name != 'static':raise ValueError('Unknown protocol')
    return dict(name=name, duration_s=duration,
                schedule=[dict(time=t, command=c) for t, c in schedule])


def tracking_metrics(data, settle_s=10.):
    """Use reference+truth sampled together BEFORE that row's action executes."""
    required = ['command_time', 'time', 'reference_command', 'request', 'tcp_at_command', 'tcp',
                'tcp_velocity_body_at_command', 'tcp_velocity_world_at_command', 'yaw_rate_at_command',
                'base_quaternion_at_command', 'outer_correction', 'outer_integral', 'outer_saturated',
                'integral', 'wheel_clipped_pd_steps']
    if not np.isfinite(settle_s) or settle_s < 0:raise ValueError('Invalid settling window')
    arrays = {k:np.asarray(data[k]) for k in required}
    if any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Non-finite trajectory')
    t = arrays['command_time']; end_t = arrays['time']
    if len(t) < 2 or any(len(a) != len(t) for a in arrays.values()):raise ValueError('Inconsistent trajectory')
    if not np.allclose(np.diff(t), .02, atol=1e-8, rtol=0) or not np.allclose(end_t-t, .02, atol=1e-8, rtol=0):
        raise ValueError('Expected aligned pre/post 50Hz clocks')
    mask = t >= settle_s-1e-9
    if not mask.any():raise ValueError('Trajectory shorter than evaluation window')
    indices = np.flatnonzero(mask); first = indices[0]
    reference = arrays['reference_command'][mask]
    body = arrays['tcp_velocity_body_at_command'][mask]
    world = arrays['tcp_velocity_world_at_command'][mask]
    p = arrays['tcp_at_command'][mask]
    yaw = arrays['yaw_rate_at_command'][mask]
    actual = np.column_stack((yaw, body[:, 0], p[:, 2]))
    error = actual-reference
    raw_error = actual-arrays['request'][mask]
    rms = lambda a:np.sqrt(np.mean(np.square(a), axis=0))
    positions = np.vstack((p, arrays['tcp'][-1]))
    position_t = np.r_[t[mask], end_t[-1]]
    delta = positions[-1]-positions[0]
    usable = position_t+20. <= position_t[-1]+1e-8
    drift20 = None
    if usable.any():
        future = np.column_stack([np.interp(position_t[usable]+20., position_t, positions[:, i]) for i in range(2)])
        drift20 = float(np.linalg.norm(future-positions[usable, :2], axis=1).max())
    q = arrays['base_quaternion_at_command']
    if 'base_quaternion' in data:
        endpoint = np.asarray(data['base_quaternion'])[-1:]
        if endpoint.shape != (1, 4) or not np.isfinite(endpoint).all():raise ValueError('Invalid completed endpoint attitude')
        q = np.vstack((q, endpoint))
    tilt = np.arccos(np.clip(1-2*(q[:, 1]**2+q[:, 2]**2), -1, 1))
    result = dict(samples=int(mask.sum()), settle_s=float(settle_s), evaluated_duration_s=float(end_t[-1]-t[first]),
        vx_mean_m_s=float(body[:, 0].mean()), vx_rms_m_s=float(rms(body[:, 0])),
        vx_bias_reference_m_s=float(error[:, 1].mean()), vx_rmse_reference_m_s=float(rms(error[:, 1])),
        height_bias_reference_m=float(error[:, 2].mean()), height_rmse_reference_m=float(rms(error[:, 2])),
        yaw_rate_rmse_reference_rad_s=float(rms(error[:, 0])), yaw_rate_rms_rad_s=float(rms(yaw)),
        rmse_raw_user_request=rms(raw_error).tolist(), world_velocity_rms_m_s=rms(world).tolist(),
        body_vy_rms_m_s=float(rms(body[:, 1])), tcp_net_displacement_m=delta.tolist(),
        xy_path_length_m=float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum()),
        max_20s_xy_drift_m=drift20, max_base_tilt_deg=float(np.degrees(tilt).max()),
        tracking_window=dict(start_s=float(t[first]), end_s=float(end_t[-1])),
        safety_window='full_completed_run_including_startup',
        max_abs_outer_correction=np.abs(arrays['outer_correction']).max(axis=0).tolist(),
        max_abs_outer_integral=np.abs(arrays['outer_integral']).max(axis=0).tolist(),
        max_abs_inner_integral=np.abs(arrays['integral']).max(axis=0).tolist(),
        outer_saturation_fraction=arrays['outer_saturated'].mean(axis=0).tolist(),
        settled_outer_saturation_fraction=arrays['outer_saturated'][mask].mean(axis=0).tolist(),
        wheel_clipped_pd_steps=arrays['wheel_clipped_pd_steps'].sum(axis=0).tolist(),
        wheel_clipped_pd_fraction=(arrays['wheel_clipped_pd_steps'].sum(axis=0)/(len(t)*4)).tolist())
    for source, name in [('outer_safety_override', 'command_safety_override_fraction'),
                         ('outer_command_amplitude_limited', 'command_amplitude_limit_fraction'),
                         ('outer_command_slew_limited', 'command_slew_limit_fraction')]:
        if source in data:
            a = np.asarray(data[source])
            if a.shape != (len(t), 3) or not np.isfinite(a).all():raise ValueError('Invalid command constraint diagnostics')
            result[name] = a.mean(axis=0).tolist()
    return result


def response_metrics(time, reference, actual, target, tolerance, hold_s=2.):
    """Fixed-target response, with None rather than a fabricated settling time."""
    t, ref, y = (np.asarray(a, dtype=float) for a in (time, reference, actual))
    if len(t) < 2 or ref.shape != t.shape or y.shape != t.shape:
        raise ValueError('Response needs matching scalar series')
    if not all(np.isfinite(a).all() for a in (t, ref, y)) or not np.all(np.diff(t) > 0):
        raise ValueError('Invalid response samples')
    if tolerance <= 0 or hold_s <= 0:raise ValueError('Positive response bands required')
    def first_band(values, band, duration):
        for i in np.flatnonzero(np.abs(values-target) <= band):
            end = np.searchsorted(t, t[i]+duration-1e-9)
            if end < len(t) and np.all(np.abs(values[i:end+1]-target) <= band):return float(t[i]-t[0])
        return None
    arrival = first_band(ref, min(tolerance*.2, .001), .1)
    settle = first_band(y, tolerance, hold_s)
    step = target-y[0]
    overshoot = None if abs(step) <= tolerance else float(max(0., (np.sign(step)*(y-target)).max()))
    return dict(reference_arrival_delay_s=arrival, settle_delay_from_event_s=settle,
        settle_delay_from_reference_s=None if settle is None or arrival is None else max(0., settle-arrival),
        overshoot_abs=overshoot, iae_reference=float(np.trapz(np.abs(y-ref), t)),
        tolerance=tolerance, required_hold_s=hold_s)


def segment_metrics(data, schedule):
    """Last5s of each segment after at least3s; fixed windows, not proof of settling."""
    time = data['command_time']; segments = []
    for i, entry in enumerate(schedule):
        end = schedule[i+1]['time'] if i+1 < len(schedule) else float(data['time'][-1])
        start = max(entry['time']+3., end-5.)
        mask = (time >= start-1e-9) & (time < end-1e-9)
        if not mask.any():
            segments.append(dict(request=entry['command'], start_s=entry['time'], end_s=end,
                                 evaluated=False, reason='Insufficient completed samples'))
            continue
        reference = data['reference_command'][mask]
        actual = np.column_stack((data['yaw_rate_at_command'][mask],
                                  data['tcp_velocity_body_at_command'][mask, 0], data['tcp_at_command'][mask, 2]))
        error = actual-reference
        whole = (time >= entry['time']-1e-9) & (time < end-1e-9)
        whole_actual = np.column_stack((data['yaw_rate_at_command'][whole],
                                       data['tcp_velocity_body_at_command'][whole, 0], data['tcp_at_command'][whole, 2]))
        # Requests outside range are scored at the physically admissible target;
        # original requests are still retained separately, never relabelled.
        target = np.clip(entry['command'], [-1.57, -1., .4], [1.57, 1., 1.4])
        response = {axis:response_metrics(time[whole], data['reference_command'][whole, j], whole_actual[:, j],
                    target[j], tolerance) for j, axis, tolerance in
                    ((0, 'yaw_world_z_rad_s', .01), (1, 'vx_base_m_s', .005), (2, 'height_world_m', .01))}
        segments.append(dict(request=entry['command'], start_s=entry['time'], end_s=end, evaluated=True,
            window_start_s=float(time[mask][0]), window_end_s=float(time[mask][-1]),
            mean_actual=actual.mean(axis=0).tolist(), reference_mean=reference.mean(axis=0).tolist(),
            bias_reference=error.mean(axis=0).tolist(), rmse_reference=np.sqrt((error**2).mean(axis=0)).tolist(),
            response=response))
    return segments


def run_case(protocol, mode, config, directory, physics_profile='robojudo'):
    directory.mkdir(parents=True, exist_ok=False)
    sim = C7Simulation(bundle_for('R0'), 'R0', outer_loop=mode,
                       outer_config=config if mode == 'truth' else None, physics_profile=physics_profile)
    rows, segment, outcome, error = [], 0, 'duration_complete', None
    try:
        for step in range(round(protocol['duration_s']/.02)):
            while segment+1 < len(protocol['schedule']) and step*.02 >= protocol['schedule'][segment+1]['time']-1e-9:
                segment += 1
            rows.append(sim.step(protocol['schedule'][segment]['command']))
            reason = sim.terminal_reason()
            if reason:
                outcome = reason; break
    except Exception as exc:
        outcome, error = 'runtime_error', repr(exc)
    data = {k:np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {}
    if data:np.savez_compressed(directory/'trajectory.npz', **data)
    meta = dict(task=sim.metadata['task_name'], checkpoint_sha256=sim.metadata['checkpoint_sha256'],
        bundle_manifest_sha256=hashlib.sha256((sim.directory/'manifest.json').read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('nandu1_c7_sim2sim.py', 'nandu1_c7_physics.py', 'nandu1_c7_outer_loop.py',
             'scripts/evaluate_nandu1_c7_outer.py')},
        protocol=protocol, outer_loop=sim.outer_metadata(), duration_s=len(rows)*.02,
        steps=len(rows), outcome=outcome, error=error,
        velocity_projection_events=sim.physics.velocity_projection_events,
        historical_torque_saturation_events=sim.physics.torque_saturation_events,
        clocks_s=sim.physics_metadata()['clocks_s'], physics=sim.physics_metadata(),
        note='Truth-assisted simulation experiment, not hardware acceptance; failed samples retained')
    if data:
        try:meta['metrics_full'] = tracking_metrics(data, settle_s=0.)
        except ValueError as exc:meta['metrics_full_error'] = str(exc)
        try:
            meta['metrics'] = tracking_metrics(data)
            meta['segments'] = segment_metrics(data, protocol['schedule'])
        except ValueError as exc:meta['metrics_error'] = str(exc)
    (directory/'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='Fresh directory, never overwrite')
    p.add_argument('--protocols', nargs='+', choices=('static', 'vx', 'height', 'yaw', 'mixed', 'recovery'),
                   default=['static', 'vx', 'height', 'yaw', 'mixed', 'recovery'])
    p.add_argument('--heights', type=float, nargs='+', default=[.9])
    p.add_argument('--modes', nargs='+', choices=('off', 'truth'), default=['off', 'truth'])
    p.add_argument('--outer-config', type=Path, default=ROOT/'configs/c7_r0_truth_outer_loop.json')
    p.add_argument('--physics-profile', choices=PHYSICS_PROFILES, default='robojudo')
    args = p.parse_args()
    if args.output.exists():p.error('Output already exists; choose a fresh path')
    if not all(np.isfinite(h) and .4 <= h <= 1.4 for h in args.heights):p.error('Heights must be within .4..1.4m')
    cfg = OuterLoopConfig.from_file(args.outer_config)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    summary = dict(config=asdict(cfg), physics_profile=args.physics_profile, cases={},
                   note='Paired fresh initial states; baseline still includes frozen learned/heading control')
    failed = False
    for height in args.heights:
        for name in args.protocols:
            protocol = protocol_for(name, height)
            for mode in args.modes:
                key = f'{name}_h{height:.2f}_{mode}'
                meta = run_case(protocol, mode, cfg, args.output/key, args.physics_profile)
                summary['cases'][key] = meta
                failed |= meta['outcome'] != 'duration_complete' or 'metrics_error' in meta
                (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
                print(key, meta['outcome'], json.dumps(meta.get('metrics', {})), flush=True)
    if failed:raise SystemExit(1)


if __name__ == '__main__':main()
