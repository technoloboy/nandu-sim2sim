"""CPU-only fixed physical-audit protocol, first-episode metrics, and selection.

Nothing here reads curriculum EMAs or changes/combines learned components.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

REPORT_SCHEMA = 'nandu_adaptive_physical_checkpoint_audit_v1'
PROTOCOL_SCHEMA = 'stage0_fixed_heights_normal_static_v1'
METRIC_NAMES = ('height_rmse_m', 'vx_rmse_m_s', 'yaw_rmse_rad_s')
TRACKING_SCALES = (.05, .10, .15)
SELECTION_CRITERIA = ['minimum_first_episode_failure_rate',
                      'minimum_mean_rmse_normalized_by_height_.05_vx_.10_yaw_.15',
                      'checkpoint_sha256_lexicographic_tiebreak']


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity_hash(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def make_protocol(seed=42, replicas=2, seconds=30., dt=.02, stage=0):
    """Common raw [yaw, vx, absolute TCP height] requests for every checkpoint.

    Heights are held at .78/.91/1.04; NORMAL velocity requests refresh every
    4--8 seconds. Each replica has the same motion trace across all heights.
    STATIC requests have zero motion. The task's causal processor is retained.
    """
    if type(stage) is not int or stage != 0:
        raise ValueError('This audit supports fixed Stage0 only')
    if type(replicas) is not int or replicas < 1:
        raise ValueError('replicas must be a positive integer')
    if not math.isfinite(seconds) or not 0 < seconds <= 30.:
        raise ValueError('seconds must be in (0, 30]')
    if not math.isfinite(dt) or dt <= 0 or not math.isclose(seconds / dt, round(seconds / dt), abs_tol=1e-6):
        raise ValueError('seconds must contain an integer number of positive control steps')
    rng = np.random.default_rng(seed)
    steps = round(seconds / dt)
    motion = np.zeros((steps + 1, replicas, 2), dtype=np.float32)
    for replica in range(replicas):
        start = 0
        while start <= steps:
            stop = min(steps + 1, start + max(1, round(rng.uniform(4., 8.) / dt)))
            motion[start:stop, replica] = [rng.uniform(-.20, .20), rng.uniform(-.08, .08)]
            start = stop
    cases = [(h, mode, replica) for h in (.78, .91, 1.04)
             for mode in (0, 1) for replica in range(replicas)]
    targets = np.zeros((steps + 1, len(cases), 3), dtype=np.float32)
    for index, (height, mode, replica) in enumerate(cases):
        targets[:, index, 2] = height
        if mode == 0:
            targets[:, index, :2] = motion[:, replica]
    return dict(targets=targets, height=np.asarray([x[0] for x in cases]),
                mode=np.asarray([x[1] for x in cases], dtype=np.int64),
                replica=np.asarray([x[2] for x in cases], dtype=np.int64),
                seed=int(seed), stage=stage, seconds=float(seconds), dt=float(dt),
                schema=PROTOCOL_SCHEMA)


def protocol_identity(protocol):
    targets = np.ascontiguousarray(protocol['targets'])
    return dict(schema=PROTOCOL_SCHEMA, seed=protocol['seed'], stage=protocol['stage'],
                seconds=protocol['seconds'], dt=protocol['dt'],
                cases=[dict(height=float(h), mode=('NORMAL', 'STATIC')[int(m)], replica=int(r))
                       for h, m, r in zip(protocol['height'], protocol['mode'], protocol['replica'])],
                target_shape=list(targets.shape), target_dtype=str(targets.dtype),
                targets_sha256=hashlib.sha256(targets.tobytes()).hexdigest(),
                metric_channels=['absolute_tcp_height', 'tcp_velocity_body_x', 'base_world_yaw_rate'],
                tracking_scales=dict(zip(METRIC_NAMES, TRACKING_SCALES)),
                metric_timing='post_physics_pre_autoreset_against_pre_action_processed_command',
                first_episode_only=True, terminal_frame_included=True)


def summarize_frames(actual, command, terminated, truncated, protocol):
    """Score truth and pre-action processed commands in [height, vx, yaw] order.

    The first done frame is included. Later reset frames are excluded. Invalid
    physics is a failure, never a perfect zero error. Window coverage exposes
    missing survivors rather than silently reporting zero late-window error.
    """
    actual, command = np.asarray(actual), np.asarray(command)
    terminated, truncated = np.asarray(terminated, bool), np.asarray(truncated, bool)
    steps, count = round(protocol['seconds'] / protocol['dt']), len(protocol['mode'])
    if actual.shape != (steps, count, 3) or command.shape != actual.shape:
        raise ValueError('Audit frames do not match the full fixed protocol horizon')
    if terminated.shape != (steps, count) or truncated.shape != terminated.shape:
        raise ValueError('Termination frames do not match the protocol population')
    finite = np.isfinite(actual).all(-1) & np.isfinite(command).all(-1)
    done = terminated | truncated | ~finite
    first = np.where(done.any(0), done.argmax(0), steps - 1)
    eligible = np.arange(steps)[:, None] <= first[None, :]
    valid = eligible & finite
    errors = actual - command
    times = (np.arange(steps) + 1) * protocol['dt']
    failed = (terminated & eligible).any(0) | (~finite & eligible).any(0)
    nonfinite = (~finite & eligible).any(0)

    def metrics(mask):
        n = int(mask.sum())
        values = np.sqrt(np.mean(errors[mask] ** 2, axis=0)) if n else [None] * 3
        result = {key: (float(value) if value is not None else None)
                  for key, value in zip(METRIC_NAMES, values)}
        result['normalized_tracking_rmse'] = (float(np.mean(np.asarray(values) / TRACKING_SCALES))
                                               if n else None)
        result['samples'] = n
        return result

    cases = []
    for index in range(count):
        mask = valid.copy()
        mask[:, np.arange(count) != index] = False
        cases.append(dict(case=index, height=float(protocol['height'][index]),
                          mode=('NORMAL', 'STATIC')[int(protocol['mode'][index])],
                          replica=int(protocol['replica'][index]), failed=bool(failed[index]),
                          nonfinite_failure=bool(nonfinite[index]),
                          truncated=bool((truncated[:, index] & eligible[:, index]).any()),
                          first_episode_seconds=float((first[index] + 1) * protocol['dt']),
                          survival_fraction=float((first[index] + 1) / steps), **metrics(mask)))
    aggregate = dict(**metrics(valid), failures=int(failed.sum()), cases=count,
                     failure_rate=float(failed.mean()),
                     mean_first_episode_seconds=float(np.mean((first + 1) * protocol['dt'])),
                     mean_survival_fraction=float(np.mean((first + 1) / steps)),
                     eligible_samples=int(eligible.sum()),
                     valid_sample_fraction=float(valid.sum() / eligible.sum()))
    windows = []
    for start, stop in ((0., 10.), (10., 20.), (20., 30.)):
        in_window = (times > start + 1e-9) & (times <= stop + 1e-9)
        mask = valid & in_window[:, None]
        windows.append(dict(start_seconds=start, end_seconds=stop,
                            available_cases=int(mask.any(0).sum()), total_cases=count,
                            coverage=float(mask.sum() / max(1, int(in_window.sum()) * count)),
                            **metrics(mask)))
    strata = []
    for height in (.78, .91, 1.04):
        for mode in (0, 1):
            selected = np.isclose(protocol['height'], height) & (protocol['mode'] == mode)
            strata.append(dict(height=height, mode=('NORMAL', 'STATIC')[mode],
                               cases=int(selected.sum()), failures=int(failed[selected].sum()),
                               failure_rate=float(failed[selected].mean()),
                               **metrics(valid & selected[None, :])))
    return dict(aggregate=aggregate, per_case=cases, strata=strata, windows=windows)


def validate_complete_checkpoint(path):
    """Validate the original compound state, including all adaptation modules/Adam.

    Temporal replay is validated but is never used as the physical audit scene.
    """
    import torch
    from .adaptation import AdaptiveBalanceLearner
    from .contract import AdaptiveBalanceExperiment, checkpoint_experiment
    from .runner import build_models, _validate_optimizer
    state = torch.load(path, map_location='cpu', weights_only=False)
    experiment = checkpoint_experiment(state)
    if not isinstance(experiment, AdaptiveBalanceExperiment):
        raise ValueError('Physical checkpoint audit supports adaptive A-F only')
    experiment.validate_checkpoint(state, require_optimizer=True)
    config = state.get('train_cfg', {})
    encoder, actor_critic = build_models(experiment, encoder_cfg=config.get('encoder'),
                                        policy_cfg=config.get('policy'), device='cpu')
    for key, module in (('model_state_dict', actor_critic), ('encoder_state_dict', encoder)):
        expected, actual = module.state_dict(), state[key]
        if (set(expected) != set(actual)
                or any(actual[name].shape != value.shape for name, value in expected.items())):
            raise ValueError(f'Compound checkpoint model dimensions/parameters differ: {key}')
    _validate_optimizer(torch.optim.Adam(actor_critic.parameters()), state['optimizer_state_dict'], 'PPO')
    _validate_optimizer(torch.optim.Adam(encoder.parameters()), state['extra_optimizer_state_dict'], 'Encoder')
    adaptation = state['adaptation_state_dict']
    learner = AdaptiveBalanceLearner(adaptation.get('config'), device='cpu')
    learner.validate_checkpoint_state(adaptation, restore_temporal=False)
    for key in ('iter', 'train_cfg', 'rng_state'):
        if key not in state:
            raise ValueError(f'Incomplete compound checkpoint: missing {key}')
    return experiment, state


def select_complete_checkpoint(report_paths, output):
    """Explicit opt-in selection; copy bytes from one whole checkpoint only."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Selection requires a fresh output directory')
    if not report_paths:
        raise ValueError('At least one complete physical audit report is required')
    candidates, comparison_id = [], None
    for report_path in report_paths:
        report_path = Path(report_path).resolve()
        report = json.loads(report_path.read_text())
        if report.get('schema') != REPORT_SCHEMA or report.get('status') != 'complete':
            raise ValueError('Selection requires complete physical audit reports')
        if (report.get('run_purpose') != 'full_30_second_fixed_protocol'
                or report.get('comparison_identity', {}).get('protocol', {}).get('seconds') != 30.):
            raise ValueError('Selection requires the full 30-second protocol; smoke reports cannot select a checkpoint')
        digest = identity_hash(report['comparison_identity'])
        if digest != report.get('comparison_id'):
            raise ValueError('Audit comparison identity hash mismatch')
        if comparison_id is not None and comparison_id != digest:
            raise ValueError('Cannot select across different protocol/seed/condition/source identities')
        comparison_id = digest
        checkpoint = Path(report['checkpoint']['path'])
        if not checkpoint.is_absolute():
            raise ValueError('Audit checkpoint path must be absolute')
        if sha256_file(checkpoint) != report['checkpoint']['sha256']:
            raise ValueError('Checkpoint hash changed after physical audit')
        experiment, state = validate_complete_checkpoint(checkpoint)
        if (report['checkpoint'].get('schema') != experiment.schema
                or report['checkpoint'].get('task') != experiment.task):
            raise ValueError('Report checkpoint identity differs from the original compound state')
        aggregate = report['summary']['aggregate']
        failure, score = aggregate['failure_rate'], aggregate['normalized_tracking_rmse']
        if (not isinstance(failure, (int, float)) or not math.isfinite(failure) or not 0 <= failure <= 1
                or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0):
            raise ValueError('Selection requires finite first-episode physical metrics')
        candidates.append(dict(report=str(report_path), report_sha256=sha256_file(report_path),
                               checkpoint=report['checkpoint'], failure_rate=failure,
                               normalized_tracking_rmse=score, behavior_revision=experiment.behavior_revision,
                               learning_diagnostic=state.get('train_cfg', {}).get('learning_diagnostic')))
    candidates.sort(key=lambda x: (x['failure_rate'], x['normalized_tracking_rmse'], x['checkpoint']['sha256']))
    chosen = candidates[0]
    output.mkdir(parents=True, exist_ok=False)
    destination = output / 'selected_checkpoint.pt'
    shutil.copy2(chosen['checkpoint']['path'], destination)
    if sha256_file(destination) != chosen['checkpoint']['sha256']:
        raise RuntimeError('Copied checkpoint hash differs from the audited original')
    result = dict(schema='nandu_adaptive_complete_checkpoint_selection_v1',
                  comparison_id=comparison_id, criteria=SELECTION_CRITERIA,
                  selected=chosen, candidates=candidates, output=str(destination.resolve()),
                  copy_mode='byte_identical_original_compound_checkpoint')
    (output / 'selection.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
