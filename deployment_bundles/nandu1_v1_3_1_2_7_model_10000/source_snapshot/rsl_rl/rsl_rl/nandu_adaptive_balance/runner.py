"""Causal adaptive rollout context and independent learning/checkpoint lifecycle."""
from collections import deque
import copy
import hashlib
import os
import time

import torch

from ..modules import DualHeadMLPEncoder, ActorCritic
from ..nandu_experiments.history import CommandHistory
from ..nandu_experiments.runner import ExperimentRunner
from .adaptation import AdaptiveBalanceLearner, build_student_input
from .contract import AdaptiveBalanceExperiment, AdaptiveBalanceActor
from .math import SlowErrorMemory
from .ppo import AdaptiveBalancePPO, build_raw_actor_input
from .update_audit import UpdateAudit


def build_models(experiment=None, seed=42, encoder_cfg=None, policy_cfg=None, device='cpu'):
    experiment = experiment or AdaptiveBalanceExperiment()
    ec = dict(encoder_cfg or {})
    ec.pop('class_name', None)
    ec.update(num_input_dim=245, num_output_dim=7, hidden_dims=[256, 128], output_detach=True)
    pc = dict(policy_cfg or {})
    pc.pop('class_name', None)
    pc.update(actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation='elu')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 1001)
        encoder = DualHeadMLPEncoder(**ec).to(device)
        torch.random.default_generator.manual_seed(seed + 2001)
        actor_critic = ActorCritic(70, 126, 13, **pc).to(device)
        torch.random.default_generator.manual_seed(seed + 2001)
        actor_critic.actor = AdaptiveBalanceActor().to(device)
        if experiment.interface['actor_raw'] == 71:
            # Preserve all A initial weights; a zero-padded observation must
            # not acquire a different fan-in initialization in B–F.
            from .ablations import AdaptiveBalanceAblationActor
            expanded = AdaptiveBalanceAblationActor().to(device)
            weights = actor_critic.actor.state_dict()
            weights['network.0.weight'] = torch.cat((weights['network.0.weight'],
                weights['network.0.weight'].new_zeros(512, 1)), -1)
            expanded.load_state_dict(weights)
            actor_critic.actor = expanded
            previous = actor_critic.critic[0]
            first = torch.nn.Linear(127, 512).to(device)
            with torch.no_grad():
                first.weight.copy_(torch.cat((previous.weight[:, :119],
                    previous.weight.new_zeros(512, 1), previous.weight[:, 119:]), -1))
                first.bias.copy_(previous.bias)
            actor_critic.critic[0] = first
            actor_critic.num_actor_obs = 71
            actor_critic.num_critic_obs = 127
    return encoder, actor_critic


def _validate_optimizer(optimizer, state, name):
    """Adam accepts wrong moment shapes at load; reject them before model mutation."""
    expected = optimizer.state_dict()['param_groups']
    groups = state.get('param_groups', [])
    if len(groups) != len(expected) or not isinstance(state.get('state'), dict):
        raise ValueError(f'Invalid {name} optimizer checkpoint groups')
    identities, parameters = [], []
    for actual, required, live in zip(groups, expected, optimizer.param_groups):
        if len(actual.get('params', [])) != len(required['params']):
            raise ValueError(f'Invalid {name} optimizer checkpoint parameters')
        identities.extend(actual['params'])
        parameters.extend(live['params'])
    if len(identities) != len(set(identities)) or set(state['state']) - set(identities):
        raise ValueError(f'Invalid {name} optimizer checkpoint identities')
    for identity, parameter in zip(identities, parameters):
        for key, value in state['state'].get(identity, {}).items():
            if isinstance(value, torch.Tensor) and (not bool(value.isfinite().all())
                    or (key != 'step' and value.shape != parameter.shape)):
                raise ValueError(f'Invalid {name} optimizer checkpoint tensor')


def encoder_trajectory_diagnostics(encoded, velocity_body, height, valid):
    """Independent position-derivative comparison; absent segments stay undefined."""
    truth = torch.cat((velocity_body, height.reshape(-1, 1)), dim=-1)
    good = valid.reshape(-1).bool() & torch.isfinite(truth).all(-1) & torch.isfinite(encoded).all(-1)
    error = torch.where(good[:, None], encoded.detach() - truth.detach(), torch.zeros_like(encoded))
    count = good.sum()
    rms = (error.square().sum(0) / count.clamp_min(1)).sqrt()
    rms = torch.where(count > 0, rms, torch.full_like(rms, float('nan')))
    names = ('base_vx_rmse', 'base_vy_rmse', 'base_vz_rmse', 'tcp_vx_rmse',
             'tcp_vy_rmse', 'tcp_vz_rmse', 'tcp_height_rmse')
    result = dict(zip(names, rms.unbind()))
    result['velocity_rmse'] = rms[:6].square().mean().sqrt()
    result['valid_fraction'] = good.float().mean()
    return result


class AdaptiveObservationAdapter:
    """Only successful physical steps advance cached, deployable temporal context."""
    def __init__(self, env, runner):
        self.env, self.runner = env, runner
        self.history = CommandHistory(env.num_envs, device=env.device)

    def __getattr__(self, name): return getattr(self.env, name)

    def _adapt(self, observations):
        result = dict(observations)
        if observations['critic'].shape[-1] != 108:
            raise ValueError('Adaptive environment must retain original critic108')
        result['critic'] = torch.cat((observations['critic'], self.runner.runtime_features(),
                                     self.runner._balance_truth), dim=-1)
        return result

    def get_observations(self):
        return self._adapt(self.env.get_observations())

    def reset(self):
        with torch.inference_mode():
            self.env.reset()
            self.history.reset()
            self.runner._clear_runtime()
            self.runner._cache_balance_truth(self.runner._snapshot())
        return self.get_observations()

    def step(self, actions):
        limit = getattr(self.env, 'clip_actions', None)
        issued = actions if limit is None else actions.clamp(-limit, limit)
        issued = issued.detach().clone()
        clipped = (issued != actions).any(-1)
        observations, rewards, dones, infos = self.env.step(issued)
        self.runner.update_runtime(observations, dones, issued, known_clipping=clipped)
        return self._adapt(observations), rewards, dones, infos


class AdaptiveBalanceRunner(ExperimentRunner):
    def __init__(self, env, train_cfg, experiment=None, log_dir=None, device='cpu'):
        self.experiment = experiment or AdaptiveBalanceExperiment()
        if not isinstance(self.experiment, AdaptiveBalanceExperiment):
            raise ValueError('AdaptiveBalanceRunner requires its independent child task')
        self.cfg = copy.deepcopy(train_cfg)
        self.learning_diagnostic = self.cfg.get('learning_diagnostic')
        if self.learning_diagnostic not in (None, 'L0', 'L1', 'L2', 'L3'):
            raise ValueError('Unknown learning diagnostic profile')
        if self.learning_diagnostic is not None and self.experiment.behavior_revision != 1:
            raise ValueError('L0–L3 explain the original failure using legacy r1 behavior only')
        raw_env = env.unwrapped
        command_cfg = getattr(raw_env.command_manager.get_term(self.experiment.command_name), 'cfg', None)
        if command_cfg is not None and (getattr(command_cfg, 'behavior_revision', 1) != self.experiment.behavior_revision
                or command_cfg.checkpoint_schema != self.experiment.schema):
            raise ValueError('Physical command revision/schema does not match the adaptive checkpoint contract')
        reward_manager = getattr(raw_env, 'reward_manager', None)
        if self.experiment.behavior_revision == 2 and reward_manager is not None:
            if not reward_manager.active_terms or reward_manager.active_terms[0] != 'yaw_tracking':
                raise ValueError('r2 requires yaw_tracking as the first enabled reward for pre-reset statistics')
        self.cfg.update(task_name=self.experiment.task, checkpoint_schema=self.experiment.schema,
                        interface_contract=self.experiment.interface, network_experiment=self.experiment.manifest())
        self.ecd_cfg, self.policy_cfg = self.cfg['encoder'], self.cfg['policy']
        self.alg_cfg = copy.deepcopy(self.cfg['algorithm'])
        self.alg_cfg.pop('class_name', None)
        self.alg_cfg.pop('obs_history_len', None)
        self.alg_cfg.update(actor_velocity_frame='body', critic_take_latent=False, sensory_gp_coef=0.)
        self.device = device
        self.num_obs, self.num_commands, self.obs_history_len = 49, 3, 5
        encoder, actor_critic = build_models(self.experiment, self.cfg.get('seed', 42), self.ecd_cfg,
                                           self.policy_cfg, device)
        self.alg = AdaptiveBalancePPO(env.num_envs, encoder, actor_critic, device=device, **self.alg_cfg)
        self.alg.encoder_updates_enabled = self.learning_diagnostic not in ('L1', 'L3')
        self.adaptation_updates_enabled = self.learning_diagnostic not in ('L2', 'L3')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.cfg.get('seed', 42) + 3001)
            self.adaptation = AdaptiveBalanceLearner(self.cfg.get('adaptation'), device=device, num_envs=env.num_envs)
        self.memory = SlowErrorMemory(env.num_envs, device=device)
        dynamic_width = self.experiment.interface['actor_raw'] - 59
        self.height_memory = None
        if dynamic_width == 12:
            from .height_memory import HeightSlowErrorMemory
            self.height_memory = HeightSlowErrorMemory(env.num_envs, device=device,
                enabled=self.experiment.height_memory_enabled)
        self._dynamic = torch.zeros(env.num_envs, dynamic_width, device=device)
        self._balance_truth = torch.zeros(env.num_envs, 4, device=device)
        self._diagnostic_truth = torch.zeros(env.num_envs, 7, device=device)
        self._diagnostic_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self.previous_issued_action = torch.zeros(env.num_envs, 13, device=device)
        self.runtime_steps = 0
        self.last_adaptation_metrics = {}
        self.update_audit = UpdateAudit(max_samples=128)
        self._audit_context = None
        self.last_update_audits = {}
        self.alg.update_phase_hook = self._audit_phase
        self.env = AdaptiveObservationAdapter(env, self)
        observations = self.env.reset()
        for name, width in [('policy', 49), ('critic', self.experiment.interface['critic']), ('commands', 3), ('obsHistory', 245)]:
            if observations[name].flatten(1).shape != (env.num_envs, width):
                raise ValueError(f'{name} violates adaptive balance interface')
        if env.num_actions != 13:
            raise ValueError('Adaptive balance requires 13 issued actions')
        self.num_steps_per_env, self.save_interval = self.cfg['num_steps_per_env'], self.cfg['save_interval']
        self.alg.init_storage(env.num_envs, self.num_steps_per_env, [49],
                              [self.experiment.interface['critic_with_commands']], [245], [3], [13])
        self.log_dir, self.writer = log_dir, None
        self.tot_timesteps, self.tot_time, self.current_learning_iteration = 0, 0., 0

    def _snapshot(self):
        term = self.env.unwrapped.command_manager.get_term(self.experiment.command_name)
        snapshot = term.adaptation_snapshot()
        required = ('teacher_params', 'counterfactual_teacher_params', 'balance_target',
                    'counterfactual_balance_target', 'valid', 'balance_state', 'balance_valid',
                    'raw_intent', 'height_rate')
        if any(key not in snapshot for key in required):
            raise ValueError('Adaptive snapshot is missing a required cached label or command state')
        return snapshot

    @torch.inference_mode()
    def _cache_balance_truth(self, snapshot):
        valid = snapshot['balance_valid'].reshape(-1, 1).to(self.device)
        balance = snapshot['balance_state'].to(self.device)
        self._balance_truth.copy_(torch.cat((balance, valid), dim=-1))
        keys = ('diagnostic_velocity_body', 'diagnostic_height', 'diagnostic_valid')
        if all(key in snapshot for key in keys):
            diagnostic = torch.cat((snapshot['diagnostic_velocity_body'].to(self.device),
                                    snapshot['diagnostic_height'].to(self.device).reshape(-1, 1)), dim=-1)
            self._diagnostic_truth.copy_(diagnostic)
            self._diagnostic_valid.copy_(snapshot['diagnostic_valid'].to(self.device).reshape(-1).bool())
        else:
            self._diagnostic_valid.zero_()

    @torch.inference_mode()
    def _clear_runtime(self):
        self.adaptation.reset()
        self.adaptation.ring.clear()
        self.memory.reset()
        if self.height_memory is not None:
            self.height_memory.reset()
        self._dynamic.zero_()
        self._balance_truth.zero_()
        self._diagnostic_truth.zero_()
        self._diagnostic_valid.zero_()
        self.previous_issued_action.zero_()
        self.runtime_steps = 0
        self._audit_context = None

    def runtime_features(self, observations=None):
        """Idempotent detached copy; never run Student, command, or memory updates."""
        return self._dynamic.detach().clone()

    def build_actor_input(self, obs, obs_history, commands):
        return build_raw_actor_input(self.alg.encoder.encode(obs_history), obs, commands, self.runtime_features())

    @torch.inference_mode()
    def update_runtime(self, observations, dones, issued_actions, known_clipping=None):
        done = dones.flatten().bool().to(self.device)
        policy = observations['policy'].to(self.device)
        commands = observations['commands'].to(self.device)
        history = observations['obsHistory'].flatten(1).to(self.device)
        snapshot = self._snapshot()
        self._cache_balance_truth(snapshot)
        self.adaptation.reset(done.nonzero(as_tuple=False).flatten())
        self.memory.reset(done.nonzero(as_tuple=False).flatten())
        if self.height_memory is not None:
            self.height_memory.reset(done.nonzero(as_tuple=False).flatten())
        issued = issued_actions.to(self.device).detach().clone()
        issued[done] = 0.
        initial_hidden = self.adaptation.hidden.detach().clone()
        initial_eta = self.adaptation.filtered_eta.detach().clone()
        student_features = build_student_input(policy, issued)
        # Autoreset returns the next episode's priming observation. It has no
        # preceding physical interval, so every dynamic channel stays zero.
        valid_student = torch.isfinite(student_features).all(-1) & ~done
        eta = self.adaptation.context_step(torch.nan_to_num(student_features), reset_mask=done)
        self.adaptation.hidden[~valid_student] = initial_hidden[~valid_student]
        self.adaptation.filtered_eta[~valid_student] = initial_eta[~valid_student]
        eta = self.adaptation.filtered_eta.detach().clone()
        self.previous_issued_action.copy_(issued)
        encoded = self.alg.encoder.encode(history).detach()
        gravity = policy[:, 3:6]
        norm = gravity.norm(dim=-1)
        up = -gravity / norm[:, None].clamp_min(1e-6)
        world_yaw = (policy[:, :3] / .25 * up).sum(-1)
        tilt = up[:, 2].clamp(-1., 1.).acos()
        valid_measurement = valid_student & torch.isfinite(encoded).all(-1) & (norm > 1e-6) & ~done
        hdot = snapshot['height_rate'].to(self.device).reshape(-1, 1)
        memory = self.memory.step_errors(commands[:, 1] - encoded[:, 3], commands[:, 0] - world_yaw,
            snapshot['raw_intent'].to(self.device), hdot, valid=valid_measurement, tilt=tilt,
            clipped=False if known_clipping is None else known_clipping,
            dt=float(getattr(self.env, 'step_dt', .02)), step_id=self.runtime_steps + 1)
        dynamic = torch.cat((memory, eta.detach(), hdot / .5), dim=-1)
        if self.height_memory is not None:
            height_memory = self.height_memory.step_errors(commands[:, 2] - encoded[:, 6], hdot,
                valid=valid_measurement, tilt=tilt,
                clipped=False if known_clipping is None else known_clipping,
                dt=float(getattr(self.env, 'step_dt', .02)), step_id=self.runtime_steps + 1)
            dynamic = torch.cat((dynamic, height_memory), dim=-1)
        self._dynamic.copy_(dynamic)
        self._dynamic[done] = 0.
        self.runtime_steps += 1
        # Bound retained diagnostic tensors independently of environment count.
        # These are pure replay entry states, not a second runtime update.
        self._audit_context = {key: value[:128].detach().clone() for key, value in dict(
            policy=policy, history=history, commands=commands, dynamic=self._dynamic,
            student_obs=student_features, hidden=initial_hidden, filtered_eta=initial_eta,
            teacher_params=snapshot['teacher_params'].to(self.device), valid=valid_student).items()}
        self.adaptation.record(dict(student_obs=student_features,
            teacher_params=snapshot['teacher_params'], counterfactual_teacher_params=snapshot['counterfactual_teacher_params'],
            balance_target=snapshot['balance_target'], counterfactual_balance_target=snapshot['counterfactual_balance_target'],
            valid=valid_student, balance_valid=snapshot['balance_valid'], reset=done,
            hidden=initial_hidden, filtered_eta=initial_eta))

    @torch.no_grad()
    def _audit_inference(self, context):
        """Fixed-observation, one-step Student replay; NOT closed-loop performance.

        Hold the recorded memory/integrals/entry hidden state fixed. Updating an
        estimator can affect future memory too; this local audit cannot quantify
        that delayed trajectory effect. Independent physical audits do so.
        """
        encoded = self.alg.encoder(context['history'])
        raw_eta, _, eta = self.adaptation.student.step(context['student_obs'], context['hidden'],
            context['filtered_eta'], self.adaptation.config.dt)
        dynamic = context['dynamic'].clone()
        dynamic[:, 2:10] = eta
        raw = build_raw_actor_input(encoded, context['policy'], context['commands'], dynamic)
        outputs = dict(actor_mean=self.alg.actor_critic.actor(raw), actor_std=self.alg.actor_critic.logstd.exp(),
            encoder=encoded, student_raw=raw_eta, eta=eta,
            teacher=self.adaptation.ema_teacher(context['teacher_params']))
        return {key: (torch.where(context['valid'][:, None], value,
                         torch.full_like(value, float('nan'))) if value.ndim == 2 else value)
                for key, value in outputs.items()}

    def _audit_phase(self, phase, before):
        if self._audit_context is None:
            return
        if before:
            self.update_audit.capture(phase, self._audit_context, self._audit_inference)
        else:
            self.last_update_audits[phase] = self.update_audit.measure(phase, self._audit_inference)

    def save(self, path, infos=None, iteration=None):
        state = dict(model_state_dict=self.alg.actor_critic.state_dict(), encoder_state_dict=self.alg.encoder.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(), extra_optimizer_state_dict=self.alg.extra_optimizer.state_dict(),
            adaptation_state_dict=self.adaptation.checkpoint_state(), environment_state_dict=self._environment_state_dict(),
            iter=self.current_learning_iteration if iteration is None else iteration, infos=infos,
            checkpoint_schema=self.experiment.schema, task_name=self.experiment.task,
            interface_contract=self.experiment.interface, network_experiment=self.experiment.manifest(),
            train_cfg=copy.deepcopy(self.cfg), tot_timesteps=self.tot_timesteps, tot_time=self.tot_time,
            source_revision=self.cfg.get('source_revision'), source_dirty=self.cfg.get('source_dirty'),
            rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        if self.height_memory is not None:
            state['height_memory_state'] = self.height_memory.state_dict()
        temporary = str(path) + '.partial'
        torch.save(state, temporary)
        os.replace(temporary, path)

    def load(self, path, load_optimizer=False, *, diagnostic_seed=False):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.experiment.validate_checkpoint(state, require_optimizer=load_optimizer)
        saved_profile = state.get('train_cfg', {}).get('learning_diagnostic')
        if diagnostic_seed:
            if (not load_optimizer or self.experiment.behavior_revision != 1
                    or self.learning_diagnostic is None or saved_profile is not None):
                raise ValueError('Diagnostic seed must fork an original r1 checkpoint with all optimizers')
        elif load_optimizer and saved_profile != self.learning_diagnostic:
            raise ValueError('Learning diagnostic profile mismatch; use the same profile to resume')
        for key, module in [('model_state_dict', self.alg.actor_critic), ('encoder_state_dict', self.alg.encoder)]:
            current = module.state_dict()
            if current.keys() != state[key].keys() or any(current[name].shape != state[key][name].shape for name in current):
                raise ValueError(f'Adaptive state dimensions mismatch: {key}')
        self.adaptation.validate_checkpoint_state(state['adaptation_state_dict'], restore_temporal=False)
        if load_optimizer:
            _validate_optimizer(self.alg.optimizer, state['optimizer_state_dict'], 'PPO')
            _validate_optimizer(self.alg.extra_optimizer, state['extra_optimizer_state_dict'], 'Encoder')
        self.alg.actor_critic.load_state_dict(state['model_state_dict'], strict=True)
        self.alg.encoder.load_state_dict(state['encoder_state_dict'], strict=True)
        self.adaptation.load_checkpoint_state(state['adaptation_state_dict'], restore_temporal=False)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.alg.extra_optimizer.load_state_dict(state['extra_optimizer_state_dict'])
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]['lr']
            self._load_environment_state_dict(state['environment_state_dict'])
            self._restart_physics_after_resume()
            torch.set_rng_state(state['rng_state'].cpu())
            if state.get('cuda_rng_state') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng_state']])
        self._clear_runtime()
        self.current_learning_iteration = int(state['iter'])
        self.alg.learning_iteration = self.current_learning_iteration
        self.tot_time, self.tot_timesteps = float(state.get('tot_time', 0.)), int(state.get('tot_timesteps', 0))
        if load_optimizer and not diagnostic_seed and state.get('train_cfg', {}).get('diagnostic_seed') is not None:
            self.cfg['diagnostic_seed'] = copy.deepcopy(state['train_cfg']['diagnostic_seed'])
        if diagnostic_seed:
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            self.cfg['diagnostic_seed'] = dict(path=os.path.abspath(path), sha256=digest.hexdigest(),
                iteration=self.current_learning_iteration, schema=state['checkpoint_schema'],
                profile=self.learning_diagnostic, physical_scene='reinitialized_not_restored')
        return state.get('infos')

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        with torch.no_grad():
            metrics = encoder_trajectory_diagnostics(self.alg.encoder.encode(locs['obs_history']).detach(),
                self._diagnostic_truth[:, :6], self._diagnostic_truth[:, 6:7], self._diagnostic_valid)
            values = torch.stack(tuple(metrics.values())).cpu().tolist()
        for name, value in zip(metrics, values):
            self.writer.add_scalar('Diagnostics/Encoder_trajectory_' + name, value, locs['it'])
        if self.height_memory is not None:
            with torch.no_grad():
                memory_values = torch.stack((self.height_memory.z.abs().mean(),
                    (self.height_memory.z.abs() >= .10 - 1e-6).float().mean())).cpu().tolist()
            self.writer.add_scalar('Memory/height_abs_m_s', memory_values[0], locs['it'])
            self.writer.add_scalar('Memory/height_saturation_fraction', memory_values[1], locs['it'])

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            self.logger_type = 'tensorboard'
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs_dict = self.env.get_observations()
        obs, obs_history, commands, critic_obs = [obs_dict[name].flatten(1).to(self.device)
                                                  for name in ('policy', 'obsHistory', 'commands', 'critic')]
        self.alg.actor_critic.train()
        ep_infos, rewbuffer, lenbuffer = [], deque(maxlen=100), deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros_like(cur_reward_sum)
        for it in range(self.current_learning_iteration, self.current_learning_iteration + num_learning_iterations):
            self.alg.learning_iteration = it
            self._begin_training_rollout(it)
            start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, obs_history, commands, critic_obs)
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    obs, obs_history, commands, critic_obs = [obs_dict[name].flatten(1).to(self.device)
                                                              for name in ('policy', 'obsHistory', 'commands', 'critic')]
                    self.alg.process_env_step(rewards.to(self.device), dones.to(self.device), infos, obs)
                    if self.log_dir is not None:
                        if 'episode' in infos: ep_infos.append(infos['episode'])
                        elif 'log' in infos: ep_infos.append(infos['log'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        ids = dones.flatten().bool()
                        rewbuffer.extend(cur_reward_sum[ids].cpu().tolist())
                        lenbuffer.extend(cur_episode_length[ids].cpu().tolist())
                        cur_reward_sum[ids] = 0.
                        cur_episode_length[ids] = 0.
                self.alg.compute_returns(torch.cat((critic_obs, commands), dim=-1))
            collection_time = time.perf_counter() - start
            start = time.perf_counter()
            self.last_update_audits = {}
            self._audit_phase('joint', True)
            mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_kl = self.alg.update()
            self._audit_phase('adaptation', True)
            self.last_adaptation_metrics = self.adaptation.update(optimize=self.adaptation_updates_enabled)
            self._audit_phase('adaptation', False)
            self._audit_phase('joint', False)
            learn_time = time.perf_counter() - start
            self.last_iteration_times = dict(collection=collection_time, learning=learn_time,
                                            total=collection_time + learn_time)
            if self.log_dir is not None:
                # A locals() snapshot captures the previous snapshot and retains
                # every earlier CUDA observation. Pass only this log's inputs.
                self.log({
                    'it': it,
                    'num_learning_iterations': self.current_learning_iteration + num_learning_iterations,
                    'collection_time': collection_time,
                    'learn_time': learn_time,
                    'mean_value_loss': mean_value_loss,
                    'mean_extra_loss': mean_extra_loss,
                    'mean_surrogate_loss': mean_surrogate_loss,
                    'mean_kl': mean_kl,
                    'ep_infos': ep_infos,
                    'rewbuffer': rewbuffer,
                    'lenbuffer': lenbuffer,
                    'obs': obs,
                    'obs_history': obs_history,
                    'commands': commands,
                })
                for name, value in self.last_adaptation_metrics.items():
                    self.writer.add_scalar('Adaptation/' + name, value, it)
                for phase, metrics in self.last_update_audits.items():
                    for name, value in metrics.items():
                        self.writer.add_scalar('UpdateAudit/' + phase + '/' + name, value, it)
                self.writer.add_scalar('Learning/encoder_updates_enabled', int(self.alg.encoder_updates_enabled), it)
                self.writer.add_scalar('Learning/adaptation_updates_enabled', int(self.adaptation_updates_enabled), it)
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f'model_{it}.pt'), iteration=it + 1)
            else:
                self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
                self.tot_time += collection_time + learn_time
            ep_infos.clear()
        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))
