"""PPO consumes immutable rollout memory/Student features, never recurrent state."""
import torch
from torch import nn

from ..algorithm.ppo import PPO, encoder_supervision_loss, encoder_supervision_metrics
from ..storage import RolloutStorage


def build_raw_actor_input(encoded, policy, commands, dynamic):
    return torch.cat((encoded.detach(), policy, commands, dynamic.detach()), dim=-1)


class AdaptiveRolloutStorage(RolloutStorage):
    @property
    def used_dynamic(self):
        return self.observations[..., 49:]


class AdaptiveBalancePPO(PPO):
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, obs_history_shape, commands_shape, action_shape):
        if actor_obs_shape != [49] or critic_obs_shape not in ([126], [127]):
            raise ValueError('Adaptive PPO requires policy49 and critic126/127')
        self.dynamic_width = critic_obs_shape[0] - 115
        self.storage = AdaptiveRolloutStorage(num_envs, num_transitions_per_env, [49 + self.dynamic_width],
                                             critic_obs_shape, obs_history_shape,
                                             commands_shape, action_shape, self.device)

    def act(self, obs, obs_history, commands, critic_obs):
        if obs.shape[-1] != 49 or critic_obs.shape[-1] != 112 + self.dynamic_width:
            raise ValueError('Adaptive PPO public observation dimensions do not match its task')
        dynamic = critic_obs[:, 108:108 + self.dynamic_width].detach().clone()
        raw = build_raw_actor_input(self.encoder.encode(obs_history), obs, commands, dynamic)
        critic = torch.cat((critic_obs, commands), dim=-1)
        self.transition.actions = self.actor_critic.act(raw).detach()
        self.transition.values = self.actor_critic.evaluate(critic).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = torch.cat((obs, dynamic), dim=-1).detach().clone()
        self.transition.critic_obs = critic.detach().clone()
        self.transition.observation_history = obs_history.detach().clone()
        self.transition.commands = commands.detach().clone()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        if next_obs is not None:
            next_obs = torch.cat((next_obs, next_obs.new_zeros(next_obs.shape[0], self.dynamic_width)), dim=-1)
        return super().process_env_step(rewards, dones, infos, next_obs)

    def update(self):
        phase = getattr(self, 'update_phase_hook', lambda name, before: None)
        phase('ppo', True)
        totals = dict(value=0., surrogate=0., kl=0.)
        count = 0
        for batch in self.storage.mini_batch_generator(self.num_group, self.num_mini_batches,
                                                       self.num_learning_epochs):
            stored, critic, history, _, commands, actions, old_value, advantage, returns, old_logp, old_mu, old_sigma = batch
            raw = build_raw_actor_input(self.encoder.encode(history), stored[:, :49], commands, stored[:, 49:49 + self.dynamic_width])
            self.actor_critic.act(raw)
            logp = self.actor_critic.get_actions_log_prob(actions)
            value = self.actor_critic.evaluate(critic)
            mu, sigma = self.actor_critic.action_mean, self.actor_critic.action_std
            with torch.no_grad():
                kl = (torch.log(sigma / old_sigma + 1e-5)
                      + (old_sigma.square() + (old_mu - mu).square()) / (2 * sigma.square()) - .5).sum(-1).mean()
                if self.desired_kl is not None and self.schedule == 'adaptive':
                    if kl > 2 * self.desired_kl:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0 < kl < self.desired_kl / 2:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups: group['lr'] = self.learning_rate
            if self.desired_kl is not None and self.early_stop and kl > 1.5 * self.desired_kl:
                break
            ratio = torch.exp(logp - old_logp.squeeze(-1))
            adv = advantage.squeeze(-1)
            surrogate = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)).mean()
            value_error = (value - returns).square()
            if self.use_clipped_value_loss:
                clipped = old_value + (value - old_value).clamp(-self.clip_param, self.clip_param)
                value_error = torch.maximum(value_error, (clipped - returns).square())
            value_loss = value_error.mean()
            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * self.actor_critic.entropy.mean()
            if self.anneal_lr:
                fraction = 1. - count / (self.num_learning_epochs * self.num_mini_batches)
                self.optimizer.param_groups[0]['lr'] = fraction * self.learning_rate
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            for name, tensor in [('value', value_loss), ('surrogate', surrogate), ('kl', kl)]:
                totals[name] += tensor.detach().item()
            count += 1
        phase('ppo', False)
        phase('encoder', True)
        extra_total = 0.
        extra_count = 0
        metric_totals = {}
        if self.extra_optimizer is not None:
            encoder_updates = getattr(self, 'encoder_updates_enabled', True)
            for _, critic, history in self.storage.encoder_mini_batch_generator(self.num_mini_batches, self.est_num_learning_epochs):
                # Consume the same minibatch RNG even when weights are frozen;
                # the L0–L3 intervention must not also change sampling streams.
                with torch.set_grad_enabled(encoder_updates):
                    self.encoder.encode(history)
                    encoded = self.encoder.get_encoder_out()
                    loss = encoder_supervision_loss(encoded, critic, self.est_target_dim, self.est_target_offset,
                        self.est_target_scales, self.est_low_speed_weight, self.est_low_speed_threshold,
                        self.est_low_speed_bias_coef)
                with torch.no_grad():
                    metrics = encoder_supervision_metrics(encoded, critic, self.est_target_dim, self.est_target_offset)
                    for name, value in metrics.items(): metric_totals[name] = metric_totals.get(name, 0.) + value.item()
                self.extra_optimizer.zero_grad(set_to_none=True)
                if encoder_updates:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.encoder.parameters(), self.est_max_grad_norm)
                    self.extra_optimizer.step()
                extra_total += loss.item()
                extra_count += 1
        self.last_encoder_metrics = {name: value / max(1, extra_count) for name, value in metric_totals.items()}
        phase('encoder', False)
        self.storage.clear()
        return (totals['value'] / max(1, count), extra_total / max(1, extra_count),
                totals['surrogate'] / max(1, count), totals['kl'] / max(1, count))
