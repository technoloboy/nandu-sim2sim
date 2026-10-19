# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import torch
import torch.nn as nn
import torch.optim as optim

from ..modules import ActorCritic, MLP_Encoder
from ..storage import RolloutStorage
from ..modules.nandu_actor_input import build_actor_input


def actor_sensory_gradient_penalty(actor, actor_input, num_commands, input_scales=None, probe=None):
    """Hutchinson estimate of the action-mean sensory Jacobian Frobenius norm.

    The random output probe is detached and fixed during differentiation. If
    x = scales*z, multiplying dmu/dx by scales measures dmu/dz; scales therefore
    express physical/normalized input excursions, not inverse excursions.
    Commands are constants and Encoder/history graphs are detached entirely.
    This is a local smoothness regularizer, not a global Lipschitz bound.
    """
    sensory_dim = actor_input.shape[-1] - num_commands
    if num_commands <= 0 or sensory_dim <= 0:
        raise ValueError("sensory GP requires a positive sensory and command dimension")
    sensory = actor_input[:, :sensory_dim].detach().requires_grad_(True)
    commands = actor_input[:, sensory_dim:].detach()
    mean = actor(torch.cat((sensory, commands), dim=-1))
    if probe is None:
        probe = torch.empty_like(mean).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    if probe.shape != mean.shape:
        raise ValueError("GP probe must have the action mean shape")
    probe = probe.detach() / mean.shape[-1] ** 0.5
    gradient = torch.autograd.grad((mean * probe).sum(), sensory, create_graph=True)[0]
    if input_scales is not None:
        scales = torch.as_tensor(input_scales, device=gradient.device, dtype=gradient.dtype)
        if scales.numel() != sensory_dim or torch.any(scales <= 0):
            raise ValueError("sensory_gp_input_scales must be positive and cover sensory inputs only")
        gradient = gradient * scales
    return gradient.square().sum(dim=-1).mean()


def encoder_supervision_loss(
    encoded: torch.Tensor,
    critic_observations: torch.Tensor,
    target_dim: int = 3,
    target_offset: int = 0,
    target_scales: tuple[float, ...] | list[float] | None = None,
    low_speed_weight: float = 1.0,
    low_speed_threshold: float = 0.05,
    low_speed_bias_coef: float = 0.0,
) -> torch.Tensor:
    """Return the normalized supervised state-estimation loss.

    The legacy locomotion tasks estimate the first three privileged states.
    Keeping those defaults makes the extension backward compatible while new
    tasks can supervise a longer, explicitly ordered target prefix.
    """
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    if encoded.shape[-1] < target_dim:
        raise ValueError(f"encoder output has {encoded.shape[-1]} dims, expected at least {target_dim}")
    stop = target_offset + target_dim
    if critic_observations.shape[-1] < stop:
        raise ValueError(f"critic observation has {critic_observations.shape[-1]} dims, expected at least {stop}")
    error = encoded[:, :target_dim] - critic_observations[:, target_offset:stop]
    if target_scales is not None:
        if len(target_scales) != target_dim:
            raise ValueError("target_scales length must equal target_dim")
        scales = torch.as_tensor(target_scales, dtype=error.dtype, device=error.device)
        error = error / scales.clamp_min(1.0e-6)
    if low_speed_weight == 1.0 and low_speed_bias_coef == 0.0:
        return error.pow(2).mean()
    if target_dim < 6:
        raise ValueError("low-speed supervision requires six true Body velocity targets")
    targets = critic_observations[:, target_offset:stop]
    low_speed = targets[:, :6].norm(dim=-1) < low_speed_threshold
    weights = torch.where(low_speed, low_speed_weight, 1.0)
    loss = (error.square().mean(dim=-1) * weights).mean()
    # A masked reduction keeps empty batches finite without host branching.
    mask = low_speed.to(error.dtype).unsqueeze(-1)
    mean_error = (error[:, :6] * mask).sum(dim=0) / mask.sum().clamp_min(1.0)
    return loss + low_speed_bias_coef * mean_error.square().mean()


def encoder_supervision_metrics(
    encoded: torch.Tensor,
    critic_observations: torch.Tensor,
    target_dim: int,
    target_offset: int = 0,
) -> dict[str, torch.Tensor]:
    """Return raw-unit Encoder RMSE metrics without affecting gradients."""
    stop = target_offset + target_dim
    if encoded.shape[-1] < target_dim or critic_observations.shape[-1] < stop:
        raise ValueError("encoder/critic dimensions do not cover the requested targets")
    error = encoded[:, :target_dim] - critic_observations[:, target_offset:stop]
    metrics = {"total_rmse": error.pow(2).mean().sqrt()}
    if target_dim >= 6:
        metrics["velocity_rmse"] = error[:, :6].pow(2).mean().sqrt()
    if target_dim >= 7:
        metrics["height_rmse"] = error[:, 6].pow(2).mean().sqrt()
    return metrics


class PPO:
    actor_critic: ActorCritic
    encoder: MLP_Encoder

    def __init__(
        self,
        num_group,
        encoder,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        vae_beta=1.0,
        est_learning_rate=1.0e-3,
        est_target_dim=3,
        est_target_offset=0,
        est_target_scales=None,
        est_num_learning_epochs=None,
        est_max_grad_norm=1.0,
        est_low_speed_weight=1.0,
        est_low_speed_threshold=0.05,
        est_low_speed_bias_coef=0.0,
        actor_velocity_frame="body",
        sensory_gp_coef=0.0,
        sensory_gp_warmup_iterations=1000,
        sensory_gp_input_scales=None,
        critic_take_latent=False,
        early_stop=False,
        anneal_lr=False,
        device="cpu",
        **kwargs,
    ):
        self.device = device
        self.num_group = num_group

        self.desired_kl = desired_kl
        self.early_stop = early_stop
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.vae_beta = vae_beta
        self.critic_take_latent = critic_take_latent
        self.est_target_dim = est_target_dim
        self.est_target_offset = est_target_offset
        self.est_target_scales = est_target_scales
        self.est_num_learning_epochs = (
            num_learning_epochs if est_num_learning_epochs is None else est_num_learning_epochs
        )
        self.est_max_grad_norm = est_max_grad_norm
        self.est_low_speed_weight = est_low_speed_weight
        self.est_low_speed_threshold = est_low_speed_threshold
        self.est_low_speed_bias_coef = est_low_speed_bias_coef
        if actor_velocity_frame not in ("body", "heading"):
            raise ValueError("actor_velocity_frame must be body or heading")
        self.actor_velocity_frame = actor_velocity_frame
        if sensory_gp_coef < 0 or sensory_gp_warmup_iterations < 0:
            raise ValueError("GP coefficient and warmup iterations must be nonnegative")
        self.sensory_gp_coef = sensory_gp_coef
        self.sensory_gp_warmup_iterations = sensory_gp_warmup_iterations
        self.sensory_gp_input_scales = sensory_gp_input_scales
        self.learning_iteration = 0
        self.last_smoothness_metrics = {}

        self.encoder = encoder
        self.last_encoder_metrics = {}

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam([{"params": self.actor_critic.parameters()}], lr=learning_rate)

        if self.encoder.num_output_dim != 0:
            self.extra_optimizer = optim.Adam(
                self.encoder.parameters(), lr=est_learning_rate
            )
        else:
            self.extra_optimizer = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    @property
    def sensory_gp_weight(self):
        if self.sensory_gp_warmup_iterations == 0:
            return self.sensory_gp_coef
        return self.sensory_gp_coef * min(1.0, max(0.0, self.learning_iteration / self.sensory_gp_warmup_iterations))

    def act(self, obs, obs_history, commands, critic_obs):
        critic_obs = torch.cat((critic_obs, commands), dim=-1)
        # act
        encoder_out = self.encoder.encode(obs_history)
        self.transition.actions = self.actor_critic.act(
            build_actor_input(encoder_out, obs, commands, self.actor_velocity_frame)
        ).detach()

        # evaluate
        if self.critic_take_latent:
            critic_obs = torch.cat((critic_obs, encoder_out), dim=-1)
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()

        # storage
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        gp_sum = 0.0
        gp_seconds = 0.0
        gp_weight = self.sensory_gp_weight
        generator = self.storage.mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for (
            obs_batch,
            critic_obs_batch,
            obs_history_batch, _,
            group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            encoder_out_batch = self.encoder.encode(obs_history_batch)
            commands_batch = group_commands_batch
            actor_input = build_actor_input(
                encoder_out_batch, obs_batch, commands_batch, self.actor_velocity_frame
            )
            self.actor_critic.act(actor_input)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )

            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.desired_kl != None and self.early_stop:
                if kl_mean > self.desired_kl * 1.5:
                    print("early stop, num_updates =", num_updates)
                    break

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            # print(ratio)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            entropy_batch_mean = entropy_batch.mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
            )
            if self.sensory_gp_coef > 0.0:
                gp_start = time.perf_counter()
                sensory_gp = actor_sensory_gradient_penalty(
                    self.actor_critic.actor, actor_input, commands_batch.shape[-1],
                    self.sensory_gp_input_scales,
                )
                loss = loss + gp_weight * sensory_gp
                gp_sum += sensory_gp.detach().item()
                gp_seconds += time.perf_counter() - gp_start

            if self.anneal_lr:
                frac = 1.0 - num_updates / (
                    self.num_learning_epochs * self.num_mini_batches
                )
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()

        num_updates_extra = 0
        mean_extra_loss = 0
        encoder_metric_sums = {}
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.est_num_learning_epochs
            )
            for (
                next_obs_batch,
                critic_obs_batch,
                obs_history_batch,
            ) in generator:
                if self.encoder.is_mlp_encoder:
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()

                if self.encoder.is_mlp_encoder:
                    extra_loss = encoder_supervision_loss(
                        encode_batch,
                        critic_obs_batch,
                        target_dim=self.est_target_dim,
                        target_offset=self.est_target_offset,
                        target_scales=self.est_target_scales,
                        low_speed_weight=self.est_low_speed_weight,
                        low_speed_threshold=self.est_low_speed_threshold,
                        low_speed_bias_coef=self.est_low_speed_bias_coef,
                    )
                    with torch.no_grad():
                        metrics = encoder_supervision_metrics(
                            encode_batch,
                            critic_obs_batch,
                            target_dim=self.est_target_dim,
                            target_offset=self.est_target_offset,
                        )
                        if self.est_low_speed_weight != 1.0 or self.est_low_speed_bias_coef > 0.0:
                            targets = critic_obs_batch[:, self.est_target_offset:self.est_target_offset + 6]
                            mask = (targets.norm(dim=-1) < self.est_low_speed_threshold).to(encode_batch.dtype).unsqueeze(-1)
                            bias = ((encode_batch[:, :6] - targets) * mask).sum(dim=0) / mask.sum().clamp_min(1.0)
                            metrics["low_speed_bias"] = bias.square().mean().sqrt()
                            metrics["low_speed_fraction"] = mask.mean()
                        for name, value in metrics.items():
                            encoder_metric_sums[name] = encoder_metric_sums.get(name, 0.0) + value.item()
                else:
                    extra_loss = torch.zeros_like(value_loss)

                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.est_max_grad_norm)
                self.extra_optimizer.step()

                num_updates_extra += 1
                mean_extra_loss += extra_loss.item()

        if self.sensory_gp_coef > 0.0:
            self.last_smoothness_metrics = {
                "sensory_gp": gp_sum / max(num_updates, 1),
                "sensory_gp_coefficient": gp_weight,
                "sensory_gp_seconds": gp_seconds,
            }
        mean_value_loss /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra
            self.last_encoder_metrics = {
                name: value / num_updates_extra for name, value in encoder_metric_sums.items()
            }
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        self.storage.clear()

        return (mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_kl)
