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
import os
from collections import deque
import statistics
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from ..algorithm import PPO
from ..modules import DualHeadHiddenMLPEncoder, DualHeadMLPEncoder, MLP_Encoder, ActorCritic
from ..modules.nandu_actor_input import build_actor_input
from ..env import VecEnv


class OnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        print(f"encoder cfg: {train_cfg.keys()}")
        self.ecd_cfg = train_cfg["encoder"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        obs_dict = self.env.get_observations()
        obs = obs_dict["policy"]
        self.num_obs = obs.shape[1]
        self.obs_history_len = self.alg_cfg.pop("obs_history_len")
        assert "commands" in obs_dict, f"Commands not found in observations"
        self.num_commands = obs_dict["commands"].shape[1]
        assert "critic" in obs_dict, f"Critic observations not found in observations"
        num_critic_obs = obs_dict["critic"].shape[1] + self.num_commands
        privileged_input_size = num_critic_obs
        self.ecd_cfg["num_input_dim"] = self.obs_history_len * self.num_obs

        self._assert_interface_contract(
            policy=self.num_obs,
            history=self.obs_history_len * self.num_obs,
            encoder=self.ecd_cfg["num_output_dim"],
            commands=self.num_commands,
            critic=obs_dict["critic"].shape[1],
            critic_with_commands=num_critic_obs,
            actions=self.env.num_actions,
        )

        encoder_classes = {
            "MLP_Encoder": MLP_Encoder,
            "DualHeadMLPEncoder": DualHeadMLPEncoder,
            "DualHeadHiddenMLPEncoder": DualHeadHiddenMLPEncoder,
        }
        encoder_class_name = self.ecd_cfg.pop("class_name", "MLP_Encoder")
        if encoder_class_name not in encoder_classes:
            raise ValueError(
                f"Unsupported encoder class {encoder_class_name!r}; "
                f"expected one of {sorted(encoder_classes)}"
            )
        encoder_cfg = dict(self.ecd_cfg)
        if encoder_class_name == "MLP_Encoder":
            for key in ("velocity_output_dim", "height_output_dim", "height_initial_bias"):
                encoder_cfg.pop(key, None)
        encoder = encoder_classes[encoder_class_name](**encoder_cfg).to(self.device)

        if train_cfg.get("empirical_normalization", False):
            from ..modules.normalized_actor_critic import (
                EmpiricalNormalizedActorCritic,
            )

            actor_critic_class = EmpiricalNormalizedActorCritic
        else:
            actor_critic_class = ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            self.num_obs
            + encoder.num_output_dim
            + self.num_commands,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        alg_class_name = self.alg_cfg.pop("class_name")
        if alg_class_name == "NormalizedPPO":
            from ..algorithm.normalized_ppo import NormalizedPPO

            alg_class = NormalizedPPO
        else:
            alg_class = eval(alg_class_name)
        self.alg = alg_class(
            self.env.num_envs,
            encoder,
            actor_critic,
            device = self.device,
            **self.alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.num_obs],
            [num_critic_obs],
            [self.obs_history_len * self.num_obs],
            [self.num_commands],
            [self.env.num_actions],
        )

        self.obs_mean = torch.tensor(
            0, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.obs_std = torch.tensor(
            1, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # _, _ = self.env.reset()
        _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Wandb & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "wandb":
                from ..utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "tensorboard":
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs_dict = self.env.get_observations()
        obs = obs_dict["policy"]
        obs_history = obs_dict.get("obsHistory")
        obs_history = obs_history.flatten(start_dim=1)
        critic_obs = obs_dict.get("critic")
        commands = obs_dict.get("commands") 

        obs, obs_history, commands, critic_obs = (
            obs.to(self.device),
            obs_history.to(self.device),
            commands.to(self.device),
            critic_obs.to(self.device),
        )
        # ???
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            self.alg.learning_iteration = it
            self._begin_training_rollout(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, obs_history, commands, critic_obs)
                    # add critic_obs_buf to step returns, make sure it updates in every for loop
                    (obs_dict, rewards, dones, infos) = self.env.step(actions)
                    obs = obs_dict["policy"]

                    critic_obs = obs_dict["critic"]
                    obs_history = obs_dict["obsHistory"].flatten(start_dim=1)
                    commands = obs_dict["commands"]

                    # critic_obs = obs
                    obs, obs_history, commands, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        obs_history.to(self.device),
                        commands.to(self.device),
                        critic_obs.to(self.device), # critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos, obs)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop

                critic_obs_ = torch.cat((critic_obs, commands), dim=-1)
                if self.alg.critic_take_latent:
                    encoder_out = self.alg.encoder.encode(obs_history)
                    self.alg.compute_returns(
                        torch.cat((critic_obs_, encoder_out), dim=-1)
                    )
                else:
                    self.alg.compute_returns(critic_obs_)

            (
                mean_value_loss,
                mean_extra_loss,
                mean_surrogate_loss,
                mean_kl,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                # The update for ``it`` has completed, so resume must start at
                # the next iteration rather than replaying this update.
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)), iteration=it + 1)
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def _begin_training_rollout(self, iteration):
        """Optional task hook; legacy tasks have no boundary command configured."""
        name = self.cfg.get("rollout_boundary_command")
        if name:
            self.env.unwrapped.command_manager.get_term(name).begin_rollout(iteration)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
        # mean_std = self.alg.actor_critic.std.mean()
        mean_std = torch.exp(self.alg.actor_critic.logstd).mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/encoder", locs["mean_extra_loss"], locs["it"])
        for name, value in self.alg.last_encoder_metrics.items():
            self.writer.add_scalar(f"Encoder/{name}", value, locs["it"])
        for name, value in self.alg.last_smoothness_metrics.items():
            self.writer.add_scalar(f"Smoothness/{name}", value, locs["it"])
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Policy/mean_kl", locs["mean_kl"], locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            if (
                self.logger_type != "wandb"
            ):  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.4f}\n"""
                f"""{'Learning rate:':>{pad}} {self.alg.learning_rate:.4f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def _assert_interface_contract(self, **actual):
        expected = self.cfg.get("interface_contract")
        if not expected:
            return
        mismatches = {
            key: (expected[key], actual.get(key))
            for key in expected
            if actual.get(key) != expected[key]
        }
        if mismatches:
            raise ValueError(f"RL interface contract mismatch: {mismatches}")

    def _environment_state_dict(self):
        state = {}
        unwrapped = getattr(self.env, "unwrapped", None)
        manager = getattr(unwrapped, "command_manager", None)
        if manager is None:
            return state
        for name in manager.active_terms:
            term = manager.get_term(name)
            if hasattr(term, "state_dict"):
                state[name] = term.state_dict()
        event_manager = getattr(unwrapped, "event_manager", None)
        event_state = {}
        if event_manager is not None:
            for mode, names in event_manager.active_terms.items():
                cfgs = event_manager._mode_term_cfgs[mode]
                for name, cfg in zip(names, cfgs):
                    if hasattr(cfg.func, "state_dict"):
                        event_state[f"{mode}/{name}"] = cfg.func.state_dict()
        if event_state:
            state["__event_terms__"] = event_state
        return state

    def _load_environment_state_dict(self, state):
        if not state:
            return
        unwrapped = getattr(self.env, "unwrapped", None)
        manager = getattr(unwrapped, "command_manager", None)
        if manager is None:
            return
        for name, term_state in state.items():
            if name in manager.active_terms:
                term = manager.get_term(name)
                if hasattr(term, "load_state_dict"):
                    term.load_state_dict(term_state)
        event_manager = getattr(unwrapped, "event_manager", None)
        event_state = state.get("__event_terms__", {})
        if event_manager is not None and event_state:
            for mode, names in event_manager.active_terms.items():
                cfgs = event_manager._mode_term_cfgs[mode]
                for name, cfg in zip(names, cfgs):
                    key = f"{mode}/{name}"
                    if key in event_state and hasattr(cfg.func, "load_state_dict"):
                        cfg.func.load_state_dict(event_state[key])

    def save(self, path, infos=None, iteration=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "encoder_state_dict": self.alg.encoder.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "extra_optimizer_state_dict": (
                    self.alg.extra_optimizer.state_dict() if self.alg.extra_optimizer is not None else None
                ),
                "environment_state_dict": self._environment_state_dict(),
                "iter": self.current_learning_iteration if iteration is None else iteration,
                "infos": infos,
                "transfer_source": self.cfg.get("transfer_source"),
                "transfer_source_schema": self.cfg.get("transfer_source_schema"),
                "transfer_source_iteration": self.cfg.get("transfer_source_iteration"),
                "checkpoint_schema": self.cfg.get("checkpoint_schema"),
                "task_name": self.cfg.get("task_name"),
                **(
                    {"interface_contract": dict(self.cfg["interface_contract"]),
                     "scheme_b_version": self.cfg["scheme_b_version"]}
                    if self.cfg.get("checkpoint_schema") == "nandu1_height_v1_2_1_adaptive_hold_v1"
                    else {}
                ),
                **(
                    {
                        "interface_contract": dict(self.cfg["interface_contract"]),
                        "smoke_fixture_only": bool(self.cfg.get("smoke_fixture_only", False)),
                    }
                    if self.cfg.get("checkpoint_schema") in (
                        "nandu1_height_v1_3_1_tcp_payload_v1", "nandu1_height_v1_4_1_smooth_transition_v1")
                    else {}
                ),
            },
            path,
        )

    def load(self, path, load_optimizer=False):
        loaded_dict = torch.load(path)
        expected_schema = self.cfg.get("checkpoint_schema")
        actual_schema = loaded_dict.get("checkpoint_schema")
        if expected_schema and actual_schema != expected_schema:
            if expected_schema == "nandu1_height_v1_4_1_smooth_transition_v1":
                raise ValueError(
                    f"v1.4.1 requires checkpoint schema {expected_schema}, got {actual_schema}; "
                    "legacy resume and weight migration are disabled"
                )
            if expected_schema == "nandu1_height_v1_4_decoupled_v1":
                raise ValueError(
                    f"v1.4 requires checkpoint schema {expected_schema}, got {actual_schema}; "
                    "legacy resume and implicit weight migration are disabled"
                )
            raise ValueError(
                f"normal resume requires checkpoint schema {expected_schema}, got {actual_schema}; "
                "use --transfer_checkpoint for a weight-only migration that restarts Stage 0"
            )
        scheme_b = expected_schema == "nandu1_height_v1_2_1_adaptive_hold_v1"
        smooth_transition = expected_schema == "nandu1_height_v1_4_1_smooth_transition_v1"
        imu_installation = expected_schema == "nandu1_height_v1_2_6_imu_installation_v1"
        if imu_installation:
            term = self.env.unwrapped.command_manager.get_term('nandu1_height_v1_1')
            state = (loaded_dict.get('environment_state_dict') or {}).get('nandu1_height_v1_1') or {}
            if (loaded_dict.get('task_name') != self.cfg.get('task_name')
                    or state.get('imu_installation_contract') != term.state_dict()['imu_installation_contract']):
                raise ValueError('v1.2.6 requires matching task and IMU installation contract')
            if not loaded_dict.get('encoder_state_dict'):
                raise ValueError('v1.2.6 requires complete Encoder state')
            if load_optimizer and (not loaded_dict.get('optimizer_state_dict') or (
                    self.alg.extra_optimizer is not None and not loaded_dict.get('extra_optimizer_state_dict'))):
                raise ValueError('v1.2.6 normal resume requires both optimizer states')
        fixed_measurement = expected_schema in (
            "nandu1_height_v1_2_3_fixed_tcp_v1", "nandu1_height_v1_2_4_tcp_hidden_heads_v1",
            "nandu1_height_v1_2_5_long_tcp_v1")
        if fixed_measurement:
            term = self.env.unwrapped.command_manager.get_term("nandu1_height_v1_1")
            state = (loaded_dict.get("environment_state_dict") or {}).get("nandu1_height_v1_1") or {}
            if (loaded_dict.get("task_name") != self.cfg.get("task_name")
                    or tuple(state.get("measurement_tcp_local", ())) != term.TCP_LOCAL):
                raise ValueError("TCP resume requires matching task and measurement point")
            if not loaded_dict.get("encoder_state_dict"):
                raise ValueError("TCP checkpoint requires the complete Encoder state")
            if load_optimizer and (not loaded_dict.get("optimizer_state_dict") or (
                    self.alg.extra_optimizer is not None and not loaded_dict.get("extra_optimizer_state_dict"))):
                raise ValueError("TCP normal resume requires both optimizer states")
        if smooth_transition:
            state = loaded_dict.get("environment_state_dict", {}).get("nandu1_height_v1_4_1", {})
            if (loaded_dict.get("interface_contract") != self.cfg["interface_contract"]
                    or loaded_dict.get("task_name") != self.cfg.get("task_name")
                    or state.get("checkpoint_schema") != expected_schema
                    or state.get("v14_processor", {}).get("schema") != "nandu1_v1_4_1_command_processor_v1"):
                raise ValueError("v1.4.1 resume requires matching task, interface and command state")
            if load_optimizer and (not loaded_dict.get("optimizer_state_dict") or (
                    self.alg.extra_optimizer is not None and not loaded_dict.get("extra_optimizer_state_dict"))):
                raise ValueError("v1.4.1 resume requires both optimizer states")
        if scheme_b and (loaded_dict.get("interface_contract") != self.cfg["interface_contract"]
                         or loaded_dict.get("scheme_b_version") != self.cfg["scheme_b_version"]
                         or "nandu1_height_v1_2_1" not in loaded_dict.get("environment_state_dict", {})):
            raise ValueError("v1.2.1 resume requires matching interface, configuration and task state")
        if scheme_b and load_optimizer:
            if not loaded_dict.get("optimizer_state_dict") or (
                self.alg.extra_optimizer is not None and not loaded_dict.get("extra_optimizer_state_dict")
            ):
                raise ValueError("v1.2.1 normal resume requires both optimizer states; use explicit transfer otherwise")
        if loaded_dict.get("transfer_source"):
            self.cfg["transfer_source"] = loaded_dict["transfer_source"]
            self.cfg["transfer_source_schema"] = loaded_dict.get("transfer_source_schema")
            self.cfg["transfer_source_iteration"] = loaded_dict.get("transfer_source_iteration")
        
        # Check for size mismatch in actor_critic and load selectively
        strict_tcp = expected_schema == "nandu1_height_v1_3_1_tcp_payload_v1" or scheme_b or smooth_transition or fixed_measurement or imu_installation
        if strict_tcp and "encoder_state_dict" not in loaded_dict:
            if smooth_transition:
                raise ValueError("v1.4.1 resume requires the complete dual-head Encoder state")
            raise ValueError("v1.3.1 resume requires the complete dual-head Encoder state")
        model_state_dict = self.alg.actor_critic.state_dict()
        checkpoint_state_dict = loaded_dict["model_state_dict"]
        
        filtered_state_dict = {}
        for k, v in (() if strict_tcp else checkpoint_state_dict.items()):
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"[Warning] Size mismatch for key '{k}': checkpoint shape {v.shape}, model shape {model_state_dict[k].shape}. Skipping this key.")
            else:
                print(f"[Warning] Key '{k}' not found in model. Skipping.")
        
        self.alg.actor_critic.load_state_dict(
            checkpoint_state_dict if strict_tcp else filtered_state_dict, strict=bool(expected_schema)
        )
        
        # Load encoder if it exists in checkpoint
        if "encoder_state_dict" in loaded_dict:
            self.alg.encoder.load_state_dict(loaded_dict["encoder_state_dict"])
            
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            except Exception as e:
                if imu_installation:
                    raise ValueError('v1.2.6 PPO optimizer state could not be restored') from e
                if fixed_measurement:
                    raise ValueError("TCP PPO optimizer state could not be restored") from e
                if smooth_transition:
                    raise ValueError("v1.4.1 optimizer state could not be restored") from e
                if scheme_b or smooth_transition:
                    raise ValueError("v1.2.1 optimizer state could not be restored") from e
                print(f"[Warning] Could not load optimizer state dict: {e}")
        if load_optimizer and self.alg.extra_optimizer is not None and loaded_dict.get("extra_optimizer_state_dict"):
            try:
                self.alg.extra_optimizer.load_state_dict(loaded_dict["extra_optimizer_state_dict"])
            except Exception as e:
                if imu_installation:
                    raise ValueError('v1.2.6 Encoder optimizer state could not be restored') from e
                if fixed_measurement:
                    raise ValueError("TCP Encoder optimizer state could not be restored") from e
                if smooth_transition:
                    raise ValueError("v1.4.1 Encoder optimizer state could not be restored") from e
                if scheme_b or smooth_transition:
                    raise ValueError("v1.2.1 Encoder optimizer state could not be restored") from e
                print(f"[Warning] Could not load encoder optimizer state dict: {e}")

        self._load_environment_state_dict(loaded_dict.get("environment_state_dict"))
        if imu_installation:
            self.env.unwrapped.command_manager.get_term('nandu1_height_v1_1').restart_after_resume()
        if expected_schema == "nandu1_height_v1_3_1_tcp_payload_v1":
            # All event loaders have completed before rebuilding the fresh world.
            self.env.unwrapped.command_manager.get_term("nandu1_height_v1_3_1").restart_after_resume()
        if scheme_b:
            self.env.unwrapped.command_manager.get_term("nandu1_height_v1_2_1").restart_after_resume()
                
        self.current_learning_iteration = loaded_dict["iter"]
        self.alg.learning_iteration = self.current_learning_iteration
        return loaded_dict["infos"]

    def build_actor_input(self, obs, obs_history, commands):
        """Shared deployed input builder; Encoder outputs remain Body-supervised."""
        encoder_out = self.alg.encoder.encode(obs_history)
        return build_actor_input(encoder_out, obs, commands, self.alg.actor_velocity_frame)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_encoder(self, device=None):
        self.alg.encoder.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.encoder.to(device)
        return self.alg.encoder.encode

    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
