"""v1.1 command/reward semantics with fixed TCP tracking and legacy origin reach."""
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from . import nandu1_v1_1_mdp as v11
from . import nandu1_v1_mdp as v1
from . import nandu1_v1_2_tcp_contract as contract


class Nandu1FixedTCPHeightCommand(v11.Nandu1CoordinatedHeightCommand):
    TCP_LOCAL = contract.TCP_LOCAL

    def _invalidate_point_cache(self):
        self._tcp_cache_key = None

    def tcp_state_world(self):
        d = self._robot.data
        key = (self._env.common_step_counter, getattr(d, '_sim_timestamp', None))
        if getattr(self, '_tcp_cache_key', None) != key:
            local = d.body_pos_w.new_tensor(self.TCP_LOCAL).expand(self.num_envs, 3)
            position = d.body_pos_w[:,self._up3_id] + quat_apply(d.body_quat_w[:,self._up3_id], local)
            velocity = d.body_lin_vel_w[:,self._up3_id] + torch.linalg.cross(
                d.body_ang_vel_w[:,self._up3_id], position - d.body_com_pos_w[:,self._up3_id], dim=-1)
            self._tcp_cache = position, velocity
            self._tcp_cache_key = key
        return self._tcp_cache

    def _up3_position_w(self):
        return self.tcp_state_world()[0]

    def _up3_velocity_w(self):
        return self.tcp_state_world()[1]

    def _reach_state_world(self):
        # Do NOT reuse tracking getters: both the old point and the old velocity
        # transport must survive unchanged in the reach reward/course chain.
        d = self._robot.data
        return d.body_pos_w[:,self._up3_id], d.body_lin_vel_w[:,self._up3_id]

    def _kinematic_up3_height(self):
        return v1.kinematic_up3_height(self._env, terminal_origin=self.TCP_LOCAL)

    def reset(self, env_ids=None):
        self._invalidate_point_cache()
        result = super().reset(env_ids)
        self._invalidate_point_cache()
        return result

    def state_dict(self):
        state = super().state_dict()
        state['measurement_tcp_local'] = list(self.TCP_LOCAL)
        return state

    def load_state_dict(self, state):
        if tuple(state.get('measurement_tcp_local', ())) != self.TCP_LOCAL:
            raise ValueError('checkpoint TCP does not match this measurement task')
        super().load_state_dict(state)
        self._invalidate_point_cache()


def tcp_body_linear_velocity(env, command_name=contract.COMMAND_NAME):
    term = env.command_manager.get_term(command_name)
    return quat_apply_inverse(term._robot.data.root_quat_w, term._up3_velocity_w())


def true_tcp_height(env, command_name=contract.COMMAND_NAME):
    return env.command_manager.get_term(command_name)._up3_position_w()[:,2:3]
