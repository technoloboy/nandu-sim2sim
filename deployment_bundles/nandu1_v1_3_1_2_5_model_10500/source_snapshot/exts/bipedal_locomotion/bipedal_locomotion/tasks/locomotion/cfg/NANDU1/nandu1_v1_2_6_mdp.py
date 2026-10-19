"""v1.2 unchanged control chain, with reset-only measurement rotations."""
import torch
from isaaclab.managers import ManagerTermBase
from . import nandu1_v1_1_mdp as parent
from .nandu1_v1_2_6_math import IMUInstallationMeasurements
from .nandu1_v1_2_6_contract import installation_contract


class IMUInstallationReset(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env._nandu_v1_imu_measurements = IMUInstallationMeasurements(env.num_envs, env.device)

    def __call__(self, env, env_ids, command_name):
        term = env.command_manager.get_term(command_name)
        mount = env._nandu_v1_imu_measurements
        mount.reset(env_ids, int(term.stage), randomize=env.cfg.imu_mount_randomization,
                    base_rpy_deg=env.cfg.base_imu_mount_rpy_deg, up3_rpy_deg=env.cfg.up3_imu_mount_rpy_deg)
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        for name in ('base_imu','up3_imu'):
            key = 'IMUInstallation/' + name + '_max_axis_deg'
            if key not in term.metrics:
                term.metrics[key] = torch.zeros(env.num_envs, device=env.device)
            term.metrics[key][ids] = mount.rpy_deg[name][ids].abs().amax(-1)


class Nandu1IMUInstallationHeightCommand(parent.Nandu1CoordinatedHeightCommand):
    """No changes to command sampling, rewards, geometry or course promotion."""
    def state_dict(self):
        state = super().state_dict()
        state['imu_installation_contract'] = installation_contract()
        return state

    def load_state_dict(self, state):
        if state.get('imu_installation_contract') != installation_contract():
            raise ValueError('v1.2.6 checkpoint IMU installation contract mismatch')
        super().load_state_dict(state)

    @torch.inference_mode()
    def restart_after_resume(self):
        """Rebuild episodes at restored stage, without updating/resetting course statistics."""
        env = self._env
        ids = torch.arange(self.num_envs, device=self.device)
        env.scene.reset(ids)
        env.event_manager.apply(mode='reset', env_ids=ids, global_env_step_count=env.common_step_counter)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.)
        env._nandu_v1_measurement_cache = {}
        env.action_manager.reset(ids)
        env.observation_manager.reset(ids)
        self.reset(ids)
        env.event_manager.reset(ids)
        env.episode_length_buf[ids] = 0
        env.obs_buf = env.observation_manager.compute(update_history=True)
