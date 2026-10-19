"""v1.2.6 identity and observation-only IMU installation contract."""
TASK = 'Isaac-NANDU1-Blind-Flat-v1-2-6'
PLAY_TASK = 'Isaac-NANDU1-Blind-Flat-Play-v1-2-6'
COMMAND_NAME = 'nandu1_height_v1_1'
SCHEMA = 'nandu1_height_v1_2_6_imu_installation_v1'
EXPERIMENT = 'nandu1_blind_flat_v1_2_6_imu_installation'
DIMENSIONS = dict(policy=49, history=245, encoder=7, commands=3, critic=108, critic_with_commands=111, actions=13)


def installation_contract():
    return dict(version='dual_imu_episode_mount_v1',
                stage_axis_limits_deg=[0., .25, .5, 1., 2.],
                sensors=['base_imu', 'up3_imu'],
                rotation='R_body_sensor=Rz(yaw)Ry(pitch)Rx(roll)',
                observation='R_body_sensor.T @ raw_body_vector; add original step noise; apply original scale',
                lifetime='independent per sensor/env; sampled only on local episode reset',
                checkpoint_scene='fresh episode mounts after restoring course; never transplant old mount/noise cache')
