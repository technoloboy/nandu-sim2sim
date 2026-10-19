"""Only the two approved static reward differences from network experiment E."""
import torch


def static_orientation_hold(env, command_name):
    """Keep base roll/up3 tilt, but do not recenter base pitch within its envelope."""
    term = env.command_manager.get_term(command_name)
    base_gravity = env.scene.sensors['base_imu'].data.projected_gravity_b
    up3_gravity = env.scene.sensors['up3_imu'].data.projected_gravity_b
    base_roll = torch.atan2(base_gravity[:, 1], -base_gravity[:, 2])
    up3_tilt = torch.acos(torch.clamp(-up3_gravity[:, 2], -1., 1.))
    penalty = torch.square(base_roll / .08) + torch.square(up3_tilt / .08)
    return term.static_gate * torch.clamp(penalty, 0., 20.)


def static_world_yaw_rate(env, command_name, scale=.15, cap=4.):
    """No local-Z/Euler-rate substitute; RewardManager applies weight and dt once."""
    term = env.command_manager.get_term(command_name)
    yaw_rate = env.scene['robot'].data.root_ang_vel_w[:, 2]
    return term.static_gate * torch.clamp(torch.square(yaw_rate / scale), 0., cap)
