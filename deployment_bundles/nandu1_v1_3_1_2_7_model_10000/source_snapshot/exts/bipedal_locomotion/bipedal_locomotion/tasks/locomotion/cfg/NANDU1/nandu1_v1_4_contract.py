"""Breaking v1.4 interface: old checkpoints cannot be resumed implicitly."""
SCHEMA = "nandu1_height_v1_4_decoupled_v1"
COMMAND_NAME = "nandu1_height_v1_4"
INTERFACE = dict(policy=49, history=245, encoder=7, commands=6,
                 critic=108, critic_with_commands=114, actions=13)
COMMAND_UNITS = ("rad/s", "m/s", "m")
COMMAND_FRAMES = ("world_z_angular_velocity", "base_heading_x", "ground_absolute_z")
