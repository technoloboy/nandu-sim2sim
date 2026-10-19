"""Stable public dimensions and semantics for NANDU1 blind-flat v1."""

POLICY_OBSERVATION_DIM = 49
HISTORY_LENGTH = 5
HISTORY_DIM = POLICY_OBSERVATION_DIM * HISTORY_LENGTH
ENCODER_OUTPUT_DIM = 7
COMMAND_DIM = 3
ACTOR_INPUT_DIM = ENCODER_OUTPUT_DIM + POLICY_OBSERVATION_DIM + COMMAND_DIM
CRITIC_OBSERVATION_DIM = 108
CRITIC_INPUT_DIM = CRITIC_OBSERVATION_DIM + COMMAND_DIM
ACTION_DIM = 13

COMMAND_ORDER = ("base_yaw_rate_world_z", "up3_vx_base_body", "up3_height_world")
COMMAND_UNITS = ("rad/s", "m/s", "m")
ENCODER_OUTPUT_ORDER = (
    "base_vx_body",
    "base_vy_body",
    "base_vz_body",
    "up3_vx_body",
    "up3_vy_body",
    "up3_vz_body",
    "up3_height_world",
)
