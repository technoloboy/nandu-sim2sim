"""Stable dimensions and ordering for the NANDU1 blind-flat task."""

POLICY_OBSERVATION_DIM = 49
HISTORY_LENGTH = 10
HISTORY_DIM = POLICY_OBSERVATION_DIM * HISTORY_LENGTH
ENCODER_OUTPUT_DIM = 6
COMMAND_DIM = 3
ACTOR_INPUT_DIM = ENCODER_OUTPUT_DIM + POLICY_OBSERVATION_DIM + COMMAND_DIM
CRITIC_OBSERVATION_DIM = 107
CRITIC_INPUT_DIM = CRITIC_OBSERVATION_DIM + COMMAND_DIM
ACTION_DIM = 13

COMMAND_ORDER = ("base_yaw_rate", "up3_vx_heading", "up3_vz_world")
ENCODER_OUTPUT_ORDER = (
    "base_vx_heading",
    "base_vy_heading",
    "base_vz_world",
    "up3_vx_heading",
    "up3_vy_heading",
    "up3_vz_world",
)
