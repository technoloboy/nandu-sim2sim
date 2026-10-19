"""RSL-RL configuration for the independent NANDU1 blind-flat task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg

from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import EncoderCfg, RslRlPpoAlgorithmMlpCfg


def _nandu1_algorithm_cfg(*, stabilized: bool) -> RslRlPpoAlgorithmMlpCfg:
    """Build an independent optimizer config for the 2x2 ablation matrix."""
    return RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15 if stabilized else 0.20,
        entropy_coef=0.01,
        num_learning_epochs=3 if stabilized else 5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008 if stabilized else 0.010,
        max_grad_norm=1.0,
        obs_history_len=10,
        est_learning_rate=1.0e-4 if stabilized else 3.0e-4,
        est_target_dim=6,
        est_target_offset=0,
        est_target_scales=[0.6, 0.4, 0.3, 0.6, 0.4, 0.3],
        est_num_learning_epochs=1 if stabilized else 2,
        est_max_grad_norm=1.0,
    )


def _nandu1_v1_algorithm_cfg() -> RslRlPpoAlgorithmMlpCfg:
    """PPO plus online 5-frame, 7D velocity-and-height estimator for v1+."""
    return RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.01,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
        obs_history_len=5,
        est_learning_rate=3.0e-4,
        est_target_dim=7,
        est_target_offset=0,
        est_target_scales=[0.6, 0.4, 0.3, 0.6, 0.4, 0.3, 0.05],
        est_num_learning_epochs=2,
        est_max_grad_norm=1.0,
    )


@configclass
class NANDU1BlindFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Production v0 runner: stabilized updater selected by the D ablation."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "nandu1_blind_flat"
    empirical_normalization = False
    interface_contract = {
        "policy": 49,
        "history": 490,
        "encoder": 6,
        "commands": 3,
        "critic": 107,
        "critic_with_commands": 110,
        "actions": 13,
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # The 2x2 ablation selected the D combination: stabilized PPO together
    # with the reduced-redundancy reward set in NANDU1BlindFlatEnvCfg.
    algorithm = _nandu1_algorithm_cfg(stabilized=True)
    encoder = EncoderCfg(
        output_detach=True,
        num_output_dim=6,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
    )


@configclass
class NANDU1BlindFlatV01PPORunnerCfg(NANDU1BlindFlatPPORunnerCfg):
    """v0 network/PPO contract with isolated adaptive-coordination logs."""

    experiment_name = "nandu1_blind_flat_v0_1_adaptive_coordination"


@configclass
class NANDU1BlindFlatV02PPORunnerCfg(NANDU1BlindFlatV01PPORunnerCfg):
    experiment_name = "nandu1_blind_flat_v0_2_robustness"


@configclass
class NANDU1BlindFlatAPPORunnerCfg(NANDU1BlindFlatPPORunnerCfg):
    experiment_name = "nandu1_blind_flat_v0_a"
    algorithm = _nandu1_algorithm_cfg(stabilized=False)


@configclass
class NANDU1BlindFlatBPPORunnerCfg(NANDU1BlindFlatPPORunnerCfg):
    experiment_name = "nandu1_blind_flat_v0_b"
    algorithm = _nandu1_algorithm_cfg(stabilized=True)


@configclass
class NANDU1BlindFlatCPPORunnerCfg(NANDU1BlindFlatPPORunnerCfg):
    experiment_name = "nandu1_blind_flat_v0_c"
    algorithm = _nandu1_algorithm_cfg(stabilized=False)


@configclass
class NANDU1BlindFlatDPPORunnerCfg(NANDU1BlindFlatPPORunnerCfg):
    experiment_name = "nandu1_blind_flat_v0_d"
    algorithm = _nandu1_algorithm_cfg(stabilized=True)


@configclass
class NANDU1BlindFlatV1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "nandu1_blind_flat_v1_height"
    empirical_normalization = False
    checkpoint_schema = "nandu1_height_v1_robust_v2"
    interface_contract = {
        "policy": 49,
        "history": 245,
        "encoder": 7,
        "commands": 3,
        "critic": 108,
        "critic_with_commands": 111,
        "actions": 13,
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = _nandu1_v1_algorithm_cfg()
    encoder = EncoderCfg(
        class_name="MLP_Encoder",
        output_detach=True,
        num_output_dim=7,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
        output_initial_bias=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
    )


@configclass
class NANDU1BlindFlatV11PPORunnerCfg(NANDU1BlindFlatV1PPORunnerCfg):
    """Separate v1-1 experiment; network and optimizer interfaces remain v1-compatible."""

    experiment_name = "nandu1_blind_flat_v1_1_coordination"
    checkpoint_schema = "nandu1_height_v1_1_robust_v2"


@configclass
class NANDU1BlindFlatV12PPORunnerCfg(NANDU1BlindFlatV11PPORunnerCfg):
    """v1.2 uses one history trunk with independent velocity and height heads."""

    experiment_name = "nandu1_blind_flat_v1_2_dual_head"
    checkpoint_schema = "nandu1_height_v1_2_robust_v2"
    encoder = EncoderCfg(
        class_name="DualHeadMLPEncoder",
        output_detach=True,
        num_output_dim=7,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
        output_initial_bias=None,
        velocity_output_dim=6,
        height_output_dim=1,
        height_initial_bias=0.9,
    )


@configclass
class NANDU1BlindFlatV13PPORunnerCfg(NANDU1BlindFlatV12PPORunnerCfg):
    """v1.3 isolates tracking-only promotion while retaining the v1.2 network."""

    experiment_name = "nandu1_blind_flat_v1_3_tracking_only"
    checkpoint_schema = "nandu1_height_v1_3_robust_v2"


@configclass
class NANDU1BlindFlatV131PPORunnerCfg(NANDU1BlindFlatV13PPORunnerCfg):
    experiment_name = "nandu1_blind_flat_v1_3_1_tcp_payload"
    checkpoint_schema = "nandu1_height_v1_3_1_tcp_payload_v1"


def _nandu1_v1_4_algorithm_cfg() -> RslRlPpoAlgorithmMlpCfg:
    cfg = _nandu1_v1_algorithm_cfg()
    cfg.actor_velocity_frame = "heading"
    cfg.est_low_speed_weight = 2.0
    cfg.est_low_speed_threshold = 0.05
    cfg.est_low_speed_bias_coef = 0.05
    cfg.sensory_gp_coef = 0.002
    cfg.sensory_gp_warmup_iterations = 1000
    # Encoder excursions in m/s and m, followed by already-scaled policy
    # observation excursions (angular velocity, gravity, q, dq, action).
    cfg.sensory_gp_input_scales = [0.6, 0.4, 0.3, 0.6, 0.4, 0.3, 0.05] + [1.0] * 49
    return cfg


@configclass
class NANDU1BlindFlatV14PPORunnerCfg(NANDU1BlindFlatV13PPORunnerCfg):
    """Independent Heading actor with Body supervision and sensory smoothing."""

    experiment_name = "nandu1_blind_flat_v1_4_decoupled"
    checkpoint_schema = "nandu1_height_v1_4_decoupled_v1"
    interface_contract = {
        "policy": 49,
        "history": 245,
        "encoder": 7,
        "commands": 6,
        "critic": 108,
        "critic_with_commands": 114,
        "actions": 13,
    }
    algorithm = _nandu1_v1_4_algorithm_cfg()


@configclass
class NANDU1BlindFlatV141PPORunnerCfg(NANDU1BlindFlatV14PPORunnerCfg):
    """Fresh smooth-transition run with unchanged detached dual-head estimation."""
    experiment_name = "nandu1_blind_flat_v1_4_1_smooth_transition"
    checkpoint_schema = "nandu1_height_v1_4_1_smooth_transition_v1"
