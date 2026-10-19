"""Exact original network-E contract; no adaptive Student or simulator imports."""
from pathlib import Path
import nandu1_contract as c

TASK=c.V1312_TASK;PLAY=c.V1312_PLAY;SCHEMA=c.V1312_SCHEMA
RUNTIME='nandu1_network_v1_3_1_2_e_r1'
CHECKPOINT_SHA256='96fbe2f9ab0c347c5877aee7b20ee1db141c27f42b57d84210504e613c3ee2ef'
# Audited fixed-checkpoint deployment graphs (opset15/batch1), independent of
# mutable bundle checksums. A new checkpoint/export requires a new audited pin.
INFERENCE_SHA256={
    'encoder.onnx':'13728443d1a9f6eff7f506be016c1f0985fc0d1b79198d4df7140748ec683a38',
    'policy.onnx':'f50a0e76a85f02f669aeacd197685c732c49720f2d02b33e9defaa46d72b4435',
    'actor_pipeline.onnx':'bee54e0076a83fd45c3b5cffc763292c2cee7add30fb8b39ac351935b60e3c4e',
}
INTERFACE=dict(policy=49,history=245,encoder=7,commands=3,actor_raw=59,actor_features=62,
    critic=108,critic_with_commands=111,actions=13)
OBSERVATION_ORDER=['base_gyro_scaled_0.25:3','base_gravity:3','up3_gyro_scaled_0.25:3',
    'up3_gravity:3','relative_joint_position:11','joint_velocity_scaled_0.05:13','previous_raw_action:13']
HISTORY_ORDER='oldest_to_newest; each frame policy49 then previous_applied_command3 if H; reset command history zero'
RAW_ORDER='Encoder7 resolved in actor velocity frame, policy49, baseline command observation3/6'
EXPERIMENT=dict(family='v1-3-1',variant=2,factors='E',revision=1,velocity_frame='body',
    interface=INTERFACE,history_command_timing='previous_applied_command_producing_current_state',
    reward_changes=[],command_gp=0.,sensory_gp=0.,gp_warmup_iterations=1000,encoder_actor_gradient=False)


def validate_metadata(m):
    if (m.get('task')!=TASK or m.get('checkpoint_schema')!=SCHEMA
        or m.get('checkpoint_sha256')!=CHECKPOINT_SHA256 or m.get('experiment')!=EXPERIMENT
        or m.get('physical_command_units')!=['rad/s','m/s','m']
        or m.get('encoder_output_frame')!='base_body' or m.get('actor_velocity_frame')!='body'
        or m.get('observation_order')!=OBSERVATION_ORDER or m.get('history_order')!=HISTORY_ORDER
        or m.get('raw_actor_input_order')!=RAW_ORDER or m.get('feature_transform_included') is not True
        or m.get('encoder_actor_gradient') is not False
        or m.get('baseline_contract',{}).get('checkpoint_schema')!=c.V131_SCHEMA):
        raise ValueError('Original network-E metadata contract mismatch')
    files=m.get('file_sha256',{})
    required={name+'.'+suffix for name in ('encoder','policy','actor_pipeline') for suffix in ('onnx','pt')}
    if not required<=files.keys():raise ValueError('Missing network-E inference fingerprints')
    for name,digest in files.items():
        if Path(name).name!=name or not isinstance(digest,str) or len(digest)!=64:
            raise ValueError('Invalid network-E fingerprint path/hash')
