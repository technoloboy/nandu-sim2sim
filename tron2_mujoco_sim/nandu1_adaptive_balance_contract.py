"""Original adaptive A/r2: dynamic11, not height-memory ablation E/r2."""
from nandu1_adaptive_contract import (BUNDLE_SCHEMA,TCP,OBSERVATION_ORDER,HISTORY_ORDER)

TASK='Isaac-NANDU1-Blind-Flat-v1-3-1-2-2'
PLAY='Isaac-NANDU1-Blind-Flat-Play-v1-3-1-2-2'
SCHEMA='nandu_network_v1_3_1_2_2_E_adaptive_balance_r2'
RUNTIME='nandu1_adaptive_balance_r2'
CHECKPOINT_SHA256='ff430c7709b9e9a5786c93fb33ebc1e4f4383fad83506d62de4ef6de6eb0d112'
INTERFACE=dict(policy=49,history=245,encoder=7,commands=3,actor_raw=70,
    actor_features=73,critic=123,critic_with_commands=126,actions=13)
HEIGHT_MEMORY_ENABLED=False
IS_ABLATION=False
EVALUATION_STAGE=4
HEIGHT_RATES=[.05,.10,.20,.35,.50]
DYNAMIC_ORDER=['memory_vx_over_0.10','memory_yaw_over_0.20']+[
    'filtered_eta_'+str(i) for i in range(8)]+['hdot_over_0.5']
RAW_ACTOR_ORDER='Encoder7,policy49,applied_commands3,normalized_memory2,filtered_student8,hdot_over_0.5'
ACTOR_FEATURES_ORDER='original_E62_in_original_order_then_normalized_memory2_filtered_student8_hdot_over_0.5'
# Filled only from an independently checked fixed-checkpoint export, not a
# runtime manifest. Future checkpoint/export changes require another audit.
INFERENCE_SHA256={
    'encoder.onnx':'3dc4dd2a409094c433e0e0cf2cdbd4bef30d9180fc047d5fd57a3d26b2443f82',
    'policy.onnx':'df9c0216da848455c7fbe960b6d202eed41a85623ff1f51031610f352c9171c2',
    'context_step.onnx':'d618f42224ed0abefe887d84aef6a7e94dbe4ee4789ddbed5e0ea95d136a0e4d',
    'actor_pipeline.onnx':'7b109dec2e0ddf09fc6bc5f126c5eca40679f93cacaff3eff51e976c2a3c439f',
}
