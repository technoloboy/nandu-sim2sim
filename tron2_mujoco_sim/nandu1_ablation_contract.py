"""Audited downloaded B/D/F identities; not inferred from model width."""
from types import SimpleNamespace
import nandu1_adaptive_contract as base


def _profile(suffix,group,iteration,stage,memory,digest):
    common={key:getattr(base,key) for key in ('BUNDLE_SCHEMA','INTERFACE','HEIGHT_MEMORY','TCP',
        'OBSERVATION_ORDER','DYNAMIC_ORDER','HISTORY_ORDER','RAW_ACTOR_ORDER','ACTOR_FEATURES_ORDER')}
    task='Isaac-NANDU1-Blind-Flat-v1-3-1-2-'+str(suffix)
    return SimpleNamespace(**common,TASK=task,PLAY=task.replace('Flat-v','Flat-Play-v'),
        SCHEMA='nandu_network_v1_3_1_2_'+str(suffix)+'_ablation_'+group+'_r2',
        RUNTIME='nandu1_adaptive_ablation_'+group.lower()+'_r2',IS_ABLATION=True,
        ABLATION_GROUP=group,MODE_SAMPLING='four_modes' if suffix==7 else 'random',
        HEIGHT_MEMORY_ENABLED=memory,HEIGHT_RATES=[.05,.10,.20,.35,.50],EVALUATION_STAGE=stage,
        CHECKPOINT_SHA256=digest,INFERENCE_SHA256={},ROLLOUT_ATOL=1e-4,
        BUNDLE_NAME='nandu1_v1_3_1_2_'+str(suffix)+'_model_'+str(iteration),
        TELEMETRY_LAYOUT='ablation_'+group.lower(),
        LAUNCH='nandu1_ablation_'+group.lower()+'_mujoco_policy.launch')


PROFILES={p.TASK:p for p in (
    _profile(3,'B',11000,1,False,'82ed2157cf031bbaed479fb9abaa38f211816abb4b60b8c0957fc06b0d0f4be3'),
    _profile(5,'D',10500,4,True,'0d6518d83b60980c82bbb0990292706b4111b20a98a9b8cb3dae6b19a597dbbf'),
    _profile(7,'F',10000,2,True,'f935187ecbaa3d950ea612f64cd91a0b482327b6586a3daf7355fb663fb7f3bf'))}

# Independent source pins from fixed-checkpoint 1024-case audited exports.
PROFILES['Isaac-NANDU1-Blind-Flat-v1-3-1-2-3'].INFERENCE_SHA256={
    'encoder.onnx':'906384b2cfbea29e8d352b1807e1eaeadb9ccd4518ce9cf7a73ffaa5fe95da07',
    'policy.onnx':'0fde0d3b9b825618f9ca6ba6c2cd25cb7082fc4e74ed98169c5ec00d83dda0b9',
    'context_step.onnx':'fb0ed4d19b9dbfa86dd84676f0465db9893f98ccbd9361a9966260d0310988a0',
    'actor_pipeline.onnx':'97305fb5fd487016bd43893a5d207a812140dc00fd1ea31f89a47d6f552541c9'}
PROFILES['Isaac-NANDU1-Blind-Flat-v1-3-1-2-5'].INFERENCE_SHA256={
    'encoder.onnx':'ff32801ad650522e2b973f5053b3f483e90cf341aee766df06bd6fd1df98973d',
    'policy.onnx':'454e096dd29b1a847c11a9bca7a0d7162df2a9362703e7f9f28e78ed35e4657a',
    'context_step.onnx':'ca859eaab96d6d2930883ebf7e4267491f552455298c7de440e1af3395d16748',
    'actor_pipeline.onnx':'9a5e1468de6b2e480494256beba6bbed560fe244f4fdc4e5ee7895730372f40c'}
PROFILES['Isaac-NANDU1-Blind-Flat-v1-3-1-2-7'].INFERENCE_SHA256={
    'encoder.onnx':'a63f892e6eafdbb6e5d24f5cd9207821a22725a0fbbacbf4c500f4d07f4b24f7',
    'policy.onnx':'e8eab8766b9e04f9757b5f2bc57a78e1f8fe16ef3609336e9a476900c0270bda',
    'context_step.onnx':'58e58b191af79845af3517185696df3a8eb99825e6af08204075a44815957488',
    'actor_pipeline.onnx':'1a8994773725e8522640732267c53d1809e4c8e4dabaa79b97a4be32652be4a5'}
