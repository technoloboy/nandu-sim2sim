"""Behavioral contract for fixed TCP and immutable composite payloads."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import mujoco

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))


def runtime():
    import nandu1_v131
    return nandu1_v131


def test_profile_does_not_relabel_legacy_policies():
    import nandu1_contract as c
    p = c.task_profile('Isaac-NANDU1-Blind-Flat-Play-v1-3-1')
    assert (p['command'], p['policy_input'], p['velocity_reference']) == (3, 59, 'fixed_tcp')
    assert c.task_profile(c.TASK)['velocity_reference'] == 'body_com'
    assert c.task_profile(c.V14_TASK)['policy_input'] == 62


def test_tcp_transport_uses_com_source_and_link_rotation():
    r = runtime()
    rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1]])
    point, velocity = r.tcp_state([1, 2, 3], rotation, [1, 2, 3], [0, 0, 0], [0, 0, 2])
    np.testing.assert_allclose(point, [0.838376280521, 1.933526948236, 2.999378515098])
    np.testing.assert_allclose(velocity, [0.132946103528, -0.323247438958, 0])


def test_heading_training_runtime_conformance_and_reset():
    r = runtime()
    path = SIM.parent/'tron2_rl_lab/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/NANDU1/nandu1_v1_3_1_math.py'
    if not path.is_file():path=SIM.parent/'deployment_bundles/nandu1_v1_3_1_model_30000/command_processor.py'
    spec = importlib.util.spec_from_file_location('v131_training_reference', path)
    training = importlib.util.module_from_spec(spec); spec.loader.exec_module(training)
    import torch
    ref = training.HeadingController(1, dtype=torch.float64)
    actual = r.CommandProcessor()
    for i in range(220):
        if i == 110:
            ref.reset(); actual.reset()
        gravity = np.array([.45, .1, -.88741195])
        gyro = np.array([.12, -.09, .15 if i < 90 else .4])
        intent = [2 if i < 35 else 0, .3, 1.2]
        expected = ref.step(.02, torch.tensor(gravity[None]), torch.tensor(gyro[None]), intent[0], 3)
        command = actual.step(.02, gravity, gyro, intent, 3)
        np.testing.assert_allclose(command, [expected.item(), .3, 1.2], atol=1e-12)
        np.testing.assert_allclose([actual.heading, actual.heading_target], [ref.heading.item(), ref.heading_target.item()], atol=1e-12)


def test_payload_rebuild_preserves_full_tensor_state_and_unrelated_body():
    r = runtime()
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="base_Link"><freejoint/><inertial mass="15.9685236" pos="-.173386093948 .000013793793 .151659601795" fullinertia=".2608819 .5650821 .4013061 .0073755 -.1217812 .0046664"/>
      <body name="up3" pos="0 0 .4"><joint/><inertial mass=".67148" pos="-.066473051764 .061623719479 -.000621484902" fullinertia=".003411 .0036588 .0035569 -.0000673 .0000689 .0014701"/></body>
      <body name="other"><inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/></body>
    </body></worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    payload = r.FixedPayload(model)
    nominal = [x.copy() for x in (model.body_mass, model.body_ipos, model.body_inertia, model.body_iquat)]
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    payload.apply(data, 6., 5., -.35)
    first = r.full_inertia(model, 1).copy()
    assert abs(model.body_mass[1] - 20.9685236) < 1e-12
    assert abs(model.body_mass[2] - 6.) < 1e-12
    np.testing.assert_allclose(model.body_ipos[2, 1], .1504323861456667, atol=1e-12)
    for array, before in zip((model.body_mass, model.body_ipos, model.body_inertia, model.body_iquat), nominal):
        np.testing.assert_array_equal(array[3], before[3])
    np.testing.assert_array_equal(data.qpos, qpos); np.testing.assert_array_equal(data.qvel, qvel)
    payload.apply(data, 6., 5., -.35)
    np.testing.assert_allclose(r.full_inertia(model, 1), first, atol=1e-14)
    payload.apply(data, .6, 0., -.2)
    np.testing.assert_allclose(r.full_inertia(model, 2), payload.nominal['up3'][2] * (.6/.67148), atol=1e-14)
    payload.apply(data, .67148, 0., -.275)
    for array, before in zip((model.body_mass, model.body_ipos, model.body_inertia, model.body_iquat), nominal):
        np.testing.assert_array_equal(array, before)
    with pytest.raises(ValueError): payload.apply(data, 6.1, 0, -.275)
    np.testing.assert_array_equal(model.body_mass, nominal[0])


def test_checkpoint_same_shapes_old_schema_rejected():
    path = SIM.parent/'tron2_rl_lab/scripts/rsl_rl/nandu1_v131_deployment.py'
    if not path.is_file():pytest.skip('Optional training exporter source is not distributed')
    spec = importlib.util.spec_from_file_location('v131_entry', path)
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    with pytest.raises(ValueError, match='schema'):
        entry.validate_checkpoint({'checkpoint_schema': 'nandu1_height_v1_3_robust_v2', 'task_name': 'Isaac-NANDU1-Blind-Flat-v1-3'})
    with pytest.raises(ValueError, match='stage'):
        entry.validate_checkpoint({'checkpoint_schema': entry.SCHEMA, 'task_name': entry.TRAIN_TASK})


def test_packet_truth_is_tcp_with_raw_com_diagnostics():
    from nandu1_simulator import Nandu1Physics
    import nandu1_contract as c
    # Use existing immutable nominal asset; bypass only bundle IO, exercise real packet.
    path = SIM.parent/'deployment_bundles/nandu1_v1_2_model_40000/assets/robot.xml'
    physics = Nandu1Physics.__new__(Nandu1Physics)
    physics.model = mujoco.MjModel.from_xml_path(str(path)); physics.data = mujoco.MjData(physics.model)
    physics.manifest = {'task': c.task_profile(c.V131_TASK)}
    physics.bids = {n:mujoco.mj_name2id(physics.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('base_Link','up3')}
    jids = [mujoco.mj_name2id(physics.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in c.JOINTS]
    physics.qadr = physics.model.jnt_qposadr[jids]; physics.vadr = physics.model.jnt_dofadr[jids]
    physics.data.qpos[:7] = [0,0,.65337,1,0,0,0]; physics.data.qpos[physics.qadr] = c.DEFAULT_Q
    physics.data.qvel[5] = .7; physics.data.qvel[physics.vadr[-1]] = .4
    physics.epoch=1; physics.velocity_projection_events=0; physics.applied_kp_max=physics.applied_dq_max=0
    physics.imu = lambda _: {}
    mujoco.mj_forward(physics.model, physics.data)
    packet = physics.packet(0)
    bid=physics.bids['up3']; base=physics.bids['base_Link']
    spatial=np.empty(6); mujoco.mj_objectVelocity(physics.model,physics.data,mujoco.mjtObj.mjOBJ_BODY,bid,spatial,0)
    point=physics.data.xpos[bid]+physics.data.xmat[bid].reshape(3,3)@np.array([-.066473051764,.161623719479,-.000621484902])
    expected_velocity=physics.data.xmat[base].reshape(3,3).T@(spatial[3:]+np.cross(spatial[:3],point-physics.data.xipos[bid]))
    np.testing.assert_allclose(packet['truth'][3:6], expected_velocity)
    assert packet['truth'][6] == pytest.approx(point[2])
    assert packet['command_truth'][1:] == pytest.approx([expected_velocity[0],point[2]])
    np.testing.assert_allclose(packet['point_diagnostics']['up3_com_world'],physics.data.xipos[bid])


def test_export_seals_bundle_and_rejects_semantic_tamper(tmp_path):
    if not (SIM.parent/'tron2_rl_lab/scripts/rsl_rl/nandu1_v131_deployment.py').is_file():
        pytest.skip('Optional training exporter/encoder source is not distributed')
    import torch
    import json
    sys.path.insert(0,str(SIM.parent/'tron2_rl_lab/rsl_rl'))
    from rsl_rl.modules.dual_head_mlp_encoder import DualHeadMLPEncoder
    entry_path=SIM.parent/'tron2_rl_lab/scripts/rsl_rl/nandu1_v131_deployment.py'
    spec=importlib.util.spec_from_file_location('v131_export_test',entry_path)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    encoder=DualHeadMLPEncoder(245,7,hidden_dims=[256,128],activation='elu')
    actor=torch.nn.Sequential(torch.nn.Linear(59,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,13))
    checkpoint=tmp_path/'mock.pt'
    torch.save(dict(checkpoint_schema=entry.SCHEMA,task_name=entry.TRAIN_TASK,iter=0,
                    environment_state_dict={entry.COMMAND_NAME:{'stage':4}},encoder_state_dict=encoder.state_dict(),
                    model_state_dict={'actor.'+k:v for k,v in actor.state_dict().items()}),checkpoint)
    result=entry.export_bundle(checkpoint,tmp_path/'bundle',SIM,6.,5.,-.35)
    assert max(result['onnx_max_abs_error'].values())<1e-5
    from nandu1_bundle import load_bundle
    manifest,_=load_bundle(tmp_path/'bundle',entry.PLAY_TASK)
    assert manifest['task']['fixed_payload']['up3_total_mass']==6.
    from nandu1_simulator import Nandu1Physics
    physics=Nandu1Physics(tmp_path/'bundle')
    assert physics.model.body_mass[physics.bids['up3']]==pytest.approx(6)
    physics.apply_fixed_payload(.6,0,-.2);physics.reset()
    assert physics.model.body_mass[physics.bids['up3']]==pytest.approx(.6)
    physics.apply_fixed_payload(.67148,0,-.275)
    np.testing.assert_array_equal(physics.model.body_inertia[physics.bids['base_Link']],physics.fixed_payload.raw['base_Link'][0])
    np.testing.assert_array_equal(physics.model.body_inertia[physics.bids['up3']],physics.fixed_payload.raw['up3'][0])
    np.testing.assert_array_equal(physics.model.body_iquat[physics.bids['up3']],physics.fixed_payload.raw['up3'][1])
    assert physics.model.body_mass[physics.bids['up3']]==physics.fixed_payload.nominal['up3'][0]
    with pytest.raises(ValueError,match='Output exists'): entry.export_bundle(checkpoint,tmp_path/'bundle',SIM)
    manifest['task']['payload_contract']['rear_cylinder_axis']='link_z'
    (tmp_path/'bundle/manifest.yaml').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='v1.3.1'):load_bundle(tmp_path/'bundle')
    manifest['task']['payload_contract']['rear_cylinder_axis']='link_y'
    manifest['task']['tcp_local_m'][1]=.061623719479
    (tmp_path/'bundle/manifest.yaml').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='v1.3.1'): load_bundle(tmp_path/'bundle')
