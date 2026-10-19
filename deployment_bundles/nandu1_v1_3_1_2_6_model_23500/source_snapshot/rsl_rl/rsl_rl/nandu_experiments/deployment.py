"""CPU-only inference/export, including exact feature transforms and command contracts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import uuid
import torch
from torch import nn
from ..modules.nandu_actor_input import build_actor_input
from .contract import Experiment
from .runner import build_models
from ..nandu_adaptive_balance.contract import AdaptiveBalanceExperiment


class Pipeline(nn.Module):
    __constants__=['frame']
    def __init__(self,encoder,actor,frame):
        super().__init__();self.encoder=encoder;self.actor=actor;self.frame=frame
    def forward(self,history:torch.Tensor,observation:torch.Tensor,commands:torch.Tensor)->torch.Tensor:
        raw=build_actor_input(self.encoder(history),observation,commands,self.frame)
        return self.actor(raw)


def load_pipeline(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    exp=Experiment.from_task(state.get('task_name',''))
    if isinstance(exp,AdaptiveBalanceExperiment):
        from ..nandu_adaptive_balance.deployment import load_pipeline as load_adaptive_pipeline
        return load_adaptive_pipeline(path,_state=state)
    exp.validate_checkpoint(state)
    cfg=state['train_cfg']
    encoder,ac=build_models(exp,cfg.get('seed',42),cfg['encoder'],cfg['policy'])
    encoder.load_state_dict(state['encoder_state_dict'],strict=True)
    ac.load_state_dict(state['model_state_dict'],strict=True)
    return exp,state,Pipeline(encoder,ac.actor,exp.velocity_frame).eval()


def _module(path):
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def export_for_play(checkpoint):
    """Publish a verified bundle beside the checkpoint, preserving old exports."""
    checkpoint=Path(checkpoint).resolve()
    identity=torch.load(checkpoint,map_location='meta',weights_only=False)
    if isinstance(Experiment.from_task(identity.get('task_name','')),AdaptiveBalanceExperiment):
        from ..nandu_adaptive_balance.deployment import export_for_play as export_adaptive_for_play
        return export_adaptive_for_play(checkpoint)
    output=checkpoint.parent/'exported'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    required={f'{name}.{suffix}' for name in ('encoder','policy','actor_pipeline')
              for suffix in ('pt','onnx')}
    try:
        metadata=json.loads((output/'deployment_metadata.json').read_text())
        files=metadata['file_sha256']
        if (metadata['checkpoint_sha256']==digest and required.issubset(files)
                and all(Path(name).name==name and
                        hashlib.sha256((output/name).read_bytes()).hexdigest()==expected
                        for name,expected in files.items())):
            return output
    except (OSError,ValueError,KeyError,TypeError):
        pass
    # Validate in a sibling temporary directory before touching an existing bundle.
    with tempfile.TemporaryDirectory(prefix='.play-export-',dir=checkpoint.parent) as staging:
        bundle=Path(staging)/'exported'
        export_bundle(checkpoint,bundle,onnx=True)
        backup=None
        if output.exists():
            backup=output.with_name('exported.backup-'+uuid.uuid4().hex)
            output.rename(backup)
        try:
            bundle.rename(output)
        except BaseException:
            if backup is not None: backup.rename(output)
            raise
    return output


def export_bundle(checkpoint,output,onnx=True):
    import numpy as np
    exp,state,pipeline=load_pipeline(checkpoint)
    if isinstance(exp,AdaptiveBalanceExperiment):
        from ..nandu_adaptive_balance.deployment import export_bundle as export_adaptive_bundle
        return export_adaptive_bundle(checkpoint,output,onnx=onnx,_loaded=(exp,state,pipeline))
    output=Path(output)
    if output.exists(): raise ValueError('Choose a new export directory; existing bundles are not overwritten')
    output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[3]
    mathdir=root/'exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/NANDU1'
    if exp.family=='v1-3-1':
        baseline=_module(mathdir/'nandu1_v1_3_1_contract.py').export_metadata()
        sources=['nandu1_v1_3_1_math.py','nandu1_v1_3_1_contract.py']
    else:
        baseline=_module(root/'scripts/rsl_rl/nandu1_v1_4_1_deployment.py').metadata()
        sources=['nandu1_v1_4_1_math.py','nandu1_v1_4_math.py']
    for name in sources: shutil.copyfile(mathdir/name,output/name)
    shutil.copyfile(Path(__file__).with_name('history.py'),output/'command_history.py')
    generator=torch.Generator().manual_seed(20261010)
    history=torch.randn(8,exp.history_dim,generator=generator)*.1
    obs=torch.randn(8,49,generator=generator)*.1;obs[:,5]=-1.
    cmd=torch.zeros(8,exp.commands);cmd[:,2]=torch.linspace(.4,1.4,8);cmd[:,1]=torch.linspace(-.3,.3,8)
    with torch.no_grad():
        latent=pipeline.encoder(history)
        raw=build_actor_input(latent,obs,cmd,exp.velocity_frame)
        expected=pipeline(history,obs,cmd)
    metrics={}
    for name,model,values,names in (
        ('encoder',pipeline.encoder,(history,),['history']),
        ('policy',pipeline.actor,(raw,),['raw_actor_input']),
        ('actor_pipeline',pipeline,(history,obs,cmd),['history','observation','commands'])):
        traced=torch.jit.trace(model,tuple(x[:1] for x in values))
        traced.save(str(output/f'{name}.pt'))
        with torch.no_grad(): reference=model(*values);actual=torch.jit.load(str(output/f'{name}.pt'))(*values)
        torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-5)
        metrics[name+'_jit_max_error']=float((actual-reference).abs().max())
        if onnx:
            import onnx as onnxlib
            torch.onnx.export(model,tuple(x[:1] for x in values),str(output/f'{name}.onnx'),
                input_names=names,output_names=['output'],opset_version=17,dynamo=False,
                dynamic_axes={key:{0:'batch'} for key in names+['output']})
            onnxlib.checker.check_model(onnxlib.load(str(output/f'{name}.onnx')))
            try:
                import onnxruntime as ort
                session=ort.InferenceSession(str(output/f'{name}.onnx'),providers=['CPUExecutionProvider'])
                metrics['onnx_validation_engine']='onnxruntime'
            except ImportError:
                from onnx.reference import ReferenceEvaluator
                session=ReferenceEvaluator(str(output/f'{name}.onnx'))
                metrics['onnx_validation_engine']='onnx_reference_evaluator'
            actual=session.run(None,{key:value.numpy() for key,value in zip(names,values)})[0]
            np.testing.assert_allclose(actual,reference.numpy(),atol=1e-5,rtol=1e-4)
            metrics[name+'_onnx_max_error']=float(np.abs(actual-reference.numpy()).max())
    np.savez(output/'conformance.npz',history=history.numpy(),observation=obs.numpy(),commands=cmd.numpy(),actions=expected.numpy())
    metadata=dict(task=exp.task,checkpoint_schema=exp.schema,experiment=exp.manifest(),baseline_contract=baseline,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        physical_command_units=['rad/s','m/s','m'],encoder_output_frame='base_body',actor_velocity_frame=exp.velocity_frame,
        observation_order=['base_gyro_scaled_0.25:3','base_gravity:3','up3_gyro_scaled_0.25:3','up3_gravity:3',
                           'relative_joint_position:11','joint_velocity_scaled_0.05:13','previous_raw_action:13'],
        history_order='oldest_to_newest; each frame policy49 then previous_applied_command3 if H; reset command history zero',
        raw_actor_input_order='Encoder7 resolved in actor velocity frame, policy49, baseline command observation3/6',
        feature_transform_included=True,encoder_actor_gradient=False,
        external_command_processing='Use copied baseline processor before network; raw joystick input is not policy command input',
        runtime_support='Network artifacts verified; existing MuJoCo/ROS loaders need explicit new-schema support',
        validation=metrics)
    metadata['file_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'deployment_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    return metadata
