#!/usr/bin/env python3
"""Fresh, explicit original-E deployment; never relabel a plain TCP network."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SIM=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SIM))
import nandu1_contract as base
import nandu1_network_e as c
from nandu1_bundle import load_bundle,sha256


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def export(args):
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    torch.set_num_threads(1)
    if args.task not in (c.TASK,c.PLAY):raise ValueError('Original network-E task required')
    checkpoint=args.checkpoint.resolve();output=args.output.resolve();lab=args.training_root.resolve()
    if sha256(checkpoint)!=c.CHECKPOINT_SHA256:raise ValueError('Pinned audited 27500 checkpoint required')
    if output.exists():raise ValueError('Choose a fresh output; old bundles are never overwritten')
    provided=args.provided_export.resolve();metadata=json.loads((provided/'deployment_metadata.json').read_text())
    c.validate_metadata(metadata)
    if metadata['checkpoint_sha256']!=c.CHECKPOINT_SHA256:raise ValueError('Provided exports differ from checkpoint')
    for name,digest in metadata['file_sha256'].items():
        if sha256(provided/name)!=digest:raise ValueError('Provided export hash mismatch: '+name)
    sys.path.insert(0,str(lab/'rsl_rl'))
    from rsl_rl.nandu_experiments.deployment import load_pipeline,export_bundle
    exp,state,pipeline=load_pipeline(checkpoint);exp.validate_checkpoint(state,require_optimizer=True)
    if exp.task!=c.TASK or exp.schema!=c.SCHEMA or state['train_cfg'].get('clip_actions','missing') is not None:
        raise ValueError('Wrong original-E checkpoint/action contract')
    stage=int(state['environment_state_dict'][exp.command_name]['stage'])
    if stage!=4:raise ValueError('This deployment is audited for Stage4')
    baseline,_=load_bundle(args.asset_bundle)
    if baseline['task']['task_id']!=base.V131_TASK:raise ValueError('Audited TCP asset source required')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.v1312-export-',dir=output.parent) as temp:
        root=Path(temp)/'bundle';meta=export_bundle(checkpoint,root,onnx=True)
        shutil.copytree(args.asset_bundle/'assets',root/'assets')
        original=root/'original_onnx_opset17';original.mkdir()
        generator=torch.Generator().manual_seed(20261018)
        history=torch.randn(1024,245,generator=generator)*.2
        obs=torch.randn(1024,49,generator=generator)*.2;obs[:,5]=-1;obs[:,11]=-1;obs[0]=0
        commands=torch.rand(1024,3,generator=generator)*torch.tensor([3.14,2.,1.])+torch.tensor([-1.57,-1.,.4])
        with torch.inference_mode():
            encoded=pipeline.encoder(history);raw=torch.cat((encoded,obs,commands),dim=-1)
            actions=pipeline(history,obs,commands)
        vectors=dict(history=history.numpy(),observation=obs.numpy(),commands=commands.numpy(),
            encoder_output_body=encoded.numpy(),raw_actor_input=raw.numpy(),actions=actions.numpy())
        metrics={}
        for name,model,keys,expected in [('encoder',pipeline.encoder,['history'],encoded.numpy()),
            ('policy',pipeline.actor,['raw_actor_input'],actions.numpy()),
            ('actor_pipeline',pipeline,['history','observation','commands'],actions.numpy())]:
            shutil.copyfile(root/(name+'.onnx'),original/(name+'.onnx'))
            values=tuple(torch.from_numpy(vectors[k][:1]) for k in keys)
            torch.onnx.export(model,values,str(root/(name+'.onnx')),input_names=keys,output_names=['output'],
                opset_version=15,dynamo=False)
            onnx.checker.check_model(onnx.load(str(root/(name+'.onnx'))))
            options=ort.SessionOptions();options.intra_op_num_threads=1
            source=ort.InferenceSession(str(provided/(name+'.onnx')),options,providers=['CPUExecutionProvider'])
            reference=source.run(None,{k:vectors[k] for k in keys})[0]
            np.testing.assert_allclose(reference,expected,atol=1e-5,rtol=1e-4)
            session=ort.InferenceSession(str(root/(name+'.onnx')),options,providers=['CPUExecutionProvider'])
            actual=np.concatenate([session.run(None,{k:vectors[k][i:i+1] for k in keys})[0] for i in range(1024)])
            np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
            metrics[name]=dict(provided_vs_torch=float(np.max(abs(reference-expected))),
                ros_opset15_vs_torch=float(np.max(abs(actual-expected))))
            meta['file_sha256'][name+'.onnx']=sha256(root/(name+'.onnx'))
        np.savez(root/'conformance.npz',**vectors)
        meta['file_sha256']['conformance.npz']=sha256(root/'conformance.npz')
        meta['onnx_compatibility']=dict(opset=15,batch=1,engine='onnxruntime',cases=1024,
            method='same weights re-exported, not opset tag rewriting',metrics=metrics)
        write_json(root/'deployment_metadata.json',meta)
        with (root/'conformance.txt').open('w') as f:
            for x,y in [(vectors['history'],vectors['encoder_output_body']),(vectors['raw_actor_input'],vectors['actions'])]:
                f.write(f'{len(x)} {x.shape[1]} {y.shape[1]}\n')
                for a,b in zip(x,y):f.write(' '.join(format(float(v),'.9g') for v in np.r_[a,b])+'\n')
        for name in ('command_processor.py','nandu1_v131.py','nandu1_contract.py','yaw_conformance.json','yaw_conformance.txt'):
            source=args.asset_bundle/name
            if source.is_file():shutil.copyfile(source,root/name)
        # Keep source snapshots and distinguish current export from historical training.
        sources={}
        for module in list(sys.modules.values()):
            source=getattr(module,'__file__',None)
            if not source:continue
            source=Path(source).resolve()
            if lab not in source.parents or source.suffix!='.py':continue
            relative=str(source.relative_to(lab));target=root/'source_snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);sources[relative]=sha256(source)
        task=copy.deepcopy(baseline['task']);task.update(task_id=c.TASK,play_alias=c.PLAY,checkpoint_schema=c.SCHEMA,
            runtime_profile=c.RUNTIME,evaluation_stage=stage,command_rate=[2.,.8,.5],
            feature_transform_included=True,actor_feature_width=62,ros_runtime_supported=True)
        if (meta['baseline_contract']['tcp_local_m']!=task['tcp_local_m']
            or meta['baseline_contract']['payload']!=task['payload_contract']):raise ValueError('TCP/payload source mismatch')
        write_json(root/'provenance.json',dict(checkpoint=str(checkpoint),sha256=sha256(checkpoint),
            original_verified_download=json.loads((checkpoint.parent/'verified-download.json').read_text()),
            training_revision=state.get('source_revision'),training_dirty=state.get('source_dirty'),
            current_export_revision=subprocess.check_output(['git','-C',str(lab),'rev-parse','HEAD'],text=True).strip(),
            current_export_sources=sources,provided_export_hashes=metadata['file_sha256'],
            asset_source=str(args.asset_bundle.resolve()),asset_files={k:v for k,v in baseline['files'].items() if k.startswith('assets/')},
            source_claim='Original checkpoint provenance retained; current inference replay checked against provided graphs on 1024 cases, not a full historical source reconstruction'))
        files={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
        m=dict(schema=base.BUNDLE_SCHEMA,task=task,dimensions=dict(observation=49,history=245,encoder=7,command=3,policy_input=59,action=13),
            checkpoint=dict(path=str(checkpoint),sha256=sha256(checkpoint),iteration=state['iter']),files=files,
            numerical_conformance=dict(cases=1024,atol=1e-5,rtol=1e-4,metrics=metrics))
        write_json(root/'manifest.yaml',m);files=dict(files,**{'manifest.yaml':sha256(root/'manifest.yaml')})
        (root/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(files.items())))
        load_bundle(root,c.TASK);root.rename(output)
    print(json.dumps(dict(bundle=str(output),stage=stage,metrics=metrics),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True)
    for name in ('checkpoint','provided-export','asset-bundle','output','training-root'):p.add_argument('--'+name,type=Path,required=True)
    export(p.parse_args())


if __name__=='__main__':main()
