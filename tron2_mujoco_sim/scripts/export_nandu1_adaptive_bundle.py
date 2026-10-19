#!/usr/bin/env python3
"""Export an audited adaptive checkpoint; never modify training or old bundles."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nandu1_adaptive_contract import contract_for
from nandu1_adaptive_bundle import load_adaptive_bundle, sha256, validate_metadata
from nandu1_bundle import load_bundle


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def seal(root):
    m = json.loads((root / 'manifest.yaml').read_text())
    m['files'] = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
                  if p.is_file() and p.name not in ('manifest.yaml', 'SHA256SUMS')}
    write_json(root / 'manifest.yaml', m)
    sums = dict(m['files'], **{'manifest.yaml': sha256(root / 'manifest.yaml')})
    (root / 'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k, v in sorted(sums.items())))


def export_ros_compatible_graphs(root, pipeline, meta):
    """Re-export weights at opset15, not just relabel opset17 operators."""
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    vectors = np.load(root / 'conformance.npz')
    raw = root / 'original_onnx_opset17'; raw.mkdir()
    graphs = [('encoder',pipeline.encoder,['history'],['encoded']),
        ('policy',pipeline.actor,['raw_actor_input'],['actions']),
        ('actor_pipeline',pipeline,['history','observation','commands','dynamic'],['actions']),
        ('context_step',pipeline.context_step,['observation','previous_issued_action','hidden','filtered_eta'],
         ['next_hidden','next_filtered_eta','raw_eta'])]
    metrics={}
    for name,model,inputs,outputs in graphs:
        file=root/(name+'.onnx');shutil.copyfile(file,raw/file.name)
        values=tuple(torch.from_numpy(vectors[k]) for k in inputs)
        torch.onnx.export(model,tuple(v[:1] for v in values),str(file),input_names=inputs,output_names=outputs,
            opset_version=15,dynamo=False,dynamic_axes={k:{0:'batch'} for k in inputs+outputs})
        onnx.checker.check_model(onnx.load(file))
        options=ort.SessionOptions();options.intra_op_num_threads=1
        result=ort.InferenceSession(str(file),options,providers=['CPUExecutionProvider']).run(None,{k:v.numpy() for k,v in zip(inputs,values)})
        metrics[name]=0.
        for k,value in zip(outputs,result):
            reference=vectors[k];np.testing.assert_allclose(value,reference,atol=1e-5,rtol=1e-4)
            metrics[name]=max(metrics[name],float(np.max(np.abs(value-reference))))
        meta['file_sha256'][file.name]=sha256(file)
    meta['onnx_compatibility']=dict(target_opset=15,reason='Existing ROS ONNX Runtime 1.10',
        method='Re-export identical weights, not opset-tag rewriting',max_errors_vs_training_torch=metrics,
        original_opset17_files={str(p.relative_to(root)):sha256(p) for p in raw.iterdir()})
    write_json(root/'deployment_metadata.json',meta)


def verify_provided_and_vectors(root,pipeline,meta,provided):
    """Compare explicit checkpoint Torch and graphs; optional supplied export."""
    import numpy as np
    import torch
    import onnxruntime as ort
    if provided is not None:
        original=json.loads((provided/'deployment_metadata.json').read_text());validate_metadata(original)
        if original['checkpoint_sha256']!=meta['checkpoint_sha256']:raise ValueError('Provided export is another checkpoint')
        for name,digest in original['file_sha256'].items():
            if Path(name).name!=name or sha256(provided/name)!=digest:raise ValueError('Provided export hash mismatch: '+name)
    rng=np.random.default_rng(13122);n=1024;width=meta['interface']['actor_raw']-59
    arrays={key:rng.normal(0,.1,(n,size)).astype(np.float32) for key,size in (
        ('history',245),('observation',49),('dynamic',width),('previous_issued_action',13),('hidden',64),('filtered_eta',8))}
    arrays['observation'][:,5]=arrays['observation'][:,11]=-1
    if width==12 and meta.get('height_memory',{}).get('enabled') is False:arrays['dynamic'][:,-1]=0
    arrays['commands']=np.c_[rng.uniform(-1.57,1.57,n),rng.uniform(-1,1,n),rng.uniform(.4,1.4,n)].astype(np.float32)
    tensor=lambda k:torch.from_numpy(arrays[k])
    with torch.inference_mode():
        arrays['encoded']=pipeline.encoder(tensor('history')).numpy()
        arrays['raw_actor_input']=np.c_[arrays['encoded'],arrays['observation'],arrays['commands'],arrays['dynamic']]
        arrays['actions']=pipeline.actor(tensor('raw_actor_input')).numpy()
        for k,value in zip(('next_hidden','next_filtered_eta','raw_eta'),pipeline.context_step(*[
            tensor(k) for k in ('observation','previous_issued_action','hidden','filtered_eta')])):arrays[k]=value.numpy()
    specs={'encoder':(['history'],['encoded']), 'policy':(['raw_actor_input'],['actions']),
        'actor_pipeline':(['history','observation','commands','dynamic'],['actions']),
        'context_step':(['observation','previous_issued_action','hidden','filtered_eta'],['next_hidden','next_filtered_eta','raw_eta'])}
    metrics={};options=ort.SessionOptions();options.intra_op_num_threads=1
    sources=([('provided',provided)] if provided is not None else [])+[('new_opset17',root)]
    for name,(inputs,outputs) in specs.items():
        for label,source in sources:
            result=ort.InferenceSession(str(source/(name+'.onnx')),options,providers=['CPUExecutionProvider']).run(None,{k:arrays[k] for k in inputs})
            maximum=0.
            for key,value in zip(outputs,result):
                np.testing.assert_allclose(value,arrays[key],atol=1e-5,rtol=1e-4)
                maximum=max(maximum,float(np.max(abs(value-arrays[key]))))
            metrics[name+'_'+label]=maximum
    np.savez_compressed(root/'conformance.npz',**arrays)
    meta['file_sha256']['conformance.npz']=sha256(root/'conformance.npz')
    meta['provided_export_validation' if provided is not None else 'checkpoint_vector_validation']=dict(
        cases=n,atol=1e-5,rtol=1e-4,max_errors=metrics,source=str(provided) if provided is not None else None,
        metadata_sha256=sha256(provided/'deployment_metadata.json') if provided is not None else None)


def export(args):
    c=contract_for(args.task)
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    digest = sha256(checkpoint)
    if digest != c.CHECKPOINT_SHA256: raise ValueError('Checkpoint differs from audited deployment identity')
    if not c.IS_ABLATION and args.provided_export is None:raise ValueError('Explicit provided export required for local adaptive balance')
    if output.exists():
        existing, _ = load_adaptive_bundle(output, args.task)
        if existing['checkpoint']['sha256'] != digest: raise ValueError('Output has another checkpoint')
        print('Reusing verified bundle:', output)
        return
    training = args.training_root.resolve()
    sys.path.insert(0, str(training / 'rsl_rl'))
    import torch
    torch.set_num_threads(1)
    from rsl_rl.nandu_adaptive_balance.deployment import load_pipeline, export_bundle
    exp, state, pipeline = load_pipeline(checkpoint)
    if exp.task != c.TASK or exp.schema != c.SCHEMA: raise ValueError('Wrong checkpoint identity')
    if state['train_cfg'].get('clip_actions', 'missing') is not None: raise ValueError('Unexpected action clipping')
    legacy, _ = load_bundle(args.asset_bundle)
    if legacy['task']['task_id'] != 'Isaac-NANDU1-Blind-Flat-v1-3-1':
        raise ValueError('Audited fixed-TCP asset bundle required')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.adaptive-export-', dir=output.parent) as temp:
        root = Path(temp) / 'bundle'
        meta = export_bundle(checkpoint, root, onnx=True, _loaded=(exp, state, pipeline))
        if hasattr(c,'INFERENCE_SHA256') or args.provided_export is not None:
            verify_provided_and_vectors(root,pipeline,meta,args.provided_export.resolve() if args.provided_export else None)
        export_ros_compatible_graphs(root, pipeline, meta)
        validate_metadata(meta)
        shutil.copytree(args.asset_bundle.resolve() / 'assets', root / 'assets')
        saved = state.get('source_files_sha256', state['train_cfg'].get('source_files_sha256', {}))
        sources, mismatches = {}, {}
        paths = {Path(module.__file__).resolve() for module in list(sys.modules.values())
                 if getattr(module, '__file__', None) and str(Path(module.__file__).resolve()).startswith(str(training) + os.sep)}
        paths.update(training / name for name in saved if (training / name).is_file())
        for p in sorted(paths):
            relative = str(p.relative_to(training))
            target = root / 'source_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            sources[relative] = sha256(p)
        for relative, expected in saved.items():
            actual = sources.get(relative)
            if actual != expected: mismatches[relative] = dict(saved=expected, current=actual)
        critical = ('nandu_adaptive_balance/', 'nandu_experiments/')
        bad = {k: v for k, v in mismatches.items() if any(part in k for part in critical)}
        if bad: raise ValueError('Checkpoint inference source fingerprints differ: ' + str(bad))
        task = copy.deepcopy(legacy['task'])
        import nandu1_contract as base
        yaw,vx=base.STAGE_LIMITS[c.EVALUATION_STAGE];low,high=base.STAGE_HEIGHT_RANGES[c.EVALUATION_STAGE]
        task.update(task_id=c.TASK, play_alias=c.PLAY, checkpoint_schema=c.SCHEMA,
                    runtime_profile=c.RUNTIME, evaluation_stage=c.EVALUATION_STAGE, command_min=[-yaw,-vx,low],
                    command_max=[yaw,vx,high], command_rate=[2.,.6,c.HEIGHT_RATES[c.EVALUATION_STAGE]], action_clipping=None,
                    height_rate_limits_m_s=c.HEIGHT_RATES,
                    history_length=5, policy_dt=.02, ros_runtime_supported=True,
                    command_processor={'implementation':c.RUNTIME[len('nandu1_'):],'control_dt':.02},
                    actor_velocity_frame='body', tcp_local_m=c.TCP)
        if c.IS_ABLATION:task['height_memory_profile']=c.HEIGHT_MEMORY
        # Physical parameter arrays are independently checked by the typed loader.
        # Baseline payload/TCP metadata is exported from this task's training source.
        baseline = meta['baseline_contract']
        if task['tcp_local_m'] != baseline['tcp_local_m'] or task['payload_contract'] != baseline['payload']:
            raise ValueError('Training TCP/payload contract differs from audited asset bundle')
        write_json(root / 'provenance.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
            checkpoint_iter=state['iter'], source_revision=state.get('source_revision'),
            source_dirty=state.get('source_dirty'), saved_source_fingerprints=saved,
            current_export_source_fingerprints=sources, source_mismatches=mismatches,
            asset_source=str(args.asset_bundle.resolve()), asset_source_fingerprints={
                k:v for k,v in legacy['files'].items() if k.startswith('assets/')},
            source_claim='Current export dependencies snapshotted; mismatches explicitly reported, not full historical dirty-tree reconstruction'))
        write_json(root / 'manifest.yaml', dict(schema=c.BUNDLE_SCHEMA, task=task,
            dimensions=dict(observation=49,history=245,encoder=7,command=3,dynamic=c.INTERFACE['actor_raw']-59,policy_input=c.INTERFACE['actor_raw'],action=13),
            checkpoint=dict(path=str(checkpoint),sha256=digest,iteration=state['iter']), files={}))
        seal(root)
        print(json.dumps({'audited_inference_sha256':{n+'.onnx':sha256(root/(n+'.onnx')) for n in ('encoder','policy','context_step','actor_pipeline')}}),flush=True)
        load_adaptive_bundle(root, c.TASK)
        root.rename(output)
    print('Exported and validated:', output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True)
    p.add_argument('--provided-export',type=Path)
    for name in ('checkpoint', 'training-root', 'asset-bundle', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    export(p.parse_args())


if __name__ == '__main__': main()
