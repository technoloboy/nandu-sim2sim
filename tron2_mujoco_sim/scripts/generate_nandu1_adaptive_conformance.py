#!/usr/bin/env python3
"""Generate independent Torch/training-math oracle for deployed E/r2 state."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nandu1_contract as base
from nandu1_adaptive_runtime import AdaptiveRuntime
from export_nandu1_adaptive_bundle import seal, write_json
from nandu1_adaptive_contract import contract_for

FIELDS = [('commands',3),('encoded',7),('dynamic',12),('hidden',64),('eta',8),
          ('actor_input',71),('action',13),('history',245),('memory_filter',2),('height_filter',1)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1200)
    a=p.parse_args();root=a.bundle.resolve();torch.set_num_threads(1)
    profile=contract_for(json.loads((root/'manifest.yaml').read_text())['task']['task_id'])
    enabled=profile.HEIGHT_MEMORY_ENABLED;ablation=profile.IS_ABLATION
    width=profile.INTERFACE['actor_raw'];dynamic_width=width-59
    # Independent recurrent rollouts feed their own float32 actions back into
    # observations. Keep graph tolerance strict; bound accumulated rollout error
    # separately for original A/r2 (single-graph tests remain 1e-5/1e-4).
    rollout_atol=getattr(profile,'ROLLOUT_ATOL',1e-5 if ablation else 1e-4)
    fields=[(key, dynamic_width if key=='dynamic' else width if key=='actor_input' else (1 if ablation else 0) if key=='height_filter' else size) for key,size in FIELDS]
    sys.path.insert(0,str(a.training_root.resolve()/'rsl_rl'))
    from rsl_rl.nandu_adaptive_balance.math import CausalCommandProcessor, SlowErrorMemory
    from rsl_rl.nandu_adaptive_balance.height_memory import HeightSlowErrorMemory
    command=CausalCommandProcessor(1);memory=SlowErrorMemory(1);height=HeightSlowErrorMemory(1,enabled=enabled)
    enc,actor,context=[torch.jit.load(str(root/(n+'.pt'))).eval() for n in ('encoder','policy','context_step')]
    r=AdaptiveRuntime(root,initial_height=.9)
    rng=np.random.default_rng(13126);records=[];max_error=0.
    with torch.inference_mode():
        for i in range(a.steps):
            reset=i%200==0;stage=(i//200)%5;local=i%200
            if reset:
                command.reset(height=.9);memory.reset();height.reset();hist=base.History()
                previous=torch.zeros(1,13);hidden=torch.zeros(1,64);eta=torch.zeros(1,8)
                r.stage=stage;r.reset(initial_height=.9);prior_clipped=False
            ob=rng.normal(0,.02,49).astype(np.float32);ob[3:6]=[.05*np.sin(i*.02),0,-1]
            ob[9:12]=[0,0,-1];ob[36:]=previous.numpy()[0]
            if local in (100,101):ob[3:6]=[.85,0,-.5]
            yl,vl=base.STAGE_LIMITS[stage];lo,hi=base.STAGE_HEIGHT_RANGES[stage]
            goal=np.array([yl*.4*np.sign(np.sin(local*.09)),vl*.6*np.sign(np.cos(local*.07)),
                           np.clip(.9+(.1 if local%80<40 else -.1),lo,hi)],np.float32)
            o=torch.from_numpy(ob).reshape(1,49);g=torch.from_numpy(goal).reshape(1,3)
            applied=command.step(g,.02,step_id=local,gravity_b=o[:,3:6],gyro_b=o[:,:3]/.25,
                                 stage=stage,height_rate_limit=profile.HEIGHT_RATES[stage])
            history=torch.from_numpy(hist.append(ob)).reshape(1,245);e=enc(history)
            dynamic=torch.zeros(1,dynamic_width)
            if local:
                hidden,eta,_=context(o,previous,hidden,eta)
                gravity=o[:,3:6];up=-gravity/gravity.norm(dim=-1,keepdim=True)
                tilt=up[:,2].clamp(-1,1).acos();yaw=(o[:,:3]/.25*up).sum(-1)
                mm=memory.step_errors(applied[:,1]-e[:,3],applied[:,0]-yaw,g,command.height_rate,
                                     tilt=tilt,clipped=prior_clipped,dt=.02,step_id=local)
                parts=[mm,eta,command.height_rate[:,None]/.5]
                if ablation:
                    hm=height.step_errors(applied[:,2]-e[:,6],command.height_rate,
                                          tilt=tilt,clipped=prior_clipped,dt=.02,step_id=local)
                    parts.append(hm)
                dynamic=torch.cat(parts,-1)
            raw=torch.cat((e,o,applied,dynamic),-1);issued=actor(raw)
            values=[applied,e,dynamic,hidden,eta,raw,issued,history,memory.error_lpf,height.error_lpf if ablation else torch.empty(0)]
            expected={key:t.numpy().reshape(-1).copy() for (key,_),t in zip(fields,values)}
            actual=r.step(ob,goal,step_id=local)
            for key,_ in fields:
                try:
                    np.testing.assert_allclose(actual[key],expected[key],atol=rollout_atol,rtol=1e-4,err_msg=f'cycle={i} field={key}')
                except AssertionError:
                    if key=='action':
                        same_input=actor(torch.from_numpy(actual['actor_input']).reshape(1,width)).numpy()[0]
                        print(json.dumps(dict(cycle=i,same_input_graph_error=float(abs(actual[key]-same_input).max()),
                            recurrent_actor_input_error=float(abs(actual['actor_input']-expected['actor_input']).max()),
                            action_error=float(abs(actual[key]-expected[key]).max()))),flush=True)
                    raise
                if actual[key].size:max_error=max(max_error,float(np.max(np.abs(actual[key]-expected[key]))))
            clipped=local%79==78
            r.report_interval_clipping(action_clipped=clipped,target_clipped=local%37==36)
            records.append(np.r_[int(reset),stage,int(prior_clipped),local,ob,goal,
                                   np.concatenate([expected[key] for key,_ in fields])])
            previous=issued;prior_clipped=clipped
    vectors=np.asarray(records)
    np.savez_compressed(root/'runtime_conformance.npz',vectors=vectors)
    np.savetxt(root/'runtime_conformance.txt',vectors,fmt='%.9g',header=f'{a.steps} {vectors.shape[1]}',comments='')
    write_json(root/'runtime_conformance.json',dict(steps=a.steps,fields=fields,
        prefix=['reset','stage','previous_action_clipped','step_id','observation49','raw_intent3'],
        max_ort_vs_training_torch_error=max_error,atol=rollout_atol,rtol=1e-4,
        single_graph_tolerance=dict(atol=1e-5,rtol=1e-4),
        phase='physics -> manager command compute -> observations -> Student/memory update -> next Actor',
        sources=['IsaacLab/.../manager_based_rl_env.py:232-238','nandu_adaptive_balance/runner.py:update_runtime'],
        known_target_clipping='separately recorded; does not impersonate training action clipping'))
    # Single-graph test vectors, including the graph's internal feature transform.
    single=np.load(root/'conformance.npz')
    with (root/'onnx_conformance.txt').open('w') as out:
        count=len(single['history']);out.write(str(count)+'\n')
        for i in range(count):
            parts=[single[k][i].reshape(-1) for k in ('history','observation','commands','dynamic',
                'previous_issued_action','hidden','filtered_eta','encoded','raw_actor_input',
                'actions','next_hidden','next_filtered_eta','raw_eta')]
            out.write(' '.join(format(float(x),'.9g') for x in np.concatenate(parts))+'\n')
    seal(root)
    print(json.dumps(dict(steps=a.steps,max_error=max_error,columns=vectors.shape[1])))


if __name__=='__main__':main()
