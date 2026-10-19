#!/usr/bin/env python3
"""Analyze real SDK+ROS telemetry, with explicit temporal/epoch quality gates."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nandu1_contract as c
from nandu1_bundle import load_bundle

def summarize(rows,manifest):
    if not rows:raise ValueError('No ROS telemetry rows')
    if any(not np.isfinite(list(row.values())).all() for row in rows):raise ValueError('Non-finite telemetry')
    epochs={int(row['epoch']) for row in rows}
    if len(epochs)!=1:raise ValueError('Separate reset epochs before computing tracking metrics')
    selected={}
    for row in rows:
        if row['sample_age']<-.05 or row['sample_age']>.1 or abs(row['tracking_age'])>.04:continue
        if int(row['epoch'])!=int(row['tracking_epoch']):continue
        key=row['tracking_stamp']
        if key not in selected or abs(row['tracking_age'])<abs(selected[key]['tracking_age']):selected[key]=row
    data=sorted(selected.values(),key=lambda r:r['tracking_stamp'])
    if len(data)<2:raise ValueError('Insufficient synchronized policy samples')
    vector=lambda prefix,n:np.array([[r[f'{prefix}_{i}'] for i in range(n)] for r in data])
    stamp=np.array([r['tracking_stamp'] for r in data]);sim=np.array([r['sim_time'] for r in data])
    command=vector('command',3);truth=vector('command_truth',3);estimated=vector('estimate',7);label=vector('truth',7)
    error=truth-command;est_error=estimated-label
    q=vector('q',13);dq=vector('dq',13);target=vector('q_target',13);tau=vector('torque_actual',13)
    lower=np.array(manifest['task']['joint_position_lower']);upper=np.array(manifest['task']['joint_position_upper'])
    pos=c.POSITION_IDS
    report={'scope':'real localhost SDK + ROS controller, not hardware','epoch':next(iter(epochs)),
            'task_id':manifest['task']['task_id'],'velocity_reference':manifest['task'].get('velocity_reference','body_com'),
            'command_frame':'base_heading' if manifest['task']['task_id'] in (c.V14_TASK,c.V141_TASK) else 'base_body',
            'raw_rows':len(rows),'synchronized_policy_samples':len(data),'excluded_or_duplicate_rows':len(rows)-len(data),
            'wall_duration_s':float(stamp[-1]-stamp[0]),'sim_duration_s':float(sim[-1]-sim[0]),
            'policy_hz_median':float(1/np.median(np.diff(stamp))),
            'policy_hz_average':float((len(stamp)-1)/(stamp[-1]-stamp[0])),
            'max_policy_gap_s':float(np.max(np.diff(stamp))),
            'tracking_rmse_yaw_vx_height':np.sqrt(np.mean(error**2,axis=0)).tolist(),
            'tracking_mean_bias':np.mean(error,axis=0).tolist(),
            'estimator_rmse_base_v_up3_v_height':np.sqrt(np.mean(est_error**2,axis=0)).tolist(),
            'max_sample_age_s':max(r['sample_age'] for r in data),'max_abs_tracking_age_s':max(abs(r['tracking_age']) for r in data),
            'torque_outside_limits':int(np.count_nonzero(np.abs(tau)>c.EFFORT+1e-4)),
            'position_target_outside_limits':int(np.count_nonzero((target[:,pos]<lower[pos])|(target[:,pos]>upper[pos]))),
            'position_state_outside_limits_005_rad_tolerance':int(np.count_nonzero((q[:,pos]<lower[pos]-.05)|(q[:,pos]>upper[pos]+.05))),
            'velocity_state_outside_limits':int(np.count_nonzero(np.abs(dq)>c.VELOCITY+1e-4)),
            'pd_saturated_policy_samples':int(np.count_nonzero(vector('torque_saturated',13)))}
    report['velocity_projection_events']=max(r.get('velocity_projection_events',-1) for r in rows)
    # v1.4 continuously corrects yaw even at constant external intent. Segment
    # by external intent, but compute tracking error against applied commands.
    segment_command=vector('external_intent',3) if 'external_intent_0' in data[0] else command
    report['segment_reference']='external_intent' if 'external_intent_0' in data[0] else 'applied_command'
    if 'actor_estimate_heading_3' in data[0]:
        report['estimator_up3_vx_heading_rmse']=float(np.sqrt(np.mean((vector('actor_estimate_heading',7)[:,3]-truth[:,1])**2)))
    boundaries=np.r_[0,np.flatnonzero(np.any(np.abs(np.diff(segment_command,axis=0))>1e-6,axis=1))+1,len(data)]
    plateaus=[];previous=None
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        if end-begin<20:continue
        duration=stamp[end-1]-stamp[begin]
        if duration<.5:continue
        settled=np.arange(begin,end)[stamp[begin:end]>=stamp[end-1]-min(2.,duration/2.)]
        item={'command':segment_command[begin].tolist(),'applied_command_mean':np.mean(command[begin:end],axis=0).tolist(),
              'start_sim_s':float(sim[begin]),'duration_s':float(duration),
              'samples':int(end-begin),'rmse':np.sqrt(np.mean(error[begin:end]**2,axis=0)).tolist(),
              'settled_mean_truth':np.mean(truth[settled],axis=0).tolist(),
              'settled_bias':np.mean(error[settled],axis=0).tolist()}
        if previous is not None:
            pb,pe=previous
            baseline=np.mean(truth[max(pb,pe-25):pe],axis=0)
            delta=segment_command[begin]-segment_command[pb]
            response=[]
            # Start at the first applied command after the previous plateau.
            # Includes command slew; does not claim sub-policy-period precision.
            onset=pe
            for axis in range(3):
                if abs(delta[axis])<1e-6:continue
                for fraction in [.5,.9]:
                    reached=(truth[onset:end,axis]-baseline[axis])*np.sign(delta[axis])>=fraction*abs(delta[axis])
                    delay=None
                    for j in range(max(0,len(reached)-4)):
                        if reached[j:j+5].all():delay=float(stamp[onset+j]-stamp[onset]);break
                    response.append({'axis':axis,'fraction':fraction,'delay_s':delay})
            item['response_from_applied_command_including_slew']=response
        plateaus.append(item)
        previous=(begin,end)
    report['command_plateaus']=plateaus
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest,_=load_bundle(a.bundle)
    with a.csv.open() as f:rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    report=summarize(rows,manifest);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
