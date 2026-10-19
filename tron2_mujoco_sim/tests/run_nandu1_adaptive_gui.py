"""Actual desktop + production launcher/Tk/SDK test; owns only its session."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

SIM=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SIM))
import nandu1_process as launcher
from nandu1_adaptive_contract import contract_for


def evaluate_gui_evidence(profile,gestures,guard,metadata,rows,resets):
    if not rows or not resets:return dict(samples=False)
    def values(i):return [float(r['requested_'+str(i)]) for r in rows]
    last_reset=max(resets,key=lambda r:float(r['stamp']));initial=float(last_reset['initial_height'])
    after=[r for r in rows if float(r['tracking_stamp'])>=float(last_reset['stamp'])]
    target=next((float(g['target'][2]) for g in gestures if g.get('phase')==4 and 'target' in g),initial)
    width=284 if profile.INTERFACE['actor_raw']==71 else 282
    from nandu1_contract import STAGE_LIMITS,STAGE_HEIGHT_RANGES
    yaw,vx=STAGE_LIMITS[profile.EVALUATION_STAGE];low,high=STAGE_HEIGHT_RANGES[profile.EVALUATION_STAGE]
    ui=next((g['ui'] for g in gestures if g.get('phase')==0 and isinstance(g.get('ui'),dict)),{})
    return dict(six_gestures=[r.get('phase') for r in gestures]==list(range(6)),
        vx_both_signs=min(values(1))<-.01 and max(values(1))>.01,
        yaw_both_signs=min(values(0))<-.01 and max(values(0))>.01,
        height_requested=target-initial>.015 and any(abs(v-target)<1e-6 for v in values(2)),
        height_applied=max(float(r['command_2']) for r in rows)>initial+.0005,
        height_reset_restored=(len(resets)>1 and bool(after) and all(
            abs(float(r['command_2'])-initial)<1e-6 and abs(float(r['requested_2'])-initial)<1e-6 for r in after)),
        source_guard=bool(guard) and guard[0].get('returncode',0)!=0 and 'Active mouse command source' in guard[0].get('output',''),
        task=metadata['task']==profile.TASK,
        stage=metadata.get('evaluation_stage')==profile.EVALUATION_STAGE,
        ui_stage=ui.get('stage_label','').startswith(f'Stage: {profile.EVALUATION_STAGE} |'),
        ui_bounds=ui.get('lower')==[-yaw,-vx,low] and ui.get('upper')==[yaw,vx,high],
        ui_slider=math.isclose(ui.get('slider_from',float('nan')),high,rel_tol=0,abs_tol=1e-9)
                  and math.isclose(ui.get('slider_to',float('nan')),low,rel_tol=0,abs_tol=1e-9),
        layout=metadata['layout']==getattr(profile,'TELEMETRY_LAYOUT','adaptive' if width==284 else 'adaptive_balance'),
        tracking_width=len(metadata['tracking_fields'])==width,samples=len(rows)>20)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();profile=contract_for(a.task);a.output=a.output.resolve()
    if a.output.exists():raise ValueError('GUI probe requires a fresh output directory')
    a.output.mkdir(parents=True)
    original=launcher.supervise
    def supervise(*args,**kw):
        gui=kw['gui_command']
        kw['gui_command']=[gui[0],str(SIM/'tests/nandu1_mouse_gui_probe.py'),
            '--report',str(a.output/'gestures.json'),'--container',kw['ros_env']['NANDU1_CONTAINER_NAME'],*gui[2:]]
        return original(*args,**kw)
    launcher.supervise=supervise
    sys.argv=[sys.argv[0],'--task',a.task,'--bundle',str(a.bundle.resolve()),
        '--telemetry',str(a.output/'tracking.csv'),'--command-source','mouse']
    code=launcher.main()
    if code:raise RuntimeError('Owned GUI launch failed: '+str(code))
    gestures=json.loads((a.output/'gestures.json').read_text())
    guard=json.loads((a.output/'gestures.guard.json').read_text())
    metadata=json.loads((a.output/'tracking.metadata.json').read_text())
    with (a.output/'tracking.csv').open() as f:rows=list(csv.DictReader(f))
    with (a.output/'tracking.reset.csv').open() as f:resets=list(csv.DictReader(f))
    checks=evaluate_gui_evidence(profile,gestures,guard,metadata,rows,resets)
    report=dict(scope='actual desktop MuJoCo + production Tk events + SDK/ROS, not mock',
        task=profile.TASK,rows=len(rows),checks=checks,passed=all(checks.values()))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise AssertionError('Actual GUI checks failed')


if __name__=='__main__':main()
