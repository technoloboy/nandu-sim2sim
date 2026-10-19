"""GUI acceptance must fail when only slider propagation or reset breaks."""
import copy
import importlib.util
from pathlib import Path
import pytest


@pytest.mark.parametrize('broken',['none','requested','applied','reset','stage','ui_stage','ui_bounds','ui_slider','ui_missing','ui_roundoff'])
def test_gui_gate_observes_height_and_reset_not_only_gesture_labels(broken):
    p=Path(__file__).with_name('run_nandu1_adaptive_gui.py')
    spec=importlib.util.spec_from_file_location('gui_acceptance_gate',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert callable(getattr(m,'evaluate_gui_evidence',None)), 'Height-aware GUI evidence gate missing'
    from nandu1_adaptive_contract import contract_for
    profile=contract_for('Isaac-NANDU1-Blind-Flat-v1-3-1-2-3')
    gestures=[dict(phase=i,target=[0,0,.92 if i==4 else .9]) for i in range(6)]
    gestures[0]['ui']=dict(stage_label='Stage: 1 | TCP',lower=[-.5,-.2,.68],upper=[.5,.2,1.12],slider_from=1.12,slider_to=.68)
    guard=[dict(returncode=1,output='Active mouse command source')]
    meta=dict(task=profile.TASK,layout='ablation_b',tracking_fields=list(range(284)),evaluation_stage=1)
    rows=[dict(tracking_stamp=1.,requested_0=.1,requested_1=.05,requested_2=.9,command_2=.9),
          dict(tracking_stamp=1.2,requested_0=-.1,requested_1=-.05,requested_2=.9,command_2=.9),
          dict(tracking_stamp=1.5,requested_0=0,requested_1=0,requested_2=.92,command_2=.905),
          dict(tracking_stamp=2.1,requested_0=0,requested_1=0,requested_2=.9,command_2=.9)]
    rows=[copy.copy(r) for r in rows for _ in range(6)]
    resets=[dict(stamp=.5,generation=1,initial_height=.9),dict(stamp=2.,generation=2,initial_height=.9)]
    if broken=='requested':
        for r in rows:r['requested_2']=.9
    if broken=='applied':
        for r in rows:r['command_2']=.9
    if broken=='reset':
        for r in rows:
            if r['tracking_stamp']>2:r['command_2']=r['requested_2']=.92
    if broken=='stage':meta['evaluation_stage']=4
    if broken=='ui_stage':gestures[0]['ui']['stage_label']='Stage: 4 | TCP'
    if broken=='ui_bounds':
        gestures[0]['ui'].update(lower=[-1.57,-1,.4],upper=[1.57,1,1.4])
    if broken=='ui_slider':gestures[0]['ui'].update(slider_from=1.4,slider_to=.4)
    if broken=='ui_missing':gestures[0].pop('ui')
    if broken=='ui_roundoff':gestures[0]['ui']['slider_to']=.6800000000000002
    checks=m.evaluate_gui_evidence(profile,gestures,guard,meta,rows,resets)
    assert all(checks.values()) is (broken in ('none','ui_roundoff'))
