"""Mouse safety contract: no ROS, desktop, or policy required."""
import importlib
import math
from pathlib import Path
import socket
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def module():
    return importlib.import_module('nandu1_mouse_protocol')


def panel():
    p = module().ManualState([-1.57, -1., .4], [1.57, 1., 1.4], .9)
    p.status(True, (2, 3), [0, 0, .9], 10.)
    p.focus(True)
    p.begin()
    return p


def test_deadzone_and_circle_do_not_boost_diagonal():
    p = panel()
    p.drag(.02, -.02)
    assert p.values() == [0., 0., .9]
    p.drag(-1, -1)
    yaw, vx, h = p.values()
    assert yaw == pytest.approx(1.57 * .25 / math.sqrt(2))
    assert vx == pytest.approx(.25 / math.sqrt(2))
    assert h == .9


def test_release_focus_and_stop_hold_height():
    p = panel()
    p.set_height(1.2)
    p.drag(0, -1)
    assert p.values() == [0., .25, 1.2]
    p.release()
    assert p.values() == [0., 0., 1.2]
    p.begin(); p.drag(0, -1); p.focus(False); p.focus(True)
    assert p.values() == [0., 0., 1.2]
    p.drag(0, -1)  # focus recovery must not resume the old gesture
    assert p.values() == [0., 0., 1.2]


def test_new_generation_and_stale_status_require_new_drag():
    p = panel(); p.drag(0, -1)
    p.status(True, (2, 4), [0, 0, .88], 10.1)
    assert p.values() == [0., 0., .88]
    p.drag(0, -1)
    assert p.values()[1] == 0
    p.begin(); p.drag(0, -1); p.expire(10.7)
    assert not p.ready
    assert p.values() == [0., 0., .88]


def test_asymmetric_limits_height_clamp_and_gain():
    p = module().ManualState([-1, -.4, .6], [2, .8, 1.2], .9)
    p.status(True, (0, 1), [0, 0, .9], 0); p.focus(True); p.begin()
    p.set_gain(.5); p.drag(-1, 0)
    assert p.values() == [1., 0., .9]
    p.drag(1, 0)
    assert p.values() == [-.5, 0., .9]
    p.set_height(3)
    assert p.values()[2] == 1.2
    p.focus(False); p.set_height(.6)
    assert p.values()[2] == 1.2
    with pytest.raises(ValueError): p.set_gain(float('nan'))


def packet(seq=1, **extra):
    return module().frame('command', 'test', 'task', seq=seq, key=[2, 3],
                          enabled=True, axes=[0., 1.], height=1., reset=False, arm_token=1, **extra)


def gate():
    g = module().CommandGate('test', 'task', [-1.57, -1, .4], [1.57, 1, 1.4], .9)
    g.sync(True, (2, 3), .9)
    neutral = packet(); neutral['enabled'] = False
    g.receive(neutral, 10)
    return g


def test_gate_timeout_holds_height_and_requires_neutral_before_rearming():
    g = gate()
    assert not g.receive(packet(2), 10.02)
    assert g.values(10.1) == [0, 1, 1.]
    assert g.values(10.3) == [0, 0, 1.]
    g.receive(packet(3), 10.31)
    assert g.values(10.31) == [0, 0, 1.]
    p = packet(4); p['enabled'] = False
    p['arm_token'] = 2
    active = packet(5); active['arm_token'] = 2
    g.receive(p, 10.32); g.receive(active, 10.34)
    assert g.values(10.34) == [0, 1, 1.]


@pytest.mark.parametrize('field,value', [('session', 'other'), ('task', 'other'),
    ('schema', 'bad'), ('key', [2, 4]), ('seq', True), ('axes', [0, float('nan')]),
    ('height', float('inf')), ('axes', [0, 2]), ('enabled', 1)])
def test_gate_rejects_invalid_packets_neutrally(field, value):
    g = gate(); g.receive(packet(2), 10.)
    p = packet(3); p[field] = value
    with pytest.raises(ValueError): g.receive(p, 10.02)
    assert g.values(10.02) == [0, 0, 1.]


def test_duplicate_sequence_reset_and_generation_invalidate_commands():
    g = gate(); g.receive(packet(2), 10.)
    with pytest.raises(ValueError): g.receive(packet(2), 10.02)
    g.sync(False, (2, 3), 1.)
    p = packet(3); p['reset'] = True; p['enabled'] = False
    p['arm_token'] = g.arm_token
    assert g.receive(p, 10.04) is True
    g.sync(True, (2, 4), .9)
    assert g.values(10.05) == [0, 0, .9]


def test_json_channel_handles_partial_frames_and_rejects_overflow():
    a, b = socket.socketpair()
    try:
        channel = module().JsonChannel(a)
        b.sendall(b'{"value":')
        assert channel.receive() == []
        b.sendall(b'1}\n{"value":2}\n')
        assert channel.receive() == [{'value': 1}, {'value': 2}]
        b.sendall(b'x' * 4097)
        with pytest.raises(ValueError): channel.receive()
    finally: a.close(); b.close()


def test_json_channel_send_is_real_json_and_rejects_nan():
    a, b = socket.socketpair()
    try:
        channel = module().JsonChannel(a)
        assert channel.send({'height': .9})
        assert b.recv(4096) == b'{"height":0.9}\n'
        with pytest.raises(ValueError): channel.send({'height': float('nan')})
    finally: a.close(); b.close()


@pytest.mark.parametrize('version,family,topic', [
    ('1','body','/nandu1/v1_1/command'), ('2','body','/nandu1/command'),
    ('3','body','/nandu1/v1_3/command'), ('2-1','body','/nandu1/v1_2_1/command'),
    ('3-1','tcp','/nandu1/v1_3_1/command'), ('4','heading','/nandu1/v1_4/command'),
    ('4-1','heading','/nandu1/v1_4_1/command')])
def test_seven_tasks_and_play_aliases_keep_command_semantics(version, family, topic):
    for middle in ('Flat-v','Flat-Play-v'):
        assert module().command_interface('Isaac-NANDU1-Blind-'+middle+'1-'+version) == (family,topic)


def test_controller_status_requires_fresh_matching_reset_and_tracking():
    s = module().ControllerStatus('body', .9)
    track = [0.] * 116
    track[:5] = [123., 2., .1, .2, .95]
    s.tracking(track, 10.)
    s.diagnostic('POLICY', 10.)
    assert not s.snapshot(10.)['ready']
    s.reset([123., 2., 3.] + [0.] * 8 + [.9])
    assert s.snapshot(10.)['ready']
    assert s.snapshot(10.)['applied'] == [.1,.2,.95]
    assert s.snapshot(10.)['key'] == [2,3]
    assert not s.snapshot(10.6)['ready']
    s.diagnostic('NANDU1 FAULT', 10.7)
    s.tracking(track, 10.7)
    assert not s.snapshot(10.7)['ready']


def test_status_wrong_shape_and_new_epoch_cannot_reuse_old_generation():
    s = module().ControllerStatus('heading', .9)
    with pytest.raises(ValueError): s.tracking([0.] * 116, 1.)
    track = [0.] * 131; track[:5] = [123., 2., 0., 0., .9]
    s.reset([123., 2., 3.] + [0.] * 8 + [.9])
    s.tracking(track, 1.); s.diagnostic('POLICY', 1.)
    assert s.snapshot(1.)['ready']
    track[1] = 3
    s.tracking(track, 1.1)
    assert not s.snapshot(1.1)['ready']


def test_new_reset_generation_waits_for_tracking_from_that_policy_cycle():
    s=module().ControllerStatus('body',.9)
    track=[0.]*116;track[:5]=[123.,2.,.1,.2,1.2]
    s.reset([123.,2.,3.]+[0.]*8+[.9])
    s.tracking(track,10.);s.diagnostic('POLICY',10.)
    assert s.snapshot(10.)['ready']
    # RESET evidence is published before its policy tracking sample. Do not
    # associate the NEW generation with OLD applied height/velocity.
    s.reset([124.,2.,4.]+[0.]*8+[.9])
    assert not s.snapshot(10.01)['ready']
    track[:5]=[124.,2.,0.,0.,.9]
    s.tracking(track,10.02)
    assert s.snapshot(10.02)['ready']
    assert s.snapshot(10.02)['applied']==[0.,0.,.9]


def test_live_mouse_owner_blocks_external_node_but_stale_metadata_does_not():
    guard = module().assert_external_available
    owner = {'node':'/nandu1_mouse_bridge', 'session':'test', 'topic':'/nandu1/command'}
    publishers = [['/nandu1/command', ['/nandu1_mouse_bridge']]]
    with pytest.raises(RuntimeError, match='mouse'): guard(owner,publishers)
    guard(owner,[])
    guard(None,publishers)


def test_gui_module_is_importable_without_tk_or_ros():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    code = f'import sys;sys.path.insert(0,{str(root)!r});sys.modules["tkinter"]=None;sys.modules["rospy"]=None;import nandu1_mouse_gui'
    result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_bridge_pause_drops_old_enabled_and_queued_neutral_enabled_pair():
    g = gate(); g.receive(packet(2),10.02)
    # No values() call during the pause; old buffered packets must not refresh age.
    g.receive(packet(3),10.42)
    assert g.values(10.42)==[0.,0.,1.]
    p=packet(4);p['enabled']=False
    g.receive(p,10.43);g.receive(packet(5),10.43)
    assert g.values(10.43)==[0.,0.,1.]
    assert g.needs_release
    # Only acknowledgement of the NEW server challenge can rearm.
    p=packet(6);p['enabled']=False;p['arm_token']=2
    g.receive(p,10.44)
    p=packet(7);p['arm_token']=2
    g.receive(p,10.46)
    assert g.values(10.46)==[0.,1.,1.]


def test_gui_rearm_challenge_disarms_an_existing_drag():
    p=panel();p.drag(0,-1)
    p.status(True,(2,3),[0,0,.9],10.02,arm_token=8)
    assert p.values()==[0.,0.,.9]
    p.drag(0,-1)
    assert p.values()[1]==0


def test_closing_packet_neutralizes_before_passive_shutdown():
    g=gate();g.receive(packet(2),10.)
    p=packet(3);p['closing']=True
    g.receive(p,10.02)
    assert g.closing
    assert g.values(10.02)==[0.,0.,1.]


def test_huge_integer_cannot_bypass_neutralization():
    g=gate();g.receive(packet(2),10.)
    p=packet(3);p['height']=10**400
    with pytest.raises(ValueError):g.receive(p,10.02)
    assert g.values(10.02)==[0.,0.,1.]


def test_deep_json_is_a_rejected_frame_not_an_unhandled_callback_exception():
    a,b=socket.socketpair()
    try:
        channel=module().JsonChannel(a)
        b.sendall(b'{"x":'+b'['*1100+b'0'+b']'*1100+b'}\n')
        with pytest.raises(ValueError):channel.receive()
    finally:a.close();b.close()


def test_socket_backpressure_keeps_partial_frame_instead_of_splicing_new_input():
    a,b=socket.socketpair()
    try:
        a.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,1024)
        sender=module().JsonChannel(a);receiver=module().JsonChannel(b)
        payload={'text':'x'*3500};sent=0
        for _ in range(10):
            assert sender.send(payload)
            sent+=1
            if sender.outgoing:break
        assert sender.outgoing
        pending=sender.outgoing
        assert not sender.send({'must_not_splice':True})
        assert sender.outgoing==pending
        received=[]
        for _ in range(20):
            received.extend(receiver.receive());sender.flush()
            if len(received)==sent:break
        assert received==[payload]*sent
        assert sender.outgoing==b''
    finally:a.close();b.close()
