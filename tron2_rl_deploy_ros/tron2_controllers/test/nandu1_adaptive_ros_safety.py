#!/usr/bin/env python3
"""Acceptance-only probe, requires local relay15015; no hardware entry."""
import argparse
import json
from pathlib import Path
import socket
import time
import rospy
from diagnostic_msgs.msg import DiagnosticArray
from std_msgs.msg import Float64MultiArray,String
from tron2_controllers.msg import NANDU1TCPCommand
from nandu1_adaptive_contract import adaptive_tasks,contract_for


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',default='Isaac-NANDU1-Blind-Flat-v1-3-1-2-6',
        choices=adaptive_tasks()+('Isaac-NANDU1-Blind-Flat-v1-3-1-2',))
    a=p.parse_args(rospy.myargv()[1:]);rospy.init_node('nandu1_adaptive_safety')
    if rospy.get_param('/tron2_hw/robot_ip','')!='127.0.0.1':raise RuntimeError('Local simulation only')
    if rospy.get_param('/model_bundle/task/task_id','')!=a.task:raise RuntimeError('Safety probe task differs from running controller')
    adaptive=a.task in adaptive_tasks()
    width=282 if adaptive and contract_for(a.task).INTERFACE['actor_raw']==70 else 284
    tracking=[None];diagnostics=[None];evidence=[None];checks=[];priming={};bad_fault_context=[]
    rospy.Subscriber('/nandu1/tracking',Float64MultiArray,lambda m:tracking.__setitem__(0,m.data),queue_size=1,tcp_nodelay=True)
    rospy.Subscriber('/nandu1/reset_evidence',Float64MultiArray,lambda m:evidence.__setitem__(0,m.data),queue_size=1,tcp_nodelay=True)
    def diagnostic(message):
        for status in message.status:
            if status.name!='nandu1/policy':continue
            diagnostics[0]=status
            context={v.key:v.value for v in status.values}
            if status.level==2 and (int(context.get('epoch',0))<=0 or int(context.get('generation',0))<=0):
                bad_fault_context.append(status.message)
    rospy.Subscriber('/nandu1/diagnostics',DiagnosticArray,diagnostic,queue_size=100,tcp_nodelay=True)
    def full(message):
        if len(message.data)==width and message.data[121]==0:priming[message.data[0]]=message.data
    rospy.Subscriber('/nandu1/adaptive/tracking',Float64MultiArray,full,queue_size=100,tcp_nodelay=True)
    from nandu1_mouse_protocol import command_interface
    pub=rospy.Publisher(command_interface(a.task)[1],NANDU1TCPCommand,queue_size=1)
    mode=rospy.Publisher('/nandu1/set_mode',String,queue_size=1)
    relay=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);relay.settimeout(.1)
    height=rospy.get_param('/model_bundle/task/initial_up3_height')
    def wait(pred,timeout=5):
        end=time.monotonic()+timeout
        while time.monotonic()<end and not rospy.is_shutdown():
            if pred():return
            time.sleep(.01)
        raise AssertionError('Adaptive safety condition timed out')
    def command(yaw=0,vx=0):
        m=NANDU1TCPCommand();m.header.stamp=rospy.Time.now();m.relative_heading_rate=yaw;m.tcp_vx_base_body=vx;m.tcp_height_world=height;pub.publish(m)
    def reset():
        generation=evidence[0][2] if evidence[0] else -1
        mode.publish(String(data='RESET'))
        wait(lambda:diagnostics[0] is not None and diagnostics[0].level==0 and evidence[0] and evidence[0][2]>generation and (not adaptive or evidence[0][0] in priming))
        assert max(abs(x) for x in evidence[0][3:8])<1e-6
        if adaptive:assert max(abs(x) for x in priming[evidence[0][0]][129:212 if width==282 else 213])<1e-6
    try:
        wait(lambda:tracking[0] is not None and pub.get_num_connections()>0 and mode.get_num_connections()>0,10)
        for _ in range(20):command(0,.05);time.sleep(.02)
        time.sleep(.8)
        assert max(abs(v) for v in tracking[0][118:120])<1e-6
        checks.append(dict(name='command_timeout_zero_intent_hold_height',passed=True))
        command(float('nan'))
        wait(lambda:diagnostics[0] is not None and diagnostics[0].level==2)
        command();time.sleep(.04);assert diagnostics[0].level==2
        checks.append(dict(name='nonfinite_command_fault_latches',passed=True))
        reset();checks.append(dict(name='explicit_reset_clears_history_and_context',passed=True))
        for perturbation in ('drop','stale','duplicate','nan'):
            relay.sendto(perturbation.encode(),('127.0.0.1',15015))
            wait(lambda:diagnostics[0].level==2,2)
            began=time.monotonic()
            def damping():
                relay.sendto(b'status',('127.0.0.1',15015));status=json.loads(relay.recvfrom(4096)[0])
                return status['applied_kp_max']==0 and status['applied_dq_max']==0
            wait(damping,.15);delay=time.monotonic()-began
            relay.sendto(b'normal',('127.0.0.1',15015));time.sleep(.04);assert diagnostics[0].level==2
            reset();time.sleep(.4)
            checks.append(dict(name='up3_'+perturbation+'_reject_latch_reset',passed=True,damping_delay_s=delay))
        assert not bad_fault_context, 'FAULT diagnostic lost epoch/generation: '+str(bad_fault_context)
        checks.append(dict(name='all_fault_diagnostics_retain_session_context',passed=True))
    except Exception as exc:
        checks.append(dict(name='failure',passed=False,error=str(exc)));raise
    finally:
        relay.sendto(b'normal',('127.0.0.1',15015));relay.close()
        a.output.write_text(json.dumps(dict(scope='local SDK+ROS relay, not hardware',checks=checks),indent=2)+'\n')
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
