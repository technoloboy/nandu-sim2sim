#!/usr/bin/env python3
"""Test-only local SDK+ROS probe: short PASSIVE/POLICY cannot reuse commands."""
import argparse
import json
from pathlib import Path
import time
import rospy
from diagnostic_msgs.msg import DiagnosticArray
from std_msgs.msg import Float64MultiArray, String
from tron2_controllers.msg import NANDU1TCPCommand
from nandu1_adaptive_contract import adaptive_tasks


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',default='Isaac-NANDU1-Blind-Flat-v1-3-1-2-6',
        choices=adaptive_tasks()+('Isaac-NANDU1-Blind-Flat-v1-3-1-2',))
    a=p.parse_args(rospy.myargv()[1:]);rospy.init_node('adaptive_lifecycle_probe')
    if rospy.get_param('/tron2_hw/robot_ip','')!='127.0.0.1':raise RuntimeError('Local simulation only')
    if rospy.get_param('/model_bundle/task/task_id','')!=a.task:raise RuntimeError('Lifecycle probe task mismatch')
    adaptive=a.task in adaptive_tasks();intent=126 if adaptive else 118
    diag=[None];track=[None];reset=[None]
    rospy.Subscriber('/nandu1/diagnostics',DiagnosticArray,lambda m:diag.__setitem__(0,m.status[0]) if m.status else None,queue_size=10,tcp_nodelay=True)
    rospy.Subscriber('/nandu1/adaptive/tracking' if adaptive else '/nandu1/tracking',Float64MultiArray,lambda m:track.__setitem__(0,m.data),queue_size=10,tcp_nodelay=True)
    rospy.Subscriber('/nandu1/reset_evidence',Float64MultiArray,lambda m:reset.__setitem__(0,m.data),queue_size=10,tcp_nodelay=True)
    mode=rospy.Publisher('/nandu1/set_mode',String,queue_size=1,tcp_nodelay=True)
    from nandu1_mouse_protocol import command_interface
    command=rospy.Publisher(command_interface(a.task)[1],NANDU1TCPCommand,queue_size=1,tcp_nodelay=True)
    def wait(pred,timeout=5):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if pred():return
            time.sleep(.002)
        raise AssertionError('Lifecycle condition timed out')
    checks=[]
    try:
        wait(lambda:track[0] and reset[0] and mode.get_num_connections() and command.get_num_connections())
        m=NANDU1TCPCommand();m.relative_heading_rate=.1;m.tcp_vx_base_body=.05
        m.tcp_height_world=rospy.get_param('/model_bundle/task/initial_up3_height')+.02
        for _ in range(10):command.publish(m);time.sleep(.02)
        wait(lambda:track[0][intent]>.05 and track[0][intent+1]>.01)
        previous=reset[0][2];last_sent=time.monotonic()
        mode.publish(String(data='PASSIVE'))
        wait(lambda:diag[0] is not None and diag[0].message=='PASSIVE',.2)
        checks.append('passive_state_visible')
        mode.publish(String(data='POLICY'))
        wait(lambda:reset[0][2]>previous and track[0][0]>=reset[0][0],.2)
        assert time.monotonic()-last_sent<.5, 'Probe must exercise still-fresh old command'
        assert abs(track[0][intent])+abs(track[0][intent+1])<1e-7
        assert abs(track[0][intent+2]-rospy.get_param('/model_bundle/task/initial_up3_height'))<1e-7
        checks.append('reentry_new_generation_and_zero_intent')
        command.publish(NANDU1TCPCommand(relative_heading_rate=float('nan')))
        wait(lambda:diag[0].level==2)
        checks.append('fault_diagnostic_persistable')
        time.sleep(.15)
        mode.publish(String(data='RESET'));wait(lambda:diag[0].level==0)
        mode.publish(String(data='PASSIVE'));wait(lambda:diag[0].message=='PASSIVE')
        time.sleep(.15)
        result=dict(passed=True,checks=checks)
    except Exception as exc:
        result=dict(passed=False,checks=checks,error=str(exc));raise
    finally:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
