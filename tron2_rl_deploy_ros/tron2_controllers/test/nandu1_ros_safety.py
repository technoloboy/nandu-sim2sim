#!/usr/bin/env python3
"""Live ROS acceptance: requires the dedicated test relay and local simulation."""
import argparse
import json
from pathlib import Path
import socket
import time
import rospy
from diagnostic_msgs.msg import DiagnosticArray
from std_msgs.msg import Float64MultiArray,String
from tron2_controllers.msg import NANDU1Command

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(rospy.myargv()[1:]);rospy.init_node('nandu1_ros_safety')
    tracking=[None];diagnostic=[None]
    rospy.Subscriber('/nandu1/tracking',Float64MultiArray,lambda m:tracking.__setitem__(0,m.data),queue_size=1)
    rospy.Subscriber('/nandu1/diagnostics',DiagnosticArray,lambda m:diagnostic.__setitem__(0,m.status[0]) if m.status else None,queue_size=1)
    pub=rospy.Publisher('/nandu1/command',NANDU1Command,queue_size=1)
    mode=rospy.Publisher('/nandu1/set_mode',String,queue_size=1)
    relay=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);relay.settimeout(.2);checks=[]
    def wait(predicate,timeout=5):
        end=time.monotonic()+timeout
        while time.monotonic()<end and not rospy.is_shutdown():
            if predicate():return
            time.sleep(.01)
        raise AssertionError('ROS acceptance wait timed out')
    def command(yaw=0,vx=0,height=.88564531185):
        msg=NANDU1Command();msg.header.stamp=rospy.Time.now()
        msg.yaw_rate_world_z=yaw;msg.up3_vx_base_body=vx;msg.up3_height_world=height;pub.publish(msg)
    def reset():
        mode.publish(String(data='RESET'))
        wait(lambda:diagnostic[0] is not None and diagnostic[0].level==0)
    try:
        wait(lambda:tracking[0] is not None and pub.get_num_connections()>0 and mode.get_num_connections()>0,20)
        for _ in range(50):command(.1,.05);time.sleep(.02)
        height=tracking[0][4];time.sleep(.9)
        assert abs(tracking[0][2])<1e-6 and abs(tracking[0][3])<1e-6
        assert abs(tracking[0][4]-height)<1e-6
        checks.append({'name':'command_timeout_zero_velocity_hold_height','passed':True})
        command(float('nan'))
        wait(lambda:diagnostic[0] is not None and diagnostic[0].level==2)
        command();time.sleep(.12)
        assert diagnostic[0].level==2
        checks.append({'name':'non_finite_command_fault_latches','passed':True})
        reset();checks.append({'name':'explicit_fault_reset','passed':True})
        for perturbation in ['drop','stale','duplicate','nan']:
            relay.sendto(perturbation.encode(),('127.0.0.1',15015))
            wait(lambda:diagnostic[0].level==2,2)
            reason=diagnostic[0].message
            began=time.monotonic()
            def physical_damping():
                relay.sendto(b'status',('127.0.0.1',15015))
                status=json.loads(relay.recvfrom(4096)[0])
                return (status['stamp_ns'] is not None and time.time()-status['stamp_ns']/1e9<.03
                        and status['applied_kp_max']==0 and status['applied_dq_max']==0)
            wait(physical_damping,.06)
            damping_delay=time.monotonic()-began
            relay.sendto(b'normal',('127.0.0.1',15015));time.sleep(.12)
            assert diagnostic[0].level==2
            checks.append({'name':'up3_'+perturbation+'_reject_and_latch','passed':True,'reason':reason,
                           'physical_damping_confirmed_within_s':damping_delay})
            reset();time.sleep(.12)
        checks.append({'name':'four_transport_fault_resets','passed':True})
    except Exception as exc:
        checks.append({'name':'acceptance_failure','passed':False,'error':str(exc)})
        raise
    finally:
        relay.sendto(b'normal',('127.0.0.1',15015));relay.close()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({'scope':'local SDK + ROS, not hardware','checks':checks},indent=2)+'\n')
        print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
