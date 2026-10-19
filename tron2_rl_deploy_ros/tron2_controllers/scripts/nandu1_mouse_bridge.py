#!/usr/bin/env python3
"""Local sim2sim mouse intents -> existing task-specific ROS command messages."""
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tron2_mujoco_sim'))
from nandu1_mouse_protocol import CommandGate, ControllerStatus, JsonChannel, frame, command_interface


def main():
    import rospy
    import rosgraph
    from diagnostic_msgs.msg import DiagnosticArray
    from std_msgs.msg import Float64MultiArray, String
    from tron2_controllers.msg import NANDU1Command, NANDU1TCPCommand, NANDU1V14Command
    from nandu1_command import assign_command
    import nandu1_contract as contract
    rospy.init_node('nandu1_mouse_bridge')
    from nandu1_adaptive_contract import deployment_profile
    task = deployment_profile(rospy.get_param('~task'))['task_id']
    if rospy.get_param('/model_bundle/task/task_id','') != task:
        raise RuntimeError('Mouse task differs from validated running bundle')
    if rospy.get_param('/tron2_hw/robot_ip','') != '127.0.0.1':
        raise RuntimeError('Mouse entry is localhost MuJoCo sim2sim only')
    family, topic = command_interface(task)
    lower = rospy.get_param('/model_bundle/task/command_min')
    upper = rospy.get_param('/model_bundle/task/command_max')
    initial = rospy.get_param('/model_bundle/task/initial_up3_height')
    session = rospy.get_param('~session')
    path = rospy.get_param('~socket')
    gate = CommandGate(session,task,lower,upper,initial)
    status = ControllerStatus(family,initial)
    master = rosgraph.Master(rospy.get_name())
    if any(name==topic and nodes for name,nodes in master.getSystemState()[0]):
        raise RuntimeError('Another command publisher is active; mouse source refuses to compete')
    message_type = {'body':NANDU1Command,'tcp':NANDU1TCPCommand,'heading':NANDU1V14Command}[family]
    pub = rospy.Publisher(topic,message_type,queue_size=1)
    mode_pub = rospy.Publisher('/nandu1/set_mode',String,queue_size=1)
    owner = dict(session=session,node=rospy.get_name(),topic=topic)
    rospy.set_param('/nandu1/mouse_owner',owner)

    def track(msg):
        try: status.tracking(msg.data,time.monotonic())
        except ValueError as exc: rospy.logwarn_throttle(1,'Mouse rejects tracking: %s',exc)

    def reset(msg):
        try: status.reset(msg.data)
        except ValueError as exc: rospy.logwarn_throttle(1,'Mouse rejects reset evidence: %s',exc)

    def diagnostics(msg):
        for entry in msg.status:
            if entry.name=='nandu1/policy':
                status.diagnostic(entry.message if entry.level==0 else 'FAULT: '+entry.message,time.monotonic())

    rospy.Subscriber('/nandu1/tracking',Float64MultiArray,track,queue_size=1)
    rospy.Subscriber('/nandu1/reset_evidence',Float64MultiArray,reset,queue_size=1)
    rospy.Subscriber('/nandu1/diagnostics',DiagnosticArray,diagnostics,queue_size=1)
    channel = None; sequence = 0; next_connect = 0.; next_conflict_check = 0.
    conflict = False

    def publish(values):
        msg = message_type(); msg.header.stamp = rospy.Time.now()
        assign_command(msg,family,values); pub.publish(msg)

    try:
        while not rospy.is_shutdown():
            now = time.monotonic()
            if now >= next_conflict_check:
                conflict = any(name==topic and any(node!=rospy.get_name() for node in nodes)
                               for name,nodes in master.getSystemState()[0])
                next_conflict_check = now+.1
            current = status.snapshot(now)
            gate.sync(current['ready'] and not conflict,current['key'],current['initial_height'])
            if conflict: gate.neutral('Conflicting ROS publisher: mouse output disabled')
            if channel is None and now >= next_connect:
                sock = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); sock.settimeout(.1)
                try:
                    sock.connect(path); channel = JsonChannel(sock)
                    gate.neutral('Reconnected: release and drag again',renew=True)
                except OSError:
                    sock.close(); next_connect = now+.1
            if channel is not None:
                try:
                    gate.values(now)
                    for packet in channel.receive():
                        if gate.key is None: continue
                        if gate.receive(packet,now) and not conflict:
                            mode_pub.publish(String(data='RESET'))
                        if gate.closing and not conflict:
                            mode_pub.publish(String(data='PASSIVE'))
                    values = gate.values(now)
                    ready = bool(current['ready'] and not conflict and not gate.needs_release)
                    if channel.send(frame('status',session,task,seq=sequence,key=current['key'],
                        ready=ready,arm_token=gate.arm_token,lower=list(lower),upper=list(upper),applied=current['applied'],
                        mode=current['mode'],reason=gate.reason)):
                        sequence += 1
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    gate.neutral('IPC disconnected: '+str(exc)); channel.sock.close(); channel=None
            if not conflict:
                publish(gate.values(now))
            time.sleep(.02)
    finally:
        if not conflict:
            try: publish([0.,0.,gate.request[2]])
            except Exception: pass
        if channel is not None: channel.sock.close()
        try:
            if rospy.get_param('/nandu1/mouse_owner',{}) == owner:
                rospy.delete_param('/nandu1/mouse_owner')
        except Exception: pass


if __name__ == '__main__':
    main()
