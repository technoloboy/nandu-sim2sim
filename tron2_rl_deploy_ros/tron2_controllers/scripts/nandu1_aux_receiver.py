#!/usr/bin/env python3
"""Localhost-only timestamped dual IMU receiver; truth is evaluation-only."""
import csv
import json
import math
import socket
import time
from pathlib import Path
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray,UInt64,String

def main():
    rospy.init_node('nandu1_aux_receiver')
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1',rospy.get_param('~port',15012)));sock.settimeout(.1)
    up=rospy.Publisher('/nandu1/up3_imu',Imu,queue_size=1)
    epoch_pub=rospy.Publisher('/nandu1/reset_epoch',UInt64,queue_size=1,latch=True)
    truth_pub=rospy.Publisher('/nandu1/evaluation_truth',Float64MultiArray,queue_size=1)
    raw_topic=rospy.get_param('~raw_packet_topic','')
    raw_pub=rospy.Publisher(raw_topic,String,queue_size=10) if raw_topic else None
    latest=[None]
    rospy.Subscriber('/nandu1/tracking',Float64MultiArray,lambda msg:latest.__setitem__(0,msg.data),queue_size=1)
    path=rospy.get_param('~telemetry_output','')
    stream=writer=reset_stream=None
    tracking_names=['tracking_stamp','tracking_epoch']+[f'command_{i}' for i in range(3)]+[f'estimate_{i}' for i in range(7)]+[f'action_{i}' for i in range(13)]
    tracking_names += [f'{prefix}_{i}' for prefix in ['q','dq','q_target','dq_target','torque_actual'] for i in range(13)]
    tracking_names += [f'{prefix}_{i}' for i in range(13) for prefix in ['torque_limited','torque_saturated']]
    profile=rospy.get_param('/model_bundle/task/runtime_profile','')
    if profile in ('nandu1_height_v1_4','nandu1_height_v1_4_1'):
        tracking_names += [f'command_trend_{i}' for i in range(3)]+[f'actor_estimate_heading_{i}' for i in range(7)]
        tracking_names += ['relative_heading','heading_target']+[f'external_intent_{i}' for i in range(3)]
    elif profile=='nandu1_height_v1_3_1':
        tracking_names += ['relative_heading','heading_target']+[f'external_intent_{i}' for i in range(3)]
    if path:
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        stream=open(path,'w',newline='');writer=csv.writer(stream)
        writer.writerow(['stamp','sim_time','epoch','sample_age','tracking_age','velocity_projection_events']+[f'truth_{i}' for i in range(7)]+[f'command_truth_{i}' for i in range(3)]+tracking_names)
        reset_stream=open(Path(path).with_suffix('.reset.csv'),'w',newline='');reset_writer=csv.writer(reset_stream)
        from nandu1_reset_evidence import FIELDS
        reset_writer.writerow(FIELDS)
        def reset_evidence(msg):
            if len(msg.data)==len(FIELDS) and all(math.isfinite(x) for x in msg.data):
                reset_writer.writerow(msg.data);reset_stream.flush()
        rospy.Subscriber('/nandu1/reset_evidence',Float64MultiArray,reset_evidence,queue_size=10)
    last_epoch=last_seq=-1
    try:
        while not rospy.is_shutdown():
            try:
                payload,address=sock.recvfrom(65535)
                p=json.loads(payload)
                if address[0]!='127.0.0.1' or p['schema']!='nandu1.dual-imu/v1':continue
                epoch=int(p['epoch']);seq=int(p['seq']);stamp=p['stamp_ns']/1e9
                if epoch<last_epoch or (epoch==last_epoch and seq<=last_seq):continue
                age=time.time()-stamp
                if age<-.05 or age>.1:continue
                b=p['up3'];numbers=b['quat']+b['gyro']+b['acc']+p['truth']+p['command_truth']
                if len(b['quat'])!=4 or len(b['gyro'])!=3 or len(b['acc'])!=3 or len(p['truth'])!=7 or len(p['command_truth'])!=3 or not all(math.isfinite(x) for x in numbers):continue
                norm=sum(x*x for x in b['quat'])
                if not .5<norm<1.5:continue
                last_epoch,last_seq=epoch,seq
                epoch_pub.publish(UInt64(data=epoch))
                # Header.seq is uint32 and managed by rospy; epochs travel on UInt64.
                msg=Imu();msg.header.stamp=rospy.Time.from_sec(stamp);msg.header.frame_id='up3'
                msg.orientation.w,msg.orientation.x,msg.orientation.y,msg.orientation.z=b['quat']
                msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z=b['gyro']
                msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z=b['acc']
                up.publish(msg)
                truth=Float64MultiArray();truth.data=[stamp,p['sim_time'],epoch]+p['truth']+p['command_truth'];truth_pub.publish(truth)
                if raw_pub is not None:raw_pub.publish(String(data=payload.decode('utf-8')))
                t=latest[0]
                if writer and t and len(t)==len(tracking_names) and int(t[1])==epoch and abs(stamp-t[0])<.04:
                    writer.writerow([stamp,p['sim_time'],epoch,age,stamp-t[0],p.get('velocity_projection_events',-1)]+p['truth']+p['command_truth']+list(t))
                    stream.flush()
            except socket.timeout:continue
            except (ValueError,KeyError,TypeError) as exc:rospy.logwarn_throttle(1,'Rejected NANDU1 auxiliary packet: %s',exc)
    finally:
        sock.close()
        if stream:stream.close()
        if reset_stream:reset_stream.close()

if __name__=='__main__':main()
