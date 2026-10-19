"""NANDU1 physical SDK bridge; policy inference lives in ROS."""
import csv
import json
import math
import socket
import threading
import time
import numpy as np
import mujoco
import mujoco.viewer
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes
import nandu1_contract as c
from nandu1_bundle import load_bundle,require_controller_runtime

class Nandu1Physics:
    def __init__(self,bundle, *, up3_total_mass=None, rear_mass=None, rear_x=None):
        self.manifest,path=load_bundle(bundle)
        self.model=mujoco.MjModel.from_xml_path(str(path))
        self.data=mujoco.MjData(self.model)
        self.jids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in c.JOINTS]
        self.aids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in c.JOINTS]
        if min(self.jids+self.aids)<0 or self.model.nu!=13: raise ValueError('NANDU1 requires 13 named joints/actuators')
        if not np.array_equal(self.model.jnt_range[self.jids],np.column_stack([
                self.manifest['task']['joint_position_lower'],self.manifest['task']['joint_position_upper']])):
            raise ValueError('Manifest physical limits do not match MJCF')
        self.qadr=self.model.jnt_qposadr[self.jids]
        self.vadr=self.model.jnt_dofadr[self.jids]
        self.bids={n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ['base_Link','up3']}
        self.sensors={}
        for prefix in ['base_imu','up3_imu']:
            for suffix,size in [('quat',4),('gyro',3),('acc',3)]:
                sid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SENSOR,prefix+'_'+suffix)
                if sid<0 or self.model.sensor_dim[sid]!=size: raise ValueError('Missing sensor '+prefix+'_'+suffix)
                self.sensors[prefix+'_'+suffix]=slice(self.model.sensor_adr[sid],self.model.sensor_adr[sid]+size)
        self.epoch=int(time.time()*1000)
        self.fixed_payload=None
        if self.manifest['task'].get('velocity_reference')=='fixed_tcp':
            from nandu1_v131 import FixedPayload
            self.fixed_payload=FixedPayload(self.model)
            parameters=self.manifest['task']['fixed_payload'].copy()
            for name,value in [('up3_total_mass',up3_total_mass),('rear_mass',rear_mass),('rear_x',rear_x)]:
                if value is not None:parameters[name]=value
            self.fixed_payload.apply(self.data,parameters['up3_total_mass'],parameters['rear_mass'],parameters['rear_x'])
        elif any(v is not None for v in (up3_total_mass,rear_mass,rear_x)):
            raise ValueError('Fixed payload overrides require v1.3.1')
        self.reset()

    def apply_fixed_payload(self, up3_total_mass=.67148, rear_mass=0., rear_x=-.275):
        if self.fixed_payload is None:raise ValueError('Fixed payload requires v1.3.1')
        self.fixed_payload.apply(self.data,up3_total_mass,rear_mass,rear_x)

    def reset(self):
        self.velocity_projection_events=0
        self.applied_kp_max=0.
        self.applied_dq_max=0.
        mujoco.mj_resetData(self.model,self.data)
        if self.fixed_payload is not None:
            p=self.fixed_payload.parameters
            self.fixed_payload.apply(self.data,p['up3_total_mass'],p['rear_mass'],p['rear_x'])
        self.data.qpos[:7]=[0,0,.65337,1,0,0,0]
        self.data.qpos[self.qadr]=c.DEFAULT_Q
        mujoco.mj_forward(self.model,self.data)
        self.epoch+=1

    def imu(self,prefix):
        return {suffix:self.data.sensordata[self.sensors[prefix+'_'+suffix]].tolist() for suffix in ['quat','gyro','acc']}

    def step(self):
        mujoco.mj_step(self.model,self.data)
        # MuJoCo has no native equivalent of PhysX maxJointVelocity. Project
        # only excessive joint speeds; report every projection in evaluation.
        velocity=self.data.qvel[self.vadr]
        count=int(np.count_nonzero(np.abs(velocity)>c.VELOCITY))
        self.velocity_projection_events+=count
        if count:
            self.data.qvel[self.vadr]=np.clip(velocity,-c.VELOCITY,c.VELOCITY)
            mujoco.mj_forward(self.model,self.data)
        return count

    def state(self):
        return self.data.qpos[self.qadr].copy(),self.data.qvel[self.vadr].copy(),self.data.qfrc_actuator[self.vadr].copy()

    def apply(self,q,dq,kp,kd,tau):
        q,dq,kp,kd,tau=[c.finite(x,13) for x in [q,dq,kp,kd,tau]]
        if np.any(kp<0) or np.any(kp>2000) or np.any(kd<0) or np.any(kd>200): raise ValueError('Invalid PD gains')
        actual_q,actual_dq,_=self.state()
        raw=kp*(q-actual_q)+kd*(dq-actual_dq)+tau
        self.data.ctrl[self.aids]=np.clip(raw,-c.EFFORT,c.EFFORT)
        self.applied_kp_max=float(np.max(kp))
        self.applied_dq_max=float(np.max(np.abs(dq)))
        return raw

    def packet(self,seq):
        fixed_tcp=self.manifest['task'].get('velocity_reference')=='fixed_tcp'
        link_origin=self.manifest['task'].get('velocity_reference')=='body_link_origin'
        velocities=[]
        for n in ['base_Link','up3']:
            v=np.empty(6)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.bids[n],v,0)
            if link_origin:
                bid=self.bids[n]
                v[3:]+=np.cross(v[:3],self.data.xpos[bid]-self.data.xipos[bid])
            elif fixed_tcp and n=='up3':
                from nandu1_v131 import tcp_state
                bid=self.bids[n]
                _,v[3:]=tcp_state(self.data.xpos[bid],self.data.xmat[bid],self.data.xipos[bid],v[3:],v[:3])
            # Training body_linear_velocity uses root_quat for BOTH labels.
            rotation=self.data.xmat[self.bids['base_Link']].reshape(3,3)
            velocities.extend((rotation.T@v[3:]).tolist())
        bid=self.bids['up3']; base=self.bids['base_Link']
        up_v=np.empty(6); base_v=np.empty(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,bid,up_v,0)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,base,base_v,0)
        height=float(self.data.xpos[bid,2])
        diagnostics={}
        if fixed_tcp:
            from nandu1_v131 import tcp_state
            tcp,velocity=tcp_state(self.data.xpos[bid],self.data.xmat[bid],self.data.xipos[bid],up_v[3:],up_v[:3])
            diagnostics=dict(up3_origin_world=self.data.xpos[bid].tolist(),up3_com_world=self.data.xipos[bid].tolist(),
                             tcp_world=tcp.tolist(),up3_com_velocity_world=up_v[3:].tolist(),tcp_velocity_world=velocity.tolist())
            up_v[3:]=velocity; height=float(tcp[2])
        if link_origin:up_v[3:]+=np.cross(up_v[:3],self.data.xpos[bid]-self.data.xipos[bid])
        velocity_body=self.data.xmat[base].reshape(3,3).T@up_v[3:]
        if link_origin:
            from nandu1_v14 import body_to_heading
            velocity_body=body_to_heading(velocity_body,self.data.xmat[base].reshape(3,3).T@np.array([0,0,-1]))
        command_truth=[float(base_v[2]),float(velocity_body[0]),height]
        q,dq,tau=self.state()
        return {'schema':'nandu1.dual-imu/v1','seq':seq,'epoch':self.epoch,'sim_time':self.data.time,
                'stamp_ns':time.time_ns(),'base':self.imu('base_imu'),'up3':self.imu('up3_imu'),
                'velocity_projection_events':self.velocity_projection_events,
                'applied_kp_max':self.applied_kp_max,'applied_dq_max':self.applied_dq_max,
                'truth':velocities+[height], 'command_truth':command_truth,
                **({'point_diagnostics':diagnostics,'fixed_payload':getattr(getattr(self,'fixed_payload',None),'parameters',None)} if fixed_tcp else {}),
                'q':q.tolist(),'dq':dq.tolist(),'tau':tau.tolist()}

def run_nandu1(args):
    import os
    if (args.ip or args.legacy_ip or '127.0.0.1')!='127.0.0.1': raise ValueError('NANDU1 sim2sim requires localhost')
    if os.getenv('ROBOT_TYPE') not in (None,'NANDU1'): raise ValueError('ROBOT_TYPE must be NANDU1 or unset')
    manifest,_=load_bundle(args.model_bundle,args.task)
    require_controller_runtime(manifest)
    physics=Nandu1Physics(args.model_bundle)
    robot=Robot(RobotType.Tron2,True)
    if not robot.init('127.0.0.1'): raise RuntimeError('SDK init failed')
    lock=threading.Lock(); command=None; received=0.; armed=False; reset_requested=False
    def callback(cmd):
        nonlocal command,received,armed
        try:
            arrays=[c.finite(getattr(cmd,k),13) for k in ['q','dq','Kp','Kd','tau']]
            mode=c.finite(cmd.mode,13)
            if np.any(mode!=0) or np.any(arrays[2]<0) or np.any(arrays[2]>2000) or np.any(arrays[3]<0) or np.any(arrays[3]>200):
                raise ValueError('Unsupported hybrid command')
        except (ValueError,AttributeError) as exc:
            print('[NANDU1] rejected command:',exc,flush=True)
            with lock: received=0
            return
        with lock:
            command=arrays;received=time.monotonic()
            if np.any(arrays[2]>0): armed=True
    robot.subscribeRobotCmdForSim(callback)
    def key(code):
        nonlocal reset_requested
        if code in (82,114): reset_requested=True
    udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    v=None
    if not args.headless:
        v=mujoco.viewer.launch_passive(physics.model,physics.data,key_callback=key)
        v.cam.distance=3;v.cam.elevation=-15;v.cam.lookat[:]=[0,0,.7]
    stream=None; writer=None
    if args.telemetry_output:
        args.telemetry_output.parent.mkdir(parents=True,exist_ok=True)
        stream=args.telemetry_output.open('w',newline='')
        writer=csv.writer(stream)
        writer.writerow(['sim_time','epoch']+[f'{p}_{i}' for p,n in [('truth',7),('q',13),('dq',13),('tau',13)] for i in range(n)])
    start=time.monotonic(); frame=seq=0;deadline=start
    timeout=min(.1,args.command_timeout)
    try:
        while v is None or v.is_running():
            now=time.monotonic()
            if args.duration>0 and now-start>=args.duration: break
            if reset_requested:
                with lock: command=None;received=0;armed=False
                physics.reset();reset_requested=False
            with lock: cmd=command;age=now-received;is_armed=armed
            if is_armed:
                if cmd is not None and age<timeout:
                    # v1.4: PD500 Hz with zero-order torque hold at 1kHz physics.
                    # Preserve the legacy v1.2 bridge's original PD scheduling.
                    if physics.manifest['task']['task_id'] not in (c.V14_TASK,c.V141_TASK) or frame%2==0:physics.apply(*cmd)
                else:
                    q,_,_=physics.state()
                    physics.apply(q,np.zeros(13),np.zeros(13),c.KD,np.zeros(13))
                physics.step()
            q,dq,tau=physics.state()
            if frame%2==0:
                state=datatypes.RobotState();state.q=q.tolist();state.dq=dq.tolist();state.tau=tau.tolist();state.stamp=time.time_ns()
                robot.publishRobotStateForSim(state)
                imu=datatypes.ImuData();b=physics.imu('base_imu')
                imu.quat=b['quat'];imu.gyro=b['gyro'];imu.acc=b['acc'];imu.stamp=state.stamp
                robot.publishImuDataForSim(imu)
            if frame%5==0:
                packet=physics.packet(seq);seq+=1
                udp.sendto(json.dumps(packet,allow_nan=False).encode(),('127.0.0.1',args.aux_port))
                if writer:
                    writer.writerow([packet['sim_time'],packet['epoch']]+packet['truth']+packet['q']+packet['dq']+packet['tau'])
                    stream.flush()
            if v is not None and frame%20==0: v.sync()
            frame+=1;deadline+=.001
            time.sleep(max(0,deadline-time.monotonic()))
            if time.monotonic()-deadline>.1: deadline=time.monotonic()
    finally:
        if v is not None: v.close()
        if stream: stream.close()
        udp.close()
