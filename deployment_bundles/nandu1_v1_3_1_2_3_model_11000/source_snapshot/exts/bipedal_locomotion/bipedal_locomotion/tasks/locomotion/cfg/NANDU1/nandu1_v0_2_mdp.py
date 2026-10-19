"""Isolated v0-2 robustness: resets, measurements, disturbances and state."""
import math
from pathlib import Path
import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply

from . import nandu1_mdp as base
from . import nandu1_v0_2_math as math02
from .nandu1_v0_1_mdp import Nandu1V01Command, Nandu1V01CommandCfg

SCHEMA = 'nandu1_v0_2_robustness'
URDF = Path(__file__).resolve().parents[4] / 'assets/usd/NANDU1/urdf/robot.urdf'

@configclass
class Nandu1V02CommandCfg(Nandu1V01CommandCfg):
    robustness_enabled: bool = True

class Nandu1V02Command(Nandu1V01Command):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self._wheel_contact_ids=math02.wheel_sensor_indices(env.scene.sensors['contact_forces'].body_names)
        print(f'[NANDU v0-2] robot wheel IDs={self._wheel_ids}; sensor wheel IDs={self._wheel_contact_ids}')
        for name in ('Reset/rejection_rate','Reset/fallback','Reset/position_offset_max','Reset/joint_speed_max',
                     'Disturbance/active','Disturbance/force_norm','Disturbance/torque_norm',
                     'Disturbance/recovery_seconds','Disturbance/recovery_timeout',
                     'Tracking/clean_error','Tracking/disturbed_error','Tracking/recovery_error',
                     'Tracking/clean_fraction','Tracking/disturbed_fraction','Tracking/recovery_fraction'):
            self.metrics[name]=torch.zeros(self.num_envs,device=self.device)
        for group in ('clean','disturbed','recovery'):
            for component in ('yaw','vx','vz','tilt'):
                self.metrics[f'Tracking/{group}_{component}_square_sum']=torch.zeros(self.num_envs,device=self.device)
        for key in math02.STAGES[0]:
            self.metrics[f'Robustness/{key}_limit']=torch.zeros(self.num_envs,device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        k=1./self._env.max_episode_length
        for key,value in math02.STAGES[self.stage].items():
            self.metrics[f'Robustness/{key}_limit']+=value*k
        scheduler=getattr(self._env,'_nandu_v02_disturbance',None)
        if scheduler is not None:
            active=scheduler.active;recover=scheduler.recovering & ~active
            clean=~(active|recover)
            velocity=base._heading_velocity(self._robot.data.root_quat_w,self._robot.data.body_lin_vel_w[:,self._up3_id])
            gravity=base.imu_projected_gravity(self._env,SceneEntityCfg('up3_imu'))
            errors=torch.stack((self._robot.data.root_ang_vel_w[:,2]-self.command[:,0],
                                velocity[:,0]-self.command[:,1],velocity[:,2]-self.command[:,2],
                                torch.acos((-gravity[:,2]).clamp(-1,1))),-1)
            error=(errors.abs()/errors.new_tensor([.25,.14,.10,.17])).mean(-1)
            for label,mask in [('clean',clean),('disturbed',active),('recovery',recover)]:
                self.metrics[f'Tracking/{label}_error']+=error*mask*k
                self.metrics[f'Tracking/{label}_fraction']+=mask*k
                for i,component in enumerate(('yaw','vx','vz','tilt')):
                    self.metrics[f'Tracking/{label}_{component}_square_sum']+=errors[:,i].square()*mask*k
            self.metrics['Disturbance/active']+=active*k
            self.metrics['Disturbance/force_norm']+=scheduler.applied_force.norm(dim=-1)*k
            self.metrics['Disturbance/torque_norm']+=scheduler.applied_torque.norm(dim=-1)*k

    def state_dict(self):
        state=super().state_dict();state['task_schema']=SCHEMA
        return state

    def load_state_dict(self,state):
        if state.get('task_schema')!=SCHEMA:
            raise ValueError('v0-2 resume requires a v0-2 checkpoint. For v0-1 use --transfer_checkpoint (stage resets to 0).')
        super().load_state_dict(state)
        # The simulator was initially spawned at stage 0. Resample its fresh
        # episode using the restored stage without recording a fake failure or
        # evaluating the curriculum a second time.
        env=self._env;ids=torch.arange(self.num_envs,device=self.device)
        env.scene.reset(ids)
        env.event_manager.apply(mode='reset',env_ids=ids,global_env_step_count=env.common_step_counter)
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        env.action_manager.reset(ids);env.observation_manager.reset(ids)
        self.reset(ids);env.event_manager.reset(ids)
        env.episode_length_buf[ids]=0

Nandu1V02CommandCfg.class_type=Nandu1V02Command

def noisy_observation(env,kind,command_name,asset_cfg=None,sensor_cfg=None):
    """Same draw for policy/history at the same observation time."""
    robot=env.scene['robot'];term=env.command_manager.get_term(command_name)
    if kind in ('gyro','gravity'):
        sensor=env.scene.sensors[sensor_cfg.name]
        value=sensor.data.ang_vel_b if kind=='gyro' else sensor.data.projected_gravity_b
    elif kind=='joint_pos':value=robot.data.joint_pos[:,asset_cfg.joint_ids]-robot.data.default_joint_pos[:,asset_cfg.joint_ids]
    else:value=robot.data.joint_vel[:,asset_cfg.joint_ids]
    if not env.cfg.observations.policy.enable_corruption:return value
    key=(env.common_step_counter,kind,sensor_cfg.name if sensor_cfg else 'robot')
    cache=getattr(env,'_nandu_v02_noise',{})
    if cache.get('step')!=env.common_step_counter:
        cache={'step':env.common_step_counter};env._nandu_v02_noise=cache
    if key not in cache:
        noise_key={'gyro':'gyro_noise','gravity':'gravity_noise','joint_pos':'joint_pos_noise','joint_vel':'joint_vel_noise'}[kind]
        cache[key]=math02.uniform_noise(value,math02.STAGES[term.stage][noise_key])
    return cache[key]

class RobustDomainRandomization(base.StageDomainRandomization):
    def __call__(self,env,env_ids,command_name,asset_cfg,material_buckets_per_stage=48):
        super().__call__(env,env_ids,command_name,asset_cfg,material_buckets_per_stage)
        ids=env_ids.to('cpu',dtype=torch.long)
        com=self._robot.root_physx_view.get_coms()
        com[ids]=self._default_com[ids]
        amplitude=math02.STAGES[env.command_manager.get_term(command_name).stage]['com']
        com[ids,:,:3]+=torch.empty(len(ids),self._robot.num_bodies,3).uniform_(-amplitude,amplitude)
        self._robot.root_physx_view.set_coms(com,ids)

class RobustReset(ManagerTermBase):
    def __init__(self,cfg,env):
        super().__init__(cfg,env);self.robot=env.scene['robot']
        self.geometry=math02.ResetGeometry(URDF,self.robot.joint_names,env.device)
        valid,z=self.geometry.evaluate(self.robot.data.default_joint_pos[:1],torch.eye(3,device=env.device)[None])
        if not bool(valid.all()):raise ValueError('NANDU default reset fails mesh geometry audit')
        self.default_z=float(z[0])

    def __call__(self,env,env_ids,command_name):
        ids=env_ids.to(env.device,dtype=torch.long);n=len(ids)
        if not n:return
        term=env.command_manager.get_term(command_name);stage=math02.STAGES[term.stage]
        r=self.robot;defaults=r.data.default_joint_pos[ids]
        limits=r.data.soft_joint_pos_limits[ids]
        severity=torch.rand(n,device=env.device)
        severity=torch.where(severity<.6,.25,torch.where(severity<.9,.6,1.))
        enabled=term.cfg.robustness_enabled
        if not enabled:severity.zero_()
        q=defaults.clone();rpy=torch.zeros(n,3,device=env.device);z=torch.full((n,),self.default_z,device=env.device)
        remaining=torch.arange(n,device=env.device);attempts=torch.zeros(n,device=env.device)
        for _ in range(16):
            if len(remaining)==0:break
            count=len(remaining);amplitude=severity[remaining,None]*stage['joint_offset']
            lo=torch.maximum(defaults[remaining]-amplitude,limits[remaining,:,0])
            hi=torch.minimum(defaults[remaining]+amplitude,limits[remaining,:,1])
            candidate=lo+torch.rand_like(lo)*(hi-lo)
            candidate_rpy=torch.zeros(count,3,device=env.device)
            candidate_rpy[:,:2]=torch.empty(count,2,device=env.device).uniform_(-1,1)*severity[remaining,None]*math.radians(stage['tilt_deg'])
            candidate_rpy[:,2]=torch.empty(count,device=env.device).uniform_(-math.pi,math.pi) if enabled else 0
            legal,heights=self.geometry.evaluate(candidate,math02.rpy_rotation(candidate_rpy))
            attempts[remaining]+=1
            accepted=remaining[legal];q[accepted]=candidate[legal];rpy[accepted]=candidate_rpy[legal];z[accepted]=heights[legal]
            remaining=remaining[~legal]
        root=r.data.default_root_state[ids].clone();root[:,:3]=env.scene.env_origins[ids]
        if enabled:root[:,:2]+=torch.empty(n,2,device=env.device).uniform_(-.1,.1)
        root[:,2]+=z
        root[:,3:7]=quat_from_euler_xyz(*rpy.unbind(-1))
        root[:,7:10]=torch.empty(n,3,device=env.device).uniform_(-1,1)*severity[:,None]*stage['root_v']
        root[:,10:13]=torch.empty(n,3,device=env.device).uniform_(-1,1)*severity[:,None]*stage['root_w']
        r.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
        r.write_root_pose_to_sim(root[:,:7],env_ids=ids);r.write_root_velocity_to_sim(root[:,7:],env_ids=ids)
        term.metrics['Reset/rejection_rate'][ids]=(attempts-1)/attempts.clamp_min(1)
        term.metrics['Reset/rejection_rate'][ids[remaining]]=1
        term.metrics['Reset/fallback'][ids]=0;term.metrics['Reset/fallback'][ids[remaining]]=1
        term.metrics['Reset/position_offset_max'][ids]=(q-defaults).abs().amax(-1)
        term.metrics['Reset/joint_speed_max'][ids]=0
        env._nandu_v02_noise={}

class RobustDisturbance(ManagerTermBase):
    """Single vectorized scheduler: five event modes, no overlapping pulses."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env);self.env=env;self.robot=env.scene['robot']
        self.up3=self.robot.find_bodies('up3')[0][0]
        n=env.num_envs;d=env.device
        self.wait=torch.empty(n,device=d).uniform_(8,12);self.enabled=torch.rand(n,device=d)>.2
        self.duration=torch.ones(n,device=d);self.elapsed=torch.ones(n,device=d)
        self.force=torch.zeros(n,3,device=d);self.torque=torch.zeros_like(self.force)
        self.applied_force=torch.zeros_like(self.force);self.applied_torque=torch.zeros_like(self.force)
        self.active=torch.zeros(n,device=d,dtype=torch.bool);self.recovering=self.active.clone()
        self.recovery_age=torch.zeros(n,device=d);self.recovery_hold=torch.zeros(n,device=d)
        self.events=0;self.recovered=0;self.failed=0
        env._nandu_v02_disturbance=self

    def __call__(self,env,env_ids,command_name):
        term=env.command_manager.get_term(command_name);s=math02.STAGES[term.stage];dt=env.step_dt
        self.wait-=dt;self.elapsed+=dt
        starting=((self.wait<=0)&self.enabled&(env.episode_length_buf*dt>1)&(term.stage>0)).nonzero().flatten()
        if len(starting):
            self.events+=len(starting);n=len(starting);device=env.device
            mode=torch.randint(0 if term.stage>=3 else 1,5,(n,),device=device)
            self.force[starting]=0;self.torque[starting]=0
            fids=starting[mode==1];tids=starting[mode==2]
            self.force[fids]=torch.empty(len(fids),3,device=device).uniform_(-s['force'],s['force'])
            self.torque[tids]=torch.empty(len(tids),3,device=device).uniform_(-s['torque'],s['torque'])
            if term.stage==4:
                combined=starting[torch.rand(n,device=device)<.1]
                self.force[combined]=torch.empty(len(combined),3,device=device).uniform_(-.7*s['force'],.7*s['force'])
                self.torque[combined]=torch.empty(len(combined),3,device=device).uniform_(-.7*s['torque'],.7*s['torque'])
                mode[torch.isin(starting,combined)]=1
            velocity=self.robot.data.root_vel_w[starting].clone()
            planar=mode==0
            # Preserve original planar push only from stage 3 onward.
            velocity[planar,:2]+=torch.empty(int(planar.sum()),2,device=device).uniform_(-.2,.2)*(term.stage>=3)
            z=mode==3;angular=mode==4
            velocity[z,2]+=torch.empty(int(z.sum()),device=device).uniform_(-s['z_kick'],s['z_kick'])
            velocity[angular,3:]+=torch.empty(int(angular.sum()),3,device=device).uniform_(-s['angular_kick'],s['angular_kick'])
            self.robot.write_root_velocity_to_sim(velocity,env_ids=starting)
            self.duration[starting]=torch.empty(n,device=device).uniform_(.10,.25)
            self.elapsed[starting]=0;self.wait[starting]=torch.empty(n,device=device).uniform_(8,12)
            self.recovering[starting]=True;self.recovery_age[starting]=0;self.recovery_hold[starting]=0
        self.active[:]=self.elapsed<self.duration
        gain=math02.pulse_gain(self.elapsed,self.duration)[:,None]
        self.applied_force[:]=self.force*gain
        self.applied_torque[:]=quat_apply(self.robot.data.body_quat_w[:,self.up3],self.torque)*gain
        self.robot.set_external_force_and_torque(self.applied_force[:,None],self.applied_torque[:,None],body_ids=[self.up3],is_global=True)
        self.force[~self.active]=0;self.torque[~self.active]=0
        self.recovery_age+=self.recovering*dt
        velocity=base._heading_velocity(self.robot.data.root_quat_w,self.robot.data.body_lin_vel_w[:,self.up3])
        tilt=base.imu_projected_gravity(env,SceneEntityCfg('up3_imu'))[:,:2].norm(dim=-1)
        yaw=(self.robot.data.root_ang_vel_w[:,2]-term.command[:,0]).abs()
        good=(velocity[:,0]-term.command[:,1]).abs()<.14
        good &= (velocity[:,2]-term.command[:,2]).abs()<.1
        good &= (tilt<math.sin(math.radians(10))) & (yaw<.25) & ~self.active
        self.recovery_hold=torch.where(good & self.recovering,self.recovery_hold+dt,0.)
        done=self.recovering & (self.recovery_hold>=.2)
        failed=self.recovering & (self.recovery_age>=3.) & ~done
        term.metrics['Disturbance/recovery_seconds'][done]=self.recovery_age[done]
        term.metrics['Disturbance/recovery_timeout'][failed]+=1
        self.recovered+=int(done.sum());self.failed+=int(failed.sum())
        self.recovering[done|failed]=False

    def reset(self,env_ids=None):
        ids=base._ids(slice(None) if env_ids is None else env_ids,len(self.wait),self.wait.device)
        self.wait[ids]=torch.empty(len(ids),device=self.wait.device).uniform_(8,12)
        self.enabled[ids]=torch.rand(len(ids),device=self.wait.device)>.2
        self.elapsed[ids]=1;self.duration[ids]=1
        self.force[ids]=0;self.torque[ids]=0;self.applied_force[ids]=0;self.applied_torque[ids]=0
        self.failed+=int(self.recovering[ids].sum())
        self.active[ids]=False;self.recovering[ids]=False;self.recovery_age[ids]=0;self.recovery_hold[ids]=0
        zero=torch.zeros(len(ids),1,3,device=self.wait.device)
        self.robot.set_external_force_and_torque(zero,zero,body_ids=[self.up3],env_ids=ids,is_global=True)

    def state_dict(self):
        return dict(events=self.events,recovered=self.recovered,failed=self.failed)

    def load_state_dict(self,state):
        self.events=state.get('events',0);self.recovered=state.get('recovered',0);self.failed=state.get('failed',0)
        # Physical world is freshly reset; replaying the old wrench would be wrong.
        self.reset()
