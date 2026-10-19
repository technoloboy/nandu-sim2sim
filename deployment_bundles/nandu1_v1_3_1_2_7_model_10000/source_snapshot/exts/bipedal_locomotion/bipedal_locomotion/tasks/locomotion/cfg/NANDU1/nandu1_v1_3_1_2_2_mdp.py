"""Isolated E child: causal reference, equivalent balance and payload supervision.

Privileged geometry below never becomes an Actor observation. Student and slow
memory are runner-owned and consume only the existing noisy measurement cache.
"""
from __future__ import annotations
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from rsl_rl.nandu_adaptive_balance import math as am
from rsl_rl.nandu_adaptive_balance.modes import TrainingModes
from . import nandu1_v1_3_1_mdp as parent
from . import nandu1_v1_3_1_math as payload
from . import nandu1_v1_mdp as v1
from . import nandu1_mdp as v0
from . import nandu1_math as nm
from . import nandu1_v1_robust_mdp as robust
from . import nandu1_v1_1_math as coordination

COMMAND_NAME = parent.COMMAND_NAME


def quiet(value, low, high):
    x = ((value.abs()-low)/(high-low)).clamp(0,1)
    return 1-x.square()*(3-2*x)


class AdaptivePayloadRandomization(parent.TCPPayloadDomainRandomization):
    """Same physical sampler; cache actual and independent counterfactual labels at reset."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.body_ids = [self._payload_base_id,self._payload_up3_id]
        dev = env.device
        self.nom = (self.nominal_masses[:,self.body_ids].to(dev),
                    self.nominal_coms[:,self.body_ids,:3].to(dev),
                    self.nominal_inertias[:,self.body_ids].to(dev))
        samples=[]
        # Additive origin inertias attain the relevant extrema at payload endpoints.
        for u in (.6,6.):
            for r in (0.,5.):
                for x in (-.35,-.20):
                    m,c,i=self._properties(torch.tensor([u]),torch.tensor([r]),torch.tensor([x]),torch.tensor([0]))
                    samples.append(am.physical_descriptor(m,c,i,*(t[:1].cpu() for t in self.nom)))
        extremes=torch.cat(samples,0).to(dev)
        self.scale=extremes.abs().amax(0).clamp_min(1e-6)
        self.active=extremes.abs().amax(0)>1e-8
        n=env.num_envs
        env._nandu_adaptive_payload=dict(teacher_params=torch.zeros(n,20,device=dev),
            counterfactual_teacher_params=torch.zeros(n,20,device=dev),
            cf_mass=self.nom[0].clone(),cf_com=self.nom[1].clone(),normalization=self.scale.clone(),
            constant_mask=~self.active)

    def _properties(self,u,r,x,ids):
        ids=ids.cpu()
        b,a=self.body_ids
        bm,bc,bi=payload.rear_mass_properties(self.nominal_masses[ids,b],self.nominal_coms[ids,b,:3],self.nominal_inertias[ids,b],r,x)
        um,uc,ui=payload.up3_mass_properties(self.nominal_masses[ids,a],self.nominal_coms[ids,a,:3],self.nominal_inertias[ids,a],u)
        return torch.stack((bm,um),1),torch.stack((bc,uc),1),torch.stack((bi,ui),1)

    def __call__(self,env,env_ids,command_name,asset_cfg,material_buckets_per_stage=48,excluded_body_ids=()):
        super().__call__(env,env_ids,command_name,asset_cfg,material_buckets_per_stage,excluded_body_ids)
        ids=env_ids.to('cpu',dtype=torch.long)
        if not len(ids):return
        d=ids.to(env.device);view=self._robot.root_physx_view
        m=view.get_masses()[ids][:,self.body_ids].to(env.device)
        c=view.get_coms()[ids][:,self.body_ids,:3].to(env.device)
        i=view.get_inertias()[ids].reshape(len(ids),self._robot.num_bodies,3,3).transpose(-1,-2)[:,self.body_ids].to(env.device)
        actual=am.physical_descriptor(m,c,i,*(t[d] for t in self.nom))
        p=payload.sample_payload(len(ids),int(env.command_manager.get_term(command_name).stage))
        cm,cc,ci=self._properties(p['up3_mass'],p['rear_mass'],p['rear_x'],ids)
        cf=am.physical_descriptor(cm.to(env.device),cc.to(env.device),ci.to(env.device),*(t[d] for t in self.nom))
        cache=env._nandu_adaptive_payload
        cache['teacher_params'][d]=actual/self.scale*self.active
        cache['counterfactual_teacher_params'][d]=cf/self.scale*self.active
        cache['cf_mass'][d]=cm.to(env.device);cache['cf_com'][d]=cc.to(env.device)


class AdaptiveBalanceCommand(parent.Nandu1TCPHeightCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.processor=am.CausalCommandProcessor(self.num_envs,self.device)
        self.modes=TrainingModes(self.num_envs,self.device)
        self.geometry=am.GeometryTrajectoryCache(self.num_envs,self.device)
        self.tracking_statistics_mask=self.modes.normal_mask()
        self._geometry_step=None
        self._coordination_step=None
        self._drift_step=None
        self._gate_step=None
        self._statistics_step=None
        self.stop_age=torch.zeros(self.num_envs,device=self.device)
        self.pulse_quiet_age=torch.zeros_like(self.stop_age)
        self.velocity_bias=torch.zeros(self.num_envs,3,device=self.device)
        self.balance_gate=torch.zeros_like(self.stop_age)
        self.fine_gate=torch.zeros_like(self.stop_age)
        self.intent_gate=torch.zeros_like(self.stop_age)
        self.com_cost=torch.zeros_like(self.stop_age)
        self._drift_start=torch.zeros(self.num_envs,3,device=self.device)
        self._drift_previous=torch.zeros_like(self._drift_start)
        self._drift_time=torch.zeros_like(self.stop_age)
        self._drift_path=torch.zeros_like(self.stop_age)
        self._wheel_net=torch.zeros(self.num_envs,2,device=self.device)
        self._wheel_abs=torch.zeros_like(self._wheel_net)
        self._wheel_previous=torch.zeros_like(self._wheel_net)
        self._wheel_quat_previous=torch.zeros(self.num_envs,2,4,device=self.device)
        self._wheel_quat_previous[:,:,0]=1
        self._wheel_absolute_roll=torch.zeros_like(self._wheel_net)
        self._yaw_previous=torch.zeros_like(self.stop_age)
        self._yaw_net=torch.zeros_like(self.stop_age)
        self._yaw_path=torch.zeros_like(self.stop_age)
        self._reference_quiet_age=torch.zeros_like(self.stop_age)
        self._point_start=torch.zeros(self.num_envs,5,3,device=self.device)
        self._point_previous=torch.zeros_like(self._point_start)
        self._point_path=torch.zeros(self.num_envs,5,device=self.device)
        self._point_velocity=torch.zeros_like(self._point_start)
        self._drift_active=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._wheel_joint_ids=self._robot.find_joints(['wheel_L_Joint','wheel_R_Joint'],preserve_order=True)[0]
        for key in ('Balance/theta','Balance/theta_dot','Balance/h','Balance/vG','Balance/valid',
                    'Static/capture_gate','Static/fine_gate','Static/bias_vx','Static/bias_yaw',
                    'Drift/tcp_net_speed','Drift/tcp_activity','Drift/wheel_activity',
                    'Drift/score_tcp_net','Drift/score_tcp_activity','Drift/score_wheel_activity','Drift/valid',
                    'Drift/api_tcp_z_residual','Drift/yaw_slope','Drift/yaw_activity','Drift/score_yaw',
                    'Drift/wheel_net_proxy','Drift/wheel_absolute_roll_proxy',
                    'Mode/normal','Mode/static','Mode/stop','Mode/cycle'):
            self.metrics[key]=torch.zeros(self.num_envs,device=self.device)
        for point in ('base_origin','base_com','up3_origin','up3_com','fixed_tcp'):
            for component in ('net_abs_x','net_abs_y','net_abs_z','activity_xy'):
                self.metrics[f'Drift/{point}/{component}']=torch.zeros(self.num_envs,device=self.device)

    def _resample_command(self,env_ids):
        if self._external_control:return
        ids=v1._ids(env_ids,self.num_envs,self.device)
        if hasattr(self,'modes'):
            ids=ids[(self.modes.mode[ids]==TrainingModes.NORMAL) |
                    ((self.modes.mode[ids]==TrainingModes.STOP)&(self.modes.age[ids]<self.modes.stop_time[ids]))]
        if len(ids):super()._resample_command(ids)

    def _update_command(self):
        previous=self._command.clone()
        if not self._external_control:
            self.modes.advance(self._env.step_dt,step_id=self._env.common_step_counter)
            self.target_command.copy_(self.modes.targets(self.target_command))
        gravity=robust.noisy_observation(self._env,'gravity',COMMAND_NAME,sensor_cfg=SceneEntityCfg('base_imu'))
        gyro=robust.noisy_observation(self._env,'gyro',COMMAND_NAME,sensor_cfg=SceneEntityCfg('base_imu'))
        self._command.copy_(self.processor.step(self.target_command,self._env.step_dt,
            step_id=self._env.common_step_counter,gravity_b=gravity,gyro_b=gyro,stage=self.active_stage,
            yaw_limit=self.STAGE_LIMITS[self.active_stage][0],height_rate_limit=self.cfg.height_rate_limits[self.active_stage]))
        self.commanded_height_rate.copy_(self.processor.height_rate)
        # Parent diagnostics must report the active controller, not its unused predecessor.
        self.heading_controller.heading.copy_(self.processor.heading)
        self.heading_controller.heading_target.copy_(self.processor.heading_target)
        self.heading_controller.yaw_rate.copy_(self._command[:,0])
        self.command_acceleration.copy_((self._command-previous)/self._env.step_dt)
        self.height_rate_limited.copy_((self.target_command[:,2]-self._command[:,2]).abs()>1e-4)
        self.sampled_height_target.copy_(self.target_command[:,2])
        self.height_step_segment.zero_();self.pending_height_step.zero_()
        self.filtered_base_vx_target.add_(self._env.step_dt/(.8+self._env.step_dt)*(self._command[:,1]-self.filtered_base_vx_target))

    def set_external_command(self,command):
        previous=self._command.clone()
        super().set_external_command(command)
        self._command.copy_(previous)  # Only physical command step advances causal processor.

    def _com_quality(self):
        if not hasattr(self,'com_cost') or self._geometry_step!=self._env.common_step_counter:
            return super()._com_quality()
        return torch.exp(-self.com_cost)

    def ensure_geometry(self):
        step=self._env.common_step_counter
        if self._geometry_step==step:return self._geo
        d=self._robot.data;dt=self._env.step_dt
        masses=parent.actual_body_mass(self._env)
        mask=torch.ones(self._robot.num_bodies,device=self.device,dtype=torch.bool);mask[self._wheel_ids]=False
        weighted=(d.body_com_pos_w[:,mask]*masses[:,mask,None]).sum(1)
        total=masses[:,mask].sum(1)
        com=weighted/total[:,None].clamp_min(1e-6)
        # Both wheel joints in the immutable NANDU1 URDF use local axis (-1,0,0).
        axes=torch.zeros(self.num_envs,2,3,device=self.device);axes[:,:,0]=-1.
        axes=quat_apply(d.body_quat_w[:,self._wheel_ids],axes)
        geo=self.geometry.step(self._up3_position_w(),d.body_pos_w[:,self._base_id],d.root_quat_w,
            com,d.body_pos_w[:,self._wheel_ids],axes,.10,dt,step_id=step)
        loads=self._env.scene.sensors['contact_forces'].data.net_forces_w[:,self._wheel_contact_ids,2].clamp_min(0)
        ratio=loads/(masses.sum(1,keepdim=True)*9.81).clamp_min(1e-6)
        floor=self._env.scene.env_origins[:,2]
        geometric_contact=((geo['contact_points'][:,:,2]-floor[:,None]).abs()<.03).all(-1)
        supported=(ratio>.05).all(-1)&(ratio.sum(-1)>.5)&geometric_contact
        self.intent_gate.copy_(quiet(self.target_command[:,1],.02,.08)*quiet(self.target_command[:,0],.03,.10))
        advance_temporal = self._gate_step != step
        if advance_temporal:
            self.stop_age.copy_(torch.where(self.intent_gate>.99,self.stop_age+dt,0.))
        ramp=((self.stop_age-.5)/1.5).clamp(0,1);ramp=ramp.square()*(3-2*ramp)
        self.fine_gate.copy_(self.intent_gate*ramp*quiet(self.commanded_height_rate,.01,.04)*quiet(self.command[:,2]-self._up3_position_w()[:,2],.02,.06))
        if advance_temporal:
            self.pulse_quiet_age.copy_(torch.where(self.disturbance_active,0.,self.pulse_quiet_age+dt))
        self.balance_gate.copy_(self.fine_gate*supported*geo['valid']*(self.pulse_quiet_age/.3).clamp(0,1))
        sample=torch.stack((geo['tcp_vel'][:,0],geo['tcp_vel'][:,1],geo['world_ang_vel'][:,2]),-1)
        if advance_temporal:
            self.velocity_bias.add_(dt/(2.+dt)*(sample-self.velocity_bias))
        self._gate_step=step
        whole_com=(d.body_com_pos_w*masses[:,:,None]).sum(1)/masses.sum(1,keepdim=True)
        delta=whole_com-d.body_pos_w[:,self._wheel_ids].mean(1)
        heading=quat_apply(d.root_quat_w,torch.tensor([1.,0.,0.],device=self.device).expand(self.num_envs,3));heading[:,2]=0
        heading=heading/heading.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        lateral=torch.stack((-heading[:,1],heading[:,0],torch.zeros_like(heading[:,0])),-1)
        dx=(delta*heading).sum(-1);dy=(delta*lateral).sum(-1)
        self.com_cost.copy_((dy/.08).square()+(1-self.balance_gate)*(dx/.08).square())
        geo.update(heading=heading,nonwheel_weighted=weighted,nonwheel_mass=total,actual_nonwheel_com=com)
        self._geo=geo;self._geometry_step=step
        self._update_drift()
        return geo

    def _update_coordination_state(self):
        if not hasattr(self,'geometry'):
            return super()._update_coordination_state()
        step=self._env.common_step_counter
        if self._coordination_step==step:return
        g=self.ensure_geometry()
        previous_low=self.height_low_frequency_velocity.clone()
        previous_alpha=self.alpha_base.clone()
        super()._update_coordination_state()
        dt=self._env.step_dt;eh=self.command[:,2]-self._up3_position_w()[:,2]
        self.actual_base_vz.copy_(g['base_vel'][:,2]);self.actual_arm_vz.copy_(g['tcp_vel'][:,2]-g['base_vel'][:,2])
        vmax=self.STAGE_VREF_LIMITS[self.active_stage]
        ref=(self.commanded_height_rate+1.5*eh-.25*(g['tcp_vel'][:,2]-self.commanded_height_rate)).clamp(-vmax,vmax)
        self.height_reference_velocity.copy_(ref)
        # Parent computes intermediate legacy/API targets; discard their temporal
        # updates, then derive this child's final targets ONCE from geometry.
        low=previous_low+dt/(.8+dt)*(ref-previous_low)
        self.height_low_frequency_velocity.copy_(torch.minimum(torch.maximum(low,torch.minimum(ref,torch.zeros_like(ref))),torch.maximum(ref,torch.zeros_like(ref))))
        self.height_high_frequency_velocity.copy_(ref-self.height_low_frequency_velocity)
        robot=self._robot
        utilization=robot.data.applied_torque.abs()/robot.data.joint_effort_limits.clamp_min(1e-6)
        bq=1-utilization[:,self._leg_ids].amax(-1).clamp(0,1)
        aq=1-utilization[:,self._arm_ids].amax(-1).clamp(0,1)
        load=1-self.wheel_load_difference.clamp(0,1)
        reach=((self.REACH_HARD_LIMIT-self.horizontal_reach)/.10).clamp(0,1)
        br=self.STAGE_BASE_HEIGHT_RANGES[self.active_stage];ar=self.STAGE_ARM_HEIGHT_RANGES[self.active_stage]
        self.base_directional_margin.copy_(nm.directional_margin(ref,self.base_height,br))
        self.arm_directional_margin.copy_(nm.directional_margin(ref,self.arm_relative_height,ar))
        raw=nm.height_allocation_ratio(ref,self.base_height,self.arm_relative_height,br,ar,bq*self._com_quality()*load,aq*reach)
        self.alpha_base.copy_(torch.where(ref.abs()>.02,previous_alpha+dt/(.30+dt)*(raw-previous_alpha),previous_alpha))
        self.target_base_vz.copy_(self.alpha_base*self.height_low_frequency_velocity)
        self.target_arm_vz.copy_(ref-self.target_base_vz)
        direction=ref.sign()
        self.cancellation.copy_((ref.abs()>.02)&((self.actual_base_vz*direction<0)|(self.actual_arm_vz*direction<0)))
        cost,occupancy=coordination.continuous_vertical_cancellation(self.actual_base_vz,self.actual_arm_vz,eh,self.commanded_height_rate)
        self.continuous_cancellation_cost.copy_(cost);self.continuous_cancellation_occupancy.copy_(occupancy)
        self._coordination_step=step

    def _capture_post_physics_statistics(self):
        """Capture the applied interval before curriculum audit or autoreset."""
        if getattr(self.cfg,'behavior_revision',1)==1:
            self._update_coordination_state()
            return
        step=self._env.common_step_counter
        if self._statistics_step==step:return
        self._accumulate_metrics()
        self._statistics_step=step

    def _update_metrics(self):
        # CommandTerm.compute runs after autoreset. Legacy checkpoints retain
        # that sampling point; r2 samples only from the first reward entry.
        if getattr(self.cfg,'behavior_revision',1)==1:
            self._accumulate_metrics()

    def _accumulate_metrics(self):
        self.tracking_statistics_mask=self.modes.normal_mask()
        super()._update_metrics()
        g=self.ensure_geometry();scale=1./self._env.max_episode_length
        values={'Balance/theta':g['theta'],'Balance/theta_dot':g['theta_dot'],'Balance/h':g['h'],'Balance/vG':g['vG'],
            'Balance/valid':g['valid'].float(),'Static/capture_gate':self.intent_gate,'Static/fine_gate':self.fine_gate,
            'Static/bias_vx':self.velocity_bias[:,0],'Static/bias_yaw':self.velocity_bias[:,2],
            'Drift/api_tcp_z_residual':self._up3_velocity_w()[:,2]-g['tcp_vel'][:,2]}
        for i,label in enumerate(('normal','static','stop','cycle')):values['Mode/'+label]=(self.modes.mode==i).float()
        for k,v in values.items():self.metrics[k].add_(v*scale)

    def adaptation_snapshot(self):
        g=self.ensure_geometry();d=self._robot.data
        cache=self._env._nandu_adaptive_payload
        body=[self._base_id,self._up3_id];mass=parent.actual_body_mass(self._env)
        actual_pair=(d.body_com_pos_w[:,body]*mass[:,body,None]).sum(1)
        cf_com_world=d.body_pos_w[:,body]+quat_apply(d.body_quat_w[:,body],cache['cf_com'])
        cf_sum=g['nonwheel_weighted']-actual_pair+(cf_com_world*cache['cf_mass'][:,:,None]).sum(1)
        cf_total=g['nonwheel_mass']-mass[:,body].sum(1)+cache['cf_mass'].sum(1)
        cf=am.equivalent_balance(cf_sum/cf_total[:,None].clamp_min(1e-6),g['contact_points'],g['heading'])
        valid=torch.isfinite(cache['teacher_params']).all(-1)
        return dict(teacher_params=cache['teacher_params'],counterfactual_teacher_params=cache['counterfactual_teacher_params'],
            balance_target=torch.stack((g['theta'],g['h']),-1),
            counterfactual_balance_target=torch.stack((cf['theta'],cf['h']),-1),valid=valid,
            balance_state=torch.stack((g['theta'],g['theta_dot'],g['vG']),-1),balance_valid=g['valid'],
            raw_intent=self.target_command,height_rate=self.commanded_height_rate[:,None],
            diagnostic_velocity_body=torch.cat((quat_apply_inverse(d.root_quat_w,self._point_velocity[:,1]),
                                                quat_apply_inverse(d.root_quat_w,g['tcp_vel_raw'])),1),
            diagnostic_height=self._up3_position_w()[:,2:3],diagnostic_valid=g['valid'])

    def motion_snapshot(self):
        """Independent truth telemetry only; never passed to actor-side context."""
        g=self.ensure_geometry();d=self._robot.data
        b,a=self._base_id,self._up3_id
        points=torch.stack((d.body_pos_w[:,b],d.body_com_pos_w[:,b],d.body_pos_w[:,a],
                            d.body_com_pos_w[:,a],self._up3_position_w()),1)
        velocities=[]
        for index,body in ((0,b),(1,b),(2,a),(3,a),(4,a)):
            velocities.append(d.body_lin_vel_w[:,body]+torch.linalg.cross(d.body_ang_vel_w[:,body],points[:,index]-d.body_com_pos_w[:,body],dim=-1))
        return dict(point_names=('base_origin','base_com','up3_origin','up3_com','fixed_tcp'),
            position_world=points,velocity_api_world=torch.stack(velocities,1),
            tcp_velocity_difference=g['tcp_vel_raw'],base_velocity_difference=g['base_vel_raw'],
            base_quaternion_world=d.root_quat_w,up3_quaternion_world=d.body_quat_w[:,a],
            point_signed_displacement_world=points-self._point_start,segment_seconds=self._drift_time,
            wheel_relative_travel=self._wheel_net,wheel_absolute_travel=self._wheel_abs,
            wheel_rotation_corrected_proxy=self._wheel_absolute_roll,valid=g['valid'])

    def _update_drift(self):
        if self._drift_step==self._env.common_step_counter:return
        self._drift_step=self._env.common_step_counter
        p=self._up3_position_w();dt=self._env.step_dt
        quiet_ref=(self.commanded_height_rate.abs()<.005)&((self.target_command[:,2]-self.command[:,2]).abs()<.001)
        self._reference_quiet_age.copy_(torch.where(quiet_ref,self._reference_quiet_age+dt,0.))
        ready=(self.target_command[:,:2].abs()<1e-4).all(-1)&(self._reference_quiet_age>=.5)
        start=ready&~self._drift_active
        self._drift_start[start]=p[start];self._drift_path[start]=0;self._drift_time[start]=0
        self._wheel_abs[start]=0;self._wheel_net[start]=0
        self._wheel_absolute_roll[start]=0;self._yaw_net[start]=0;self._yaw_path[start]=0
        d=self._robot.data;b,a=self._base_id,self._up3_id
        points=torch.stack((d.body_pos_w[:,b],d.body_com_pos_w[:,b],d.body_pos_w[:,a],d.body_com_pos_w[:,a],p),1)
        self._point_start[start]=points[start];self._point_path[start]=0
        point_delta=points-self._point_previous
        self._point_velocity.copy_(torch.where(self._geo['valid'][:,None,None],point_delta/dt,0.))
        self._point_path.add_(point_delta[:,:,:2].norm(dim=-1)*ready[:,None]*self._drift_active[:,None])
        self._point_previous.copy_(points)
        q=self._robot.data.joint_pos[:,self._wheel_joint_ids]
        active=ready&self._drift_active
        self._drift_path.add_((p-self._drift_previous)[:,:2].norm(dim=-1)*active)
        self._drift_time.add_(dt*ready)
        delta=torch.atan2(torch.sin(q-self._wheel_previous),torch.cos(q-self._wheel_previous))*.10
        self._wheel_net.add_(delta*active[:,None]);self._wheel_abs.add_(delta.abs()*active[:,None])
        d=self._robot.data;quat=d.body_quat_w[:,self._wheel_ids]
        omega=am.quaternion_angular_velocity(self._wheel_quat_previous,quat,dt)
        axle=quat_apply(quat,torch.tensor([-1.,0.,0.],device=self.device).expand(self.num_envs,2,3))
        # Absolute wheel spin includes parent rotation; still only a rolling proxy.
        spin=(omega*axle).sum(-1)*.10*dt
        self._wheel_absolute_roll.add_(spin*active[:,None])
        self._wheel_quat_previous.copy_(quat)
        qroot=d.root_quat_w
        yaw=torch.atan2(2*(qroot[:,0]*qroot[:,3]+qroot[:,1]*qroot[:,2]),1-2*(qroot[:,2].square()+qroot[:,3].square()))
        dyaw=am.wrap_angle(yaw-self._yaw_previous)
        self._yaw_net.add_(dyaw*active);self._yaw_path.add_(dyaw.abs()*active);self._yaw_previous.copy_(yaw)
        self._drift_previous.copy_(p);self._wheel_previous.copy_(q);self._drift_active.copy_(ready)
        valid=(self._drift_time>=2.)&ready
        t=self._drift_time.clamp_min(dt)
        net=(p-self._drift_start)[:,:2].norm(dim=-1)/t;path=self._drift_path/t;wheel=self._wheel_abs.mean(-1)/t
        scale=1./self._env.max_episode_length
        for k,v in [('tcp_net_speed',net),('tcp_activity',path),('wheel_activity',wheel),
                    ('score_tcp_net',torch.exp(-(net/.002).square())),('score_tcp_activity',torch.exp(-(path/.02).square())),
                    ('score_wheel_activity',torch.exp(-(wheel/.02).square())),('yaw_slope',self._yaw_net.abs()/t),
                    ('yaw_activity',self._yaw_path/t),('score_yaw',torch.exp(-(self._yaw_net/t/.001).square())),
                    ('wheel_net_proxy',self._wheel_net.mean(-1)/t),('wheel_absolute_roll_proxy',self._wheel_absolute_roll.mean(-1)/t)]:
            self.metrics['Drift/'+k].add_(torch.where(valid,v,0.)*scale)
        self.metrics['Drift/valid'].add_(valid.float()*scale)
        per_point=(points-self._point_start)/t[:,None,None]
        for j,point in enumerate(('base_origin','base_com','up3_origin','up3_com','fixed_tcp')):
            for k,axis in enumerate(('x','y','z')):
                self.metrics[f'Drift/{point}/net_abs_{axis}'].add_(torch.where(valid,per_point[:,j,k].abs(),0.)*scale)
            self.metrics[f'Drift/{point}/activity_xy'].add_(torch.where(valid,self._point_path[:,j]/t,0.)*scale)

    def reset(self,env_ids=None):
        ids=v1._ids(env_ids,self.num_envs,self.device)
        # CommandTerm.reset dispatches _resample_command from the parent reset;
        # it must see the NEW episode's mode and a fresh STOP age.
        if hasattr(self,'modes'):
            self.modes.reset(ids,*self.STAGE_HEIGHT_RANGES[self.active_stage])
            self.tracking_statistics_mask=self.modes.normal_mask()
        result=super().reset(ids)
        if hasattr(self,'processor'):
            self.processor.reset(ids,height=self._command[ids,2])
            self.geometry.reset(ids)
            # The parent clears vref but leaves the previous episode's
            # decomposition cached; keep reset rows coherent until next physics.
            for name in ('target_base_vz','target_arm_vz','actual_base_vz','actual_arm_vz',
                         'target_base_vx','target_arm_vx','actual_base_vx','actual_arm_vx',
                         'continuous_cancellation_cost','continuous_cancellation_occupancy','cancellation'):
                getattr(self,name)[ids]=0
            self.stop_age[ids]=0;self.velocity_bias[ids]=0;self.pulse_quiet_age[ids]=0
            self.balance_gate[ids]=0;self.fine_gate[ids]=0;self.intent_gate[ids]=0
            self._drift_active[ids]=False;self._drift_time[ids]=0
            self._reference_quiet_age[ids]=0
            self._geometry_step=None
        return result

    def state_dict(self):
        state=super().state_dict()
        state['adaptive_modes']=self.modes.state_dict()
        state['adaptive_normalization']=self._env._nandu_adaptive_payload['normalization'].detach().cpu()
        state['adaptive_processor']={k:v.detach().cpu().clone() for k,v in vars(self.processor).items() if isinstance(v,torch.Tensor)}
        return state

    def load_state_dict(self,state):
        saved=torch.as_tensor(state['adaptive_normalization'],device=self.device)
        if not torch.allclose(saved,self._env._nandu_adaptive_payload['normalization']):
            raise ValueError('Adaptive payload normalization mismatch; start a fresh experiment')
        super().load_state_dict(state)
        if 'adaptive_modes' in state:self.modes.load_state_dict(state['adaptive_modes'])


def yaw_tracking(env,command_name,std):
    term=env.command_manager.get_term(command_name)
    # This is the first reward in A–F. In r2 the complete inherited metrics
    # chain also refreshes tracking_gate before allocation rewards consume it.
    term._capture_post_physics_statistics()
    return v0.yaw_rate_tracking(env,command_name,std)


def adaptive_static_cost(env,command_name,kind):
    t=env.command_manager.get_term(command_name);g=t.ensure_geometry()
    def h(x):
        a=x.abs();return torch.where(a<=1,.5*x.square(),a-.5)
    tcp=g['tcp_vel'];yaw=g['world_ang_vel'][:,2];valid=g['valid'].float()
    if kind=='capture':cost=h(tcp[:,:2]/.08).sum(-1)+h(yaw/.12);gate=t.intent_gate
    elif kind=='fine':cost=h(tcp[:,:2]/.02).sum(-1)+h(yaw/.03);gate=t.fine_gate
    elif kind=='bias':
        cost=h(t.velocity_bias/.005).sum(-1);gate=t.intent_gate*((t.stop_age-.5)/1.5).clamp(0,1)
    elif kind=='residual':
        cost=h(tcp[:,2]/.05)+.5*h(g['base_vel'][:,2]/.05)+.5*h(g['world_ang_vel'][:,:2]/.15).sum(-1);gate=t.fine_gate
    elif kind=='balance':
        cost=h(g['theta']/.08)+.5*h(g['theta_dot']/.15)+.5*h(g['vG']/.02);gate=t.balance_gate
    else:raise ValueError(kind)
    return torch.nan_to_num(cost,nan=20.,posinf=20.).clamp(0,20)*gate*valid


def static_orientation_hold(env,command_name):
    t=env.command_manager.get_term(command_name)
    b=env.scene.sensors['base_imu'].data.projected_gravity_b
    e=env.scene.sensors['up3_imu'].data.projected_gravity_b
    roll=torch.atan2(b[:,1],-b[:,2]);tilt=torch.acos((-e[:,2]).clamp(-1,1))
    return t.static_gate*((roll/.08).square()+(tilt/.08).square()).clamp(0,20)


def com_to_wheel_support(env,command_name,scale=.08):
    t=env.command_manager.get_term(command_name);t.ensure_geometry()
    return t.com_cost*(.08/scale)**2
