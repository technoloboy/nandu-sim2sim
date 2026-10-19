"""Sensor-only v1.4 command/Actor adapter; external-command subset of training.

No simulator state is used. Commands contain applied yaw/vx/height followed by
three normalized per-period trends. Sampling/curriculum/rewards stay in training.
"""
import math
import numpy as np
from nandu1_contract import finite

STAGE_LIMITS=((.20,.08),(.50,.20),(.85,.40),(1.20,.70),(1.57,1.))
STAGE_HEIGHT_RANGES=((.78,1.04),(.68,1.12),(.58,1.22),(.48,1.32),(.40,1.40))
HEIGHT_RATE_LIMITS=(.05,.10,.20,.35,.50)
PROCESSOR_CONTRACT={'schema':'nandu1_v1_4_command_processor_v1','heading_kp':1.5,
    'yaw_deadzone':1e-5,'dt_min':1e-6,'dt_max':.1,'vx_processing':'immediate_stage_clamp',
    'height_processing':'external_slew','trend':'applied_difference_over_dt_stage_limits',
    'heading_observation':'body_gyro_and_projected_gravity','step_once_per_policy_period':True}

def axes(gravity):
    gravity=finite(gravity,3).astype(np.float32);norm=np.linalg.norm(gravity)
    up=-gravity/max(norm,1e-6) if norm>1e-6 else np.array([0,0,1],np.float32)
    forward=np.array([1,0,0],np.float32)-up[0]*up
    if np.linalg.norm(forward)<=1e-6:forward=np.array([0,1,0],np.float32)-up[1]*up
    forward/=max(np.linalg.norm(forward),1e-6)
    return forward,np.cross(up,forward),up

def body_to_heading(velocity,gravity):
    velocity=finite(velocity,3)
    return np.array([np.dot(velocity,axis) for axis in axes(gravity)],np.float32)

def actor_input(estimate,observation,commands):
    estimate=finite(estimate,7);observation=finite(observation,49);commands=finite(commands,6)
    features=np.r_[body_to_heading(estimate[:3],observation[3:6]),
                   body_to_heading(estimate[3:6],observation[3:6]),estimate[6]]
    return np.r_[features,observation,commands].astype(np.float32)

def wrap(angle):return np.float32(math.atan2(math.sin(angle),math.cos(angle)))

class CommandProcessor:
    def __init__(self,stage,height):
        if isinstance(stage,bool) or stage not in range(5):raise ValueError('stage must be 0..4')
        self.stage=int(stage);self.reset(height)
    def reset(self,height):
        if not np.isfinite(height):raise ValueError('Non-finite initial height')
        self.commands=np.array([0,0,np.clip(height,.4,1.4)],np.float32)
        self.trend=np.zeros(3,np.float32);self.heading=self.heading_target=np.float32(0)
        self.was_turning=False
    def step(self,external,gravity,gyro,dt=.02):
        if not np.isfinite(dt) or not 1e-6<=dt<=.1:raise ValueError('dt must be finite in [1e-6,.1]')
        goal=finite(external,3).astype(np.float32);gyro=finite(gyro,3).astype(np.float32)
        yaw_limit,vx_limit=STAGE_LIMITS[self.stage];lo,hi=STAGE_HEIGHT_RANGES[self.stage]
        goal=np.clip(goal,[-yaw_limit,-vx_limit,lo],[yaw_limit,vx_limit,hi]).astype(np.float32)
        previous=self.commands.copy();_,left,up=axes(gravity)
        projection=np.float32(math.sqrt(max(1-float(up[0]**2),1e-4)))
        heading_rate=np.dot(np.cross(gyro,np.array([1,0,0],np.float32)),left)/projection
        if projection<=.01:heading_rate=np.dot(gyro,up)
        self.heading=wrap(self.heading+heading_rate*dt)
        joystick=goal[0];turning=abs(joystick)>1e-5
        if self.was_turning and not turning:self.heading_target=self.heading
        self.heading_target=wrap(self.heading_target+(joystick*dt if turning else 0))
        error=wrap(self.heading_target-self.heading)
        coupling=np.dot(gyro,up)-heading_rate
        raw=float(joystick+1.5*error+coupling)
        yaw=np.clip(raw,-yaw_limit,yaw_limit)
        if raw!=yaw:self.heading_target=wrap(self.heading+(yaw-coupling-joystick)/1.5)
        self.was_turning=bool(turning)
        rate=HEIGHT_RATE_LIMITS[self.stage]
        height=goal[2] if previous[2]<lo or previous[2]>hi else previous[2]+np.clip(goal[2]-previous[2],-rate*dt,rate*dt)
        self.commands[:]=[yaw,goal[1],np.clip(height,lo,hi)]
        self.trend[:]=np.clip((self.commands-previous)/np.float32(dt)/np.array([yaw_limit,vx_limit,rate],np.float32),-1,1)
        return np.r_[self.commands,self.trend].astype(np.float32)
