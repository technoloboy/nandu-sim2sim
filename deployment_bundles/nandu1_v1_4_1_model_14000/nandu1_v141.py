"""Independent NumPy v1.4.1 external-command processor; no simulator truth."""
import math
import numpy as np
from nandu1_contract import finite
from nandu1_v14 import axes, body_to_heading, actor_input, wrap, STAGE_LIMITS, STAGE_HEIGHT_RANGES, HEIGHT_RATE_LIMITS

PROCESSOR_CONTRACT = dict(schema='nandu1_v1_4_1_command_processor_v1',heading_kp=1.5,
    yaw_deadzone=1e-5,dt_min=1e-6,dt_max=.1,vx_processing='immediate_stage_clamp',
    height_processing='external_slew',heading_observation='body_gyro_and_projected_gravity',
    step_once_per_policy_period=True,cue_scales=[10.,5.],cue_tau=.08,
    cue_formula='u=tanh((applied-previous)/(dt*scale)); T=(1-alpha)*Tprev+alpha*u; alpha=dt/(tau+dt)',
    height_cue_scale=.50,height_cue='clip(commanded_height_rate/0.50,-1,1)',
    finite_height_step_rate=0.,reset='clear_filter_and_heading_memory_hold_supplied_height',
    state_restore='same_processor_schema_preserves_filter_memory',
    setter='latch_only; resolved_applied_commands_advance_once_per_policy_tick')
COMMAND_ORDER=['base_yaw_rate_world_z','up3_vx_base_heading','up3_height_world_z',
               'yaw_change_cue','vx_change_cue','commanded_height_rate_cue']

class CommandProcessor:
    def __init__(self,stage,height):
        self.set_stage(stage); self.reset(height)

    def set_stage(self,stage):
        if isinstance(stage,bool) or stage not in range(5): raise ValueError('stage must be 0..4')
        self.stage=int(stage)

    def reset(self,height):
        if not np.isfinite(height): raise ValueError('Non-finite initial height')
        self.commands=np.array([0,0,np.clip(height,.4,1.4)],np.float32)
        self.goal=self.commands.copy(); self.trend=np.zeros(3,np.float32)
        self.heading=self.heading_target=np.float32(0)
        self.was_turning=False; self.height_rate=np.float32(0); self.is_step=False
        self.pending_step=False; self.last_tick=None

    def set_external(self,external,stage=None,*,finite_height_step=False):
        if stage is not None:self.set_stage(stage)
        goal=finite(external,3).astype(np.float32)
        yl,vl=STAGE_LIMITS[self.stage];lo,hi=STAGE_HEIGHT_RANGES[self.stage]
        self.goal=np.clip(goal,[-yl,-vl,lo],[yl,vl,hi]).astype(np.float32)
        self.pending_step=bool(finite_height_step)

    def step(self,external,gravity,gyro,dt=.02,*,tick=None):
        if not np.isfinite(dt) or not 1e-6<=dt<=.1:raise ValueError('dt must be finite in [1e-6,.1]')
        if tick is not None and tick==self.last_tick:return np.r_[self.commands,self.trend].astype(np.float32)
        if external is not None:self.set_external(external)
        gyro=finite(gyro,3).astype(np.float32);_,left,up=axes(gravity)
        yl,vl=STAGE_LIMITS[self.stage];lo,hi=STAGE_HEIGHT_RANGES[self.stage]
        goal=np.clip(self.goal,[-yl,-vl,lo],[yl,vl,hi]).astype(np.float32)
        previous=self.commands.copy()
        projection=np.float32(math.sqrt(max(1-float(up[0]**2),1e-4)))
        heading_rate=np.dot(np.cross(gyro,np.array([1,0,0],np.float32)),left)/projection
        if projection<=.01:heading_rate=np.dot(gyro,up)
        self.heading=wrap(self.heading+heading_rate*dt)
        joystick=goal[0];turning=abs(joystick)>1e-5
        if self.was_turning and not turning:self.heading_target=self.heading
        self.heading_target=wrap(self.heading_target+(joystick*dt if turning else 0))
        coupling=np.dot(gyro,up)-heading_rate
        raw=float(joystick+1.5*wrap(self.heading_target-self.heading)+coupling)
        yaw=np.clip(raw,-yl,yl)
        if raw!=yaw:self.heading_target=wrap(self.heading+(yaw-coupling-joystick)/1.5)
        self.was_turning=bool(turning)
        self.is_step=self.pending_step or previous[2]<lo or previous[2]>hi
        delta=np.clip(goal[2]-previous[2],-HEIGHT_RATE_LIMITS[self.stage]*dt,HEIGHT_RATE_LIMITS[self.stage]*dt)
        height=goal[2] if self.is_step else previous[2]+delta
        self.commands[:]=[yaw,goal[1],np.clip(height,lo,hi)]
        self.height_rate=np.float32(0 if self.is_step else (self.commands[2]-previous[2])/dt)
        gain=dt/(.08+dt)
        cue=np.tanh((self.commands[:2]-previous[:2])/np.float32(dt)/np.array([10,5],np.float32))
        self.trend[:2]=(1-gain)*self.trend[:2]+gain*cue
        self.trend[2]=np.clip(self.height_rate/.5,-1,1)
        self.pending_step=False;self.last_tick=tick
        return np.r_[self.commands,self.trend].astype(np.float32)

    def state_dict(self):
        return dict(schema=PROCESSOR_CONTRACT['schema'],stage=self.stage,commands=self.commands.copy(),
            goal=self.goal.copy(),trend=self.trend.copy(),heading=float(self.heading),
            heading_target=float(self.heading_target),was_turning=self.was_turning,
            height_rate=float(self.height_rate),is_step=self.is_step,pending_step=self.pending_step,last_tick=self.last_tick)

    def load_state_dict(self,state):
        if state.get('schema')!=PROCESSOR_CONTRACT['schema']:raise ValueError('processor schema mismatch')
        vectors={name:finite(state[name],3).astype(np.float32) for name in ('commands','goal','trend')}
        scalars={name:float(state[name]) for name in ('heading','heading_target','height_rate')}
        if not all(math.isfinite(x) for x in scalars.values()):raise ValueError('Non-finite processor state')
        self.set_stage(state['stage'])
        for name,value in {**vectors,**scalars}.items():setattr(self,name,value)
        for name in ('was_turning','is_step','pending_step','last_tick'):setattr(self,name,state[name])
