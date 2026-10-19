"""Deployable NANDU1 v1.2 contract. No simulator truth enters observations."""
from collections import deque
import numpy as np

TASK = 'Isaac-NANDU1-Blind-Flat-v1-2'
PLAY = 'Isaac-NANDU1-Blind-Flat-Play-v1-2'
SCHEMA = 'nandu1_height_v1_2_robust_v2'
BUNDLE_SCHEMA = 'limx.nandu1-deployment-bundle/v1'
V14_TASK = 'Isaac-NANDU1-Blind-Flat-v1-4'
V14_PLAY = 'Isaac-NANDU1-Blind-Flat-Play-v1-4'
V14_SCHEMA = 'nandu1_height_v1_4_decoupled_v1'
V141_TASK = 'Isaac-NANDU1-Blind-Flat-v1-4-1'
V141_PLAY = 'Isaac-NANDU1-Blind-Flat-Play-v1-4-1'
V141_SCHEMA = 'nandu1_height_v1_4_1_smooth_transition_v1'
V131_TASK = 'Isaac-NANDU1-Blind-Flat-v1-3-1'
V131_PLAY = 'Isaac-NANDU1-Blind-Flat-Play-v1-3-1'
V131_SCHEMA = 'nandu1_height_v1_3_1_tcp_payload_v1'
V11_TASK = 'Isaac-NANDU1-Blind-Flat-v1-1'
V13_TASK = 'Isaac-NANDU1-Blind-Flat-v1-3'
V121_TASK = 'Isaac-NANDU1-Blind-Flat-v1-2-1'
STAGE_LIMITS = ((.2,.08),(.5,.2),(.85,.4),(1.2,.7),(1.57,1.))
STAGE_HEIGHT_RANGES = ((.78,1.04),(.68,1.12),(.58,1.22),(.48,1.32),(.4,1.4))

_PROFILES = [
    (V11_TASK,'nandu1_height_v1_1_robust_v2','single','body_com',59),
    (TASK,SCHEMA,'dual','body_com',59),
    (V13_TASK,'nandu1_height_v1_3_robust_v2','dual','body_com',59),
    (V121_TASK,'nandu1_height_v1_2_1_adaptive_hold_v1','dual','body_com',59),
    (V131_TASK,V131_SCHEMA,'dual','fixed_tcp',59),
    (V14_TASK,V14_SCHEMA,'dual','body_link_origin',62),
    (V141_TASK,V141_SCHEMA,'dual','body_link_origin',62),
]
TASKS = tuple(row[0] for row in _PROFILES)
PLAY_ALIASES = tuple(task.replace('Flat-v','Flat-Play-v') for task in TASKS)

def task_profile(task):
    for canonical,schema,encoder,reference,width in _PROFILES:
        play=canonical.replace('Flat-v','Flat-Play-v')
        if task in (canonical,play):
            return dict(task_id=canonical,play_alias=play,checkpoint_schema=schema,
                        runtime_profile='nandu1_height_'+canonical.split('Flat-')[1].replace('-','_'),
                        encoder_architecture=encoder,command=width-56,policy_input=width,
                        velocity_reference=reference,actor_velocity_frame='heading' if width==62 else 'body')
    raise ValueError('Unsupported NANDU1 task or Play alias: '+str(task))
JOINTS = [f'{name}_{side}_Joint' for side in ('L', 'R')
          for name in ('proximal_pitch', 'proximal_roll', 'proximal_yaw', 'knee', 'wheel')] + ['arm1', 'arm2', 'arm3']
POSITION_IDS = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12]
ACTION_TO_HARDWARE = POSITION_IDS + [4, 9]
DEFAULT_Q = np.array([-0.475, 0, 0, 0.95, 0, 0.475, 0, 0, 0.95, 0, -0.15, -1.78, 1.93])
KP = np.array([159.67,159.67,53.22,159.67,0]*2 + [480,240,160])
KD = np.array([10.16,10.16,3.39,10.16,0.6]*2 + [50,25,17])
EFFORT = np.array([140,140,40,140,20]*2 + [181,136,136])
ARMATURE = np.array([.161777558,.161777558,.053923687,.161777558,0]*2 + [.7,.45,.45])
VELOCITY = np.array([12.57,12.57,14.66,12.57,40]*2 + [4.14]*3)

def finite(values, size):
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'Expected {size} finite values')
    return a

def rotation(q):
    w,x,y,z = finite(q,4)
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        raise ValueError('Invalid IMU quaternion')
    w,x,y,z = np.array([w,x,y,z])/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def observe(base_q, base_gyro, up_q, up_gyro, q, dq, previous_action):
    gravity = np.array([0.,0.,-1.])
    return np.concatenate([finite(base_gyro,3)*.25, rotation(base_q).T@gravity,
        finite(up_gyro,3)*.25, rotation(up_q).T@gravity,
        (finite(q,13)-DEFAULT_Q)[POSITION_IDS], finite(dq,13)*.05,
        finite(previous_action,13)]).astype(np.float32)

def decode(action):
    a = finite(action,13)
    q = DEFAULT_Q.copy()
    q[POSITION_IDS] += .25*a[:11]
    dq = np.zeros(13)
    dq[[4,9]] = a[11:]
    return q,dq

def clip_position_targets(target, lower, upper):
    """Post-decode projection; inclusive physical limits, no wheel/action clip."""
    q, lo, hi = finite(target,13).copy(), finite(lower,13), finite(upper,13)
    if np.any(lo >= hi):
        raise ValueError('Invalid NANDU1 physical position limits')
    q[POSITION_IDS] = np.clip(q[POSITION_IDS], lo[POSITION_IDS], hi[POSITION_IDS])
    return q

class History:
    def __init__(self):
        self.frames = deque(maxlen=5)
    def reset(self):
        self.frames.clear()
    def append(self, obs):
        frame = finite(obs,49).astype(np.float32)
        if not self.frames:
            self.frames.extend(frame.copy() for _ in range(4))
        self.frames.append(frame.copy())
        return np.concatenate(list(self.frames))
