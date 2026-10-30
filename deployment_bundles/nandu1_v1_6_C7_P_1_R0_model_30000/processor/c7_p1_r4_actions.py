"""R0/R1/R2 only: bounded positions, latent likelihood, physical wheel increments.

PPO stores the latent sample. log_prob subtracts the stable Jacobian into fixed
normalized target units; entropy includes that Jacobian. No inverse tanh of a
rounded hardware target is ever used. Wheel outputs are normalized increments.
"""
import math
from functools import lru_cache
import torch
from torch.nn import functional as F
from .actions import POSITION_DEFAULT, POSITION_LOWER, POSITION_UPPER, JOINT_NAMES

WHEEL_SCALE = .6 / 10.4


class BoundedJointMap:
    def __init__(self, device='cpu'):
        lower = list(POSITION_LOWER)
        lower[3] = lower[7] = .20
        self.lower = torch.tensor(lower, device=device)
        self.upper = torch.tensor(POSITION_UPPER, device=device)
        self.zero = torch.tensor(POSITION_DEFAULT, device=device)
        self.mid = (self.lower + self.upper) / 2
        self.half = (self.upper - self.lower) / 2
        fraction = (self.zero - self.mid) / self.half
        self.bias = torch.atanh(fraction)
        self.gain = .25 / (self.half * (1 - fraction.square()))

    def argument(self, latent):
        return self.bias.to(latent) + self.gain.to(latent) * latent

    def to_target(self, latent):
        if latent.shape[-1] != 11 or not torch.isfinite(latent).all():
            raise ValueError('Expected finite nonwheel latent [...,11]')
        return self.mid.to(latent) + self.half.to(latent) * self.argument(latent).tanh()

    def slope(self, latent):
        return self.half.to(latent) * self.gain.to(latent) * (1-self.argument(latent).tanh().square())

    def log_abs_det_jacobian(self, latent):
        x = self.argument(latent)
        return torch.log(self.half.to(latent) * self.gain.to(latent)) + 2*(math.log(2)-x-F.softplus(-2*x))


@lru_cache(maxsize=16)
def _device_joint_map(device):
    # Constants only: never cache latent-dependent outputs or recurrent state.
    # Build outside inference_mode so subsequent PPO backward can use them.
    with torch.inference_mode(False), torch.no_grad():
        return BoundedJointMap(device)


def joint_map_for(device):
    """Process-local, read-only constants shared by R4 consumers on one device."""
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return _device_joint_map(device)


class LatentActionDistribution(torch.distributions.Normal):
    """Samples/mean/std stay latent for analytic KL and reproducible PPO replay."""
    def __init__(self, loc, scale):
        super().__init__(loc, scale, validate_args=False)
        self.mapping = joint_map_for(loc.device)

    def log_prob(self, value):
        raw = super().log_prob(value)
        jac = self.mapping.log_abs_det_jacobian(value[..., :11]) - math.log(.25)
        return torch.cat((raw[..., :11] - jac, raw[..., 11:]), -1)

    def entropy(self):
        # Reparameterized Monte Carlo expectation: gradients penalize tanh tails.
        sample = self.rsample()
        jac = self.mapping.log_abs_det_jacobian(sample[..., :11]) - math.log(.25)
        gaussian = super().entropy()
        return torch.cat((gaussian[..., :11] + jac, gaussian[..., 11:]), -1)


class R4ActionInterface:
    def __init__(self, device='cpu'):
        self.mapping = joint_map_for(device)
        self.scale = torch.tensor([.25]*11 + [1.,1.], device=device)
        self.offset = torch.tensor(list(POSITION_DEFAULT)+[0.,0.], device=device)

    def map(self, latent):
        if latent.shape[-1] != 13 or not torch.isfinite(latent).all():
            raise ValueError('Expected finite r4 action [...,13]')
        position = (self.mapping.to_target(latent[..., :11])-self.mapping.zero.to(latent))/.25
        return torch.cat((position, latent[..., 11:]*WHEEL_SCALE), -1)

    def metadata(self):
        return dict(schema='nandu_c7_p1_r4_bounded_wheel_increment_v1',
            joint_names=list(JOINT_NAMES), mapping='affine_tanh_nonwheel_linear_wheel',
            scale=self.scale.tolist(), offset=self.offset.tolist(),
            distribution_log_probability='latent_normal_minus_normalized_target_log_jacobian',
            feedback='interval_mean_executed_wheel_increment_and_issued_nonwheel',
            position_hard_lower_rad=self.mapping.lower.tolist(),
            position_hard_upper_rad=self.mapping.upper.tolist(),
            position_default_rad=list(POSITION_DEFAULT), zero_slope_rad=.25,
            wheel_increment_scale_rad=WHEEL_SCALE, wheel_kp=10.4, wheel_kd=1.32,
            wheel_effort_limit_Nm=22., wheel_velocity_limit_rad_s=40.,
            wheel_static_friction_Nm=.6, wheel_dynamic_friction_Nm=.3,
            policy_dt_s=.02, physics_dt_s=.005)

    def validate_articulation(self,robot,action_manager=None):
        from .actions import ActionInterface
        close=ActionInterface._require_close
        names=list(robot.joint_names)
        ids=[names.index(n) for n in JOINT_NAMES[:11]]
        wheels=[names.index(n) for n in JOINT_NAMES[11:]]
        d=robot.data
        hard=torch.stack((self.mapping.lower,self.mapping.upper),-1)
        for name in ('joint_pos_limits','default_joint_pos_limits'):
            close(getattr(d,name)[:,ids],hard,name)
        half=.45*(hard[:,1]-hard[:,0]);mid=hard.mean(-1)
        close(d.soft_joint_pos_limits[:,ids],torch.stack((mid-half,mid+half),-1),'soft limits')
        close(d.default_joint_pos[:,ids],POSITION_DEFAULT,'action zero')
        close(d.joint_vel_limits[:,wheels],40.,'wheel velocity limit')
        close(d.joint_effort_limits[:,wheels],22.,'wheel effort limit')
        close(d.joint_friction_coeff[:,wheels],.6,'native static joint friction')
        close(d.joint_dynamic_friction_coeff[:,wheels],.3,'native dynamic joint friction')
        if action_manager is not None:
            if list(action_manager.active_terms)!=['joint_position','wheel_position_delta']:
                raise ValueError('r4 action ordering mismatch')
            p=action_manager.get_term('joint_position')
            if tuple(p._joint_names)!=JOINT_NAMES[:11] or p.cfg.clip is not None:
                raise ValueError('r4 nonwheel action mapping mismatch')
            close(p._scale,.25,'position scale');close(p._offset,POSITION_DEFAULT,'position offset')
            w=action_manager.get_term('wheel_position_delta')
            if tuple(w._joint_names)!=JOINT_NAMES[11:] or w.cfg.clip is not None:
                raise ValueError('r4 wheel action mapping mismatch')
            close(w._scale,1.,'wheel scale');close(w._offset,0.,'wheel offset')


def physical_profile(spec):
    from .c7_p1 import physical_profile as baseline
    p=baseline()
    p['profile_schema']='nandu_c7_p1_r4_'+spec.revision_profile
    p['checkpoint_schema']=spec.schema
    p['joint_limits'].pop('lower_bounds')
    p['joint_limits']['lower_rad']={'knee_L_Joint':.20,'knee_R_Joint':.20}
    p['joint_limits']['knee_work_region_rad']=[.30,1.10]
    p['wheel_control']=R4ActionInterface().metadata()
    p['posture']['static_orientation']['gate']='raw_user_intent_height_gate'
    if spec.revision_profile!='R0':p['posture']['static_orientation']['weight']=0.
    return p


def write_joint_limits(robot):
    names=list(robot.joint_names)
    ids=[names.index(n) for n in JOINT_NAMES[:11]]
    limits=robot.data.joint_pos_limits[:,ids].clone()
    m=BoundedJointMap(limits.device)
    limits[:,:,0]=m.lower;limits[:,:,1]=m.upper
    robot.write_joint_position_limit_to_sim(limits,joint_ids=ids)
    robot.data.joint_pos_limits[:,ids]=limits
    robot.data.default_joint_pos_limits[:,ids]=limits
    mid=limits.mean(-1);half=.45*(limits[...,1]-limits[...,0])
    robot.data.soft_joint_pos_limits[:,ids]=torch.stack((mid-half,mid+half),-1)
