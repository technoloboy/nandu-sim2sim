"""Deployable, per-task-channel anti-windup; no contact/pose/effort truth inputs."""
import math
import torch
from .state import ControllerState, physical_errors, configuration
from .c7_p1_r4_actions import joint_map_for, WHEEL_SCALE


def channel_permission(effect, saturation_direction):
    outward = effect * saturation_direction.unsqueeze(-1) > 0
    weight = effect.abs()
    total = weight.sum(-2)
    blocked = (weight*outward).sum(-2)
    return torch.where(total > 1e-8, 1-blocked/total.clamp_min(1e-8), 1.).clamp(0,1)


def integral_effect(mapping, integral, delta, latent, wheel_kp=10.4):
    """Normalize joint position and wheel effort contributions before aggregation."""
    derivative = .20*(1-((mapping*integral.unsqueeze(-2)).sum(-1)).tanh().square())
    effect = derivative.unsqueeze(-1)*mapping*delta.unsqueeze(-2)
    slope = joint_map_for(latent.device).slope(latent[...,:11])
    # A vanishing tanh derivative must not make an outward contributor invisible.
    slope = torch.where(slope < .0025, torch.full_like(slope,.25), slope)/.25
    position = effect[...,:11,:]*slope.unsqueeze(-1)
    wheel = effect[...,11:,:]*(WHEEL_SCALE*wheel_kp/22.)
    return torch.cat((position,wheel),-2)


def nonwheel_saturation(measured_position, latent):
    m=joint_map_for(latent.device)
    high=measured_position >= m.upper.to(latent)-.01
    low=measured_position <= m.lower.to(latent)+.01
    deep=m.slope(latent[...,:11]) < .0025
    direction=torch.where(high,1.,torch.where(low,-1.,0.))
    return torch.where(deep,m.argument(latent[...,:11]).sign(),direction)


def packet_records(policy,packet,wheel_records):
    """Joint feedback is the same noisy measurement as Policy, never reward truth.

    Nonwheel hard/map saturation uses measured q. Wheel budget directions already
    carry the exact delayed packet; they must not be replaced by the newest map.
    """
    records=list(wheel_records or [])
    if packet is not None:
        m=joint_map_for(policy.device)
        direction=torch.zeros(len(policy),13,device=policy.device)
        direction[:,:11]=nonwheel_saturation(policy[:,12:23]+m.zero,packet['latent'])
        mask=torch.zeros_like(direction);mask[:,:11]=1.
        # Four equal-weight nonwheel records match the four wheel physics samples.
        for _ in range(4):records.append(dict(**packet,direction=direction,joint_mask=mask))
    return records


def motor_packet_record(packet,joint_columns,measured_q,requested_effort,emitted_effort):
    """Known motor-controller budget and encoder q, not constraint/contact forces."""
    n=len(measured_q);latent=packet['latent'];m=joint_map_for(latent.device)
    ids=torch.as_tensor(joint_columns,device=latent.device,dtype=torch.long)
    high=measured_q>=m.upper[ids]-.01;low=measured_q<=m.lower[ids]+.01
    deep=m.slope(latent[:,:11])[:,ids]<.0025
    local=torch.where(high,1.,torch.where(low,-1.,0.))
    local=torch.where(deep,m.argument(latent[:,:11])[:,ids].sign(),local)
    residual=requested_effort-emitted_effort
    local=torch.where(residual.abs()>1e-5,residual.sign(),local)
    direction=torch.zeros(n,13,device=latent.device);direction[:,ids]=local
    mask=torch.zeros_like(direction);mask[:,ids]=1
    return dict(**packet,direction=direction,joint_mask=mask)


def packet_permission(records, delta):
    """Batch the existing per-packet rule without mixing delayed I contexts.

    The leading axis is physical packet, not environment. Keep every packet's
    own mapping/integral/latent/direction and joint mask; only reduce afterwards.
    This is called under no_grad by the controller, not retained in PPO graphs.
    """
    if not records:
        return torch.ones_like(delta)
    mapping = torch.stack([r['mapping'] for r in records])
    integral = torch.stack([r['integral'] for r in records])
    latent = torch.stack([r['latent'] for r in records])
    direction = torch.stack([r['direction'] for r in records])
    masks = torch.stack([r['joint_mask'] if 'joint_mask' in r else
                         torch.ones_like(r['direction']) for r in records])
    effect = integral_effect(mapping, integral, delta.unsqueeze(0), latent)
    effect = effect * masks.unsqueeze(-1)
    weight = effect.abs().sum(-2)
    blocked = (weight * (1 - channel_permission(effect, direction))).sum(0)
    total = weight.sum(0)
    return torch.where(total > 1e-8, 1 - blocked / total.clamp_min(1e-8), 1.)


class R4ControllerState(ControllerState):
    def __init__(self,num_envs,device='cpu',use_integral=True):
        super().__init__(num_envs,device,use_integral)
        self.permission=torch.ones(num_envs,3,device=device)

    @torch.no_grad()
    def reset(self,ids,policy,estimate):
        super().reset(ids,policy,estimate)
        self.permission[ids]=1

    @torch.no_grad()
    def advance(self,policy,estimate,command,rate,saturation,dt,step_id,valid=None,intent=None):
        if step_id == self.last_step:return
        if self.last_step is not None and step_id < self.last_step:
            raise ValueError('Controller step counter went backwards')
        finite=torch.isfinite(policy).all(-1)&torch.isfinite(estimate).all(-1)&torch.isfinite(command).all(-1)&torch.isfinite(rate).all(-1)
        upright=-policy[:,5]/policy[:,3:6].norm(dim=-1).clamp_min(1e-8)>math.cos(math.pi/4)
        ok=finite&upright
        if valid is not None:ok &= valid
        error=physical_errors(policy,estimate,command)
        scaled=torch.nan_to_num(error/error.new_tensor([.10,.05,.05])).clamp(-3,3)
        self.error_lp += torch.where(ok[:,None],dt/(.25+dt)*(scaled-self.error_lp),0.)
        gate=(1-rate.abs()/rate.new_tensor([.5,.2,.02])).clamp(0,1)
        delta=dt*gate*self.error_lp
        self.permission.fill_(1)
        if saturation is not None:
            # A list permits each delayed wheel packet to carry its own B/z/latent.
            records=saturation if isinstance(saturation,list) else [saturation]
            self.permission.copy_(packet_permission(records, delta))
        # Opposing increments unwind a saturated accumulator even when the learned
        # map couples this channel into a currently blocked motor direction.
        self.permission.copy_(torch.where(self.integral*delta<0,1.,self.permission))
        if self.use_integral:
            self.integral.add_(torch.where(ok[:,None],delta*self.permission,0.)).clamp_(-2,2)
        else:self.integral.zero_()
        measured=torch.nan_to_num(configuration(policy,estimate))
        self.slow += torch.where(finite[:,None],dt/(.30+dt)*(measured-self.slow),0.)
        raw=command if intent is None else intent
        stopped=(raw[:,0].abs()<.03)&(raw[:,1].abs()<.02)&(rate[:,2].abs()<.01)
        self.stop_clock.copy_(torch.where(stopped[:,None],self.stop_clock+dt,0.))
        self.rate.copy_(rate);self.last_step=step_id
