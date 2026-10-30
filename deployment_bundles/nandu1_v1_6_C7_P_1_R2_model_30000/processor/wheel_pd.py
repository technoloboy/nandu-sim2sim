"""Vectorized policy-rate references and physics-rate position-increment PD."""
import torch

KP, KD, TORQUE_LIMIT, POSITION_SCALE = 10.4, 1.32, 22., 1.
INITIAL_ACTION_SCALE = .6 / KP


def wheel_pd_budget(q_ref, q, dq, a_req, kp=KP, kd=KD, scale=POSITION_SCALE,
                    limit=TORQUE_LIMIT, *, validate=True):
    """Return executed increment, target, requested torque, emitted motor torque.

    q/ref use continuous joint radians. Validation is optional only for the
    hot runtime, whose owner records invalid inputs and terminates the episode.
    """
    if not isinstance(kp,torch.Tensor): kp = torch.as_tensor(kp, device=q.device, dtype=q.dtype)
    if not isinstance(kd,torch.Tensor): kd = torch.as_tensor(kd, device=q.device, dtype=q.dtype)
    if validate:
        if not all(torch.isfinite(v).all() for v in (q_ref,q,dq,a_req,kp,kd)):
            raise ValueError('Wheel PD requires finite measurements and requests')
        if not (kp > 0).all() or not (kd >= 0).all() or scale <= 0 or limit <= 0:
            raise ValueError('Wheel PD gains/scale/budget must be positive')
    requested = kp*(q_ref + scale*a_req-q) - kd*dq
    low = ((-limit+kd*dq)/kp + q-q_ref)/scale
    high = ((limit+kd*dq)/kp + q-q_ref)/scale
    executed = torch.maximum(torch.minimum(a_req,high),low)
    target = q_ref + scale*executed
    emitted = (kp*(target-q)-kd*dq).clamp(-limit,limit)
    return executed,target,requested,emitted


class WheelPDState:
    """Frozen q_ref + action packets, delayed together; no per-env Python loop."""
    def __init__(self,num_envs,*,max_delay=4,device='cpu',dtype=torch.float32):
        self.num_envs = num_envs
        self.max_delay = max_delay
        self.rows = torch.arange(num_envs,device=device)
        self.reference = torch.zeros(num_envs,2,device=device,dtype=dtype)
        self.request = torch.zeros_like(self.reference)
        self.executed_increment = torch.zeros_like(self.reference)
        self.target = torch.zeros_like(self.reference)
        self.requested_torque = torch.zeros_like(self.reference)
        self.motor_torque = torch.zeros_like(self.reference)
        self.sequence = torch.zeros(num_envs,device=device,dtype=torch.long)
        self.delay_steps = torch.zeros_like(self.sequence)
        self.reference_queue = torch.zeros(max_delay+1,num_envs,2,device=device,dtype=dtype)
        self.request_queue = torch.zeros_like(self.reference_queue)
        self.sequence_queue = torch.zeros(max_delay+1,num_envs,device=device,dtype=torch.long)
        self.cursor = 0
        self.invalid = torch.zeros(num_envs,device=device,dtype=torch.bool)
        self.delayed_reference = torch.zeros_like(self.reference)
        self.packet_age = torch.zeros_like(self.sequence)
        self.peak_torque = torch.zeros_like(self.reference)
        self.torque_square_sum = torch.zeros_like(self.reference)
        self.clipped_steps = torch.zeros_like(self.reference)
        self.velocity_guard_steps = torch.zeros_like(self.reference)
        self.substeps = 0

    def reset(self,env_ids,q,*,delay_steps):
        ids = torch.as_tensor(env_ids,device=self.reference.device,dtype=torch.long)
        self.reference[ids] = q
        self.request[ids] = self.executed_increment[ids] = 0
        self.target[ids] = self.delayed_reference[ids] = q
        self.reference_queue[:,ids] = q
        self.request_queue[:,ids] = 0
        self.sequence[ids] = self.sequence_queue[:,ids] = 0
        self.delay_steps[ids] = delay_steps
        self.invalid[ids] = False
        self.packet_age[ids] = 0
        for value in (self.requested_torque,self.motor_torque,self.peak_torque,
                      self.torque_square_sum,self.clipped_steps,self.velocity_guard_steps):
            value[ids] = 0

    def capture(self,q,actions):
        self.reference.copy_(q)
        self.request.copy_(actions.clamp(-100.,100.))
        self.sequence.add_(1)
        self.invalid |= ~torch.isfinite(q).all(-1) | ~torch.isfinite(actions).all(-1)
        for value in (self.peak_torque,self.torque_square_sum,self.clipped_steps,self.velocity_guard_steps):
            value.zero_()
        self.substeps = 0

    def step(self,q,dq,kp=KP,kd=KD):
        self.reference_queue[self.cursor] = self.reference
        self.request_queue[self.cursor] = self.request
        self.sequence_queue[self.cursor] = self.sequence
        read = (self.cursor-self.delay_steps) % (self.max_delay+1)
        ref = self.reference_queue[read,self.rows]
        request = self.request_queue[read,self.rows]
        self.packet_age.copy_(self.sequence-self.sequence_queue[read,self.rows])
        self.delayed_reference.copy_(ref)
        self.cursor = (self.cursor+1) % (self.max_delay+1)
        kp = torch.as_tensor(kp,device=q.device,dtype=q.dtype)
        kd = torch.as_tensor(kd,device=q.device,dtype=q.dtype)
        valid = (torch.isfinite(q)&torch.isfinite(dq)&torch.isfinite(ref)&
                 torch.isfinite(request)&torch.isfinite(kp)&torch.isfinite(kd)&(kp>0)&(kd>=0)).all(-1)
        self.invalid |= ~valid
        executed,target,requested,emitted = wheel_pd_budget(ref,q,dq,request,kp,kd,validate=False)
        # Quarantine invalid rows and expose them to the task's non-finite
        # termination. Never send NaN efforts to PhysX or conceal the failure.
        keep = (~self.invalid)[:,None]
        self.executed_increment.copy_(torch.where(keep,executed,0.))
        self.target.copy_(torch.where(keep,target,0.))
        self.requested_torque.copy_(torch.where(keep,requested,0.))
        self.motor_torque.copy_(torch.where(keep,emitted,0.))
        self.peak_torque.copy_(torch.maximum(self.peak_torque,self.motor_torque.abs()))
        self.torque_square_sum.add_(self.motor_torque.square())
        self.clipped_steps.add_((self.requested_torque.abs()>TORQUE_LIMIT).to(q.dtype))
        self.velocity_guard_steps.add_((dq.abs()>=40.).to(q.dtype))
        self.substeps += 1
        return self.executed_increment,self.target,self.requested_torque,self.motor_torque


class WheelActionCosts:
    """Policy-rate requested-action differences, with episode-local histories."""
    def __init__(self, num_envs, *, device='cpu'):
        self.previous = torch.zeros(num_envs,13,device=device)
        self.previous_previous = torch.zeros_like(self.previous)
        self.count = torch.zeros(num_envs,device=device,dtype=torch.long)
        self.rate = torch.zeros(num_envs,device=device)
        self.smooth = torch.zeros_like(self.rate)
        self.rate_groups = torch.zeros(num_envs,3,device=device)
        self.smooth_groups = torch.zeros_like(self.rate_groups)

    @staticmethod
    def grouped(delta):
        # E action order: leg8, arm3, wheel2. Original eleven terms stay squared.
        z = delta[:,-2:] / INITIAL_ACTION_SCALE
        huber_twice = torch.where(z.abs()<=1, z.square(), 2*z.abs()-1).clamp_max(20.)
        return torch.stack((delta[:,:8].square().sum(-1),
                            delta[:,8:11].square().sum(-1),huber_twice.sum(-1)),dim=-1)

    def capture(self, action):
        self.rate_groups.copy_(self.grouped(action-self.previous))
        smooth = self.grouped(action-2*self.previous+self.previous_previous)
        self.smooth_groups.copy_(torch.where((self.count>=2)[:,None],smooth,0.))
        self.rate.copy_(self.rate_groups.sum(-1))
        self.smooth.copy_(self.smooth_groups.sum(-1))
        self.previous_previous.copy_(self.previous)
        self.previous.copy_(action)
        self.count.add_(1)

    def reset(self, env_ids):
        for value in (self.previous,self.previous_previous,self.count,self.rate,self.smooth,
                      self.rate_groups,self.smooth_groups):
            value[env_ids] = 0
