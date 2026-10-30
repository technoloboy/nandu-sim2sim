"""C7 r4 wheel packets; delay applies to reference, request and I context together."""
import torch
try:  # Self-contained exported processor bundle.
    from .wheel_pd import WheelPDState
except ImportError:
    from ..nandu_six_change_e.wheel_pd import WheelPDState


class WheelPacketState(WheelPDState):
    def __init__(self,num_envs,*,max_delay=4,device='cpu',dtype=torch.float32):
        super().__init__(num_envs,max_delay=max_delay,device=device,dtype=dtype)
        self.context={k:torch.zeros((num_envs,*shape),device=device,dtype=dtype)
                      for k,shape in dict(mapping=(13,3),integral=(3,),latent=(13,)).items()}
        self.context_queue={k:torch.zeros((max_delay+1,*v.shape),device=device,dtype=dtype)
                            for k,v in self.context.items()}
        self.sum_executed=torch.zeros_like(self.reference)
        self.interval_count=torch.zeros_like(self.sequence)
        self.residual_sum=torch.zeros(num_envs,device=device,dtype=dtype)
        self.budget_cost=torch.zeros_like(self.residual_sum)
        self.executed_packet_id=torch.zeros_like(self.sequence)
        self.saturation_records=[]
        self.pending_context=None

    def reset(self,env_ids,q,*,delay_steps):
        super().reset(env_ids,q,delay_steps=delay_steps)
        for value in self.context.values():value[env_ids]=0
        for value in self.context_queue.values():value[:,env_ids]=0
        for value in (self.sum_executed,self.interval_count,self.residual_sum,self.budget_cost,self.executed_packet_id):
            value[env_ids]=0
        for record in self.saturation_records:
            for value in record.values():value[env_ids]=0
        if self.pending_context is not None:
            for value in self.pending_context.values():value[env_ids]=0

    def capture(self,q,actions,context=None):
        super().capture(q,actions)
        # Do not silently flatten all raw requests beyond 100 rad.
        self.request.copy_(torch.where(torch.isfinite(actions),actions,0.))
        self.reference.copy_(torch.where(torch.isfinite(q),q,0.))
        context=self.pending_context if context is None else context
        self.pending_context=None
        for key,value in self.context.items():
            value.copy_(context[key].detach() if context is not None else torch.zeros_like(value))
        self.sum_executed.zero_();self.interval_count.zero_();self.residual_sum.zero_();self.budget_cost.zero_()
        self.saturation_records=[]

    def step(self,q,dq,kp=10.4,kd=1.32):
        cursor=self.cursor
        read=(cursor-self.delay_steps)%(self.max_delay+1)
        for key,value in self.context.items():self.context_queue[key][cursor]=value
        record={k:v[read,self.rows].clone() for k,v in self.context_queue.items()}
        result=super().step(q,dq,kp,kd)
        self.executed_packet_id.copy_(self.sequence_queue[read,self.rows])
        self.invalid |= ~torch.isfinite(self.requested_torque).all(-1)
        # The joint's 40rad/s limit remains in PhysX. Suppress only further
        # acceleration past it; braking is never removed.
        guard=(dq.abs()>=40.)&(self.motor_torque*dq>0)
        self.motor_torque.copy_(torch.where(guard|self.invalid[:,None],0.,self.motor_torque))
        kp=torch.as_tensor(kp,device=q.device,dtype=q.dtype)
        kd=torch.as_tensor(kd,device=q.device,dtype=q.dtype)
        effective=(self.motor_torque+kd*dq)/kp+q-self.delayed_reference
        self.executed_increment.copy_(torch.where(self.invalid[:,None],0.,effective))
        self.target.copy_(self.delayed_reference+self.executed_increment)
        residual=torch.where(self.invalid[:,None],0.,self.requested_torque-self.motor_torque)
        z=(residual/22.).abs()
        huber=torch.where(z<=1,.5*z.square(),z-.5).mean(-1)
        self.residual_sum.add_(huber);self.interval_count.add_(1)
        self.budget_cost.copy_(self.residual_sum/self.interval_count.clamp_min(1))
        self.sum_executed.add_(self.executed_increment)
        direction=torch.zeros(q.shape[0],13,device=q.device,dtype=q.dtype)
        direction[:,11:]=torch.where(residual.abs()>1e-6,residual.sign(),0.)
        mask=torch.zeros_like(direction);mask[:,11:]=1.
        record.update(direction=direction,joint_mask=mask)
        self.saturation_records.append(record)
        return self.executed_increment,self.target,self.requested_torque,self.motor_torque

    def mean_executed(self):
        return self.sum_executed/self.interval_count.clamp_min(1)[:,None]
