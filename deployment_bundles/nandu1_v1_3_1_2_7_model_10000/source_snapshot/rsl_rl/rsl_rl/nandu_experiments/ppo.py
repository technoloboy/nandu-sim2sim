"""Opt-in split input-Jacobian regularization, with the baseline PPO objective."""
import time
import torch
from torch import nn
from ..algorithm.ppo import PPO, encoder_supervision_loss, encoder_supervision_metrics
from ..modules.nandu_actor_input import build_actor_input


def split_gradient_penalty(actor, raw, commands, probe=None):
    x=raw.detach().requires_grad_(True)
    mean=actor(x)
    if probe is None: probe=torch.empty_like(mean).bernoulli_(.5).mul_(2).sub_(1)
    if probe.shape != mean.shape: raise ValueError('GP probe shape must match actions')
    gradient=torch.autograd.grad((mean*probe.detach()/mean.shape[-1]**.5).sum(),x,create_graph=True)[0]
    scales=x.new_tensor([.6,.4,.3,.6,.4,.3,.05]+[1.]*49+[.25,.12,.10]+([1.]*3 if commands==6 else []))
    gradient=gradient*scales
    return gradient[:,:56].square().sum(-1).mean(),gradient[:,56:].square().sum(-1).mean()


class ExperimentPPO(PPO):
    def __init__(self,*args,split_gp=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.split_gp=split_gp

    def update(self):
        if not self.split_gp:
            return super().update()
        # Same on-policy storage, clipped objective, KL adaptation and independent
        # supervised optimizer. Only the additional actor loss differs.
        totals=dict(value=0.,surrogate=0.,kl=0.,sensory=0.,command=0.,gp_seconds=0.)
        count=0
        ramp=min(1.,max(0.,self.learning_iteration/1000.))
        for batch in self.storage.mini_batch_generator(self.num_group,self.num_mini_batches,self.num_learning_epochs):
            obs,critic,history,_,commands,actions,old_value,advantage,returns,old_logp,old_mu,old_sigma=batch
            latent=self.encoder.encode(history).detach()
            raw=build_actor_input(latent,obs,commands,self.actor_velocity_frame)
            self.actor_critic.act(raw)
            logp=self.actor_critic.get_actions_log_prob(actions)
            value=self.actor_critic.evaluate(critic)
            mu,sigma=self.actor_critic.action_mean,self.actor_critic.action_std
            with torch.no_grad():
                kl=(torch.log(sigma/old_sigma+1e-5)+(old_sigma.square()+(old_mu-mu).square())/(2*sigma.square())-.5).sum(-1).mean()
                if self.desired_kl is not None and self.schedule=='adaptive':
                    if kl>2*self.desired_kl: self.learning_rate=max(1e-5,self.learning_rate/1.5)
                    elif 0<kl<self.desired_kl/2: self.learning_rate=min(1e-2,self.learning_rate*1.5)
                    for group in self.optimizer.param_groups: group['lr']=self.learning_rate
            ratio=torch.exp(logp-old_logp.squeeze(-1))
            adv=advantage.squeeze(-1)
            surrogate=torch.maximum(-adv*ratio,-adv*ratio.clamp(1-self.clip_param,1+self.clip_param)).mean()
            value_error=(value-returns).square()
            if self.use_clipped_value_loss:
                clipped=old_value+(value-old_value).clamp(-self.clip_param,self.clip_param)
                value_error=torch.maximum(value_error,(clipped-returns).square())
            value_loss=value_error.mean()
            start=time.perf_counter()
            sensory,command=split_gradient_penalty(self.actor_critic.actor,raw,commands.shape[-1])
            totals['gp_seconds']+=time.perf_counter()-start
            loss=surrogate+self.value_loss_coef*value_loss-self.entropy_coef*self.actor_critic.entropy.mean()
            loss=loss+ramp*(.002*sensory+.0002*command)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),self.max_grad_norm)
            self.optimizer.step()
            for key,tensor in [('value',value_loss),('surrogate',surrogate),('kl',kl),('sensory',sensory),('command',command)]:
                totals[key]+=tensor.detach().item()
            count+=1
        extra_total=0.;extra_count=0;metrics_sum={}
        if self.extra_optimizer is not None:
            for _,critic,history in self.storage.encoder_mini_batch_generator(self.num_mini_batches,self.est_num_learning_epochs):
                self.encoder.encode(history)
                encoded=self.encoder.get_encoder_out()
                loss=encoder_supervision_loss(encoded,critic,self.est_target_dim,self.est_target_offset,
                    self.est_target_scales,self.est_low_speed_weight,self.est_low_speed_threshold,self.est_low_speed_bias_coef)
                with torch.no_grad():
                    metrics=encoder_supervision_metrics(encoded,critic,self.est_target_dim,self.est_target_offset)
                    if self.est_low_speed_weight!=1. or self.est_low_speed_bias_coef>0.:
                        target=critic[:,self.est_target_offset:self.est_target_offset+6]
                        mask=(target.norm(dim=-1)<self.est_low_speed_threshold).to(encoded.dtype)[:,None]
                        bias=((encoded[:,:6]-target)*mask).sum(0)/mask.sum().clamp_min(1.)
                        metrics.update(low_speed_bias=bias.square().mean().sqrt(),low_speed_fraction=mask.mean())
                    for key,tensor in metrics.items(): metrics_sum[key]=metrics_sum.get(key,0.)+tensor.item()
                self.extra_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(),self.est_max_grad_norm)
                self.extra_optimizer.step()
                extra_count+=1;extra_total+=loss.item()
        self.last_encoder_metrics={k:v/max(1,extra_count) for k,v in metrics_sum.items()}
        self.last_smoothness_metrics={
            'sensory_gp':totals['sensory']/max(count,1),'command_gp':totals['command']/max(count,1),
            'sensory_gp_coefficient':.002*ramp,'command_gp_coefficient':.0002*ramp,
            'sensory_gp_seconds':totals['gp_seconds']}
        self.storage.clear()
        return totals['value']/max(count,1),extra_total/max(extra_count,1),totals['surrogate']/max(count,1),totals['kl']/max(count,1)
