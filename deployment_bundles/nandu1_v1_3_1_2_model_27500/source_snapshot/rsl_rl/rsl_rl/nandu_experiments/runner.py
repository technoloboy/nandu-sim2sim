"""Independent construction and strict checkpoint handling; reuse baseline rollout/logging."""
import copy
import os
import torch
from ..runner.on_policy_runner import OnPolicyRunner
from ..modules import DualHeadMLPEncoder, ActorCritic
from .network import ExperimentActor
from .history import ObservationAdapter
from .ppo import ExperimentPPO
from .features import raw_tracking_errors, activity_features


def build_models(exp,seed=42,encoder_cfg=None,policy_cfg=None,device='cpu'):
    ec=dict(encoder_cfg or {})
    ec.pop('class_name',None)
    ec.update(num_input_dim=exp.history_dim,num_output_dim=7,output_detach=True)
    pc=dict(policy_cfg or {})
    pc.pop('class_name',None)
    pc.update(actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu')
    # Independent streams prevent a wider Encoder from changing Critic/Actor initialization.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed+1001)
        encoder=DualHeadMLPEncoder(**ec).to(device)
        torch.random.default_generator.manual_seed(seed+2001)
        ac=ActorCritic(56+exp.commands,108+exp.commands,13,**pc).to(device)
        torch.random.default_generator.manual_seed(seed+2001)
        ac.actor=ExperimentActor(exp).to(device)
    return encoder,ac


class ExperimentRunner(OnPolicyRunner):
    def __init__(self,env,train_cfg,experiment,log_dir=None,device='cpu'):
        self.experiment=experiment
        self.cfg=copy.deepcopy(train_cfg)
        self.cfg.update(task_name=experiment.task,checkpoint_schema=experiment.schema,
                        interface_contract=experiment.interface,network_experiment=experiment.manifest())
        self.ecd_cfg=self.cfg['encoder'];self.policy_cfg=self.cfg['policy']
        self.alg_cfg=copy.deepcopy(self.cfg['algorithm'])
        self.alg_cfg.pop('class_name',None);self.alg_cfg.pop('obs_history_len',None)
        self.alg_cfg['actor_velocity_frame']=experiment.velocity_frame
        self.env=ObservationAdapter(env,'H' in experiment.factors)
        self.device=device;self.num_obs=49;self.num_commands=experiment.commands;self.obs_history_len=5
        obs=self.env.reset()
        expected={'policy':49,'critic':108,'commands':experiment.commands,'obsHistory':experiment.history_dim}
        for name,width in expected.items():
            if obs[name].flatten(1).shape!=(env.num_envs,width):
                raise ValueError(f'{name} violates network experiment interface')
        if env.num_actions!=13: raise ValueError('Network experiment needs 13 actions')
        encoder,ac=build_models(experiment,self.cfg.get('seed',42),self.ecd_cfg,self.policy_cfg,device)
        self.alg=ExperimentPPO(env.num_envs,encoder,ac,device=device,split_gp='P' in experiment.factors,**self.alg_cfg)
        self.num_steps_per_env=self.cfg['num_steps_per_env'];self.save_interval=self.cfg['save_interval']
        self.alg.init_storage(env.num_envs,self.num_steps_per_env,[49],[108+experiment.commands],
                              [experiment.history_dim],[experiment.commands],[13])
        self.log_dir=log_dir;self.writer=None;self.tot_timesteps=0;self.tot_time=0.;self.current_learning_iteration=0
        print(f'[NETWORK EXPERIMENT] {experiment.task}: {experiment.factors or "control"}; '
              f'history={experiment.history_dim}, raw_actor={56+experiment.commands}, features={experiment.actor_dim}')

    def log(self,locs,width=80,pad=35):
        super().log(locs,width,pad)
        with torch.no_grad():
            raw=self.build_actor_input(locs['obs'],locs['obs_history'],locs['commands'])
            errors=raw_tracking_errors(raw)
            gates=activity_features(raw,self.experiment.commands)
            # One host copy; these are estimated input diagnostics, not truth-based acceptance metrics.
            values=torch.cat((errors.square().mean(0).sqrt(),gates.mean(0))).cpu().tolist()
        for name,value in zip(('estimated_yaw_error_rms','estimated_vx_error_rms','estimated_height_error_rms',
                               'yaw_activity','vx_activity','height_activity'),values):
            self.writer.add_scalar('Network/'+name,value,locs['it'])
        self.writer.add_scalar('Network/actor_parameters',sum(p.numel() for p in self.alg.actor_critic.actor.parameters()),locs['it'])
        self.writer.add_scalar('Network/encoder_parameters',sum(p.numel() for p in self.alg.encoder.parameters()),locs['it'])

    def save(self,path,infos=None,iteration=None):
        state=dict(model_state_dict=self.alg.actor_critic.state_dict(),encoder_state_dict=self.alg.encoder.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(),extra_optimizer_state_dict=self.alg.extra_optimizer.state_dict(),
            environment_state_dict=self._environment_state_dict(),
            iter=self.current_learning_iteration if iteration is None else iteration,infos=infos,
            checkpoint_schema=self.experiment.schema,task_name=self.experiment.task,
            interface_contract=self.experiment.interface,network_experiment=self.experiment.manifest(),
            train_cfg=copy.deepcopy(self.cfg),tot_timesteps=self.tot_timesteps,tot_time=self.tot_time,
            source_revision=self.cfg.get('source_revision'),source_dirty=self.cfg.get('source_dirty'),
            rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        temporary=str(path)+'.partial'
        torch.save(state,temporary)
        os.replace(temporary,path)

    def load(self,path,load_optimizer=False):
        state=torch.load(path,map_location=self.device,weights_only=False)
        self.experiment.validate_checkpoint(state,require_optimizer=load_optimizer)
        # Validate every key/shape before applying any tensors.
        for key,module in [('model_state_dict',self.alg.actor_critic),('encoder_state_dict',self.alg.encoder)]:
            current=module.state_dict()
            if current.keys()!=state[key].keys() or any(current[k].shape!=state[key][k].shape for k in current):
                raise ValueError(f'Network experiment state dimensions mismatch: {key}')
        self.alg.actor_critic.load_state_dict(state['model_state_dict'],strict=True)
        self.alg.encoder.load_state_dict(state['encoder_state_dict'],strict=True)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.alg.extra_optimizer.load_state_dict(state['extra_optimizer_state_dict'])
            self.alg.learning_rate=self.alg.optimizer.param_groups[0]['lr']
            self._load_environment_state_dict(state['environment_state_dict'])
            # Resume learning state, but never combine old temporal state with new physics.
            self._restart_physics_after_resume()
            torch.set_rng_state(state['rng_state'].cpu())
            if state.get('cuda_rng_state') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng_state']])
        self.current_learning_iteration=int(state['iter'])
        self.alg.learning_iteration=self.current_learning_iteration
        self.tot_time=float(state.get('tot_time',0.));self.tot_timesteps=int(state.get('tot_timesteps',0))
        return state.get('infos')

    @torch.inference_mode()
    def _restart_physics_after_resume(self):
        term=self.env.unwrapped.command_manager.get_term(self.experiment.command_name)
        if hasattr(term,'restart_after_resume'):
            term.restart_after_resume()
        else:
            # Restart physics without counting a synthetic curriculum failure.
            raw=self.env.unwrapped
            ids=torch.arange(raw.num_envs,device=raw.device)
            raw.scene.reset(ids)
            raw.event_manager.apply(mode='reset',env_ids=ids,global_env_step_count=raw.common_step_counter)
            raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.)
            raw._nandu_v1_measurement_cache={}
            raw.action_manager.reset(ids);raw.observation_manager.reset(ids)
            term.reset(ids);raw.event_manager.reset(ids)
            raw.episode_length_buf[ids]=0
            raw.obs_buf=raw.observation_manager.compute(update_history=True)
        self.env.history.reset()
