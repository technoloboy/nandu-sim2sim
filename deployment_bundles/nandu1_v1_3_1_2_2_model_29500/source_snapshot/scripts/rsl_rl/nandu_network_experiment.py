"""Isolated NANDU network experiments: list, train, play, export. Baseline scripts stay unchanged."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'rsl_rl'))
from rsl_rl.nandu_experiments.contract import Experiment, VARIANTS, StaticPitchYawExperiment


def parser_for(mode):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('list','train','play','export'))
    parser.add_argument('--task')
    parser.add_argument('--checkpoint_path',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--learning-diagnostic', choices=('L0','L1','L2','L3'),
                        help='A-only legacy r1 learning diagnosis; L1 freezes Encoder, L2 adaptation, L3 both')
    parser.add_argument('--diagnostic-seed-checkpoint', type=Path,
                        help='Explicit fork of an original r1 A checkpoint AND all optimizer states; not normal resume')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--num_envs',type=int,default=None)
    parser.add_argument('--max_iterations',type=int,default=30000,help='Total target iteration, including completed iterations on resume')
    parser.add_argument('--save_interval',type=int,default=500)
    parser.add_argument('--run_name',default='network_ablation')
    parser.add_argument('--max_steps',type=int,default=1000)
    parser.add_argument('--random_commands',action='store_true')
    parser.add_argument('--joystick',action='store_true',help='Play with the standalone mouse joystick (one environment, desktop only)')
    parser.add_argument('--evaluation_stage',type=int,choices=range(5))
    parser.add_argument('--allow_untrained_stage',action='store_true')
    parser.add_argument('--up3-total-mass',type=float)
    parser.add_argument('--rear-mass',type=float)
    parser.add_argument('--rear-x',type=float)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--no-onnx',action='store_true')
    parser.add_argument('--benchmark-fixed-inputs',action='store_true',
                        help='Train-only performance harness: Stage0 command (0,0,.9), zero issued actions')
    if mode in ('train','play'):
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    return parser


def validate_joystick_options(args):
    if not getattr(args,'joystick',False):return
    if args.mode!='play':raise ValueError('--joystick is a Play-only option')
    if getattr(args,'headless',False):raise ValueError('--joystick requires a desktop; do not use --headless')
    if args.random_commands:raise ValueError('--joystick cannot be combined with --random_commands')
    if args.num_envs not in (None,1):raise ValueError('--joystick requires --num_envs 1')


def validate_args(args):
    validate_joystick_options(args)
    if args.benchmark_fixed_inputs and args.mode!='train':
        raise ValueError('--benchmark-fixed-inputs is a train-only performance harness')
    if args.mode=='list': return None
    if not args.task: raise ValueError('--task is required')
    exp=Experiment.from_task(args.task)
    from rsl_rl.nandu_adaptive_balance.contract import AdaptiveBalanceExperiment, checkpoint_experiment
    diagnostic=getattr(args,'learning_diagnostic',None)
    seed_checkpoint=getattr(args,'diagnostic_seed_checkpoint',None)
    if diagnostic is not None:
        if args.mode!='train' or type(exp) is not AdaptiveBalanceExperiment:
            raise ValueError('Learning diagnostics are train-only and belong to adaptive A, not other tasks')
        exp=replace(exp,behavior_revision=1)
        if seed_checkpoint is None and not (args.resume and args.checkpoint_path is not None):
            raise ValueError('Diagnostic profile requires --diagnostic-seed-checkpoint or its own resume checkpoint')
    if seed_checkpoint is not None:
        if diagnostic is None:raise ValueError('Diagnostic seed requires an explicit learning diagnostic profile')
        if args.resume or args.checkpoint_path is not None:
            raise ValueError('Diagnostic seed and ordinary --resume/--checkpoint_path are exclusive')
    if args.num_envs is not None and args.num_envs<=0: raise ValueError('--num_envs must be positive')
    if args.max_steps<0 or args.max_iterations<1 or args.save_interval<1:
        raise ValueError('Invalid rollout/iteration/save limit')
    if args.mode=='train':
        if args.task!=exp.task: raise ValueError('Training requires the training task, not Play alias')
        if seed_checkpoint is None and args.resume != (args.checkpoint_path is not None):
            raise ValueError('Normal resume requires both --resume and --checkpoint_path; no implicit transfers')
        if any(x is not None for x in (args.up3_total_mass,args.rear_mass,args.rear_x)):
            raise ValueError('Fixed payload arguments are only supported by v1.3.1-family Play')
    elif args.mode in ('play','export') and args.checkpoint_path is None:
        raise ValueError('Play/export requires explicit --checkpoint_path')
    if args.mode=='export' and args.output is None: raise ValueError('Export requires --output')
    if args.resume and args.mode!='train': raise ValueError('--resume is a training option')
    if exp.family!='v1-3-1' and any(x is not None for x in (args.up3_total_mass,args.rear_mass,args.rear_x)):
        raise ValueError('Payload override belongs only to v1.3.1 family')
    path=seed_checkpoint or args.checkpoint_path
    if path is not None:
        import torch
        state=torch.load(path,map_location='cpu',weights_only=False)
        if args.mode in ('play','export') and isinstance(exp,AdaptiveBalanceExperiment):
            saved=checkpoint_experiment(state)
            if saved.task!=exp.task:raise ValueError('Requested task does not match checkpoint task')
            exp=saved
        exp.validate_checkpoint(state,require_optimizer=args.resume or seed_checkpoint is not None)
        saved_profile=state.get('train_cfg',{}).get('learning_diagnostic')
        if seed_checkpoint is not None and saved_profile is not None:
            raise ValueError('Diagnostic seed must be the original checkpoint, not another profile run')
        if args.resume and isinstance(exp,AdaptiveBalanceExperiment) and saved_profile!=diagnostic:
            raise ValueError('Diagnostic profile mismatch on resume')
    return exp


def configure_fixed_input_benchmark(runner):
    """Measure matched physics streams while still executing policy and PPO work."""
    import torch
    runner.cfg['benchmark_fixed_inputs']=True
    term=runner.env.unwrapped.command_manager.get_term(runner.experiment.command_name)
    term.set_evaluation_stage(0)
    command=torch.tensor([0.,0.,.9],device=runner.device).expand(runner.env.num_envs,-1).clone()
    term.set_external_command(command)
    original_act=runner.alg.act
    def fixed_act(*args,**kwargs):
        sampled=original_act(*args,**kwargs)
        issued=torch.zeros_like(sampled)
        runner.alg.transition.actions=issued
        runner.alg.transition.actions_log_prob=runner.alg.actor_critic.get_actions_log_prob(issued).detach()
        return issued
    runner.alg.act=fixed_act


def run_simulation(args,exp):
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(args)
    app=launcher.app
    env=None
    joystick=None
    try:
        import gymnasium as gym
        import torch
        import bipedal_locomotion
        from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.nandu_experiments.runner import ExperimentRunner
        from rsl_rl.nandu_adaptive_balance.contract import AdaptiveBalanceExperiment
        task=exp.task if args.mode=='train' else exp.play_task
        cfg=parse_env_cfg(task,device=args.device,num_envs=args.num_envs or (4096 if args.mode=='train' else 1))
        cfg.seed=args.seed
        agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point').to_dict()
        agent.update(seed=args.seed,save_interval=args.save_interval)
        if isinstance(exp,AdaptiveBalanceExperiment):
            term_cfg=getattr(cfg.commands,exp.command_name)
            term_cfg.behavior_revision=exp.behavior_revision
            term_cfg.checkpoint_schema=exp.schema
            cfg.network_experiment=exp.manifest()
            agent.update(learning_diagnostic=getattr(args,'learning_diagnostic',None),
                         checkpoint_schema=exp.schema,network_experiment=exp.manifest())
        if args.benchmark_fixed_inputs:agent['benchmark_fixed_inputs']=True
        try:
            agent['source_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            agent['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
        except subprocess.CalledProcessError:
            agent['source_revision']='unavailable'
        if isinstance(exp,AdaptiveBalanceExperiment):
            sources=list((ROOT/'rsl_rl/rsl_rl/nandu_adaptive_balance').glob('*.py'))
            mdp_dir=ROOT/'exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/NANDU1'
            sources.extend(mdp_dir.glob('*.py'))  # Include inherited command/reward/randomization semantics.
            sources.extend((ROOT/'rsl_rl/rsl_rl/nandu_experiments').glob('*.py'))
            sources.extend(ROOT/name for name in (
                'scripts/rsl_rl/nandu_network_experiment.py',
                'exts/bipedal_locomotion/bipedal_locomotion/assets/config/nandu1_cfg.py',
                'exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/nandu1_rsl_rl_ppo_cfg.py',
                'rsl_rl/rsl_rl/algorithm/ppo.py', 'rsl_rl/rsl_rl/storage/rollout_storage.py',
                'rsl_rl/rsl_rl/modules/actor_critic.py', 'rsl_rl/rsl_rl/modules/dual_head_mlp_encoder.py',
                'IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py',
                'IsaacLab/source/isaaclab/isaaclab/managers/command_manager.py'))
            agent['source_files_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(set(sources))}
        if args.mode=='play' and exp.family=='v1-3-1':
            from nandu1_v131_deployment import configure_payload
            configure_payload(cfg,.67148 if args.up3_total_mass is None else args.up3_total_mass,
                0. if args.rear_mass is None else args.rear_mass,-.275 if args.rear_x is None else args.rear_x)
        log_dir=None
        if args.mode=='train':
            log_dir=ROOT/'logs/rsl_rl'/exp.experiment_name/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')+'_'+args.run_name)
            log_dir.mkdir(parents=True,exist_ok=False)
            (log_dir/'experiment.json').write_text(json.dumps({**exp.manifest(),'seed':args.seed,
                'source_revision':agent['source_revision'],'source_dirty':agent.get('source_dirty'),
                'source_files_sha256':agent.get('source_files_sha256'),
                'learning_diagnostic':getattr(args,'learning_diagnostic',None),
                'diagnostic_seed_checkpoint':str(args.diagnostic_seed_checkpoint) if args.diagnostic_seed_checkpoint else None,
                **({'benchmark_fixed_inputs':True} if args.benchmark_fixed_inputs else {})},indent=2)+'\n')
        env=RslRlVecEnvWrapper(gym.make(task,cfg=cfg))
        runner_type=ExperimentRunner
        if isinstance(exp,AdaptiveBalanceExperiment):
            from rsl_rl.nandu_adaptive_balance.runner import AdaptiveBalanceRunner
            runner_type=AdaptiveBalanceRunner
        runner=runner_type(env,agent,exp,log_dir=str(log_dir) if log_dir else None,device=args.device)
        if args.checkpoint_path: runner.load(args.checkpoint_path,load_optimizer=args.resume)
        if args.diagnostic_seed_checkpoint:
            runner.load(args.diagnostic_seed_checkpoint,load_optimizer=True,diagnostic_seed=True)
            (log_dir/'diagnostic_seed.json').write_text(json.dumps(runner.cfg['diagnostic_seed'],indent=2)+'\n')
        if args.benchmark_fixed_inputs:configure_fixed_input_benchmark(runner)
        if args.mode=='train':
            remaining=args.max_iterations-runner.current_learning_iteration
            if remaining<=0: raise ValueError('--max_iterations must exceed checkpoint iteration')
            runner.learn(remaining,init_at_random_ep_len=not (args.resume or args.diagnostic_seed_checkpoint))
            if runner.writer is not None:
                runner.writer.flush()
            print('NETWORK_TRAIN_PASS',exp.task,'iteration',runner.current_learning_iteration,
                  'checkpoint',log_dir/f'model_{runner.current_learning_iteration}.pt',flush=True)
            if args.benchmark_fixed_inputs and torch.cuda.is_available():
                print('NETWORK_BENCHMARK_MEMORY',json.dumps(dict(
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())),flush=True)
        else:
            state=torch.load(args.checkpoint_path,map_location='cpu',weights_only=False)
            learned=int(state['environment_state_dict'][exp.command_name]['stage'])
            stage=learned if args.evaluation_stage is None else args.evaluation_stage
            if stage>learned and not args.allow_untrained_stage:
                raise ValueError('Requested stage exceeds trained stage; use --allow_untrained_stage for a labeled stress test')
            from rsl_rl.nandu_experiments.deployment import export_for_play
            print('NETWORK_PLAY_EXPORT',export_for_play(args.checkpoint_path),flush=True)
            term=env.unwrapped.command_manager.get_term(exp.command_name)
            term.set_evaluation_stage(stage)
            runner.env.reset()
            keyboard=None
            if args.joystick:
                from nandu_network_joystick import NetworkJoystick
                joystick=NetworkJoystick(exp,term,env.unwrapped,stage)
            elif not args.random_commands:
                if args.headless:
                    term.set_external_command(term.command.detach().clone())
                elif exp.family=='v1-3-1':
                    from nandu1_v131_deployment import Nandu1V131Keyboard
                    keyboard=Nandu1V131Keyboard(term)
                else:
                    import importlib.util
                    spec=importlib.util.spec_from_file_location('_network_keyboard',ROOT/'scripts/rsl_rl/play-nandu1-v1-4-1.py')
                    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                    keyboard=module.Nandu1V141Keyboard(term,stage)
            runner.alg.actor_critic.eval();runner.alg.encoder.eval()
            steps=0
            with torch.inference_mode():
                while app.is_running() and (args.max_steps==0 or steps<args.max_steps):
                    if joystick:
                        ready=joystick.apply()
                        if ready is False:break
                        if ready is None:continue
                    if keyboard: keyboard.apply()
                    obs=runner.env.get_observations()
                    actions=runner.alg.actor_critic.act_inference(runner.build_actor_input(
                        obs['policy'],obs['obsHistory'].flatten(1),obs['commands']))
                    _,_,dones,_=runner.env.step(actions)
                    if joystick:joystick.after_step(bool(dones.any()))
                    steps+=1
            print('NETWORK_PLAY_PASS',exp.task,'steps',steps,'stage',stage,flush=True)
    except BaseException:
        # SimulationApp.close may exit the interpreter: preserve failures in logs.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if joystick is not None:joystick.close()
        if env is not None: env.close()
        app.close()


def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'list'
    parser=parser_for(mode)
    args=parser.parse_args()
    try: exp=validate_args(args)
    except ValueError as exc: parser.error(str(exc))
    if args.mode=='list':
        for family in ('v1-3-1','v1-4-1'):
            for variant in VARIANTS:
                e=Experiment(family,variant)
                print(f'{e.task:43s} {e.factors or "control":7s} history={e.history_dim} actor_features={e.actor_dim} critic={108+e.commands}')
        e=StaticPitchYawExperiment('v1-3-1',2)
        print(f'{e.task:43s} {e.factors:7s} history={e.history_dim} actor_features={e.actor_dim} critic={108+e.commands} reward_ablation=static_pitch_yaw')
        from rsl_rl.nandu_adaptive_balance.contract import AdaptiveBalanceExperiment
        e=AdaptiveBalanceExperiment()
        print(f'{e.task:43s} {e.factors:7s} history={e.history_dim} actor_features={e.actor_dim} critic={e.interface["critic_with_commands"]} adaptive_balance')
        from rsl_rl.nandu_adaptive_balance.ablations import ABLATION_SUFFIXES, AdaptiveBalanceAblationExperiment
        for suffix in ABLATION_SUFFIXES:
            e=AdaptiveBalanceAblationExperiment(suffix=suffix)
            print(f'{e.task:43s} {e.factors:7s} history={e.history_dim} actor_features={e.actor_dim} '
                  f'critic={e.interface["critic_with_commands"]} ablation={e.group} mode={e.mode_sampling} '
                  f'height_memory={e.height_memory_enabled}')
    elif args.mode=='export':
        from rsl_rl.nandu_experiments.deployment import export_bundle
        print(json.dumps(export_bundle(args.checkpoint_path,args.output,onnx=not args.no_onnx),indent=2))
    else:
        run_simulation(args,exp)


if __name__=='__main__': main()
