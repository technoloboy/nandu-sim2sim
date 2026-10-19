"""Chronological previous-command history, shared by training and deployment."""
import torch


class CommandHistory:
    def __init__(self, num_envs, device='cpu', dtype=torch.float32):
        self.buffer = torch.zeros(num_envs,5,3,device=device,dtype=dtype)

    @torch.inference_mode()
    def reset(self, ids=None):
        if ids is None: self.buffer.zero_()
        else: self.buffer[ids] = 0.

    def advance(self, previous_applied, dones):
        self.buffer = torch.cat((self.buffer[:,1:],previous_applied[:,:3,None].transpose(1,2)),dim=1)
        self.buffer[dones.flatten().bool()] = 0.

    def combine(self, policy_history):
        states = policy_history.reshape(-1,5,49)
        return torch.cat((states,self.buffer),-1).flatten(1)


class ObservationAdapter:
    """Wrap an RSL env without altering baseline observation terms or noise caches."""
    def __init__(self, env, use_commands):
        self.env = env
        self.use_commands = use_commands
        self.history = CommandHistory(env.num_envs,device=env.device)
        self.current = None

    def __getattr__(self,name): return getattr(self.env,name)

    def _adapt(self,obs):
        result = dict(obs)
        if self.use_commands: result['obsHistory'] = self.history.combine(obs['obsHistory'])
        self.current = obs['commands'].detach().clone()
        return result

    def get_observations(self):
        return self._adapt(self.env.get_observations())

    def reset(self):
        # Isaac may cache inference tensors during rollout; reset must obey
        # the same execution context without creating inference parameters.
        with torch.inference_mode():
            self.env.reset()
            self.history.reset()
        return self.get_observations()

    def step(self,actions):
        if self.current is None: self.get_observations()
        previous = self.current.clone()
        obs,reward,dones,infos = self.env.step(actions)
        if self.use_commands: self.history.advance(previous,dones)
        return self._adapt(obs),reward,dones,infos
