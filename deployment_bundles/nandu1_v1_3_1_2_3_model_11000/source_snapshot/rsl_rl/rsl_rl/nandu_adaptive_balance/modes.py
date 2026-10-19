"""Episode-local training coverage, not performance-conditioned sample filtering."""
import torch


class TrainingModes:
    NORMAL, STATIC, STOP, CYCLE = range(4)

    def __init__(self, num_envs, device='cpu'):
        self.mode = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.age = torch.zeros(num_envs, device=device)
        self.stop_time = torch.zeros_like(self.age)
        self.height_a = torch.ones_like(self.age)
        self.height_b = torch.ones_like(self.age)
        self._step_id = None

    def reset(self, ids, low, high):
        if ids is None:
            ids = torch.arange(self.mode.numel(),device=self.mode.device)
        n = len(ids)
        u = torch.rand(n, 5, device=self.mode.device)
        self.mode[ids] = torch.bucketize(u[:,0].contiguous(), u.new_tensor([.30,.55,.80]))
        self.age[ids] = 0
        self.stop_time[ids] = 4.+4.*u[:,1]
        # Opposite half-ranges guarantee separation >= half the stage span.
        # Eight equal bins: select one of four lower bins, then its opposite.
        lower_bin = (u[:,2]*4).long()
        a = low + (lower_bin+u[:,4])*(high-low)/8
        b = a + .5*(high-low)
        swap = u[:,3] < .5
        self.height_a[ids] = torch.where(swap,b,a)
        self.height_b[ids] = torch.where(swap,a,b)

    def advance(self, dt, step_id=None):
        if step_id is not None and self._step_id == step_id:
            return
        self.age.add_(dt)
        self._step_id = step_id

    def normal_mask(self):
        return self.mode == self.NORMAL

    def targets(self, sampled):
        result = sampled.clone()
        held = (self.mode == self.STATIC) | (self.mode == self.CYCLE)
        stopped = (self.mode == self.STOP) & (self.age >= self.stop_time)
        result[held | stopped, :2] = 0
        result[:,2] = torch.where(self.mode == self.STATIC,self.height_a,result[:,2])
        cycle_h = torch.where((self.age >= 8.) & (self.age < 16.), self.height_b,self.height_a)
        result[:,2] = torch.where(self.mode == self.CYCLE,cycle_h,result[:,2])
        return result

    def state_dict(self):
        return {k:getattr(self,k).detach().cpu().clone() for k in ('mode','age','stop_time','height_a','height_b')}

    def load_state_dict(self, state):
        for k in ('mode','age','stop_time','height_a','height_b'):
            value = torch.as_tensor(state[k], device=self.mode.device)
            if value.shape == getattr(self,k).shape:
                getattr(self,k).copy_(value)
