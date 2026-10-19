"""One shared whole-body Actor with optional explicit errors and FiLM."""
import torch
from torch import nn
from .features import tracking_errors, activity_features


class ExperimentActor(nn.Module):
    __constants__ = ['commands', 'use_errors', 'use_gates', 'use_film']

    def __init__(self, experiment):
        super().__init__()
        self.commands = experiment.commands
        self.use_errors = 'E' in experiment.factors
        self.use_gates = 'G' in experiment.factors
        self.use_film = 'F' in experiment.factors
        state_dim = 56 + 3*self.use_errors
        if self.use_film:
            self.network = nn.Sequential(nn.Linear(state_dim,512),nn.ELU(),nn.Linear(512,256),nn.ELU())
            self.conditioner = nn.Sequential(nn.Linear(self.commands+3,64),nn.ELU(),nn.Linear(64,512))
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)
            self.output = nn.Sequential(nn.Linear(256,128),nn.ELU(),nn.Linear(128,13))
        else:
            self.network = nn.Sequential(nn.Linear(experiment.actor_dim,512),nn.ELU(),nn.Linear(512,256),
                                         nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,13))
            self.conditioner = nn.Identity()
            self.output = nn.Identity()

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        state = raw[..., :56]
        commands = raw[..., 56:]
        if self.use_errors:
            state = torch.cat((state,tracking_errors(raw)),dim=-1)
        if self.use_gates:
            commands = torch.cat((commands,activity_features(raw,self.commands)),dim=-1)
        if self.use_film:
            feature = self.network(state)
            modulation = .1*torch.tanh(self.conditioner(commands))
            feature = feature*(1.+modulation[...,:256])+modulation[...,256:]
            return self.output(feature)
        return self.network(torch.cat((state,commands),dim=-1))
