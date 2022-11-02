from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        dist = self(observation)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        return ptu.to_numpy(action)

    def forward(self, observation: torch.FloatTensor):
        mean = self.mean_net(observation)
        log_std = torch.tanh(self.logstd)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clip(log_std, log_std_min, log_std_max)
        std = log_std.exp()
        dist = sac_utils.SquashedNormal(mean, std)
        return dist

    def update(self, obs, critic):
        obs = ptu.from_numpy(obs)

        dist = self(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Qs = critic(obs, action)
        actor_Q = torch.min(*actor_Qs)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), self.alpha.item()