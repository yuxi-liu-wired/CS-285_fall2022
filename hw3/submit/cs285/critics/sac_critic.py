from .base_critic import BaseCritic
from torch import nn
from torch import optim
import numpy as np
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import sac_utils
import torch

class SACCritic(nn.Module, BaseCritic):
    """
        Clipped double Q critic. 
        It contains two neural networks with the same architecture, each implementing
        a function of type Q(s_t, a_t).
        
        Running `forward(s_t, a_t)` on the critic results in 
            min(Q_1(s_t, a_t), Q_2(s_t, a_t))
    """
    def __init__(self, hparams):
        super(SACCritic, self).__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.gamma = hparams['gamma']
        self.Q1 = ptu.build_mlp(
            self.ob_dim + self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu'
        )
        self.Q2 = ptu.build_mlp(
            self.ob_dim + self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu'
        )
        self.Q1.to(ptu.device)
        self.Q2.to(ptu.device)
        self.loss = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.parameters(),
            self.learning_rate,
        )

    def forward(self, ob_no: torch.Tensor, ac_na: torch.Tensor):
        obs_action = torch.cat([ob_no, ac_na], dim=1)
        assert obs_action.shape == (ob_no.shape[0], self.ob_dim + self.ac_dim)
        
        q1 = self.Q1(obs_action).squeeze(1)
        q2 = self.Q2(obs_action).squeeze(1)
        assert q1.shape == (ob_no.shape[0],)
        assert q2.shape == (ob_no.shape[0],)
        
        q = torch.minimum(q1, q2) # clipped double Q-learning
        assert q.shape == (ob_no.shape[0],)
        
        return q

    def update(self, ob_no: torch.Tensor, ac_na: torch.Tensor, target_n: torch.Tensor):
        assert target_n.shape ==(target_n.shape[0],)
        
        obs_action = torch.cat([ob_no, ac_na], dim=1)
        assert obs_action.shape == (target_n.shape[0], self.ob_dim + self.ac_dim)
        
        q1 = self.Q1(obs_action).squeeze(1)
        q2 = self.Q2(obs_action).squeeze(1)
        assert q1.shape == target_n.shape
        assert q2.shape == target_n.shape
        
        # loss = self.loss(q1, target_n) + self.loss(q2, target_n)
        loss = ((q1 - target_n) ** 2 + (q2 - target_n) ** 2).sum() / target_n.shape[0]
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()