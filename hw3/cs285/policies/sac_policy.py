from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools
from torch import distributions

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
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        """ 
            Return sample from action distribution if sampling, else return the mean.
        """
        observation = ptu.from_numpy(obs)
        action_distribution = self(observation)
        if sample:
            action = action_distribution.sample()
        else:
            assert not self.discrete, "why are you taking the mean action of a discrete action space?"
            action = action_distribution.mean
        return ptu.to_numpy(action)


    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            batch_dim = batch_mean.shape[0]
            
            log_scale = self.logstd.clip(min=self.log_std_bounds[0], max=self.log_std_bounds[1])
            scale = torch.diag(torch.exp(log_scale))
            batch_scale = scale.repeat(batch_dim, 1, 1)
            
            action_distribution = sac_utils.SquashedNormal(loc=batch_mean, scale=batch_scale, 
                                                           min=self.action_range[0], max=self.action_range[1])
            return action_distribution

    def update(self, ob_no: torch.Tensor, critic):
        # policy gradient on actor network
        n_trajectory = ob_no.shape[0]
        ac_t_dist = self.forward(ob_no)
        ac_t_na = ac_t_dist.sample()
        q_t_n = self.critic(ob_no, ac_t_na)
        adv_n = q_t_n
        
        log_action_probability_n = ac_t_dist.log_prob(ac_t_na)
        assert log_action_probability_n.shape == (n_trajectory,)
        
        actor_loss = -(log_action_probability_n * adv_n).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # update alpha (entropy regularizer)
        
        # alpha_loss = - self.alpha * (log_action_probability_n + self.target_entropy).mean()
        alpha_loss = - self.alpha * (log_action_probability_n.mean() + self.target_entropy)
        
        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()
        
        return actor_loss.item(), alpha_loss.item(), self.alpha.item()