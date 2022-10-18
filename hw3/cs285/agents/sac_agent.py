from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
from cs285.infrastructure import sac_utils

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        n_samples = ob_no.shape[0]

        # sample next action
        ac_tp1_dist = self.actor(next_ob_no)
        # ac_tp1_na = ac_tp1_dist.mean
        ac_tp1_na = ac_tp1_dist.sample() # the critic doesn't need to do reparametrization trick.
        
        # compute entropy reward
        ac_tp1_logprob_n = ac_tp1_dist.log_prob(ac_tp1_na).sum(dim=1)
        assert ac_tp1_logprob_n.shape == (n_samples,)
        
        # compute target Q(s_{t+1}, a_{t+1})
        q_tp1_n = self.critic_target(next_ob_no, ac_tp1_na)
        assert q_tp1_n.shape == (n_samples,)
        
        # compute target for the two Q networks of the critic
        target_q_t_n = re_n + self.gamma * (1.0 - terminal_n) * (q_tp1_n - self.actor.alpha * ac_tp1_logprob_n)
        target_q_t_n = target_q_t_n.detach()
        assert target_q_t_n.shape == (n_samples,)
        
        # update critic
        critic_loss = self.critic.update(ob_no, ac_na, target_q_t_n)
        
        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)
        
        # update online critic
        critic_loss = 0.0
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss += self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        critic_loss /= self.agent_params['num_critic_updates_per_agent_update']
        
        # softly update (moving exponential average) target critic
        if self.training_step % self.critic_target_update_frequency == 0 and True:
            net1 = self.critic.Q1
            target_net1 = self.critic_target.Q1
            sac_utils.soft_update_params(net1, target_net1, self.critic_tau)
            
            net2 = self.critic.Q2
            target_net2 = self.critic_target.Q2
            sac_utils.soft_update_params(net2, target_net2, self.critic_tau)
        
        # update actor
        actor_loss, alpha_loss, alpha = 0.0, 0.0, self.actor.alpha
        if self.training_step % self.actor_update_frequency == 0 and True:
            alpha = 0.0
            for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
                actor_loss_t, alpha_loss_t, alpha_t = self.actor.update(ob_no, self.critic)
                actor_loss += actor_loss_t
                alpha_loss += alpha_loss_t
                alpha += alpha_t
            actor_loss /= self.agent_params['num_actor_updates_per_agent_update']
            alpha_loss /= self.agent_params['num_actor_updates_per_agent_update']
            alpha      /= self.agent_params['num_actor_updates_per_agent_update']

        # logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = alpha

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
