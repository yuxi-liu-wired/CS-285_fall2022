from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent

class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, reward_n, next_ob_no, terminal_n):
        critic_loss = 0
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss += self.critic.update(ob_no, ac_na, next_ob_no, reward_n, terminal_n)
        critic_loss /= self.agent_params['num_critic_updates_per_agent_update']
        
        adv_n = self.estimate_advantage(ob_no, next_ob_no, reward_n, terminal_n)

        actor_loss = 0
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss += self.actor.update(ob_no, ac_na, adv_n)
        actor_loss /= self.agent_params['num_actor_updates_per_agent_update']

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, reward_n, terminal_n):
        v_t = self.critic.forward_np(ob_no)
        v_tp1 = self.critic.forward_np(next_ob_no)
        q_t = reward_n + self.gamma * v_tp1 * (1.0 - terminal_n)
        adv_n = q_t - v_t
        if self.standardize_advantages:
            adv_n = normalize(adv_n, np.mean(adv_n), np.std(adv_n))
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
