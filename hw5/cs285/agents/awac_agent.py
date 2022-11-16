from collections import OrderedDict

from cs285.critics.dqn_critic import DQNCritic
from cs285.critics.cql_critic import CQLCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from cs285.exploration.rnd_model import RNDModel
from .dqn_agent import DQNAgent
from cs285.policies.MLP_policy import MLPPolicyAWAC
import numpy as np
import torch


class AWACAgent(DQNAgent):
    def __init__(self, env, agent_params, normalize_rnd=True, rnd_gamma=0.99):
        super(AWACAgent, self).__init__(env, agent_params)
        
        self.replay_buffer = MemoryOptimizedReplayBuffer(100000, 1, float_obs=True)
        self.num_exploration_steps = agent_params['num_exploration_steps']
        self.offline_exploitation = agent_params['offline_exploitation']

        self.exploitation_critic = DQNCritic(agent_params, self.optimizer_spec)
        self.exploration_critic = DQNCritic(agent_params, self.optimizer_spec)
        
        self.exploration_model = RNDModel(agent_params, self.optimizer_spec)
        self.explore_weight_schedule = agent_params['explore_weight_schedule']
        self.exploit_weight_schedule = agent_params['exploit_weight_schedule']
        
        self.actor = ArgMaxPolicy(self.exploitation_critic)
        self.eval_policy = self.awac_actor = MLPPolicyAWAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            self.agent_params['awac_lambda'],
        )

        self.exploit_rew_shift = agent_params['exploit_rew_shift']
        self.exploit_rew_scale = agent_params['exploit_rew_scale']
        self.eps = agent_params['eps']

        self.running_rnd_rew_mean = 0
        self.running_rnd_rew_std = 1
        self.normalize_rnd = normalize_rnd
        self.rnd_gamma = rnd_gamma

    def _get_qvals(self, critic, ob_no, ac_na):
        # Get the online critic Q value. NOT the target critic Q value
        if self.agent_params['discrete']:
            qa_values = critic.q_net(ob_no)
            q_values = torch.gather(qa_values, 1, ac_na.type(torch.int64).unsqueeze(1)).squeeze(1)
        else:
            obs_action = torch.cat([ob_no, ac_na], dim=1)
            assert obs_action.shape == (ob_no.shape[0], self.ob_dim + self.ac_dim)
            q_values = critic.q_net(obs_action).squeeze(1)
            
        assert q_values.shape == (ob_no.shape[0],)
        return q_values

    def estimate_advantage(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, n_actions=10):
        samples_n = ob_no.shape[0]
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        re_n = ptu.from_numpy(re_n)
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)
        
        # The tutor said: use the q_net for estimating the advantage,
        # the target Q is only used to update the q_net and do double Q-learning.
        
        # Compute (when discrete) or estimate (when continuous) value V(s)
        # Use the exploitation critic.
        ac_dist = self.awac_actor(ob_no)
        if self.agent_params['discrete']:
            # explicit summation
            assert ac_na.shape == (samples_n,)
            ac_probs = ac_dist.probs
            qa_values = self.exploitation_critic.q_net(ob_no)
            assert ac_probs.shape == qa_values.shape == (samples_n, self.agent_params['ac_dim'])
            v_pi = (qa_values * ac_probs).sum(dim=1)
        else: 
            # estimate by samples
            assert ac_na.shape == (samples_n, self.agent_params['ac_dim'])
            v_pi = torch.zeros((samples_n,))
            for _ in range(n_actions):
                ac_samples = ac_dist.sample()
                assert ac_samples.shape == ac_na.shape 
                v_pi += self._get_qvals(self.exploitation_critic, ob_no, ac_na)
            v_pi /= n_actions
        assert v_pi.shape == (samples_n,)

        q_vals = self._get_qvals(self.exploitation_critic, ob_no, ac_na)

        return q_vals - v_pi

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}

        if self.t > self.num_exploration_steps:
            # exploration is over; set the actor to optimize the exploitation critic
            self.actor.set_critic(self.exploitation_critic)

        if (        self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # Get Reward Weights
            # COMMENT: Until part 3, explore_weight = 1, and exploit_weight = 0
            explore_weight = self.explore_weight_schedule.value(self.t)
            exploit_weight = self.exploit_weight_schedule.value(self.t)

            # Exploration reward on observation
            expl_bonus = self.exploration_model.forward_np(next_ob_no)
            
            if self.normalize_rnd:
                rew_mean = expl_bonus.mean()
                rew_std = expl_bonus.std()

                # Normalize the exploration bonus, as RND values vary highly in magnitude.
                expl_bonus = normalize(expl_bonus, rew_mean, rew_std)
                
                # exponential moving average
                self.running_rnd_rew_mean = self.running_rnd_rew_mean * self.rnd_gamma + rew_mean * (1 - self.rnd_gamma)
                self.running_rnd_rew_std = self.running_rnd_rew_std * self.rnd_gamma + rew_std * (1 - self.rnd_gamma)
            
            # Reward Calculations
            assert expl_bonus.shape == re_n.shape
            env_reward = (re_n + self.exploit_rew_shift) * self.exploit_rew_scale
            mixed_reward = explore_weight * expl_bonus + exploit_weight * re_n # The tutor said: do not use env_reward here.
            # https://edstem.org/us/courses/24422/discussion/2082447?comment=4934521
            assert env_reward.shape == mixed_reward.shape == re_n.shape
            # Update Exploration Model and Critics

            # Update the exploration model (based off s')
            expl_model_loss = self.exploration_model.update(next_ob_no)
            # Update the exploration critic (based off mixed_reward)
            exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)
            # Update the exploitation critic (based off env_reward)
            exploitation_critic_loss = self.exploitation_critic.update(ob_no, ac_na, next_ob_no, env_reward, terminal_n)

            # Update AWAC actor
            adv_n = self.estimate_advantage(ob_no, ac_na, re_n, next_ob_no, terminal_n)
            actor_loss = self.awac_actor.update(ob_no, ac_na, adv_n)

            # Update critic target networks
            if self.num_param_updates % self.target_update_freq == 0:
                self.exploration_critic.update_target_network()
                self.exploitation_critic.update_target_network()

            # Logging
            log['Exploration Critic Loss'] = exploration_critic_loss['Training Loss']
            log['Exploitation Critic Loss'] = exploitation_critic_loss['Training Loss']
            log['Exploration Model Loss'] = expl_model_loss
            log['Actor Loss'] = actor_loss

            self.num_param_updates += 1

        self.t += 1
        return log

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        perform_random_action = np.random.random() < self.eps or self.t < self.learning_starts

        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            processed = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(processed)

        next_obs, reward, done, info = self.env.step(action)
        self.last_obs = next_obs.copy()

        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.env.reset()
