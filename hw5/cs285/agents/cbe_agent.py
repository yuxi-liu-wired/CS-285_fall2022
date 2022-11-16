from collections import OrderedDict

from cs285.critics.dqn_critic import DQNCritic
from cs285.critics.cql_critic import CQLCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from cs285.exploration.rnd_model import RNDModel
from .dqn_agent import DQNAgent
import numpy as np


class CBEAgent(DQNAgent):
    def __init__(self, env, agent_params):
        super(CBEAgent, self).__init__(env, agent_params)
        
        self.replay_buffer = MemoryOptimizedReplayBuffer(100000, 1, float_obs=True)
        self.num_exploration_steps = agent_params['num_exploration_steps']
        self.offline_exploitation = agent_params['offline_exploitation'] 
        # boolean. If set to TRUE, then the agent would explore for `num_exploration_steps`, then start exploiting
        # If set to FALSE, then the agent would always explore.

        self.exploitation_critic = CQLCritic(agent_params, self.optimizer_spec) # offline learning by CQL
        self.exploration_critic = DQNCritic(agent_params, self.optimizer_spec)  # online learning by DQN
        
        self.exploration_model = RNDModel(agent_params, self.optimizer_spec)    # compute state novelty
        self.explore_weight_schedule = agent_params['explore_weight_schedule']
        self.exploit_weight_schedule = agent_params['exploit_weight_schedule']
        
        self.actor = ArgMaxPolicy(self.exploration_critic)
        self.eval_policy = ArgMaxPolicy(self.exploitation_critic)
        self.exploit_rew_shift = agent_params['exploit_rew_shift']
        self.exploit_rew_scale = agent_params['exploit_rew_scale']
        self.eps = agent_params['eps']
        
        self.state_record = {}

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        samples_n = ob_no.shape[0]

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
            expl_bonus = np.zeros((samples_n,))
            for i in range(samples_n):
                ob = ob_no[i,:].astype(int)
                if ob not in self.state_record: # update the records
                    self.state_record[ob] = 1
                else:
                    self.state_record[ob] += 1
                expl_bonus[i] = 1/np.sqrt(self.state_record[ob])
                                    
            # Reward Calculations
            assert expl_bonus.shape == re_n.shape
            env_reward = (re_n + self.exploit_rew_shift) * self.exploit_rew_scale
            mixed_reward = explore_weight * expl_bonus + exploit_weight * re_n
            assert env_reward.shape == mixed_reward.shape == re_n.shape

            # Update Exploration Model and Critics

            # Update the exploration critic (based off mixed_reward)
            exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)
            # Update the exploitation critic (based off env_reward)
            exploitation_critic_loss = self.exploitation_critic.update(ob_no, ac_na, next_ob_no, env_reward, terminal_n)
            
            # Update critic target networks
            if self.num_param_updates % self.target_update_freq == 0:
                self.exploration_critic.update_target_network()
                self.exploitation_critic.update_target_network()

            # Logging
            log['Exploration Critic Loss'] = exploration_critic_loss['Training Loss']
            log['Exploitation Critic Loss'] = exploitation_critic_loss['Training Loss']

            if self.exploitation_critic.cql_alpha >= 0:
                log['Exploitation Data q-values'] = exploitation_critic_loss['Data q-values']
                log['Exploitation OOD q-values'] = exploitation_critic_loss['OOD q-values']
                log['Exploitation CQL Loss'] = exploitation_critic_loss['CQL Loss']

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
