from typing import List
import numpy as np

from cs285.infrastructure.utils import unnormalize

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import normalize, unnormalize

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations: np.ndarray, 
                    actions: np.ndarray, 
                    rewards_list: List[np.ndarray], 
                    next_observations: np.ndarray, 
                    terminals: np.ndarray):

        """
            Updates the policy (actor), using the given observations, actions,
            and the Q-values/advantages computed from the given rewards.
            
            OUTPUT. training log from updating the policy.
        """
        # WE DONT USE `next_observations`??!!
        
        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)

        train_log = self.actor.update(observations, actions, advantages, q_values)
        
        return train_log

    def calculate_q_vals(self, rewards_list: List[np.ndarray]) -> np.ndarray:

        """
            Monte Carlo estimation of the Q function, using either
            - the full trajectory-based estimator, or 
            - the reward-to-go estimator
            
            INPUT. `rewards_list`, a list of arrays of rewards. 
                rewards_list[i][t] is the reward at step t for trajectory i.
                [
                    [r_{0,0}, r_{0,1}, ..., r_{0, T_0}],
                    [r_{1,0}, r_{1,1}, ..., r_{1, T_1}],
                    ...
                    [r_{N-1,0}, r_{N-1,1}, ..., r_{N-1, T_{N-1}}]
                ]
            
            OUTPUT. `q_values`: a list of 1D-array obtained by first computing 
                the Monte Carlo Q-value estimation on each trajectory in rewards_list, 
                then concatenating them together.
                For example, 
                    - q_values[0][0] is the Q-value estimate for trajectory 0, step 0.
                    - q_values[1][1] is the Q-value estimate for trajectory 1, step 1.
        """

        q_values_list = []
        
        # Case 1: full-trajectory PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory

        if not self.reward_to_go:
            for reward_trajectory in rewards_list:
                disc_return = self._discounted_return(reward_trajectory, self.gamma)
                q_values_list.append(disc_return)

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            for reward_trajectory in rewards_list:
                q_values_list.append(self._discounted_cumsum(reward_trajectory, self.gamma))

        return np.concatenate(q_values_list)

    def estimate_advantage(self, obs: np.ndarray, rews_list: List[np.ndarray], q_values: np.ndarray, terminals: np.ndarray) -> np.ndarray:

        """
            Computes advantages by one of the following methods: 
                - A_{GAE}^π(s_t, a_t)
                - A^π(s_t, a_t) = Q^π(s_t, a_t) - V_φ(s_t), where V is computed by a "baseline" neural network
                - A^π(s_t, a_t) = Q^π(s_t, a_t)
            
            OUTPUT. array of estimated advantage at each step.
                A^π(s_t, a_t)   ∀ t = 0, ..., T-1
        """ 

        # Estimate the advantage A = Q - V, where V is computed by the "baseline" neural network.
        if self.nn_baseline:
            values_normalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_normalized.ndim == q_values.ndim
            ## values were trained with standardized q_values, so ensure
            ## that the predictions have the same mean and standard deviation as
            ## the current batch of q_values
            values_list = []
            count = 0
            for i in range(len(rews_list)):
                episode_len = len(rews_list[i])
                q_values_episode = q_values[count:count+episode_len]
                values_list.append(unnormalize(values_normalized[count:count+episode_len], 
                                               np.mean(q_values_episode), 
                                               np.std(q_values_episode)))
                count += episode_len
            assert count == q_values.size
            values = np.concatenate(values_list)

            if self.gae_lambda is not None:
                advantages = np.zeros(shape=q_values.shape)
                count = 0
                for i in range(len(rews_list)):
                    episode_len = len(rews_list[i])
                    advantages_episode = advantages[count:count+episode_len]
                    values_episode = values[count:count+episode_len]
                    
                    delta_episode = rews_list[i] - values_episode
                    delta_episode[:-1] += self.gamma * values_episode[1:]
                    
                    advantages_episode = self._discounted_cumsum(delta_episode, self.gamma * self.gae_lambda)
                    count += episode_len

            else: 
                # There is no lambda, so we do 1-step advantage estimation: A = Q - V
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize advantages to (mean = 0, std = 1)
        if self.standardize_advantages:
            advantages = normalize(advantages)
            # advantages_list = self._split_array_according_to_list(q_values, rews_list)
            # for i in range(advantages_list):
            #     advantages_list[i] = normalize(advantages_list[i])
            # advantages = np.concatenate(advantages_list)

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards: np.ndarray, gamma: float) -> float:
        """
            Helper function

            Input: np.ndarray of rewards [r_0, r_1, ..., r_t', ... r_{T-1}] from a single rollout of length T

            Output: sum_{t'=0}^{T-1} gamma^t' r_{t'}
        """
        reversed_rewards = np.copy(np.flip(rewards))
        discounted_return = 0.0
        with np.nditer(reversed_rewards) as it:
            for x in it:
                discounted_return = x[...] + gamma * discounted_return
        return np.full(shape=rewards.size, fill_value=discounted_return)

    def _discounted_cumsum(self, rewards: np.ndarray, gamma: float) -> np.ndarray:
        """
            Helper function
            
            INPUT. np.ndarray of rewards [r_0, r_1, ..., r_t', ... r_{T-1}] from a single rollout of length T
            
            OUTPUT. np.ndarray where the entry in each index t' is sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
        """
        running_cum = np.copy(np.flip(rewards))
        discounted_return = 0.0
        with np.nditer(running_cum, op_flags=['readwrite']) as it:
            for x in it:
                discounted_return = x[...] + gamma * discounted_return
                x[...] = discounted_return
        return np.copy(np.flip(running_cum))

    def _split_array_according_to_list(self, array: np.ndarray, reference_list: List) -> List[np.ndarray]:
        split_list = []
        counter = 0
        for item in reference_list:
            split_list.append(array[counter:counter+len(item)])
            counter += len(item)
        return split_list