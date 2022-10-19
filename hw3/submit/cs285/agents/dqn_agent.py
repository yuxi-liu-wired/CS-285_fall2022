import numpy as np

from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.critics.dqn_critic import DQNCritic
import gym 

class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size'] # batch size per training step
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        
        self.learning_starts = agent_params['learning_starts'] 
        # Before learning_starts of training steps are taken, always act by random exploration.
        self.learning_freq = agent_params['learning_freq']
        # perform one parameter update per `learning_freq` train steps 
        # (the other steps only increment self.t).
        self.target_update_freq = agent_params['target_update_freq']
        # one target-Q-network update per `target_update_freq` Q-network updates. 

        self.replay_buffer_idx = None # temporary variable, used only in `step_env`
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0 # training steps taken so far
        self.num_param_updates = 0 # Q network updates so far

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """

        # store the latest observation ("frame") into the replay buffer
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        # decide if random explore or greedy
        eps = self.exploration.value(self.t)
        perform_random_action = (self.t < self.learning_starts) or (np.random.rand() < eps)
        
        ## Perform action
        if perform_random_action:
            assert isinstance(self.env.action_space, gym.spaces.Discrete)
            action = self.env.action_space.sample()
        else:
        # Actor takes `frame_history_len` previous observations, for partial observability.
            obs = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(obs)
        
        ## Review and clean up
        obs, reward, done, info = self.env.step(action)
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            obs = self.env.reset()
        self.last_obs = obs

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (        self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)):
            
            log = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
