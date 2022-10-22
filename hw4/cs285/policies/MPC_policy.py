import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None) -> np.array:
        """ Returns a numpy array of shape (num_sequences, horizon, ac_dim),
            representing `N = num_sequences` number of action trajectories of length `H = horizon`:
            [[a_{1, t}, ..., a_{1, t+H-1}],
             [a_{2, t}, ..., a_{2, t+H-1}],
             ...
             [a_{N, t}, ..., a_{N, t+H-1}]]
        """
        # initialize with random actions
        candidate_acs = np.random.uniform(low=self.low, high=self.high, 
                                     size=(num_sequences, horizon, self.ac_dim))
        
        if self.sample_strategy == 'random' or (self.sample_strategy == 'cem' and obs is None):
            return candidate_acs
            
        elif self.sample_strategy == 'cem':
            # Implement action selection using CEM, as described in 
            # Section 3.3, "Iterative Random-Shooting with Refinement"
            # https://arxiv.org/pdf/1909.11652.pdf
            
            elite_means = candidate_acs.mean(axis=0)
            elite_vars = candidate_acs.var(axis=0)
            assert elite_means.shape == (horizon, self.ac_dim)
            assert elite_vars.shape == (horizon, self.ac_dim)
            
            # refine the sampling distribution iteratively
            for _ in range(self.cem_iterations):
                # Find elites
                reward_acs = self.evaluate_candidate_sequences(candidate_acs, obs)
                assert reward_acs.shape == (self.N,)
                
                elite_acs = candidate_acs[np.argsort(reward_acs)[-self.cem_num_elites:], :]
                assert elite_acs.shape == (self.cem_num_elites, horizon, self.ac_dim)
                
                # Update the elite mean and variance
                elite_means = (1.0 - self.cem_alpha) * elite_means + self.cem_alpha * elite_acs.mean(axis=0)
                elite_vars = (1.0 - self.cem_alpha) * elite_vars + self.cem_alpha * elite_acs.var(axis=0)
                
                # Sample candidate sequences from a Gaussian with the current elite mean and variance
                candidate_acs = np.random.normal(loc=elite_means, scale=np.sqrt(elite_vars))
                candidate_acs = candidate_acs.clip(self.low, self.high) # Is this a good idea??

            assert candidate_acs.shape == (num_sequences, horizon, self.ac_dim)
            return candidate_acs
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences: np.array, obs: np.array) -> np.array:
        """ For each model in ensemble, compute the predicted sum of rewards.
            Returns a numpy array of shape (N,), the mean predictions across all ensembles.
        """
        horizon = candidate_action_sequences.shape[1]
        assert candidate_action_sequences.shape == (self.N, horizon, self.ac_dim)
        assert obs.shape == (self.N, self.ob_dim)
        
        reward_candidates = np.zeros((self.N,))
        for model in self.dyn_models:
            rews = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            assert rews.shape == (self.N,)
            reward_candidates += rews
        reward_candidates /= len(self.dyn_models)
        assert reward_candidates.shape == (self.N,)
        return reward_candidates

    def get_action(self, obs: np.array):
        """ 
        :return: a numpy array of shape (1, ac_dim), representing the MPC action to take.
        """
        if self.data_statistics is None: # take random action
            action_to_take = self.sample_action_sequences(num_sequences=1, horizon=1)[0]
            assert action_to_take.shape == (1, self.ac_dim)
            return action_to_take

        # MPC
        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            best_index = 0
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            best_index = np.argmax(predicted_rewards)
        action_to_take = candidate_action_sequences[best_index,0][None] 
        assert action_to_take.shape == (1, self.ac_dim)
        return action_to_take

    def calculate_sum_of_rewards(self, obs: np.array, candidate_action_sequences: np.array, model) -> np.array:
        """
        :param obs: numpy array with the current observation. Shape [N, D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence. Shape [N].
        """
        assert obs.shape == (self.N, self.ob_dim)
        horizon = candidate_action_sequences.shape[1]
        assert candidate_action_sequences.shape == (self.N, horizon, self.ac_dim)
        
        assert self.data_statistics is not None
        
        sum_of_rewards = np.zeros((self.N,))
        for i in range(horizon):
            acs = candidate_action_sequences[:,i,:]
            rew, _ = self.env.get_reward(obs, acs) # Is it okay to ignore terminal??
            assert rew.shape == (self.N,)
            sum_of_rewards += rew
            obs = model.get_prediction(obs, acs, self.data_statistics) 
        
        return sum_of_rewards

    def predict(self, ob_no, ac_na):
        """ Compute the average next state according to the ensemble of models.
        """
        assert self.data_statistics is not None
        next_ob_no = np.zeros(shape=ob_no.shape)
        for model in self.dyn_models:
            next_ob_no += model.get_prediction(ob_no, ac_na, self.data_statistics)
        next_ob_no /= len(self.dyn_models)
        return next_ob_no