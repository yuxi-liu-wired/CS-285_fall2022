from .base_agent import BaseAgent
from cs285.models.ff_model import FFModel
from cs285.policies.MPC_policy import MPCPolicy
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from math import floor
from typing import Dict, Tuple, NoneType

class MBAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBAgent, self).__init__()

        self.env = env.unwrapped
        self.agent_params = agent_params
        self.ensemble_size = self.agent_params['ensemble_size']

        self.dyn_models = []
        for _ in range(self.ensemble_size):
            model = FFModel(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['learning_rate'],
            )
            self.dyn_models.append(model)

        self.actor = MPCPolicy(
            self.env,
            ac_dim=self.agent_params['ac_dim'],
            dyn_models=self.dyn_models,
            horizon=self.agent_params['mpc_horizon'],
            N=self.agent_params['mpc_num_action_sequences'],
            sample_strategy=self.agent_params['mpc_action_sampling_strategy'],
            cem_iterations=self.agent_params['cem_iterations'],
            cem_num_elites=self.agent_params['cem_num_elites'],
            cem_alpha=self.agent_params['cem_alpha'],
        )

        self.replay_buffer = ReplayBuffer()
        
        self.data_statistics = None

    def train(self, ob_no: np.array, ac_na: np.array, re_n: np.array, next_ob_no: np.array, terminal_n: np.array) -> Dict:
        """ Train the ensemble of predictive models using observed state transitions.
            NOTE: each model in the ensemble is trained on a different random batch
            
            Returns average training loss for the predictive models.
        """
        n_batch = ob_no.shape[0]
        num_data_per_env = floor(n_batch / self.ensemble_size)
        rand_indices = np.random.permutation(n_batch)

        loss = 0
        for i in range(self.ensemble_size):
            model = self.dyn_models[i]
            indices = rand_indices[i*num_data_per_env:(i+1)*num_data_per_env]
            observations = ob_no[indices,:]
            actions = ac_na[indices,:]
            next_observations = next_ob_no[indices,:]

            # update model
            assert self.data_statistics is not None
            log = model.update(observations, actions, next_observations,
                                self.data_statistics)
            loss += log['Training Loss']
        loss /= self.ensemble_size
        
        return {
            'Training Loss': loss,
        }

    def add_to_replay_buffer(self, paths, add_sl_noise=False) -> NoneType:

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {
            'obs_mean': np.mean(self.replay_buffer.obs, axis=0),
            'obs_std': np.std(self.replay_buffer.obs, axis=0),
            'acs_mean': np.mean(self.replay_buffer.acs, axis=0),
            'acs_std': np.std(self.replay_buffer.acs, axis=0),
            'delta_mean': np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
            'delta_std': np.std(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size):
        # NOTE: sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(
            batch_size * self.ensemble_size)

    def predict(self, ob_no, ac_na):
        """ Compute the average next state according to the ensemble of models.
        """
        return self.actor.predict(ob_no, ac_na)