from torch import nn
import torch
from torch import optim
from cs285.models.base_model import BaseModel
from cs285.infrastructure.utils import normalize, unnormalize
from cs285.infrastructure import pytorch_util as ptu
import numpy as np
from typing import Dict, Tuple, NoneType

class FFModel(nn.Module, BaseModel):

    def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate=0.001):
        super(FFModel, self).__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.delta_network = ptu.build_mlp(
            input_size=self.ob_dim + self.ac_dim,
            output_size=self.ob_dim,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.delta_network.to(ptu.device)
        self.optimizer = optim.Adam(
            self.delta_network.parameters(),
            self.learning_rate,
        )
        self.loss = nn.MSELoss()
        self.obs_mean = None
        self.obs_std = None
        self.acs_mean = None
        self.acs_std = None
        self.delta_mean = None
        self.delta_std = None

    def update_statistics(
            self,
            obs_mean: np.array,
            obs_std: np.array,
            acs_mean: np.array,
            acs_std: np.array,
            delta_mean: np.array,
            delta_std,
    ) -> NoneType:
        self.obs_mean = ptu.from_numpy(obs_mean)
        self.obs_std = ptu.from_numpy(obs_std)
        self.acs_mean = ptu.from_numpy(acs_mean)
        self.acs_std = ptu.from_numpy(acs_std)
        self.delta_mean = ptu.from_numpy(delta_mean)
        self.delta_std = ptu.from_numpy(delta_std)

    def forward(
            self,
            obs_unnormalized: torch.Tensor,
            acs_unnormalized: torch.Tensor,
            obs_mean: torch.Tensor,
            obs_std: torch.Tensor,
            acs_mean: torch.Tensor,
            acs_std: torch.Tensor,
            delta_mean: torch.Tensor,
            delta_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param obs_unnormalized: Unnormalized observations
        :param acs_unnormalized: Unnormalized actions
        :param obs_mean: Mean of observations
        :param obs_std: Standard deviation of observations
        :param acs_mean: Mean of actions
        :param acs_std: Standard deviation of actions
        :param delta_mean: Mean of state difference `s_t+1 - s_t`.
        :param delta_std: Standard deviation of state difference `s_t+1 - s_t`.
        :return: tuple `(next_obs_pred, delta_pred_normalized)`
        This forward function should return a tuple of two items
            1. `next_obs_pred` which is the predicted `s_t+1`
            2. `delta_pred_normalized` which is the normalized (i.e. not
                unnormalized) output of the delta network. This is needed
        """
        n_batch = obs_unnormalized.shape[0]
        assert obs_unnormalized.shape == (n_batch, self.ob_dim)
        assert obs_mean.shape == (n_batch, self.ob_dim)
        assert obs_std.shape == (n_batch, self.ob_dim)
        assert delta_mean.shape == (n_batch, self.ob_dim)
        assert delta_std.shape == (n_batch, self.ob_dim)
        assert acs_unnormalized.shape == (n_batch, self.ac_dim)
        assert acs_mean.shape == (n_batch, self.ac_dim)
        assert acs_std.shape == (n_batch, self.ac_dim)
        
        # normalize input data to mean 0, std 1
        obs_normalized = normalize(obs_unnormalized, obs_mean, obs_std)
        acs_normalized = normalize(acs_unnormalized, acs_mean, acs_std)
        assert obs_normalized.shape == (n_batch, self.ob_dim)
        assert acs_normalized.shape == (n_batch, self.ac_dim)

        # predicted change in obs
        concatenated_input = torch.cat([obs_normalized, acs_normalized], dim=1)
        assert concatenated_input.shape == (n_batch, self.ob_dim + self.ac_dim)

        # compute delta_pred_normalized and next_obs_pred
        # the output of the network is normalized(s_t+1 - s_t).
        delta_pred_normalized = self.delta_network(concatenated_input)
        assert delta_pred_normalized.shape == (n_batch, self.ob_dim)
        
        delta_pred_unnormalized = unnormalize(delta_pred_normalized, delta_mean, delta_std)
        assert delta_pred_unnormalized.shape == (n_batch, self.ob_dim)
        next_obs_pred = obs_unnormalized + delta_pred_unnormalized
        assert next_obs_pred.shape == (n_batch, self.ob_dim)
        
        return next_obs_pred, delta_pred_normalized

    def get_prediction(self, obs: np.array, acs: np.array, data_statistics: Dict) -> np.array:
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: a numpy array of the predicted next-states (s_t+1)
        """
        
        assert obs.shape[0] > 0
        assert obs.shape[1] == self.ob_dim
        
        obs_unnormalized=ptu.from_numpy(obs)
        acs_unnormalized=ptu.from_numpy(acs)
        obs_mean=ptu.from_numpy(data_statistics["obs_mean"])
        obs_std=ptu.from_numpy(data_statistics["obs_std"])
        acs_mean=ptu.from_numpy(data_statistics["acs_mean"])
        acs_std=ptu.from_numpy(data_statistics["acs_std"])
        delta_mean=ptu.from_numpy(data_statistics["delta_mean"])
        delta_std=ptu.from_numpy(data_statistics["delta_std"])
        
        prediction, _ = self(
            obs_unnormalized,
            acs_unnormalized,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
            )

        return prediction

    def update(self, obs: np.array, acs: np.array, next_obs: np.array, data_statistics: Dict) -> Dict:
        """
        :param obs: numpy array of observations
        :param acs: numpy array of actions
        :param next_obs: numpy array of next observations
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: training loss
        """
        delta_target_normalized = normalize(next_obs - obs, 
                                            data_statistics["delta_mean"], 
                                            data_statistics["delta_std"])
        delta_target_normalized = ptu.from_numpy(delta_target_normalized)

        obs_unnormalized=ptu.from_numpy(obs)
        acs_unnormalized=ptu.from_numpy(acs)
        obs_mean=ptu.from_numpy(data_statistics["obs_mean"])
        obs_std=ptu.from_numpy(data_statistics["obs_std"])
        acs_mean=ptu.from_numpy(data_statistics["acs_mean"])
        acs_std=ptu.from_numpy(data_statistics["acs_std"])
        delta_mean=ptu.from_numpy(data_statistics["delta_mean"])
        delta_std=ptu.from_numpy(data_statistics["delta_std"])
        
        self.update_statistics(
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
        )
        
        _, delta_pred_normalized = self(
            obs_unnormalized,
            acs_unnormalized,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
        )
        
        loss = self.loss(delta_pred_normalized, delta_target_normalized)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }
