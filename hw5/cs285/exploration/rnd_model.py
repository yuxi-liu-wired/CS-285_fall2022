from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']

        # the random function we are trying to learn. Fixed.
        self.f = ptu.build_mlp(input_size=self.ob_dim,
                                output_size=self.output_size,
                                n_layers=self.n_layers, size=self.size)
        # the function we are using to learn f. Learned.
        self.f_hat = ptu.build_mlp(input_size=self.ob_dim,
                                output_size=self.output_size,
                                n_layers=self.n_layers, size=self.size)
        
        # They must be initialized differently to avoid trivial learning
        init_method_1(self.f)
        init_method_2(self.f_hat)
        
        self.optimizer_spec = optimizer_spec
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )

    def forward(self, ob_no):
        """ Get the prediction error for ob_no
        
        Returns:
            torch array: L2 prediction error, with f(ob_no) detached, but f_hat(ob_no) attached.
        """
        # detach the output of self.f, but not that of self.f_hat
        target = self.f(ob_no).detach() 
        pred = self.f_hat(ob_no)
        return nn.MSELoss()(pred, target)

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        loss = self(ob_no).sum() # Take the mean prediction error across the batch
        
        assert loss.shape == ()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
