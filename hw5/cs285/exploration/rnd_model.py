from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
import numpy as np

def init_method_uniform(model):
    radius = 1.73205 # sqrt(3), making uniform(-r, r) have std 1
    for module in model:
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-radius, radius)
            module.bias.data.uniform_(-radius, radius)

def init_method_uniform_1(model):
    for module in model:
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_()
            module.bias.data.uniform_()

def init_method_normal(model):
    for module in model:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_()
            module.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']

        # the fixed random function we are trying to learn
        self.f = ptu.build_mlp(input_size=self.ob_dim,
                                output_size=self.output_size,
                                n_layers=self.n_layers, size=self.size)
        # the learned function we are using to learn f
        self.f_hat = ptu.build_mlp(input_size=self.ob_dim,
                                output_size=self.output_size,
                                n_layers=self.n_layers, size=self.size)
        
        # They must be initialized differently to avoid trivial learning
        # init_method_normal(self.f_hat)
        # init_method_uniform(self.f)
        init_method_normal(self.f) # TODO: try this initialization as a last resort
        init_method_uniform_1(self.f_hat)
        
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
        target = self.f(ob_no).detach() # target value f(o), detached
        pred = self.f_hat(ob_no)        # predicted value f_hat(o), attached
        l2_loss = ((pred - target)**2).sum(dim=1)
        # l2_loss = torch.linalg.norm(pred - target, dim=1)

        assert l2_loss.shape == (ob_no.shape[0],)
        return l2_loss

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        if isinstance(ob_no, np.ndarray):
            ob_no = ptu.from_numpy(ob_no)
        # loss = self(ob_no).mean()
        loss = torch.sqrt(self(ob_no)).mean() # TODO idk, but try this
        
        assert loss.shape == ()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
