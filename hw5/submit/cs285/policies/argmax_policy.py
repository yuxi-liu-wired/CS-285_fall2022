import numpy as np
import pdb


class ArgMaxPolicy(object):

    def __init__(self, critic, use_boltzmann=False):
        self.critic = critic
        self.use_boltzmann = use_boltzmann

    def set_critic(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        q_values = self.critic.qa_values(observation)

        if self.use_boltzmann: # boltzmann sampling
            # distribution = np.exp(q_values) / np.sum(np.exp(q_values))
            distribution = np.exp(q_values) 
            distribution /= distribution.sum()
            action = self.sample_discrete(distribution)
        else: # always argmax 
            action = q_values.argmax(-1)

        return action[0]

    def sample_discrete(self, p):
        # https://stackoverflow.com/questions/40474436/how-to-apply-numpy-random-choice-to-a-matrix-of-probability-values-vectorized-s
        c = p.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        choices = (u < c).argmax(axis=1)
        return choices

    ####################################
    ####################################