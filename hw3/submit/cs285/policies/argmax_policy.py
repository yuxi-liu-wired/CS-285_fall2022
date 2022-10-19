import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs): 
        if len(obs.shape) <= 3: # There's only one frame! We need to add an axis to it.
            obs = obs[None]
        qa_t_values = self.critic.qa_values(obs)
        
        return qa_t_values.argmax(axis=1).squeeze()