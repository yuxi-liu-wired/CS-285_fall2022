import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs): 
        if len(obs.shape) > 3: # it's an Atari environment
            observation = obs
        else:
            observation = obs[None]
        
        qa_t_values = self.critic.qa_values(obs) # DOING. What if I do this for Lunar Lander???
        # qa_t_values = self.critic.qa_values(observation)
        
        return qa_t_values.argmax(dim=1)