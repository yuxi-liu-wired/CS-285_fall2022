import gym
import d4rl

# Create the environment
env = gym.make('maze2d-open-v0')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Automatically download and return the dataset
dataset = env.get_dataset()
# d4rl.qlearning_dataset adds next_observations.
dataset = d4rl.qlearning_dataset(env)
# Dataset is a dictionary of numpy arrays.
# ['observations', 'actions', 'next_observations', 'rewards', 'terminals']

print("\nobservations: shape", dataset['observations'].shape)
print(dataset['observations']) # An (N, dim_observation)-dimensional numpy array of observations


print("\nactions: shape", dataset['actions'].shape)
print(dataset['actions']) # An (N, dim_action)-dimensional numpy array of actions

print("\nrewards: shape", dataset['rewards'].shape)
print(dataset['rewards']) # An (N,)-dimensional numpy array of rewards

# observations: shape (977851, 4)
# [[ 3.0275512   2.9867747   0.10218666 -0.05524508]
#  [ 3.0269444   2.9886053  -0.06068372  0.18305033]
#  [ 3.025115    2.992813   -0.1829311   0.42077821]
#  ...
#  [ 0.9125538   3.6278195   0.0691421   3.7245796 ]
#  [ 0.9131342   3.662595    0.05804501  3.4775453 ]
#  [ 0.91361654  3.694906    0.04823486  3.2310991 ]]

# actions: shape (977851, 2)
# [[-0.6828367   1.        ]
#  [-0.51389796  1.        ]
#  [-0.37335745  1.        ]
#  ...
#  [-0.0459029  -1.        ]
#  [-0.04061032 -1.        ]
#  [-0.03562365 -1.        ]]

# rewards: shape (977851,)
# [0. 0. 0. ... 0. 0. 0.]