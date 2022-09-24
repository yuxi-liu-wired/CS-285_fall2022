# HW1 solutions
There's multiple ways to implement the code inside `MLP_policy.py`.

## `policies/MLP_policy.py`
### MLPPolicy
```
def get_action(self, obs: np.ndarray) -> np.ndarray:
    if len(obs.shape) > 1:
        observation = obs
    else:
        observation = obs[None]

    observation = ptu.from_numpy(observation)
    action_distribution = self(observation)
    action = action_distribution.sample()  # don't bother with rsample
    return ptu.to_numpy(action)

def forward(self, observation: torch.FloatTensor):
    if self.discrete:
        logits = self.logits_na(observation)
        action_distribution = distributions.Categorical(logits=logits)
        return action_distribution
    else:
        batch_mean = self.mean_net(observation)
        scale_tril = torch.diag(torch.exp(self.logstd))
        batch_dim = batch_mean.shape[0]
        batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
        action_distribution = distributions.MultivariateNormal(
            batch_mean,
            scale_tril=batch_scale_tril,
        )
        return action_distribution
```
### MLPPolicySL
```
def update(
        self, observations, actions,
        adv_n=None, acs_labels_na=None, qvals=None
):
    observations = ptu.from_numpy(observations)
    actions = ptu.from_numpy(actions)
    action_distribution = self(observations)
    predicted_actions = action_distribution.rsample()
    loss = self.loss(predicted_actions, actions)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return {
        'Training Loss': ptu.to_numpy(loss),
    }
```
Note that another solution could ignore `self.loss` and simply use
```
    loss = - action_distribution.log_prob(actions).mean()
```

## `infrastructure/rl_trainer.py`
```
def train_agent(self):
    all_logs = []
    for train_step in range(self.params['num_agent_train_steps_per_iter']):
        ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
        train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
        all_logs.append(train_log)
    return all_logs
```

```
def collect_training_trajectories(
        self,
        itr,
        load_initial_expertdata,
        collect_policy,
        batch_size,
):
    if itr == 0:
        if load_initial_expertdata:
            paths = pickle.load(open(self.params['expert_data'], 'rb'))
            return paths, 0, None
        else:
            num_transitions_to_sample = self.params['batch_size_initial']
    else:
        num_transitions_to_sample = self.params['batch_size']

    print("\nCollecting data to be used for training...")
    paths, envsteps_this_batch = utils.sample_trajectories(
        self.env, collect_policy, num_transitions_to_sample, self.params['ep_len'])

    train_video_paths = None
    if self.log_video:
        print('\nCollecting train rollouts to be used for saving videos...')
        train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

    return paths, envsteps_this_batch, train_video_paths
```
## `infrastructure/utils.py`
```
def sample_trajectory(env, policy, max_path_length, render=False):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render image of the simulated env
        if render:
            if hasattr(env, 'sim'):
                image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
            else:
                image_obs.append(env.render())
        obs.append(ob)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done or steps > max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)
```
    
```
def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)
        print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')

    return paths, timesteps_this_batch
```
```
def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):

    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

    return paths
```

## `infrastructure/pytorch_util.py`
```
def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)
```
## `infrastructure/replay_buffer.py`
```
def sample_random_data(self, batch_size):
    assert (
            self.obs.shape[0]
            == self.acs.shape[0]
            == self.rews.shape[0]
            == self.next_obs.shape[0]
            == self.terminals.shape[0]
    )
    rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
    return self.obs[rand_indices], self.acs[rand_indices], self.rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices]
```
