# HW2 solutions
These solutions only show the relevant functions and the ones that were specific to this assignment.
The solutions make use of this function:
```
from cs285.infrastructure.utils import normalize
```

## `agents/pg_agent.py`
```
def train(self, obs, acs, rews_list, next_obs, terminals):

    """
        Training a PG agent refers to updating its actor using the given observations/actions
        and the calculated qvals/advantages that come from the seen rewards.
    """

    # step 1: calculate q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
    q_values = self.calculate_q_vals(rews_list)

    # step 2: calculate advantages that correspond to each (s_t, a_t) point
    advantage_values = self.estimate_advantage(obs, q_values)

    # step 3: use all datapoints (s_t, a_t, q_t, adv_t) to update the PG actor/policy
    log = self.actor.update(obs, acs, advantage_values, qvals=q_values)
    return log
    
def estimate_advantage(self, obs, q_values):

    """
        Computes advantages by (possibly) subtracting a baseline from the estimated Q values
    """

    # Estimate the advantage as [Q-b], when nn_baseline is True,
    # by querying the neural network that you're using to learn the baseline
    if self.nn_baseline:
        b_n = self.actor.run_baseline_prediction(obs)
        assert b_n.ndim == q_values.ndim
        b_n = b_n * np.std(q_values) + np.mean(q_values)
        adv_n = q_values - b_n

    # Else, just set the advantage to [Q]
    else:
        adv_n = q_values.copy()

    # Normalize the resulting advantages
    if self.standardize_advantages:
        adv_n = normalize(adv_n, np.mean(adv_n), np.std(adv_n))

    return adv_n

def _discounted_return(self, rewards):
    """
        Helper function

        Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

        Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
            note that all entries of this output are equivalent
            because each sum is from 0 to T (and doesnt involve t)
    """

    # create a list of indices (t'): from 0 to T
    indices = np.arange(len(rewards))

    # create a list where the entry at each index (t') is gamma^(t')
    discounts = self.gamma**indices

    # create a list where the entry at each index (t') is gamma^(t') * r_{t'}
    discounted_rewards = discounts * rewards

    # scalar: sum_{t'=0}^T gamma^(t') * r_{t'}
    sum_of_discounted_rewards = sum(discounted_rewards)

    # list where each entry t contains the same thing
        # it contains sum_{t'=0}^T gamma^t' r_{t'}
    list_of_discounted_returns = np.ones_like(rewards) * sum_of_discounted_rewards

    return list_of_discounted_returns

def _discounted_cumsum(self, rewards):
    """
        Helper function which
        -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
        -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
    """

    all_discounted_cumsums = []

    # for loop over steps (t) of the given rollout
    for start_time_index in range(len(rewards)):

        # create a list of indices (t'): goes from t to T
        indices = np.arange(start_time_index, len(rewards))

        # create a list of indices (t'-t)
        indices_adjusted = indices - start_time_index

        # create a list where the entry at each index (t') is gamma^(t'-t)
        discounts = self.gamma**(indices_adjusted) # each entry is gamma^(t'-t)

        # create a list where the entry at each index (t') is gamma^(t'-t) * r_{t'}
        discounted_rtg = discounts * rewards[start_time_index:]

        # scalar: sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        sum_discounted_rtg = sum(discounted_rtg)
        all_discounted_cumsums.append(sum_discounted_rtg)

    list_of_discounted_cumsums = np.array(all_discounted_cumsums)
    return list_of_discounted_cumsums
```

## `policies/MLP_policy`
```
def update(self, observations, acs_na, adv_n=None, acs_labels_na=None,
               qvals=None):
    observations = ptu.from_numpy(observations)
    actions = ptu.from_numpy(acs_na)
    adv_n = ptu.from_numpy(adv_n)

    action_distribution = self(observations)
    loss = - action_distribution.log_prob(actions) * adv_n
    loss = loss.mean()

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    if self.nn_baseline:
        targets_n = normalize(qvals, np.mean(qvals), np.std(qvals))
        targets_n = ptu.from_numpy(targets_n)
        baseline_predictions = self.baseline(observations).squeeze()
        assert baseline_predictions.dim() == baseline_predictions.dim()
        
        baseline_loss = F.mse_loss(baseline_predictions, targets_n)
        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()

    return {
        'Training Loss': ptu.to_numpy(loss),
    }
```
