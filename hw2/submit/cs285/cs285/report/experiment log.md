# Experiment logs

## Meaning of flags

* `-n` : Number of iterations.
* `-b` : Batch size (number of state-action pairs sampled while acting according to the current policy at each iteration).
* `-dsa` : Flag: if present, sets standardize_advantages to False. Otherwise, by default, standardizes advantages to have a mean of zero and standard deviation of one.
* `-rtg` : Flag: if present, sets reward_to_go=True. Otherwise, reward_to_go=False by default.
* `--exp_name` : Name for experiment, which goes into the name for the data logging directory.

## Small-Scale Experiments

### Experiment 1 (CartPole-v0)

Run multiple experiments with the PG algorithm on the discrete CartPole-v environment, using the following commands:

```bash

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na

```

Successful. The experiment data are put into

```
q2_pg_q1_lb_no_rtg_dsa_CartPole-v0_23-09-2022_23-42-06
q2_pg_q1_lb_rtg_dsa_CartPole-v0_24-09-2022_12-12-58
q2_pg_q1_lb_rtg_na_CartPole-v0_23-09-2022_23-45-10
q2_pg_q1_sb_no_rtg_dsa_CartPole-v0_23-09-2022_23-31-14
q2_pg_q1_sb_rtg_dsa_CartPole-v0_23-09-2022_23-33-44
q2_pg_q1_sb_rtg_na_CartPole-v0_23-09-2022_23-38-45
```

### Experiment 2 (InvertedPendulum-v4)

Run experiments on the InvertedPendulum-v4 continuous control environment as follows:

Start with the defaults:

* batch size = 1000
* learning rate = 5e-3

and then try some more

```bash
# default setting -- optimum (1000) around iteration 92
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 1000 -lr 5e-3 -rtg --exp_name q2_b1000_r5e-3

# trying higher learning rate -- total collapse at iteration 80.
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 1000 -lr 5e-2 -rtg --exp_name q2_b1000_r5e-2

# trying lower leawing rate -- did not reach convergence
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 1000 -lr 5e-4 -rtg --exp_name q2_b1000_r5e-4

# trying less batch -- converged around iteration 80, but then it suddenly forgot how to play around iter 90
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 5e-3 -rtg --exp_name q2_b500_r5e-3

# trying higher learning rate -- it worked!
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 9e-3 -rtg --exp_name q2_b500_r9e-3

# trying higher learning rate -- failure
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 500 -lr 5e-2 -rtg --exp_name q2_b500_r5e-2

# trying lower batch size -- somehow, it worked?
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 300 -lr 9e-3 -rtg --exp_name q2_b300_r9e-3

# trying lower batch size -- failure
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 100 -lr 9e-3 -rtg --exp_name q2_b100_r9e-3

# trying higher lr -- failure
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 100 -lr 5e-2 -rtg --exp_name q2_b100_r5e-2
```

The best we've got so far is:

```bash
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 300 -lr 9e-3 -rtg --exp_name q2_b300_r9e-3
```

This data is put into `q2_pg_q2_b300_r9e-3_InvertedPendulum-v4_24-09-2022_00-18-39`.

## More Complex Experiments

### Experiment 3 (LunarLander-v2)

```bash
python cs285/scripts/run_hw2.py --env_name LunarLanderContinuous-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 5e-3 --reward_to_go --nn_baseline --exp_name q3_b40000_r5e-3
```

Meaning:

* episode length = 1000
* discounte rate = 0.99
* number of iterations = 100
* neural network layer (both the policy and the baseline) = 2
* number of hidden neurons = 64
* batch size = 40000
* learning rate = 5e-3
* using reward-to-go
* using baseline

The speed is pretty awful tbh. And for some reason turning on video logging slows down *all* training episodes significantly.

Most of the time is spent on "collecting data to be used for training", and almost no time spent on the other parts. "Training agent using sampled data from replay buffer" and "Beginning logging procedure".

The data is in `q2_pg_q3_b40000_r5e-3_LunarLanderContinuous-v2_24-09-2022_08-39-02`. Its eval returns steeply to 156 by step 32, then slowly to 180 by step 74, then at step 86 suffered a catastrophic failure back to around 100. As of iteration 100 it did not succeed in recovering. Presumably early stopping would have helped.

I made another run with video-logging to see how it is working. The videos showed the RL agent learned around step 40 to soft-land and *usually* close to the target area, and then did not improve. `q_pg_q_b_re-3_LunarLanderContinuous-v_24-09-2022_22-59-18`	

### Experiment 4 (HalfCheetah-v4)

You will be using your policy gradient implementation to learn a controller for the HalfCheetah-v4 benchmark environment with an episode length of 150. This is shorter than the default episode length (1000), which speeds up training significantly. Search over batch sizes b ∈ [10000, 30000, 50000] and learning rates r ∈ [0.005, 0.01, 0.02] to replace <b> and <r> below.

Run the following script with `python .\cs285\scripts\exp_4_1.py`.

```python
import os

if __name__ == "__main__":
    batch_size_list = [10000, 30000, 50000]
    lr_list = [0.005, 0.01, 0.02]
    for batch_size in batch_size_list:
        for lr in lr_list:
            command = "python ./cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {lr} -rtg --nn_baseline --exp_name q4_search_b{batch_size}_lr{lr}_rtg_nnbaseline".format(batch_size=batch_size, lr=lr)
            print(command)
            os.system(command)
```

Experiment data put into 

```
q2_pg_q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_24-09-2022_10-27-01
q2_pg_q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_24-09-2022_10-36-49
q2_pg_q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_24-09-2022_10-48-06
q2_pg_q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_24-09-2022_10-59-05
q2_pg_q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_24-09-2022_11-26-53
q2_pg_q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_24-09-2022_11-58-31
q2_pg_q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_24-09-2022_12-29-41
q2_pg_q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_24-09-2022_13-16-05
q2_pg_q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_24-09-2022_13-59-00
```

Inspection of the graphs show that the performance increases as batch size and learning rate increases. We choose batch size 50000 and learning rate 0.02 as the best. 

After that, substitute those rates by the following script, run with `python .\cs285\scripts\exp_4_2.py`.

```python
import os

batch_size = 50000
lr = 0.02
if __name__ == "__main__":
    command = "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {lr} --exp_name q4_b{batch_size}_r{lr}".format(batch_size=batch_size, lr=lr)
    os.system(command)
    command = "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {lr} -rtg --exp_name q4_b{batch_size}_r{lr}_rtg".format(batch_size=batch_size, lr=lr)
    os.system(command)
    command = "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {lr} --nn_baseline --exp_name q4_b{batch_size}_r{lr}_nnbaseline".format(batch_size=batch_size, lr=lr)
    os.system(command)
    command = "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {lr} -rtg --nn_baseline --exp_name q4_b{batch_size}_r{lr}_rtg_nnbaseline".format(batch_size=batch_size, lr=lr)
    os.system(command)
```

Best performance is only at around 170, by reward-to-go without baseline. Experiment data put into 

```
q2_pg_q4_b50000_r0.02_HalfCheetah-v4_25-09-2022_00-20-45
q2_pg_q4_b50000_r0.02_nnbaseline_HalfCheetah-v4_25-09-2022_01-40-29
q2_pg_q4_b50000_r0.02_rtg_HalfCheetah-v4_25-09-2022_01-00-50
q2_pg_q4_b50000_r0.02_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_02-20-02
```

### Experiment 5 (Hopper-V4)

You will now use your implementation of policy gradient with generalized advantage estimation to learn a controller for a version of Hopper-v with noisy actions.

Search over λ ∈ [0, 0.95, 0.98, 0.99, 1] to replace <λ> below. Note that with a correct implementation, λ = 1 is equivalent to the vanilla neural network baseline estimator. Do not change any of the other hyperparameters (e.g. batch size, learning rate).

```python
import os

if __name__ == "__main__":
    lam_list = [0, 0.95, 0.98, 0.99, 1]
    for lam in lam_list:
        command = "python cs285/scripts/run_hw2.py --env_name Hopper-v4 --ep_len 1000 --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda {lam} --exp_name q5_b2000_r0.001_lambda{lam}".format(lam=lam)
        print(command)
        os.system(command)
```

Experiment data put into

```
q2_pg_q5_b2000_r0.001_lambda0.95_Hopper-v4_25-09-2022_08-51-31
q2_pg_q5_b2000_r0.001_lambda0.98_Hopper-v4_25-09-2022_08-59-19
q2_pg_q5_b2000_r0.001_lambda0.99_Hopper-v4_25-09-2022_09-08-15
q2_pg_q5_b2000_r0.001_lambda0_Hopper-v4_25-09-2022_08-42-53
q2_pg_q5_b2000_r0.001_lambda1_Hopper-v4_25-09-2022_09-17-53
```

Provide a single plot with the learning curves for the Hopper-v4 experiments that you tried. Describe in words how λ affected task performance. The run with the best performance should achieve an average score close to 400.

For better or for worse, my results showed that the performance increases with $\lambda$ and $\lambda = 1$ is the best, consistently plateauing around eval reward 500. In that case, what even is the point of GAE?