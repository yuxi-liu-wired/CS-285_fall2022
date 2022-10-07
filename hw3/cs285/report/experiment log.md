# Experiment log

Meaning of parameters

```python
'--env_name', type=str, default='CartPole-v0'
'--ep_len', type=int, default=200
'--exp_name', type=str, default='todo'
'--n_iter', '-n', type=int, default=200

'--num_agent_train_steps_per_iter', type=int, default=1
'--num_critic_updates_per_agent_update', type=int, default=1
'--num_actor_updates_per_agent_update', type=int, default=1

'--batch_size', '-b', type=int, default=1000 # steps collected per train iteration
'--eval_batch_size', '-eb', type=int, default=400 # steps collected per eval iteration
'--train_batch_size', '-tb', type=int, default=1000 # steps used per gradient step

'--discount', type=float, default=1.0
'--learning_rate', '-lr', type=float, default=5e-3
'--dont_standardize_advantages', '-dsa', action='store_true'
'--num_target_updates', '-ntu', type=int, default=10
'--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10
'--n_layers', '-l', type=int, default=2
'--size', '-s', type=int, default=64

'--seed', type=int, default=1
'--no_gpu', '-ngpu', action='store_true'
'--which_gpu', '-gpu_id', default=0
'--video_log_freq', type=int, default=-1
'--scalar_log_freq', type=int, default=10

'--save_params', action='store_true'
```

Tensorboard commands

* `(Eval|Train).*(Average|Std)Return`

## Part 1: Q learning

The first phase of the assignment is to implement a working version of Q-learning. The default code will run the Ms. Pac-Man game with reasonable hyperparameter settings.

You may want to look inside infrastructure/dqn utils.py to understand how the (memory-optimized) replay buffer works, but you will not need to modify it.

### Question 1: basic Q-learning performance (DQN)

#### LunarLander-v3 (optional)

To accelerate debugging, you may also test on LunarLander-v3.

Our reference solution with the default hyperparameters achieves around 150 reward after 350k timesteps, but there is considerable variation between runs and without the double-Q trick the average return often decreases after reaching 150.


```bash
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q1
```

|![](images/1_1.png)|
|:--:|
| <b>Fig 1.1. Learning curves for `LunarLander-v3`.</b>|

Data put in `q1_LunarLander-v3_06-10-2022_12-16-15`.

#### MsPacman-v0

Run it with the default hyperparameters on the Ms. Pac-Man game for 1 million steps using the command below. Our reference solution gets a return of 1500 in this timeframe. On Colab, this will take roughly 3 GPU hours. 


```bash
python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q1
```

Data in `q1_MsPacman-v0_06-10-2022_00-43-20`. Left it running overnight, and it accidentally ran for over 3.7 million steps. 1 million steps took about 2 hours on my laptop. The training curve has two phases: 

* Phase 1: fast learning in the first 1 million steps, reaching 1600 (average training reward).
* Phase 2: slow learning in the next 1.4 million steps, reaching 1850.
* Phase 3: plateau between 1650 and 1850.

|![](images/1_2.png)|
|:--:|
| <b>Fig 1.2. Learning curves for `MsPacman-v0`.</b>|

Include a learning curve plot showing the performance of your implementation on Ms. Pac-Man. The x-axis should correspond to number of time steps and the y-axis should show the average per-epoch reward as well as the best mean reward so far. Be sure to label the y-axis, since we need to verify that your implementation achieves similar reward as ours.

These quantities are already computed and printed in the starter code. They are also logged to the data folder, and can be visualized using Tensorboard as in previous assignments.

You should not need to modify the default hyperparameters in order to obtain good performance, but if you modify any of the parameters, list them in the caption of the figure.

### Question 2: double Q-learning (DDQN)

Use the double estimator to improve the accuracy of your learned Q values. This amounts to using the online Q network (instead of the target Q network) to select the best action when computing target values. Compare the performance of DDQN to vanilla DQN. Since there is considerable variance between runs, you must run at least three random seeds for both DQN and DDQN. You may use LunarLander-v3 for this question.

```bash
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --double_q --seed 1
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --double_q --seed 2
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --double_q --seed 3
```

|![](images/2.png)|
|:--:|
| <b>Fig 2. Learning curves for `LunarLander-v3`, comparison between the average over 3 runs of DDQN and the average over 3 runs of DQN.</b>|

make a single graph that averages the performance across three runs for both DQN and double DQN. See `scripts/read results.py` for an example of how to read the evaluation returns from Tensorboard logs.

### Question 3: experimenting with hyperparameters

Choose one hyperparameter of your choice and run at least three other settings of this hyperparameter, in addition to the one used in Question 1, and plot all four values on the same graph. Your choice what you experiment with, but you should explain why you chose this hyperparameter in the caption.

Examples include: learning rates; neural network architecture for the Q network, e.g., number of layers, hidden layer size, etc; exploration schedule or exploration rule (e.g. you may implement an alternative to ϵ-greedy and set different values of hyperparameters), etc. Discuss the effect of this hyperparameter on performance in the caption. You should find a hyperparameter that makes a nontrivial difference on performance.

Note: you might consider performing a hyperparameter sweep for getting good results in Question 1, in which case it’s fine to just include the results of this sweep for Question 3 as well, while plotting only the best hyperparameter setting in Question 1. The final results should use the following experiment name:

```bash
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam1 
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam2 
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam3 
```

|![](images/.png)|
|:--:|
| <b>Fig  Learning curves for .</b>|

You can replace LunarLander-v3 with PongNoFrameskip-v4 or MsPacman-v0 if you would like to test on a different environment.

## Part 2: Actor-Critic

### Question 4: Sanity check with Cartpole

Compare the results for the following settings and report which worked best. Do this by plotting all the runs on a single plot and writing your takeaway in the caption.

```bash
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v1 -n 100 -b 1000 --exp_name q4_1_1 -ntu 1 -ngsptu 1
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v1 -n 100 -b 1000 --exp_name q4_100_1 -ntu 100 -ngsptu 1
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v1 -n 100 -b 1000 --exp_name q4_1_100 -ntu 1 -ngsptu 100
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v1 -n 100 -b 1000 --exp_name q4_10_10 -ntu 10 -ngsptu 10
```

|![](images/4.png)|
|:--:|
| <b>Fig 4. Learning curves for `CartPole-v1`.</b>|

At the end, the best setting from above should match the policy gradient results from Cartpole in hw2 (200).

Indeed, in all cases, we have matched the 200 reward. The best one uses `-ntu 10 -ngsptu 10`: it reached 200 reward the fastest, and suffered no catastrophic forgetting.

Data put in

```
q4_1_1_CartPole-v0_06-10-2022_11-14-43
q4_10_10_CartPole-v0_06-10-2022_10-34-17
q4_100_1_CartPole-v0_06-10-2022_10-34-17
q4_1_100_CartPole-v0_06-10-2022_10-34-17
```

### Question 5: Run soft actor-critic with more difficult tasks

Use the best setting from the previous question to run InvertedPendulum and HalfCheetah:

```python
import shlex, subprocess

ntu = 10 # number of critic network updates
ngsptu = 10 # number of gradient steps per critic network update

commands = []
commands.append("python cs285/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name q5_{ntu}_{ngsptu} -ntu {ntu} -ngsptu {ngsptu}".format(ntu=ntu, ngsptu=ngsptu))
commands.append("python cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_{ntu}_{ngsptu} -ntu {ntu} -ngsptu {ngsptu}".format(ntu=ntu, ngsptu=ngsptu))

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
```

Data put in

```

```

Plots with the eval returns for both enviornments:

|![](images/5_1.png)|
|:--:|
| <b>Fig 5.1. Learning curves for InvertedPendulum-v4.</b>|

|![](images/5_2.png)|
|:--:|
| <b>Fig 5.2. Learning curves for HalfCheetah-v2.</b>|

Your results should roughly match those of policy gradient. After 150 iterations, your HalfCheetah return should be around 150. After 100 iterations, your InvertedPendulum return should be around 1000.

The returns should start going up immediately. For example, after 20 iterations, your HalfCheetah return should be above -40 and your InvertedPendulum return should near or above 100. However, there is some variance between runs, so the 150-iteration (for HalfCheetah) and 100-iteration (for InvertedPendulum) results are the numbers we use to grade.

## Part 3: Soft Actor-Critic

### Question 6: Run soft actor-critic more difficult tasks

Use the best setting from the previous question to run InvertedPendulum and HalfCheetah. You may use InvertedPendulum as a debugging environment, as it is much faster to train.

`python cs285/scripts/run_hw3_sac.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.99 --scalar_log_freq 1000 -n 100000 -l 2 -s 256 -b 1000 -eb 2000 -lr 0.0003 --init_temperature 0.1 ----exp_name q6a_sac_InvertedPendulum_<parameters> --seed 1`

where <parameters> is replaced with the parameters you chose.

|![](images/6_1.png)|
|:--:|
| <b>Fig 6.1. Learning curves for InvertedPendulum-v4.</b>|

`python cs285/scripts/run_hw3_sac.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.99 --scalar_log_freq 1500 -n 2000000 -l 2 -s 256 -b 1500 -eb 1500 -lr 0.0003 --init_temperature 0.1 --exp_name q6b_sac_HalfCheetah_<parameters> --seed 1`

Here the number of iterations stands for the number of environment steps taken. Your results should roughly match and exceed those of policy gradient. After 50000 steps, your HalfCheetah return should be around 200. After 20000 steps, your InvertedPendulum return should reach 1000. Your deliverables for this section are plots with the eval returns for both enviornments.

As a debugging tip, the returns should start going up immediately. For example, after 10000 steps, your HalfCheetah return should be above -40 (trending toward positive) and your InvertedPendulum return should be near or above 100. However, there is some variance between seeds, so the 1000 eval average return under 100000 steps (for InvertedPendulum) and 300 average return under 200000 steps (for HalfCheetah) results are the numbers we use to grade.


|![](images/6_2.png)|
|:--:|
| <b>Fig 6.2. Learning curves for HalfCheetah-v4.</b>|