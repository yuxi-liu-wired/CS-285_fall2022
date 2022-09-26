# New experiment logs

This is for my other experiment where I change the norm from "normalize per trajectory" to "normalize per batch". It sounds stupid, but the tutor stated that "The q-values should also be normalized over all trajectories".

The fix is really simple: just one line in `pg_agent.py`:

```python
values = unnormalize(values_normalized, np.mean(q_values), np.std(q_values))

# values_list = []
# count = 0
# for i in range(len(rews_list)):
#     episode_len = len(rews_list[i])
#     q_values_episode = q_values[count:count+episode_len]
#     values_list.append(unnormalize(values_normalized[count:count+episode_len], 
#                                    np.mean(q_values_episode), 
#                                    np.std(q_values_episode)))
#     count += episode_len
# assert count == q_values.size
# values = np.concatenate(values_list)
```

New data obtained: Since the code change affects only the neural baseline, the first two experiments are unaffected.

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

Old data: `q2_pg_q3_b40000_r5e-3_LunarLanderContinuous-v2_24-09-2022_08-39-02`

Its eval returns steeply to 156 by step 32, then slowly to 180 by step 74, then at step 86 suffered a catastrophic failure back to around 100. As of iteration 100 it did not succeed in recovering. Presumably early stopping would have helped.

I made another run with video-logging to see how it is working. The videos showed the RL agent learned around step 40 to soft-land and *usually* close to the target area, and then did not improve. 

New data: `q2_pg_q3_b40000_r5e-3_LunarLanderContinuous-v2_25-09-2022_12-02-13`. It did not suffer from catastrophic forgetting. It achieved around 180 eval reward at step 47 and remained there until the end (step 99).

### Experiment 4 (HalfCheetah-v4)

```
New data:

q2_pg_q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_11-49-27
q2_pg_q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_11-49-27
q2_pg_q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_11-49-27
q2_pg_q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_11-49-27
q2_pg_q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_11-49-27
q2_pg_q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_11-49-27
q2_pg_q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_11-49-27
q2_pg_q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_11-49-27
q2_pg_q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_11-49-27

Old data:

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

For old data, inspection of the graphs show that the performance increases as batch size and learning rate increases. We choose batch size 50000 and learning rate 0.02 as the best. The best eval reward is around 150.

Same for the new data, except that we witnessed smoother curves, and also a sudden catastrophic forgetting for batch size 30000, learning rate 0.02. Best performance is batch size 50000, learning rate 0.02, with eval reward around 180.

After that, substitute those rates by the following script, run with `python .\cs285\scripts\exp_4_2.py`.

Old data: Best performance is only at around 170, by reward-to-go without baseline.

New data: Best performance reaches 180 -- 200, with reward-to-go with baseline.

```
New data:

q2_pg_q4_b50000_r0.02_HalfCheetah-v4_25-09-2022_12-41-52
q2_pg_q4_b50000_r0.02_nnbaseline_HalfCheetah-v4_25-09-2022_12-41-52
q2_pg_q4_b50000_r0.02_rtg_HalfCheetah-v4_25-09-2022_12-41-52
q2_pg_q4_b50000_r0.02_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_12-41-52

Old data:

q2_pg_q4_b50000_r0.02_HalfCheetah-v4_25-09-2022_00-20-45
q2_pg_q4_b50000_r0.02_nnbaseline_HalfCheetah-v4_25-09-2022_01-40-29
q2_pg_q4_b50000_r0.02_rtg_HalfCheetah-v4_25-09-2022_01-00-50
q2_pg_q4_b50000_r0.02_rtg_nnbaseline_HalfCheetah-v4_25-09-2022_02-20-02
```

For fun, I logged the videos: `python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name q4_b50000_r0.02_rtg_nnbaseline --video_log_freq 10`. The cheetah learned to run by.flipping upside down and edging forward by pumping its legs. Not what we expected, but still effective.

### Experiment 5 (Hopper-V4)

Provide a single plot with the learning curves for the Hopper-v4 experiments that you tried. Describe in words how λ affected task performance. The run with the best performance should achieve an average score close to 400.

Old data showed that the performance increases with $\lambda$ and $\lambda = 1$ is the best, consistently plateauing around eval reward 500. In that case, what even is the point of GAE?

Using the new implementation actually showed that $\lambda = 0.98$ is the best with eval reward 484, better than $\lambda = 1$ by about 100 points.

Still, it is fishy that my previous implementation with $\lambda = 1$ reaches eval reward 500! I suspect it shows the superiority of using per-episode normalization of Q-values!

```
New data:

q2_pg_q5_b2000_r0.001_lambda0.95_Hopper-v4_25-09-2022_11-28-47
q2_pg_q5_b2000_r0.001_lambda0.98_Hopper-v4_25-09-2022_11-28-47
q2_pg_q5_b2000_r0.001_lambda0.99_Hopper-v4_25-09-2022_11-28-47
q2_pg_q5_b2000_r0.001_lambda0_Hopper-v4_25-09-2022_11-28-47
q2_pg_q5_b2000_r0.001_lambda1_Hopper-v4_25-09-2022_11-28-47

Old data:

q2_pg_q5_b2000_r0.001_lambda0.95_Hopper-v4_25-09-2022_08-51-31
q2_pg_q5_b2000_r0.001_lambda0.98_Hopper-v4_25-09-2022_08-59-19
q2_pg_q5_b2000_r0.001_lambda0.99_Hopper-v4_25-09-2022_09-08-15
q2_pg_q5_b2000_r0.001_lambda0_Hopper-v4_25-09-2022_08-42-53
q2_pg_q5_b2000_r0.001_lambda1_Hopper-v4_25-09-2022_09-17-53
```

For fun, I generated videos. Data put in `q2_pg_q5_b2000_r0.001_lambda0.98_video_Hopper-v4_26-09-2022_01-40-59`... well, it certainly learned to leap forward by flexing its ankle really hard, but it didn't learn to leap with the entire knee.
