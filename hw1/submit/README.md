Run BC Ant experiment

```bash
python cs285/scripts/run_hw1.py \
  --expert_policy_file 'cs285/policies/experts/Ant.pkl' \
  --expert_data 'cs285/expert_data/expert_data_Ant-v4.pkl' \
  --env_name 'Ant-v4' \
  --exp_name 'bc_Ant' \
  --ep_len 10000 \
  --num_agent_train_steps_per_iter 10000 \
  --n_iter 1 \
  --batch_size 10000 \
  --eval_batch_size 10000 \
  --train_batch_size 100 \
  --max_replay_buffer_size 1000000 \
  --n_layers 2 \
  --size 64 \
  --learning_rate 5e-3 \
  --video_log_freq -1 \
  --scalar_log_freq 1 \
  --which_gpu 0 \
  --seed 1 
```

Run BC Hopper experiment

```bash
python cs285/scripts/run_hw1.py \
  --expert_policy_file 'cs285/policies/experts/Hopper.pkl' \
  --expert_data 'cs285/expert_data/expert_data_Hopper-v4.pkl' \
  --env_name 'Hopper-v4' \
  --exp_name 'bc_Hopper' \
  --ep_len 10000 \
  --num_agent_train_steps_per_iter 10000 \
  --n_iter 1 \
  --batch_size 10000 \
  --eval_batch_size 10000 \
  --train_batch_size 100 \
  --max_replay_buffer_size 1000000 \
  --n_layers 2 \
  --size 64 \
  --learning_rate 5e-3 \
  --video_log_freq -1 \
  --scalar_log_freq 1 \
  --which_gpu 0 \
  --seed 1 
```

Run BC Hopper experiment with varying iterations

```bash
python cs285/scripts/run_hw1.py \
  --expert_policy_file 'cs285/policies/experts/Hopper.pkl' \
  --expert_data 'cs285/expert_data/expert_data_Hopper-v4.pkl' \
  --env_name 'Hopper-v4' \
  --exp_name 'bc_Hopper' \
  --ep_len 10000 \
  --num_agent_train_steps_per_iter 100 \
  --n_iter 1 \
  --batch_size 10000 \
  --eval_batch_size 10000 \
  --train_batch_size 100 \
  --max_replay_buffer_size 1000000 \
  --n_layers 2 \
  --size 64 \
  --learning_rate 5e-3 \
  --video_log_freq -1 \
  --scalar_log_freq 1 \
  --which_gpu 0 \
  --seed 1 
```

```bash
python cs285/scripts/run_hw1.py \
  --expert_policy_file 'cs285/policies/experts/Hopper.pkl' \
  --expert_data 'cs285/expert_data/expert_data_Hopper-v4.pkl' \
  --env_name 'Hopper-v4' \
  --exp_name 'bc_Hopper' \
  --ep_len 10000 \
  --num_agent_train_steps_per_iter 1000 \
  --n_iter 1 \
  --batch_size 10000 \
  --eval_batch_size 10000 \
  --train_batch_size 100 \
  --max_replay_buffer_size 1000000 \
  --n_layers 2 \
  --size 64 \
  --learning_rate 5e-3 \
  --video_log_freq -1 \
  --scalar_log_freq 1 \
  --which_gpu 0 \
  --seed 1 
```

```bash
python cs285/scripts/run_hw1.py \
  --expert_policy_file 'cs285/policies/experts/Hopper.pkl' \
  --expert_data 'cs285/expert_data/expert_data_Hopper-v4.pkl' \
  --env_name 'Hopper-v4' \
  --exp_name 'bc_Hopper' \
  --ep_len 10000 \
  --num_agent_train_steps_per_iter 10000 \
  --n_iter 1 \
  --batch_size 10000 \
  --eval_batch_size 10000 \
  --train_batch_size 100 \
  --max_replay_buffer_size 1000000 \
  --n_layers 2 \
  --size 64 \
  --learning_rate 5e-3 \
  --video_log_freq -1 \
  --scalar_log_freq 1 \
  --which_gpu 0 \
  --seed 1 
```

```bash
python cs285/scripts/run_hw1.py \
  --expert_policy_file 'cs285/policies/experts/Hopper.pkl' \
  --expert_data 'cs285/expert_data/expert_data_Hopper-v4.pkl' \
  --env_name 'Hopper-v4' \
  --exp_name 'bc_Hopper' \
  --ep_len 10000 \
  --num_agent_train_steps_per_iter 100000 \
  --n_iter 1 \
  --batch_size 10000 \
  --eval_batch_size 10000 \
  --train_batch_size 100 \
  --max_replay_buffer_size 1000000 \
  --n_layers 2 \
  --size 64 \
  --learning_rate 5e-3 \
  --video_log_freq -1 \
  --scalar_log_freq 1 \
  --which_gpu 0 \
  --seed 1 
```

```bash
python cs285/scripts/run_hw1.py \
  --expert_policy_file 'cs285/policies/experts/Hopper.pkl' \
  --expert_data 'cs285/expert_data/expert_data_Hopper-v4.pkl' \
  --env_name 'Hopper-v4' \
  --exp_name 'bc_Hopper' \
  --ep_len 10000 \
  --num_agent_train_steps_per_iter 1000000 \
  --n_iter 1 \
  --batch_size 10000 \
  --eval_batch_size 10000 \
  --train_batch_size 100 \
  --max_replay_buffer_size 1000000 \
  --n_layers 2 \
  --size 64 \
  --learning_rate 5e-3 \
  --video_log_freq -1 \
  --scalar_log_freq 1 \
  --which_gpu 0 \
  --seed 1 
```

Run DAgger Ant experiment

```bash
python cs285/scripts/run_hw1.py \
  --expert_policy_file 'cs285/policies/experts/Ant.pkl' \
  --expert_data 'cs285/expert_data/expert_data_Ant-v4.pkl' \
  --env_name 'Ant-v4' \
  --exp_name 'dagger_Ant' \
  --do_dagger \
  --ep_len 10000 \
  --num_agent_train_steps_per_iter 10000 \
  --n_iter 40 \
  --batch_size 10000 \
  --eval_batch_size 10000 \
  --train_batch_size 100 \
  --max_replay_buffer_size 1000000 \
  --n_layers 2 \
  --size 64 \
  --learning_rate 5e-3 \
  --video_log_freq -1 \
  --scalar_log_freq 1 \
  --which_gpu 0 \
  --seed 1 
```

Run DAgger Hopper experiment

```bash
python cs285/scripts/run_hw1.py \
  --expert_policy_file 'cs285/policies/experts/Hopper.pkl' \
  --expert_data 'cs285/expert_data/expert_data_Hopper-v4.pkl' \
  --env_name 'Hopper-v4' \
  --exp_name 'dagger_Hopper' \
  --do_dagger \
  --ep_len 10000 \
  --num_agent_train_steps_per_iter 10000 \
  --n_iter 40 \
  --batch_size 10000 \
  --eval_batch_size 10000 \
  --train_batch_size 100 \
  --max_replay_buffer_size 1000000 \
  --n_layers 2 \
  --size 64 \
  --learning_rate 5e-3 \
  --video_log_freq -1 \
  --scalar_log_freq 1 \

  --which_gpu 0 \
  --seed 1 
```