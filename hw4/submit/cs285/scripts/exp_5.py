import shlex, subprocess

commands = []
command_stem = "python cs285/scripts/run_hw4_mb.py \
--exp_name q5_cheetah_{exp_name} \
--env_name 'cheetah-cs285-v0' --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 5 --video_log_freq -1 \
--mpc_action_sampling_strategy {sampling_strategy}"

params = [("random", "'random'"), 
          ("cem_2", "'cem' --cem_iterations 2"),
          ("cem_4", "'cem' --cem_iterations 4")]

for exp_name, sampling_strategy in params:
    commands.append(command_stem.format(exp_name=exp_name, sampling_strategy=sampling_strategy))

if __name__ == "__main__":
    for command in commands:
        print(command)
    user_input = None
    while user_input not in ['y', 'n']:
        user_input = input('Run experiment with above commands? (y/n): ')
        user_input = user_input.lower()[:1]
    if user_input == 'n':
        exit(0)
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
