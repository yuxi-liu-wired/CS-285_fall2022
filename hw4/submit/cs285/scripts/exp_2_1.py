import shlex, subprocess


commands = []
command_stem = "python cs285/scripts/run_hw4_mb.py --exp_name q2_obstacles_{seed} --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --video_log_freq -1 --mpc_action_sampling_strategy 'random' \
--seed {seed}"

params = [0,1,2,3,4,5]

for seed in params:
    commands.append(command_stem.format(seed=seed))            

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
