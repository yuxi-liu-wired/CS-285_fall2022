import shlex, subprocess

commands = []
command_stem = "python cs285/scripts/run_hw4_mb.py \
--exp_name q3_{envname} --env_name {envname}-cs285-v0 --add_sl_noise \
--mpc_horizon {mpc_horizon} \
--num_agent_train_steps_per_iter {natspi} --batch_size_initial 5000 --batch_size {batch_size} --n_iter {n_iter} \
--video_log_freq -1 --mpc_action_sampling_strategy 'random'"

params = [('obstacles', 10, 20, 1000, 12), ('reacher', 10, 1000, 5000, 15), ('cheetah', 15, 1500, 5000, 20)]

for envname, mpc_horizon, natspi, batch_size, n_iter in params:
    commands.append(command_stem.format(envname=envname, mpc_horizon=mpc_horizon, natspi=natspi, batch_size=batch_size, n_iter=n_iter))

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
