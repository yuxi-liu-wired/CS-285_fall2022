import shlex, subprocess

commands = []
command_stem = "python cs285/scripts/run_hw4_mb.py \
--exp_name q1_cheetah_n{natspi}_arch{nlayers}x{nsize} --env_name cheetah-cs285-v0 \
--add_sl_noise --n_iter 1 --batch_size_initial 20000 \
--num_agent_train_steps_per_iter {natspi} --n_layers {nlayers} --size {nsize} \
--scalar_log_freq -1 --video_log_freq -1 --mpc_action_sampling_strategy 'random'"

params = [(500, 1, 32), (5, 2, 250), (500, 2, 250)]
for natspi, nlayers, nsize in params:
    commands.append(command_stem.format(natspi=natspi, nlayers=nlayers, nsize=nsize))

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
