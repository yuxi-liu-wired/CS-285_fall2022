import shlex, subprocess

commands = []
command_stem = "python cs285/scripts/run_hw4_mbpo.py \
--exp_name q6_cheetah_rlen{rollout_len} --env_name 'cheetah-cs285-v0' --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 10 --video_log_freq -1 --sac_discount 0.99 --sac_n_layers 2 --sac_size 256 --sac_batch_size 1500 --sac_learning_rate 0.0003 --sac_init_temperature 0.1 \
--sac_n_iter {sac_n_iter} \
--mbpo_rollout_length {rollout_len}"

params = [(0, 1000), (1, 5000), (10, 5000)]

for rollout_len, sac_n_iter in params:
    commands.append(command_stem.format(rollout_len=rollout_len, sac_n_iter=sac_n_iter))

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
