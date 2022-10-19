import shlex, subprocess

command_stem = "python cs285/scripts/run_hw3_sac.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.99 --scalar_log_freq 1500 -n 200000 -l 2 -s 256 -b 1500 -eb 1500 --init_temperature 0.1 --seed 1 --exp_name q6b_HalfCheetah_lr{lr}_tb{tb}"
commands = []
for lr in [3e-4, 1e-4]:
    for tb in [256, 512]:
        commands.append(command_stem.format(lr=lr, tb=tb))

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
