import shlex, subprocess

commands = []
command_stem = "python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_{lr} -lr {lr}"
params = []
for lr in params:
    commands.append(command_stem.format(lr=lr))

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
