import shlex, subprocess

commands = []
command_stem = "python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v1 -n 100 -b 1000 --exp_name q4_{ntu}_{ngsptu} -ntu {ntu} -ngsptu {ngsptu}"
params = [(1, 1), (1, 100), (10, 10), (100, 1)]
for ntu, ngsptu in params:
    commands.append(command_stem.format(ntu=ntu, ngsptu=ngsptu))
    
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
