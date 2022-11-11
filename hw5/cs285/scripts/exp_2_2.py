import shlex, subprocess

commands = []
command_stem = [   
"python cs285/scripts/run_hw5_expl.py --no_gpu --env_name {env} --use_rnd --num_exploration_steps={nes} --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_{nes}",
"python cs285/scripts/run_hw5_expl.py --no_gpu --env_name {env} --use_rnd --num_exploration_steps={nes} --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_{nes}",
]

env = "PointmassMedium-v0" # or PointmassHard-v0
for s in command_stem:
    for nes in [5000, 15000]:
        commands.append(s.format(env=env, nes=nes))

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
