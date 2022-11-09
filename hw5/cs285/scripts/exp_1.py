import shlex, subprocess

env1 = "PointmassEasy-v0"
env2 = "PointmassMedium-v0"

commands = [
"python cs285/scripts/run_hw5_expl.py --env_name {env1} --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd".format(env1=env1, env2=env2),
"python cs285/scripts/run_hw5_expl.py --env_name {env1}           --unsupervised_exploration --exp_name q1_env1_random".format(env1=env1, env2=env2),
"python cs285/scripts/run_hw5_expl.py --env_name {env2} --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd".format(env1=env1, env2=env2),
"python cs285/scripts/run_hw5_expl.py --env_name {env2}           --unsupervised_exploration --exp_name q1_env2_random".format(env1=env1, env2=env2),
]

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
