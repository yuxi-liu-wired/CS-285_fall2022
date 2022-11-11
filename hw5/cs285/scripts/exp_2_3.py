import shlex, subprocess

commands = []
command_stem = [
"python cs285/scripts/run_hw5_expl.py --no_gpu --env_name {env} --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha={a} --exp_name q2_alpha{a}"
]

# (optional) scaled and shifted rewards for CQL

env = "PointmassMedium-v0"
for s in command_stem:
    for a in [0.02, 0.1, 0.5]:
        commands.append(s.format(env=env, a=a))

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
