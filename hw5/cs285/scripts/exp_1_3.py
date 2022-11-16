import shlex, subprocess

# Problem 1.2. Using count-based exploration

command_stem = [
    "python cs285/scripts/run_hw5_expl.py --no_gpu --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --cbe --seed {s} --exp_name q1_alg_med_seed{s}",
    "python cs285/scripts/run_hw5_expl.py --no_gpu --env_name PointmassHard-v0 --use_rnd --unsupervised_exploration --cbe --seed {s} --exp_name q1_alg_hard_seed{s}",
]

commands = []
for s in range(6):
    for stem in command_stem:
        commands.append(stem.format(s=s))

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
