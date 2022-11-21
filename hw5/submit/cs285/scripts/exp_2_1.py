import shlex, subprocess

command_stem = [
"python cs285/scripts/run_hw5_expl.py --no_gpu --env_name PointmassMedium-v0 --exp_name q2_dqn_seed{s} --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 --seed {s}",
"python cs285/scripts/run_hw5_expl.py --no_gpu --env_name PointmassMedium-v0 --exp_name q2_dqn_ss_seed{s} --use_rnd --exploit_rew_shift 1 --exploit_rew_scale 100 --unsupervised_exploration --offline_exploitation --cql_alpha=0 --seed {s}",
"python cs285/scripts/run_hw5_expl.py --no_gpu --env_name PointmassMedium-v0 --exp_name q2_cql_seed{s} --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --seed {s}",
"python cs285/scripts/run_hw5_expl.py --no_gpu --env_name PointmassMedium-v0 --exp_name q2_cql_ss_seed{s} --use_rnd --exploit_rew_shift 1 --exploit_rew_scale 100 --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --seed {s}",
]
# "ss" means "scaled and shifted"

commands = []
for stem in command_stem:
    for s in range(5):
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
