import shlex, subprocess

command_stem = [
"python cs285/scripts/run_hw5_iql.py --no_gpu --env_name PointmassEasy-v0   --exp_name q5_easy_supervised_lam{l}_tau{t}                                    --use_rnd --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={t}",
"python cs285/scripts/run_hw5_iql.py --no_gpu --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam{l}_tau{t}                              --use_rnd --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={t}",
"python cs285/scripts/run_hw5_iql.py --no_gpu --env_name PointmassEasy-v0   --exp_name q5_easy_unsupervised_lam{l}_tau{t}       --unsupervised_exploration --use_rnd --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={t}",
"python cs285/scripts/run_hw5_iql.py --no_gpu --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam{l}_tau{t} --unsupervised_exploration --use_rnd --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={t}",
]


iql_tau = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
# Best lambda for AWAC, found in part 4.
awac_l = ? 

commands = []
for command in command_stem:
    for tau in iql_tau:
        commands.append(command.format(l=awac_l, t=tau))

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
