import shlex, subprocess

commands = []
commands.append("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam1 ")
commands.append("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam2 ")
commands.append("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam3 ")

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
