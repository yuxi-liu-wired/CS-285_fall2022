import shlex, subprocess

commands = []
commands.append("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1")
commands.append("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2")
commands.append("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3")
commands.append("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --double_q --seed 1")
commands.append("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --double_q --seed 2")
commands.append("python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --double_q --seed 3")

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
