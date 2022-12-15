import shlex, subprocess

config="experiments/config/d4rl/gym_rvs_r.cfg"
envs = ["halfcheetah-medium-replay-v2", "halfcheetah-medium-expert-v2", 
        "hopper-medium-replay-v2"     , "hopper-medium-expert-v2", 
        "walker2d-medium-replay-v2"   , "walker2d-medium-expert-v2"]
seeds=5

command_stem = "python src/rvs/train.py --configs {config} --env_name {env} --seed {seed} --use_gpu"
commands = []
for env in envs:
    for seed in range(seeds):
        commands.append(command_stem.format(config=config, env=env, seed=seed))

if __name__ == "__main__":
    for command in commands:
        print(command)
    user_input = None
    while user_input not in ['y', 'n']:
        user_input = input('Run experiment with above commands? (y/n): ')
        user_input = user_input.lower()[:1]
    if user_input == 'n':
        exit(0)

    # for command in commands:
    #     args = shlex.split(command)
    #     subprocess.Popen(args)
    
    n_simultaneous_processes = 3
    processes = []
    for i, command in enumerate(commands):
        args = shlex.split(command)
        processes.append(subprocess.Popen(args))
        if (i+1) % n_simultaneous_processes == 0 or i+1 == len(commands):
            for process in processes:
                process.wait()
            processes = []
