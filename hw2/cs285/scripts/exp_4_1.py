import os

if __name__ == "__main__":
    batch_size_list = [10000, 30000, 50000]
    lr_list = [0.005, 0.01, 0.02]
    for batch_size in batch_size_list:
        for lr in lr_list:
            command = "python ./cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {lr} -rtg --nn_baseline --exp_name q4_search_b{batch_size}_lr{lr}_rtg_nnbaseline".format(batch_size=batch_size, lr=lr)
            print(command)
            os.system(command)
