# static arguments
num_gpus=4
dry_run=0
which_experiment=0

# hyperparameters
lrs=(1e-3 3e-4 1e-4 3e-5)
sizes=(256 1024)
n_layers=(2 3 4)
batch_sizes=(256)

for lr in ${lrs[@]}; do
    for size in ${sizes[@]}; do
        for n_layer in ${n_layers[@]}; do
            for batch_size in ${batch_sizes[@]}; do
                which_gpu=$((which_experiment % num_gpus))
                gpu_command="export CUDA_VISIBLE_DEVICES=$which_gpu"

                command="python cs285/scripts/run_hw3_sac.py \
                --env_name Hopper-v4 --ep_len 1000 \
                --discount 0.99 --scalar_log_freq 1000 \
                -n 1000000 -l $n_layer -s $size -b $batch_size -eb 1500 \
                -lr $lr --exp_name hc_sac_cheetah_${lr}_${n_layer}_${size}_${batch_size} \
                --seed 1024 &"

                echo $command

                if [ $dry_run -eq 0 ]; then
                    eval $gpu_command
                    eval $command
                    
                    sleep 5

                fi

                which_experiment=$((which_experiment + 1))
            done
        done
    done
done

echo "Done running all $which_experiment experiments"