## cartpole

# python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 \
# 	-n 100 -b 1000 --exp_name 1_1 -ntu 1 -ngsptu 1

## inverted-pendulum

# python cs285/scripts/run_hw3_actor_critic.py \
# 	--env_name InvertedPendulum-v2 --ep_len 1000 \
# 	--discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 \
#     --exp_name q5_ip_10_10 -ntu 10 -ngsptu 10

## half cheetah

 python cs285/scripts/run_hw3_actor_critic.py \
 	--env_name HalfCheetah-v2 --ep_len 150 \
 	--discount 0.90 --scalar_log_freq 1 \
 	-n 150 -l 2 -s 32 -b 30000 -eb 1500 \
 	-lr 0.02 --exp_name hc_10_10 -ntu 10 -ngsptu 10