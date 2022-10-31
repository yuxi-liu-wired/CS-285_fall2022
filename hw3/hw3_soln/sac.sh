# python cs285/scripts/run_hw3_sac.py \
#     --env_name Reacher-v4 --ep_len 50 \
# 	--discount 0.99 --scalar_log_freq 1000 \
# 	-n 1000000 -l 2 -s 256 -b 1000 -eb 500 \
# 	-lr 0.0003 --init_temperature 0.1 --exp_name hc_sac_reacher --seed 3 \
	# --actor_update_frequency 50 --critic_target_update_frequency 50 \

# half cheetah
python cs285/scripts/run_hw3_sac.py \
    --env_name HalfCheetah-v4 --ep_len 150 \
	--discount 0.99 --scalar_log_freq 1500 \
	-n 2000000 -l 2 -s 256 -b 1500 -eb 1500 \
	-lr 0.0003 --init_temperature 0.1 --exp_name hc_sac_cheetah --seed 2

#lunar lander
# python cs285/scripts/run_hw3_sac.py \
#     --env_name LunarLander-v2 --ep_len 150 \
# 	--discount 0.99 --scalar_log_freq 1500 \
# 	-n 2000000 -l 2 -s 256 -b 1500 -eb 1500 \
# 	-lr 0.001 --init_temperature 0.1 --exp_name hc_sac_cheetah --seed 4

# python cs285/scripts/run_hw3_sac.py \
#     --env_name InvertedPendulum-v4 --ep_len 1000 \
# 	--discount 0.99 --scalar_log_freq 1000 \
# 	-n 200000 -l 2 -s 256 -b 1000 -eb 2000 \
# 	-lr 0.0003 --init_temperature 0.1 --exp_name hc_sac_inverted --seed 3

# python cs285/scripts/run_hw3_sac.py \
#     --env_name Hopper-v4 --ep_len 1000 \
# 	--discount 0.99 --scalar_log_freq 1000 \
# 	-n 1000000 -l 2 -s 256 -b 1000 -eb 1000 \
# 	-lr 0.0003 --exp_name hc_sac_hopper --seed 114514

# python cs285/scripts/run_hw3_sac.py \
#     --env_name Ant-v4 --ep_len 100 \
# 	--discount 0.99 --scalar_log_freq 1000 \
# 	-n 1000000 -l 2 -s 256 -b 1000 -eb 1000 \
# 	-lr 0.0003 --exp_name hc_sac_ant --seed 114514

# python cs285/scripts/run_hw3_sac.py \
#     --env_name Walker2d-v4 --ep_len 1000 \
# 	--discount 0.99 --scalar_log_freq 1000 \
# 	-n 1000000 -l 2 -s 256 -b 1000 -eb 1000 \
# 	-lr 0.0003 --exp_name hc_sac_walker --seed 114514