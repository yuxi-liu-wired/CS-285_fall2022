for i in 1 2 3
do
	echo $i
	python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_$i --seed $i
done