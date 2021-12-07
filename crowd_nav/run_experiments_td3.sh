#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
a=0.2
b=-0.25
c=0.25
d=1.0
# Script to reproduce results
for ((i=1;i<3;i+=1))
do
	python train.py \
	--policy td3_rl \
	--output_dir data/$day/td3_rl/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/td3.py \
	--safe_weight $d \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--human_num 1

#	python train.py \
#	--policy model-predictive-rl \
#	--output_dir data/$day/mprl/$i \
#	--randomseed $i  \
#	--config configs/icra_benchmark/mp_separate.py
done

