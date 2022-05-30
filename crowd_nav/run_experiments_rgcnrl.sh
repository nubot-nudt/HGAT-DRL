#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
a=0.2
b=-0.25
c=0.25
d=0.01
e=0.01
# Script to reproduce results
for ((i=0;i<5;i+=1))
do
	python train.py \
	--policy rgcn_rl \
	--output_dir data/$day/rgcn_rl2/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/rgcnrl.py \
	--safe_weight $d \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--re_rvo  $e    \
	--human_num 5   

#	python train.py \
#	--policy model-predictive-rl \
#	--output_dir data/$day/mprl/$i \
#	--randomseed $i  \
#	--config configs/icra_benchmark/mp_separate.py
done

