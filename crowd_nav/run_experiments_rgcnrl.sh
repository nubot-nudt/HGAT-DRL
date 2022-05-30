#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
a=0.2
b=-0.25
c=0.25
d=0.01
e=5
# Script to reproduce results
for ((i=0;i<5;i+=1))
do
	let "k=10**$i"
	python train.py \
	--policy rgcn_rl \
	--output_dir data/0528/resume/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/rgcnrl.py \
	--safe_weight $d \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--re_rvo  $k    \
	--human_num 5  \
	--resume 

#	python train.py \
#	--policy model-predictive-rl \
#	--output_dir data/$day/mprl/$i \
#	--randomseed $i  \
#	--config configs/icra_benchmark/mp_separate.py
done

