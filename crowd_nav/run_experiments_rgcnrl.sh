#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
a=0.1
b=-0.25
c=0.25
d=0.5
e=0.00
f=0.01
# Script to reproduce results
for ((i=$2;i<$3;i+=1))
do
	python3 train.py \
	--policy rgcn_rl \
	--output_dir data/ablution/$day/$1/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/rgcnrl.py \
	--safe_weight $d \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--re_rvo  $e    \
	--human_num 5  \
	--re_theta $f \
        --gnn $1
#	python train.py \
#	--policy model-predictive-rl \
#	--output_dir data/$day/mprl/$i \
#	--randomseed $i  \
#	--config configs/icra_benchmark/mp_separate.py
done

