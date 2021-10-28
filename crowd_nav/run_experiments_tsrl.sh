#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
a=0.1
b=-0.25
c=0.25
# Script to reproduce results
for ((i=0;i<3;i+=1))
do
	python train.py \
	--policy tree-search-rl \
	--output_dir data/$day/tsrl/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/ts_separate.py \
	--safe_weight 1.0 \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--human_num 5

#	python train.py \
#	--policy model-predictive-rl \
#	--output_dir data/$day/mprl/$i \
#	--randomseed $i  \
#	--config configs/icra_benchmark/mp_separate.py
done

