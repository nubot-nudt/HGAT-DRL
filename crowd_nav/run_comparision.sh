#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
a=0.1
b=-1.0
c=1.0
d=0.2
e=0.05
f=0.01
# Script to reproduce results
for ((i=$1;i<$2;i+=1))
do
	python3 train.py \
	--policy rgcn_rl \
	--output_dir data/$day/gat/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/rgcnrl.py \
	--safe_weight $d \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--re_rvo  $e    \
	--re_theta $f \
	--human_num 5  \
  --gnn gat

	python3 train.py \
	--policy rgcn_rl \
	--output_dir data/$day/rgcn/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/rgcnrl.py \
	--safe_weight $d \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--re_rvo  $e    \
	--re_theta $f \
	--human_num 5  \
  --gnn rgcn

	python3 train.py \
	--policy rgcn_rl \
	--output_dir data/$day/gcn/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/rgcnrl.py \
	--safe_weight $d \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--re_rvo  $e    \
	--re_theta $f \
	--human_num 5  \
  --gnn gcn

	python3 train.py \
	--policy rgcn_rl \
	--output_dir data/$day/transformer/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/rgcnrl.py \
	--safe_weight $d \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--re_rvo  $e    \
	--re_theta $f \
	--human_num 5  \
        --gnn transformer


#	python train.py \
#	--policy model-predictive-rl \
#	--output_dir data/$day/mprl/$i \
#	--randomseed $i  \
#	--config configs/icra_benchmark/mp_separate.py
done

