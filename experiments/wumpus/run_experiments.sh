source ../experiments.sh

SEEDS=$(seq 0 9)
PARAMS_COMMON="-n 50000 -I 200 -l 100"

# vanilla
params="--disable-actions forget_and_infer --disable-shields is_safe --disable-rewards infer_wumpus_position,new_pos_smell,new_neg_smell"
for instance_steps in "wumpus_7_7_2 49" # "wumpus_5_5_1 15"
do
  set -- $instance_steps
  instance=$1
  steps=$2
  for seed in $SEEDS; do
      frompath wumpus single_experiment $instance _vanilla_$seed $PARAMS_COMMON $params -L $steps -s $seed  
  done
done

# reward shaping
params="--disable-actions forget_and_infer --disable-shields is_safe --disable-rewards infer_wumpus_position,new_neg_smell"
for instance_steps in "wumpus_7_7_2 49" # "wumpus_5_5_1 15"
do
  set -- $instance_steps
  instance=$1
  steps=$2
  for seed in $SEEDS; do
      frompath wumpus single_experiment $instance _reward_$seed $PARAMS_COMMON $params -L $steps -s $seed  
  done
done

# shielding
params="--disable-actions forget_and_infer --disable-rewards infer_wumpus_position,new_pos_smell,new_neg_smell"
for instance_steps in "wumpus_7_7_2 49" # "wumpus_5_5_1 15"
do
  set -- $instance_steps
  instance=$1
  steps=$2
  for seed in $SEEDS; do
      frompath wumpus single_experiment $instance _shielding_$seed $PARAMS_COMMON $params -L $steps -s $seed  
  done
done

# shielding + reward shaping
params="--disable-actions forget_and_infer --disable-rewards infer_wumpus_position,new_neg_smell"
for instance_steps in "wumpus_7_7_2 49" # "wumpus_5_5_1 15"
do
  set -- $instance_steps
  instance=$1
  steps=$2
  for seed in $SEEDS; do
      frompath wumpus single_experiment $instance _sreward_$seed $PARAMS_COMMON $params -L $steps -s $seed  
  done
done

wait
