#!/bin/bash

DEMETER_HOME=$HOME/dev/demeter
pushd $DEMETER_HOME/examples

# reinforce
python reinforce_acrobot.py \
  --batch_size 1 \
  --num_batches 1000 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --discount 0.99 \
  --hidden_sizes '' \
  --output_dir "$DEMETER_HOME/tmp/pg" \
  --experiment_name "reinforce_acrobot" 

# baseline reinforce
python reinforce_baseline_acrobot.py \
  --batch_size 1 \
  --num_batches 1000 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --value_learning_rate 0.1 \
  --discount 0.99 \
  --hidden_sizes '' \
  --output_dir "$DEMETER_HOME/tmp/pg" \
  --experiment_name "reinforce_baseline_acrobot" 

# vanilla_actor_critic
python vanilla_actor_critic_acrobot.py \
  --batch_size 1 \
  --num_batches 1000 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --value_learning_rate 0.1 \
  --discount 0.99 \
  --gae_lambda 0.99 \
  --hidden_sizes '' \
  --output_dir "$DEMETER_HOME/tmp/pg" \
  --experiment_name "actor_critic_acrobot" 
