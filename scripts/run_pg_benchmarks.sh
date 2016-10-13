#!/bin/bash

DEMETER_HOME=$HOME/dev/demeter
pushd $DEMETER_HOME/examples

# reinforce
python reinforce_cartpole.py \
  --env_name 'CartPole-v0' \
  --batch_size 1 \
  --num_batches 1000 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --discount 0.99 \
  --hidden_sizes '' \
  --anneal_init 0.0 \
  --anneal_final 0.0 \
  --anneal_steps 0 \
  --plot_dir "$DEMETER_HOME/scripts/plots/pg" \
  --experiment_name "default_reinforce" 

python reinforce_cartpole.py \
  --env_name 'CartPole-v0' \
  --batch_size 10 \
  --num_batches 250 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --discount 0.99 \
  --hidden_sizes '' \
  --anneal_init 0.0 \
  --anneal_final 0.0 \
  --anneal_steps 0 \
  --plot_dir "$DEMETER_HOME/scripts/plots/pg" \
  --experiment_name "reinforce_batch10" 

python reinforce_cartpole.py \
  --env_name 'CartPole-v0' \
  --batch_size 1 \
  --num_batches 250 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --discount 0.99 \
  --hidden_sizes '' \
  --anneal_init 0.2 \
  --anneal_final 0.0 \
  --anneal_steps 500 \
  --plot_dir "$DEMETER_HOME/scripts/plots/pg" \
  --experiment_name "reinforce_anneal" 

# baseline reinforce
python reinforce_baseline_cartpole.py \
  --env_name 'CartPole-v0' \
  --batch_size 1 \
  --num_batches 1000 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --value_learning_rate 0.1 \
  --discount 0.99 \
  --hidden_sizes '' \
  --anneal_init 0.0 \
  --anneal_final 0.0 \
  --anneal_steps 0 \
  --plot_dir "$DEMETER_HOME/scripts/plots/pg" \
  --experiment_name "reinforce_baseline" 

python reinforce_baseline_cartpole.py \
  --env_name 'CartPole-v0' \
  --batch_size 10 \
  --num_batches 1000 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --value_learning_rate 0.1 \
  --discount 0.99 \
  --hidden_sizes '' \
  --anneal_init 0.0 \
  --anneal_final 0.0 \
  --anneal_steps 0 \
  --plot_dir "$DEMETER_HOME/scripts/plots/pg" \
  --experiment_name "reinforce_baseline_batch10" 

python reinforce_baseline_cartpole.py \
  --env_name 'CartPole-v0' \
  --batch_size 1 \
  --num_batches 1000 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --value_learning_rate 0.1 \
  --discount 0.99 \
  --hidden_sizes '' \
  --anneal_init 0.2 \
  --anneal_final 0.0 \
  --anneal_steps 500 \
  --plot_dir "$DEMETER_HOME/scripts/plots/pg" \
  --experiment_name "reinforce_baseline_anneal" 

# vanilla_actor_critic
python vanilla_actor_critic_cartpole.py \
  --env_name 'CartPole-v0' \
  --batch_size 1 \
  --num_batches 1000 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --value_learning_rate 0.1 \
  --discount 0.99 \
  --gae_lambda 0.99 \
  --hidden_sizes '' \
  --anneal_init 0.0 \
  --anneal_final 0.0 \
  --anneal_steps 0 \
  --plot_dir "$DEMETER_HOME/scripts/plots/pg" \
  --experiment_name "actor_critic_cartpole" 

python vanilla_actor_critic_cartpole.py \
  --env_name 'CartPole-v0' \
  --batch_size 10 \
  --num_batches 250 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --value_learning_rate 0.1 \
  --discount 0.99 \
  --gae_lambda 0.99 \
  --hidden_sizes '' \
  --anneal_init 0.0 \
  --anneal_final 0.0 \
  --anneal_steps 0 \
  --plot_dir "$DEMETER_HOME/scripts/plots/pg" \
  --experiment_name "actor_critic_cartpole_batch10" 

python vanilla_actor_critic_cartpole.py \
  --env_name 'CartPole-v0' \
  --batch_size 1 \
  --num_batches 1000 \
  --max_steps 200 \
  --policy_learning_rate 0.01 \
  --value_learning_rate 0.1 \
  --discount 0.99 \
  --gae_lambda 0.99 \
  --hidden_sizes '' \
  --anneal_init 0.2 \
  --anneal_final 0.0 \
  --anneal_steps 500 \
  --plot_dir "$DEMETER_HOME/scripts/plots/pg" \
  --experiment_name "actor_critic_cartpole_anneal" 
