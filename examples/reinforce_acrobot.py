from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import argparse
import gym
import numpy as np
import tensorflow as tf

from agents.reinforce import REINFORCE
from samplers.batch_sampler import BatchSampler
from lib.evaluator import BaseEvaluator
from policies.discrete_stochastic_mlp import DiscreteStochasticMLPPolicy

# arg parsing
parser = argparse.ArgumentParser(description=None)
parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size to use during learning')
parser.add_argument('-nb', '--num_batches', default=1000, type=int, help='number of batches to use during learning')
parser.add_argument('-m', '--max_steps', default=1000, type=int, help='max number of steps to run for')
parser.add_argument('-pl', '--policy_learning_rate', default=0.1, type=float, help='policy network learning rate')
parser.add_argument('-d', '--discount', default=0.99, type=float, help='reward discount rate to use')
parser.add_argument('-nh', '--hidden_sizes', default="", type=str, help='number of hidden units per layer (comma delimited)')
parser.add_argument('-p', '--output_dir', default='../tmp', type=str, help='directory to save the reward outputs')
parser.add_argument('-ex', '--experiment_name', default='reinforce_acrobot', type=str, help='name of the experiment')
args = parser.parse_args()
print(args)

# set random seeds
np.random.seed(0)
tf.set_random_seed(1234)

# init gym
env = gym.make('Acrobot-v0')

# env vars
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# paths
output_path = args.output_dir + "/" + args.experiment_name

with tf.Session() as sess:

    policy = DiscreteStochasticMLPPolicy(
        network_name = "action-network",
        sess = sess,
        optimizer = tf.train.AdamOptimizer(args.policy_learning_rate),
        hidden_layers = args.hidden_sizes,
        num_inputs = state_dim,
        num_actions = num_actions)

    writer = tf.train.SummaryWriter(output_path)

    agent = REINFORCE(
        sess = sess,
        state_dim = state_dim,
        num_actions = num_actions,
        summary_writer = writer,
        summary_every = 100,
        action_policy = policy)

    sampler = BatchSampler(
        env = env,
        policy = policy,
        norm_reward = lambda x: 5.0 if x else -0.1,
        discount = args.discount)

    evaluator = BaseEvaluator(
        agent = agent,
        sampler = sampler,
        batch_size = args.batch_size,
        max_steps = args.max_steps,
        num_batches = args.num_batches)

    averaged_stats = evaluator.run_avg(5)
    np.savez(output_path, stats=averaged_stats)
