from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import argparse
import gym
import numpy as np
import tensorflow as tf

from agents.reinforce import REINFORCE
from lib.batch_trajectory import BatchTrajectory
from lib.stats import Stats
from policies.discrete_stochastic_mlp import DiscreteStochasticMLPPolicy
from policies.annealers.linear_annealer import LinearAnnealer

# arg parsing
parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env_name', default='CartPole-v0', type=str, help='gym environment')
parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size to use during learning')
parser.add_argument('-nb', '--num_batches', default=1000, type=int, help='number of batches to use during learning')
parser.add_argument('-m', '--max_steps', default=200, type=int, help='max number of steps to run for')
parser.add_argument('-pl', '--policy_learning_rate', default=0.1, type=float, help='policy network learning rate')
parser.add_argument('-d', '--discount', default=0.99, type=float, help='reward discount rate to use')
parser.add_argument('-nh', '--hidden_sizes', default="", type=str, help='number of hidden units per layer (comma delimited)')
parser.add_argument('-ai', '--anneal_init', default=0.0, type=float, help='initial exploration rate')
parser.add_argument('-af', '--anneal_final', default=0.0, type=float, help='final exploration rate')
parser.add_argument('-as', '--anneal_steps', default=0, type=float, help='number of steps to anneal over')
parser.add_argument('-p', '--plot_dir', default='.', type=str, help='directory to save the reward plots')
parser.add_argument('-ex', '--experiment_name', default='default', type=str, help='name of the experiment')
args = parser.parse_args()
print(args)

# set random seeds
np.random.seed(0)
tf.set_random_seed(1234)

# init gym
env = gym.make(args.env_name)

# env vars
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

with tf.Session() as sess:

    annealer = LinearAnnealer(
            init_exp = args.anneal_init,
            final_exp = args.anneal_final,
            anneal_steps = args.anneal_steps)

    policy = DiscreteStochasticMLPPolicy(
                network_name = "action-network",
                sess = sess, 
                optimizer = tf.train.AdamOptimizer(args.policy_learning_rate),
                hidden_layers = args.hidden_sizes,
                num_inputs = state_dim,
                num_actions = num_actions,
                annealer = annealer)

    writer = tf.train.SummaryWriter("/tmp/{}".format(args.experiment_name))

    agent = REINFORCE(
        sess = sess,
        state_dim = state_dim,
        num_actions = num_actions,
        summary_writer = writer,
        summary_every = 100,
        action_policy = policy)

    tf.initialize_all_variables().run()

    stats = Stats(args.batch_size, args.max_steps)

    for i_batch in xrange(args.num_batches):
        traj = BatchTrajectory()
        
        for i_eps in xrange(args.batch_size):
            state = env.reset()

            for t in xrange(args.max_steps):
                action = policy.sample_action(state)

                next_state, reward, is_terminal, info = env.step(action)

                norm_reward = -10 if is_terminal else 0.1
                traj.store_step(state.tolist(), action, norm_reward)
                stats.store_reward(reward)

                state = next_state
    
                if is_terminal: break
            
            # discounts the rewards over a single episode
            eps_rewards = traj.rewards[-t-1:]
            traj.calc_and_store_discounted_returns(eps_rewards, args.discount) 

            stats.mark_eps_finished(i_batch, i_eps)

        agent.train(traj)

    # plotting
    stats.plot_batch_stats(args.plot_dir, args.experiment_name)
