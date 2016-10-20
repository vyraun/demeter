from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from lib.batch_stats import BatchStats
from lib.batch_trajectory import BatchTrajectory

class BatchSampler(object):

    def __init__(self, env,
                       policy,
                       norm_reward,
                       discount):

        self.env = env
        self.policy = policy
        self.norm_reward = norm_reward
        self.discount = discount

    def sample(self, i_batch, batch_size, max_steps, verbose = True):
        traj = BatchTrajectory()
        stats = BatchStats(batch_size, max_steps)

        for i_eps in xrange(batch_size):
            state = self.env.reset()

            for t in xrange(max_steps):
                action = self.policy.sample_action(state)

                next_state, reward, is_terminal, info = self.env.step(action)

                traj.store_step(state.tolist(), action, self.norm_reward(is_terminal))
                stats.store_reward(reward)

                state = next_state

                if is_terminal: break

            # discounts the rewards over a single episode
            eps_rewards = traj.rewards[-t - 1:]
            traj.calc_and_store_discounted_returns(eps_rewards, self.discount)

            stats.summarize_eps(i_batch, i_eps, verbose)

        return stats, traj
