import numpy as np

class BatchStats(object):
    """
    Stats collection for a single batch

    """
    def __init__(self, batch_size, eps_max_steps):
        # constants
        self.batch_size = batch_size
        self.eps_max_steps = eps_max_steps

        # eps summaries
        self.total_eps_rewards = []
        self.mean_eps_rewards = []
        self.eps_lengths = []

        # bookkeeping
        self.eps_rewards_buffer = []

    def store_reward(self, reward):
        self.eps_rewards_buffer.append(reward)

    def summarize_eps(self, i_batch, i_eps, verbose=True):
        # update batch summaries
        self.total_eps_rewards.append(np.sum(self.eps_rewards_buffer))
        self.mean_eps_rewards.append(np.mean(np.cumsum(self.eps_rewards_buffer)))
        self.eps_lengths.append(len(self.eps_rewards_buffer))

        # clear rewards buffer
        self.eps_rewards_buffer = []

        # print a summary of the episode
        if verbose: self._print_eps_summary(i_batch, i_eps)

    def _print_eps_summary(self, i_batch, i_eps):
        print("Batch {}, episode {}".format(i_batch + 1, i_eps + 1))
        print("Finished after {} timesteps".format(self.eps_lengths[-1]))
        print("Reward for this episode: {}".format(self.total_eps_rewards[-1]))
        print("Average reward for this episode: {}".format(self.mean_eps_rewards[-1]))
