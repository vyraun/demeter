import numpy as np

import matplotlib.pyplot as plt

class Stats(object):
    """
    Bookkeeping for various reward statistics over a batch
    of an arbitrary number of episodes.

    TODO: Add batch summary capabilities
    """
    def __init__(self, batch_size, eps_max_steps):
        # constants
        self.batch_size = batch_size
        self.eps_max_steps = eps_max_steps

        # batch stats
        self.batch_total_eps_rewards = []
        self.batch_mean_eps_rewards = []
        self.batch_eps_lengths = []

        # bookkeeping
        self.eps_rewards_buffer = []

    def store_reward(self, reward):
        self.eps_rewards_buffer.append(reward)

    def mark_eps_finished(self, i_batch, i_eps):
        """
        Called after every eps.

        Clears reward buffer, updates the batch stats, and prints the episode summary
        """
        
        eps_rewards_sum = np.sum(self.eps_rewards_buffer)

        # update batch stats
        self.batch_total_eps_rewards.append(eps_rewards_sum)
        self.batch_mean_eps_rewards.append(np.mean(np.cumsum(self.eps_rewards_buffer)))
        self.batch_eps_lengths.append(len(self.eps_rewards_buffer))

        # clear rewards buffer
        self.eps_rewards_buffer = []

        self._print_eps_stats(i_batch, i_eps)

    def _print_eps_stats(self, i_batch, i_eps):
        idx = (i_batch * self.batch_size) + i_eps
        print("Batch {}, episode {}".format(i_batch + 1, i_eps + 1))
        print("Finished after {} timesteps".format(self.batch_eps_lengths[idx]))
        print("Reward for this episode: {}".format(self.batch_total_eps_rewards[idx]))
        print("Average reward for this episode: {}".format(self.batch_mean_eps_rewards[idx]))
        print("Average reward for the last 100 episodes: {}").format(np.mean(self.batch_total_eps_rewards[-100:]))
    
    def check_finished(self, i_eps, upper_limit):
        mean_reward = self.batch_mean_eps_rewards[i_eps]
        if mean_reward >= upper_limit: True 
        else: False

    def plot_batch_stats(self, plot_dir, name):
        """
        TODO: move plotting abstractions out to a separate class
        """

        chunks = np.array_split(self.batch_total_eps_rewards, int(len(self.batch_total_eps_rewards) / 25))
        chunk_means = [np.mean(chunk) for chunk in chunks]
        chunk_indexes = [i * 25 for i in range(1,len(chunks) + 1)]

        plt.plot(chunk_indexes,chunk_means)
        plt.legend(['total episode rewards'])

        plt.savefig(plot_dir + "/" + name)
