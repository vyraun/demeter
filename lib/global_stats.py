import numpy as np

class GlobalStats(object):
    """
    Stats collection across all batches
    """

    def __init__(self, total_rewards=None,
                       mean_rewards=None,
                       eps_lengths=None):

        # global metrics
        self._total_rewards = [] if total_rewards is None else total_rewards
        self._mean_rewards = [] if mean_rewards is None else mean_rewards
        self._eps_lengths = [] if eps_lengths is None else eps_lengths

    @property
    def total_rewards(self):
        return self._total_rewards

    @property
    def mean_rewards(self):
        return self._mean_rewards

    @property
    def eps_lengths(self):
        return self._eps_lengths

    def print_lookback_summary(self, eps_lookback=100):
        moving_reward_avg = np.mean(self.total_rewards[-eps_lookback:])
        print("Average reward for the last 100 episodes: {}").format(moving_reward_avg)

    def add_batch_stats(self, batch_stats):
        # update global history
        self._total_rewards += batch_stats.total_eps_rewards
        self._mean_rewards += batch_stats.mean_eps_rewards
        self._eps_lengths += batch_stats.eps_lengths

    @classmethod
    def merge(cls, global_stats):
        total_rewards = np.mean([s.total_rewards for s in global_stats], axis=0)
        mean_rewards = np.mean([s.mean_rewards for s in global_stats], axis=0)
        eps_lengths = np.mean([s.eps_lengths for s in global_stats], axis=0)

        return cls(total_rewards, mean_rewards, eps_lengths)
