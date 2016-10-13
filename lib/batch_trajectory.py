from collections import defaultdict

import numpy as np

from . import utils

class BatchTrajectory(object):
    """
    In charge of storing all trajectory related data
    """

    def __init__(self):
        self.paths = defaultdict(list)

    @property
    def states(self):
        return self.paths["states"]

    @property
    def actions(self):
        return self.paths["actions"]

    @property
    def rewards(self):
        return self.paths["rewards"]

    @property
    def returns(self):
        if self.paths["returns"] == []:
            # TODO: convert to error
            print("Returns have not been set")
        return self.paths["returns"]

    def store_step(self, state, action, reward):
        """
        @state list
        @action int
        @reward int
        """

        self.paths["states"] += [state]
        self.paths["actions"].append([action])
        self.paths["rewards"].append([reward])

    def calc_and_store_discounted_returns(self, last_eps_rewards, discount_factor):
        """
        Extract rewards for the episode and calc returns
        """
        discounted = utils.discount_cum_sum(last_eps_rewards, discount_factor)
        standardized = utils.standardize(discounted)
        self.paths["returns"] += standardized.tolist()
