from __future__ import print_function
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
import numpy as np

from . import base_agent

class REINFORCEBaseline(base_agent.BaseAgent):
    def __init__(self, sess,
                       state_dim,
                       num_actions,
                       summary_writer,
                       summary_every,
                       action_policy,
                       value_estimator):

        super(REINFORCEBaseline, self).__init__(sess, state_dim, num_actions, summary_writer, summary_every)

        # agent specific 
        self.action_policy = action_policy
        self.value_estimator = value_estimator

    def train(self, traj):

        # update value network
        value_summary = self.value_estimator.train(traj.states, traj.returns, self.value_estimator.summary_op)

        # get baselines
        baselines = self.value_estimator.predict(traj.states)
        advantages = traj.returns - baselines

        # update policy network
        policy_summary = self.action_policy.train(traj.states, traj.actions, advantages, self.action_policy.summary_op)
        
        # bookkeeping
        self.train_iter += 1
        self.write_scalar_summaries([value_summary, policy_summary])
        if self.action_policy.annealer is not None:
            self.action_policy.annealer.anneal(self.train_iter)
